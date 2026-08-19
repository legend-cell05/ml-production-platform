"""The promotion gate.

A model reaches production by passing six checks, each of which exists
because of a way a model can look better than it is.

1. **Feature version.** A model trained on one feature definition cannot be
   served against another. This is a match, not a comparison.
2. **It beats the champion, by a margin.** A gain of 0.002 PR-AUC on twenty
   thousand rows is noise, and promoting on noise means the "best" model is
   whichever was trained most recently.
3. **No segment regresses.** The check this project exists to demonstrate. A
   challenger can be better overall and worse for enterprise accounts -- and
   enterprise accounts are worth ten times the others, so the average is the
   wrong thing to look at. A segment is only judged when it has enough
   positive examples to be judged; four churns cannot support a decision, and
   a gate that blocks on four churns is a gate that gets switched off.
4. **The probabilities are probabilities.** The decision threshold is an
   expected-value calculation, and an expected value computed from an
   uncalibrated score is arithmetic on a number that does not mean what it
   says.
5. **It is fast enough.** Measured, not assumed, by scoring rows and taking
   the p95.
6. **It is worth more than doing nothing.** A model that loses money at its
   own chosen threshold should not be promoted however good its AUC is.

``--force`` exists, because a controlled exception is sometimes right. What it
cannot be is accidental: it is a named flag, it is logged, and the reason is
written onto the model version.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from polaris.config import Settings, get_settings
from polaris.exceptions import PromotionBlocked
from polaris.features.definitions import FEATURE_VERSION
from polaris.logging_config import get_logger
from polaris.registry.store import ModelRecord, get_production, segment_metrics, set_stage

logger = get_logger(__name__)

# A segment with fewer positives than this is reported as "not evaluable"
# rather than compared. With four churns, the difference between two PR-AUCs
# is noise, and blocking on it teaches everyone to use --force.
MIN_SEGMENT_POSITIVES = 20


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str
    value: float | None = None
    threshold: float | None = None
    skipped: bool = False


@dataclass
class GateDecision:
    """The verdict, with every check that produced it."""

    model_name: str
    version: int
    checks: list[Check] = field(default_factory=list)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and not c.skipped]

    @property
    def allowed(self) -> bool:
        return not self.failed

    @property
    def summary(self) -> str:
        if self.allowed:
            return f"{self.model_name} v{self.version} passed {len(self.checks)} checks"
        return "; ".join(f"{c.name}: {c.detail}" for c in self.failed)

    def raise_if_blocked(self) -> None:
        if not self.allowed:
            raise PromotionBlocked(self.summary, failed=[c.name for c in self.failed])


def _measure_latency(record: ModelRecord, sample: pd.DataFrame, *, repeats: int = 200) -> float:
    """p95 milliseconds for a single-row prediction.

    Single-row rather than batched on purpose: the serving path answers one
    account at a time when a CSM opens an account page, and a batch
    measurement hides the per-call overhead that dominates there.
    """
    payload = record.load()
    model = payload["model"]
    rows = sample.iloc[: min(len(sample), repeats)]
    timings: list[float] = []
    for index in range(len(rows)):
        single = rows.iloc[[index]]
        start = time.perf_counter()
        model.predict_proba(single)
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.percentile(timings, 95)) if timings else 0.0


def evaluate_gate(
    challenger: ModelRecord,
    settings: Settings | None = None,
    *,
    latency_sample: pd.DataFrame | None = None,
) -> GateDecision:
    """Run every check against the current champion."""
    settings = settings or get_settings()
    champion = get_production(challenger.model_name, settings)
    decision = GateDecision(model_name=challenger.model_name, version=challenger.version)

    # 1. Feature version
    decision.checks.append(
        Check(
            name="feature_version",
            passed=challenger.feature_version == FEATURE_VERSION,
            detail=(f"trained on {challenger.feature_version}, the store holds {FEATURE_VERSION}"),
        )
    )

    # 2. Beats the champion by a margin
    challenger_pr = challenger.pr_auc or 0.0
    if champion is None:
        decision.checks.append(
            Check(
                name="beats_champion",
                passed=True,
                detail="no champion yet; this becomes the baseline",
                value=challenger_pr,
                skipped=True,
            )
        )
    else:
        champion_pr = champion.pr_auc or 0.0
        gain = challenger_pr - champion_pr
        decision.checks.append(
            Check(
                name="beats_champion",
                passed=gain >= settings.min_pr_auc_gain,
                detail=(
                    f"PR-AUC {challenger_pr:.4f} against champion v{champion.version}'s "
                    f"{champion_pr:.4f} (gain {gain:+.4f}, need {settings.min_pr_auc_gain:+.4f})"
                ),
                value=gain,
                threshold=settings.min_pr_auc_gain,
            )
        )

    # 3. No segment regresses
    if champion is None:
        decision.checks.append(
            Check(
                name="no_segment_regression",
                passed=True,
                detail="no champion to compare segments against",
                skipped=True,
            )
        )
    else:
        challenger_segments = segment_metrics(
            challenger.model_name, challenger.version, settings=settings
        )
        champion_segments = segment_metrics(
            champion.model_name, champion.version, settings=settings
        )
        regressions: list[str] = []
        judged = 0
        for segment, (value, _) in challenger_segments.items():
            if segment == "overall" or segment not in champion_segments:
                continue
            champion_value, _ = champion_segments[segment]
            positives = challenger_segments[segment][1]
            if positives < MIN_SEGMENT_POSITIVES:
                continue  # not enough evidence to judge; reported below
            judged += 1
            drop = champion_value - value
            if drop > settings.max_segment_regression:
                regressions.append(f"{segment} {value:.4f} vs {champion_value:.4f} (-{drop:.4f})")
        not_judged = [
            segment
            for segment, (_, positives) in challenger_segments.items()
            if segment != "overall" and positives < MIN_SEGMENT_POSITIVES
        ]
        detail = f"{judged} segment(s) judged; " + (
            f"regressions: {', '.join(regressions)}" if regressions else "none regressed"
        )
        if not_judged:
            detail += f"; not evaluable (under {MIN_SEGMENT_POSITIVES} positives): " + ", ".join(
                sorted(not_judged)
            )
        decision.checks.append(
            Check(
                name="no_segment_regression",
                passed=not regressions,
                detail=detail,
                threshold=settings.max_segment_regression,
            )
        )

    # 4. Calibration
    brier = float(challenger.metrics.get("test_brier", 1.0))
    decision.checks.append(
        Check(
            name="calibration",
            passed=brier <= settings.max_brier_score,
            detail=f"Brier {brier:.4f}, ceiling {settings.max_brier_score:.4f}",
            value=brier,
            threshold=settings.max_brier_score,
        )
    )

    # 5. Latency
    if latency_sample is None or latency_sample.empty:
        decision.checks.append(
            Check(name="latency", passed=True, detail="no sample supplied", skipped=True)
        )
    else:
        p95 = _measure_latency(challenger, latency_sample)
        decision.checks.append(
            Check(
                name="latency",
                passed=p95 <= settings.max_p95_latency_ms,
                detail=f"p95 {p95:.1f} ms, budget {settings.max_p95_latency_ms:.0f} ms",
                value=p95,
                threshold=settings.max_p95_latency_ms,
            )
        )

    # 6. Worth more than doing nothing
    value_per_1000 = float(challenger.metrics.get("validation_expected_value_per_1000", 0.0))
    decision.checks.append(
        Check(
            name="positive_expected_value",
            passed=value_per_1000 > 0,
            detail=f"{value_per_1000:,.0f} EUR per 1000 accounts at the chosen threshold",
            value=value_per_1000,
            threshold=0.0,
        )
    )

    logger.info(
        "promotion gate evaluated",
        extra={
            "model": challenger.model_name,
            "version": challenger.version,
            "allowed": decision.allowed,
            "failed": [c.name for c in decision.failed],
        },
    )
    return decision


def promote(
    challenger: ModelRecord,
    settings: Settings | None = None,
    *,
    latency_sample: pd.DataFrame | None = None,
    force: bool = False,
) -> tuple[ModelRecord, GateDecision]:
    """Promote a candidate, or refuse and say why."""
    settings = settings or get_settings()
    decision = evaluate_gate(challenger, settings, latency_sample=latency_sample)

    if not decision.allowed and not force:
        set_stage(
            challenger.model_name,
            challenger.version,
            "rejected",
            notes=decision.summary,
            settings=settings,
        )
        decision.raise_if_blocked()

    note = decision.summary if not force else f"FORCED: {decision.summary}"
    if force and not decision.allowed:
        logger.warning(
            "promotion gate overridden",
            extra={
                "model": challenger.model_name,
                "version": challenger.version,
                "failed": [c.name for c in decision.failed],
            },
        )
    record = set_stage(
        challenger.model_name, challenger.version, "production", notes=note, settings=settings
    )
    return record, decision
