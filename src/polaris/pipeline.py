"""The platform as a set of operations.

One function per thing a person can ask for, each independently runnable. The
CLI is a thin shell over this module and contains no logic of its own, so what
CI exercises, what Docker runs and what an operator types all go through the
same code.

The order, and what each step depends on:

    simulate -> features -> train -> promote -> score -> monitor

``promote`` is the one that can refuse. Everything else either works or
raises.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import pandas as pd

from polaris.config import Settings, get_settings
from polaris.data.dataset import build_dataset, load_scoreable
from polaris.db.schema import initialise_database
from polaris.features.builder import BuildReport, build_all, reference_dates
from polaris.features.leakage import LeakageFinding, screen_frame
from polaris.generation import simulate, write_simulation
from polaris.logging_config import get_logger
from polaris.monitoring.drift import DriftResult, check_drift, prediction_drift, record_drift
from polaris.monitoring.performance import LivePerformance, live_performance, record_outcomes
from polaris.registry.promotion import GateDecision, evaluate_gate, promote
from polaris.registry.store import ModelRecord, get_production, register
from polaris.serving.predictor import Prediction, Predictor
from polaris.training.pipelines import Algorithm
from polaris.training.train import TrainingResult, train

logger = get_logger(__name__)


@dataclass
class CycleReport:
    """What a full train-promote-score cycle did."""

    training: TrainingResult
    record: ModelRecord
    decision: GateDecision
    promoted: bool
    scored: int = 0
    flagged: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.training.algorithm,
            "version": self.record.version,
            "promoted": self.promoted,
            "gate": self.decision.summary,
            **{k: round(v, 4) for k, v in self.training.headline.items()},
            "scored": self.scored,
            "flagged": self.flagged,
        }


def prepare(settings: Settings | None = None) -> dict[str, Any]:
    """Create the schemas and generate the simulated business."""
    settings = settings or get_settings()
    initialise_database(settings)
    result = simulate(settings)
    counts = write_simulation(result, settings)
    return {"rows": counts, "stats": result.stats}


def build_features(settings: Settings | None = None, *, rebuild: bool = True) -> BuildReport:
    settings = settings or get_settings()
    initialise_database(settings)
    return build_all(settings, rebuild=rebuild)


def screen(settings: Settings | None = None) -> list[LeakageFinding]:
    """Run the leakage screens over the labelled feature store."""
    settings = settings or get_settings()
    dataset = build_dataset(settings)
    return screen_frame(dataset.train.frame)


def train_and_register(
    algorithm: Algorithm = "gradient_boosting",
    settings: Settings | None = None,
    *,
    model_name: str = "churn-60d",
) -> tuple[TrainingResult, ModelRecord]:
    settings = settings or get_settings()
    result = train(algorithm, settings, model_name=model_name)
    record = register(result, settings)
    return result, record


def promote_candidate(
    record: ModelRecord,
    settings: Settings | None = None,
    *,
    latency_sample: pd.DataFrame | None = None,
    force: bool = False,
) -> tuple[ModelRecord, GateDecision]:
    return promote(record, settings, latency_sample=latency_sample, force=force)


def gate_for(record: ModelRecord, settings: Settings | None = None) -> GateDecision:
    return evaluate_gate(record, settings)


def score_batch(
    settings: Settings | None = None,
    *,
    reference_date: dt.date | None = None,
    limit: int | None = None,
    explain: bool = False,
    model_name: str = "churn-60d",
) -> list[Prediction]:
    """Score a reference date and record every prediction."""
    settings = settings or get_settings()
    predictor = Predictor.from_production(model_name, settings)
    target = reference_date or reference_dates(settings)[-1]
    frame = load_scoreable(target, settings)
    if limit:
        frame = frame.head(limit)
    predictions = predictor.predict(frame, explain=explain)
    predictions.sort(key=lambda p: p.probability, reverse=True)
    return predictions


def monitor(settings: Settings | None = None, *, model_name: str = "churn-60d") -> dict[str, Any]:
    """Drift while the labels are pending, performance once they arrive."""
    settings = settings or get_settings()
    record = get_production(model_name, settings)
    if record is None:
        return {"error": "no production model"}

    dataset = build_dataset(settings)
    drift: list[DriftResult] = check_drift(dataset.train.frame, dataset.test.frame, settings)
    record_drift(
        drift,
        model_name=record.model_name,
        version=record.version,
        baseline_period=(dataset.train.start, dataset.train.end),
        current_period=(dataset.test.start, dataset.test.end),
        settings=settings,
    )
    record_outcomes(settings)
    performance: LivePerformance | None = live_performance(
        record.model_name, record.version, settings
    )
    return {
        "model": f"{record.model_name} v{record.version}",
        "drift": drift,
        "prediction_drift": prediction_drift(record.model_name, record.version, settings),
        "performance": performance,
    }


def full_cycle(
    algorithm: Algorithm = "gradient_boosting",
    settings: Settings | None = None,
    *,
    force: bool = False,
    score_limit: int = 2000,
) -> CycleReport:
    """Train, register, try to promote, and score if promoted."""
    settings = settings or get_settings()
    result, record = train_and_register(algorithm, settings)
    sample = result.dataset.test.X.head(200)

    promoted = False
    try:
        record, decision = promote_candidate(record, settings, latency_sample=sample, force=force)
        promoted = True
    except Exception as exc:
        from polaris.exceptions import PromotionBlocked

        if not isinstance(exc, PromotionBlocked):
            raise
        decision = evaluate_gate(record, settings, latency_sample=sample)
        logger.warning("promotion refused", extra={"reason": str(exc)})

    report = CycleReport(training=result, record=record, decision=decision, promoted=promoted)
    if promoted:
        predictions = score_batch(settings, limit=score_limit)
        report.scored = len(predictions)
        report.flagged = sum(1 for p in predictions if p.decision)
    return report
