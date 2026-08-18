"""Screening the feature store for leakage.

This module exists because of a bug in this repository's own data generator.

The simulator originally set an account's renewal date only for accounts that
had *not* churned. Nothing in the feature SQL was wrong; the source data was.
``days_to_renewal`` was simply missing for every account that eventually left,
so "no renewal date" was a perfect proxy for the label. The first model
reached a ROC-AUC of 0.995 and a linear coefficient of 7.8 on that one column.

A number that good is not good news. It is the single most reliable signal
that something is wrong, and the only reliable defence is a check that runs
every time the feature store is built.

Three screens, each catching a different shape of the same mistake:

* **value** -- one feature predicts the label almost perfectly;
* **missingness** -- whether a feature is *present* predicts the label. This
  is the one that catches the renewal-date bug, because the values themselves
  were innocent;
* **timing** -- rebuilding an old reference date after new data arrives
  changes its features, which means those features were reading the future.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from polaris.features.definitions import FEATURE_NAMES, LABEL_COLUMN
from polaris.logging_config import get_logger

logger = get_logger(__name__)

Verdict = Literal["OK", "REVIEW", "LEAK"]

# A single feature above this is treated as a leak. It is set high on purpose:
# in this dataset usage and error-rate decline are genuinely strong signals and
# the strongest of them reaches 0.873 on its own, so a lower bar would cry wolf
# on the features the model is meant to rely on. Above 0.95, no honest business feature explains the label that
# well sixty days ahead.
LEAK_AUC = 0.95
REVIEW_AUC = 0.90

# Missingness is held to a stricter standard, because a feature whose mere
# presence predicts the outcome is almost never legitimate.
LEAK_MISSING_AUC = 0.80
REVIEW_MISSING_AUC = 0.70

# AUC on its own is not enough here, and the renewal-date bug is why. Fifty-nine
# per cent of accounts legitimately have no renewal date -- they are on monthly
# contracts -- and that legitimate missingness dilutes the AUC into the "worth
# a look" band rather than "stop". The ratio between the label rate among
# missing values and among present ones is not diluted by it: on a frame where
# the date is missing for exactly the churners it runs into the thousands, while
# the legitimate case in the current store sits at 3.4x. Both criteria run; the
# worse verdict wins.
LEAK_MISSING_LIFT = 8.0
REVIEW_MISSING_LIFT = 3.0
MIN_GROUP = 200


@dataclass(frozen=True)
class LeakageFinding:
    feature: str
    screen: str
    score: float
    verdict: Verdict
    detail: str

    @property
    def blocking(self) -> bool:
        return self.verdict == "LEAK"


def _directionless_auc(values: pd.Series, labels: pd.Series) -> float | None:
    """AUC of a single column, ignoring direction.

    A feature that predicts the negative class perfectly is exactly as leaky
    as one that predicts the positive class perfectly, so the score is folded
    around 0.5.
    """
    mask = values.notna()
    if mask.sum() < 100 or labels[mask].nunique() < 2 or values[mask].nunique() < 2:
        return None
    try:
        score = roc_auc_score(labels[mask].astype(int), values[mask].astype(float))
    except ValueError:  # pragma: no cover - guarded above, kept for safety
        return None
    return float(max(score, 1.0 - score))


def screen_values(frame: pd.DataFrame) -> list[LeakageFinding]:
    """Is any single feature almost sufficient on its own?"""
    findings: list[LeakageFinding] = []
    labels = frame[LABEL_COLUMN]
    for name in FEATURE_NAMES:
        if name not in frame.columns:
            continue
        column = frame[name]
        if column.dtype == object or isinstance(column.dtype, pd.CategoricalDtype):
            continue  # categoricals are screened by missingness only
        score = _directionless_auc(pd.to_numeric(column, errors="coerce"), labels)
        if score is None:
            continue
        verdict: Verdict = (
            "LEAK" if score >= LEAK_AUC else "REVIEW" if score >= REVIEW_AUC else "OK"
        )
        if verdict != "OK":
            findings.append(
                LeakageFinding(
                    feature=name,
                    screen="value",
                    score=score,
                    verdict=verdict,
                    detail=f"single-feature AUC {score:.3f}",
                )
            )
    return findings


def screen_missingness(frame: pd.DataFrame) -> list[LeakageFinding]:
    """Does whether a feature is present predict the label?

    The screen that would have caught the renewal-date bug on the first run.
    """
    findings: list[LeakageFinding] = []
    labels = frame[LABEL_COLUMN].astype(int)
    for name in FEATURE_NAMES:
        if name not in frame.columns:
            continue
        present = frame[name].notna().astype(int)
        if present.nunique() < 2 or present.mean() > 0.999 or present.mean() < 0.001:
            continue
        score = _directionless_auc(present.astype(float), labels)
        if score is None:
            continue

        by_auc: Verdict = (
            "LEAK"
            if score >= LEAK_MISSING_AUC
            else "REVIEW"
            if score >= REVIEW_MISSING_AUC
            else "OK"
        )

        missing_mask = present == 0
        n_missing, n_present = int(missing_mask.sum()), int((~missing_mask).sum())
        by_lift: Verdict = "OK"
        lift = 1.0
        lift_measured = n_missing >= MIN_GROUP and n_present >= MIN_GROUP
        if lift_measured:
            rate_missing = float(labels[missing_mask].mean())
            rate_present = float(labels[~missing_mask].mean())
            epsilon = 1.0 / len(labels)
            lift = (rate_missing + epsilon) / (rate_present + epsilon)
            extremity = max(lift, 1.0 / lift)
            by_lift = (
                "LEAK"
                if extremity >= LEAK_MISSING_LIFT
                else "REVIEW"
                if extremity >= REVIEW_MISSING_LIFT
                else "OK"
            )

        order: dict[Verdict, int] = {"OK": 0, "REVIEW": 1, "LEAK": 2}
        verdict: Verdict = by_auc if order[by_auc] >= order[by_lift] else by_lift
        if verdict != "OK":
            missing_rate = 1.0 - float(present.mean())
            extremity = max(lift, 1.0 / max(lift, 1e-9))
            # A lift of "1.0x" when the groups were too small to compare would
            # read as "the label rate is identical", which is a measurement
            # nobody made. Say which of the two criteria actually ran.
            lift_phrase = (
                f"the label rate is {extremity:.1f}x different between missing and present"
                if lift_measured
                else (f"the label rates were not compared (one side has under {MIN_GROUP} rows)")
            )
            findings.append(
                LeakageFinding(
                    feature=name,
                    screen="missingness",
                    score=max(score, min(1.0, 0.5 + 0.05 * extremity)),
                    verdict=verdict,
                    detail=(
                        f"presence alone separates the classes at AUC {score:.3f}; "
                        f"{lift_phrase} ({missing_rate:.1%} missing)"
                    ),
                )
            )
    return findings


def screen_frame(frame: pd.DataFrame) -> list[LeakageFinding]:
    """Both statistical screens, worst first."""
    if LABEL_COLUMN not in frame.columns or frame[LABEL_COLUMN].nunique() < 2:
        return []
    labelled = frame[frame[LABEL_COLUMN].notna()]
    findings = screen_values(labelled) + screen_missingness(labelled)
    findings.sort(key=lambda f: f.score, reverse=True)
    logger.info(
        "leakage screen complete",
        extra={
            "rows": len(labelled),
            "findings": len(findings),
            "blocking": sum(1 for f in findings if f.blocking),
        },
    )
    return findings


def stability_check(
    first: pd.DataFrame, second: pd.DataFrame, *, reference_date: dt.date
) -> list[LeakageFinding]:
    """Do a reference date's features change when later data arrives?

    A point-in-time feature is a statement about a moment that has already
    happened, so recomputing it tomorrow must produce the same number. If it
    does not, the query is reading data that did not exist at the time --
    which is leakage even when no single feature looks suspicious.
    """
    findings: list[LeakageFinding] = []
    key = ["account_id"]
    merged = first.merge(second, on=key, suffixes=("_before", "_after"))
    for name in FEATURE_NAMES:
        before, after = f"{name}_before", f"{name}_after"
        if before not in merged.columns or after not in merged.columns:
            continue
        left, right = merged[before], merged[after]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            differs = ~np.isclose(
                left.astype(float).fillna(-999.0), right.astype(float).fillna(-999.0)
            )
        else:
            differs = left.astype(str) != right.astype(str)
        changed = int(differs.sum())
        if changed:
            findings.append(
                LeakageFinding(
                    feature=name,
                    screen="timing",
                    score=changed / max(1, len(merged)),
                    verdict="LEAK",
                    detail=(
                        f"{changed} of {len(merged)} rows changed when {reference_date} "
                        "was rebuilt after later data arrived"
                    ),
                )
            )
    return findings
