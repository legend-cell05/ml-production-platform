"""Drift, and why it is the only thing watching between labels.

A churn model predicts sixty days ahead, so its accuracy cannot be measured
for sixty days. That is a long time to be flying on nothing, and it is the
reason drift monitoring exists: it is the only signal available in the gap.

Population Stability Index, on the conventional thresholds:

    PSI < 0.10   the population has not moved
    0.10 - 0.25  it has moved enough to look at
    PSI > 0.25   it has moved enough to distrust the model

The bins come from the **training** distribution and are then frozen. Binning
each period independently would compare two sets of quantiles and report zero
drift no matter what happened -- the most common way a PSI implementation is
quietly wrong.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from polaris.config import Settings, get_settings
from polaris.db.engine import get_engine
from polaris.exceptions import DatabaseError
from polaris.features.definitions import CATEGORICAL_FEATURES, FEATURE_NAMES
from polaris.logging_config import get_logger

logger = get_logger(__name__)

# Added to every bin so that a category present in one period and absent in
# the other produces a large number rather than an infinite one.
EPSILON = 1e-6
DEFAULT_BINS = 10


@dataclass(frozen=True)
class DriftResult:
    feature: str
    psi: float
    status: str
    baseline_n: int
    current_n: int
    detail: str = ""

    @property
    def alerting(self) -> bool:
        return self.status == "ALERT"


def population_stability_index(
    baseline: pd.Series, current: pd.Series, *, bins: int = DEFAULT_BINS
) -> tuple[float, str]:
    """PSI between two samples of one feature, plus how the bins were made."""
    baseline = baseline.dropna()
    current = current.dropna()
    if len(baseline) < 50 or len(current) < 50:
        return float("nan"), "too few rows"

    if baseline.dtype == object or str(baseline.dtype).startswith("category"):
        categories = sorted(set(baseline.unique()) | set(current.unique()))
        expected = np.array([(baseline == c).mean() for c in categories]) + EPSILON
        actual = np.array([(current == c).mean() for c in categories]) + EPSILON
        detail = f"{len(categories)} categories"
    else:
        # Bins frozen from the baseline. Re-binning the current period would
        # compare a distribution to itself.
        edges = np.unique(np.quantile(baseline.astype(float), np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0, "constant in the baseline"
        edges[0], edges[-1] = -np.inf, np.inf
        expected = np.histogram(baseline.astype(float), bins=edges)[0] / len(baseline) + EPSILON
        actual = np.histogram(current.astype(float), bins=edges)[0] / len(current) + EPSILON
        detail = f"{len(edges) - 1} bins"

    psi = float(np.sum((actual - expected) * np.log(actual / expected)))
    return psi, detail


def check_drift(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    settings: Settings | None = None,
) -> list[DriftResult]:
    """PSI for every feature, worst first."""
    settings = settings or get_settings()
    results: list[DriftResult] = []
    for feature in FEATURE_NAMES:
        if feature not in baseline.columns or feature not in current.columns:
            continue
        left, right = baseline[feature], current[feature]
        if feature in CATEGORICAL_FEATURES:
            left, right = left.astype(str), right.astype(str)
        psi, detail = population_stability_index(left, right)
        if np.isnan(psi):
            continue
        status = (
            "ALERT" if psi >= settings.psi_alert else "WARN" if psi >= settings.psi_warn else "OK"
        )
        results.append(
            DriftResult(
                feature=feature,
                psi=psi,
                status=status,
                baseline_n=int(left.notna().sum()),
                current_n=int(right.notna().sum()),
                detail=detail,
            )
        )
    results.sort(key=lambda r: r.psi, reverse=True)
    logger.info(
        "drift check complete",
        extra={
            "features": len(results),
            "alerting": sum(1 for r in results if r.alerting),
            "warning": sum(1 for r in results if r.status == "WARN"),
        },
    )
    return results


def record_drift(
    results: list[DriftResult],
    *,
    model_name: str,
    version: int,
    baseline_period: tuple[dt.date, dt.date],
    current_period: tuple[dt.date, dt.date],
    settings: Settings | None = None,
) -> int:
    settings = settings or get_settings()
    if not results:
        return 0
    schema = settings.ml_schema
    rows = [
        {
            "model_name": model_name,
            "version": version,
            "feature": r.feature,
            "psi": r.psi,
            "status": r.status,
            "baseline_start": baseline_period[0],
            "baseline_end": baseline_period[1],
            "current_start": current_period[0],
            "current_end": current_period[1],
        }
        for r in results
    ]
    try:
        with get_engine(settings).begin() as conn:
            conn.execute(
                text(
                    f"""INSERT INTO {schema}.drift_check
                            (model_name, version, feature, psi, status,
                             baseline_start, baseline_end, current_start, current_end)
                        VALUES (:model_name, :version, :feature, :psi, :status,
                                :baseline_start, :baseline_end, :current_start, :current_end)"""
                ),
                rows,
            )
    except SQLAlchemyError as exc:
        raise DatabaseError(f"recording drift failed: {exc}") from exc
    return len(rows)


def prediction_drift(
    model_name: str, version: int, settings: Settings | None = None
) -> dict[str, Any]:
    """How the scores themselves have moved between the last two scoring runs.

    Cheaper than feature drift and often faster to react: a model whose mean
    predicted probability has doubled is telling you something before any
    individual feature crosses a threshold.

    The comparison is between **reference dates**, not between wall-clock
    windows. Scoring the whole history in an afternoon -- which is what a
    backfill or a demo does -- gives every prediction the same ``scored_at``,
    so a wall-clock comparison has nothing to compare and reports a mean of
    zero. Reporting zero where the answer is "there is only one batch" is the
    kind of monitoring bug that gets believed.
    """
    settings = settings or get_settings()
    schema = settings.ml_schema
    try:
        with get_engine(settings).connect() as conn:
            batches = (
                conn.execute(
                    text(
                        f"""SELECT reference_date,
                                   AVG(probability) AS mean_probability,
                                   AVG(CASE WHEN decision THEN 1.0 ELSE 0.0 END) AS flag_rate,
                                   COUNT(*) AS n
                              FROM {schema}.prediction
                             WHERE model_name = :name AND version = :version
                             GROUP BY reference_date
                             ORDER BY reference_date DESC
                             LIMIT 2"""
                    ),
                    {"name": model_name, "version": version},
                )
                .mappings()
                .all()
            )
            total: int = (
                conn.execute(
                    text(
                        f"""SELECT COUNT(*) FROM {schema}.prediction
                             WHERE model_name = :name AND version = :version"""
                    ),
                    {"name": model_name, "version": version},
                ).scalar_one()
                or 0
            )
    except SQLAlchemyError as exc:
        raise DatabaseError(f"reading prediction drift failed: {exc}") from exc

    if not batches:
        return {"predictions": 0, "status": "nothing scored yet"}

    recent = batches[0]
    result: dict[str, Any] = {
        "reference_date": str(recent["reference_date"]),
        "recent_mean_probability": round(float(recent["mean_probability"]), 5),
        "recent_flag_rate": round(float(recent["flag_rate"]), 5),
        "batch_size": int(recent["n"]),
        "predictions": int(total),
    }
    if len(batches) == 1:
        # No second batch: say so, rather than dividing by a zero that was
        # never a measurement.
        result["status"] = "one batch only -- no comparison available"
        return result

    earlier = batches[1]
    earlier_mean = float(earlier["mean_probability"])
    result["compared_with"] = str(earlier["reference_date"])
    result["earlier_mean_probability"] = round(earlier_mean, 5)
    result["ratio"] = (
        round(float(recent["mean_probability"]) / earlier_mean, 4) if earlier_mean else None
    )
    result["status"] = "compared with the previous scoring run"
    return result
