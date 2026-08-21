"""Measuring the model on the outcomes, once they arrive.

Sixty days after every prediction, the answer becomes knowable. This module
joins the two and reports what the model actually achieved -- which is a
different question from what it scored on a test set, and the only one that
matters after go-live.

The lag is the point. Live performance is always one horizon behind, so a
model that broke today will be measurably broken in two months. That is why
drift monitoring exists alongside this, and why neither replaces the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from polaris.config import Settings, get_settings
from polaris.db.engine import get_engine
from polaris.economics import Economics
from polaris.exceptions import DatabaseError
from polaris.logging_config import get_logger
from polaris.training.evaluate import compute_metrics

logger = get_logger(__name__)


@dataclass(frozen=True)
class LivePerformance:
    """What the model achieved on predictions whose labels are now known."""

    model_name: str
    version: int
    matched: int
    positives: int
    pr_auc: float
    roc_auc: float
    brier: float
    precision_at_threshold: float
    recall_at_threshold: float
    realised_value_eur: float
    pending_labels: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": f"{self.model_name} v{self.version}",
            "predictions_with_labels": self.matched,
            "positives": self.positives,
            "pr_auc": round(self.pr_auc, 4),
            "roc_auc": round(self.roc_auc, 4),
            "brier": round(self.brier, 5),
            "precision_at_threshold": round(self.precision_at_threshold, 4),
            "recall_at_threshold": round(self.recall_at_threshold, 4),
            "realised_value_eur": round(self.realised_value_eur, 2),
            "predictions_awaiting_labels": self.pending_labels,
        }


def record_outcomes(settings: Settings | None = None) -> int:
    """Copy labels from the feature store into the outcome table as they mature.

    The feature store already knows: a row's label stops being NULL once the
    horizon has elapsed. Copying rather than joining across schemas keeps the
    monitoring query simple and keeps a record of when each outcome was
    observed, which is what makes the reporting lag measurable rather than
    assumed.
    """
    settings = settings or get_settings()
    try:
        with get_engine(settings).begin() as conn:
            result = conn.execute(
                text(
                    f"""INSERT INTO {settings.ml_schema}.outcome
                            (account_id, reference_date, churned, observed_at)
                        SELECT account_id, reference_date, churned_in_horizon, label_known_at
                        FROM {settings.feature_schema}.churn_features
                        WHERE churned_in_horizon IS NOT NULL
                        ON CONFLICT (account_id, reference_date) DO NOTHING"""
                )
            )
    except SQLAlchemyError as exc:
        raise DatabaseError(f"recording outcomes failed: {exc}") from exc
    written = int(result.rowcount or 0)
    logger.info("outcomes recorded", extra={"rows": written})
    return written


def live_performance(
    model_name: str, version: int, settings: Settings | None = None
) -> LivePerformance | None:
    """Score the predictions whose outcome is now known."""
    settings = settings or get_settings()
    schema = settings.ml_schema
    try:
        with get_engine(settings).connect() as conn:
            rows = conn.execute(
                text(
                    f"""SELECT p.probability, p.decision, o.churned
                        FROM {schema}.prediction p
                        JOIN {schema}.outcome o
                          ON o.account_id = p.account_id
                         AND o.reference_date = p.reference_date
                        WHERE p.model_name = :name AND p.version = :version"""
                ),
                {"name": model_name, "version": version},
            ).all()
            pending: int = conn.execute(
                text(
                    f"""SELECT COUNT(*) FROM {schema}.prediction p
                        LEFT JOIN {schema}.outcome o
                          ON o.account_id = p.account_id
                         AND o.reference_date = p.reference_date
                        WHERE p.model_name = :name AND p.version = :version
                          AND o.account_id IS NULL"""
                ),
                {"name": model_name, "version": version},
            ).scalar_one()
    except SQLAlchemyError as exc:
        raise DatabaseError(f"reading live performance failed: {exc}") from exc

    if not rows:
        return None

    probabilities = np.array([float(r.probability) for r in rows])
    decisions = np.array([bool(r.decision) for r in rows])
    labels = np.array([1 if r.churned else 0 for r in rows])

    metrics = compute_metrics(labels, probabilities)
    economics = Economics.from_settings(settings)
    true_positives = int(np.sum(decisions & (labels == 1)))
    false_positives = int(np.sum(decisions & (labels == 0)))
    flagged = int(decisions.sum())
    positives = int(labels.sum())

    performance = LivePerformance(
        model_name=model_name,
        version=version,
        matched=len(rows),
        positives=positives,
        pr_auc=metrics.pr_auc,
        roc_auc=metrics.roc_auc,
        brier=metrics.brier,
        precision_at_threshold=true_positives / flagged if flagged else 0.0,
        recall_at_threshold=true_positives / positives if positives else 0.0,
        realised_value_eur=(
            true_positives * economics.value_of_true_positive
            - false_positives * economics.cost_of_false_positive
        ),
        pending_labels=int(pending),
    )
    logger.info("live performance computed", extra=performance.as_dict())
    return performance
