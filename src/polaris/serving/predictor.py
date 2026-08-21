"""Scoring an account, and saying why.

Two responsibilities that are usually split across three files and get out of
step: producing the probability, and producing the sentence a customer success
manager reads underneath it.

**The explanation method.** For each *group* of related features -- usage,
support, billing, sentiment and so on -- the account's values are replaced by
typical ones, the training medians, and the model is asked again. The drop in
probability is that group's contribution for this account. All eight perturbed
rows go through in a single batched call, so an explanation costs one extra
prediction rather than one per feature.

Groups rather than individual features, and the reason is worth stating
because the first implementation did it per feature and was wrong. An account
with 0.14 average active users and zero sessions is coherent. The same account
with the median eight active users and still zero sessions is not -- the model
has never seen that contradiction and scores it as *more* risky, which flips
the sign and tells a customer success manager the opposite of the truth.
Perturbing the whole usage group at once keeps the row plausible.

It is a local sensitivity, not a Shapley value, and the difference matters:
contributions do not sum to the prediction, and two correlated groups will
each claim part of the credit. It is used anyway because it is honest about
what it measures, needs no extra dependency, and answers the question actually
being asked -- which part of this account's behaviour is driving the score.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from polaris.config import Settings, get_settings
from polaris.db.engine import get_engine
from polaris.economics import Economics
from polaris.exceptions import DatabaseError, ModelNotFound, ServingError
from polaris.features.definitions import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    FEATURE_VERSION,
    GROUP_LABELS,
)
from polaris.logging_config import get_logger
from polaris.registry.store import ModelRecord, get_production

logger = get_logger(__name__)

# How many groups an explanation names. There are eight in total and the tail
# is usually flat, so five is what fits in a panel without hiding anything.
EXPLAIN_TOP_K = 5


def _plain(value: Any) -> Any:
    """A JSON-serialisable version of a numpy or pandas scalar.

    pandas hands back numpy types, and pydantic refuses to serialise an
    ``int64`` -- a failure that appears at the API boundary rather than where
    the value was produced, and is therefore worth converting once here.
    """
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


@dataclass(frozen=True)
class Contribution:
    """One group's effect on this account's score."""

    group: str
    label: str
    contribution: float
    # The group's most extreme member, as evidence rather than as an
    # independent claim: it is what a reader wants to see next, and it is not
    # a second measurement.
    driver: str | None = None
    driver_value: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "label": self.label,
            "contribution": round(self.contribution, 5),
            "driver": self.driver,
            "driver_value": _plain(self.driver_value),
        }


@dataclass(frozen=True)
class Prediction:
    account_id: str
    reference_date: dt.date
    probability: float
    decision: bool
    threshold: float
    expected_value_eur: float
    model_name: str
    version: int
    contributions: list[Contribution] = field(default_factory=list)
    latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "reference_date": self.reference_date.isoformat(),
            "probability": round(self.probability, 6),
            "decision": self.decision,
            "threshold": round(self.threshold, 5),
            "expected_value_eur": round(self.expected_value_eur, 2),
            "model": {"name": self.model_name, "version": self.version},
            "contributions": [c.as_dict() for c in self.contributions],
            "latency_ms": round(self.latency_ms, 2),
        }


class Predictor:
    """A loaded production model, ready to answer."""

    def __init__(self, record: ModelRecord, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.record = record
        payload = record.load()

        if payload.get("feature_version") != FEATURE_VERSION:
            raise ServingError(
                f"{record.model_name} v{record.version} was trained on feature version "
                f"{payload.get('feature_version')}, but this process computes "
                f"{FEATURE_VERSION}. Serving it would score one thing and predict another."
            )

        self.model = payload["model"]
        self.threshold = float(payload.get("threshold", record.decision_threshold))
        self.baseline: dict[str, Any] = payload.get("feature_baseline", {})
        self.algorithm = payload.get("algorithm", "unknown")
        self.economics = Economics.from_settings(self.settings)

    @classmethod
    def from_production(
        cls, model_name: str = "churn-60d", settings: Settings | None = None
    ) -> Predictor:
        settings = settings or get_settings()
        record = get_production(model_name, settings)
        if record is None:
            raise ModelNotFound(f"no production model for {model_name!r}; train one and promote it")
        return cls(record, settings)

    def _driver(self, row: pd.Series, group: str) -> tuple[str | None, Any]:
        """The member of a group furthest from typical, in median absolute deviations.

        Evidence for the group's contribution, not a measurement of its own.
        """
        best_feature: str | None = None
        best_score = 0.0
        best_value: Any = None
        for feature in FEATURE_GROUPS[group]:
            if feature not in row.index or feature not in self.baseline:
                continue
            value, typical = row[feature], self.baseline[feature]
            if not isinstance(typical, (int, float)) or pd.isna(value):
                continue
            scale = abs(float(typical)) or 1.0
            score = abs(float(value) - float(typical)) / scale
            if score > best_score:
                best_feature, best_score, best_value = feature, score, value
        return best_feature, best_value

    def _explain(self, row: pd.DataFrame, probability: float) -> list[Contribution]:
        if not self.baseline:
            return []
        groups = [g for g in FEATURE_GROUPS if any(f in row.columns for f in FEATURE_GROUPS[g])]
        if not groups:
            return []

        perturbed = pd.concat([row] * len(groups), ignore_index=True)
        for index, group in enumerate(groups):
            for feature in FEATURE_GROUPS[group]:
                if feature in perturbed.columns and feature in self.baseline:
                    perturbed.loc[index, feature] = self.baseline[feature]

        try:
            probabilities = self.model.predict_proba(perturbed)[:, 1]
        except Exception as exc:
            logger.warning("explanation failed", extra={"error": str(exc)})
            return []

        single = row.iloc[0]
        contributions = []
        for index, group in enumerate(groups):
            driver, driver_value = self._driver(single, group)
            contributions.append(
                Contribution(
                    group=group,
                    label=GROUP_LABELS.get(group, group),
                    contribution=float(probability - probabilities[index]),
                    driver=driver,
                    driver_value=driver_value,
                )
            )
        contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
        return contributions[:EXPLAIN_TOP_K]

    def predict(
        self, features: pd.DataFrame, *, explain: bool = True, log: bool = True
    ) -> list[Prediction]:
        """Score a frame of feature rows."""
        missing = [c for c in FEATURE_NAMES if c not in features.columns]
        if missing:
            raise ServingError(f"missing feature column(s): {', '.join(missing)}")

        start = time.perf_counter()
        X = features[list(FEATURE_NAMES)]
        probabilities = self.model.predict_proba(X)[:, 1]
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(1, len(features))

        predictions: list[Prediction] = []
        for index in range(len(features)):
            probability = float(probabilities[index])
            row = X.iloc[[index]]
            contributions = self._explain(row, probability) if explain else []
            decision = probability >= self.threshold
            expected = (
                probability * self.economics.value_of_true_positive
                - (1 - probability) * self.economics.cost_of_false_positive
            )
            predictions.append(
                Prediction(
                    account_id=str(features.iloc[index]["account_id"]),
                    reference_date=features.iloc[index]["reference_date"],
                    probability=probability,
                    decision=bool(decision),
                    threshold=self.threshold,
                    expected_value_eur=float(expected),
                    model_name=self.record.model_name,
                    version=self.record.version,
                    contributions=contributions,
                    latency_ms=elapsed_ms,
                )
            )

        if log:
            self.log_predictions(predictions)
        return predictions

    def log_predictions(self, predictions: list[Prediction]) -> int:
        """Write every prediction down.

        A prediction nobody recorded cannot be monitored for drift, cannot be
        explained back to the person who acted on it, and cannot be joined to
        the outcome when the label arrives sixty days later.
        """
        if not predictions:
            return 0
        import json

        schema = self.settings.ml_schema
        rows = [
            {
                "model_name": p.model_name,
                "version": p.version,
                "account_id": p.account_id,
                "reference_date": p.reference_date,
                "probability": p.probability,
                "decision": p.decision,
                "threshold": round(p.threshold, 5),
                "expected_value_eur": round(p.expected_value_eur, 2),
                "top_features": json.dumps([c.as_dict() for c in p.contributions], default=str),
                "latency_ms": round(p.latency_ms, 2),
            }
            for p in predictions
        ]
        try:
            with get_engine(self.settings).begin() as conn:
                conn.execute(
                    text(
                        f"""INSERT INTO {schema}.prediction
                                (model_name, version, account_id, reference_date, probability,
                                 decision, threshold, expected_value_eur, top_features, latency_ms)
                            VALUES (:model_name, :version, :account_id, :reference_date,
                                    :probability, :decision, :threshold, :expected_value_eur,
                                    CAST(:top_features AS JSONB), :latency_ms)"""
                    ),
                    rows,
                )
        except SQLAlchemyError as exc:
            raise DatabaseError(f"logging predictions failed: {exc}") from exc
        return len(rows)
