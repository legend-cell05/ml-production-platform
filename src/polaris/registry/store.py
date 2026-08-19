"""The model registry.

A model is not a file. A model is a file, plus the threshold it is meant to be
used with, plus the feature version it was trained against, plus the evidence
that it works, plus a statement about whether it is the one currently serving
traffic. Keeping those five things together is the entire job of this module,
and keeping them apart is how a team ends up serving a model with somebody
else's threshold.

One rule is enforced by the database rather than by convention: at most one
version per model may be in `production`, via a partial unique index. A
convention would be broken during an incident, at night, by someone in a
hurry -- which is exactly when it matters.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from polaris.config import Settings, get_settings
from polaris.db.engine import get_engine
from polaris.exceptions import DatabaseError, ModelNotFound
from polaris.logging_config import get_logger
from polaris.training.train import TrainingResult

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelRecord:
    """One registered version."""

    model_name: str
    version: int
    stage: str
    run_id: str
    artifact_path: str
    decision_threshold: float
    metrics: dict[str, Any]
    created_at: dt.datetime
    promoted_at: dt.datetime | None = None
    notes: str | None = None

    @property
    def feature_version(self) -> str:
        return str(self.metrics.get("feature_version", "unknown"))

    @property
    def pr_auc(self) -> float | None:
        value = self.metrics.get("test_pr_auc")
        return float(value) if value is not None else None

    def load(self) -> Any:
        """Load the artefact from disk.

        Returns the whole payload rather than the estimator, because the
        threshold and the feature version travel with it -- a model loaded
        without them is a model somebody will use with the wrong ones.
        """
        path = Path(self.artifact_path)
        if not path.is_file():
            raise ModelNotFound(f"artefact missing for {self.model_name} v{self.version}: {path}")
        return joblib.load(path)


def _row_to_record(row: Any) -> ModelRecord:
    metrics = row["metrics"]
    if isinstance(metrics, str):
        metrics = json.loads(metrics)
    return ModelRecord(
        model_name=row["model_name"],
        version=int(row["version"]),
        stage=row["stage"],
        run_id=str(row["run_id"]),
        artifact_path=row["artifact_path"],
        decision_threshold=float(row["decision_threshold"]),
        metrics=metrics or {},
        created_at=row["created_at"],
        promoted_at=row["promoted_at"],
        notes=row["notes"],
    )


def register(result: TrainingResult, settings: Settings | None = None) -> ModelRecord:
    """Record a trained model as a candidate, with its evaluations."""
    settings = settings or get_settings()
    schema = settings.ml_schema

    metrics: dict[str, Any] = {
        "feature_version": result.dataset.feature_version,
        "algorithm": result.algorithm,
        "dataset_fingerprint": result.dataset.fingerprint,
        "train_rows": len(result.dataset.train),
        "test_rows": len(result.dataset.test),
        "validation_expected_value_per_1000": result.threshold.expected_value_per_1000,
        **{f"test_{k}": v for k, v in result.test.overall.as_dict().items()},
        **{f"validation_{k}": v for k, v in result.validation.overall.as_dict().items()},
    }

    try:
        with get_engine(settings).begin() as conn:
            version: int = conn.execute(
                text(
                    f"""SELECT COALESCE(MAX(version), 0) + 1 FROM {schema}.model_version
                        WHERE model_name = :name"""
                ),
                {"name": result.model_name},
            ).scalar_one()

            conn.execute(
                text(
                    f"""INSERT INTO {schema}.model_version
                            (model_name, version, run_id, stage, artifact_path,
                             decision_threshold, metrics)
                        VALUES (:name, :version, CAST(:run_id AS UUID), 'candidate',
                                :artifact_path, :threshold, CAST(:metrics AS JSONB))"""
                ),
                {
                    "name": result.model_name,
                    "version": version,
                    "run_id": result.run_id,
                    "artifact_path": str(result.artifact_path),
                    "threshold": round(result.threshold.threshold, 5),
                    "metrics": json.dumps(metrics, default=str),
                },
            )

            evaluations = []
            for evaluation in (result.validation, result.test):
                for segment, metric, value, n_rows, n_positives in evaluation.rows():
                    if value != value:  # NaN: the slice had one class
                        continue
                    evaluations.append(
                        {
                            "name": result.model_name,
                            "version": version,
                            "split": evaluation.split,
                            "segment": segment,
                            "metric": metric,
                            "value": float(value),
                            "n_rows": n_rows,
                            "n_positives": n_positives,
                        }
                    )
            if evaluations:
                conn.execute(
                    text(
                        f"""INSERT INTO {schema}.evaluation
                                (model_name, version, split, segment, metric, value,
                                 n_rows, n_positives)
                            VALUES (:name, :version, :split, :segment, :metric, :value,
                                    :n_rows, :n_positives)
                            ON CONFLICT (model_name, version, split, segment, metric)
                            DO UPDATE SET value = EXCLUDED.value,
                                          n_positives = EXCLUDED.n_positives"""
                    ),
                    evaluations,
                )
    except SQLAlchemyError as exc:
        raise DatabaseError(f"registering the model failed: {exc}") from exc

    logger.info(
        "model registered",
        extra={"model": result.model_name, "version": version, "stage": "candidate"},
    )
    return get_version(result.model_name, int(version), settings)


def get_version(model_name: str, version: int, settings: Settings | None = None) -> ModelRecord:
    settings = settings or get_settings()
    schema = settings.ml_schema
    try:
        with get_engine(settings).connect() as conn:
            row = (
                conn.execute(
                    text(
                        f"""SELECT * FROM {schema}.model_version
                        WHERE model_name = :name AND version = :version"""
                    ),
                    {"name": model_name, "version": version},
                )
                .mappings()
                .first()
            )
    except SQLAlchemyError as exc:
        raise DatabaseError(f"reading the registry failed: {exc}") from exc
    if row is None:
        raise ModelNotFound(f"{model_name} v{version} is not registered")
    return _row_to_record(row)


def get_production(model_name: str, settings: Settings | None = None) -> ModelRecord | None:
    """The version currently serving, or None if nothing has been promoted."""
    settings = settings or get_settings()
    schema = settings.ml_schema
    try:
        with get_engine(settings).connect() as conn:
            row = (
                conn.execute(
                    text(
                        f"""SELECT * FROM {schema}.model_version
                        WHERE model_name = :name AND stage = 'production'"""
                    ),
                    {"name": model_name},
                )
                .mappings()
                .first()
            )
    except SQLAlchemyError as exc:
        raise DatabaseError(f"reading the registry failed: {exc}") from exc
    return _row_to_record(row) if row else None


def list_versions(
    model_name: str | None = None, settings: Settings | None = None, limit: int = 50
) -> list[ModelRecord]:
    settings = settings or get_settings()
    schema = settings.ml_schema
    try:
        with get_engine(settings).connect() as conn:
            rows = (
                conn.execute(
                    text(
                        f"""SELECT * FROM {schema}.model_version
                        WHERE (CAST(:name AS TEXT) IS NULL OR model_name = :name)
                        ORDER BY model_name, version DESC LIMIT :limit"""
                    ),
                    {"name": model_name, "limit": limit},
                )
                .mappings()
                .all()
            )
    except SQLAlchemyError as exc:
        raise DatabaseError(f"reading the registry failed: {exc}") from exc
    return [_row_to_record(row) for row in rows]


def segment_metrics(
    model_name: str,
    version: int,
    metric: str = "pr_auc",
    split: str = "test",
    settings: Settings | None = None,
) -> dict[str, tuple[float, int]]:
    """One metric per segment, with the number of positives behind it.

    The count is returned because a segment with four churns cannot support a
    promotion decision, and the gate needs to know that rather than compare
    two noisy numbers.
    """
    settings = settings or get_settings()
    schema = settings.ml_schema
    try:
        with get_engine(settings).connect() as conn:
            rows = conn.execute(
                text(
                    f"""SELECT segment, value, n_positives FROM {schema}.evaluation
                        WHERE model_name = :name AND version = :version
                          AND split = :split AND metric = :metric"""
                ),
                {"name": model_name, "version": version, "split": split, "metric": metric},
            ).all()
    except SQLAlchemyError as exc:
        raise DatabaseError(f"reading evaluations failed: {exc}") from exc
    return {row.segment: (float(row.value), int(row.n_positives)) for row in rows}


def set_stage(
    model_name: str,
    version: int,
    stage: str,
    *,
    notes: str | None = None,
    settings: Settings | None = None,
) -> ModelRecord:
    """Move a version to a stage, archiving the incumbent when promoting.

    Both writes happen in one transaction: a failure between them would leave
    either two production versions (which the index refuses) or none (which
    stops serving).
    """
    settings = settings or get_settings()
    schema = settings.ml_schema
    try:
        with get_engine(settings).begin() as conn:
            if stage == "production":
                conn.execute(
                    text(
                        f"""UPDATE {schema}.model_version
                            SET stage = 'archived', archived_at = now()
                            WHERE model_name = :name AND stage = 'production'
                              AND version <> :version"""
                    ),
                    {"name": model_name, "version": version},
                )
            conn.execute(
                text(
                    f"""UPDATE {schema}.model_version
                        SET stage = :stage,
                            promoted_at = CASE WHEN :stage = 'production' THEN now()
                                               ELSE promoted_at END,
                            notes = COALESCE(:notes, notes)
                        WHERE model_name = :name AND version = :version"""
                ),
                {"name": model_name, "version": version, "stage": stage, "notes": notes},
            )
    except SQLAlchemyError as exc:
        raise DatabaseError(f"changing the stage failed: {exc}") from exc

    logger.info("stage changed", extra={"model": model_name, "version": version, "stage": stage})
    return get_version(model_name, version, settings)
