"""Training a model, end to end and on the record.

The order of operations is the part worth reading, because each step uses a
different slice of the data and using the wrong one is how a model comes to
report a number it will never reproduce:

1. **fit** on train;
2. **calibrate** by cross-fitting on train -- not on validation, which is
   needed intact for the next step;
3. **choose the threshold** on validation, by expected value;
4. **evaluate** on test, once, and never touch it again.

Everything is recorded twice: in MLflow, where an experiment can be browsed
and compared, and in ``ml.training_run``, where the platform's own gates and
reports read it. The duplication is deliberate -- the registry must not
depend on a tracking server being reachable to answer "what is in production
and what was it trained on?".
"""

from __future__ import annotations

import datetime as dt
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from polaris.config import Settings, get_settings
from polaris.data.dataset import Dataset, build_dataset
from polaris.db.engine import get_engine
from polaris.economics import Economics, ThresholdChoice, baseline_values, choose_threshold
from polaris.exceptions import DatabaseError, TrainingError
from polaris.features.definitions import FEATURE_VERSION
from polaris.features.leakage import LeakageFinding, screen_frame
from polaris.logging_config import get_logger
from polaris.training.evaluate import Evaluation, calibration_table, evaluate_split
from polaris.training.pipelines import Algorithm, build_pipeline, default_params

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Everything one training run produced."""

    run_id: str
    mlflow_run_id: str | None
    model_name: str
    algorithm: Algorithm
    pipeline: Pipeline
    dataset: Dataset
    threshold: ThresholdChoice
    validation: Evaluation
    test: Evaluation
    artifact_path: Path
    leakage: list[LeakageFinding] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def headline(self) -> dict[str, float]:
        return {
            "test_pr_auc": self.test.overall.pr_auc,
            "test_roc_auc": self.test.overall.roc_auc,
            "test_brier": self.test.overall.brier,
            "test_lift_at_100": self.test.overall.lift_at_100,
            "threshold": self.threshold.threshold,
            "validation_expected_value_per_1000": self.threshold.expected_value_per_1000,
        }


def _feature_baseline(dataset: Dataset) -> dict[str, Any]:
    """A typical value per feature, from the training split only.

    Numeric features use the median, categorical ones the mode. Used by the
    serving layer's explanation, never by the model itself.
    """
    baseline: dict[str, Any] = {}
    frame = dataset.train.X
    for column in frame.columns:
        series = frame[column]
        if series.dtype == object:
            mode = series.mode()
            baseline[column] = str(mode.iloc[0]) if len(mode) else None
        else:
            baseline[column] = float(series.median()) if series.notna().any() else 0.0
    return baseline


def _calibrated(pipeline: Pipeline, dataset: Dataset) -> Pipeline:
    """Cross-fitted calibration on the training data.

    Isotonic rather than sigmoid because gradient boosting with balanced class
    weights is badly miscalibrated in a specific, non-sigmoidal way -- the
    reweighting inflates every probability by roughly the weight ratio.

    Cross-fitted on **train** so that validation stays untouched for the
    threshold. Calibrating on validation and then choosing the threshold on
    the same rows would tune two things on one set and report the result as if
    it were out of sample.
    """
    calibrated = CalibratedClassifierCV(pipeline, method="isotonic", cv=3, ensemble=True)
    calibrated.fit(dataset.train.X, dataset.train.y)
    return calibrated


def _record_run(result: TrainingResult, settings: Settings, status: str) -> None:
    schema = settings.ml_schema
    metrics = {
        **{f"test_{k}": v for k, v in result.test.overall.as_dict().items()},
        **{f"validation_{k}": v for k, v in result.validation.overall.as_dict().items()},
        **result.threshold.as_dict(),
    }
    clean = {
        k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else v)
        for k, v in metrics.items()
    }
    try:
        with get_engine(settings).begin() as conn:
            conn.execute(
                text(
                    f"""INSERT INTO {schema}.training_run
                            (run_id, mlflow_run_id, model_name, algorithm, feature_version,
                             dataset_fingerprint, params, metrics, train_rows,
                             train_period, test_period, status, finished_at)
                        VALUES (CAST(:run_id AS UUID), :mlflow_run_id, :model_name, :algorithm,
                                :feature_version, :fingerprint, CAST(:params AS JSONB),
                                CAST(:metrics AS JSONB), :train_rows,
                                DATERANGE(:train_start, :train_end, '[]'),
                                DATERANGE(:test_start, :test_end, '[]'),
                                :status, now())
                        ON CONFLICT (run_id) DO UPDATE
                        SET metrics = EXCLUDED.metrics, status = EXCLUDED.status,
                            finished_at = EXCLUDED.finished_at"""
                ),
                {
                    "run_id": result.run_id,
                    "mlflow_run_id": result.mlflow_run_id,
                    "model_name": result.model_name,
                    "algorithm": result.algorithm,
                    "feature_version": result.dataset.feature_version,
                    "fingerprint": result.dataset.fingerprint,
                    "params": json.dumps(default_params(result.algorithm), default=str),
                    "metrics": json.dumps(clean, default=str),
                    "train_rows": len(result.dataset.train),
                    "train_start": result.dataset.train.start,
                    "train_end": result.dataset.train.end,
                    "test_start": result.dataset.test.start,
                    "test_end": result.dataset.test.end,
                    "status": status,
                },
            )
    except SQLAlchemyError as exc:
        raise DatabaseError(f"recording the training run failed: {exc}") from exc


def train(
    algorithm: Algorithm = "gradient_boosting",
    settings: Settings | None = None,
    *,
    model_name: str = "churn-60d",
    dataset: Dataset | None = None,
    screen_for_leakage: bool = True,
) -> TrainingResult:
    """Train, calibrate, threshold, evaluate and record one model."""
    settings = settings or get_settings()
    started = time.perf_counter()
    dataset = dataset or build_dataset(settings)

    if dataset.feature_version != FEATURE_VERSION:
        raise TrainingError(
            f"the feature store holds version {dataset.feature_version} but this code "
            f"expects {FEATURE_VERSION}; rebuild the features"
        )

    # The screen runs before training rather than after, so a leak is found
    # before anyone has a number to become attached to.
    findings: list[LeakageFinding] = []
    if screen_for_leakage:
        findings = screen_frame(dataset.train.frame)
        blocking = [f for f in findings if f.blocking]
        if blocking:
            raise TrainingError(
                "leakage screen blocked training: "
                + "; ".join(f"{f.feature} ({f.detail})" for f in blocking)
            )

    run_id = str(uuid.uuid4())
    economics = Economics.from_settings(settings)

    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    with mlflow.start_run(run_name=f"{model_name}-{algorithm}") as active:
        mlflow_run_id = active.info.run_id
        mlflow.log_params(
            {
                "algorithm": algorithm,
                "feature_version": dataset.feature_version,
                "horizon_days": settings.horizon_days,
                "embargo_days": dataset.embargo_days,
                "train_rows": len(dataset.train),
                "train_start": dataset.train.start,
                "train_end": dataset.train.end,
                "dataset_fingerprint": dataset.fingerprint[:16],
                **{f"model_{k}": v for k, v in default_params(algorithm).items()},
            }
        )

        pipeline = build_pipeline(algorithm, random_state=settings.random_seed)
        model = _calibrated(pipeline, dataset)

        validation_prob = model.predict_proba(dataset.validation.X)[:, 1]
        threshold = choose_threshold(dataset.validation.y.to_numpy(), validation_prob, economics)
        validation = evaluate_split(
            dataset.validation.frame,
            dataset.validation.y.to_numpy(),
            validation_prob,
            split="validation",
        )

        test_prob = model.predict_proba(dataset.test.X)[:, 1]
        test = evaluate_split(
            dataset.test.frame, dataset.test.y.to_numpy(), test_prob, split="test"
        )

        settings.artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = settings.artifact_dir / f"{model_name}-{run_id[:8]}.joblib"
        joblib.dump(
            {
                "model": model,
                "feature_version": dataset.feature_version,
                "threshold": threshold.threshold,
                "algorithm": algorithm,
                "trained_at": dt.datetime.now(dt.UTC).isoformat(),
                "dataset_fingerprint": dataset.fingerprint,
                # The training medians travel with the model because the
                # serving layer explains a score by asking "what would this
                # account's probability be if this one feature were typical?".
                # Recomputing them at serving time from live data would make
                # the explanation drift away from the model that produced it.
                "feature_baseline": _feature_baseline(dataset),
            },
            artifact_path,
        )

        baselines = baseline_values(dataset.validation.y.to_numpy(), economics)
        mlflow.log_metrics(
            {
                **{f"test_{k}": v for k, v in test.overall.as_dict().items() if not np.isnan(v)},
                **{
                    f"validation_{k}": v
                    for k, v in validation.overall.as_dict().items()
                    if not np.isnan(v)
                },
                **threshold.as_dict(),
                **baselines,
            }
        )
        for metrics in test.segments:
            for name, value in metrics.as_dict().items():
                if not np.isnan(value):
                    mlflow.log_metric(f"test_{metrics.segment}_{name}", value)

        calibration = calibration_table(dataset.test.y.to_numpy(), test_prob)
        mlflow.log_text(calibration.to_string(index=False), "calibration_test.txt")
        mlflow.log_artifact(str(artifact_path))

        result = TrainingResult(
            run_id=run_id,
            mlflow_run_id=mlflow_run_id,
            model_name=model_name,
            algorithm=algorithm,
            pipeline=model,
            dataset=dataset,
            threshold=threshold,
            validation=validation,
            test=test,
            artifact_path=artifact_path,
            leakage=findings,
            duration_seconds=time.perf_counter() - started,
        )

    _record_run(result, settings, status="SUCCESS")
    logger.info(
        "model trained",
        extra={
            "run_id": run_id,
            "algorithm": algorithm,
            **{k: round(v, 4) for k, v in result.headline.items()},
            "duration_seconds": round(result.duration_seconds, 1),
        },
    )
    return result


def training_runs(limit: int = 20, settings: Settings | None = None) -> list[dict[str, Any]]:
    settings = settings or get_settings()
    schema = settings.ml_schema
    try:
        with get_engine(settings).connect() as conn:
            rows = (
                conn.execute(
                    text(
                        f"""SELECT run_id, model_name, algorithm, feature_version, status,
                               started_at, train_rows, metrics
                        FROM {schema}.training_run
                        ORDER BY started_at DESC LIMIT :limit"""
                    ),
                    {"limit": limit},
                )
                .mappings()
                .all()
            )
    except SQLAlchemyError as exc:
        raise DatabaseError(f"reading run history failed: {exc}") from exc
    return [dict(row) for row in rows]
