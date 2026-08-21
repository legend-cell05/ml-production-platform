"""Training: pipelines, evaluation, and the run that ties them together."""

from polaris.training.evaluate import Evaluation, MetricSet, calibration_table, evaluate_split
from polaris.training.pipelines import ALGORITHMS, Algorithm, build_pipeline
from polaris.training.train import TrainingResult, train, training_runs

__all__ = [
    "ALGORITHMS",
    "Algorithm",
    "Evaluation",
    "MetricSet",
    "TrainingResult",
    "build_pipeline",
    "calibration_table",
    "evaluate_split",
    "train",
    "training_runs",
]
