"""Scoring a model, overall and where it matters.

Four metrics, each answering a different question, and none of them
sufficient alone:

* **PR-AUC** -- the headline. With a 2.7% base rate, ROC-AUC flatters
  everything: a model can be 0.88 on ROC and useless in the top 100 accounts,
  which is the only part anybody will look at.
* **ROC-AUC** -- reported because everyone asks for it, and because it is
  comparable across datasets with different base rates.
* **Brier score** -- whether the probabilities are *probabilities*. A model
  used with an expected-value threshold needs calibrated output; a ranking is
  not enough.
* **Lift at k** -- what the customer success team actually experiences: of
  the hundred accounts we call this fortnight, how many were really leaving?

Everything is computed per segment as well as overall, because a model that
is better on average and worse for enterprise accounts is not an improvement
-- those are the accounts worth ten times the others, and the promotion gate
reads these rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


@dataclass(frozen=True)
class MetricSet:
    """Metrics for one slice of the data."""

    segment: str
    n_rows: int
    n_positives: int
    pr_auc: float
    roc_auc: float
    brier: float
    log_loss: float
    lift_at_100: float
    recall_at_100: float

    def as_dict(self) -> dict[str, float]:
        return {
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "lift_at_100": self.lift_at_100,
            "recall_at_100": self.recall_at_100,
        }


@dataclass
class Evaluation:
    """Overall and per-segment metrics for one split."""

    split: str
    overall: MetricSet
    segments: list[MetricSet] = field(default_factory=list)

    def rows(self) -> list[tuple[str, str, float, int, int]]:
        """Flattened for storage: (segment, metric, value, n_rows, n_positives)."""
        out: list[tuple[str, str, float, int, int]] = []
        for metrics in [self.overall, *self.segments]:
            for name, value in metrics.as_dict().items():
                out.append(
                    (metrics.segment, name, float(value), metrics.n_rows, metrics.n_positives)
                )
        return out


def _lift_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> tuple[float, float]:
    """Precision in the top k, as a multiple of the base rate, and its recall."""
    if len(y_true) == 0 or y_true.sum() == 0:
        return 0.0, 0.0
    k = min(k, len(y_true))
    order = np.argsort(-y_prob)[:k]
    captured = int(y_true[order].sum())
    precision_at_k = captured / k
    base_rate = float(y_true.mean())
    lift = precision_at_k / base_rate if base_rate > 0 else 0.0
    return float(lift), float(captured / y_true.sum())


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, *, segment: str = "overall", k: int = 100
) -> MetricSet:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    positives = int(y_true.sum())

    if positives == 0 or positives == len(y_true):
        # A slice with one class has no AUC. Reporting 0.5 would be a lie
        # dressed as a number; the row is kept with NaN so the gate can see
        # that it could not be evaluated rather than that it scored badly.
        nan = float("nan")
        return MetricSet(segment, len(y_true), positives, nan, nan, nan, nan, nan, nan)

    lift, recall = _lift_at_k(y_true, y_prob, k)
    return MetricSet(
        segment=segment,
        n_rows=len(y_true),
        n_positives=positives,
        pr_auc=float(average_precision_score(y_true, y_prob)),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
        log_loss=float(log_loss(y_true, y_prob, labels=[0, 1])),
        lift_at_100=lift,
        recall_at_100=recall,
    )


def evaluate_split(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    split: str,
    segment_column: str = "segment",
) -> Evaluation:
    """Metrics overall and for every value of ``segment_column``."""
    overall = compute_metrics(y_true, y_prob, segment="overall")
    segments: list[MetricSet] = []
    for value in sorted(frame[segment_column].dropna().unique()):
        mask = (frame[segment_column] == value).to_numpy()
        if mask.sum() < 50:
            continue  # too small to say anything honest about
        segments.append(
            compute_metrics(np.asarray(y_true)[mask], np.asarray(y_prob)[mask], segment=str(value))
        )
    return Evaluation(split=split, overall=overall, segments=segments)


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, *, bins: int = 10) -> pd.DataFrame:
    """Predicted probability against observed frequency, by decile of score.

    The table a reader should look at before believing an expected-value
    threshold: if the bucket predicted at 20% churns at 5%, the threshold is
    being applied to a number that does not mean what it says.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    edges = np.quantile(y_prob, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return pd.DataFrame(columns=["bucket", "n", "mean_predicted", "observed", "gap"])

    index = np.clip(np.digitize(y_prob, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for bucket in range(len(edges) - 1):
        mask = index == bucket
        if mask.sum() == 0:
            continue
        predicted = float(y_prob[mask].mean())
        observed = float(y_true[mask].mean())
        rows.append(
            {
                "bucket": bucket + 1,
                "n": int(mask.sum()),
                "mean_predicted": round(predicted, 5),
                "observed": round(observed, 5),
                "gap": round(observed - predicted, 5),
            }
        )
    return pd.DataFrame(rows)
