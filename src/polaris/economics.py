"""Turning a probability into a decision.

A churn model does not produce value by being accurate. It produces value by
causing someone to pick up a phone, and picking up the phone costs money
whether or not the account was going to leave. That makes the threshold an
economic question, not a statistical one, and it is the reason this module
exists instead of a call to ``f1_score``.

The arithmetic, per account flagged:

* the account really was going to churn (**true positive**): the retention
  play works a fraction ``p`` of the time, so the expected gain is
  ``value x p - cost``;
* it was not (**false positive**): the play costs ``cost`` and saves nothing;
* it was going to churn and was not flagged (**false negative**): nothing is
  spent and nothing is saved -- the loss is the customer, and it is the same
  loss as doing nothing at all, which is why it does not appear as a negative
  term here.

With the defaults -- 9 000 EUR of margin, a 350 EUR play that works three
times in ten -- a true positive is worth 2 350 EUR and a false positive costs
350. Acting is worth it as long as roughly one account in eight is right,
which puts the optimal threshold near 0.13 rather than 0.5. Choosing 0.5,
the sklearn default, would leave most of the value on the table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from polaris.config import Settings, get_settings


@dataclass(frozen=True)
class Economics:
    """The three numbers the business owns."""

    value_of_saved_account: float
    cost_of_intervention: float
    intervention_success_rate: float

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Economics:
        settings = settings or get_settings()
        return cls(
            value_of_saved_account=settings.value_of_saved_account,
            cost_of_intervention=settings.cost_of_intervention,
            intervention_success_rate=settings.intervention_success_rate,
        )

    @property
    def value_of_true_positive(self) -> float:
        return (
            self.value_of_saved_account * self.intervention_success_rate - self.cost_of_intervention
        )

    @property
    def cost_of_false_positive(self) -> float:
        return self.cost_of_intervention

    @property
    def break_even_probability(self) -> float:
        """The probability at which acting stops being worth it.

        ``cost / (value + cost)``. Below this, the expected gain from the
        accounts that really would have churned no longer covers the calls
        wasted on the ones that would not. It is a lower bound on any sensible
        threshold, and a useful sanity check on the one the optimiser picks.
        """
        total = self.value_of_true_positive + self.cost_of_false_positive
        return self.cost_of_false_positive / total if total > 0 else 1.0


@dataclass(frozen=True)
class ThresholdChoice:
    """The chosen threshold and what it is expected to be worth."""

    threshold: float
    expected_value: float
    expected_value_per_1000: float
    flagged: int
    true_positives: int
    false_positives: int
    precision: float
    recall: float
    n_rows: int

    def as_dict(self) -> dict[str, float]:
        return {
            "threshold": round(self.threshold, 5),
            "expected_value_eur": round(self.expected_value, 2),
            "expected_value_per_1000_accounts": round(self.expected_value_per_1000, 2),
            "flagged": float(self.flagged),
            "true_positives": float(self.true_positives),
            "false_positives": float(self.false_positives),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
        }


def expected_value(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, economics: Economics
) -> float:
    """Expected euros from acting on everything at or above ``threshold``."""
    flagged = y_prob >= threshold
    true_positives = int(np.sum(flagged & (y_true == 1)))
    false_positives = int(np.sum(flagged & (y_true == 0)))
    return (
        true_positives * economics.value_of_true_positive
        - false_positives * economics.cost_of_false_positive
    )


def choose_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    economics: Economics,
    *,
    grid: int = 500,
) -> ThresholdChoice:
    """Pick the threshold that maximises expected value on this data.

    Swept over a grid rather than solved analytically because the optimum
    depends on the empirical distribution of scores, not only on the costs --
    a model that never predicts above 0.3 has its optimum inside that range
    whatever the arithmetic says.

    Chosen on **validation** data. Choosing it on the test set would make the
    reported value an upper bound that nobody will ever see again.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    if len(y_true) == 0:
        raise ValueError("cannot choose a threshold on an empty set")

    candidates = np.unique(np.concatenate([np.linspace(0.001, 0.999, grid), y_prob]))
    values = np.array([expected_value(y_true, y_prob, t, economics) for t in candidates])
    best_index = int(np.argmax(values))
    best = float(candidates[best_index])

    flagged = y_prob >= best
    true_positives = int(np.sum(flagged & (y_true == 1)))
    false_positives = int(np.sum(flagged & (y_true == 0)))
    positives = int(np.sum(y_true == 1))

    return ThresholdChoice(
        threshold=best,
        expected_value=float(values[best_index]),
        expected_value_per_1000=float(values[best_index]) / len(y_true) * 1000.0,
        flagged=int(flagged.sum()),
        true_positives=true_positives,
        false_positives=false_positives,
        precision=true_positives / flagged.sum() if flagged.sum() else 0.0,
        recall=true_positives / positives if positives else 0.0,
        n_rows=len(y_true),
    )


def baseline_values(y_true: np.ndarray, economics: Economics) -> dict[str, float]:
    """What the two strategies that need no model are worth.

    Reported next to the model so that "the model is worth X" is always
    accompanied by "and doing the obvious thing is worth Y". A model that does
    not beat calling everybody is not a model worth deploying, and a model
    that does not beat calling nobody is worse than that.
    """
    y_true = np.asarray(y_true).astype(int)
    n = len(y_true)
    positives = int(y_true.sum())
    call_everyone = (
        positives * economics.value_of_true_positive
        - (n - positives) * economics.cost_of_false_positive
    )
    return {
        "call_nobody_eur": 0.0,
        "call_everyone_eur": float(call_everyone),
        "call_everyone_per_1000": float(call_everyone) / n * 1000.0 if n else 0.0,
    }
