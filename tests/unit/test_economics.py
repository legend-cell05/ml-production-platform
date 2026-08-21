"""The decision layer.

These tests are short because the arithmetic is short. They are here because
the arithmetic decides how many people get phoned, and an off-by-one in the
sign would be invisible in every accuracy metric.
"""

from __future__ import annotations

import numpy as np
import pytest

from polaris.economics import Economics, baseline_values, choose_threshold, expected_value

ECONOMICS = Economics(
    value_of_saved_account=9000.0, cost_of_intervention=350.0, intervention_success_rate=0.30
)


class TestArithmetic:
    def test_a_true_positive_is_worth_the_save_minus_the_call(self) -> None:
        assert ECONOMICS.value_of_true_positive == pytest.approx(9000 * 0.30 - 350)

    def test_break_even_is_where_the_two_cancel(self) -> None:
        assert ECONOMICS.break_even_probability == pytest.approx(350 / (2350 + 350))

    def test_a_more_expensive_play_raises_the_bar(self) -> None:
        cheap = Economics(9000, 100, 0.3).break_even_probability
        dear = Economics(9000, 900, 0.3).break_even_probability
        assert dear > cheap

    def test_a_worthless_intervention_makes_acting_never_pay(self) -> None:
        """If the play never works, no probability justifies making the call."""
        useless = Economics(9000, 350, 0.01)
        assert useless.value_of_true_positive < 0
        assert useless.break_even_probability > 1


class TestExpectedValue:
    def test_counts_hits_and_misses_the_right_way_round(self) -> None:
        y = np.array([1, 1, 0, 0])
        p = np.array([0.9, 0.8, 0.7, 0.1])
        # At 0.5: two true positives, one false positive.
        assert expected_value(y, p, 0.5, ECONOMICS) == pytest.approx(2 * 2350 - 350)

    def test_a_threshold_above_everything_is_worth_nothing(self) -> None:
        y = np.array([1, 0, 1])
        p = np.array([0.4, 0.2, 0.3])
        assert expected_value(y, p, 0.99, ECONOMICS) == 0.0


class TestThresholdChoice:
    def test_picks_a_threshold_above_break_even(self) -> None:
        rng = np.random.default_rng(3)
        y = (rng.random(4000) < 0.04).astype(int)
        p = np.clip(rng.beta(1.3, 22, 4000) + y * rng.beta(4, 5, 4000), 0, 1)
        choice = choose_threshold(y, p, ECONOMICS)
        assert 0 < choice.threshold < 1
        assert choice.expected_value > 0
        assert choice.precision > ECONOMICS.break_even_probability

    def test_a_useless_model_is_told_to_call_nobody(self) -> None:
        """With no signal, the best threshold flags nobody rather than everybody."""
        rng = np.random.default_rng(5)
        y = (rng.random(3000) < 0.03).astype(int)
        p = rng.random(3000) * 0.2  # independent of y
        choice = choose_threshold(y, p, ECONOMICS)
        assert choice.expected_value >= 0

    def test_refuses_an_empty_set(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            choose_threshold(np.array([]), np.array([]), ECONOMICS)

    def test_refuses_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            choose_threshold(np.array([1, 0]), np.array([0.5]), ECONOMICS)


class TestBaselines:
    def test_calling_everyone_loses_money_at_a_low_base_rate(self) -> None:
        y = (np.arange(1000) < 30).astype(int)  # 3%
        baselines = baseline_values(y, ECONOMICS)
        assert baselines["call_everyone_eur"] < 0
        assert baselines["call_nobody_eur"] == 0.0

    def test_calling_everyone_wins_when_almost_everyone_churns(self) -> None:
        y = (np.arange(1000) < 900).astype(int)
        assert baseline_values(y, ECONOMICS)["call_everyone_eur"] > 0
