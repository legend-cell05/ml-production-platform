"""Metrics and drift.

Both are small pieces of arithmetic that everybody assumes are right, and
both have a standard way of being wrong: a metric that silently returns 0.5
on a degenerate slice, and a PSI that re-bins each period and therefore never
detects anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from polaris.config import Settings
from polaris.monitoring.drift import check_drift, population_stability_index
from polaris.training.evaluate import calibration_table, compute_metrics, evaluate_split

SETTINGS = Settings(psi_warn=0.10, psi_alert=0.25)


class TestMetrics:
    def test_a_perfect_ranking_scores_one(self) -> None:
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = compute_metrics(y, p)
        assert metrics.roc_auc == pytest.approx(1.0)
        assert metrics.pr_auc == pytest.approx(1.0)

    def test_a_single_class_slice_returns_nan_not_a_guess(self) -> None:
        """Reporting 0.5 would be a lie dressed as a number."""
        metrics = compute_metrics(np.zeros(100), np.random.default_rng(0).random(100))
        assert np.isnan(metrics.roc_auc)
        assert metrics.n_positives == 0

    def test_lift_is_relative_to_the_base_rate(self) -> None:
        y = np.zeros(1000, dtype=int)
        y[:10] = 1  # 1% base rate
        p = np.zeros(1000)
        p[:10] = 0.9  # the model puts all ten at the top
        metrics = compute_metrics(y, p, k=100)
        assert metrics.lift_at_100 == pytest.approx(10.0)
        assert metrics.recall_at_100 == pytest.approx(1.0)

    def test_segments_smaller_than_fifty_rows_are_skipped(self) -> None:
        frame = pd.DataFrame({"segment": ["a"] * 200 + ["tiny"] * 10})
        y = np.array([0, 1] * 105)
        p = np.random.default_rng(1).random(210)
        evaluation = evaluate_split(frame, y, p, split="test")
        assert {m.segment for m in evaluation.segments} == {"a"}


class TestCalibration:
    def test_a_calibrated_model_has_small_gaps(self) -> None:
        rng = np.random.default_rng(7)
        p = rng.random(5000) * 0.5
        y = (rng.random(5000) < p).astype(int)
        table = calibration_table(y, p, bins=10)
        assert table.gap.abs().max() < 0.06

    def test_an_overconfident_model_shows_a_positive_gap(self) -> None:
        rng = np.random.default_rng(8)
        p = np.clip(rng.random(5000) * 0.5, 0.01, 1)
        y = (rng.random(5000) < p / 3).astype(int)  # reality is a third of the claim
        table = calibration_table(y, p, bins=10)
        assert table.gap.min() < -0.05


class TestPSI:
    def test_identical_distributions_score_zero(self) -> None:
        values = pd.Series(np.random.default_rng(2).normal(size=3000))
        psi, _ = population_stability_index(values, values.copy())
        assert psi == pytest.approx(0.0, abs=1e-9)

    def test_a_shifted_distribution_is_detected(self) -> None:
        rng = np.random.default_rng(3)
        baseline = pd.Series(rng.normal(0, 1, 4000))
        shifted = pd.Series(rng.normal(1.5, 1, 4000))
        psi, _ = population_stability_index(baseline, shifted)
        assert psi > 0.25

    def test_bins_come_from_the_baseline_only(self) -> None:
        """Re-binning each period would compare a distribution to itself.

        The classic broken PSI returns ~0 here, because quantile bins of the
        shifted data reproduce the same proportions.
        """
        rng = np.random.default_rng(4)
        baseline = pd.Series(rng.normal(0, 1, 4000))
        shifted = pd.Series(rng.normal(0, 1, 4000) * 4 + 6)
        psi, _ = population_stability_index(baseline, shifted)
        assert psi > 1.0

    def test_handles_categoricals(self) -> None:
        baseline = pd.Series(["a"] * 900 + ["b"] * 100)
        current = pd.Series(["a"] * 500 + ["b"] * 500)
        psi, detail = population_stability_index(baseline, current)
        assert psi > 0.25
        assert "categories" in detail

    def test_a_category_appearing_from_nowhere_is_finite(self) -> None:
        baseline = pd.Series(["a"] * 1000)
        current = pd.Series(["a"] * 500 + ["new"] * 500)
        psi, _ = population_stability_index(baseline, current)
        assert np.isfinite(psi) and psi > 0

    def test_too_few_rows_is_reported_rather_than_guessed(self) -> None:
        psi, detail = population_stability_index(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))
        assert np.isnan(psi) and "too few" in detail


class TestDriftStatuses:
    def test_thresholds_map_to_statuses(self) -> None:
        rng = np.random.default_rng(9)
        baseline = pd.DataFrame({"tenure_days": rng.normal(500, 100, 3000)})
        current = pd.DataFrame({"tenure_days": rng.normal(900, 100, 3000)})
        results = check_drift(baseline, current, SETTINGS)
        assert results and results[0].status == "ALERT"
