"""The leakage screens.

The first test in this file is a regression test for a bug that was in this
repository: the simulator set a renewal date only for accounts that had not
churned, so the *absence* of that feature was the label. The model reached
0.995 ROC-AUC and looked wonderful.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from polaris.features.leakage import (
    screen_frame,
    screen_missingness,
    screen_values,
    stability_check,
)


class TestMissingnessScreen:
    def test_catches_the_renewal_date_bug(self, labelled_frame: pd.DataFrame) -> None:
        """A feature present only for the negative class must be flagged as a leak."""
        bugged = labelled_frame.copy()
        bugged["days_to_renewal"] = np.where(bugged.churned_in_horizon, np.nan, 180.0)

        findings = screen_missingness(bugged)
        leaks = [f for f in findings if f.feature == "days_to_renewal" and f.blocking]
        assert leaks, "the screen missed a feature whose absence is the label"
        assert "label rate" in leaks[0].detail

    def test_ignores_missingness_that_means_nothing(self, labelled_frame: pd.DataFrame) -> None:
        findings = screen_missingness(labelled_frame)
        assert not [f for f in findings if f.feature == "last_nps_score" and f.blocking]

    def test_ignores_a_column_that_is_never_missing(self, labelled_frame: pd.DataFrame) -> None:
        assert not [f for f in screen_missingness(labelled_frame) if f.feature == "tickets_90d"]


class TestValueScreen:
    def test_catches_a_feature_that_is_the_label(self, labelled_frame: pd.DataFrame) -> None:
        leaked = labelled_frame.copy()
        leaked["mrr_eur"] = leaked.churned_in_horizon.astype(float) * 1000 + 1
        findings = [f for f in screen_values(leaked) if f.feature == "mrr_eur"]
        assert findings and findings[0].blocking

    def test_catches_it_the_other_way_round_too(self, labelled_frame: pd.DataFrame) -> None:
        """A feature that perfectly predicts *not* churning is just as leaky."""
        leaked = labelled_frame.copy()
        leaked["mrr_eur"] = (~leaked.churned_in_horizon).astype(float) * 1000 + 1
        findings = [f for f in screen_values(leaked) if f.feature == "mrr_eur"]
        assert findings and findings[0].blocking

    def test_leaves_a_genuinely_strong_feature_alone(self, labelled_frame: pd.DataFrame) -> None:
        """Usage decline is a real signal and must not be treated as a leak.

        This is why the threshold is 0.95 and not 0.80: the feature the model
        is supposed to rely on reaches 0.85 on its own in the real data.
        """
        findings = [f for f in screen_values(labelled_frame) if f.blocking]
        assert not findings


class TestStabilityScreen:
    def test_identical_rebuilds_are_clean(self) -> None:
        frame = pd.DataFrame({"account_id": ["A", "B"], "tenure_days": [10, 20], "seats": [5, 6]})
        assert stability_check(frame, frame.copy(), reference_date=dt.date(2026, 1, 1)) == []

    def test_a_changed_feature_is_a_leak(self) -> None:
        """A past reference date that changes when new data arrives read the future."""
        before = pd.DataFrame({"account_id": ["A", "B"], "tenure_days": [10, 20]})
        after = pd.DataFrame({"account_id": ["A", "B"], "tenure_days": [10, 25]})
        findings = stability_check(before, after, reference_date=dt.date(2026, 1, 1))
        assert findings and findings[0].feature == "tenure_days"
        assert findings[0].blocking


class TestScreenFrame:
    def test_returns_nothing_without_a_label(self) -> None:
        assert screen_frame(pd.DataFrame({"seats": [1, 2, 3]})) == []

    def test_orders_findings_worst_first(self, labelled_frame: pd.DataFrame) -> None:
        leaked = labelled_frame.copy()
        leaked["mrr_eur"] = leaked.churned_in_horizon.astype(float)
        findings = screen_frame(leaked)
        assert findings
        assert findings == sorted(findings, key=lambda f: f.score, reverse=True)
