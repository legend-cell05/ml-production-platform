"""The platform end to end, against a real PostgreSQL.

Every test here is a property the README claims. The expensive fixture -- a
simulated business, a feature store and two trained models -- is built once
for the module, because each test is a different question about the same
outcome rather than a different scenario.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from sqlalchemy import text

from polaris import pipeline
from polaris.config import Settings
from polaris.data import build_dataset
from polaris.db.engine import get_engine
from polaris.exceptions import PromotionBlocked
from polaris.features.builder import build_reference_date, reference_dates
from polaris.features.leakage import stability_check
from polaris.monitoring.drift import prediction_drift
from polaris.registry.promotion import evaluate_gate
from polaris.registry.store import get_production, get_version, list_versions
from polaris.serving.predictor import Predictor
from tests.conftest import requires_database

pytestmark = [requires_database, pytest.mark.integration]


@pytest.fixture(scope="module")
def platform(request: pytest.FixtureRequest) -> tuple[Settings, dict]:
    """Simulate, build features, train two models and promote the first."""
    settings: Settings = request.getfixturevalue("integration_settings")
    pipeline.prepare(settings)
    pipeline.build_features(settings)

    baseline, baseline_record = pipeline.train_and_register("logistic", settings)
    sample = baseline.dataset.test.X.head(50)
    pipeline.promote_candidate(baseline_record, settings, latency_sample=sample)

    challenger, challenger_record = pipeline.train_and_register("gradient_boosting", settings)
    return settings, {
        "baseline": baseline,
        "baseline_record": baseline_record,
        "challenger": challenger,
        "challenger_record": challenger_record,
        "sample": sample,
    }


class TestPointInTimeCorrectness:
    """The property the whole project rests on."""

    def test_rebuilding_a_past_date_after_new_data_changes_nothing(
        self, platform: tuple[Settings, dict]
    ) -> None:
        """If a past reference date moves when later data arrives, it read the future.

        This is the strongest available check: it does not ask whether a
        feature *looks* leaky, it asks whether the query is a statement about
        a moment that has already happened.
        """
        settings, _ = platform
        dates = reference_dates(settings)
        target = dates[len(dates) // 2]

        before = pd.read_sql(
            text(
                f"""SELECT * FROM {settings.feature_schema}.churn_features
                    WHERE reference_date = :d ORDER BY account_id"""
            ),
            get_engine(settings),
            params={"d": target},
        )

        # Add a day of usage far in the future for every account.
        future_day = dt.date(2026, 6, 1)
        with get_engine(settings).begin() as conn:
            conn.execute(
                text(
                    f"""INSERT INTO {settings.source_schema}.usage_daily
                            (account_id, usage_date, active_users, sessions, api_calls,
                             features_used, error_events)
                        SELECT account_id, :d, 999, 999, 9999, 14, 0
                        FROM {settings.source_schema}.account
                        ON CONFLICT DO NOTHING"""
                ),
                {"d": future_day},
            )
            conn.execute(
                text(
                    f"""DELETE FROM {settings.feature_schema}.churn_features
                        WHERE reference_date = :d"""
                ),
                {"d": target},
            )
        build_reference_date(target, settings)

        after = pd.read_sql(
            text(
                f"""SELECT * FROM {settings.feature_schema}.churn_features
                    WHERE reference_date = :d ORDER BY account_id"""
            ),
            get_engine(settings),
            params={"d": target},
        )

        findings = stability_check(before, after, reference_date=target)
        assert not findings, [f.detail for f in findings]

        with get_engine(settings).begin() as conn:
            conn.execute(
                text(f"DELETE FROM {settings.source_schema}.usage_daily WHERE usage_date = :d"),
                {"d": future_day},
            )

    def test_a_ticket_closed_after_the_date_counted_as_open(
        self, platform: tuple[Settings, dict]
    ) -> None:
        """The subtle one: `closed_at IS NULL` uses today's state, not the date's."""
        settings, _ = platform
        with get_engine(settings).connect() as conn:
            wrong = conn.execute(
                text(
                    f"""SELECT COUNT(*) FROM {settings.feature_schema}.churn_features f
                        JOIN LATERAL (
                            SELECT COUNT(*) AS open_then
                            FROM {settings.source_schema}.support_ticket t
                            WHERE t.account_id = f.account_id
                              AND t.opened_at < f.reference_date
                              AND (t.closed_at IS NULL OR t.closed_at >= f.reference_date)
                        ) expected ON TRUE
                        WHERE f.open_tickets <> expected.open_then"""
                )
            ).scalar_one()
        assert wrong == 0, f"{wrong} rows disagree with a point-in-time recount"


class TestDataset:
    def test_splits_are_ordered_in_time(self, platform: tuple[Settings, dict]) -> None:
        settings, _ = platform
        dataset = build_dataset(settings)
        assert dataset.train.end < dataset.validation.start
        assert dataset.validation.end < dataset.test.start

    def test_the_embargo_is_at_least_one_horizon(self, platform: tuple[Settings, dict]) -> None:
        """A training row's label describes the next sixty days, so the gap must exist."""
        settings, _ = platform
        dataset = build_dataset(settings)
        gap = (dataset.validation.start - dataset.train.end).days
        assert gap >= settings.horizon_days

    def test_no_account_and_date_appears_in_two_splits(
        self, platform: tuple[Settings, dict]
    ) -> None:
        settings, _ = platform
        dataset = build_dataset(settings)
        keys = [
            set(zip(s.frame.account_id, s.frame.reference_date, strict=True))
            for s in dataset.splits
        ]
        assert not (keys[0] & keys[1]) and not (keys[1] & keys[2]) and not (keys[0] & keys[2])

    def test_the_fingerprint_is_stable(self, platform: tuple[Settings, dict]) -> None:
        settings, _ = platform
        assert build_dataset(settings).fingerprint == build_dataset(settings).fingerprint


class TestTrainingAndRegistry:
    def test_both_models_beat_the_base_rate_by_a_wide_margin(
        self, platform: tuple[Settings, dict]
    ) -> None:
        _, objects = platform
        for key in ("baseline", "challenger"):
            result = objects[key]
            assert result.test.overall.pr_auc > 5 * result.dataset.test.base_rate

    def test_probabilities_are_calibrated(self, platform: tuple[Settings, dict]) -> None:
        settings, objects = platform
        assert objects["challenger"].test.overall.brier < settings.max_brier_score

    def test_exactly_one_version_is_in_production(self, platform: tuple[Settings, dict]) -> None:
        """Enforced by a partial unique index, not by a convention."""
        settings, _ = platform
        versions = list_versions("churn-60d", settings)
        assert sum(1 for v in versions if v.stage == "production") == 1

    def test_evaluations_are_stored_per_segment(self, platform: tuple[Settings, dict]) -> None:
        settings, objects = platform
        from polaris.registry.store import segment_metrics

        metrics = segment_metrics(
            "churn-60d", objects["challenger_record"].version, settings=settings
        )
        assert "overall" in metrics
        assert len(metrics) > 1


class TestPromotionGate:
    def test_a_stricter_segment_rule_can_block_a_better_model(
        self, platform: tuple[Settings, dict]
    ) -> None:
        """The check this project exists to demonstrate."""
        settings, objects = platform
        strict = settings.model_copy(update={"max_segment_regression": 0.0})
        decision = evaluate_gate(
            get_version("churn-60d", objects["challenger_record"].version, strict), strict
        )
        segment_check = next(c for c in decision.checks if c.name == "no_segment_regression")
        assert segment_check.detail  # always says what it judged
        if not segment_check.passed:
            assert "regressions" in segment_check.detail

    def test_a_model_that_does_not_improve_enough_is_refused(
        self, platform: tuple[Settings, dict]
    ) -> None:
        settings, objects = platform
        impossible = settings.model_copy(update={"min_pr_auc_gain": 0.99})
        with pytest.raises(PromotionBlocked, match=r"beats_champion|PR-AUC"):
            pipeline.promote_candidate(
                get_version("churn-60d", objects["challenger_record"].version, impossible),
                impossible,
                latency_sample=objects["sample"],
            )

    def test_force_overrides_the_gate_and_records_why(
        self, platform: tuple[Settings, dict]
    ) -> None:
        settings, objects = platform
        impossible = settings.model_copy(update={"min_pr_auc_gain": 0.99})
        record, decision = pipeline.promote_candidate(
            get_version("churn-60d", objects["challenger_record"].version, impossible),
            impossible,
            latency_sample=objects["sample"],
            force=True,
        )
        assert record.stage == "production"
        assert record.notes and record.notes.startswith("FORCED")
        assert not decision.allowed


class TestServing:
    def test_the_served_model_is_the_promoted_one(self, platform: tuple[Settings, dict]) -> None:
        settings, _ = platform
        production = get_production("churn-60d", settings)
        assert production is not None
        predictor = Predictor.from_production("churn-60d", settings)
        assert predictor.record.version == production.version
        # The registry column is NUMERIC(6,5), so the stored threshold is the
        # artefact's rounded to five places. The tolerance is that rounding,
        # not a tolerance for disagreement.
        assert predictor.threshold == pytest.approx(float(production.decision_threshold), abs=1e-5)

    def test_scoring_records_every_prediction(self, platform: tuple[Settings, dict]) -> None:
        settings, _ = platform
        with get_engine(settings).connect() as conn:
            before = conn.execute(
                text(f"SELECT COUNT(*) FROM {settings.ml_schema}.prediction")
            ).scalar_one()
        predictions = pipeline.score_batch(settings, limit=120)
        with get_engine(settings).connect() as conn:
            after = conn.execute(
                text(f"SELECT COUNT(*) FROM {settings.ml_schema}.prediction")
            ).scalar_one()
        assert after - before == len(predictions)

    def test_an_explanation_names_a_group_and_its_driver(
        self, platform: tuple[Settings, dict]
    ) -> None:
        settings, _ = platform
        predictions = pipeline.score_batch(settings, limit=25, explain=True)
        explained = [p for p in predictions if p.contributions]
        assert explained
        contribution = explained[0].contributions[0]
        assert contribution.group and contribution.label

    def test_missing_feature_columns_are_refused(self, platform: tuple[Settings, dict]) -> None:
        settings, _ = platform
        from polaris.exceptions import ServingError

        predictor = Predictor.from_production("churn-60d", settings)
        with pytest.raises(ServingError, match="missing feature"):
            predictor.predict(
                pd.DataFrame({"account_id": ["A"], "reference_date": [dt.date.today()]})
            )


class TestMonitoring:
    def test_drift_is_computed_and_recorded(self, platform: tuple[Settings, dict]) -> None:
        settings, _ = platform
        result = pipeline.monitor(settings)
        assert result["drift"]
        with get_engine(settings).connect() as conn:
            stored = conn.execute(
                text(f"SELECT COUNT(*) FROM {settings.ml_schema}.drift_check")
            ).scalar_one()
        assert stored >= len(result["drift"])

    def test_outcomes_arrive_only_for_matured_predictions(
        self, platform: tuple[Settings, dict]
    ) -> None:
        """The reporting lag is a property of the problem, not a bug."""
        settings, _ = platform
        dates = reference_dates(settings)
        pipeline.score_batch(settings, reference_date=dates[-6], limit=200)
        pipeline.score_batch(settings, reference_date=dates[-1], limit=200)
        result = pipeline.monitor(settings)
        performance = result["performance"]
        assert performance is not None
        assert performance.matched > 0
        assert performance.pending_labels > 0

    def test_prediction_drift_says_when_it_cannot_compare(
        self, platform: tuple[Settings, dict]
    ) -> None:
        """One batch is not zero drift.

        The first implementation compared wall-clock windows. Scoring several
        reference dates in one afternoon put every prediction in the same
        window, so the "earlier" mean was NULL, reported as 0.0, and the ratio
        came out as a confident zero -- a monitor that says the score
        distribution collapsed when it means it has nothing to compare.
        """
        settings, _ = platform
        production = get_production("churn-60d", settings)
        assert production is not None
        dates = reference_dates(settings)

        with get_engine(settings).begin() as conn:
            conn.execute(text(f"DELETE FROM {settings.ml_schema}.prediction"))

        pipeline.score_batch(settings, reference_date=dates[-1], limit=100)
        single = prediction_drift("churn-60d", production.version, settings)
        assert "no comparison available" in single["status"]
        assert "ratio" not in single

        pipeline.score_batch(settings, reference_date=dates[-4], limit=100)
        both = prediction_drift("churn-60d", production.version, settings)
        assert both["compared_with"] == str(dates[-4])
        assert both["ratio"] is not None and both["ratio"] > 0
