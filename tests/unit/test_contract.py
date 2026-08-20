"""The feature contract's own coherence.

Cheap tests that catch the class of mistake which is otherwise found in
production: a feature added to the definitions and forgotten in a group, or a
column the model was never supposed to see quietly ending up in the matrix.
"""

from __future__ import annotations

import pytest

from polaris.features.definitions import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    FEATURE_GROUPS,
    FEATURE_NAMES,
    FEATURE_VERSION,
    FEATURES,
    LABEL_COLUMN,
    NUMERIC_FEATURES,
    feature_by_name,
    group_of,
)
from polaris.training.pipelines import ALGORITHMS, MEANINGFUL_MISSING, build_pipeline


class TestContract:
    def test_every_feature_belongs_to_exactly_one_group(self) -> None:
        for name in FEATURE_NAMES:
            assert group_of(name)
        members = [f for group in FEATURE_GROUPS.values() for f in group]
        assert len(members) == len(set(members)), "a feature is in two groups"
        assert set(members) == set(FEATURE_NAMES)

    def test_no_feature_is_named_twice(self) -> None:
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

    def test_kinds_partition_the_features(self) -> None:
        assert set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) | set(BOOLEAN_FEATURES) == set(
            FEATURE_NAMES
        )

    def test_the_label_is_not_a_feature(self) -> None:
        assert LABEL_COLUMN not in FEATURE_NAMES
        assert LABEL_COLUMN in EXCLUDED_COLUMNS

    def test_identifiers_are_excluded(self) -> None:
        """`reference_date` as a feature teaches the model the calendar."""
        for column in ("reference_date", "account_id", "feature_version"):
            assert column in EXCLUDED_COLUMNS
            assert column not in FEATURE_NAMES

    def test_every_feature_has_a_description(self) -> None:
        for feature in FEATURES:
            assert feature.description.strip()

    def test_unknown_names_raise(self) -> None:
        with pytest.raises(KeyError):
            feature_by_name("not_a_feature")
        with pytest.raises(KeyError):
            group_of("not_a_feature")

    def test_the_version_looks_like_a_version(self) -> None:
        parts = FEATURE_VERSION.split(".")
        assert len(parts) == 3 and all(p.isdigit() for p in parts)


class TestPipelines:
    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_every_algorithm_builds(self, algorithm: str) -> None:
        pipeline = build_pipeline(algorithm)  # type: ignore[arg-type]
        assert "prep" in pipeline.named_steps and "model" in pipeline.named_steps

    def test_an_unknown_algorithm_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown algorithm"):
            build_pipeline("random_forest")  # type: ignore[arg-type]

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_preprocessing_drops_anything_not_declared(self, algorithm: str) -> None:
        """`remainder="drop"` is what stops a stray column becoming a feature."""
        pipeline = build_pipeline(algorithm)  # type: ignore[arg-type]
        assert pipeline.named_steps["prep"].remainder == "drop"

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_the_class_imbalance_is_handled(self, algorithm: str) -> None:
        """Without it the model predicts 'stays' for everyone and is right 97% of the time."""
        model = build_pipeline(algorithm).named_steps["model"]  # type: ignore[arg-type]
        assert model.get_params().get("class_weight") == "balanced"

    def test_a_meaningful_gap_becomes_its_own_column(self) -> None:
        """Imputing the median over a meaningful NULL throws the meaning away.

        Fitted on a frame where every declared meaningful-missing column has a
        gap, the numeric branch must produce an indicator for each of them --
        so the model can learn "never answered a survey" rather than only
        "answered, with the median score".
        """
        import numpy as np
        import pandas as pd

        rows = 40
        frame = pd.DataFrame(
            {name: np.arange(rows, dtype=float) for name in NUMERIC_FEATURES}
            | {name: ["smb"] * rows for name in CATEGORICAL_FEATURES}
            # Booleans reach the pipeline as floats -- `dataset.py` casts them,
            # because SimpleImputer refuses a bool dtype.
            | {name: [1.0] * rows for name in BOOLEAN_FEATURES}
        )
        for name in MEANINGFUL_MISSING:
            frame.loc[: rows // 2, name] = np.nan

        prep = build_pipeline("logistic").named_steps["prep"]
        prep.fit(frame)
        imputer = prep.named_transformers_["numeric"].named_steps["impute"]
        indicated = {NUMERIC_FEATURES[i] for i in imputer.indicator_.features_}
        missing = sorted(set(MEANINGFUL_MISSING) - indicated)
        assert not missing, f"no missingness indicator for {missing}"
