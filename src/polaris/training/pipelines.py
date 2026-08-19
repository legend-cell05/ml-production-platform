"""The model pipelines.

Two families, deliberately:

* **logistic regression**, as a baseline that can be read. Its coefficients
  are the thing to show a customer success manager who asks why an account
  was flagged, and it is the number every other model has to beat before
  anyone should accept the extra complexity;
* **gradient boosting**, because the relationships here are not linear -- a
  usage ratio of 0.4 means something very different at 5 seats and at 500.

Everything about the preprocessing is inside the pipeline, never applied to a
frame beforehand. A scaler fitted outside the pipeline is fitted on the test
set too, which is the most common way a good validation number turns out to
be fiction.

**Missingness is a feature.** ``avg_satisfaction_180d`` is NULL when the
customer never answered a survey, and customers who never answer are not a
random sample. Imputing the median throws that away, so the numeric branch
keeps an explicit indicator column alongside the imputed value.
``SimpleImputer(add_indicator=True)`` produces one for every numeric column
that had a missing value at fit time; ``MEANINGFUL_MISSING`` names the columns
where that is known in advance to carry meaning, and a test asserts each of
them gets its indicator rather than trusting that it happened.
"""

from __future__ import annotations

from typing import Literal

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from polaris.features.definitions import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
)

Algorithm = Literal["logistic", "gradient_boosting"]

ALGORITHMS: tuple[Algorithm, ...] = ("logistic", "gradient_boosting")

# Columns whose absence is information rather than an accident.
MEANINGFUL_MISSING: tuple[str, ...] = tuple(
    f.name for f in FEATURES if f.missing_means is not None and f.kind == "numeric"
)


def _numeric_branch() -> Pipeline:
    return Pipeline(
        [
            # The indicator is added per column that was ever missing during
            # fit, so a column that is always present costs nothing.
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", _numeric_branch(), list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=25, sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
            ("boolean", SimpleImputer(strategy="most_frequent"), list(BOOLEAN_FEATURES)),
        ],
        remainder="drop",  # anything not named is not a feature, by construction
        verbose_feature_names_out=True,
    )


def build_pipeline(algorithm: Algorithm, *, random_state: int = 0) -> Pipeline:
    """A fitted-from-scratch pipeline for one algorithm."""
    if algorithm == "logistic":
        estimator = LogisticRegression(
            max_iter=2000,
            # The positive class is 2.7% of the data. Without the reweighting
            # the model is technically excellent and practically useless: it
            # predicts "will not churn" for everyone and is right 97% of the
            # time.
            class_weight="balanced",
            C=0.5,
            random_state=random_state,
        )
        return Pipeline([("prep", _preprocessor()), ("model", estimator)])

    if algorithm == "gradient_boosting":
        # HistGradientBoosting handles NaN natively, so the imputer is not
        # needed for its sake -- but the pipeline stays identical across
        # algorithms so that a comparison between them is a comparison of the
        # algorithms rather than of two different preprocessings.
        estimator = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            class_weight="balanced",
            random_state=random_state,
        )
        return Pipeline([("prep", _preprocessor()), ("model", estimator)])

    raise ValueError(f"unknown algorithm {algorithm!r}; expected one of {ALGORITHMS}")


def default_params(algorithm: Algorithm) -> dict[str, object]:
    """The parameters worth recording with a run."""
    pipeline = build_pipeline(algorithm)
    model = pipeline.named_steps["model"]
    return {
        key: value
        for key, value in model.get_params().items()
        if isinstance(value, (int, float, str, bool)) or value is None
    }
