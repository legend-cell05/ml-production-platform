"""Turning the feature store into a dataset.

Two decisions live here, and both are the kind that quietly ruin a churn
model.

**The split is temporal, never random.** A random split puts January and March
rows for the same account on both sides, and the model is scored on its
ability to remember accounts rather than to predict them. Every reported
number would be optimistic, and the optimism would only be discovered in
production.

**There is an embargo between the splits.** A reference date has a label that
describes the following sixty days. A training row dated 1 June therefore
knows about July -- so if the validation period starts on 15 June, the
training set has already seen part of it. The gap between splits is one
horizon wide, which costs one horizon of data and buys a number that means
something.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from polaris.config import Settings, get_settings
from polaris.db.engine import get_engine
from polaris.exceptions import DatabaseError, DataError
from polaris.features.definitions import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    LABEL_COLUMN,
    NUMERIC_FEATURES,
)
from polaris.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Split:
    """One temporal slice of the dataset."""

    name: str
    frame: pd.DataFrame
    start: dt.date
    end: dt.date

    @property
    def X(self) -> pd.DataFrame:  # noqa: N802 - X and y are the names this domain uses
        return self.frame[list(FEATURE_NAMES)]

    @property
    def y(self) -> pd.Series:
        return self.frame[LABEL_COLUMN].astype(int)

    @property
    def base_rate(self) -> float:
        return float(self.y.mean()) if len(self.frame) else 0.0

    def __len__(self) -> int:
        return len(self.frame)


@dataclass
class Dataset:
    """Train, validation and test, with the provenance to reproduce them."""

    train: Split
    validation: Split
    test: Split
    fingerprint: str
    feature_version: str
    horizon_days: int
    embargo_days: int
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def splits(self) -> tuple[Split, Split, Split]:
        return (self.train, self.validation, self.test)


def load_labelled(settings: Settings | None = None) -> pd.DataFrame:
    """Every feature row whose label is known."""
    settings = settings or get_settings()
    schema = settings.feature_schema
    try:
        frame = pd.read_sql(
            text(
                f"""SELECT * FROM {schema}.churn_features
                    WHERE {LABEL_COLUMN} IS NOT NULL
                    ORDER BY reference_date, account_id"""
            ),
            get_engine(settings),
        )
    except SQLAlchemyError as exc:
        raise DatabaseError(f"reading the feature store failed: {exc}") from exc
    if frame.empty:
        raise DataError("no labelled rows -- run `polaris features build` first")
    return _coerce(frame)


def load_scoreable(reference_date: dt.date, settings: Settings | None = None) -> pd.DataFrame:
    """The rows to score for one date, labelled or not."""
    settings = settings or get_settings()
    schema = settings.feature_schema
    try:
        frame = pd.read_sql(
            text(
                f"""SELECT * FROM {schema}.churn_features
                    WHERE reference_date = :reference_date
                    ORDER BY account_id"""
            ),
            get_engine(settings),
            params={"reference_date": reference_date},
        )
    except SQLAlchemyError as exc:
        raise DatabaseError(f"reading the feature store failed: {exc}") from exc
    return _coerce(frame)


def _coerce(frame: pd.DataFrame) -> pd.DataFrame:
    """Types the model expects, applied in one place.

    Booleans become floats with NaN preserved rather than 0/1 with NaN
    silently becoming False, because "we do not know whether a QBR happened"
    and "no QBR happened" are different statements.
    """
    frame = frame.copy()
    frame["reference_date"] = pd.to_datetime(frame["reference_date"]).dt.date
    for column in NUMERIC_FEATURES:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in BOOLEAN_FEATURES:
        if column in frame.columns:
            frame[column] = frame[column].astype("boolean").astype("Float64").astype(float)
    for column in CATEGORICAL_FEATURES:
        if column in frame.columns:
            frame[column] = frame[column].astype(str)
    return frame


def fingerprint(frame: pd.DataFrame) -> str:
    """A stable hash of the rows a model was trained on.

    Recorded with every run so that "the same model on the same data" is a
    claim somebody can check rather than remember. Built from the identifiers
    and the label, not from the feature values, so it survives a float being
    rendered with one more decimal.
    """
    key = (
        frame[["reference_date", "account_id", LABEL_COLUMN]]
        .astype(str)
        .agg("|".join, axis=1)
        .sort_values()
    )
    digest = hashlib.sha256()
    for value in key:
        digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def build_dataset(
    settings: Settings | None = None,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.20,
) -> Dataset:
    """Split the labelled feature store by time, with an embargo between parts."""
    settings = settings or get_settings()
    frame = load_labelled(settings)

    dates = sorted(frame["reference_date"].unique())
    if len(dates) < 5:
        raise DataError(
            f"only {len(dates)} reference date(s) available; a temporal split needs at least 5"
        )

    embargo = dt.timedelta(days=settings.horizon_days)
    n = len(dates)
    test_start_index = max(1, int(n * (1 - test_fraction)))
    validation_start_index = max(1, int(n * (1 - test_fraction - validation_fraction)))

    validation_start = dates[validation_start_index]
    test_start = dates[test_start_index]

    # The embargo removes training rows whose label window reaches into
    # validation, and validation rows whose window reaches into test.
    train_frame = frame[frame["reference_date"] < validation_start - embargo]
    validation_frame = frame[
        (frame["reference_date"] >= validation_start)
        & (frame["reference_date"] < test_start - embargo)
    ]
    test_frame = frame[frame["reference_date"] >= test_start]

    for name, part in (
        ("train", train_frame),
        ("validation", validation_frame),
        ("test", test_frame),
    ):
        if part.empty:
            raise DataError(
                f"the {name} split is empty after the {settings.horizon_days}-day embargo; "
                "the feature store needs a longer history"
            )
        if part[LABEL_COLUMN].nunique() < 2:
            raise DataError(f"the {name} split contains only one class")

    def _split(name: str, part: pd.DataFrame) -> Split:
        return Split(
            name=name,
            frame=part.reset_index(drop=True),
            start=min(part["reference_date"]),
            end=max(part["reference_date"]),
        )

    train = _split("train", train_frame)
    validation = _split("validation", validation_frame)
    test = _split("test", test_frame)

    dataset = Dataset(
        train=train,
        validation=validation,
        test=test,
        fingerprint=fingerprint(frame),
        feature_version=str(frame["feature_version"].iloc[0]),
        horizon_days=settings.horizon_days,
        embargo_days=settings.horizon_days,
        stats={
            "reference_dates": len(dates),
            "rows_total": len(frame),
            "rows_dropped_to_embargo": len(frame) - len(train) - len(validation) - len(test),
            **{f"{split.name}_rows": len(split) for split in (train, validation, test)},
            **{
                f"{split.name}_base_rate_pct": round(100 * split.base_rate, 3)
                for split in (train, validation, test)
            },
        },
    )
    logger.info("dataset built", extra=dataset.stats)
    return dataset
