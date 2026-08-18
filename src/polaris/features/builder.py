"""Building the feature store, one reference date at a time.

A reference date is a moment the model is asked "who will churn in the next
sixty days?". The feature store holds one row per (reference date, account),
and the whole platform rests on those rows containing nothing that was
unknowable on that date.

Why monthly reference dates rather than one row per account: churn is a
*repeated* decision. An account that survived January can churn in February,
and a dataset with one row per account throws away most of the evidence and
introduces a subtle survivorship bias in what is left.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from polaris.config import Settings, get_settings
from polaris.db.engine import get_engine
from polaris.db.sql_files import read_sql, split_statements
from polaris.exceptions import DatabaseError
from polaris.features.definitions import FEATURE_VERSION
from polaris.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BuildReport:
    reference_dates: list[dt.date]
    rows: int
    labelled_rows: int
    positive_rows: int
    feature_version: str

    @property
    def base_rate_pct(self) -> float:
        return 100.0 * self.positive_rows / self.labelled_rows if self.labelled_rows else 0.0


def data_bounds(settings: Settings | None = None) -> tuple[dt.date, dt.date]:
    """The first and last day the source system has any evidence for.

    The upper bound is what makes a label knowable: a reference date whose
    horizon extends past it has no answer yet, and the SQL writes NULL rather
    than guessing.
    """
    settings = settings or get_settings()
    schema = settings.source_schema
    try:
        with get_engine(settings).connect() as conn:
            row = conn.execute(
                text(f"SELECT MIN(usage_date), MAX(usage_date) FROM {schema}.usage_daily")
            ).one()
    except SQLAlchemyError as exc:
        raise DatabaseError(f"reading the data bounds failed: {exc}") from exc
    if row[0] is None:
        raise DatabaseError("the source system is empty -- run `polaris simulate` first")
    return row[0], row[1]


def reference_dates(settings: Settings | None = None) -> list[dt.date]:
    """Every reference date the data can support.

    Starts 120 days after the first usage record, because the 90-day windows
    need history to be meaningful, and stops at the last day of data. Dates
    whose horizon has not elapsed still get feature rows -- they are what
    scoring uses -- but no label.
    """
    settings = settings or get_settings()
    first, last = data_bounds(settings)
    start = first + dt.timedelta(days=120)
    step = settings.reference_interval_days

    dates: list[dt.date] = []
    current = start
    while current <= last:
        dates.append(current)
        current += dt.timedelta(days=step)
    return dates


def build_reference_date(reference_date: dt.date, settings: Settings | None = None) -> int:
    """Compute and store the features for one date. Idempotent."""
    settings = settings or get_settings()
    _, last_known = data_bounds(settings)
    sql = read_sql("features/build_features.sql", settings)
    statements = split_statements(sql)
    if len(statements) != 1:
        raise DatabaseError(
            f"build_features.sql must be a single statement, found {len(statements)}"
        )

    try:
        with get_engine(settings).begin() as conn:
            result = conn.execute(
                text(statements[0]),
                {
                    "reference_date": reference_date,
                    "horizon_days": settings.horizon_days,
                    "feature_version": FEATURE_VERSION,
                    "max_known_date": last_known,
                },
            )
    except SQLAlchemyError as exc:
        raise DatabaseError(f"building features for {reference_date} failed: {exc}") from exc
    return int(result.rowcount or 0)


def build_all(settings: Settings | None = None, *, rebuild: bool = False) -> BuildReport:
    """Build every reference date the data supports."""
    settings = settings or get_settings()
    schema = settings.feature_schema

    if rebuild:
        # A feature definition change invalidates every stored row. Rebuilding
        # is cheap; serving a model on rows computed by a different definition
        # is not.
        try:
            with get_engine(settings).begin() as conn:
                conn.execute(text(f"TRUNCATE TABLE {schema}.churn_features"))
        except SQLAlchemyError as exc:
            raise DatabaseError(f"truncating the feature store failed: {exc}") from exc

    dates = reference_dates(settings)
    total = 0
    for reference_date in dates:
        total += build_reference_date(reference_date, settings)

    try:
        with get_engine(settings).connect() as conn:
            counted = conn.execute(
                text(
                    f"""SELECT COUNT(*) FILTER (WHERE churned_in_horizon IS NOT NULL),
                               COUNT(*) FILTER (WHERE churned_in_horizon)
                        FROM {schema}.churn_features"""
                )
            ).one()
            labelled, positives = int(counted[0]), int(counted[1])
    except SQLAlchemyError as exc:
        raise DatabaseError(f"counting the feature store failed: {exc}") from exc

    report = BuildReport(
        reference_dates=dates,
        rows=total,
        labelled_rows=labelled,
        positive_rows=positives,
        feature_version=FEATURE_VERSION,
    )
    logger.info(
        "feature store built",
        extra={
            "reference_dates": len(dates),
            "rows": report.rows,
            "labelled": report.labelled_rows,
            "base_rate_pct": round(report.base_rate_pct, 3),
            "feature_version": FEATURE_VERSION,
        },
    )
    return report
