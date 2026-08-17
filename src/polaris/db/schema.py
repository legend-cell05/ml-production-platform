"""Schema lifecycle: create, inspect, drop.

``initialise_database`` is idempotent and is what every entry point calls
first. Dropping is a separate, explicitly named operation, because a tool that
can silently destroy the crosswalk is a tool nobody should run twice.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from polaris.config import Settings, get_settings
from polaris.db.engine import get_engine
from polaris.db.sql_files import read_sql, split_statements, sql_dir
from polaris.exceptions import DatabaseError
from polaris.logging_config import get_logger

logger = get_logger(__name__)


def _apply_file(relative_path: str, settings: Settings) -> int:
    sql = read_sql(relative_path, settings)
    statements = split_statements(sql)
    engine = get_engine(settings)
    try:
        with engine.begin() as conn:
            for statement in statements:
                conn.execute(text(statement))
    except SQLAlchemyError as exc:
        raise DatabaseError(f"{relative_path}: {exc}") from exc
    return len(statements)


def initialise_database(settings: Settings | None = None) -> dict[str, int]:
    """Create both schemas and every table. Safe to re-run."""
    settings = settings or get_settings()
    applied: dict[str, int] = {}
    schema_files = sorted((sql_dir(settings) / "schema").glob("*.sql"))
    if not schema_files:
        raise DatabaseError(f"no schema files under {sql_dir(settings) / 'schema'}")
    for path in schema_files:
        applied[path.name] = _apply_file(f"schema/{path.name}", settings)
    logger.info("database initialised", extra={"files": len(applied)})
    return applied


def drop_schemas(settings: Settings | None = None, *, include_source: bool = True) -> None:
    """Drop polaris's own schemas.

    ``include_source`` exists because "retrain from scratch" and "throw away
    the source system as well" are different intentions, and a function that
    offers only the second will eventually be called when the first was
    meant.
    """
    settings = settings or get_settings()
    engine = get_engine(settings)
    targets = [settings.ml_schema, settings.feature_schema]
    if include_source:
        targets.append(settings.source_schema)
    try:
        with engine.begin() as conn:
            for schema in targets:
                conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
    except SQLAlchemyError as exc:
        raise DatabaseError(f"failed to drop schemas: {exc}") from exc
    logger.warning("schemas dropped", extra={"schemas": targets})


def table_counts(settings: Settings | None = None) -> dict[str, int]:
    """Exact row counts per table in all three schemas.

    Exact, not ``pg_class.reltuples``: an estimate is zero until autovacuum
    has been past, which makes a freshly loaded table look empty in a report
    that is meant to prove it is not.
    """
    settings = settings or get_settings()
    engine = get_engine(settings)
    counts: dict[str, int] = {}
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """SELECT table_schema, table_name
                       FROM information_schema.tables
                       WHERE table_schema = ANY(:schemas) AND table_type = 'BASE TABLE'
                       ORDER BY table_schema, table_name"""
                ),
                {
                    "schemas": [
                        settings.source_schema,
                        settings.feature_schema,
                        settings.ml_schema,
                    ]
                },
            ).all()
            for schema, table in rows:
                # Annotated because SQLAlchemy 2.1 types scalar_one() as a
                # TypeVar that cannot be solved here: the query is built from
                # a string, so there is no Select[...] to infer from.
                count: int = conn.execute(
                    text(f"SELECT COUNT(*) FROM {schema}.{table}")
                ).scalar_one()
                counts[f"{schema}.{table}"] = count
    except SQLAlchemyError as exc:
        raise DatabaseError(f"failed to count tables: {exc}") from exc
    return counts
