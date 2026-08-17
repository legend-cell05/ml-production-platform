"""Engine management.

One engine per database URL, kept in a module-level dict. ``lru_cache`` on a
function taking ``Settings`` looks tidier and does not work: a Pydantic model
is not hashable, and even when it is made so, two settings objects that differ
only in a log level would open two connection pools to the same database.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from polaris.config import Settings, get_settings
from polaris.exceptions import DatabaseError
from polaris.logging_config import get_logger

logger = get_logger(__name__)

_ENGINES: dict[str, Engine] = {}


def get_engine(settings: Settings | None = None) -> Engine:
    """Return the engine for these settings, creating it on first use."""
    settings = settings or get_settings()
    dsn = settings.dsn
    engine = _ENGINES.get(dsn)
    if engine is None:
        engine = create_engine(
            dsn,
            pool_pre_ping=True,  # a pooled connection can be dead after an idle night
            pool_size=5,
            max_overflow=5,
            future=True,
        )
        _ENGINES[dsn] = engine
        logger.debug("engine created", extra={"dsn": settings.safe_dsn})
    return engine


def dispose_engines() -> None:
    """Close every pool. Used by tests and by the CLI before exiting."""
    for engine in _ENGINES.values():
        engine.dispose()
    _ENGINES.clear()


@contextmanager
def raw_connection(settings: Settings | None = None) -> Iterator[Any]:
    """Yield the underlying psycopg connection.

    COPY is not part of the SQLAlchemy Core API, so the loader needs the
    driver connection itself. Committed on success, rolled back on failure.
    """
    engine = get_engine(settings)
    connection = engine.raw_connection()
    try:
        yield connection.driver_connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def check_connection(settings: Settings | None = None, *, retries: int = 1) -> bool:
    """True when the database answers ``SELECT 1``.

    Retries because ``docker compose up`` starts the application while
    PostgreSQL is still initialising, and a migration tool that dies on that
    race is a migration tool nobody can run from compose.
    """
    settings = settings or get_settings()
    for attempt in range(1, retries + 1):
        try:
            with get_engine(settings).connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError as exc:
            if attempt == retries:
                logger.error(
                    "database unreachable",
                    extra={"dsn": settings.safe_dsn, "error": str(exc).splitlines()[0]},
                )
                return False
            time.sleep(min(2.0 * attempt, 5.0))
    return False


def execute_returning_scalar(
    sql: str, params: dict[str, Any] | None = None, settings: Settings | None = None
) -> Any:
    """Run a single scalar query, translating driver errors."""
    try:
        with get_engine(settings).connect() as conn:
            return conn.execute(text(sql), params or {}).scalar_one()
    except SQLAlchemyError as exc:
        raise DatabaseError(f"query failed: {exc}") from exc
