"""Writing the simulated business into PostgreSQL.

Usage is written with ``COPY`` because it is two orders of magnitude larger
than everything else -- 1.7 million rows for the default configuration -- and
a multi-row ``INSERT`` on that volume turns a ten-second step into a coffee
break. The other five tables are small enough that an executemany is both
fast enough and easier to read.
"""

from __future__ import annotations

import csv
import io

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from polaris.config import Settings, get_settings
from polaris.db.engine import get_engine
from polaris.exceptions import DatabaseError
from polaris.generation.simulator import SimulationResult
from polaris.logging_config import get_logger

logger = get_logger(__name__)

_USAGE_COLUMNS = (
    "account_id",
    "usage_date",
    "active_users",
    "sessions",
    "api_calls",
    "features_used",
    "error_events",
)


def _copy_usage(usage: pd.DataFrame, settings: Settings) -> int:
    schema = settings.source_schema
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    for row in usage[list(_USAGE_COLUMNS)].itertuples(index=False, name=None):
        writer.writerow(row)
    buffer.seek(0)

    engine = get_engine(settings)
    raw = engine.raw_connection()
    try:
        driver = raw.driver_connection
        if driver is None:  # pragma: no cover - a pooled connection always has one
            raise DatabaseError("no driver connection available for COPY")
        cursor = driver.cursor()
        with cursor.copy(
            f"COPY {schema}.usage_daily ({', '.join(_USAGE_COLUMNS)}) FROM STDIN WITH (FORMAT CSV)"
        ) as copy:
            copy.write(buffer.getvalue())
        raw.commit()
    except Exception as exc:  # pragma: no cover - surfaced with context
        raw.rollback()
        raise DatabaseError(f"COPY into usage_daily failed: {exc}") from exc
    finally:
        raw.close()
    return len(usage)


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Rows as dicts, with pandas' missing values turned back into None.

    pandas represents a missing integer as NaN, which is a float, and a
    missing timestamp as NaT. Both reach the driver as values the column
    cannot hold -- `satisfaction` is a smallint, and NaN is not one. Converting
    here rather than at every call site is the difference between one rule and
    six places to forget it.
    """
    cleaned = frame.astype(object).where(frame.notna(), None)
    return [{str(k): v for k, v in row.items()} for row in cleaned.to_dict("records")]


_INSERTS: dict[str, str] = {
    "account": """INSERT INTO {s}.account
        (account_id, company_name, segment, industry, country, signup_date, plan,
         seats, mrr_eur, contract_term_months, renewal_date, has_csm, churn_date)
        VALUES (:account_id, :company_name, :segment, :industry, :country, :signup_date,
                :plan, :seats, :mrr_eur, :contract_term_months, :renewal_date,
                :has_csm, :churn_date)""",
    "support_ticket": """INSERT INTO {s}.support_ticket
        (ticket_id, account_id, opened_at, closed_at, priority, category, satisfaction)
        VALUES (:ticket_id, :account_id, :opened_at, :closed_at, :priority, :category,
                :satisfaction)""",
    "invoice": """INSERT INTO {s}.invoice
        (invoice_id, account_id, issued_at, due_at, paid_at, amount_eur, status)
        VALUES (:invoice_id, :account_id, :issued_at, :due_at, :paid_at, :amount_eur, :status)""",
    "nps_response": """INSERT INTO {s}.nps_response (response_id, account_id, responded_at, score)
        VALUES (:response_id, :account_id, :responded_at, :score)""",
    "account_event": """INSERT INTO {s}.account_event
        (event_id, account_id, occurred_at, event_type, detail)
        VALUES (:event_id, :account_id, :occurred_at, :event_type, CAST(:detail AS JSONB))""",
}


def write_simulation(result: SimulationResult, settings: Settings | None = None) -> dict[str, int]:
    """Truncate and reload the source schema."""
    import json
    import uuid

    settings = settings or get_settings()
    schema = settings.source_schema
    engine = get_engine(settings)

    events = result.events.copy()
    if not events.empty:
        events["event_id"] = [f"EVT-{uuid.uuid4().hex[:12]}" for _ in range(len(events))]
        events["detail"] = events["detail"].map(json.dumps)

    try:
        with engine.begin() as conn:
            # Order matters: everything references account.
            for table in (
                "account_event",
                "nps_response",
                "invoice",
                "support_ticket",
                "usage_daily",
                "account",
            ):
                conn.execute(text(f"TRUNCATE TABLE {schema}.{table} CASCADE"))

            conn.execute(
                text(_INSERTS["account"].format(s=schema)),
                _records(result.accounts),
            )
            for name, frame in (
                ("support_ticket", result.tickets),
                ("invoice", result.invoices),
                ("nps_response", result.nps),
                ("account_event", events),
            ):
                if not frame.empty:
                    conn.execute(text(_INSERTS[name].format(s=schema)), _records(frame))
    except SQLAlchemyError as exc:
        raise DatabaseError(f"writing the simulated source failed: {exc}") from exc

    usage_rows = _copy_usage(result.usage, settings) if not result.usage.empty else 0

    counts = {
        "account": len(result.accounts),
        "usage_daily": usage_rows,
        "support_ticket": len(result.tickets),
        "invoice": len(result.invoices),
        "nps_response": len(result.nps),
        "account_event": len(events),
    }
    logger.info("simulated source written", extra=counts)
    return counts
