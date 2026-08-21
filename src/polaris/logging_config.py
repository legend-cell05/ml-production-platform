"""Logging setup.

Text for humans at a terminal, JSON for anything that ships logs somewhere.
The JSON formatter keeps the ``extra`` fields as real keys rather than
interpolating them into the message, so a migration run can be queried by
entity and batch rather than grepped.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from typing import Any

_RESERVED = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": dt.datetime.fromtimestamp(record.created, tz=dt.UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _SafeLogger(logging.Logger):
    """A logger whose ``extra`` cannot crash the program.

    ``LogRecord`` owns a set of attribute names, and passing any of them in
    ``extra`` raises ``KeyError`` -- at the log call, in production, long after
    the code was reviewed. That is a real trap here: a load report naturally
    has a field called ``created``, which is also the timestamp every
    ``LogRecord`` carries.

    Renaming the collision to ``ctx_created`` keeps the value, keeps the log
    line, and keeps the program running. Logging must never be the thing that
    breaks a migration.
    """

    def makeRecord(  # noqa: N802 - the name is logging's API, not a choice
        self,
        name: str,
        level: int,
        fn: str,
        lno: int,
        msg: object,
        args: Any,
        exc_info: Any,
        func: str | None = None,
        extra: Any = None,
        sinfo: str | None = None,
    ) -> logging.LogRecord:
        if extra:
            extra = {
                (f"ctx_{key}" if key in _RESERVED else key): value for key, value in extra.items()
            }
        return super().makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra, sinfo)


logging.setLoggerClass(_SafeLogger)


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    """Install a single stdout handler. Safe to call more than once."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if fmt.lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root.addHandler(handler)
    root.setLevel(level.upper())

    # These two are chatty at INFO and say nothing polaris does not already log.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
