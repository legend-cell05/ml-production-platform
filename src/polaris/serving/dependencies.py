"""Shared state for the serving API.

The predictor is loaded once and cached, because loading a calibrated
gradient-boosting pipeline takes long enough that doing it per request would
dominate the latency budget. ``reload_predictor`` exists so that a promotion
takes effect without a restart -- the alternative is an operator restarting
pods after every model change, which is how a team ends up serving last
month's model for a week.
"""

from __future__ import annotations

from fastapi import HTTPException, status

from polaris.config import get_settings
from polaris.exceptions import ModelNotFound, ServingError
from polaris.logging_config import get_logger
from polaris.serving.predictor import Predictor

logger = get_logger(__name__)

_PREDICTOR: Predictor | None = None


def reload_predictor(model_name: str = "churn-60d") -> Predictor:
    global _PREDICTOR
    _PREDICTOR = Predictor.from_production(model_name, get_settings())
    logger.info(
        "predictor loaded",
        extra={"model": _PREDICTOR.record.model_name, "version": _PREDICTOR.record.version},
    )
    return _PREDICTOR


def get_predictor() -> Predictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        try:
            _PREDICTOR = reload_predictor()
        except (ModelNotFound, ServingError) as exc:
            # 503 rather than 500: nothing is broken, there is simply no model
            # in production yet, and a caller should retry after one is
            # promoted.
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "NO_MODEL_IN_PRODUCTION", "message": str(exc)},
            ) from exc
    return _PREDICTOR


def clear_predictor() -> None:
    global _PREDICTOR
    _PREDICTOR = None
