"""The serving API.

Kept deliberately small. It loads whatever is in the registry's production
stage, scores accounts from the feature store, and records what it answered.
Everything else -- how features are computed, what the threshold should be,
which model deserves to be here -- belongs upstream, and an API that decided
any of it would be a second place for those decisions to live.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from polaris import __version__
from polaris.config import get_settings
from polaris.db.engine import check_connection
from polaris.logging_config import configure_logging, get_logger
from polaris.registry.store import get_production
from polaris.serving.dependencies import clear_predictor, reload_predictor
from polaris.serving.routers import scoring

logger = get_logger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    app = FastAPI(
        title="polaris churn scoring",
        version=__version__,
        description=(
            "Scores B2B SaaS accounts for churn within 60 days. Synthetic data "
            "only. Features come from the point-in-time store, never from the "
            "request body -- a caller who could supply feature values could "
            "supply ones computed differently from the model's training data."
        ),
    )
    app.include_router(scoring.router)

    @app.get("/health", summary="Liveness")
    def health() -> dict[str, Any]:
        """Liveness, and honest about what is missing.

        Returns 200 with `model_loaded: false` rather than failing when no
        model is in production: the service is up, it simply has nothing to
        serve yet, and a load balancer should not take it out of rotation for
        that.
        """
        record = None
        try:
            record = get_production("churn-60d", settings)
        except Exception as exc:
            logger.warning("registry unreachable", extra={"error": str(exc)})
        return {
            "status": "ok",
            "version": __version__,
            "database": check_connection(settings),
            "model_loaded": record is not None,
            "model": ({"name": record.model_name, "version": record.version} if record else None),
        }

    @app.post("/admin/reload", summary="Pick up a newly promoted model")
    def reload() -> dict[str, Any]:
        """Applied without a restart, because a promotion should not need one."""
        clear_predictor()
        predictor = reload_predictor()
        return {
            "model": predictor.record.model_name,
            "version": predictor.record.version,
            "threshold": predictor.threshold,
        }

    logger.info("serving api ready", extra={"version": __version__})
    return app


app = create_app()
