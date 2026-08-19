"""Scoring endpoints.

Features are read from the store rather than accepted in the request body, on
purpose. A caller who can supply arbitrary feature values can supply values
computed a different way from the ones the model was trained on -- which is
training/serving skew, arriving through the front door. The API takes account
identifiers and a date; the platform decides what those mean.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from polaris.config import Settings, get_settings
from polaris.data.dataset import load_scoreable
from polaris.exceptions import ServingError
from polaris.features.builder import reference_dates
from polaris.serving.dependencies import get_predictor
from polaris.serving.predictor import Predictor
from polaris.serving.schemas import (
    ModelInfo,
    ScoreAllRequest,
    ScoreRequest,
    ScoreResponse,
)

router = APIRouter(prefix="/api/v1", tags=["scoring"])


def _resolve_date(requested: dt.date | None, settings: Settings) -> dt.date:
    if requested is not None:
        return requested
    dates = reference_dates(settings)
    if not dates:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "NO_FEATURES", "message": "the feature store is empty"},
        )
    return dates[-1]


@router.post("/score", response_model=ScoreResponse, summary="Score named accounts")
def score(
    request: ScoreRequest,
    predictor: Annotated[Predictor, Depends(get_predictor)],
) -> ScoreResponse:
    settings = predictor.settings
    reference_date = _resolve_date(request.reference_date, settings)
    frame = load_scoreable(reference_date, settings)
    frame = frame[frame["account_id"].isin(request.account_ids)]

    if frame.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NO_FEATURES_FOR_ACCOUNTS",
                "message": f"no feature rows for those accounts on {reference_date}",
            },
        )

    try:
        predictions = predictor.predict(frame, explain=request.explain)
    except ServingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "CANNOT_SCORE", "message": str(exc)},
        ) from exc

    return ScoreResponse(
        model_name=predictor.record.model_name,
        version=predictor.record.version,
        reference_date=reference_date,
        scored=len(predictions),
        flagged=sum(1 for p in predictions if p.decision),
        predictions=[p.as_dict() for p in predictions],  # type: ignore[misc]
    )


@router.post("/score-all", response_model=ScoreResponse, summary="Score every account")
def score_all(
    request: ScoreAllRequest,
    predictor: Annotated[Predictor, Depends(get_predictor)],
) -> ScoreResponse:
    """The fortnightly batch the customer success team works from.

    Explanations default to off here: they cost an extra prediction per
    account and nobody reads five thousand of them. The list is worked from
    the top, and the accounts that get opened get explained individually.
    """
    settings = predictor.settings
    reference_date = _resolve_date(request.reference_date, settings)
    frame = load_scoreable(reference_date, settings).head(request.limit)
    if frame.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NO_FEATURES", "message": f"no feature rows for {reference_date}"},
        )
    predictions = predictor.predict(frame, explain=request.explain)
    predictions.sort(key=lambda p: p.probability, reverse=True)
    return ScoreResponse(
        model_name=predictor.record.model_name,
        version=predictor.record.version,
        reference_date=reference_date,
        scored=len(predictions),
        flagged=sum(1 for p in predictions if p.decision),
        predictions=[p.as_dict() for p in predictions],  # type: ignore[misc]
    )


@router.get("/model", response_model=ModelInfo, summary="What is serving")
def model_info(predictor: Annotated[Predictor, Depends(get_predictor)]) -> ModelInfo:
    record = predictor.record
    return ModelInfo(
        model_name=record.model_name,
        version=record.version,
        stage=record.stage,
        algorithm=predictor.algorithm,
        feature_version=record.feature_version,
        decision_threshold=predictor.threshold,
        trained_metrics={
            k: v for k, v in record.metrics.items() if isinstance(v, (int, float, str))
        },
        promoted_at=record.promoted_at,
    )


@router.get("/economics", summary="The numbers behind the threshold")
def economics(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, float]:
    """Published, because the threshold is an economic decision.

    Anyone questioning why an account was or was not flagged is really
    questioning these three numbers, and they should not have to read the
    source to find them.
    """
    from polaris.economics import Economics

    e = Economics.from_settings(settings)
    return {
        "value_of_saved_account_eur": e.value_of_saved_account,
        "cost_of_intervention_eur": e.cost_of_intervention,
        "intervention_success_rate": e.intervention_success_rate,
        "value_of_true_positive_eur": e.value_of_true_positive,
        "cost_of_false_positive_eur": e.cost_of_false_positive,
        "break_even_probability": round(e.break_even_probability, 5),
    }
