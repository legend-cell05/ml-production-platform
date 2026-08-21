"""Request and response shapes for the serving API."""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    """Score accounts that already have features in the store."""

    account_ids: list[str] = Field(min_length=1, max_length=5000)
    reference_date: dt.date | None = Field(
        default=None,
        description="Defaults to the most recent reference date in the feature store.",
    )
    explain: bool = True


class ScoreAllRequest(BaseModel):
    """Score every account for a reference date."""

    reference_date: dt.date | None = None
    explain: bool = False
    limit: int = Field(default=5000, ge=1, le=50_000)


class ContributionOut(BaseModel):
    group: str
    label: str
    contribution: float
    driver: str | None = None
    driver_value: Any = None


class PredictionOut(BaseModel):
    account_id: str
    reference_date: dt.date
    probability: float
    decision: bool
    threshold: float
    expected_value_eur: float
    model: dict[str, Any]
    contributions: list[ContributionOut] = Field(default_factory=list)
    latency_ms: float


class ScoreResponse(BaseModel):
    model_name: str
    version: int
    reference_date: dt.date
    scored: int
    flagged: int
    predictions: list[PredictionOut]


class ModelInfo(BaseModel):
    model_name: str
    version: int
    stage: str
    algorithm: str
    feature_version: str
    decision_threshold: float
    trained_metrics: dict[str, Any]
    promoted_at: dt.datetime | None = None
