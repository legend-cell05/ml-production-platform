"""The model registry and the promotion gate."""

from polaris.registry.promotion import Check, GateDecision, evaluate_gate, promote
from polaris.registry.store import (
    ModelRecord,
    get_production,
    get_version,
    list_versions,
    register,
    segment_metrics,
    set_stage,
)

__all__ = [
    "Check",
    "GateDecision",
    "ModelRecord",
    "evaluate_gate",
    "get_production",
    "get_version",
    "list_versions",
    "promote",
    "register",
    "segment_metrics",
    "set_stage",
]
