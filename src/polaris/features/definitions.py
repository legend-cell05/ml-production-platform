"""The feature contract.

The list below is the single source of truth for what the model may see. It
drives the training pipeline's column selection, the serving layer's payload
validation, the drift monitor's per-feature checks, and the documentation.
Keeping one list rather than four is what stops a feature from being added to
training and forgotten in serving -- the failure that produces a model which
works in the notebook and predicts nonsense in production.

``FEATURE_VERSION`` changes whenever this list or the SQL that computes it
changes. A model trained under one version cannot be served under another,
and the registry refuses to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FEATURE_VERSION = "1.2.0"

Kind = Literal["numeric", "categorical", "boolean"]


@dataclass(frozen=True)
class Feature:
    """One column the model is allowed to use."""

    name: str
    kind: Kind
    description: str
    # Set when a missing value carries meaning rather than being an accident.
    # "No NPS response" is not a gap to impute, it is a fact about the account.
    missing_means: str | None = None


FEATURES: tuple[Feature, ...] = (
    # --- contract and identity ---------------------------------------------
    Feature("segment", "categorical", "SMB, mid-market or enterprise."),
    Feature("industry", "categorical", "Customer's industry."),
    Feature("country", "categorical", "Billing country."),
    Feature("plan", "categorical", "Subscription tier."),
    Feature("contract_term_months", "numeric", "1 for monthly, 12 or 24 for committed terms."),
    Feature("has_csm", "boolean", "Whether a customer success manager is assigned."),
    Feature("tenure_days", "numeric", "Days since signup."),
    Feature(
        "days_to_renewal",
        "numeric",
        "Days until the contract renews.",
        missing_means="no scheduled renewal (monthly contract)",
    ),
    # --- commercial ---------------------------------------------------------
    Feature("mrr_eur", "numeric", "Monthly recurring revenue."),
    Feature("seats", "numeric", "Licensed seats."),
    Feature("had_downgrade_180d", "boolean", "A plan downgrade in the last 180 days."),
    Feature("had_upgrade_180d", "boolean", "A plan upgrade in the last 180 days."),
    # --- usage --------------------------------------------------------------
    Feature(
        "active_users_28d_avg",
        "numeric",
        "Mean daily active users over 28 days.",
        missing_means="no usage records at all in the window",
    ),
    Feature(
        "seat_utilisation_28d",
        "numeric",
        "Active users over licensed seats.",
        missing_means="no usage records, or zero seats",
    ),
    Feature("sessions_28d", "numeric", "Sessions in the last 28 days."),
    Feature("api_calls_28d", "numeric", "API calls in the last 28 days."),
    Feature("features_used_28d", "numeric", "Mean distinct features used per day."),
    Feature(
        "usage_trend_28d_vs_90d",
        "numeric",
        "28-day average over 90-day average. Below 1 means the account is cooling.",
        missing_means="not enough history for a trend",
    ),
    Feature("days_since_last_use", "numeric", "Days since anyone last logged in."),
    Feature("zero_usage_days_28d", "numeric", "Days with no active user in the last 28."),
    Feature(
        "error_rate_28d",
        "numeric",
        "Error events per session.",
        missing_means="no sessions to divide by",
    ),
    # --- support ------------------------------------------------------------
    Feature("tickets_90d", "numeric", "Tickets opened in the last 90 days."),
    Feature("tickets_p1_90d", "numeric", "P1 tickets opened in the last 90 days."),
    Feature("open_tickets", "numeric", "Tickets still open as of the reference date."),
    Feature(
        "avg_resolution_hours_90d",
        "numeric",
        "Mean resolution time, tickets closed in window.",
        missing_means="nothing was resolved in the window",
    ),
    Feature(
        "avg_satisfaction_180d",
        "numeric",
        "Mean CSAT, 1-5.",
        missing_means="the customer never answered a survey",
    ),
    # --- billing ------------------------------------------------------------
    Feature("late_payments_180d", "numeric", "Invoices paid late or overdue in 180 days."),
    Feature("failed_payments_90d", "numeric", "Failed payments in 90 days."),
    Feature("max_days_overdue_180d", "numeric", "Worst overdue stretch in 180 days."),
    # --- sentiment and relationship ----------------------------------------
    Feature(
        "last_nps_score",
        "numeric",
        "Most recent NPS score, 0-10.",
        missing_means="never responded to a survey",
    ),
    Feature(
        "days_since_nps",
        "numeric",
        "Days since the last NPS response.",
        missing_means="never responded to a survey",
    ),
    Feature("champion_left_180d", "boolean", "The main contact left in the last 180 days."),
    Feature("qbr_held_180d", "boolean", "A quarterly business review took place."),
)

# Columns present in the feature store that the model must NOT see. Listed
# explicitly so that "everything except the label" is never used as the
# selection rule -- that rule is how `reference_date` ends up as a feature and
# the model learns the calendar.
EXCLUDED_COLUMNS: tuple[str, ...] = (
    "reference_date",
    "account_id",
    "churned_in_horizon",
    "label_known_at",
    "feature_version",
    "computed_at",
    # Computed by the SQL but left NULL: the simulated billing system does not
    # version price changes, so an honest value cannot be produced.
    "mrr_change_90d_pct",
    "seats_change_90d",
)

LABEL_COLUMN = "churned_in_horizon"

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)
NUMERIC_FEATURES: tuple[str, ...] = tuple(f.name for f in FEATURES if f.kind == "numeric")
CATEGORICAL_FEATURES: tuple[str, ...] = tuple(f.name for f in FEATURES if f.kind == "categorical")
BOOLEAN_FEATURES: tuple[str, ...] = tuple(f.name for f in FEATURES if f.kind == "boolean")


# ---------------------------------------------------------------------------
# Feature groups.
#
# Used by the serving layer's explanation, and the reason it exists: features
# inside a group are strongly correlated, and perturbing one of them on its own
# produces a row the model has never seen. An account with 0.14 active users
# and zero sessions is coherent; the same account with the median eight active
# users and still zero sessions is not, and the model scores that
# contradiction as *more* risky -- which flips the sign of the explanation and
# tells a customer success manager the opposite of the truth.
#
# Perturbing a whole group keeps the row coherent, and answers the question
# actually being asked: which part of this account's behaviour is driving the
# score?
# ---------------------------------------------------------------------------
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "product_usage": (
        "active_users_28d_avg",
        "seat_utilisation_28d",
        "sessions_28d",
        "api_calls_28d",
        "features_used_28d",
        "usage_trend_28d_vs_90d",
        "days_since_last_use",
        "zero_usage_days_28d",
        "error_rate_28d",
    ),
    "support": (
        "tickets_90d",
        "tickets_p1_90d",
        "open_tickets",
        "avg_resolution_hours_90d",
        "avg_satisfaction_180d",
    ),
    "billing": ("late_payments_180d", "failed_payments_90d", "max_days_overdue_180d"),
    "sentiment": ("last_nps_score", "days_since_nps"),
    "relationship": ("champion_left_180d", "qbr_held_180d", "has_csm"),
    "commercial": ("mrr_eur", "seats", "had_downgrade_180d", "had_upgrade_180d", "plan"),
    "contract": ("contract_term_months", "tenure_days", "days_to_renewal"),
    "firmographics": ("segment", "industry", "country"),
}

GROUP_LABELS: dict[str, str] = {
    "product_usage": "how much the product is being used",
    "support": "support history",
    "billing": "payment behaviour",
    "sentiment": "survey responses",
    "relationship": "the account relationship",
    "commercial": "the commercial shape of the account",
    "contract": "contract and tenure",
    "firmographics": "who the customer is",
}


def group_of(feature: str) -> str:
    for group, members in FEATURE_GROUPS.items():
        if feature in members:
            return group
    raise KeyError(f"feature {feature!r} belongs to no group")


def feature_by_name(name: str) -> Feature:
    for feature in FEATURES:
        if feature.name == name:
            return feature
    raise KeyError(f"unknown feature {name!r}")
