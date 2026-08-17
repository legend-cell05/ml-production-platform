"""Simulating Vertex Systems.

The hard part of a synthetic churn dataset is not generating rows, it is
making the problem *the right difficulty*. Two failure modes, both useless:

* churn drawn independently of the features -- nothing is learnable, and the
  whole platform is scored against noise;
* churn drawn from a clean function of two columns -- a logistic regression
  reaches 0.99 AUC, every gate passes trivially, and none of the machinery
  is exercised.

So accounts here have a **latent health state** that drifts over time and
reacts to events: a champion leaving, a ticket storm, a downgrade, a failed
payment. Churn is drawn from a hazard built on that state, which is only
partially observable through the product usage and support records the
feature store can see. The signal is real, it is noisy, and it arrives before
the event -- usage decays in the months leading up to a churn, which is the
leading indicator a model is supposed to find.

Everything is deterministic given ``POLARIS_RANDOM_SEED``.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from polaris.config import Settings, get_settings
from polaris.logging_config import get_logger

logger = get_logger(__name__)

SEGMENTS: tuple[str, ...] = ("smb", "mid_market", "enterprise")
SEGMENT_WEIGHTS: tuple[float, ...] = (0.62, 0.28, 0.10)

PLANS: tuple[str, ...] = ("starter", "business", "premium")

INDUSTRIES: tuple[str, ...] = (
    "manufacturing",
    "retail",
    "logistics",
    "professional_services",
    "healthcare",
    "software",
    "public_sector",
    "finance",
    "construction",
    "education",
)

COUNTRIES: tuple[str, ...] = ("FR", "BE", "DE", "ES", "IT", "NL", "CH", "UK")
COUNTRY_WEIGHTS: tuple[float, ...] = (0.42, 0.10, 0.12, 0.09, 0.08, 0.07, 0.06, 0.06)

TICKET_CATEGORIES: tuple[str, ...] = (
    "bug",
    "how_to",
    "integration",
    "performance",
    "billing",
    "feature_request",
)

# How many months before a churn the decline becomes visible in usage. This is
# the single most consequential number in the simulation: set it to zero and
# the problem is unlearnable, set it to twelve and it is trivial.
DECLINE_MONTHS = 3


@dataclass
class SimulationResult:
    """Everything the simulation produced, as frames ready to be written."""

    accounts: pd.DataFrame
    usage: pd.DataFrame
    tickets: pd.DataFrame
    invoices: pd.DataFrame
    nps: pd.DataFrame
    events: pd.DataFrame
    stats: dict[str, Any] = field(default_factory=dict)


def _month_starts(start: dt.date, months: int) -> list[dt.date]:
    out: list[dt.date] = []
    year, month = start.year, start.month
    for _ in range(months):
        out.append(dt.date(year, month, 1))
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    result: np.ndarray = 1.0 / (1.0 + np.exp(-x))
    return result


def _build_accounts(
    rng: np.random.Generator, settings: Settings, first_month: dt.date, last_month: dt.date
) -> pd.DataFrame:
    """Companies, with the attributes that exist before any behaviour."""
    n = settings.n_accounts
    segment = rng.choice(SEGMENTS, size=n, p=SEGMENT_WEIGHTS)

    # Seats and price follow the segment, with enough spread that segment is
    # not a proxy for size.
    seat_mean = np.where(segment == "smb", 12, np.where(segment == "mid_market", 60, 320))
    seats = np.maximum(2, rng.poisson(seat_mean)).astype(int)

    plan_p = np.where(segment == "smb", 0.15, np.where(segment == "mid_market", 0.45, 0.80))
    plan_roll = rng.random(n)
    plan = np.where(
        plan_roll < plan_p * 0.45,
        "premium",
        np.where(plan_roll < plan_p * 0.45 + 0.45, "business", "starter"),
    )

    price_per_seat = np.where(plan == "premium", 42.0, np.where(plan == "business", 26.0, 14.0))
    mrr = np.round(seats * price_per_seat * rng.normal(1.0, 0.08, n), 2)
    mrr = np.maximum(mrr, 50.0)

    term = np.where(segment == "enterprise", 24, np.where(segment == "mid_market", 12, 1))
    # A CSM is assigned by revenue, with exceptions in both directions.
    has_csm = (mrr > 2500) ^ (rng.random(n) < 0.08)

    # Signup dates: most accounts predate the observation window, which is
    # what a real customer base looks like.
    span_days = (last_month - first_month).days
    signup_offset = rng.integers(-1500, span_days - 30, size=n)
    signup = np.array([first_month + dt.timedelta(days=int(d)) for d in signup_offset])

    accounts = pd.DataFrame(
        {
            "account_id": [f"ACC-{i:06d}" for i in range(1, n + 1)],
            "company_name": [
                f"{rng.choice(['Nord', 'Val', 'Mont', 'Rive', 'Cap', 'Pont', 'Haut', 'Clair'])}"
                f"{rng.choice(['tech', 'log', 'ware', 'flow', 'core', 'works', 'group', 'lab'])}"
                f" {rng.choice(['SA', 'SAS', 'GmbH', 'BV', 'Ltd', 'SpA'])}"
                for _ in range(n)
            ],
            "segment": segment,
            "industry": rng.choice(INDUSTRIES, size=n),
            "country": rng.choice(COUNTRIES, size=n, p=COUNTRY_WEIGHTS),
            "signup_date": signup,
            "plan": plan,
            "seats": seats,
            "mrr_eur": mrr,
            "contract_term_months": term,
            "has_csm": has_csm,
        }
    )
    return accounts


def _simulate_health(
    rng: np.random.Generator, accounts: pd.DataFrame, months: list[dt.date]
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Walk every account through the months, drawing events and churn.

    Returns the monthly health matrix, the month index at which each account
    churned (or -1), and the commercial events that happened along the way.

    Health is a latent variable in [0, 1]. Nothing observes it directly; the
    feature store sees only its consequences -- how much the product is used,
    how many tickets are opened, whether invoices are paid on time.
    """
    n = len(accounts)
    n_months = len(months)
    segment = accounts["segment"].to_numpy()
    has_csm = accounts["has_csm"].to_numpy()
    term = accounts["contract_term_months"].to_numpy()
    signup = accounts["signup_date"].to_numpy()

    health = np.zeros((n, n_months), dtype=float)
    current = rng.beta(6, 2, size=n)  # most accounts start healthy
    churn_month = np.full(n, -1, dtype=int)
    champion_left_month = np.full(n, -1, dtype=int)
    downgrade_month = np.full(n, -1, dtype=int)
    upgrade_month = np.full(n, -1, dtype=int)
    events: list[dict[str, Any]] = []

    # Baseline monthly hazard by segment: SMB churns far more than enterprise,
    # which is what makes the segment-wise promotion gate meaningful.
    # Roughly 15%, 11% and 8% a year for SMB, mid-market and enterprise --
    # the spread a B2B SaaS business actually sees, and wide enough that the
    # per-segment promotion gate has something to measure. An earlier version
    # put enterprise at -5.3, which produced test windows containing zero
    # enterprise churns and a gate that could not evaluate its most valuable
    # segment.
    base_hazard = np.where(segment == "smb", -3.9, np.where(segment == "mid_market", -4.6, -4.9))

    for m, month in enumerate(months):
        active = (churn_month < 0) & (signup <= month)

        # --- drift and shocks ------------------------------------------
        drift = rng.normal(0.0, 0.06, size=n)
        # A slow secular improvement for accounts with a CSM: the product team
        # believes this and the data will let a model half-confirm it.
        drift += np.where(has_csm, 0.012, -0.004)
        current = np.clip(current + drift, 0.02, 1.0)

        champion_leaves = active & (rng.random(n) < 0.010)
        current = np.where(
            champion_leaves, np.clip(current - rng.uniform(0.15, 0.35, n), 0.02, 1.0), current
        )
        for idx in np.flatnonzero(champion_leaves):
            champion_left_month[idx] = m
            events.append(
                {
                    "account_id": accounts["account_id"].iloc[idx],
                    "occurred_at": month,
                    "event_type": "champion_left",
                    "detail": {},
                }
            )

        downgrades = active & (current < 0.45) & (rng.random(n) < 0.05)
        for idx in np.flatnonzero(downgrades):
            downgrade_month[idx] = m
            events.append(
                {
                    "account_id": accounts["account_id"].iloc[idx],
                    "occurred_at": month,
                    "event_type": "plan_downgrade",
                    "detail": {},
                }
            )

        upgrades = active & (current > 0.75) & (rng.random(n) < 0.03)
        for idx in np.flatnonzero(upgrades):
            upgrade_month[idx] = m
            current[idx] = min(1.0, current[idx] + 0.05)
            events.append(
                {
                    "account_id": accounts["account_id"].iloc[idx],
                    "occurred_at": month,
                    "event_type": "plan_upgrade",
                    "detail": {},
                }
            )

        # --- renewal pressure ------------------------------------------
        # Churn concentrates at contract boundaries: an annual contract is
        # hard to leave in month seven and easy to leave in month twelve.
        months_since_signup = np.array(
            [max(0, (month.year - s.year) * 12 + month.month - s.month) for s in signup]
        )
        at_renewal = (term > 1) & (months_since_signup > 0) & (months_since_signup % term == 0)
        renewal_bonus = np.where(at_renewal, 1.6, 0.0)
        # Monthly contracts can leave at any time.
        renewal_bonus = np.where(term == 1, 0.35, renewal_bonus)

        # --- hazard ----------------------------------------------------
        logit = (
            base_hazard
            + renewal_bonus
            + 3.4 * (0.55 - current)  # unhealthy accounts leave
            + np.where(champion_left_month >= 0, 0.55, 0)  # and stay fragile afterwards
            + np.where(downgrade_month >= 0, 0.45, 0)
            - np.where(has_csm, 0.30, 0.0)
            - 0.25 * np.log1p(months_since_signup / 12.0)  # tenure protects, mildly
        )
        hazard = _sigmoid(logit)
        churns = active & (rng.random(n) < hazard)
        for idx in np.flatnonzero(churns):
            churn_month[idx] = m
            events.append(
                {
                    "account_id": accounts["account_id"].iloc[idx],
                    "occurred_at": month,
                    "event_type": "churn",
                    "detail": {},
                }
            )

        health[:, m] = np.where(active, current, np.nan)

    # --- the leading indicator -----------------------------------------
    # An account on its way out disengages before it leaves. Without this the
    # problem is unlearnable from usage; with too much of it, it is trivial.
    for idx in np.flatnonzero(churn_month >= 0):
        end = churn_month[idx]
        for k in range(1, DECLINE_MONTHS + 1):
            m = end - k
            if m < 0 or np.isnan(health[idx, m]):
                continue
            # Ramped decay, strongest in the final month, and noisy enough
            # that plenty of declining accounts do not churn at all.
            decay = 1.0 - (0.30 / k) * rng.uniform(0.5, 1.5)
            health[idx, m] = float(np.clip(health[idx, m] * decay, 0.02, 1.0))

    return health, churn_month, events


def _materialise_usage(
    rng: np.random.Generator,
    accounts: pd.DataFrame,
    months: list[dt.date],
    health: np.ndarray,
    churn_month: np.ndarray,
    end_date: dt.date,
) -> pd.DataFrame:
    """Daily product usage, generated from the monthly health state.

    Daily grain because that is the grain a product database has, and because
    features like "days since last use" and "zero-usage days in the last 28"
    cannot be computed from monthly aggregates.
    """
    rows_account: list[str] = []
    rows_date: list[dt.date] = []
    active_users: list[int] = []
    sessions: list[int] = []
    api_calls: list[int] = []
    features_used: list[int] = []
    errors: list[int] = []

    seats = accounts["seats"].to_numpy()
    signup = accounts["signup_date"].to_numpy()
    account_ids = accounts["account_id"].to_numpy()

    for i in range(len(accounts)):
        start = max(signup[i], months[0])
        last = end_date if churn_month[i] < 0 else months[churn_month[i]]
        if last <= start:
            continue
        n_days = (last - start).days
        if n_days <= 0:
            continue

        days = np.array([start + dt.timedelta(days=d) for d in range(n_days)])
        month_index = np.array(
            [
                min(len(months) - 1, (d.year - months[0].year) * 12 + d.month - months[0].month)
                for d in days
            ]
        )
        h = health[i, month_index]
        h = np.where(np.isnan(h), 0.05, h)

        # Weekends are quiet in B2B software, which is why a 28-day window is
        # the right length: it contains four of them and the noise cancels.
        weekday = np.array([d.weekday() for d in days])
        weekend = (weekday >= 5).astype(float)
        factor = h * (1.0 - 0.72 * weekend)

        expected_users = np.maximum(0.0, seats[i] * factor * rng.normal(0.85, 0.12, n_days))
        users = rng.poisson(np.clip(expected_users, 0, seats[i] * 1.2)).astype(int)
        users = np.minimum(users, seats[i])

        sess = rng.poisson(np.maximum(0.0, users * 2.6 * np.clip(h, 0.05, 1.0)))
        api = rng.poisson(np.maximum(0.0, users * 48.0 * np.clip(h, 0.05, 1.0)))
        feats = np.minimum(14, rng.poisson(np.maximum(0.0, 3.0 + 9.0 * h)))
        errs = rng.poisson(np.maximum(0.0, 0.4 + 6.0 * (1.0 - h) * (users > 0)))

        rows_account.extend([account_ids[i]] * n_days)
        rows_date.extend(days.tolist())
        active_users.extend(users.tolist())
        sessions.extend(sess.tolist())
        api_calls.extend(api.tolist())
        features_used.extend(feats.tolist())
        errors.extend(errs.tolist())

    return pd.DataFrame(
        {
            "account_id": rows_account,
            "usage_date": rows_date,
            "active_users": active_users,
            "sessions": sessions,
            "api_calls": api_calls,
            "features_used": features_used,
            "error_events": errors,
        }
    )


def _support_billing_sentiment(
    rng: np.random.Generator,
    accounts: pd.DataFrame,
    months: list[dt.date],
    health: np.ndarray,
    churn_month: np.ndarray,
    end_date: dt.date,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Tickets, invoices, NPS responses and the remaining events.

    All three are driven by the same latent health, with different lags and
    different amounts of noise -- which is what gives the model several weak,
    partially redundant signals rather than one strong one.
    """
    tickets: list[dict[str, Any]] = []
    invoices: list[dict[str, Any]] = []
    nps: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    for i in range(len(accounts)):
        account_id = accounts["account_id"].iloc[i]
        seats = int(accounts["seats"].iloc[i])
        mrr = float(accounts["mrr_eur"].iloc[i])
        signup = accounts["signup_date"].iloc[i]
        term = int(accounts["contract_term_months"].iloc[i])
        has_csm = bool(accounts["has_csm"].iloc[i])
        last_month_index = churn_month[i] if churn_month[i] >= 0 else len(months) - 1

        events.append(
            {"account_id": account_id, "occurred_at": signup, "event_type": "signup", "detail": {}}
        )

        for m in range(len(months)):
            month = months[m]
            if month < signup or m > last_month_index:
                continue
            h = health[i, m]
            if np.isnan(h):
                continue

            # --- support -------------------------------------------------
            # Unhealthy accounts open more tickets, and more of them are P1.
            rate = 0.35 + 0.05 * np.log1p(seats) + 2.2 * (1.0 - h) ** 2
            for _ in range(int(rng.poisson(rate))):
                day = int(rng.integers(0, 28))
                opened = dt.datetime.combine(
                    month + dt.timedelta(days=day), dt.time(hour=int(rng.integers(7, 19)))
                ).replace(tzinfo=dt.UTC)
                priority = str(
                    rng.choice(
                        ["p1", "p2", "p3"], p=[0.10 + 0.25 * (1 - h), 0.35, 0.55 - 0.25 * (1 - h)]
                    )
                )
                # Resolution time is the feature that matters here, and it is
                # worse for the accounts that can least afford it.
                hours = float(rng.gamma(2.0, 6.0 + 22.0 * (1.0 - h)))
                unresolved = rng.random() < (0.05 + 0.22 * (1.0 - h))
                closed = None if unresolved else opened + dt.timedelta(hours=hours)
                satisfaction = None
                if closed is not None and rng.random() < 0.45:
                    satisfaction = int(np.clip(round(rng.normal(2.0 + 3.0 * h, 0.9)), 1, 5))
                tickets.append(
                    {
                        "ticket_id": f"TCK-{uuid.uuid4().hex[:12]}",
                        "account_id": account_id,
                        "opened_at": opened,
                        "closed_at": closed,
                        "priority": priority,
                        "category": str(rng.choice(TICKET_CATEGORIES)),
                        "satisfaction": satisfaction,
                    }
                )

            # --- billing -------------------------------------------------
            issued = month
            due = month + dt.timedelta(days=30)
            # Financial stress correlates with disengagement but is not the
            # same thing: plenty of healthy accounts pay late.
            late_p = 0.04 + 0.28 * (1.0 - h)
            failed_p = 0.01 + 0.10 * (1.0 - h)
            roll = rng.random()
            if roll < failed_p:
                status, paid = "failed", None
            elif roll < failed_p + late_p:
                status = "paid"
                paid = due + dt.timedelta(days=int(rng.integers(1, 45)))
            else:
                status = "paid"
                paid = due - dt.timedelta(days=int(rng.integers(0, 25)))
            invoices.append(
                {
                    "invoice_id": f"INV-{uuid.uuid4().hex[:12]}",
                    "account_id": account_id,
                    "issued_at": issued,
                    "due_at": due,
                    "paid_at": paid,
                    "amount_eur": round(mrr, 2),
                    "status": status,
                }
            )

            # --- sentiment -----------------------------------------------
            # Surveyed quarterly; answered by fewer than half, and the ones
            # who answer are not a random sample -- which is exactly why
            # `days_since_nps` is a feature in its own right.
            if m % 3 == 0 and rng.random() < 0.42:
                score = int(np.clip(round(rng.normal(3.0 + 6.5 * h, 1.6)), 0, 10))
                nps.append(
                    {
                        "response_id": f"NPS-{uuid.uuid4().hex[:12]}",
                        "account_id": account_id,
                        "responded_at": month + dt.timedelta(days=int(rng.integers(0, 28))),
                        "score": score,
                    }
                )

            # --- relationship --------------------------------------------
            if has_csm and m % 3 == 1 and rng.random() < 0.7:
                events.append(
                    {
                        "account_id": account_id,
                        "occurred_at": month,
                        "event_type": "qbr_held",
                        "detail": {},
                    }
                )

            months_since = (month.year - signup.year) * 12 + month.month - signup.month
            if term > 1 and months_since > 0 and months_since % term == 0:
                events.append(
                    {
                        "account_id": account_id,
                        "occurred_at": month,
                        "event_type": "renewal",
                        "detail": {"term_months": term},
                    }
                )

    return (
        pd.DataFrame(tickets),
        pd.DataFrame(invoices),
        pd.DataFrame(nps),
        events,
    )


def simulate(settings: Settings | None = None) -> SimulationResult:
    """Run the whole simulation. Deterministic for a given seed."""
    settings = settings or get_settings()
    rng = np.random.default_rng(settings.random_seed)

    end_date = dt.date(2026, 3, 1)
    months = _month_starts(
        dt.date(end_date.year, end_date.month, 1) - dt.timedelta(days=31 * settings.history_months),
        settings.history_months,
    )
    first_month, last_month = months[0], months[-1]

    logger.info(
        "simulating Vertex Systems",
        extra={
            "accounts": settings.n_accounts,
            "months": len(months),
            "seed": settings.random_seed,
        },
    )

    accounts = _build_accounts(rng, settings, first_month, last_month)
    health, churn_month, events = _simulate_health(rng, accounts, months)
    usage = _materialise_usage(rng, accounts, months, health, churn_month, end_date)
    tickets, invoices, nps, more_events = _support_billing_sentiment(
        rng, accounts, months, health, churn_month, end_date
    )

    accounts["churn_date"] = [
        months[m] + dt.timedelta(days=int(rng.integers(0, 28))) if m >= 0 else None
        for m in churn_month
    ]

    # A renewal date belongs to the contract, so it exists for every
    # committed-term account whatever happened to it afterwards. An earlier
    # version set it only for survivors, which made "no renewal date" a
    # perfect proxy for "this account churned" -- a textbook target leak, and
    # one the leakage screen in `features/leakage.py` now catches.
    def _next_anniversary(signup: dt.date, term_months: int, after: dt.date) -> dt.date | None:
        if term_months <= 1:
            return None
        elapsed = (after.year - signup.year) * 12 + after.month - signup.month
        periods = elapsed // term_months + 1
        total = periods * term_months
        year = signup.year + (signup.month - 1 + total) // 12
        month = (signup.month - 1 + total) % 12 + 1
        day = min(signup.day, 28)
        return dt.date(year, month, day)

    accounts["renewal_date"] = [
        _next_anniversary(
            accounts["signup_date"].iloc[i], int(accounts["contract_term_months"].iloc[i]), end_date
        )
        for i in range(len(accounts))
    ]

    all_events = pd.DataFrame(events + more_events)
    stats = {
        "accounts": len(accounts),
        "churned": int((churn_month >= 0).sum()),
        "churn_rate_pct": round(100.0 * float((churn_month >= 0).mean()), 2),
        "usage_rows": len(usage),
        "tickets": len(tickets),
        "invoices": len(invoices),
        "nps_responses": len(nps),
        "events": len(all_events),
        "months": len(months),
        "first_month": months[0].isoformat(),
        "last_month": months[-1].isoformat(),
    }
    logger.info("simulation complete", extra=stats)

    return SimulationResult(
        accounts=accounts,
        usage=usage,
        tickets=tickets,
        invoices=invoices,
        nps=nps,
        events=all_events,
        stats=stats,
    )
