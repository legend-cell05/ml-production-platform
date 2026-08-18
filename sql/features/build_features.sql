-- ===========================================================================
-- Build one reference date's features.
--
-- THE RULE, and the only one that matters here: every row in this query is
-- filtered by a date **strictly before** :reference_date. Not `<=` -- a
-- prediction made on the morning of the 1st cannot see the 1st.
--
-- Three places where that rule is easy to break and expensive to break:
--
--   1. A ticket that was closed AFTER the reference date was still OPEN on
--      it. Joining on `closed_at IS NULL` uses today's state and leaks.
--      Everything below asks `closed_at < :reference_date` instead.
--   2. An invoice paid after the reference date was UNPAID on it, and its
--      days-overdue must be measured against the reference date, not against
--      now().
--   3. `account.churn_date` must never appear in a feature. It is used once,
--      at the very bottom, to build the label -- and only for the window that
--      starts after the reference date.
--
-- Parameters: :reference_date, :horizon_days, :feature_version, :max_known_date
-- ===========================================================================

WITH eligible AS (
    -- An account is scoreable on a date if it had signed up and had not yet
    -- churned. Using churn_date here is legitimate: it decides who exists,
    -- not what is known about them.
    SELECT
        a.account_id,
        a.segment,
        a.industry,
        a.country,
        a.plan,
        a.contract_term_months,
        a.has_csm,
        a.seats,
        a.mrr_eur,
        a.signup_date,
        a.renewal_date,
        a.churn_date
    FROM ${SOURCE}.account a
    WHERE a.signup_date < CAST(:reference_date AS DATE)
      AND (a.churn_date IS NULL OR a.churn_date >= CAST(:reference_date AS DATE))
),

-- --- Usage, two windows -------------------------------------------------
usage_28 AS (
    SELECT
        u.account_id,
        AVG(u.active_users)::NUMERIC(10, 3)                       AS active_users_28d_avg,
        SUM(u.sessions)                                           AS sessions_28d,
        SUM(u.api_calls)                                          AS api_calls_28d,
        AVG(u.features_used)::NUMERIC(10, 3)                      AS features_used_28d,
        COUNT(*) FILTER (WHERE u.active_users = 0)                AS zero_usage_days_28d,
        (SUM(u.error_events)::NUMERIC
            / NULLIF(SUM(u.sessions), 0))::NUMERIC(8, 5)          AS error_rate_28d,
        MAX(u.usage_date) FILTER (WHERE u.active_users > 0)       AS last_active_date
    FROM ${SOURCE}.usage_daily u
    WHERE u.usage_date < CAST(:reference_date AS DATE)
      AND u.usage_date >= CAST(:reference_date AS DATE) - 28
    GROUP BY u.account_id
),
usage_90 AS (
    SELECT
        u.account_id,
        AVG(u.active_users)::NUMERIC(10, 3) AS active_users_90d_avg
    FROM ${SOURCE}.usage_daily u
    WHERE u.usage_date < CAST(:reference_date AS DATE)
      AND u.usage_date >= CAST(:reference_date AS DATE) - 90
    GROUP BY u.account_id
),
-- The last day anybody logged in, over a long lookback, so "days since last
-- use" is right even for an account that has been silent for months.
last_use AS (
    SELECT u.account_id, MAX(u.usage_date) AS last_active_date
    FROM ${SOURCE}.usage_daily u
    WHERE u.usage_date < CAST(:reference_date AS DATE)
      AND u.active_users > 0
      AND u.usage_date >= CAST(:reference_date AS DATE) - 400
    GROUP BY u.account_id
),

-- --- Support ------------------------------------------------------------
tickets AS (
    SELECT
        t.account_id,
        COUNT(*) FILTER (
            WHERE t.opened_at >= CAST(:reference_date AS DATE) - 90
        )                                                         AS tickets_90d,
        COUNT(*) FILTER (
            WHERE t.priority = 'p1'
              AND t.opened_at >= CAST(:reference_date AS DATE) - 90
        )                                                         AS tickets_p1_90d,
        -- Open tickets are counted by the CTE below, not here: a ticket open
        -- for five hundred days is exactly the kind of thing that predicts
        -- churn, and the 400-day lookback this CTE uses for its windowed
        -- aggregates would silently drop it.
        AVG(
            EXTRACT(EPOCH FROM (t.closed_at - t.opened_at)) / 3600.0
        ) FILTER (
            WHERE t.closed_at < CAST(:reference_date AS DATE)
              AND t.opened_at >= CAST(:reference_date AS DATE) - 90
        )::NUMERIC(10, 2)                                         AS avg_resolution_hours_90d,
        AVG(t.satisfaction) FILTER (
            WHERE t.satisfaction IS NOT NULL
              AND t.closed_at < CAST(:reference_date AS DATE)
              AND t.opened_at >= CAST(:reference_date AS DATE) - 180
        )::NUMERIC(4, 2)                                          AS avg_satisfaction_180d
    FROM ${SOURCE}.support_ticket t
    WHERE t.opened_at < CAST(:reference_date AS DATE)
      AND t.opened_at >= CAST(:reference_date AS DATE) - 400
    GROUP BY t.account_id
),
-- Tickets still open as of the reference date, over ALL history rather than
-- the 400-day window above. The distinction cost a failing test: an account
-- with a ticket open since 2023 is not an account with no open tickets.
open_tickets AS (
    SELECT
        t.account_id,
        COUNT(*) AS open_tickets
    FROM ${SOURCE}.support_ticket t
    WHERE t.opened_at < CAST(:reference_date AS DATE)
      AND (t.closed_at IS NULL OR t.closed_at >= CAST(:reference_date AS DATE))
    GROUP BY t.account_id
),

-- --- Billing ------------------------------------------------------------
billing AS (
    SELECT
        i.account_id,
        COUNT(*) FILTER (
            WHERE i.issued_at >= CAST(:reference_date AS DATE) - 180
              AND (
                    -- Paid, but paid late
                    (i.paid_at IS NOT NULL
                     AND i.paid_at < CAST(:reference_date AS DATE)
                     AND i.paid_at > i.due_at)
                    -- Or still unpaid as of the reference date, past due
                 OR ((i.paid_at IS NULL OR i.paid_at >= CAST(:reference_date AS DATE))
                     AND i.due_at < CAST(:reference_date AS DATE))
              )
        )                                                         AS late_payments_180d,
        COUNT(*) FILTER (
            WHERE i.status = 'failed'
              AND i.issued_at >= CAST(:reference_date AS DATE) - 90
        )                                                         AS failed_payments_90d,
        COALESCE(MAX(
            CASE
                WHEN i.issued_at >= CAST(:reference_date AS DATE) - 180
                 AND (i.paid_at IS NULL OR i.paid_at >= CAST(:reference_date AS DATE))
                 AND i.due_at < CAST(:reference_date AS DATE)
                THEN CAST(:reference_date AS DATE) - i.due_at
                WHEN i.issued_at >= CAST(:reference_date AS DATE) - 180
                 AND i.paid_at < CAST(:reference_date AS DATE)
                 AND i.paid_at > i.due_at
                THEN i.paid_at - i.due_at
                ELSE 0
            END
        ), 0)                                                     AS max_days_overdue_180d
    FROM ${SOURCE}.invoice i
    WHERE i.issued_at < CAST(:reference_date AS DATE)
      AND i.issued_at >= CAST(:reference_date AS DATE) - 400
    GROUP BY i.account_id
),

-- --- Sentiment ----------------------------------------------------------
nps AS (
    SELECT DISTINCT ON (n.account_id)
        n.account_id,
        n.score                                                   AS last_nps_score,
        CAST(:reference_date AS DATE) - n.responded_at            AS days_since_nps
    FROM ${SOURCE}.nps_response n
    WHERE n.responded_at < CAST(:reference_date AS DATE)
    ORDER BY n.account_id, n.responded_at DESC
),

-- --- Relationship and commercial events ---------------------------------
events AS (
    SELECT
        e.account_id,
        BOOL_OR(e.event_type = 'champion_left'
                AND e.occurred_at >= CAST(:reference_date AS DATE) - 180)  AS champion_left_180d,
        BOOL_OR(e.event_type = 'qbr_held'
                AND e.occurred_at >= CAST(:reference_date AS DATE) - 180)  AS qbr_held_180d,
        BOOL_OR(e.event_type = 'plan_downgrade'
                AND e.occurred_at >= CAST(:reference_date AS DATE) - 180)  AS had_downgrade_180d,
        BOOL_OR(e.event_type = 'plan_upgrade'
                AND e.occurred_at >= CAST(:reference_date AS DATE) - 180)  AS had_upgrade_180d
    FROM ${SOURCE}.account_event e
    WHERE e.occurred_at < CAST(:reference_date AS DATE)
      AND e.occurred_at >= CAST(:reference_date AS DATE) - 400
    GROUP BY e.account_id
)

INSERT INTO ${FEATURES}.churn_features (
    reference_date, account_id,
    segment, industry, country, plan, contract_term_months, has_csm,
    tenure_days, days_to_renewal,
    mrr_eur, seats, mrr_change_90d_pct, seats_change_90d,
    had_downgrade_180d, had_upgrade_180d,
    active_users_28d_avg, seat_utilisation_28d, sessions_28d, api_calls_28d,
    features_used_28d, usage_trend_28d_vs_90d, days_since_last_use,
    zero_usage_days_28d, error_rate_28d,
    tickets_90d, tickets_p1_90d, open_tickets, avg_resolution_hours_90d,
    avg_satisfaction_180d,
    late_payments_180d, failed_payments_90d, max_days_overdue_180d,
    last_nps_score, days_since_nps, champion_left_180d, qbr_held_180d,
    churned_in_horizon, label_known_at, feature_version
)
SELECT
    CAST(:reference_date AS DATE),
    e.account_id,
    e.segment, e.industry, e.country, e.plan, e.contract_term_months, e.has_csm,
    (CAST(:reference_date AS DATE) - e.signup_date)               AS tenure_days,
    -- Days to the NEXT contract anniversary after the reference date,
    -- derived from the signup date and the term. Deliberately not read from
    -- account.renewal_date: that column describes the contract as it stands
    -- today, and reading it would be a statement about the future.
    CASE
        WHEN e.contract_term_months > 1 THEN
            (e.signup_date + (
                ((((EXTRACT(YEAR FROM AGE(CAST(:reference_date AS DATE), e.signup_date)) * 12
                    + EXTRACT(MONTH FROM AGE(CAST(:reference_date AS DATE), e.signup_date)))::INT
                   / e.contract_term_months) + 1) * e.contract_term_months
                ) || ' months')::INTERVAL)::DATE
            - CAST(:reference_date AS DATE)
    END                                                           AS days_to_renewal,

    e.mrr_eur,
    e.seats,
    -- The simulated billing system does not version price changes, so these
    -- two are NULL rather than invented. A feature that cannot be computed
    -- honestly is left empty and documented, not approximated.
    NULL::NUMERIC(8, 4)                                           AS mrr_change_90d_pct,
    NULL::INT                                                     AS seats_change_90d,
    COALESCE(ev.had_downgrade_180d, FALSE),
    COALESCE(ev.had_upgrade_180d, FALSE),

    u28.active_users_28d_avg,
    (u28.active_users_28d_avg / NULLIF(e.seats, 0))::NUMERIC(8, 4) AS seat_utilisation_28d,
    u28.sessions_28d,
    u28.api_calls_28d,
    u28.features_used_28d,
    -- Below 1 means the last four weeks are quieter than the last three
    -- months. This ratio is the leading indicator the whole model rests on.
    (u28.active_users_28d_avg
        / NULLIF(u90.active_users_90d_avg, 0))::NUMERIC(8, 4)     AS usage_trend_28d_vs_90d,
    COALESCE(CAST(:reference_date AS DATE) - lu.last_active_date, 400) AS days_since_last_use,
    COALESCE(u28.zero_usage_days_28d, 28)                         AS zero_usage_days_28d,
    u28.error_rate_28d,

    COALESCE(t.tickets_90d, 0),
    COALESCE(t.tickets_p1_90d, 0),
    COALESCE(ot.open_tickets, 0),
    t.avg_resolution_hours_90d,
    t.avg_satisfaction_180d,

    COALESCE(b.late_payments_180d, 0),
    COALESCE(b.failed_payments_90d, 0),
    COALESCE(b.max_days_overdue_180d, 0),

    n.last_nps_score,
    n.days_since_nps,
    COALESCE(ev.champion_left_180d, FALSE),
    COALESCE(ev.qbr_held_180d, FALSE),

    -- --- The label ------------------------------------------------------
    -- Churn strictly after the reference date and within the horizon. It is
    -- NULL when the horizon has not elapsed in the data, because at that
    -- point the answer is genuinely unknown -- and a NULL is the only honest
    -- way to say so. Filling it with FALSE would teach the model that the
    -- most recent accounts never churn.
    CASE
        WHEN CAST(:reference_date AS DATE) + CAST(:horizon_days AS INT)
             > CAST(:max_known_date AS DATE)
        THEN NULL
        ELSE (
            e.churn_date IS NOT NULL
            AND e.churn_date > CAST(:reference_date AS DATE)
            AND e.churn_date <= CAST(:reference_date AS DATE) + CAST(:horizon_days AS INT)
        )
    END                                                           AS churned_in_horizon,
    CASE
        WHEN CAST(:reference_date AS DATE) + CAST(:horizon_days AS INT)
             > CAST(:max_known_date AS DATE)
        THEN NULL
        ELSE CAST(:reference_date AS DATE) + CAST(:horizon_days AS INT)
    END                                                           AS label_known_at,
    CAST(:feature_version AS TEXT)
FROM eligible e
LEFT JOIN usage_28 u28 ON u28.account_id = e.account_id
LEFT JOIN usage_90 u90 ON u90.account_id = e.account_id
LEFT JOIN last_use lu  ON lu.account_id  = e.account_id
LEFT JOIN tickets t    ON t.account_id   = e.account_id
LEFT JOIN open_tickets ot ON ot.account_id = e.account_id
LEFT JOIN billing b    ON b.account_id   = e.account_id
LEFT JOIN nps n        ON n.account_id   = e.account_id
LEFT JOIN events ev    ON ev.account_id  = e.account_id
ON CONFLICT (reference_date, account_id) DO NOTHING;
