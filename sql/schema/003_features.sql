-- ===========================================================================
-- The feature store.
--
-- One row per (reference_date, account_id), holding what was knowable about
-- that account on that date and nothing else. The label is stored alongside,
-- with the date it became knowable -- which is the horizon after the
-- reference date, and the reason the most recent rows have no label yet.
--
-- Why a wide typed table rather than JSONB or a generic key/value store: the
-- column list IS the feature contract. Adding a feature is a migration, which
-- is friction, and that friction is worth having -- a feature nobody can name
-- is a feature nobody can explain to a customer success manager who asks why
-- their account was flagged.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS ${FEATURES}.churn_features (
    reference_date          DATE NOT NULL,
    account_id              TEXT NOT NULL,

    -- Identity and contract (slow-moving)
    segment                 TEXT NOT NULL,
    industry                TEXT NOT NULL,
    country                 TEXT NOT NULL,
    plan                    TEXT NOT NULL,
    contract_term_months    INT  NOT NULL,
    has_csm                 BOOLEAN NOT NULL,
    tenure_days             INT  NOT NULL,
    days_to_renewal         INT,

    -- Commercial
    mrr_eur                 NUMERIC(12, 2) NOT NULL,
    seats                   INT NOT NULL,
    mrr_change_90d_pct      NUMERIC(8, 4),
    seats_change_90d        INT,
    had_downgrade_180d      BOOLEAN NOT NULL DEFAULT FALSE,
    had_upgrade_180d        BOOLEAN NOT NULL DEFAULT FALSE,

    -- Usage, over two windows so the trend between them is a feature
    active_users_28d_avg    NUMERIC(10, 3),
    seat_utilisation_28d    NUMERIC(8, 4),
    sessions_28d            INT,
    api_calls_28d           INT,
    features_used_28d       NUMERIC(10, 3),
    usage_trend_28d_vs_90d  NUMERIC(8, 4),
    days_since_last_use     INT,
    zero_usage_days_28d     INT,
    error_rate_28d          NUMERIC(8, 5),

    -- Support
    tickets_90d             INT NOT NULL DEFAULT 0,
    tickets_p1_90d          INT NOT NULL DEFAULT 0,
    open_tickets            INT NOT NULL DEFAULT 0,
    avg_resolution_hours_90d NUMERIC(10, 2),
    avg_satisfaction_180d   NUMERIC(4, 2),

    -- Billing
    late_payments_180d      INT NOT NULL DEFAULT 0,
    failed_payments_90d     INT NOT NULL DEFAULT 0,
    max_days_overdue_180d   INT NOT NULL DEFAULT 0,

    -- Sentiment and relationship
    last_nps_score          SMALLINT,
    days_since_nps          INT,
    champion_left_180d      BOOLEAN NOT NULL DEFAULT FALSE,
    qbr_held_180d           BOOLEAN NOT NULL DEFAULT FALSE,

    -- Label. NULL until the horizon has elapsed.
    churned_in_horizon      BOOLEAN,
    label_known_at          DATE,

    -- Provenance: which feature definition produced this row.
    feature_version         TEXT NOT NULL,
    computed_at             TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (reference_date, account_id),
    CONSTRAINT ck_features_tenure CHECK (tenure_days >= 0),
    CONSTRAINT ck_features_util   CHECK (seat_utilisation_28d IS NULL OR seat_utilisation_28d >= 0),
    -- A labelled row must say when the label became knowable, and an
    -- unlabelled one must not pretend to.
    CONSTRAINT ck_features_label  CHECK (
        (churned_in_horizon IS NULL AND label_known_at IS NULL)
        OR (churned_in_horizon IS NOT NULL AND label_known_at IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS ix_features_reference ON ${FEATURES}.churn_features (reference_date);
CREATE INDEX IF NOT EXISTS ix_features_labelled  ON ${FEATURES}.churn_features (reference_date)
    WHERE churned_in_horizon IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_features_segment   ON ${FEATURES}.churn_features (segment, reference_date);
