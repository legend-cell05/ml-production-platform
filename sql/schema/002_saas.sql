-- ===========================================================================
-- The operational system: Vertex Systems, a B2B SaaS platform.
--
-- Modelled on what a churn project actually has to work with -- a product
-- database that was designed for running the product, not for predicting
-- anything. In particular: usage lives at daily grain in a table that is two
-- orders of magnitude larger than every other one, and the fact that an
-- account churned is a column on the account rather than an event, which is
-- exactly the shape that tempts people into leakage.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS ${SOURCE}.account (
    account_id          TEXT PRIMARY KEY,
    company_name        TEXT NOT NULL,
    segment             TEXT NOT NULL,      -- smb | mid_market | enterprise
    industry            TEXT NOT NULL,
    country             TEXT NOT NULL,
    signup_date         DATE NOT NULL,
    plan                TEXT NOT NULL,      -- starter | business | premium
    seats               INT  NOT NULL,
    mrr_eur             NUMERIC(12, 2) NOT NULL,
    contract_term_months INT NOT NULL,
    renewal_date        DATE,
    has_csm             BOOLEAN NOT NULL DEFAULT FALSE,
    -- The churn date is the only thing in this schema that a feature must
    -- never read directly. It is here because it is here in every real CRM.
    churn_date          DATE,
    CONSTRAINT ck_account_segment CHECK (segment IN ('smb', 'mid_market', 'enterprise')),
    CONSTRAINT ck_account_seats   CHECK (seats > 0),
    CONSTRAINT ck_account_mrr     CHECK (mrr_eur >= 0)
);

CREATE INDEX IF NOT EXISTS ix_account_segment ON ${SOURCE}.account (segment);
CREATE INDEX IF NOT EXISTS ix_account_churn   ON ${SOURCE}.account (churn_date)
    WHERE churn_date IS NOT NULL;

-- Daily product usage. The big table.
CREATE TABLE IF NOT EXISTS ${SOURCE}.usage_daily (
    account_id          TEXT NOT NULL REFERENCES ${SOURCE}.account (account_id),
    usage_date          DATE NOT NULL,
    active_users        INT  NOT NULL DEFAULT 0,
    sessions            INT  NOT NULL DEFAULT 0,
    api_calls           INT  NOT NULL DEFAULT 0,
    features_used       INT  NOT NULL DEFAULT 0,
    error_events        INT  NOT NULL DEFAULT 0,
    PRIMARY KEY (account_id, usage_date)
);

-- BRIN rather than B-tree: the table is written in date order and scanned by
-- date range, which is the case BRIN exists for. A B-tree here would be
-- roughly forty times larger for the same pruning.
CREATE INDEX IF NOT EXISTS ix_usage_date ON ${SOURCE}.usage_daily USING BRIN (usage_date);

CREATE TABLE IF NOT EXISTS ${SOURCE}.support_ticket (
    ticket_id           TEXT PRIMARY KEY,
    account_id          TEXT NOT NULL REFERENCES ${SOURCE}.account (account_id),
    opened_at           TIMESTAMPTZ NOT NULL,
    closed_at           TIMESTAMPTZ,
    priority            TEXT NOT NULL,      -- p1 | p2 | p3
    category            TEXT NOT NULL,
    satisfaction        SMALLINT,           -- 1-5, only when the customer answered
    CONSTRAINT ck_ticket_priority CHECK (priority IN ('p1', 'p2', 'p3')),
    CONSTRAINT ck_ticket_csat     CHECK (satisfaction IS NULL OR satisfaction BETWEEN 1 AND 5)
);

CREATE INDEX IF NOT EXISTS ix_ticket_account ON ${SOURCE}.support_ticket (account_id, opened_at);

CREATE TABLE IF NOT EXISTS ${SOURCE}.invoice (
    invoice_id          TEXT PRIMARY KEY,
    account_id          TEXT NOT NULL REFERENCES ${SOURCE}.account (account_id),
    issued_at           DATE NOT NULL,
    due_at              DATE NOT NULL,
    paid_at             DATE,
    amount_eur          NUMERIC(12, 2) NOT NULL,
    status              TEXT NOT NULL,      -- paid | open | failed | written_off
    CONSTRAINT ck_invoice_status CHECK (status IN ('paid', 'open', 'failed', 'written_off'))
);

CREATE INDEX IF NOT EXISTS ix_invoice_account ON ${SOURCE}.invoice (account_id, issued_at);

CREATE TABLE IF NOT EXISTS ${SOURCE}.nps_response (
    response_id         TEXT PRIMARY KEY,
    account_id          TEXT NOT NULL REFERENCES ${SOURCE}.account (account_id),
    responded_at        DATE NOT NULL,
    score               SMALLINT NOT NULL,
    CONSTRAINT ck_nps_score CHECK (score BETWEEN 0 AND 10)
);

CREATE INDEX IF NOT EXISTS ix_nps_account ON ${SOURCE}.nps_response (account_id, responded_at);

-- Commercial and relationship events. Includes the churn event itself, which
-- is what the label is built from.
CREATE TABLE IF NOT EXISTS ${SOURCE}.account_event (
    event_id            TEXT PRIMARY KEY,
    account_id          TEXT NOT NULL REFERENCES ${SOURCE}.account (account_id),
    occurred_at         DATE NOT NULL,
    event_type          TEXT NOT NULL,
    detail              JSONB NOT NULL DEFAULT '{}'::JSONB,
    CONSTRAINT ck_event_type CHECK (event_type IN (
        'signup', 'plan_upgrade', 'plan_downgrade', 'seat_increase', 'seat_decrease',
        'champion_left', 'renewal', 'qbr_held', 'churn'
    ))
);

CREATE INDEX IF NOT EXISTS ix_event_account ON ${SOURCE}.account_event (account_id, occurred_at);
CREATE INDEX IF NOT EXISTS ix_event_type    ON ${SOURCE}.account_event (event_type, occurred_at);
