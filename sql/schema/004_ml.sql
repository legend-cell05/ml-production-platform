-- ===========================================================================
-- The platform's own memory.
--
-- Every table here answers a question that a model artefact on a disk cannot:
-- what was trained, on what, how well did it do, which version is serving,
-- what did it predict, and has the world moved since.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS ${ML}.training_run (
    run_id              UUID PRIMARY KEY,
    mlflow_run_id       TEXT,
    model_name          TEXT NOT NULL,
    algorithm           TEXT NOT NULL,
    feature_version     TEXT NOT NULL,
    dataset_fingerprint CHAR(64) NOT NULL,
    params              JSONB NOT NULL DEFAULT '{}'::JSONB,
    metrics             JSONB NOT NULL DEFAULT '{}'::JSONB,
    train_rows          INT NOT NULL,
    train_period        DATERANGE,
    test_period         DATERANGE,
    status              TEXT NOT NULL DEFAULT 'RUNNING',
    started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at         TIMESTAMPTZ,
    CONSTRAINT ck_run_status CHECK (status IN ('RUNNING', 'SUCCESS', 'FAILED'))
);

CREATE INDEX IF NOT EXISTS ix_run_started ON ${ML}.training_run (started_at DESC);

-- ---------------------------------------------------------------------------
-- The registry.
--
-- `stage` is the only thing the serving layer reads, and only one version per
-- model may be `production` -- enforced by a partial unique index rather than
-- by a convention everybody has to remember.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ${ML}.model_version (
    model_name          TEXT NOT NULL,
    version             INT  NOT NULL,
    run_id              UUID NOT NULL REFERENCES ${ML}.training_run (run_id),
    stage               TEXT NOT NULL DEFAULT 'candidate',
    artifact_path       TEXT NOT NULL,
    decision_threshold  NUMERIC(6, 5) NOT NULL,
    metrics             JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    promoted_at         TIMESTAMPTZ,
    archived_at         TIMESTAMPTZ,
    notes               TEXT,
    PRIMARY KEY (model_name, version),
    CONSTRAINT ck_model_stage CHECK (stage IN ('candidate', 'production', 'archived', 'rejected')),
    CONSTRAINT ck_model_threshold CHECK (decision_threshold > 0 AND decision_threshold < 1)
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_model_one_production
    ON ${ML}.model_version (model_name) WHERE stage = 'production';

-- ---------------------------------------------------------------------------
-- Evaluations, stored per segment as well as overall.
--
-- Storing only the headline number is what allows a model that is better on
-- average and worse for enterprise accounts to be promoted. The segment rows
-- are what the promotion gate reads.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ${ML}.evaluation (
    model_name          TEXT NOT NULL,
    version             INT  NOT NULL,
    split               TEXT NOT NULL,      -- validation | test
    segment             TEXT NOT NULL,      -- 'overall' or a segment value
    metric              TEXT NOT NULL,
    value               DOUBLE PRECISION NOT NULL,
    n_rows              INT NOT NULL,
    -- How many churns the slice actually contained. The promotion gate needs
    -- this to tell "this segment got worse" from "this segment had four
    -- positive examples and the number is noise".
    n_positives         INT NOT NULL DEFAULT 0,
    PRIMARY KEY (model_name, version, split, segment, metric),
    FOREIGN KEY (model_name, version) REFERENCES ${ML}.model_version (model_name, version)
        ON DELETE CASCADE
);

-- ---------------------------------------------------------------------------
-- Everything the serving layer answered.
--
-- Kept because a prediction without a record of it cannot be monitored, cannot
-- be explained back to the person who acted on it, and cannot be joined to
-- the outcome when the label finally arrives sixty days later.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ${ML}.prediction (
    prediction_id       BIGSERIAL PRIMARY KEY,
    model_name          TEXT NOT NULL,
    version             INT  NOT NULL,
    account_id          TEXT NOT NULL,
    reference_date      DATE NOT NULL,
    scored_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    probability         DOUBLE PRECISION NOT NULL,
    decision            BOOLEAN NOT NULL,
    threshold           NUMERIC(6, 5) NOT NULL,
    expected_value_eur  NUMERIC(12, 2),
    top_features        JSONB NOT NULL DEFAULT '[]'::JSONB,
    latency_ms          NUMERIC(8, 2),
    CONSTRAINT ck_prediction_prob CHECK (probability >= 0 AND probability <= 1)
);

CREATE INDEX IF NOT EXISTS ix_prediction_account ON ${ML}.prediction (account_id, scored_at DESC);
CREATE INDEX IF NOT EXISTS ix_prediction_model   ON ${ML}.prediction (model_name, version, scored_at);
CREATE INDEX IF NOT EXISTS ix_prediction_date    ON ${ML}.prediction USING BRIN (scored_at);

-- ---------------------------------------------------------------------------
-- Drift.
--
-- One row per feature per check, so a report can say which feature moved
-- rather than that "the data drifted".
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ${ML}.drift_check (
    check_id            BIGSERIAL PRIMARY KEY,
    checked_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_name          TEXT NOT NULL,
    version             INT  NOT NULL,
    feature             TEXT NOT NULL,
    psi                 DOUBLE PRECISION NOT NULL,
    status              TEXT NOT NULL,
    baseline_start      DATE NOT NULL,
    baseline_end        DATE NOT NULL,
    current_start       DATE NOT NULL,
    current_end         DATE NOT NULL,
    CONSTRAINT ck_drift_status CHECK (status IN ('OK', 'WARN', 'ALERT'))
);

CREATE INDEX IF NOT EXISTS ix_drift_recent ON ${ML}.drift_check (model_name, checked_at DESC);

-- ---------------------------------------------------------------------------
-- Outcomes: the label, as and when it arrives.
--
-- Separate from the prediction because they are separated by the horizon.
-- Joining them is how live performance is measured, and the gap between the
-- two dates is why that measurement is always sixty days behind.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ${ML}.outcome (
    account_id          TEXT NOT NULL,
    reference_date      DATE NOT NULL,
    churned             BOOLEAN NOT NULL,
    observed_at         DATE NOT NULL,
    PRIMARY KEY (account_id, reference_date)
);
