-- ===========================================================================
-- Three schemas, three owners.
--
--   saas      the operational system, as the product team runs it. polaris
--             only ever reads from it.
--   features  what the model is allowed to see, computed as of a reference
--             date and never touched afterwards.
--   ml        the platform's own memory: runs, model versions, evaluations,
--             predictions, drift and outcomes.
--
-- The separation between `saas` and `features` is the one that prevents the
-- mistake this whole project is about: a feature is not a query against the
-- current state of the business, it is a query against the state of the
-- business at a point in the past.
-- ===========================================================================

CREATE SCHEMA IF NOT EXISTS ${SOURCE};
CREATE SCHEMA IF NOT EXISTS ${FEATURES};
CREATE SCHEMA IF NOT EXISTS ${ML};

COMMENT ON SCHEMA ${SOURCE}   IS 'Simulated Vertex Systems SaaS platform. Read-only. Synthetic data.';
COMMENT ON SCHEMA ${FEATURES} IS 'Point-in-time feature store: one row per (reference_date, account).';
COMMENT ON SCHEMA ${ML}       IS 'Runs, model versions, evaluations, predictions, drift, outcomes.';
