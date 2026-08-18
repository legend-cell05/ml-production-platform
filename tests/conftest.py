"""Shared fixtures.

Integration tests use a real PostgreSQL and a real (small) simulated business,
because the things worth testing here are the point-in-time SQL, the temporal
splits and the promotion gate -- all of which a mock would simply agree with.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from polaris.config import Settings
from polaris.db.engine import check_connection, dispose_engines
from polaris.db.schema import drop_schemas, initialise_database

REPO_ROOT = Path(__file__).resolve().parents[1]

requires_database = pytest.mark.skipif(
    os.environ.get("POLARIS_RUN_INTEGRATION") != "1",
    reason="set POLARIS_RUN_INTEGRATION=1 and provide a PostgreSQL to run these",
)


@pytest.fixture(scope="session")
def integration_settings(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Settings]:
    """A small, isolated platform: its own schemas, data directory and tracking store."""
    tmp = tmp_path_factory.mktemp("polaris")
    settings = Settings(
        source_schema="test_saas",
        feature_schema="test_features",
        ml_schema="test_ml",
        data_dir=tmp,
        mlflow_tracking_uri=f"sqlite:///{tmp / 'mlflow.db'}",
        mlflow_experiment="test-churn",
        n_accounts=400,
        history_months=24,
        reference_interval_days=14,
        log_level="WARNING",
    )
    if not check_connection(settings, retries=2):
        pytest.skip("no PostgreSQL available")
    drop_schemas(settings)
    initialise_database(settings)
    yield settings
    drop_schemas(settings)
    dispose_engines()


@pytest.fixture
def labelled_frame() -> pd.DataFrame:
    """A small hand-built frame with a known signal and no leak."""
    rng = np.random.default_rng(11)
    n = 1200
    label = (rng.random(n) < 0.08).astype(int)
    return pd.DataFrame(
        {
            "churned_in_horizon": label.astype(bool),
            # Genuinely predictive, at a believable strength.
            "seat_utilisation_28d": np.clip(rng.normal(0.6 - 0.25 * label, 0.18), 0, 2),
            # Pure noise.
            "tickets_90d": rng.poisson(2.0, n),
            # Missing at random, so its absence says nothing.
            "last_nps_score": np.where(rng.random(n) < 0.4, np.nan, rng.integers(0, 11, n)),
            "segment": rng.choice(["smb", "mid_market", "enterprise"], n),
        }
    )
