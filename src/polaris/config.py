"""Configuration.

Every setting is an environment variable prefixed ``POLARIS_``, read from
``.env`` when present. Three groups of them are worth reading even if you skip
the rest, because they encode decisions rather than plumbing:

* **the horizon**, which defines the label -- change it and every model
  trained before is answering a different question;
* **the economics**, which decide the threshold -- a model without them can
  only be scored on abstractions like F1;
* **the gates**, which decide whether a challenger ships.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


def _detect_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


class Settings(BaseSettings):
    """Runtime configuration."""

    model_config = SettingsConfigDict(
        env_prefix="POLARIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),  # `model_*` is a legitimate prefix in this domain
    )

    # --- Database ----------------------------------------------------------
    db_host: str = "localhost"
    db_port: int = Field(default=5432, ge=1, le=65535)
    db_name: str = "polaris"
    db_user: str = "polaris_app"
    db_password: SecretStr = SecretStr("change_me_local_only")

    source_schema: str = "saas"
    feature_schema: str = "features"
    ml_schema: str = "ml"

    # --- Experiment tracking -----------------------------------------------
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "churn-60d"

    # --- The prediction problem --------------------------------------------
    horizon_days: int = Field(default=60, ge=7, le=365)
    reference_interval_days: int = Field(default=14, ge=1, le=90)

    # --- Economics ---------------------------------------------------------
    value_of_saved_account: float = Field(default=9000.0, gt=0)
    cost_of_intervention: float = Field(default=350.0, gt=0)
    intervention_success_rate: float = Field(default=0.30, gt=0, le=1.0)

    # --- Promotion gates ---------------------------------------------------
    min_pr_auc_gain: float = Field(default=0.005, ge=0.0, le=1.0)
    max_segment_regression: float = Field(default=0.03, ge=0.0, le=1.0)
    max_brier_score: float = Field(default=0.12, gt=0.0, le=1.0)
    max_p95_latency_ms: float = Field(default=50.0, gt=0)

    # --- Drift -------------------------------------------------------------
    psi_warn: float = Field(default=0.10, gt=0)
    psi_alert: float = Field(default=0.25, gt=0)

    # --- Generation --------------------------------------------------------
    random_seed: int = 20260301
    n_accounts: int = Field(default=3000, ge=50, le=200_000)
    history_months: int = Field(default=24, ge=6, le=120)

    # --- Paths -------------------------------------------------------------
    project_root: Path = Field(default_factory=_detect_project_root)
    data_dir: Path = Path("data")

    # --- Runtime -----------------------------------------------------------
    log_level: str = "INFO"
    log_format: str = "text"

    # --- Validation --------------------------------------------------------

    @field_validator("source_schema", "feature_schema", "ml_schema")
    @classmethod
    def _valid_schema_name(cls, value: str) -> str:
        """Schema names reach DDL, where they cannot be bound parameters."""
        if not _IDENTIFIER.match(value):
            raise ValueError(f"schema name {value!r} must match [a-z_][a-z0-9_]*")
        return value

    @field_validator("log_level")
    @classmethod
    def _known_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log level must be one of {sorted(allowed)}")
        return upper

    @model_validator(mode="after")
    def _coherent(self) -> Settings:
        if self.psi_alert <= self.psi_warn:
            raise ValueError("psi_alert must be above psi_warn")
        if not self.data_dir.is_absolute():
            object.__setattr__(self, "data_dir", self.project_root / self.data_dir)
        return self

    # --- Derived -----------------------------------------------------------

    @property
    def dsn(self) -> str:
        """SQLAlchemy URL including the password -- never log this."""
        pwd = self.db_password.get_secret_value()
        return f"postgresql+psycopg://{self.db_user}:{pwd}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def safe_dsn(self) -> str:
        """Connection string with the password masked -- safe to log."""
        return (
            f"postgresql+psycopg://{self.db_user}:***@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def artifact_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def report_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def expected_value_of_action(self) -> float:
        """What acting on a true positive is worth, net of the intervention.

        value x success rate - cost. This single number is what turns a
        probability into a decision, and it is why the threshold is not 0.5.
        """
        return (
            self.value_of_saved_account * self.intervention_success_rate - self.cost_of_intervention
        )

    @property
    def mlflow_uri(self) -> str:
        """Resolve a relative local store against the project root.

        Otherwise the tracking store depends on the working directory, and a
        run launched from `notebooks/` lands somewhere else than one launched
        by the CLI -- which is how two people end up comparing experiments
        that are not in the same place.

        The default is SQLite rather than the old ``file:./mlruns`` store:
        MLflow 3 put the filesystem backend into maintenance mode and refuses
        to open one without an opt-out environment variable. A local SQLite
        file is a better default anyway -- it supports the full API, and
        moving to a tracking server later is a change to this one string.
        """
        uri = self.mlflow_tracking_uri
        for scheme in ("sqlite:///", "file:"):
            if uri.startswith(scheme):
                relative = uri[len(scheme) :]
                path = Path(relative)
                if not path.is_absolute():
                    path = self.project_root / path
                return f"{scheme}{path}"
        return uri


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read once."""
    return Settings()
