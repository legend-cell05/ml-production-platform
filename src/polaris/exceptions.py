"""Exception taxonomy.

The distinction that matters here is between *a run that cannot proceed* and
*a model that must not ship*. The second is not an error in the usual sense --
nothing crashed, the pipeline worked, and the answer is no. Conflating the two
produces a promotion gate that people learn to ignore because it looks like a
flaky build.
"""

from __future__ import annotations


class PolarisError(Exception):
    """Base class for every error this package raises deliberately."""


class ConfigurationError(PolarisError):
    """Settings are missing or contradictory."""


class DatabaseError(PolarisError):
    """The database refused an operation."""


class DataError(PolarisError):
    """The data cannot support what was asked of it."""


class DataLeakage(DataError):
    """A feature used information that did not exist at prediction time.

    Raised by the point-in-time checks rather than discovered six weeks after
    a model with an implausible AUC reached production.
    """

    def __init__(self, message: str, *, feature: str | None = None) -> None:
        super().__init__(message if feature is None else f"{feature}: {message}")
        self.feature = feature


class TrainingError(PolarisError):
    """Training could not complete."""


class ModelNotFound(PolarisError):
    """No model matches the requested name, version or stage."""


class PromotionBlocked(PolarisError):
    """A gate refused to promote a challenger.

    Carries the failed checks so the message can name them, rather than
    reporting only that the answer was no.
    """

    def __init__(self, message: str, *, failed: list[str] | None = None) -> None:
        super().__init__(message)
        self.failed = failed or []


class ServingError(PolarisError):
    """The serving layer cannot answer."""
