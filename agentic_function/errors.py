"""Unified exception hierarchy for agentic-function.

All custom exceptions derive from ``AgenticFunctionError`` so callers can catch
one base type when they don't care about specifics.
"""
from __future__ import annotations

from typing import Any  # used by RetryExhaustedError.metrics / attempt_errors


class AgenticFunctionError(Exception):
    """Base exception for every error raised by this library."""


# ---------------------------------------------------------------------------
# Configuration / registration errors
# ---------------------------------------------------------------------------
class ConfigError(AgenticFunctionError):
    """Raised when a configuration value is invalid or missing."""


class SchemaError(AgenticFunctionError):
    """Raised when an ``output_schema`` declaration is malformed."""


class RegistrationError(AgenticFunctionError):
    """Raised when a function fails to register (duplicate name, bad metadata)."""


# ---------------------------------------------------------------------------
# Runtime / execution errors
# ---------------------------------------------------------------------------
class BackendError(AgenticFunctionError):
    """Raised when a backend fails in an unexpected way (network, auth, ...)."""

    def __init__(self, message: str, *, backend: str | None = None,
                 status_code: int | None = None, raw: Any = None) -> None:
        super().__init__(message)
        self.backend = backend
        self.status_code = status_code
        self.raw = raw


class ValidationError(AgenticFunctionError):
    """Raised when an LLM response fails to validate against the output schema."""

    def __init__(self, message: str, *, errors: list[dict[str, Any]] | None = None,
                 raw_output: Any = None) -> None:
        super().__init__(message)
        self.errors = errors or []
        self.raw_output = raw_output


class ParseError(AgenticFunctionError):
    """Raised when we cannot even parse the raw LLM output (not JSON, etc.)."""

    def __init__(self, message: str, *, raw_output: Any = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output


class RetryExhaustedError(AgenticFunctionError):
    """Raised when every retry attempt has been used up and the last call still failed.

    Carries the full ``CallMetrics`` (when available) so unit tests / model
    evals can assert on ``attempts``, ``retries``, and per-attempt error
    categories without scraping log lines.
    """

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        last_exception: BaseException,
        metrics: Any = None,
        attempt_errors: list[Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception
        self.metrics = metrics
        self.attempt_errors = list(attempt_errors or [])
        # Convenience mirror — prefer metrics.retries when present.
        if metrics is not None and getattr(metrics, "retries", None) is not None:
            self.retries = int(metrics.retries)
        else:
            self.retries = max(0, attempts - 1)

    @property
    def error_category(self) -> str | None:
        if self.metrics is not None:
            return getattr(self.metrics, "error_category", None)
        if self.attempt_errors:
            last = self.attempt_errors[-1]
            return getattr(last, "category", None) or (
                last.get("category") if isinstance(last, dict) else None
            )
        return None


def error_category_of(exc: BaseException) -> str:
    """Normalize an exception into a stable category label for metrics / evals.

    Categories: ``validation`` | ``parse`` | ``backend`` | ``timeout`` |
    ``schema`` | ``retry_exhausted`` | ``budget`` | ``cache`` |
    ``composition`` | ``config`` | ``registration`` | ``agentic`` | ``unknown``.
    """
    # Local imports avoided — all types live in this module. BudgetExceededError
    # is defined in runtime.budget to keep budget deps out of errors.py; we
    # match it by name / base class as a fallback.
    if isinstance(exc, RetryExhaustedError):
        # Prefer the *root* failure category (why we retried), not the wrapper.
        root = exc.last_exception
        if root is not None and root is not exc:
            return error_category_of(root)
        return "retry_exhausted"
    if isinstance(exc, ValidationError):
        return "validation"
    if isinstance(exc, ParseError):
        return "parse"
    if isinstance(exc, BackendError):
        return "backend"
    if isinstance(exc, TimeoutError_):
        return "timeout"
    if isinstance(exc, SchemaError):
        return "schema"
    if isinstance(exc, CacheError):
        return "cache"
    if isinstance(exc, CompositionError):
        return "composition"
    if isinstance(exc, ConfigError):
        return "config"
    if isinstance(exc, RegistrationError):
        return "registration"
    name = type(exc).__name__
    if name == "BudgetExceededError":
        return "budget"
    if isinstance(exc, AgenticFunctionError):
        return "agentic"
    return "unknown"


class CacheError(AgenticFunctionError):
    """Raised when a cache backend fails to read or write."""


class CompositionError(AgenticFunctionError):
    """Raised when composing agentic functions fails (cycles, missing tool, ...)."""


class TimeoutError_(AgenticFunctionError):
    """Raised when an agentic function call exceeds its timeout."""