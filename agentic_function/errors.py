"""Unified exception hierarchy for agentic-function.

All custom exceptions derive from ``AgenticFunctionError`` so callers can catch
one base type when they don't care about specifics.
"""
from __future__ import annotations

from typing import Any


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
    """Raised when every retry attempt has been used up and the last call still failed."""

    def __init__(self, message: str, *, attempts: int, last_exception: BaseException) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


class CacheError(AgenticFunctionError):
    """Raised when a cache backend fails to read or write."""


class CompositionError(AgenticFunctionError):
    """Raised when composing agentic functions fails (cycles, missing tool, ...)."""


class TimeoutError_(AgenticFunctionError):
    """Raised when an agentic function call exceeds its timeout."""