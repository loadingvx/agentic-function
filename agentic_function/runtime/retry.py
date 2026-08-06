"""Retry policies with exponential backoff + jitter."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from ..errors import BackendError, ParseError, ValidationError


@dataclass(frozen=True)
class RetryPolicy:
    """Describes when and how to retry after a transient failure."""
    max_retries: int = 2
    initial_delay: float = 0.5
    max_delay: float = 8.0
    exponential_base: float = 2.0
    jitter: bool = True

    def delay_for(self, attempt: int) -> float:
        """Return the seconds to sleep before attempt N (0-indexed)."""
        if attempt <= 0:
            return 0.0
        raw = self.initial_delay * (self.exponential_base ** (attempt - 1))
        capped = min(raw, self.max_delay)
        if self.jitter:
            capped = random.uniform(0.5 * capped, capped)  # nosec — not for crypto
        return capped


def default_retry_policy(max_retries: int = 2) -> RetryPolicy:
    return RetryPolicy(max_retries=max_retries)


# Errors that are always worth retrying — these are typically transient.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    BackendError,
    ParseError,
    ValidationError,
)


def is_retryable(exc: BaseException, policy: RetryPolicy, attempt: int) -> bool:
    """Decide whether ``exc`` should be retried.

    ``attempt`` is the 0-based index of the attempt that just failed
    (i.e. how many retries have already been consumed). When
    ``attempt >= max_retries``, no further retries are allowed — so
    ``max_retries=2`` permits attempts ``0, 1, 2`` (3 total).
    """
    if attempt >= policy.max_retries:
        return False
    if isinstance(exc, TRANSIENT_ERRORS):
        # ValidationError: yes if it's a "shape" failure, not a semantic one.
        if isinstance(exc, ValidationError):
            # Heuristic: if the raw_output is None (no JSON at all), keep retrying.
            return True
        return True
    return False


def sleep_or_yield(delay: float, *, is_async: bool) -> None:
    """Sleep in either sync or async context. Centralised so tests can patch it."""
    if delay <= 0:
        return
    if is_async:
        import asyncio
        asyncio.sleep(delay)  # type: ignore[arg-type]
    else:
        import time
        time.sleep(delay)