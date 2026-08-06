"""Token usage and per-call metrics.

This module is intentionally side-effect-free: nothing here touches global
state. The runtime/aggregator.py module is responsible for *aggregating*
metrics across many calls.

CallMetrics now carries:

- Phase timings (prompt render, cache lookup, backend round-trip, parse +
  validate, cache write, retry backoff) so a single ``latency_ms`` can be
  decomposed when debugging.
- Backend identity (backend name, model, finish_reason) — was previously
  only available via ``extra`` dict, now first-class fields.
- Per-attempt accounting: ``attempts`` of usage tokens + cost, so when
  retries happen the *total* tokens charged are visible.
- Retry / failure observability: ``retries``, ``error_category``, and a
  full ``attempt_errors`` history — critical for model unit-test evals
  (retry rate = how often the model needs a second chance).
- An optional ``request_snapshot`` / ``response_snapshot`` (truncated,
  sanitised) for debugging without re-running.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AttemptError:
    """One failed attempt inside a call's retry loop.

    ``attempt`` is 0-based (matching the executor's loop index).
    ``category`` is a stable label from :func:`errors.error_category_of`.
    """
    attempt: int
    category: str
    error_type: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "category": self.category,
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token accounting for a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def of(cls, prompt: int, completion: int) -> "TokenUsage":
        return cls(prompt_tokens=prompt, completion_tokens=completion,
                   total_tokens=prompt + completion)

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def __sub__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens - other.prompt_tokens,
            completion_tokens=self.completion_tokens - other.completion_tokens,
            total_tokens=self.total_tokens - other.total_tokens,
        )


@dataclass(slots=True)
class PhaseTimings:
    """Milliseconds spent in each Executor phase.

    All fields default to 0.0; populated by ``Executor._run_sync/_run_async``
    via ``perf_counter`` deltas. Use ``total_ms()`` to recover the wall-clock
    sum (overlaps with ``latency_ms`` because the Executor is strictly
    sequential).
    """
    prompt_render_ms: float = 0.0
    cache_lookup_ms: float = 0.0
    backend_call_ms: float = 0.0     # only the last successful attempt's round-trip
    parse_validate_ms: float = 0.0
    cache_write_ms: float = 0.0
    retry_backoff_ms: float = 0.0    # total time spent sleeping between retries

    def total_ms(self) -> float:
        return (
            self.prompt_render_ms
            + self.cache_lookup_ms
            + self.backend_call_ms
            + self.parse_validate_ms
            + self.cache_write_ms
            + self.retry_backoff_ms
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "prompt_render_ms": self.prompt_render_ms,
            "cache_lookup_ms": self.cache_lookup_ms,
            "backend_call_ms": self.backend_call_ms,
            "parse_validate_ms": self.parse_validate_ms,
            "cache_write_ms": self.cache_write_ms,
            "retry_backoff_ms": self.retry_backoff_ms,
            "total_ms": self.total_ms(),
        }


@dataclass(slots=True)
class CallMetrics:
    """Aggregated metrics for a single ``@agentic_function`` invocation.

    These accumulate across the inner retry attempts so you can see how many
    tokens the whole call cost in the worst case.

    Retry semantics
    ---------------
    - ``attempts``: how many backend calls were made (always >= 1 on a live call)
    - ``retries``: ``attempts - 1`` after the call finishes (0 = first-try success)
    - ``recovered``: succeeded only after one or more failed attempts — a key
      model-quality signal for unit-test evals
    - ``attempt_errors``: ordered history of every failed attempt
    - ``error_category``: category of the *last* failure (set on final failure;
      also retained briefly during the loop, cleared on clean/recovered success
      after being copied into ``attempt_errors``)
    """
    attempts: int = 0
    successful: bool = False
    latency_ms: float = 0.0
    function_name: str = ""        # Set by the Executor to its qualified name.

    # Token accounting for the LAST successful response (or zero if failed).
    usage: TokenUsage = field(default_factory=TokenUsage)

    # Tokens charged across ALL attempts (including failed retries). When
    # ``retries == 0`` this equals ``usage``.
    total_usage: TokenUsage = field(default_factory=TokenUsage)

    cache_hit: bool = False
    cost_usd: float | None = None
    total_cost_usd: float | None = None    # sum across all attempts
    retries: int = 0
    error: str | None = None
    error_category: str | None = None
    attempt_errors: list[AttemptError] = field(default_factory=list)

    # New in 0.5 — first-class backend identity. Older code paths populated
    # ``extra`` with these, but the public API now has stable field names.
    backend: str = ""                     # "openai" / "anthropic" / "minimax" / "mock"
    model: str = ""
    finish_reason: str | None = None

    # Per-phase timings. See PhaseTimings docstring.
    timings: PhaseTimings = field(default_factory=PhaseTimings)

    # Optional debug snapshots. ``None`` unless the user enabled debug mode
    # via the decorator's ``debug`` arg or the ``AGENTIC_DEBUG`` env var.
    # We only store a *truncated* copy of the messages + raw response to
    # avoid bloating every call.
    request_snapshot: list[dict[str, Any]] | None = None
    response_snapshot: Any = None

    # Free-form backend-specific extras. Kept for backwards compatibility
    # (the field has been here since v0.2).
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived helpers — used by tests / Aggregator / diagnose()
    # ------------------------------------------------------------------
    @property
    def recovered(self) -> bool:
        """True when the call eventually succeeded after at least one retry."""
        return self.successful and self.retries > 0

    def note_attempt_failure(self, attempt: int, exc: BaseException) -> AttemptError:
        """Append a failed attempt and refresh ``error`` / ``error_category``.

        ``retries`` is set to ``attempt`` (0-based index of the failure) so
        that even mid-loop introspection sees a meaningful value; the
        executor overwrites it with the final count on success / exhaustion.
        """
        from ..errors import error_category_of

        record = AttemptError(
            attempt=attempt,
            category=error_category_of(exc),
            error_type=type(exc).__name__,
            message=str(exc),
        )
        self.attempt_errors.append(record)
        self.retries = attempt
        self.error = f"{record.error_type}: {record.message}"
        self.error_category = record.category
        return record

    def mark_success(self, *, attempt: int) -> None:
        """Finalize metrics for a successful (possibly recovered) call."""
        self.successful = True
        self.retries = attempt          # prior failures = current 0-based index
        self.attempts = attempt + 1
        # Keep the failure history for evals, but clear the live error fields
        # so aggregators don't treat a recovered call as a failure.
        self.error = None
        self.error_category = None

    def mark_failure(self) -> None:
        """Finalize metrics after the retry loop is exhausted / aborted."""
        self.successful = False
        if self.attempts > 0:
            self.retries = max(self.retries, self.attempts - 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "successful": self.successful,
            "recovered": self.recovered,
            "latency_ms": self.latency_ms,
            "cache_hit": self.cache_hit,
            "cost_usd": self.cost_usd,
            "total_cost_usd": self.total_cost_usd,
            "retries": self.retries,
            "error": self.error,
            "error_category": self.error_category,
            "attempt_errors": [a.as_dict() for a in self.attempt_errors],
            "backend": self.backend,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
            "total_usage": {
                "prompt_tokens": self.total_usage.prompt_tokens,
                "completion_tokens": self.total_usage.completion_tokens,
                "total_tokens": self.total_usage.total_tokens,
            },
            "timings": self.timings.as_dict(),
            **({"request_snapshot": self.request_snapshot}
               if self.request_snapshot is not None else {}),
            **({"response_snapshot": self.response_snapshot}
               if self.response_snapshot is not None else {}),
            **self.extra,
        }