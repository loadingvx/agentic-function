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
- An optional ``request_snapshot`` / ``response_snapshot`` (truncated,
  sanitised) for debugging without re-running.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "successful": self.successful,
            "latency_ms": self.latency_ms,
            "cache_hit": self.cache_hit,
            "cost_usd": self.cost_usd,
            "total_cost_usd": self.total_cost_usd,
            "retries": self.retries,
            "error": self.error,
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