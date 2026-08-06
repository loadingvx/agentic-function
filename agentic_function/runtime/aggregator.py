"""Process-wide aggregator — accumulates ``CallMetrics`` across many calls.

Use cases:

* Show the user a one-line "you spent $0.42 across 1,247 calls today"
* Export Prometheus / OpenMetrics for Grafana dashboards
* Enforce a process-wide cost ceiling via ``install_budget_tracker``
* Drive a "Top-N most expensive agentic functions" report
* **Model unit-test evals**: retry rate, recovered rate, failure categories —
  how often a model needed a second chance to match the schema

The aggregator is **thread-safe**. It lives at module level so users don't
have to thread it through their application.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from .metrics import CallMetrics


@dataclass(slots=True)
class FunctionStats:
    """Per-function aggregate — success, retries, and failure taxonomy."""
    function: str
    call_count: int = 0
    success_count: int = 0
    cache_hit_count: int = 0
    failure_count: int = 0
    # Retry / recovery — first-class eval signals
    total_retries: int = 0            # sum of CallMetrics.retries
    total_attempts: int = 0           # sum of CallMetrics.attempts
    calls_with_retries: int = 0       # calls where retries > 0 (success or fail)
    recovered_count: int = 0          # successful after >= 1 retry
    total_latency_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    last_error: str | None = None
    last_error_category: str | None = None
    errors_by_category: dict[str, int] = field(default_factory=dict)
    # Histogram of retries-per-call: key = retries count, value = #calls
    retries_histogram: dict[int, int] = field(default_factory=dict)
    # Histogram-lite: latency buckets for percentiles / SLOs.
    latency_buckets_ms: list[float] = field(default_factory=lambda: [
        10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10_000, 30_000,
    ])
    latency_counts: list[int] = field(default_factory=list)

    def avg_latency_ms(self) -> float:
        if not self.call_count:
            return 0.0
        return self.total_latency_ms / self.call_count

    def avg_retries(self) -> float:
        if not self.call_count:
            return 0.0
        return self.total_retries / self.call_count

    def avg_attempts(self) -> float:
        if not self.call_count:
            return 0.0
        return self.total_attempts / self.call_count

    def retry_rate(self) -> float:
        """Fraction of calls that needed at least one retry."""
        if not self.call_count:
            return 0.0
        return self.calls_with_retries / self.call_count

    def recovery_rate(self) -> float:
        """Of calls that retried, what fraction eventually succeeded."""
        if not self.calls_with_retries:
            return 0.0
        return self.recovered_count / self.calls_with_retries

    def failure_rate(self) -> float:
        if not self.call_count:
            return 0.0
        return self.failure_count / self.call_count


class Aggregator:
    """Thread-safe accumulator for ``CallMetrics``.

    Parameters
    ----------
    history_limit:
        If > 0, retain the most recent ``CallMetrics`` objects (for
        ``capture_metrics`` / post-hoc eval). 0 means no history (default,
        cheapest).
    on_record:
        Optional callback invoked after each successful ``record()``.
        Exceptions from the callback are swallowed so they never break calls.
    """

    def __init__(
        self,
        *,
        history_limit: int = 0,
        on_record: Callable[[CallMetrics], None] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._by_function: dict[str, FunctionStats] = {}
        self._history_limit = max(0, int(history_limit))
        self._history: list[CallMetrics] = []
        self._on_record = on_record
        # Cumulative totals across all functions.
        self._totals: dict[str, Any] = {
            "call_count": 0,
            "success_count": 0,
            "cache_hit_count": 0,
            "failure_count": 0,
            "total_retries": 0,
            "total_attempts": 0,
            "calls_with_retries": 0,
            "recovered_count": 0,
            "total_latency_ms": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost_usd": 0.0,
        }
        self._errors_by_category: dict[str, int] = defaultdict(int)
        # Also keep the legacy string-prefix counter for backwards compat.
        self._errors_by_type: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Record one call.
    # ------------------------------------------------------------------
    def record(self, metrics: CallMetrics, *, function_name: str) -> None:
        with self._lock:
            stats = self._by_function.get(function_name)
            if stats is None:
                stats = FunctionStats(function=function_name)
                stats.latency_counts = [0] * len(stats.latency_buckets_ms)
                self._by_function[function_name] = stats

            retries = max(0, int(metrics.retries or 0))
            attempts = int(metrics.attempts or 0)
            if attempts <= 0:
                attempts = retries + 1 if (retries or metrics.successful or metrics.cache_hit) else 0

            stats.call_count += 1
            stats.total_latency_ms += metrics.latency_ms
            # Prefer total_usage (includes failed retries) when present.
            usage = metrics.total_usage if (
                metrics.total_usage.total_tokens
                or metrics.total_usage.prompt_tokens
                or metrics.total_usage.completion_tokens
            ) else metrics.usage
            stats.total_prompt_tokens += usage.prompt_tokens
            stats.total_completion_tokens += usage.completion_tokens
            cost = metrics.total_cost_usd if metrics.total_cost_usd is not None else metrics.cost_usd
            stats.total_cost_usd += (cost or 0.0)

            stats.total_retries += retries
            stats.total_attempts += attempts
            stats.retries_histogram[retries] = stats.retries_histogram.get(retries, 0) + 1
            if retries > 0:
                stats.calls_with_retries += 1

            if metrics.cache_hit:
                stats.cache_hit_count += 1
            if metrics.successful:
                stats.success_count += 1
                if retries > 0:
                    stats.recovered_count += 1
            else:
                stats.failure_count += 1
                if metrics.error:
                    stats.last_error = metrics.error
                if metrics.error_category:
                    stats.last_error_category = metrics.error_category

            # Count every failed attempt once (model-eval signal). Fallback to
            # the final error_category only when attempt history is empty.
            if metrics.attempt_errors:
                for ae in metrics.attempt_errors:
                    stats.errors_by_category[ae.category] = (
                        stats.errors_by_category.get(ae.category, 0) + 1
                    )
                    self._errors_by_category[ae.category] += 1
            elif metrics.error_category:
                stats.errors_by_category[metrics.error_category] = (
                    stats.errors_by_category.get(metrics.error_category, 0) + 1
                )
                self._errors_by_category[metrics.error_category] += 1

            for i, ub in enumerate(stats.latency_buckets_ms):
                if metrics.latency_ms <= ub:
                    stats.latency_counts[i] += 1
                    break

            self._totals["call_count"] += 1
            self._totals["total_latency_ms"] += metrics.latency_ms
            self._totals["total_prompt_tokens"] += usage.prompt_tokens
            self._totals["total_completion_tokens"] += usage.completion_tokens
            self._totals["total_cost_usd"] += (cost or 0.0)
            self._totals["total_retries"] += retries
            self._totals["total_attempts"] += attempts
            if retries > 0:
                self._totals["calls_with_retries"] += 1
            if metrics.cache_hit:
                self._totals["cache_hit_count"] += 1
            if metrics.successful:
                self._totals["success_count"] += 1
                if retries > 0:
                    self._totals["recovered_count"] += 1
            else:
                self._totals["failure_count"] += 1

            if metrics.error:
                err_type = metrics.error.split(":", 1)[0].strip()
                self._errors_by_type[err_type] += 1

            if self._history_limit > 0:
                self._history.append(metrics)
                overflow = len(self._history) - self._history_limit
                if overflow > 0:
                    del self._history[:overflow]

        if self._on_record is not None:
            try:
                self._on_record(metrics)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        with self._lock:
            totals = dict(self._totals)
            call_count = totals["call_count"] or 0
            totals["avg_retries"] = (
                totals["total_retries"] / call_count if call_count else 0.0
            )
            totals["avg_attempts"] = (
                totals["total_attempts"] / call_count if call_count else 0.0
            )
            totals["retry_rate"] = (
                totals["calls_with_retries"] / call_count if call_count else 0.0
            )
            totals["recovery_rate"] = (
                totals["recovered_count"] / totals["calls_with_retries"]
                if totals["calls_with_retries"] else 0.0
            )
            totals["failure_rate"] = (
                totals["failure_count"] / call_count if call_count else 0.0
            )
            return {
                "totals": totals,
                "functions": {name: _stats_to_dict(s)
                              for name, s in self._by_function.items()},
                "errors_by_category": dict(self._errors_by_category),
                # Back-compat alias (string prefix of metrics.error).
                "errors_by_type": dict(self._errors_by_type),
            }

    def top_functions(self, n: int = 10, *, by: str = "total_cost_usd") -> list[dict[str, Any]]:
        with self._lock:
            items = sorted(
                self._by_function.values(),
                key=lambda s: _sort_key(s, by),
                reverse=True,
            )
        return [_stats_to_dict(s) for s in items[:n]]

    def history(self) -> list[CallMetrics]:
        """Return a copy of retained CallMetrics (empty unless history_limit > 0)."""
        with self._lock:
            return list(self._history)

    def reset(self) -> None:
        with self._lock:
            self._by_function.clear()
            for k, v in list(self._totals.items()):
                self._totals[k] = 0 if isinstance(v, int) else 0.0
            self._errors_by_category.clear()
            self._errors_by_type.clear()
            self._history.clear()

    # ------------------------------------------------------------------
    # Prometheus exposition format — zero external deps.
    # ------------------------------------------------------------------
    def to_prometheus(self) -> str:
        """Render as Prometheus text exposition (v0.0.4)."""
        lines: list[str] = []
        with self._lock:
            lines.append("# HELP agentic_calls_total Total agentic function calls")
            lines.append("# TYPE agentic_calls_total counter")
            for name, s in self._by_function.items():
                lines.append(f'agentic_calls_total{{function="{name}"}} {s.call_count}')

            lines.append("# HELP agentic_call_errors_total Calls that ended in error")
            lines.append("# TYPE agentic_call_errors_total counter")
            for name, s in self._by_function.items():
                if s.failure_count:
                    lines.append(f'agentic_call_errors_total{{function="{name}"}} {s.failure_count}')

            lines.append("# HELP agentic_retries_total Sum of per-call retry counts")
            lines.append("# TYPE agentic_retries_total counter")
            for name, s in self._by_function.items():
                lines.append(f'agentic_retries_total{{function="{name}"}} {s.total_retries}')

            lines.append("# HELP agentic_attempts_total Sum of per-call attempt counts")
            lines.append("# TYPE agentic_attempts_total counter")
            for name, s in self._by_function.items():
                lines.append(f'agentic_attempts_total{{function="{name}"}} {s.total_attempts}')

            lines.append("# HELP agentic_calls_with_retries_total Calls that needed >=1 retry")
            lines.append("# TYPE agentic_calls_with_retries_total counter")
            for name, s in self._by_function.items():
                if s.calls_with_retries:
                    lines.append(
                        f'agentic_calls_with_retries_total{{function="{name}"}} {s.calls_with_retries}'
                    )

            lines.append("# HELP agentic_recovered_total Calls that succeeded after retries")
            lines.append("# TYPE agentic_recovered_total counter")
            for name, s in self._by_function.items():
                if s.recovered_count:
                    lines.append(
                        f'agentic_recovered_total{{function="{name}"}} {s.recovered_count}'
                    )

            lines.append("# HELP agentic_errors_by_category_total Failures by category")
            lines.append("# TYPE agentic_errors_by_category_total counter")
            for name, s in self._by_function.items():
                for cat, cnt in s.errors_by_category.items():
                    lines.append(
                        f'agentic_errors_by_category_total{{function="{name}",category="{cat}"}} {cnt}'
                    )

            lines.append("# HELP agentic_cache_hits_total Calls served from cache")
            lines.append("# TYPE agentic_cache_hits_total counter")
            for name, s in self._by_function.items():
                if s.cache_hit_count:
                    lines.append(f'agentic_cache_hits_total{{function="{name}"}} {s.cache_hit_count}')

            lines.append("# HELP agentic_tokens_total Total LLM tokens charged")
            lines.append("# TYPE agentic_tokens_total counter")
            for name, s in self._by_function.items():
                lines.append(f'agentic_tokens_total{{function="{name}",kind="prompt"}} {s.total_prompt_tokens}')
                lines.append(f'agentic_tokens_total{{function="{name}",kind="completion"}} {s.total_completion_tokens}')

            lines.append("# HELP agentic_cost_usd_total Total cost in USD")
            lines.append("# TYPE agentic_cost_usd_total counter")
            for name, s in self._by_function.items():
                lines.append(f'agentic_cost_usd_total{{function="{name}"}} {s.total_cost_usd}')

            lines.append("# HELP agentic_latency_ms_sum Cumulative latency")
            lines.append("# TYPE agentic_latency_ms_sum counter")
            for name, s in self._by_function.items():
                lines.append(f'agentic_latency_ms_sum{{function="{name}"}} {s.total_latency_ms}')

            lines.append("# HELP agentic_latency_ms_bucket Latency histogram")
            lines.append("# TYPE agentic_latency_ms_bucket histogram")
            for name, s in self._by_function.items():
                running = 0
                for ub, cnt in zip(s.latency_buckets_ms, s.latency_counts):
                    running += cnt
                    lines.append(f'agentic_latency_ms_bucket{{function="{name}",le="{ub}"}} {running}')
                lines.append(f'agentic_latency_ms_bucket{{function="{name}",le="+Inf"}} {s.call_count}')
        return "\n".join(lines) + "\n"


def _sort_key(s: FunctionStats, by: str) -> Any:
    if hasattr(s, by):
        return getattr(s, by)
    # Allow sorting by derived rates.
    fn = getattr(s, by, None)
    if callable(fn):
        return fn()
    return 0.0


def _stats_to_dict(s: FunctionStats) -> dict[str, Any]:
    return {
        "function": s.function,
        "call_count": s.call_count,
        "success_count": s.success_count,
        "cache_hit_count": s.cache_hit_count,
        "failure_count": s.failure_count,
        "total_retries": s.total_retries,
        "total_attempts": s.total_attempts,
        "calls_with_retries": s.calls_with_retries,
        "recovered_count": s.recovered_count,
        "avg_retries": s.avg_retries(),
        "avg_attempts": s.avg_attempts(),
        "retry_rate": s.retry_rate(),
        "recovery_rate": s.recovery_rate(),
        "failure_rate": s.failure_rate(),
        "retries_histogram": dict(sorted(s.retries_histogram.items())),
        "errors_by_category": dict(s.errors_by_category),
        "avg_latency_ms": s.avg_latency_ms(),
        "total_latency_ms": s.total_latency_ms,
        "total_prompt_tokens": s.total_prompt_tokens,
        "total_completion_tokens": s.total_completion_tokens,
        "total_cost_usd": s.total_cost_usd,
        "last_error": s.last_error,
        "last_error_category": s.last_error_category,
    }


# ----------------------------------------------------------------------
# Process-wide default aggregator
# ----------------------------------------------------------------------
_default: Aggregator | None = None


def get_default_aggregator() -> Aggregator | None:
    return _default


def install_default_aggregator(agg: Aggregator | None) -> Aggregator | None:
    """Install / clear the process-wide aggregator. Returns the previous one."""
    global _default
    prev = _default
    _default = agg
    return prev


__all__ = [
    "Aggregator",
    "FunctionStats",
    "get_default_aggregator",
    "install_default_aggregator",
]
