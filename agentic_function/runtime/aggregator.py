"""Process-wide aggregator — accumulates ``CallMetrics`` across many calls.

Use cases:

* Show the user a one-line "you spent $0.42 across 1,247 calls today"
* Export Prometheus / OpenMetrics for Grafana dashboards
* Enforce a process-wide cost ceiling via ``install_budget_tracker``
* Drive a "Top-N most expensive agentic functions" report

The aggregator is **thread-safe**. It lives at module level so users don't
have to thread it through their application.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .metrics import CallMetrics


@dataclass(slots=True)
class FunctionStats:
    """Per-function aggregate."""
    function: str
    call_count: int = 0
    success_count: int = 0
    cache_hit_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    last_error: str | None = None
    # Histogram-lite: latency buckets for percentiles / SLOs.
    latency_buckets_ms: list[float] = field(default_factory=lambda: [
        10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10_000, 30_000,
    ])
    latency_counts: list[int] = field(default_factory=list)

    def avg_latency_ms(self) -> float:
        if not self.call_count:
            return 0.0
        return self.total_latency_ms / self.call_count


class Aggregator:
    """Thread-safe accumulator for ``CallMetrics``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_function: dict[str, FunctionStats] = {}
        # Cumulative totals across all functions.
        self._totals = {
            "call_count": 0,
            "success_count": 0,
            "cache_hit_count": 0,
            "failure_count": 0,
            "total_latency_ms": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost_usd": 0.0,
        }
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
            stats.call_count += 1
            stats.total_latency_ms += metrics.latency_ms
            stats.total_prompt_tokens += metrics.usage.prompt_tokens
            stats.total_completion_tokens += metrics.usage.completion_tokens
            stats.total_cost_usd += (metrics.cost_usd or 0.0)
            if metrics.cache_hit:
                stats.cache_hit_count += 1
            if metrics.successful:
                stats.success_count += 1
            else:
                stats.failure_count += 1
                if metrics.error:
                    stats.last_error = metrics.error
            # Bucket the latency
            for i, ub in enumerate(stats.latency_buckets_ms):
                if metrics.latency_ms <= ub:
                    stats.latency_counts[i] += 1
                    break

            self._totals["call_count"] += 1
            self._totals["total_latency_ms"] += metrics.latency_ms
            self._totals["total_prompt_tokens"] += metrics.usage.prompt_tokens
            self._totals["total_completion_tokens"] += metrics.usage.completion_tokens
            self._totals["total_cost_usd"] += (metrics.cost_usd or 0.0)
            if metrics.cache_hit:
                self._totals["cache_hit_count"] += 1
            if metrics.successful:
                self._totals["success_count"] += 1
            else:
                self._totals["failure_count"] += 1

            if metrics.error:
                err_type = metrics.error.split(":", 1)[0].strip()
                self._errors_by_type[err_type] += 1

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "totals": dict(self._totals),
                "functions": {name: _stats_to_dict(s)
                              for name, s in self._by_function.items()},
                "errors_by_type": dict(self._errors_by_type),
            }

    def top_functions(self, n: int = 10, *, by: str = "total_cost_usd") -> list[dict[str, Any]]:
        with self._lock:
            items = sorted(
                self._by_function.values(),
                key=lambda s: getattr(s, by, 0.0),
                reverse=True,
            )
        return [_stats_to_dict(s) for s in items[:n]]

    def reset(self) -> None:
        with self._lock:
            self._by_function.clear()
            for k in self._totals:
                self._totals[k] = 0 if isinstance(self._totals[k], int) else 0.0
            self._errors_by_type.clear()

    # ------------------------------------------------------------------
    # Prometheus exposition format — zero external deps.
    # ------------------------------------------------------------------
    def to_prometheus(self) -> str:
        """Render as Prometheus text exposition (v0.0.4).

        Example::

            # HELP agentic_calls_total Total agentic function calls
            # TYPE agentic_calls_total counter
            agentic_calls_total{function="classify"} 42
            # HELP agentic_tokens_total Total LLM tokens charged
            # TYPE agentic_tokens_total counter
            agentic_tokens_total{function="classify",kind="prompt"} 1234
        """
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
                # +Inf bucket
                lines.append(f'agentic_latency_ms_bucket{{function="{name}",le="+Inf"}} {s.call_count}')
        return "\n".join(lines) + "\n"


def _stats_to_dict(s: FunctionStats) -> dict[str, Any]:
    return {
        "function": s.function,
        "call_count": s.call_count,
        "success_count": s.success_count,
        "cache_hit_count": s.cache_hit_count,
        "failure_count": s.failure_count,
        "avg_latency_ms": s.avg_latency_ms(),
        "total_latency_ms": s.total_latency_ms,
        "total_prompt_tokens": s.total_prompt_tokens,
        "total_completion_tokens": s.total_completion_tokens,
        "total_cost_usd": s.total_cost_usd,
        "last_error": s.last_error,
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