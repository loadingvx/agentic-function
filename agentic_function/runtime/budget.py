"""Per-function and process-wide cost / latency budgets.

A ``BudgetTracker`` accumulates ``CallMetrics`` from completed calls and
raises ``BudgetExceededError`` when a configured ceiling is crossed. There
are two scopes:

- **Function-scoped budgets** — set via ``@agentic_function(
  budget_usd=0.05)``. Cheap and local; the check happens after each call
  to *that* function.
- **Process-scoped budgets** — set via ``install_budget_tracker(...)``.
  Aggregates across every function. Useful for "spend no more than $5/day
  on agentic features in this worker".

The default tracker is ``None`` (no enforcement). When set, every successful
call records its ``CallMetrics`` into the tracker; subsequent calls check
thresholds *before* running the backend call.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from ..errors import AgenticFunctionError
from .metrics import CallMetrics


class BudgetExceededError(AgenticFunctionError):
    """Raised when a configured budget ceiling is crossed.

    Attributes:
        scope:  "function" or "process"
        metric: "cost_usd" or "latency_ms" or "tokens"
        limit:  the configured limit
        used:   the running total that triggered the breach
        metric_value: the value of *this* call that would have pushed the total over
    """

    def __init__(self, message: str, *, scope: str, metric: str,
                 limit: float, used: float, metric_value: float) -> None:
        super().__init__(message)
        self.scope = scope
        self.metric = metric
        self.limit = limit
        self.used = used
        self.metric_value = metric_value


@dataclass
class Budget:
    """A single ceiling on one metric."""
    metric: str                          # "cost_usd" | "latency_ms" | "tokens"
    limit: float
    scope: str = "process"               # "function" | "process" — default to process so it's the common ceiling.


@dataclass
class BudgetTracker:
    """Accumulator for cost / latency / token budgets.

    A tracker is **thread-safe** — ``record()`` and ``check()`` take a lock
    so concurrent agentic functions don't race past the limit.
    """
    budgets: list[Budget] = field(default_factory=list)
    _totals: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _record_count: int = 0

    def add(self, budget: Budget) -> None:
        with self._lock:
            self.budgets.append(budget)
            self._totals.setdefault(budget.metric, 0.0)

    def totals(self) -> dict[str, float]:
        with self._lock:
            return dict(self._totals)

    def record_count(self) -> int:
        with self._lock:
            return self._record_count

    # ------------------------------------------------------------------
    # Core: record a completed call's metrics and check budgets.
    # ------------------------------------------------------------------
    def check(self, metrics: CallMetrics, *, scope: str = "process") -> None:
        """Check *prospective* budgets — called BEFORE the backend runs.

        Looks at the running total + the cost we'd add for this call
        (estimated from previous runs or ``0`` for the first call). If a
        budget would be exceeded, raise ``BudgetExceededError``.

        For the first call we don't have a previous-cost estimate, so we
        skip the check; the post-call ``record()`` will catch sustained
        overruns.
        """
        candidate_costs = self._estimate_candidate_cost(metrics)
        with self._lock:
            for budget in self.budgets:
                if budget.scope != scope:
                    continue
                used = self._totals.get(budget.metric, 0.0)
                cand = candidate_costs.get(budget.metric, 0.0)
                if used + cand > budget.limit:
                    raise BudgetExceededError(
                        f"{budget.metric} budget exceeded: "
                        f"{used + cand:.6f} > limit {budget.limit} "
                        f"(scope={budget.scope})",
                        scope=budget.scope,
                        metric=budget.metric,
                        limit=budget.limit,
                        used=used + cand,
                        metric_value=cand,
                    )

    def record(self, metrics: CallMetrics, *, scope: str = "process") -> None:
        """Record a completed call's metrics, then check post-hoc budgets."""
        spent = self._extract_value(metrics)
        with self._lock:
            self._record_count += 1
            for budget in self.budgets:
                if budget.scope != scope:
                    continue
                self._totals[budget.metric] = self._totals.get(budget.metric, 0.0) + spent.get(budget.metric, 0.0)
            for budget in self.budgets:
                if budget.scope != scope:
                    continue
                used = self._totals.get(budget.metric, 0.0)
                if used > budget.limit:
                    raise BudgetExceededError(
                        f"{budget.metric} budget exceeded: "
                        f"{used:.6f} > limit {budget.limit} "
                        f"(scope={budget.scope})",
                        scope=budget.scope,
                        metric=budget.metric,
                        limit=budget.limit,
                        used=used,
                        metric_value=spent.get(budget.metric, 0.0),
                    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_value(metrics: CallMetrics) -> dict[str, float]:
        return {
            "cost_usd": metrics.total_cost_usd or metrics.cost_usd or 0.0,
            "latency_ms": metrics.latency_ms,
            "tokens": float(metrics.total_usage.total_tokens or metrics.usage.total_tokens),
        }

    @staticmethod
    def _estimate_candidate_cost(metrics: CallMetrics) -> dict[str, float]:
        """Best-effort estimate of the *upcoming* call's footprint.

        We don't have the response yet, so we can't know the real numbers.
        For the *first* call we just assume 0; the post-call ``record()``
        path will catch overruns. Once we have at least one prior call,
        we use the *last* ``CallMetrics`` value as a proxy.
        """
        return {
            "cost_usd": metrics.cost_usd or 0.0,
            "latency_ms": metrics.latency_ms,
            "tokens": float(metrics.usage.total_tokens),
        }

    def reset(self) -> None:
        with self._lock:
            self._totals.clear()
            self._record_count = 0


# ----------------------------------------------------------------------
# Process-wide default tracker.
# ----------------------------------------------------------------------
_default_tracker: BudgetTracker | None = None


def get_default_budget_tracker() -> BudgetTracker | None:
    return _default_tracker


def install_budget_tracker(tracker: BudgetTracker | None) -> BudgetTracker | None:
    """Install (or clear) the process-wide ``BudgetTracker``.

    Pass ``None`` to remove. Returns the previously-installed tracker (or
    ``None``) so callers can save/restore in tests.
    """
    global _default_tracker
    prev = _default_tracker
    _default_tracker = tracker
    return prev


__all__ = [
    "Budget",
    "BudgetTracker",
    "BudgetExceededError",
    "get_default_budget_tracker",
    "install_budget_tracker",
]