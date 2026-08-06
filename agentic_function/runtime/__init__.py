"""Runtime subpackage: configuration, retry, cache, tracing, executor,
budgets, aggregators, diagnostics."""
from .config import global_config, GlobalConfig, configure
from .metrics import TokenUsage, CallMetrics, PhaseTimings, AttemptError
from .trace import TraceSpan, TraceContext, TraceRecorder, get_current_trace, trace
from .retry import RetryPolicy, default_retry_policy
from .cache import CacheBackend, InMemoryCache, DiskCache, NullCache, get_cache, set_cache
from .executor import Executor
from .budget import (
    Budget, BudgetTracker, BudgetExceededError,
    get_default_budget_tracker, install_budget_tracker,
)
from .aggregator import (
    Aggregator, FunctionStats,
    get_default_aggregator, install_default_aggregator,
)
from .diagnostics import (
    Diagnostic, diagnose, diagnose_metrics, explain_failure, snapshot,
)

__all__ = [
    # config + metrics
    "global_config", "GlobalConfig", "configure",
    "TokenUsage", "CallMetrics", "PhaseTimings", "AttemptError",
    # trace
    "TraceSpan", "TraceContext", "TraceRecorder", "get_current_trace", "trace",
    # retry
    "RetryPolicy", "default_retry_policy",
    # cache
    "CacheBackend", "InMemoryCache", "DiskCache", "NullCache", "get_cache", "set_cache",
    # executor
    "Executor",
    # budgets
    "Budget", "BudgetTracker", "BudgetExceededError",
    "get_default_budget_tracker", "install_budget_tracker",
    # aggregator
    "Aggregator", "FunctionStats",
    "get_default_aggregator", "install_default_aggregator",
    # diagnostics
    "Diagnostic", "diagnose", "diagnose_metrics", "explain_failure", "snapshot",
]