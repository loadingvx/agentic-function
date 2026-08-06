"""agentic-function — type-safe LLM-powered functions.

Public API::

    from agentic_function import (
        agentic_function, AgenticResult, configure,
        register_backend, get_backend, MockBackend,
        as_openai_tool, trace,
    )

Quick start::

    @agentic_function(
        model="gpt-4o-mini",
        output_schema={"label": str, "score": float},
    )
    def classify(text: str) -> AgenticResult:
        '''Classify the sentiment of ``text``.'''

    result = classify("I love this!")
    result.label, result.score
"""
from __future__ import annotations

__version__ = "0.1.0a0"

# ---- Core ----
from .core import (
    agentic_function,
    AgenticFunction,
    AgenticResult,
    DynamicResult,
    SchemaSpec,
    resolve_schema,
    render_prompt,
)

# ---- Errors ----
from .errors import (
    AgenticFunctionError,
    BackendError,
    CacheError,
    CompositionError,
    ConfigError,
    ParseError,
    RegistrationError,
    RetryExhaustedError,
    SchemaError,
    TimeoutError_,
    ValidationError,
)

# ---- Runtime ----
from .runtime import (
    Executor,
    GlobalConfig,
    TraceContext,
    TraceSpan,
    TraceRecorder,
    CallMetrics,
    PhaseTimings,
    TokenUsage,
    RetryPolicy,
    default_retry_policy,
    CacheBackend,
    InMemoryCache,
    DiskCache,
    NullCache,
    configure,
    global_config,
    trace,
    get_current_trace,
    # 0.5 additions — observability + debug + quant
    Budget,
    BudgetTracker,
    BudgetExceededError,
    install_budget_tracker,
    get_default_budget_tracker,
    Aggregator,
    FunctionStats,
    install_default_aggregator,
    get_default_aggregator,
    Diagnostic,
    diagnose,
    diagnose_metrics,
    explain_failure,
    snapshot,
)
from .core.decorator import get_default_executor, set_default_executor

# ---- Backends ----
from .backends import (
    LLMBackend,
    LLMResponse,
    StreamChunk,
    MockBackend,
    OpenAIBackend,
    register_backend,
    get_backend,
    get_default_backend,
    set_default_backend,
    known_backends,
)
from . import testing

# ---- Composition ----
from .composition import (
    FunctionRegistry,
    get_global_registry,
    register,
    get_function,
    as_openai_tool,
    as_anthropic_tool,
)


__all__ = [
    # version
    "__version__",
    # core
    "agentic_function",
    "AgenticFunction",
    "AgenticResult",
    "DynamicResult",
    "SchemaSpec",
    "resolve_schema",
    "render_prompt",
    # errors
    "AgenticFunctionError",
    "BackendError",
    "CacheError",
    "CompositionError",
    "ConfigError",
    "ParseError",
    "RegistrationError",
    "RetryExhaustedError",
    "SchemaError",
    "TimeoutError_",
    "ValidationError",
    # runtime
    "Executor",
    "GlobalConfig",
    "TraceContext",
    "TraceSpan",
    "TraceRecorder",
    "CallMetrics",
    "TokenUsage",
    "RetryPolicy",
    "default_retry_policy",
    "CacheBackend",
    "InMemoryCache",
    "DiskCache",
    "NullCache",
    "configure",
    "global_config",
    "trace",
    "get_current_trace",
    "get_default_executor",
    "set_default_executor",
    # backends
    "LLMBackend",
    "LLMResponse",
    "StreamChunk",
    "MockBackend",
    "OpenAIBackend",
    "register_backend",
    "get_backend",
    "get_default_backend",
    "set_default_backend",
    "known_backends",
    "testing",
    # composition
    "FunctionRegistry",
    "get_global_registry",
    "register",
    "get_function",
    "as_openai_tool",
    "as_anthropic_tool",
]