# agentic-function

Turn LLM capabilities into ordinary Python functions with `@agentic_function`: describe the task in the docstring, constrain the output with a schema; the library handles prompt rendering, validation, retries, caching, tracing, and multi-backend execution.

**Languages:** [English](README.md) · [中文](README.zh.md)

- License: [MIT](LICENSE)
- Python: 3.10+
- Status: Alpha (`0.0.1a0`) · [PyPI](https://pypi.org/project/agentic-function/)

---

## Installation

The current PyPI release is a pre-release: `0.0.1a0`. pip does not install alpha / beta versions by default, so `pip install agentic-function` alone will not pick it up. Pre-release APIs may still change; install explicitly:

```bash
pip install --pre agentic-function
# or pin the version
pip install agentic-function==0.0.1a0
```

Optional provider SDKs:

```bash
pip install --pre "agentic-function[openai]"
pip install --pre "agentic-function[anthropic]"
pip install --pre "agentic-function[openai,anthropic]"
```

Editable install from source:

```bash
pip install -e ".[dev,openai,anthropic]"
```

---

## Quick start

```python
from agentic_function import agentic_function, AgenticResult, set_default_backend
from agentic_function.backends.mock_backend import MockBackend
from agentic_function.testing import mock_llm

set_default_backend(MockBackend())
mock_llm({"category": "positive", "confidence": 0.94, "reasoning": "..."})

@agentic_function(
    output_schema={
        "category": str,
        "confidence": float,
        "reasoning": str,
    },
)
def classify_sentiment(text: str) -> AgenticResult:
    """Classify the sentiment of ``text``.

    - ``category``: "positive" | "negative" | "neutral"
    - ``confidence``: [0.0, 1.0]
    - ``reasoning``: short explanation
    """

result = classify_sentiment("Amazing launch today!")
print(result.category, result.confidence, result.reasoning)
print(result.metrics.latency_ms, result.metrics.usage.prompt_tokens)
```

Runnable examples under [`examples/`](examples/):

```bash
python examples/01_sentiment_classification.py
python examples/02_information_extraction.py
python examples/03_summarization.py
python examples/04_intent_routing.py
python examples/05_composition.py
python examples/06_real_minimax.py   # requires an API key
```

Examples `01`–`05` use `MockBackend` and need no API key.

---

## Features

| Area | Notes |
| ---- | ----- |
| Decorator API | `@agentic_function`; call it like a normal function |
| Output schema | `dict`, pydantic `BaseModel`, or `Literal[...]` |
| Composition | Plain Python calls between functions |
| Tool export | `as_openai_tool` / `as_anthropic_tool`, `FunctionRegistry` |
| Backends | Mock, OpenAI, Anthropic, MiniMax, plus `register_backend` |
| Validation & retry | pydantic validation; retry on parse / validation failure |
| Cache | `InMemoryCache` / `DiskCache` / `NullCache` |
| Metrics & cost | `CallMetrics` on every result (latency, tokens, estimated USD, …) |
| Trace & budget | `trace`, `BudgetTracker`, `Aggregator` (incl. Prometheus text) |
| Diagnostics | `diagnose` / `explain_failure` / `snapshot`; `debug=` / `AGENTIC_DEBUG` |
| Testing helpers | `mock_llm`, `mock_llm_table`, `freeze_time`, `capture_metrics` |
| Async | `.acall()` primary path; sync `__call__` available |
| Errors | `ValidationError`, `RetryExhaustedError`, `BudgetExceededError`, … |

---

## Usage

### Schema and structured output

Declare `output_schema` as a `dict`, a `BaseModel`, or a `Literal[...]` return annotation. The library injects the JSON schema into the prompt (or uses provider tool / json_schema mode), applies common coercions, and validates with pydantic. On failure it retries according to policy. Callers receive typed fields, not a raw string to parse.

```python
@agentic_function(output_schema={"label": str, "score": float})
def classify(text: str) -> AgenticResult:
    """Classify sentiment of ``text``."""
```

### Composition

```python
topic = extract_topic(article)
summary = make_summary(article, topic.topic, topic.tone)
```

See [`examples/05_composition.py`](examples/05_composition.py).

### Tool export

Export a function as OpenAI / Anthropic tool JSON for an external agent or custom tool loop:

```python
from agentic_function import as_openai_tool, as_anthropic_tool, register, get_function

openai_tool = as_openai_tool(make_summary)
anthropic_tool = as_anthropic_tool(make_summary)

register(make_summary)
fn = get_function(make_summary.qualified_name)
```

### Backends

Built-ins: `MockBackend`, `OpenAIBackend`, `AnthropicBackend`, and the MiniMax-CN preset `minimax`. Register custom backends with `register_backend(...)`.

```python
@agentic_function(backend="mock", output_schema={"label": str})
@agentic_function(backend="openai", model="gpt-4o-mini", output_schema={"label": str})
@agentic_function(backend="anthropic", model="claude-sonnet-4-20250514", output_schema={"label": str})
@agentic_function(backend="minimax", model="MiniMax-M3", output_schema={"label": str})
```

### Prompt parameters

| Parameter | Purpose |
| --------- | ------- |
| docstring | Task description (system prompt body) |
| `few_shots` | `[(input, output), …]` exemplars |
| `prompt_template` / `system_template` | Custom `{placeholder}` templates |
| `include_schema_in_prompt` | Whether to inject the schema |
| `description` | Tool-export blurb (defaults to first docstring line) |
| `render_prompt(fn, args, kwargs)` | Inspect the message list before calling |

### Async

```python
out = await classify.acall("terrible")
```

Works as free functions, methods, and classmethods (descriptor protocol).

---

## Metrics, tracing, and cost

Every result includes `CallMetrics`:

```python
result.metrics.latency_ms
result.metrics.usage.prompt_tokens
result.metrics.usage.completion_tokens
result.metrics.usage.total_tokens
result.metrics.cost_usd
result.metrics.cache_hit
result.metrics.attempts
result.metrics.retries
result.metrics.recovered
result.metrics.attempt_errors
result.metrics.timings
result.metrics.total_cost_usd
```

Tracing, budgets, and aggregation:

```python
from agentic_function import (
    trace,
    Budget, BudgetTracker, install_budget_tracker,
    Aggregator, install_default_aggregator,
)

with trace("nightly_eval") as ctx:
    out = classify(sample.text)
    ctx.span.set_attribute("sample.id", sample.id)

install_budget_tracker(BudgetTracker(budgets=[
    Budget(metric="cost_usd", limit=5.0),
]))

agg = install_default_aggregator(Aggregator())
print(agg.summary())
print(agg.to_prometheus())
```

Diagnostics:

```python
from agentic_function import diagnose, explain_failure, snapshot

print(diagnose(result).to_dict())
print(explain_failure(exc))
print(snapshot(result))
```

Cache keys cover `(model, schema, few_shots, prompt_hash)`. Bound retries with `RetryPolicy(max_retries=..., base_delay=..., max_delay=...)`.

`examples/06_real_minimax.py` exercises a live backend (API key required). The default `tests/` suite does not use the network.

---

## Testing

```python
from agentic_function.testing import mock_llm, mock_llm_table, freeze_time, capture_metrics

mock_llm({"label": "positive", "score": 0.95})
assert classify("amazing").label == "positive"

mock_llm_table([
    {"label": "positive", "score": 0.9},
    {"label": "negative", "score": 0.8},
])

with freeze_time():
    classify("text")

with capture_metrics() as bag:
    classify("a")
    classify("b")
assert len(bag) == 2
```

You can also use `MockBackend`, or patch the `openai` / `anthropic` clients to assert request shaping (see `tests/test_anthropic_backend.py`).

```bash
pytest tests/
pytest tests/test_anthropic_backend.py
pytest --cov=agentic_function
```

Isolation points:

| Concern | Approach |
| ------- | -------- |
| Schema | Test the pydantic model directly |
| Prompt | `render_prompt(fn, args, kwargs)` |
| Backend request | Patch the provider SDK |
| Backend response | Pass a synthetic `LLMResponse` |
| Retry | Raise synthetic retryable errors |
| Cache | Swap cache implementations |
| Tool export | Assert tool JSON shape |
| Metrics | `capture_metrics` / aggregator |

---

## Decorator parameters

| Parameter | Default | Meaning |
| --------- | ------- | ------- |
| `model` | global default | Model id for the backend |
| `output_schema` | inferred from return annotation | `dict` / `BaseModel` / `Literal[...]` |
| `backend` | `set_default_backend(...)` | Instance or registered name |
| `temperature`, `top_p`, `max_tokens`, `stop` | global config | Sampling params |
| `max_retries` | global config | Retries on parse / validation failure |
| `retry_policy` | `RetryPolicy(...)` | Backoff and retryable exceptions |
| `cache` | global default | Per-call cache override |
| `timeout` | global config | Request timeout (seconds) |
| `include_schema_in_prompt` | `True` | Inject JSON schema into the system message |
| `few_shots` | `[]` | Exemplar pairs |
| `prompt_template` / `system_template` | `None` | Custom templates |
| `description` | first docstring line | Tool-export description |
| `debug` | `False` / `AGENTIC_DEBUG` | Attach request/response snapshots |
| `executor` | global default | Custom `Executor` |

Environment variables: `AGENTIC_FUNCTION_MODEL`, `AGENTIC_FUNCTION_BACKEND`, `AGENTIC_FUNCTION_CACHE`, `AGENTIC_FUNCTION_CACHE_DIR`, plus `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `MINIMAX_CN_API_KEY`, etc.

---

## Public API

```python
from agentic_function import (
    agentic_function, AgenticFunction, AgenticResult, DynamicResult,
    SchemaSpec, resolve_schema, render_prompt,
    LLMBackend, LLMResponse, StreamChunk,
    MockBackend, OpenAIBackend,
    register_backend, get_backend, get_default_backend, set_default_backend,
    known_backends,
    Executor, GlobalConfig, configure, global_config,
    TraceContext, TraceSpan, TraceRecorder, trace, get_current_trace,
    CallMetrics, TokenUsage, PhaseTimings,
    RetryPolicy, default_retry_policy,
    CacheBackend, InMemoryCache, DiskCache, NullCache,
    get_default_executor, set_default_executor,
    Budget, BudgetTracker, BudgetExceededError,
    install_budget_tracker, get_default_budget_tracker,
    Aggregator, FunctionStats,
    install_default_aggregator, get_default_aggregator,
    Diagnostic, diagnose, diagnose_metrics, explain_failure, snapshot,
    FunctionRegistry, get_global_registry, register, get_function,
    as_openai_tool, as_anthropic_tool,
    testing,
    AgenticFunctionError, BackendError, CacheError, CompositionError,
    ConfigError, ParseError, RegistrationError, RetryExhaustedError,
    SchemaError, TimeoutError_ as TimeoutError, ValidationError,
)
```

`AnthropicBackend` and MiniMax live under `agentic_function.backends` and register as `"anthropic"` / `"minimax"`.

---

## Architecture

```
@agentic_function(...)
        │
        ▼
  AgenticFunction  (descriptor / call / await)
        │ ExecutionRequest
        ▼
     Executor
   trace → cache → retry → schema → backend
        │
        ├── BudgetTracker
        ├── Aggregator
        └── as_*_tool / Registry
```

---

## Project layout

```
agentic-function/
├── agentic_function/
│   ├── core/
│   ├── backends/
│   ├── runtime/
│   ├── composition/
│   ├── validation/
│   ├── utils/
│   └── testing.py
├── tests/
├── examples/
└── docs/
```

---

## Roadmap

- [x] v0.1 — decorator + pydantic schema validation
- [x] v0.2 — pluggable backends (Mock, OpenAI)
- [x] v0.3 — function composition
- [x] v0.4 — cache, cost, `mock_llm`
- [x] v0.5 — async, tracing, tool export, Anthropic / MiniMax, budget / aggregator / diagnostics
- [ ] v0.6 — Ollama adapter and richer streaming
- [ ] v0.7 — streaming output public API
- [ ] v1.0 — stable API and complete docs

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
