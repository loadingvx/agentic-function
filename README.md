# agentic-function

> **Type-safe Python functions powered by LLMs.**
> Define a function with `@agentic_function`, describe what it should do in
> the docstring, declare the output schema — and the framework handles
> prompt rendering, schema validation, retries, caching, tracing, and
> multi-backend execution.

**Languages:** [English](README.md) · [中文](README.zh.md)

[License: MIT](LICENSE)
[Python 3.10+](https://www.python.org/downloads/)
[Status: Alpha `0.0.1a0`](https://pypi.org/project/agentic-function/)

---

## Installation (Alpha — read this)

> **Current PyPI release is an alpha: `0.0.1a0`.**
> pip **does not install alpha/beta by default**, so a plain
> `pip install agentic-function` will fail or skip this version until a
> stable release exists. That is intentional: pre-releases are hidden so
> production installs do not pick up APIs that may still change.

**Install the current alpha (pick one):**

```bash
# Recommended: opt into pre-releases
pip install --pre agentic-function

# Or pin the exact alpha version
pip install agentic-function==0.0.1a0
```

**With LLM provider SDKs:**

```bash
pip install --pre "agentic-function[openai]"
pip install --pre "agentic-function[anthropic]"
pip install --pre "agentic-function[openai,anthropic]"
```

Requires **Python 3.10+**. For local development from source:

```bash
pip install -e ".[dev,openai,anthropic]"
```

---

## What is an Agentic Function?

```python
from agentic_function import agentic_function, AgenticResult

@agentic_function(
    model="gpt-4o-mini",
    output_schema={
        "category": str,        # "positive" | "negative" | "neutral"
        "confidence": float,    # 0.0 .. 1.0
        "reasoning": str,       # <= 80 chars
    },
)
def classify_sentiment(text: str) -> AgenticResult:
    """Classify the sentiment of ``text``.

    - ``category``: one of "positive", "negative", "neutral"
    - ``confidence``: a float in [0.0, 1.0]
    - ``reasoning``: a short explanation (<= 80 chars)
    """

result = classify_sentiment("The launch today was absolutely amazing!")
result.category    # → "positive"
result.confidence  # → 0.94
result.reasoning   # → "Strong positive adjectives, exclamation mark."
result.metrics.latency_ms
result.metrics.usage.prompt_tokens
```

The function looks and behaves like a regular Python function — but its body
is a docstring/prompt, and the implementation is "any LLM that can match the
declared output schema". The decorator wires up everything else.

---

## Framework strengths — everything you get

Most LLM code is glue: prompt strings, fragile JSON parsing, ad-hoc retries,
and a pile of provider SDKs. **Agentic Function turns that glue into a typed
Python function** — and keeps every production concern (validation, cost,
cache, tools, tests) as a first-class capability rather than homework.

### 1. Decorator-first — zero new concepts

One decorator. No Agent / Tool / Chain / Memory vocabulary to learn.
If you can write a Python function, you can ship an LLM step.

```python
@agentic_function(output_schema={"label": str, "score": float})
def classify(text: str) -> AgenticResult:
    """Classify sentiment of ``text``."""
```

### 2. Schema-enforced structured output

Declare the contract once — as a `dict`, a pydantic `BaseModel`, or a
`Literal[...]` return annotation. The framework:

- injects the JSON schema into the prompt (or uses provider tool / json_schema mode)
- coerces common LLM quirks (`"0.9"` → `0.9`, CSV → `list`)
- validates with pydantic
- auto-retries on parse / validation failure

Your callers always get typed attributes, not a string you have to parse.

### 3. Plain-Python composition

Agentic Functions call Agentic Functions like any other Python code.
No orchestration DSL. Pipelines stay readable and unit-testable.

```python
topic = extract_topic(article)
summary = make_summary(article, topic.topic, topic.tone)
```

See [`examples/05_composition.py`](examples/05_composition.py).

### 4. Drop into any Agent as a tool (`as_tool`)

Need a multi-step planner? Keep the planner in your Agent framework — and
export each Agentic Function as a native OpenAI / Anthropic tool in one line.
You get **typed, validated, metered tools** without rewriting them for the Agent.

```python
from agentic_function import as_openai_tool, as_anthropic_tool, register, get_function

# OpenAI Agents / LangChain / custom tool loops
openai_tool = as_openai_tool(make_summary)
# → {"type": "function", "function": {"name": ..., "parameters": {...}}}

# Anthropic tool_use
anthropic_tool = as_anthropic_tool(make_summary)
# → {"name": ..., "description": ..., "input_schema": {...}}

# Dynamic lookup for Agent dispatchers
register(make_summary)
fn = get_function(make_summary.qualified_name)
```

This is the intended bridge: **Agentic Function for single-step reliability;
Agent frameworks for multi-step planning** — same functions, both worlds.

### 5. Pluggable backends — swap with one argument

Built-in: `MockBackend`, `OpenAIBackend`, `AnthropicBackend`, and `minimax`
(Anthropic-protocol preset for MiniMax-CN). Register your own via
`register_backend(...)`.

```python
@agentic_function(backend="mock", output_schema={"label": str})          # local
@agentic_function(backend="openai", model="gpt-4o-mini", ...)            # staging
@agentic_function(backend="anthropic", model="claude-3-5-sonnet-latest") # prod
@agentic_function(backend="minimax", model="MiniMax-M3", ...)            # MiniMax-CN
```

Callers and the function body never change.

### 6. Small / cheap models work reliably

Because the contract is a **filled-in schema** (often via tool-input /
json_schema mode), a 7B-class model only has to fill known slots — the same
reason Claude Haiku function-calling is reliable. The framework absorbs
format variability so success rates stay high on cheap models.

| Layer | Recommendation |
| ----- | -------------- |
| Dev / unit tests | `MockBackend` + `mock_llm()` — zero cost, deterministic |
| Staging / CI | DeepSeek / Qwen 7B / Llama-3.1-8B / Haiku / GPT-4o-mini |
| Production | Promote only where metrics show the cheap model underperforms |

### 7. Production metrics & cost on every call

Every result carries a `CallMetrics` object — latency, tokens, USD cost,
cache hit, **attempts / retries**, phase timings — no extra instrumentation.
Retry count is a first-class model-quality signal for unit-test evals.

```python
result.metrics.latency_ms
result.metrics.usage.total_tokens
result.metrics.cost_usd
result.metrics.cache_hit
result.metrics.attempts         # backend calls made
result.metrics.retries          # attempts - 1 (0 = first-try success)
result.metrics.recovered        # True if succeeded only after retries
result.metrics.attempt_errors   # [{attempt, category, error_type, message}, …]
result.metrics.timings          # PhaseTimings: prompt / backend / validate / …
result.metrics.total_cost_usd   # across retries
```

```python
from agentic_function import Aggregator, install_default_aggregator
from agentic_function.testing import capture_metrics, eval_summary
from agentic_function.errors import RetryExhaustedError

# Per-suite eval: retry_rate / recovery_rate / errors_by_category
with capture_metrics() as bag:
    for sample in dataset:
        try:
            classify(sample.text)
        except RetryExhaustedError as exc:
            assert exc.metrics.retries >= 1
            assert exc.error_category in {"validation", "parse", "backend"}

summary = eval_summary(bag)
assert summary["totals"]["retry_rate"] < 0.1   # model rarely needs a second chance
assert summary["errors_by_category"].get("validation", 0) == 0
```

### 8. Nested tracing, budgets, and Prometheus aggregation

```python
from agentic_function import (
    trace,
    Budget, BudgetTracker, install_budget_tracker,
    Aggregator, install_default_aggregator,
)

with trace("nightly_eval") as ctx:
    out = classify(sample.text)
    ctx.span.set_attribute("sample.id", sample.id)

# Process-wide spend / latency / token ceilings
install_budget_tracker(BudgetTracker(budgets=[
    Budget(metric="cost_usd", limit=5.0),
]))

# Per-function stats + Prometheus exposition
agg = install_default_aggregator(Aggregator())
# … after traffic …
print(agg.summary())
print(agg.to_prometheus())
```

### 9. Diagnostics that speak human

When something fails in staging, you get structured explanations — not a
raw SDK stacktrace.

```python
from agentic_function import diagnose, explain_failure, snapshot

print(diagnose(result).to_dict())       # success / cache_hit / failed breakdown
print(explain_failure(exc))             # category, attempts, actionable hint
print(snapshot(result))                 # log-shippable dict
```

Enable request/response snapshots with `debug=True` on the decorator or
`AGENTIC_DEBUG=1`.

### 10. Caching that actually saves money

Stable cache key over `(model, schema, few_shots, prompt_hash)`.
Swap `InMemoryCache` / `DiskCache` / `NullCache`. Identical calls return in
microseconds and mark `metrics.cache_hit = True`.

### 11. Async-first, sync-friendly

`.acall()` is the primary path; `__call__` is a thin sync wrapper.
Works as free functions, methods, and classmethods via the descriptor protocol.

```python
out = await classify.acall("terrible")
```

### 12. Prompt ergonomics without a template engine

| Knob | Purpose |
| ---- | ------- |
| docstring | System prompt (the task description) |
| `few_shots=[(in, out), …]` | In-context exemplars as chat turns |
| `prompt_template` / `system_template` | Custom `{placeholder}` formatting |
| `include_schema_in_prompt` | Toggle schema injection |
| `description` | Tool-export blurb (defaults to first docstring line) |
| `render_prompt(fn, args, kwargs)` | Inspect the exact message list before calling |

### 13. Testability without the network

```python
from agentic_function.testing import (
    mock_llm, mock_llm_table, freeze_time, isolate_execution, capture_metrics,
)

mock_llm({"label": "positive", "score": 0.95})
assert classify("amazing").label == "positive"

mock_llm_table([
    {"label": "positive", "score": 0.9},
    {"label": "negative", "score": 0.8},
])

with freeze_time():
    r = classify("text")          # deterministic PhaseTimings

with capture_metrics() as bag:
    classify("a"); classify("b")
assert len(bag) == 2
```

Backend request shaping is also unit-testable by patching the provider SDK
(see `tests/test_anthropic_backend.py`).

### 14. Typed error hierarchy

Catch what you mean: `ValidationError`, `ParseError`, `BackendError`,
`RetryExhaustedError`, `BudgetExceededError`, `CacheError`,
`CompositionError`, `TimeoutError_`, …

---

## Why Agentic Function instead of a general Agent framework?

Most "AI-powered" code is **single-step**: classify, extract, summarise,
route, rewrite, score. None of it needs a multi-step Agent runtime.
Wrapping such a step in a generic Agent framework is overkill:

| Dimension         | General Agent framework                      | Agentic Function                                  |
| ----------------- | -------------------------------------------- | ------------------------------------------------- |
| Onboarding cost   | Learn Agent / Tool / Chain / Memory concepts | One decorator                                     |
| Type safety       | Output is usually a string                   | Inputs annotated, output validated against schema |
| Structured output | Hand-rolled, frequently fails                | Framework-enforced (validate + retry)             |
| Composition       | Constrained by framework protocol            | Plain Python functions calling each other         |
| Agent integration | Rewrite steps as tools                       | `as_openai_tool` / `as_anthropic_tool` — one line |
| Unit testing      | Mock the whole Agent runtime                 | `mock_llm()` — one line, pure unit test           |
| Observability     | Framework-internal trace                     | Metrics + spans + budget + Prometheus             |
| Learning curve    | Steep                                        | Flat — it's just a function                       |

**Use both when you need both.** Keep Agentic Functions as the reliable
single-step building blocks; expose them to an Agent planner via `as_*_tool`
only when multi-step planning is actually required.

---

## Testing

`agentic-function` is designed to be **test-friendly at every layer.**

### 1. Pure unit tests with `mock_llm()`

```python
from agentic_function import agentic_function, AgenticResult
from agentic_function.testing import mock_llm

@agentic_function(output_schema={"label": str, "score": float})
def classify(text: str) -> AgenticResult:
    """Classify sentiment. - label: positive|negative|neutral"""

def test_classify_positive():
    mock_llm({"label": "positive", "score": 0.95})
    out = classify("amazing launch!")
    assert out.label == "positive"
    assert out.score == pytest.approx(0.95)
```

### 2. Deterministic tests with `MockBackend`

```python
from agentic_function.backends.mock_backend import MockBackend
from agentic_function import set_default_backend

set_default_backend(MockBackend())  # every call returns a schema-shaped dict
```

### 3. Multi-call fixtures with `mock_llm_table`

```python
from agentic_function.testing import mock_llm_table

mock_llm_table([
    {"topic": "OpenAI", "tone": "informational"},
    {"summary": "...", "tags": ["ai"], "score": 0.9},
])
```

### 4. Async-path tests are first-class

```python
@pytest.mark.asyncio
async def test_async_classify():
    mock_llm({"label": "negative", "score": 0.8})
    out = await classify.acall("terrible")
    assert out.label == "negative"
```

### 5. Backend tests with mocked SDK clients

The `AnthropicBackend` / `OpenAIBackend` tests demonstrate how to patch
`anthropic.Anthropic` / `openai.OpenAI` to verify **request shaping**
(temperature, max_tokens, tool schemas, timeouts, base URLs) without
hitting the network. See `tests/test_anthropic_backend.py`.

### 6. Run the suite

```bash
pytest tests/                           # full suite, <1s, no network
pytest tests/test_anthropic_backend.py  # backend-specific
pytest --cov=agentic_function           # with coverage
```

---

## Quantification & observability

Every call returns a `CallMetrics` object — production-ready telemetry
out of the box, no extra setup.

```python
result = classify("amazing launch!")

result.metrics.latency_ms                  # end-to-end wall time
result.metrics.usage.prompt_tokens         # tokens in
result.metrics.usage.completion_tokens     # tokens out
result.metrics.usage.total_tokens
result.metrics.cost_usd                    # USD, computed from pricing table
result.metrics.cache_hit                   # True / False
result.metrics.attempts                    # usually 1, more if retried
result.metrics.retries                     # number of retries
result.metrics.successful                  # True / False
result.metrics.error                       # last error message if any
result.metrics.timings                     # PhaseTimings breakdown
result.metrics.extra                       # backend-specific extras
```

### Real numbers from a live `minimax` (MiniMax-M3, Anthropic-protocol) call

The numbers below are from `examples/06_real_minimax.py` against the live
LLM, with no caching:

```
── sentiment classification ──
latency       : 1678.9 ms
tokens        : 85 in / 39 out (total 124)
cost_usd      : 3.6e-05
attempts      : 1
cache_hit     : False

── deal extraction ──
latency       : 2172.0 ms
tokens        : 422 in / 60 out (total 482)
cost_usd      : 9.9e-05
attempts      : 1

── async language detection ──
latency       : 1769.1 ms
tokens        : 322 in / 31 out (total 353)
cost_usd      : 6.7e-05
attempts      : 1
```

The full test suite runs in **<1 second** because nothing in `tests/` hits
the network.

### Cost control in production

- **Cache hit** is the single biggest win: identical `(model, schema, few_shots, prompt_hash)` keys return cached output in microseconds.
- **Retry policy** is bounded: `RetryPolicy(max_retries=2, base_delay=0.5, max_delay=4.0)` stops runaway loops.
- **BudgetTracker** stops spend / latency / tokens before they blow past a ceiling.
- `cost_usd` per call gives you a per-feature P&L line item.

---

## Development workflow

### Iteration loop

```bash
# 1. Build / edit a function with MockBackend — instant feedback
python -c "from agentic_function import *; ..."

# 2. Run unit tests
pytest tests/ -x

# 3. Run a real LLM example (cheap, schema-validated)
set -a && source ~/.hermes/.env && set +a
python examples/06_real_minimax.py

# 4. Promote to production model when quality is acceptable
```

### Promoting between backends is one decorator arg

```python
# dev
@agentic_function(backend="mock", output_schema={"label": str})

# staging
@agentic_function(backend="openai", model="gpt-4o-mini",
                  output_schema={"label": str})

# production
@agentic_function(backend="anthropic", model="claude-3-5-sonnet-latest",
                  output_schema={"label": str})
```

### Project layout

```
agentic-function/
├── agentic_function/
│   ├── core/           # decorator, schema, prompt, result
│   ├── backends/       # mock / openai / anthropic / minimax
│   ├── runtime/        # executor, cache, retry, trace, metrics, budget, aggregator, diagnostics
│   ├── composition/    # as_tool, registry
│   ├── validation/     # coerce + validator
│   ├── utils/          # hashing, logging, cost
│   └── testing.py      # mock_llm, mock_llm_table, freeze_time, …
├── tests/
├── examples/           # 5 mock + 1 real LLM
└── docs/
```

---

## Testability

The framework is **deliberately layered so that every concern can be
isolated in tests.**

| Concern                         | How it's isolated for testing                               | What you can verify                              |
| ------------------------------- | ----------------------------------------------------------- | ------------------------------------------------ |
| **Schema validation**           | pydantic model alone                                        | Field types, ranges, `Literal` enums             |
| **Prompt rendering**           | `render_prompt(fn, args, kwargs)` standalone fn             | Exact message list sent to backend               |
| **Backend request shaping**     | Patch `anthropic.Anthropic` / `openai.OpenAI`               | Tool schemas, sampling params, timeouts, headers |
| **Backend response conversion** | Pass a synthetic `LLMResponse`                              | Dict vs raw string, token parsing, JSON repair   |
| **Retry policy**                | Synthetic `BackendError` / `ParseError` / `ValidationError` | `is_retryable()`, backoff timing, max attempts   |
| **Cache**                       | `InMemoryCache`, `DiskCache`, `NullCache` are all swappable | Hit/miss, TTL, key stability                     |
| **Executor orchestration**      | Substitute any layer above with a stub                      | Order of operations, trace span lifecycle        |
| **Composition / as-tool**       | Mock an inner call; assert tool JSON shape                  | Typed inner result; OpenAI/Anthropic tool schema |
| **Observability**               | `capture_metrics`, `freeze_time`, aggregator                | Latency buckets, cost totals, Prometheus text    |
| **End-to-end with real LLM**    | `examples/06_real_minimax.py`                               | Real latency, tokens, cost, error recovery       |

### CI-friendly patterns

```yaml
# .github/workflows/test.yml
- run: pytest tests/ --tb=short            # always runs, <1s, no secrets
- run: pytest tests/test_*.py -m "not live" # opt-in live tests skipped
```

Mark live-network tests with `@pytest.mark.live` and skip them in CI; the
default suite is hermetic.

---

## Capabilities at a glance

| Strength | Capability |
| -------- | ---------- |
| 🎯 Decorator-first API | `@agentic_function` — no new concepts |
| 📋 Flexible schemas | `dict` / `BaseModel` / `Literal[...]` |
| 🔄 Plain-Python composition | Functions call functions |
| 🔧 Agent tool export | `as_openai_tool` / `as_anthropic_tool` + `FunctionRegistry` |
| 🔌 Multi-backend | Mock · OpenAI · Anthropic · MiniMax · custom |
| 🛡️ Validate + coerce + retry | Schema mismatch never silently ships |
| 💾 Pluggable cache | In-memory / disk / null |
| 📊 Per-call metrics & cost | Latency, tokens, USD, phase timings |
| 🌳 Nested tracing | `contextvars` spans, OTel-ready recorder |
| 💰 Budget ceilings | Process-wide cost / latency / token limits |
| 📈 Aggregator | Per-function stats + `to_prometheus()` |
| 🩺 Diagnostics | `diagnose` / `explain_failure` / `snapshot` / `debug=` |
| 🧪 Test helpers | `mock_llm`, `mock_llm_table`, `freeze_time`, `capture_metrics` |
| 🌊 Async-first | `.acall()` primary; sync `__call__` wrapper |
| 📝 Prompt knobs | few-shots, templates, schema injection |
| ⚠️ Typed errors | Catch `ValidationError`, `RetryExhaustedError`, … |

---

## Installation

See **[Installation (Alpha — read this)](#installation-alpha--read-this)** at the top.
Summary: current version is `0.0.1a0`; use `pip install --pre agentic-function`
(or pin `==0.0.1a0`). Plain `pip install agentic-function` will not pick up alphas.

---

## Quick start

```python
from agentic_function import agentic_function, AgenticResult, set_default_backend
from agentic_function.backends.mock_backend import MockBackend
from agentic_function.testing import mock_llm

# 1. Choose your backend.
set_default_backend(MockBackend())    # or OpenAIBackend() once you have a key

# 2. Pre-register a canned response (only needed for tests).
mock_llm({"category": "positive", "confidence": 0.94, "reasoning": "..."})

# 3. Declare the function. The docstring is the prompt.
@agentic_function(output_schema={"category": str, "confidence": float, "reasoning": str})
def classify_sentiment(text: str) -> AgenticResult:
    """Classify the sentiment of ``text``."""

# 4. Call it.
result = classify_sentiment("Amazing launch today!")
print(result.category, result.confidence, result.reasoning)
print(result.metrics.latency_ms, result.metrics.usage.prompt_tokens)
```

Run the runnable examples in [`examples/`](examples/):

```bash
python examples/01_sentiment_classification.py
python examples/02_information_extraction.py
python examples/03_summarization.py
python examples/04_intent_routing.py
python examples/05_composition.py          # composition + as_openai_tool
python examples/06_real_minimax.py         # live MiniMax (needs API key)
```

Examples `01`–`05` use `MockBackend` and need no API key.

---

## Decorator parameters (cheat sheet)

| Parameter                                    | Default                                           | Meaning                                                    |
| -------------------------------------------- | ------------------------------------------------- | ---------------------------------------------------------- |
| `model`                                      | global default                                    | Model identifier passed to the backend                     |
| `output_schema`                              | inferred from return annotation                   | `dict[str, type]`, `BaseModel` subclass, or `Literal[...]` |
| `backend`                                    | `None` (falls back to `set_default_backend(...)`) | `LLMBackend` instance or registered name                   |
| `temperature`, `top_p`, `max_tokens`, `stop` | from global config                                | Sampling params                                            |
| `max_retries`                                | from global config                                | Auto-retry on parse / validation errors                    |
| `retry_policy`                               | `RetryPolicy(max_retries=...)`                    | Customise backoff / retryable exceptions                   |
| `cache`                                      | global default                                    | Per-call caching override                                  |
| `timeout`                                    | from global config                                | Per-call request timeout (seconds)                         |
| `include_schema_in_prompt`                   | `True`                                            | Inject JSON schema into the system message                 |
| `few_shots`                                  | `[]`                                              | `[(input, output), …]` exemplars                           |
| `prompt_template`                            | `None`                                            | Custom user-message template (str-format style)            |
| `system_template`                            | `None`                                            | Custom system-message template                             |
| `description`                                | first line of docstring                           | For tool export                                            |
| `debug`                                      | `False` / `AGENTIC_DEBUG`                         | Attach request/response snapshots to metrics               |
| `executor`                                   | global default                                    | Custom `Executor` instance                                 |

Environment defaults: `AGENTIC_FUNCTION_MODEL`, `AGENTIC_FUNCTION_BACKEND`,
`AGENTIC_FUNCTION_CACHE`, `AGENTIC_FUNCTION_CACHE_DIR`, plus provider keys
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `MINIMAX_CN_API_KEY`, …).

---

## Public API

```python
from agentic_function import (
    # core
    agentic_function,            # the decorator
    AgenticFunction,             # the descriptor class returned by the decorator
    AgenticResult,               # marker base for user output models
    DynamicResult,               # returned when schema was a dict
    SchemaSpec, resolve_schema,  # schema machinery
    render_prompt,               # build a message list for one call
    # backends
    LLMBackend, LLMResponse, StreamChunk,
    MockBackend, OpenAIBackend,
    register_backend, get_backend, get_default_backend, set_default_backend,
    known_backends,
    # runtime
    Executor, GlobalConfig, configure, global_config,
    TraceContext, TraceSpan, TraceRecorder, trace, get_current_trace,
    CallMetrics, TokenUsage, PhaseTimings,
    RetryPolicy, default_retry_policy,
    CacheBackend, InMemoryCache, DiskCache, NullCache,
    get_default_executor, set_default_executor,
    # observability (v0.5)
    Budget, BudgetTracker, BudgetExceededError,
    install_budget_tracker, get_default_budget_tracker,
    Aggregator, FunctionStats,
    install_default_aggregator, get_default_aggregator,
    Diagnostic, diagnose, diagnose_metrics, explain_failure, snapshot,
    # composition
    FunctionRegistry, get_global_registry, register, get_function,
    as_openai_tool, as_anthropic_tool,
    # testing
    testing,                     # mock_llm, mock_llm_table, freeze_time, …
    # errors
    AgenticFunctionError, BackendError, CacheError, CompositionError,
    ConfigError, ParseError, RegistrationError, RetryExhaustedError,
    SchemaError, TimeoutError_ as TimeoutError, ValidationError,
)
```

`AnthropicBackend` / MiniMax live under `agentic_function.backends` and are
auto-registered as `"anthropic"` / `"minimax"`.

---

## Architecture

```
                        ┌────────────────────────────┐
                        │   @agentic_function(...)   │
                        └─────────────┬──────────────┘
                                      │ builds
                                      ▼
                        ┌────────────────────────────┐
                        │       AgenticFunction      │  (descriptor, call/await)
                        └─────────────┬──────────────┘
                                      │ ExecutionRequest
                                      ▼
        ┌─────────────────────────────────────────────────────────┐
        │                       Executor                           │
        │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────────┐  │
        │  │trace │→ │cache │→ │retry │→ │schema│→ │ backend  │  │
        │  └──────┘  └──────┘  └──────┘  └──────┘  └──────────┘  │
        └─────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         BudgetTracker   Aggregator     as_*_tool / Registry
```

Each layer is independently testable:

- **Executor** — orchestrates everything
- **Trace** — `contextvars`-based nested spans
- **Cache** — pluggable (`InMemory`, `Disk`, `Null`)
- **Retry** — `RetryPolicy` + `is_retryable` registry
- **Schema** — pydantic-driven validation + coercion
- **Backend** — `LLMBackend` subclass (provider adapter)
- **Composition** — tool export + name registry for Agent bridges

---

## Roadmap

- [x] v0.1 — `@agentic_function` decorator + pydantic schema validation
- [x] v0.2 — pluggable backends (`Mock`, `OpenAI`)
- [x] v0.3 — composition (function-as-function)
- [x] v0.4 — caching + cost tracking + `mock_llm()` testing helper
- [x] v0.5 — async + tracing + `as_openai_tool` / `as_anthropic_tool` + Anthropic / MiniMax + observability (budget / aggregator / diagnostics)
- [ ] v0.6 — Ollama adapter + richer streaming
- [ ] v0.7 — streaming output (public API)
- [ ] v1.0 — stable API + complete docs

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports, ideas, docs improvements
and PRs all welcome.

## License

[MIT](LICENSE)
