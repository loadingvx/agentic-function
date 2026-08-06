# agentic-function

> **Type-safe Python functions powered by LLMs.**
> Define a function with `@agentic_function`, describe what it should do in
> the docstring, declare the output schema — and the framework handles
> prompt rendering, schema validation, retries, caching, tracing, and
> multi-backend execution.

[License: MIT](LICENSE)
[Python 3.10+](https://www.python.org/downloads/)
[Status: Alpha]()

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



## Why Agentic Function instead of a general Agent framework?

Most "AI-powered" code is **single-step**: classify, extract, summarise,
route, rewrite, score. None of it needs a multi-step Agent runtime, tool
dispatch loop, or long-term memory. Wrapping such a step in a generic Agent
framework is overkill:


| Dimension         | General Agent framework                      | Agentic Function                                  |
| ----------------- | -------------------------------------------- | ------------------------------------------------- |
| Onboarding cost   | Learn Agent / Tool / Chain / Memory concepts | One decorator                                     |
| Type safety       | Output is usually a string                   | Inputs annotated, output validated against schema |
| Structured output | Hand-rolled, frequently fails                | Framework-enforced (validate + retry)             |
| Composition       | Constrained by framework protocol            | Plain Python functions calling each other         |
| Unit testing      | Mock the whole Agent runtime                 | `mock_llm()` — one line, pure unit test           |
| Observability     | Framework-internal trace                     | Per-call metrics + nested spans via contextvars   |
| Learning curve    | Steep                                        | Flat — it's just a function                       |


**Agentic Function is also composable with Agents.** Anything declared with
`@agentic_function` can be exposed as an OpenAI / Anthropic tool via
`as_openai_tool(...)` / `as_anthropic_tool(...)` and consumed by an Agent
framework when you actually do need a multi-step planner.

### Why small / cheap models work great with Agentic Functions

This is a deliberate design property of the framework: an agentic function
is **structurally much easier for an LLM than an open-ended chat turn.**


| What the model has to do     | Raw prompt + string output                                   | `@agentic_function`                                     |
| ---------------------------- | ------------------------------------------------------------ | ------------------------------------------------------- |
| Reasoning depth              | Decide what's important, format consistently, stay on topic  | Same as left — you still write the docstring            |
| Output format                | Free-form text → hand-rolled regex / `json.loads` / fallback | Strict JSON-schema match enforced by the framework      |
| Recovery from sloppy output  | You write the parser; one bad call breaks your pipeline      | Auto-retry on schema mismatch + pydantic coercion       |
| Few-shot in-context learning | Cut & paste into the system prompt by hand                   | `few_shots=[(input, output), ...]` on the decorator     |
| Prompt caching               | DIY                                                          | Stable prompt hash → built-in cache backend             |
| Eval reproducibility         | Varies with temperature, prompt phrasing, chat history       | Same `input + schema + few_shots` → identical cache key |


Concretely: when the schema is provided as a **tool-input schema** (Claude /
Anthropic-compatible models), the model only has to *fill in known slots*
— the same thing that makes function-calling reliable on Claude Haiku also
makes `@agentic_function` reliable on small models. The framework absorbs
the variability so the **per-call success rate of a 7B-class model is
comparable to a frontier model on the same task.**

In practice this means:

- **Dev / test loop:** use `MockBackend` → zero cost, deterministic.
- **Staging / CI:** use a cheap open model (DeepSeek-V4-Flash, Qwen 7B,
Llama-3.1-8B, Claude Haiku, GPT-4o-mini). Schema validation catches
drift.
- **Production:** swap to a larger model only where you observe real
quality issues — measured through the framework's `metrics` and the
per-call `error` field on retry exhaustion.

A 3-tier cost-quality matrix that works out of the box:

```python
# dev: zero cost, deterministic
set_default_backend(MockBackend())

# staging / cheap eval: open 7B class
@agentic_function(backend="openai", model="deepseek-v4-flash",
                  output_schema={"label": str, "score": float})
def classify(text): ...

# production: promote only after metrics show the cheap model underperforms
@agentic_function(backend="anthropic", model="claude-3-5-sonnet-latest",
                  output_schema={"label": str, "score": float})
def classify(text): ...
```

The function body and callers don't change — that's the point.

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
    mock_llm(output={"label": "positive", "score": 0.95})
    out = classify("amazing launch!")
    assert out.label == "positive"
    assert out.score == pytest.approx(0.95)
```

`mock_llm()` is one line — no HTTP, no API key, no asyncio loop ceremony.

### 2. Deterministic tests with `MockBackend`

```python
from agentic_function.backends.mock_backend import MockBackend
from agentic_function import set_default_backend

set_default_backend(MockBackend())  # every call returns a schema-shaped dict
```



### 3. Behavioural tests with fixture sequences

```python
backend = MockBackend()
backend.register_with_schema(MyOutputModel)
set_default_backend(backend)
```



### 4. Async-path tests are first-class

```python
@pytest.mark.asyncio
async def test_async_classify():
    mock_llm(output={"label": "negative", "score": 0.8})
    out = await classify.acall("terrible")
    assert out.label == "negative"
```



### 5. Backend tests with mocked SDK clients

The `AnthropicBackend` / `OpenAIBackend` tests demonstrate how to patch
`anthropic.Anthropic` / `openai.OpenAI` to verify **request shaping**
(temperature, max_tokens, tool schemas, timeouts, base URLs) without
hitting the network. See `tests/test_anthropic_backend.py` for 29 examples.

### 6. Run the suite

```bash
pytest tests/                           # 101 tests, <1s
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

The full test suite (101 tests) runs in **<1 second** because nothing
in `tests/` hits the network.

### Per-call tracing (nested spans via `contextvars`)

```python
from agentic_function import trace

with trace("nightly_eval") as ctx:
    for sample in dataset:
        out = classify(sample.text)
        ctx.span.set_attribute("sample.id", sample.id)

# Spans nest naturally; integrate with OpenTelemetry by passing your own
# TraceRecorder (see runtime/trace.py).
```



### Cost control in production

- **Cache hit** is the single biggest win: identical `(model, schema, few_shots, prompt_hash)` keys return cached output in microseconds.
- **Retry policy** is bounded: `RetryPolicy(max_retries=2, base_delay=0.5, max_delay=4.0)` stops runaway loops.
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

Callers and the function body never change. Roll forward / roll back is
literally a model-string swap.

### Project layout

```
agentic-function/
├── agentic_function/
│   ├── core/           # decorator, schema, prompt, result
│   ├── backends/       # mock / openai / anthropic / minimax
│   ├── runtime/        # executor, cache, retry, trace, metrics, config
│   ├── composition/    # as_tool, registry
│   ├── validation/     # coerce + validator
│   ├── utils/          # hashing, logging, cost
│   └── testing.py      # mock_llm
├── tests/              # 101 tests
├── examples/           # 6 examples (5 mock + 1 real LLM)
└── docs/               # additional docs
```

---



## Testability

The framework is **deliberately layered so that every concern can be
isolated in tests.** No layer knows about the layer below it concretely.


| Concern                         | How it's isolated for testing                               | What you can verify                              |
| ------------------------------- | ----------------------------------------------------------- | ------------------------------------------------ |
| **Schema validation**           | pydantic model alone                                        | Field types, ranges, `Literal` enums             |
| **Prompt rendering**            | `render_prompt(fn, args, kwargs)` standalone fn             | Exact message list sent to backend               |
| **Backend request shaping**     | Patch `anthropic.Anthropic` / `openai.OpenAI`               | Tool schemas, sampling params, timeouts, headers |
| **Backend response conversion** | Pass a synthetic `LLMResponse`                              | Dict vs raw string, token parsing, JSON repair   |
| **Retry policy**                | Synthetic `BackendError` / `ParseError` / `ValidationError` | `is_retryable()`, backoff timing, max attempts   |
| **Cache**                       | `InMemoryCache`, `DiskCache`, `NullCache` are all swappable | Hit/miss, TTL, key stability                     |
| **Executor orchestration**      | Substitute any layer above with a stub                      | Order of operations, trace span lifecycle        |
| **Composition**                 | Mock an inner function call                                 | Outer function gets correctly typed inner result |
| **End-to-end with real LLM**    | `examples/06_real_minimax.py`                               | Real latency, tokens, cost, error recovery       |




### What `mock_llm()` actually does

```python
def mock_llm(*, output=None, side_effect=None, schema=None) -> MockRegistration:
    """Register a canned response on the default MockBackend for the next call.
    Returns a handle you can .reset() to clear early."""
```

This means a unit test like:

```python
def test_retry_on_schema_mismatch():
    call_count = {"n": 0}
    def side_effect(req):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"label": "INVALID"}      # bad value
        return {"label": "positive", "score": 0.9}
    mock_llm(side_effect=side_effect)
    out = classify("text")
    assert call_count["n"] == 2              # framework retried
    assert out.label == "positive"
```

can verify retry behaviour without ever calling the network — **the
behavioural contract is testable independently of any LLM provider.**

### CI-friendly patterns

```yaml
# .github/workflows/test.yml
- run: pytest tests/ --tb=short            # always runs, <1s, no secrets
- run: pytest tests/test_*.py -m "not live" # opt-in live tests skipped
```

Mark live-network tests with `@pytest.mark.live` and skip them in CI; the
default suite is hermetic.

---



## Features


|                                       |                                                                   |
| ------------------------------------- | ----------------------------------------------------------------- |
| 🎯 **Decorator-first API**            | `@agentic_function` — no new concepts                             |
| 📋 **Schema-as-decorator-arg**        | Dict or `BaseModel` subclass — both work                          |
| 🔄 **Composition**                    | Agentic Functions call Agentic Functions                          |
| 🧪 `mock_llm()`                       | One-liner to turn tests into pure unit tests                      |
| 📊 **Per-call metrics**               | latency, tokens, cost, retries, cache hits                        |
| 🔌 **Pluggable backends**             | `MockBackend`, `OpenAIBackend`, or write your own                 |
| 💾 **Caching**                        | In-memory, disk, or null — keyed by stable hash                   |
| 🛡️ **Validation + retry**            | Auto-retry on schema mismatch / parse errors                      |
| 🌊 **Async-first**                    | `.acall()` is primary; `__call__` is a thin sync wrapper          |
| 🌳 **Tracing**                        | `contextvars`-based nested spans                                  |
| 🔧 **OpenAI / Anthropic tool export** | Drop any agentic function into an Agent loop                      |
| 🔁 **Retry policy**                   | Configurable backoff, max attempts, transient vs permanent errors |


---



## Installation

```bash
pip install agentic-function          # coming soon
pip install -e .                      # current dev install
```



## Quick start

```python
from agentic_function import agentic_function, AgenticResult, set_default_backend
from agentic_function.backends.mock_backend import MockBackend
from agentic_function.testing import mock_llm

# 1. Choose your backend.
set_default_backend(MockBackend())    # or OpenAIBackend() once you have a key

# 2. Pre-register a canned response (only needed for tests).
mock_llm(output={"category": "positive", "confidence": 0.94, "reasoning": "..."})

# 3. Declare the function. The docstring is the prompt.
@agentic_function(output_schema={"category": str, "confidence": float, "reasoning": str})
def classify_sentiment(text: str) -> AgenticResult:
    """Classify the sentiment of ``text``."""

# 4. Call it.
result = classify_sentiment("Amazing launch today!")
print(result.category, result.confidence, result.reasoning)
print(result.metrics.latency_ms, result.metrics.usage.prompt_tokens)
```

Run the runnable examples in `[examples/](examples/)`:

```bash
python examples/01_sentiment_classification.py
python examples/02_information_extraction.py
python examples/03_summarization.py
python examples/04_intent_routing.py
python examples/05_composition.py
```

None of them require an API key — they all use `MockBackend`.

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
| `executor`                                   | global default                                    | Custom `Executor` instance                                 |


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
    CallMetrics, TokenUsage,
    RetryPolicy, default_retry_policy,
    CacheBackend, InMemoryCache, DiskCache, NullCache,
    get_default_executor, set_default_executor,
    # composition
    FunctionRegistry, get_global_registry, register, get_function,
    as_openai_tool, as_anthropic_tool,
    # testing
    testing,                     # sub-module — `agentic_function.testing.mock_llm`
    # errors
    AgenticFunctionError, BackendError, CacheError, CompositionError,
    ConfigError, ParseError, RegistrationError, RetryExhaustedError,
    SchemaError, TimeoutError_ as TimeoutError, ValidationError,
)
```

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
```

Each layer is independently testable:

- **Executor** — orchestrates everything
- **Trace** — `contextvars`-based nested spans
- **Cache** — pluggable (`InMemory`, `Disk`, `Null`)
- **Retry** — `RetryPolicy` + `is_retryable` registry
- **Schema** — pydantic-driven validation + coercion
- **Backend** — `LLMBackend` subclass (provider adapter)

---



## Roadmap

- [x] v0.1 — `@agentic_function` decorator + pydantic schema validation
- [x] v0.2 — pluggable backends (`Mock`, `OpenAI`)
- [x] v0.3 — composition (function-as-function)
- [x] v0.4 — caching + cost tracking + `mock_llm()` testing helper
- [x] v0.5 — async + tracing + `as_openai_tool` / `as_anthropic_tool`
- [ ] v0.6 — Anthropic + Ollama adapters (placeholders in place)
- [ ] v0.7 — streaming output
- [ ] v1.0 — stable API + complete docs

---



## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports, ideas, docs improvements
and PRs all welcome.

## License

[MIT](LICENSE)