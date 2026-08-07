# agentic-function

用 `@agentic_function` 把 LLM 能力写成普通 Python 函数：在 docstring 中描述任务，用 schema 约束输出；框架负责 prompt 渲染、校验、重试、缓存、追踪与多后端调用。

**语言：** [English](README.md) · [中文](README.zh.md)

- 许可证：[MIT](LICENSE)
- Python：3.10+
- 状态：Alpha（`0.0.1a0`）· [PyPI](https://pypi.org/project/agentic-function/)

---

## 安装

当前 PyPI 版本为预发布版 `0.0.1a0`。pip 默认不安装 alpha / beta，因此不能只写 `pip install agentic-function`，否则会装不上或跳过该版本。预发布版 API 仍可能变更，请按需显式安装：

```bash
pip install --pre agentic-function
# 或锁定版本
pip install agentic-function==0.0.1a0
```

可选依赖（厂商 SDK）：

```bash
pip install --pre "agentic-function[openai]"
pip install --pre "agentic-function[anthropic]"
pip install --pre "agentic-function[openai,anthropic]"
```

从源码开发：

```bash
pip install -e ".[dev,openai,anthropic]"
```

---

## 快速开始

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
    """对 ``text`` 做情感分类。

    - ``category``: "positive" | "negative" | "neutral"
    - ``confidence``: [0.0, 1.0]
    - ``reasoning``: 简短说明
    """

result = classify_sentiment("Amazing launch today!")
print(result.category, result.confidence, result.reasoning)
print(result.metrics.latency_ms, result.metrics.usage.prompt_tokens)
```

示例脚本（`examples/`）：

```bash
python examples/01_sentiment_classification.py
python examples/02_information_extraction.py
python examples/03_summarization.py
python examples/04_intent_routing.py
python examples/05_composition.py
python examples/06_real_minimax.py   # 需要 API key
```

`01`–`05` 使用 `MockBackend`，无需密钥。

---

## 功能概览

| 能力 | 说明 |
| ---- | ---- |
| 装饰器 API | `@agentic_function`，调用方式与普通函数相同 |
| 输出 Schema | 支持 `dict` / pydantic `BaseModel` / `Literal[...]` |
| 函数组合 | 普通 Python 调用即可串联 |
| 工具导出 | `as_openai_tool` / `as_anthropic_tool`、`FunctionRegistry` |
| 后端 | Mock、OpenAI、Anthropic、MiniMax，以及自定义 `register_backend` |
| 校验与重试 | pydantic 校验；解析 / 校验失败可按策略重试 |
| 缓存 | `InMemoryCache` / `DiskCache` / `NullCache` |
| 指标与成本 | 每次调用返回 `CallMetrics`（延迟、token、成本估算等） |
| 追踪与预算 | `trace`、`BudgetTracker`、`Aggregator`（含 Prometheus 文本） |
| 诊断 | `diagnose` / `explain_failure` / `snapshot`，`debug=` / `AGENTIC_DEBUG` |
| 测试辅助 | `mock_llm`、`mock_llm_table`、`freeze_time`、`capture_metrics` |
| 异步 | `.acall()` 为主路径；同步 `__call__` 可用 |
| 错误类型 | `ValidationError`、`RetryExhaustedError`、`BudgetExceededError` 等 |

---

## 核心用法

### Schema 与结构化输出

`output_schema` 可用 `dict`、`BaseModel` 或返回注解中的 `Literal[...]`。框架会将 JSON schema 注入 prompt（或走厂商 tool / json_schema 模式），做常见类型强制转换，并用 pydantic 校验；失败时按重试策略再次请求。调用方拿到的是字段属性，而不是未解析的原始字符串。

```python
@agentic_function(output_schema={"label": str, "score": float})
def classify(text: str) -> AgenticResult:
    """对 ``text`` 做情感分类。"""
```

### 组合

```python
topic = extract_topic(article)
summary = make_summary(article, topic.topic, topic.tone)
```

完整示例见 [`examples/05_composition.py`](examples/05_composition.py)。

### 导出为工具

可将函数导出为 OpenAI / Anthropic 工具 JSON，供外部 Agent 或自建 tool loop 调用：

```python
from agentic_function import as_openai_tool, as_anthropic_tool, register, get_function

openai_tool = as_openai_tool(make_summary)
anthropic_tool = as_anthropic_tool(make_summary)

register(make_summary)
fn = get_function(make_summary.qualified_name)
```

### 后端切换

内置 `MockBackend`、`OpenAIBackend`、`AnthropicBackend`，以及面向 MiniMax-CN 的 `minimax` 预设。自定义后端通过 `register_backend(...)` 注册。

```python
@agentic_function(backend="mock", output_schema={"label": str})
@agentic_function(backend="openai", model="gpt-4o-mini", output_schema={"label": str})
@agentic_function(backend="anthropic", model="claude-sonnet-4-20250514", output_schema={"label": str})
@agentic_function(backend="minimax", model="MiniMax-M3", output_schema={"label": str})
```

### Prompt 相关参数

| 参数 | 用途 |
| ---- | ---- |
| docstring | 任务描述（系统提示主体） |
| `few_shots` | `[(input, output), …]` 示例轮次 |
| `prompt_template` / `system_template` | 自定义 `{placeholder}` 模板 |
| `include_schema_in_prompt` | 是否注入 schema |
| `description` | 工具导出说明（默认取 docstring 首行） |
| `render_prompt(fn, args, kwargs)` | 调用前查看将发送的消息列表 |

### 异步

```python
out = await classify.acall("terrible")
```

支持自由函数、实例方法与类方法（描述符协议）。

---

## 指标、追踪与成本

每次调用结果包含 `CallMetrics`：

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

追踪、预算与聚合示例：

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

诊断：

```python
from agentic_function import diagnose, explain_failure, snapshot

print(diagnose(result).to_dict())
print(explain_failure(exc))
print(snapshot(result))
```

缓存键覆盖 `(model, schema, few_shots, prompt_hash)`。重试可用 `RetryPolicy(max_retries=..., base_delay=..., max_delay=...)` 限制次数与退避。

`examples/06_real_minimax.py` 可对真实后端做一次联调（需配置 API key）。默认 `tests/` 不访问网络。

---

## 测试

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

也可使用 `MockBackend`，或 patch `openai` / `anthropic` 客户端验证请求参数（见 `tests/test_anthropic_backend.py`）。

```bash
pytest tests/
pytest tests/test_anthropic_backend.py
pytest --cov=agentic_function
```

分层隔离方式：

| 关注点 | 测试方式 |
| ------ | -------- |
| Schema | 直接测 pydantic 模型 |
| Prompt | `render_prompt(fn, args, kwargs)` |
| 后端请求 | Patch 厂商 SDK |
| 后端响应 | 传入合成 `LLMResponse` |
| 重试 | 合成可重试异常 |
| 缓存 | 替换 cache 实现 |
| 工具导出 | 断言 tool JSON 结构 |
| 指标 | `capture_metrics` / aggregator |

---

## 装饰器参数

| 参数 | 默认 | 含义 |
| ---- | ---- | ---- |
| `model` | 全局默认 | 传给后端的模型名 |
| `output_schema` | 从返回注解推断 | `dict` / `BaseModel` / `Literal[...]` |
| `backend` | `set_default_backend(...)` | 实例或已注册名称 |
| `temperature`, `top_p`, `max_tokens`, `stop` | 全局配置 | 采样参数 |
| `max_retries` | 全局配置 | 解析 / 校验失败重试次数 |
| `retry_policy` | `RetryPolicy(...)` | 退避与可重试异常 |
| `cache` | 全局默认 | 单次调用缓存覆盖 |
| `timeout` | 全局配置 | 请求超时（秒） |
| `include_schema_in_prompt` | `True` | 是否注入 JSON schema |
| `few_shots` | `[]` | 示例对 |
| `prompt_template` / `system_template` | `None` | 自定义模板 |
| `description` | docstring 首行 | 工具导出说明 |
| `debug` | `False` / `AGENTIC_DEBUG` | 附加请求/响应快照 |
| `executor` | 全局默认 | 自定义 `Executor` |

环境变量：`AGENTIC_FUNCTION_MODEL`、`AGENTIC_FUNCTION_BACKEND`、`AGENTIC_FUNCTION_CACHE`、`AGENTIC_FUNCTION_CACHE_DIR`，以及 `OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`MINIMAX_CN_API_KEY` 等。

---

## 公开 API

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

`AnthropicBackend` 与 MiniMax 位于 `agentic_function.backends`，注册名为 `"anthropic"` / `"minimax"`。

---

## 架构

```
@agentic_function(...)
        │
        ▼
  AgenticFunction  (描述符 / call / await)
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

## 项目结构

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

## 路线图

- [x] v0.1 — 装饰器 + pydantic schema 校验
- [x] v0.2 — 可插拔后端（Mock、OpenAI）
- [x] v0.3 — 函数组合
- [x] v0.4 — 缓存、成本、`mock_llm`
- [x] v0.5 — 异步、追踪、工具导出、Anthropic / MiniMax、预算 / 聚合 / 诊断
- [ ] v0.6 — Ollama 适配与更完整的流式能力
- [ ] v0.7 — 流式输出公开 API
- [ ] v1.0 — 稳定 API 与完整文档

---

## 贡献

见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

[MIT](LICENSE)
