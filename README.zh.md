# agentic-function

> **由 LLM 驱动的类型安全 Python 函数。**
> 用 `@agentic_function` 声明函数，在 docstring 里描述任务，声明输出 schema —
> 框架负责 prompt 渲染、schema 校验、重试、缓存、追踪与多后端执行。

**语言：** [English](README.md) · [中文](README.zh.md)

[License: MIT](LICENSE)
[Python 3.10+](https://www.python.org/downloads/)
[Status: Alpha `0.0.1a0`](https://pypi.org/project/agentic-function/)

---

## 安装（Alpha 版 — 请先读）

> **当前 PyPI 上只有 alpha 预发布版：`0.0.1a0`。**
> pip **默认不会安装** alpha / beta 等预发布版本，因此直接执行
> `pip install agentic-function` 会装不上或跳过该版本，直到出现正式版。
> 这是 pip 的保护机制：避免生产环境在不知情时装上 API 仍可能变更的预发布包。

**安装当前 alpha（二选一）：**

```bash
# 推荐：显式允许预发布版
pip install --pre agentic-function

# 或锁定确切的 alpha 版本号
pip install agentic-function==0.0.1a0
```

**需要厂商 SDK 时：**

```bash
pip install --pre "agentic-function[openai]"
pip install --pre "agentic-function[anthropic]"
pip install --pre "agentic-function[openai,anthropic]"
```

需要 **Python 3.10+**。从源码做本地开发：

```bash
pip install -e ".[dev,openai,anthropic]"
```

---

## 什么是 Agentic Function？

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
    """对 ``text`` 做情感分类。

    - ``category``: "positive" | "negative" | "neutral" 之一
    - ``confidence``: [0.0, 1.0] 之间的浮点数
    - ``reasoning``: 简短说明（建议 ≤ 80 字符）
    """

result = classify_sentiment("The launch today was absolutely amazing!")
result.category    # → "positive"
result.confidence  # → 0.94
result.reasoning   # → "Strong positive adjectives, exclamation mark."
result.metrics.latency_ms
result.metrics.usage.prompt_tokens
```

调用方式与普通 Python 函数一致 — 函数体是 docstring/prompt，实现则是
「任意能匹配声明 schema 的 LLM」。装饰器把其余链路全部接好。

---

## 框架优势 — 你能拿到的全部能力

多数 LLM 代码是胶水：拼 prompt、脆弱的 JSON 解析、手写重试，再叠一堆
厂商 SDK。**Agentic Function 把这些胶水收成一个有类型的 Python 函数**，
并把校验、成本、缓存、工具导出、测试等生产关切做成一等公民，而不是作业。

### 1. 装饰器优先 — 零新概念

一个装饰器即可。不必学 Agent / Tool / Chain / Memory 术语。
会写 Python 函数，就能交付一个 LLM 步骤。

```python
@agentic_function(output_schema={"label": str, "score": float})
def classify(text: str) -> AgenticResult:
    """对 ``text`` 做情感分类。"""
```

### 2. Schema 强制的结构化输出

契约只声明一次 — 用 `dict`、pydantic `BaseModel`，或 `Literal[...]` 返回注解。
框架会：

- 把 JSON schema 注入 prompt（或走厂商 tool / json_schema 模式）
- 纠正常见 LLM 输出毛刺（`"0.9"` → `0.9`、CSV → `list`）
- 用 pydantic 校验
- 在解析 / 校验失败时自动重试

调用方始终拿到带类型的属性，而不是还要自己解析的字符串。

### 3. 纯 Python 组合

Agentic Function 之间像普通函数一样互相调用。没有编排 DSL。
流水线可读、可单测。

```python
topic = extract_topic(article)
summary = make_summary(article, topic.topic, topic.tone)
```

参见 [`examples/05_composition.py`](examples/05_composition.py)。

### 4. 一键导出为 Agent 工具（`as_tool`）

需要多步规划？规划器继续放在 Agent 框架里 — 每个 Agentic Function
用一行就能导出为原生 OpenAI / Anthropic 工具。你得到的是**带类型、
带校验、带计量的工具**，不必为 Agent 重写一遍。

```python
from agentic_function import as_openai_tool, as_anthropic_tool, register, get_function

# OpenAI Agents / LangChain / 自建 tool loop
openai_tool = as_openai_tool(make_summary)
# → {"type": "function", "function": {"name": ..., "parameters": {...}}}

# Anthropic tool_use
anthropic_tool = as_anthropic_tool(make_summary)
# → {"name": ..., "description": ..., "input_schema": {...}}

# Agent 调度时的动态查找
register(make_summary)
fn = get_function(make_summary.qualified_name)
```

这是刻意设计的桥梁：**单步可靠用 Agentic Function；多步规划用 Agent 框架** —
同一批函数，两套场景。

### 5. 可插拔后端 — 改一个参数即可切换

内置：`MockBackend`、`OpenAIBackend`、`AnthropicBackend`，以及 `minimax`
（面向 MiniMax-CN 的 Anthropic 协议预设）。可通过 `register_backend(...)`
注册自定义后端。

```python
@agentic_function(backend="mock", output_schema={"label": str})          # 本地
@agentic_function(backend="openai", model="gpt-4o-mini", ...)            # 预发
@agentic_function(backend="anthropic", model="claude-3-5-sonnet-latest") # 生产
@agentic_function(backend="minimax", model="MiniMax-M3", ...)            # MiniMax-CN
```

调用方与函数体完全不用改。

### 6. 小模型 / 便宜模型也能稳

因为契约是**填满已知槽位的 schema**（常走 tool-input / json_schema），
7B 级模型只需填空 — 这也是 Claude Haiku 函数调用可靠的原因。框架吸收
格式抖动，便宜模型上的成功率也能保持高位。

| 层级 | 建议 |
| ---- | ---- |
| 开发 / 单测 | `MockBackend` + `mock_llm()` — 零成本、确定性 |
| 预发 / CI | DeepSeek / Qwen 7B / Llama-3.1-8B / Haiku / GPT-4o-mini |
| 生产 | 仅在 metrics 证明便宜模型不够用时再升配 |

### 7. 每次调用自带生产级指标与成本

每个结果都带 `CallMetrics` — 延迟、token、美元成本、缓存命中、**attempts / retries**、
分阶段耗时 — 无需额外埋点。**重试次数是模型单测 / 评测的一等指标。**

```python
result.metrics.latency_ms
result.metrics.usage.total_tokens
result.metrics.cost_usd
result.metrics.cache_hit
result.metrics.attempts         # 实际后端调用次数
result.metrics.retries          # attempts - 1（0 = 一次成功）
result.metrics.recovered        # 仅在重试后才成功
result.metrics.attempt_errors   # [{attempt, category, error_type, message}, …]
result.metrics.timings          # PhaseTimings：prompt / backend / validate / …
result.metrics.total_cost_usd   # 含重试累计
```

```python
from agentic_function import Aggregator, install_default_aggregator
from agentic_function.testing import capture_metrics, eval_summary
from agentic_function.errors import RetryExhaustedError

# 套件级评测：retry_rate / recovery_rate / errors_by_category
with capture_metrics() as bag:
    for sample in dataset:
        try:
            classify(sample.text)
        except RetryExhaustedError as exc:
            assert exc.metrics.retries >= 1
            assert exc.error_category in {"validation", "parse", "backend"}

summary = eval_summary(bag)
assert summary["totals"]["retry_rate"] < 0.1   # 模型很少需要第二次机会
assert summary["errors_by_category"].get("validation", 0) == 0
```

### 8. 嵌套追踪、预算与 Prometheus 聚合

```python
from agentic_function import (
    trace,
    Budget, BudgetTracker, install_budget_tracker,
    Aggregator, install_default_aggregator,
)

with trace("nightly_eval") as ctx:
    out = classify(sample.text)
    ctx.span.set_attribute("sample.id", sample.id)

# 进程级花费 / 延迟 / token 上限
install_budget_tracker(BudgetTracker(budgets=[
    Budget(metric="cost_usd", limit=5.0),
]))

# 按函数统计 + Prometheus 文本
agg = install_default_aggregator(Aggregator())
# … 业务流量之后 …
print(agg.summary())
print(agg.to_prometheus())
```

### 9. 说人话的诊断信息

预发环境出问题，拿到的是结构化解释，而不是原始 SDK 堆栈。

```python
from agentic_function import diagnose, explain_failure, snapshot

print(diagnose(result).to_dict())       # success / cache_hit / failed 分解
print(explain_failure(exc))             # 类别、尝试次数、可操作提示
print(snapshot(result))                 # 可落日志的字典
```

装饰器加 `debug=True`，或设置环境变量 `AGENTIC_DEBUG=1`，可附带
请求/响应快照。

### 10. 真能省钱的缓存

缓存键稳定覆盖 `(model, schema, few_shots, prompt_hash)`。
可切换 `InMemoryCache` / `DiskCache` / `NullCache`。相同调用微秒级返回，
并标记 `metrics.cache_hit = True`。

### 11. Async 优先，同步也好用

`.acall()` 是主路径；`__call__` 是薄同步封装。
通过描述符协议支持自由函数、实例方法与类方法。

```python
out = await classify.acall("terrible")
```

### 12. 不需要模板引擎的 Prompt 能力

| 旋钮 | 用途 |
| ---- | ---- |
| docstring | 系统提示（任务描述） |
| `few_shots=[(in, out), …]` | 上下文示例，写成 chat 轮次 |
| `prompt_template` / `system_template` | 自定义 `{placeholder}` 格式化 |
| `include_schema_in_prompt` | 是否注入 schema |
| `description` | 工具导出说明（默认取 docstring 首行） |
| `render_prompt(fn, args, kwargs)` | 调用前检查将发给后端的完整消息列表 |

### 13. 无需联网的可测性

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
    r = classify("text")          # 确定性 PhaseTimings

with capture_metrics() as bag:
    classify("a"); classify("b")
assert len(bag) == 2
```

也可通过 patch 厂商 SDK 单测请求塑形（见 `tests/test_anthropic_backend.py`）。

### 14. 类型化错误体系

按意图捕获：`ValidationError`、`ParseError`、`BackendError`、
`RetryExhaustedError`、`BudgetExceededError`、`CacheError`、
`CompositionError`、`TimeoutError_`，……

---

## 为什么不用通用 Agent 框架？

多数「AI 能力」其实是**单步**：分类、抽取、摘要、路由、改写、打分。
这些都不需要多步 Agent 运行时。硬套通用 Agent 框架成本过高：

| 维度 | 通用 Agent 框架 | Agentic Function |
| ---- | --------------- | ---------------- |
| 上手成本 | 要学 Agent / Tool / Chain / Memory | 一个装饰器 |
| 类型安全 | 输出常常是字符串 | 入参有注解，出参按 schema 校验 |
| 结构化输出 | 手写解析，经常炸 | 框架强制（校验 + 重试） |
| 组合 | 受框架协议约束 | 普通 Python 函数互调 |
| Agent 集成 | 把步骤重写成 tool | `as_openai_tool` / `as_anthropic_tool` 一行搞定 |
| 单元测试 | 要 mock 整套 Agent 运行时 | `mock_llm()` 一行纯单测 |
| 可观测性 | 框架内部 trace | Metrics + spans + 预算 + Prometheus |
| 学习曲线 | 陡 | 平 — 就是函数 |

**需要两者时就一起用。** 把 Agentic Function 当作可靠的单步积木；
只有真正需要多步规划时，再用 `as_*_tool` 暴露给 Agent 规划器。

---

## 测试

`agentic-function` 在每一层都为可测性做了设计。

### 1. 用 `mock_llm()` 做纯单元测试

```python
from agentic_function import agentic_function, AgenticResult
from agentic_function.testing import mock_llm

@agentic_function(output_schema={"label": str, "score": float})
def classify(text: str) -> AgenticResult:
    """情感分类。 - label: positive|negative|neutral"""

def test_classify_positive():
    mock_llm({"label": "positive", "score": 0.95})
    out = classify("amazing launch!")
    assert out.label == "positive"
    assert out.score == pytest.approx(0.95)
```

### 2. 用 `MockBackend` 做确定性测试

```python
from agentic_function.backends.mock_backend import MockBackend
from agentic_function import set_default_backend

set_default_backend(MockBackend())  # 每次调用返回符合 schema 形状的 dict
```

### 3. 用 `mock_llm_table` 做多调用夹具

```python
from agentic_function.testing import mock_llm_table

mock_llm_table([
    {"topic": "OpenAI", "tone": "informational"},
    {"summary": "...", "tags": ["ai"], "score": 0.9},
])
```

### 4. 异步路径一等公民

```python
@pytest.mark.asyncio
async def test_async_classify():
    mock_llm({"label": "negative", "score": 0.8})
    out = await classify.acall("terrible")
    assert out.label == "negative"
```

### 5. 用 mock SDK 客户端测后端

`AnthropicBackend` / `OpenAIBackend` 测试展示了如何 patch
`anthropic.Anthropic` / `openai.OpenAI`，在不联网的情况下验证**请求塑形**
（temperature、max_tokens、tool schema、超时、base URL）。
见 `tests/test_anthropic_backend.py`。

### 6. 跑测试套件

```bash
pytest tests/                           # 全量，<1s，无网络
pytest tests/test_anthropic_backend.py  # 后端专项
pytest --cov=agentic_function           # 带覆盖率
```

---

## 量化与可观测性

每次调用都返回 `CallMetrics` — 开箱即用的生产遥测，无需额外配置。

```python
result = classify("amazing launch!")

result.metrics.latency_ms                  # 端到端墙钟时间
result.metrics.usage.prompt_tokens         # 输入 token
result.metrics.usage.completion_tokens     # 输出 token
result.metrics.usage.total_tokens
result.metrics.cost_usd                    # 美元成本（按价目表估算）
result.metrics.cache_hit                   # True / False
result.metrics.attempts                    # 通常为 1，重试时更大
result.metrics.retries                     # 重试次数
result.metrics.successful                  # True / False
result.metrics.error                       # 若失败，最后一次错误信息
result.metrics.timings                     # PhaseTimings 分解
result.metrics.extra                       # 后端特有扩展字段
```

### 真实 `minimax`（MiniMax-M3，Anthropic 协议）调用数字

以下来自 `examples/06_real_minimax.py` 对线上 LLM 的实测（无缓存）：

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

完整测试套件在 **<1 秒** 内跑完，因为 `tests/` 不访问网络。

### 生产环境成本控制

- **缓存命中**是最大赢家：相同 `(model, schema, few_shots, prompt_hash)` 微秒级返回。
- **重试策略有上界**：`RetryPolicy(max_retries=2, base_delay=0.5, max_delay=4.0)` 防止失控循环。
- **BudgetTracker** 在花费 / 延迟 / token 突破天花板之前拦截。
- 每次调用的 `cost_usd` 可直接做功能级损益账。

---

## 开发工作流

### 迭代循环

```bash
# 1. 用 MockBackend 编写 / 修改函数 — 即时反馈
python -c "from agentic_function import *; ..."

# 2. 跑单元测试
pytest tests/ -x

# 3. 跑真实 LLM 示例（便宜、带 schema 校验）
set -a && source ~/.hermes/.env && set +a
python examples/06_real_minimax.py

# 4. 质量达标后再升到生产模型
```

### 后端切换只改装饰器一个参数

```python
# 开发
@agentic_function(backend="mock", output_schema={"label": str})

# 预发
@agentic_function(backend="openai", model="gpt-4o-mini",
                  output_schema={"label": str})

# 生产
@agentic_function(backend="anthropic", model="claude-3-5-sonnet-latest",
                  output_schema={"label": str})
```

### 项目结构

```
agentic-function/
├── agentic_function/
│   ├── core/           # 装饰器、schema、prompt、result
│   ├── backends/       # mock / openai / anthropic / minimax
│   ├── runtime/        # executor、cache、retry、trace、metrics、budget、aggregator、diagnostics
│   ├── composition/    # as_tool、registry
│   ├── validation/     # coerce + validator
│   ├── utils/          # hashing、logging、cost
│   └── testing.py      # mock_llm、mock_llm_table、freeze_time、…
├── tests/
├── examples/           # 5 个 mock + 1 个真实 LLM
└── docs/
```

---

## 可测性分层

框架刻意分层，使每一层关注点都能单独测。

| 关注点 | 如何隔离测试 | 能验证什么 |
| ------ | ------------ | ---------- |
| **Schema 校验** | 单独测 pydantic 模型 | 字段类型、范围、`Literal` 枚举 |
| **Prompt 渲染** | 独立函数 `render_prompt(fn, args, kwargs)` | 发给后端的精确消息列表 |
| **后端请求塑形** | Patch `anthropic.Anthropic` / `openai.OpenAI` | Tool schema、采样参数、超时、headers |
| **后端响应转换** | 传入合成 `LLMResponse` | Dict vs 原始字符串、token 解析、JSON 修复 |
| **重试策略** | 合成 `BackendError` / `ParseError` / `ValidationError` | `is_retryable()`、退避时序、最大次数 |
| **缓存** | `InMemoryCache` / `DiskCache` / `NullCache` 可替换 | 命中/未命中、TTL、键稳定性 |
| **Executor 编排** | 上层用 stub 替换 | 操作顺序、trace span 生命周期 |
| **组合 / as-tool** | Mock 内层调用；断言 tool JSON 形状 | 带类型的内层结果；OpenAI/Anthropic 工具 schema |
| **可观测性** | `capture_metrics`、`freeze_time`、aggregator | 延迟分桶、成本合计、Prometheus 文本 |
| **真实 LLM 端到端** | `examples/06_real_minimax.py` | 真实延迟、token、成本、错误恢复 |

### CI 友好模式

```yaml
# .github/workflows/test.yml
- run: pytest tests/ --tb=short            # 始终跑，<1s，无密钥
- run: pytest tests/test_*.py -m "not live" # 可选联网测试跳过
```

联网测试用 `@pytest.mark.live` 标记并在 CI 跳过；默认套件是封闭的。

---

## 能力一览

| 优势 | 能力 |
| ---- | ---- |
| 🎯 装饰器优先 API | `@agentic_function` — 零新概念 |
| 📋 灵活 schema | `dict` / `BaseModel` / `Literal[...]` |
| 🔄 纯 Python 组合 | 函数调函数 |
| 🔧 Agent 工具导出 | `as_openai_tool` / `as_anthropic_tool` + `FunctionRegistry` |
| 🔌 多后端 | Mock · OpenAI · Anthropic · MiniMax · 自定义 |
| 🛡️ 校验 + 强制转换 + 重试 | schema 不匹配不会悄悄上线 |
| 💾 可插拔缓存 | 内存 / 磁盘 / 空实现 |
| 📊 每次调用指标与成本 | 延迟、token、美元、分阶段耗时 |
| 🌳 嵌套追踪 | `contextvars` spans，可接 OTel |
| 💰 预算天花板 | 进程级成本 / 延迟 / token 限制 |
| 📈 聚合器 | 按函数统计 + `to_prometheus()` |
| 🩺 诊断 | `diagnose` / `explain_failure` / `snapshot` / `debug=` |
| 🧪 测试助手 | `mock_llm`、`mock_llm_table`、`freeze_time`、`capture_metrics` |
| 🌊 Async 优先 | `.acall()` 为主；同步 `__call__` 封装 |
| 📝 Prompt 旋钮 | few-shots、模板、schema 注入 |
| ⚠️ 类型化错误 | 捕获 `ValidationError`、`RetryExhaustedError` 等 |

---

## 安装

详见文首 **[安装（Alpha 版 — 请先读）](#安装alpha-版--请先读)**。
摘要：当前版本为 `0.0.1a0`；请用 `pip install --pre agentic-function`
（或锁定 `==0.0.1a0`）。直接 `pip install agentic-function` 不会装到 alpha。

---

## 快速开始

```python
from agentic_function import agentic_function, AgenticResult, set_default_backend
from agentic_function.backends.mock_backend import MockBackend
from agentic_function.testing import mock_llm

# 1. 选择后端。
set_default_backend(MockBackend())    # 有密钥后可换 OpenAIBackend()

# 2. 预注册固定响应（仅测试需要）。
mock_llm({"category": "positive", "confidence": 0.94, "reasoning": "..."})

# 3. 声明函数。docstring 就是 prompt。
@agentic_function(output_schema={"category": str, "confidence": float, "reasoning": str})
def classify_sentiment(text: str) -> AgenticResult:
    """对 ``text`` 做情感分类。"""

# 4. 调用。
result = classify_sentiment("Amazing launch today!")
print(result.category, result.confidence, result.reasoning)
print(result.metrics.latency_ms, result.metrics.usage.prompt_tokens)
```

可运行示例见 [`examples/`](examples/)：

```bash
python examples/01_sentiment_classification.py
python examples/02_information_extraction.py
python examples/03_summarization.py
python examples/04_intent_routing.py
python examples/05_composition.py          # 组合 + as_openai_tool
python examples/06_real_minimax.py         # 真实 MiniMax（需要 API key）
```

`01`–`05` 使用 `MockBackend`，无需 API key。

---

## 装饰器参数速查

| 参数 | 默认值 | 含义 |
| ---- | ------ | ---- |
| `model` | 全局默认 | 传给后端的模型标识 |
| `output_schema` | 从返回注解推断 | `dict[str, type]`、`BaseModel` 子类，或 `Literal[...]` |
| `backend` | `None`（回退到 `set_default_backend(...)`） | `LLMBackend` 实例或已注册名称 |
| `temperature`, `top_p`, `max_tokens`, `stop` | 来自全局配置 | 采样参数 |
| `max_retries` | 来自全局配置 | 解析 / 校验错误时的自动重试 |
| `retry_policy` | `RetryPolicy(max_retries=...)` | 自定义退避 / 可重试异常 |
| `cache` | 全局默认 | 单次调用缓存覆盖 |
| `timeout` | 来自全局配置 | 单次请求超时（秒） |
| `include_schema_in_prompt` | `True` | 是否把 JSON schema 注入系统消息 |
| `few_shots` | `[]` | `[(input, output), …]` 示例 |
| `prompt_template` | `None` | 自定义用户消息模板（str-format 风格） |
| `system_template` | `None` | 自定义系统消息模板 |
| `description` | docstring 首行 | 用于工具导出 |
| `debug` | `False` / `AGENTIC_DEBUG` | 把请求/响应快照挂到 metrics |
| `executor` | 全局默认 | 自定义 `Executor` 实例 |

环境变量默认：`AGENTIC_FUNCTION_MODEL`、`AGENTIC_FUNCTION_BACKEND`、
`AGENTIC_FUNCTION_CACHE`、`AGENTIC_FUNCTION_CACHE_DIR`，以及厂商密钥
（`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`MINIMAX_CN_API_KEY` 等）。

---

## 公开 API

```python
from agentic_function import (
    # 核心
    agentic_function,            # 装饰器
    AgenticFunction,             # 装饰器返回的描述符类
    AgenticResult,               # 用户输出模型的标记基类
    DynamicResult,               # schema 为 dict 时的返回类型
    SchemaSpec, resolve_schema,  # schema 机制
    render_prompt,               # 为一次调用构建消息列表
    # 后端
    LLMBackend, LLMResponse, StreamChunk,
    MockBackend, OpenAIBackend,
    register_backend, get_backend, get_default_backend, set_default_backend,
    known_backends,
    # 运行时
    Executor, GlobalConfig, configure, global_config,
    TraceContext, TraceSpan, TraceRecorder, trace, get_current_trace,
    CallMetrics, TokenUsage, PhaseTimings,
    RetryPolicy, default_retry_policy,
    CacheBackend, InMemoryCache, DiskCache, NullCache,
    get_default_executor, set_default_executor,
    # 可观测性（v0.5）
    Budget, BudgetTracker, BudgetExceededError,
    install_budget_tracker, get_default_budget_tracker,
    Aggregator, FunctionStats,
    install_default_aggregator, get_default_aggregator,
    Diagnostic, diagnose, diagnose_metrics, explain_failure, snapshot,
    # 组合
    FunctionRegistry, get_global_registry, register, get_function,
    as_openai_tool, as_anthropic_tool,
    # 测试
    testing,                     # mock_llm、mock_llm_table、freeze_time、…
    # 错误
    AgenticFunctionError, BackendError, CacheError, CompositionError,
    ConfigError, ParseError, RegistrationError, RetryExhaustedError,
    SchemaError, TimeoutError_ as TimeoutError, ValidationError,
)
```

`AnthropicBackend` / MiniMax 位于 `agentic_function.backends`，并自动注册为
`"anthropic"` / `"minimax"`。

---

## 架构

```
                        ┌────────────────────────────┐
                        │   @agentic_function(...)   │
                        └─────────────┬──────────────┘
                                      │ 构建
                                      ▼
                        ┌────────────────────────────┐
                        │       AgenticFunction      │  (描述符，call/await)
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

每一层都可独立测试：

- **Executor** — 编排一切
- **Trace** — 基于 `contextvars` 的嵌套 span
- **Cache** — 可插拔（`InMemory` / `Disk` / `Null`）
- **Retry** — `RetryPolicy` + `is_retryable` 注册表
- **Schema** — pydantic 驱动的校验与强制转换
- **Backend** — `LLMBackend` 子类（厂商适配器）
- **Composition** — 工具导出 + 名称注册表，用于 Agent 桥接

---

## 路线图

- [x] v0.1 — `@agentic_function` 装饰器 + pydantic schema 校验
- [x] v0.2 — 可插拔后端（`Mock`、`OpenAI`）
- [x] v0.3 — 组合（函数调函数）
- [x] v0.4 — 缓存 + 成本追踪 + `mock_llm()` 测试助手
- [x] v0.5 — 异步 + 追踪 + `as_openai_tool` / `as_anthropic_tool` + Anthropic / MiniMax + 可观测性（预算 / 聚合 / 诊断）
- [ ] v0.6 — Ollama 适配器 + 更完整的流式能力
- [ ] v0.7 — 流式输出（公开 API）
- [ ] v1.0 — 稳定 API + 完整文档

---

## 贡献

见 [CONTRIBUTING.md](CONTRIBUTING.md)。欢迎提交缺陷报告、想法、文档改进与 PR。

## 许可证

[MIT](LICENSE)
