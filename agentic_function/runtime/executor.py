"""Executor — orchestrates the full call lifecycle.

For one ``@agentic_function`` invocation, the executor is responsible for:

1. Building the prompt messages from the function's template + bound args.
2. Checking the cache (optional, based on per-call config).
3. Calling the LLM backend.
4. Parsing + validating the raw response against the output schema.
5. Retrying with exponential backoff on transient failures.
6. Recording traces, metrics, and usage — including per-phase timings.
7. Returning a final ``AgenticResult`` (or raising the last exception).

Phase-level timings
-------------------
Every call records milliseconds spent in each phase so you can decompose
``latency_ms`` when something is slow. The phases are:

* ``prompt_render``   — string-format the docstring + bound args into messages
* ``cache_lookup``    — hash + read from the cache backend
* ``backend_call``    — the LLM HTTP round-trip (last successful attempt)
* ``parse_validate``  — JSON extraction + pydantic coercion
* ``cache_write``     — write-back to the cache backend
* ``retry_backoff``   — total time sleeping between retries
"""
from __future__ import annotations

import asyncio
import inspect
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..backends.base import LLMBackend, LLMResponse, StreamChunk
from ..core.function import AgenticFunction
from ..core.prompt import render_prompt
from ..core.result import AgenticResult, DynamicResult
from ..core.schema import SchemaSpec, resolve_schema
from ..errors import (
    BackendError,
    ParseError,
    RetryExhaustedError,
    TimeoutError_,
    ValidationError,
)
from ..utils.cost import estimate_cost
from ..utils.logging import get_logger, kv
from ..utils.hashing import stable_hash
from .cache import CacheBackend, CacheEntry, get_cache
from .config import global_config
from .metrics import CallMetrics, PhaseTimings, TokenUsage
from .retry import RetryPolicy, is_retryable, sleep_or_yield
from .trace import TraceContext, get_current_trace
from .aggregator import Aggregator, get_default_aggregator

_log = get_logger("executor")

# ----------------------------------------------------------------------
# Debug mode — controlled by env var or per-function ``debug=True``.
# When enabled, every call's CallMetrics carries a request/response
# snapshot (truncated + secrets-redacted) so post-mortem debugging is
# trivial: ``result.metrics.request_snapshot`` / ``response_snapshot``.
# ----------------------------------------------------------------------
_DEBUG_ENV = os.environ.get("AGENTIC_DEBUG", "").lower() in ("1", "true", "yes", "on")

# Maximum message-list length stored in a snapshot. 50 messages is enough
# for any reasonable function; longer messages are still sent to the LLM,
# just not snapshotted.
_SNAPSHOT_MAX_MESSAGES = 50
_SNAPSHOT_MAX_CHARS_PER_MESSAGE = 4000


def _is_debug(fn: AgenticFunction) -> bool:
    if _DEBUG_ENV:
        return True
    return bool(getattr(fn, "debug", False))


def _truncate(text: str, limit: int = _SNAPSHOT_MAX_CHARS_PER_MESSAGE) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... <truncated {len(text) - limit} chars>"


def _snapshot_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages[:_SNAPSHOT_MAX_MESSAGES]:
        snap = dict(m)
        content = snap.get("content")
        if isinstance(content, str):
            snap["content"] = _truncate(content)
        out.append(snap)
    if len(messages) > _SNAPSHOT_MAX_MESSAGES:
        out.append({"role": "system", "content": f"... <{len(messages) - _SNAPSHOT_MAX_MESSAGES} more messages truncated>"})
    return out


def _snapshot_response(content: Any) -> Any:
    """Snapshot the response in a way that's safe to JSON-serialise.

    Strings are truncated; dicts/lists are kept verbatim but only the first
    4 KB of any string value. Pydantic models are dumped to ``dict``.
    """
    if isinstance(content, str):
        return _truncate(content, _SNAPSHOT_MAX_CHARS_PER_MESSAGE)
    if isinstance(content, dict):
        return {k: _truncate(v) if isinstance(v, str) else v
                for k, v in content.items()}
    if hasattr(content, "model_dump"):
        try:
            return content.model_dump()
        except Exception:
            return repr(content)[:1000]
    if isinstance(content, list):
        return [_snapshot_response(c) for c in content[:20]]
    return repr(content)[:1000]


@dataclass
class ExecutionRequest:
    """Everything the executor needs to run one call."""
    fn: AgenticFunction
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    is_async: bool
    # Per-call overrides
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    cache_enabled: bool | None = None
    extra_context: dict[str, Any] | None = None
    on_chunk: Callable[[StreamChunk], None] | None = None


class Executor:
    """Stateless executor — one instance is shared by all functions."""

    def __init__(self, *, cache: CacheBackend | None = None) -> None:
        self._cache = cache

    @property
    def cache(self) -> CacheBackend:
        if self._cache is None:
            self._cache = get_cache()
        return self._cache

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def execute(self, request: ExecutionRequest) -> AgenticResult:
        return self._run_sync(request)

    async def aexecute(self, request: ExecutionRequest) -> AgenticResult:
        return await self._run_async(request)

    # ------------------------------------------------------------------
    # Sync implementation
    # ------------------------------------------------------------------
    def _run_sync(self, request: ExecutionRequest) -> AgenticResult:
        from .budget import BudgetTracker, get_default_budget_tracker

        metrics = CallMetrics()
        metrics.function_name = request.fn.qualified_name
        # Pre-populate backend + model so a failed call still tells you which
        # backend was selected.
        metrics.backend = request.fn.backend_name
        metrics.model = request.fn.model

        ctx = get_current_trace()
        span = ctx.start_span("agentic_function.call", function=request.fn.name) if ctx else None
        t_start = time.perf_counter()
        agg = get_default_aggregator()

        def _record(metrics: CallMetrics) -> None:
            if agg is not None:
                try:
                    agg.record(metrics, function_name=request.fn.name)
                except Exception:
                    pass  # aggregator failures must never break a call

        try:
            cfg = global_config()
            cache_enabled = request.cache_enabled
            if cache_enabled is None:
                cache_enabled = bool(request.fn.cache) if request.fn.cache is not None \
                    else bool(cfg.cache_enabled)
            retry_policy = request.fn.retry_policy
            debug_on = _is_debug(request.fn)
            budget: BudgetTracker | None = get_default_budget_tracker()

            # 1. Render prompt + compute cache key
            t = time.perf_counter()
            messages = render_prompt(request.fn, request.args, request.kwargs,
                                     extra=request.extra_context)
            metrics.timings.prompt_render_ms = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            cache_key = self.cache.make_key(
                request.fn.qualified_name,
                request.fn.model,
                messages,
                request.args, request.kwargs,
                request.fn.output_schema.spec_hash,
            )
            metrics.timings.cache_lookup_ms += (time.perf_counter() - t) * 1000

            if debug_on:
                metrics.request_snapshot = _snapshot_messages(messages)

            # 2. Cache lookup
            if cache_enabled:
                t = time.perf_counter()
                cached = self.cache.get(cache_key)
                metrics.timings.cache_lookup_ms += (time.perf_counter() - t) * 1000
                if cached is not None:
                    metrics.cache_hit = True
                    metrics.usage = TokenUsage(
                        prompt_tokens=cached.usage.get("prompt_tokens", 0),
                        completion_tokens=cached.usage.get("completion_tokens", 0),
                        total_tokens=cached.usage.get("total_tokens", 0),
                    )
                    metrics.total_usage = metrics.usage
                    metrics.cost_usd = estimate_cost(request.fn.model,
                                                    metrics.usage.prompt_tokens,
                                                    metrics.usage.completion_tokens)
                    metrics.total_cost_usd = metrics.cost_usd
                    metrics.successful = True
                    metrics.latency_ms = (time.perf_counter() - t_start) * 1000
                    metrics.attempts = 1
                    if span:
                        span.set_attribute("cache.hit", True)
                        span.finish("cached")
                    if budget is not None:
                        budget.record(metrics); _record(metrics)
                    result = self._rehydrate_result(request.fn, cached)
                    result._metrics = metrics  # type: ignore[attr-defined]
                    return result

            # 3. Backend call with retry
            last_exc: BaseException | None = None
            for attempt in range(retry_policy.max_retries + 1):
                metrics.attempts = attempt + 1
                attempt_span = ctx.start_span("agentic_function.attempt",
                                              attempt=attempt) if ctx else None
                try:
                    t = time.perf_counter()
                    response = self._call_backend(request, messages)
                    metrics.timings.backend_call_ms = (time.perf_counter() - t) * 1000

                    t = time.perf_counter()
                    result = self._parse_and_validate(request.fn, response, messages)
                    metrics.timings.parse_validate_ms = (time.perf_counter() - t) * 1000

                    # Update token accounting (last successful attempt wins
                    # for ``usage``; ``total_usage`` accumulates across
                    # attempts).
                    last_usage = TokenUsage(
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        total_tokens=response.total_tokens,
                    )
                    if attempt == 0:
                        metrics.total_usage = last_usage
                    else:
                        metrics.total_usage = TokenUsage(
                            prompt_tokens=metrics.total_usage.prompt_tokens + last_usage.prompt_tokens,
                            completion_tokens=metrics.total_usage.completion_tokens + last_usage.completion_tokens,
                            total_tokens=metrics.total_usage.total_tokens + last_usage.total_tokens,
                        )
                    metrics.usage = last_usage

                    last_cost = estimate_cost(
                        request.fn.model,
                        response.prompt_tokens,
                        response.completion_tokens,
                    )
                    metrics.cost_usd = last_cost
                    metrics.total_cost_usd = (metrics.total_cost_usd or 0.0) + (last_cost or 0.0)

                    # First-class backend identity (was previously buried in
                    # ``extra``). Backends populate ``response.model`` and
                    # ``response.finish_reason``; ``backend.name`` is the
                    # registry key.
                    metrics.backend = getattr(self._backend_for_metrics(request),
                                              "name", metrics.backend) or metrics.backend
                    metrics.model = response.model or metrics.model
                    metrics.finish_reason = response.finish_reason

                    metrics.successful = True
                    metrics.latency_ms = (time.perf_counter() - t_start) * 1000

                    if debug_on:
                        metrics.response_snapshot = _snapshot_response(response.content)

                    if ctx and attempt_span:
                        attempt_span.set_attribute("usage", metrics.usage.__dict__)
                        ctx.end_span(attempt_span, "ok")
                    if span:
                        span.set_attribute("model", request.fn.model)
                        span.set_attribute("backend", metrics.backend)
                        span.set_attribute("usage.prompt_tokens", metrics.usage.prompt_tokens)
                        span.set_attribute("usage.completion_tokens", metrics.usage.completion_tokens)
                        span.set_attribute("cost_usd", metrics.cost_usd)
                        span.finish("ok")

                    # 4. Write cache
                    if cache_enabled:
                        t = time.perf_counter()
                        self.cache.set(cache_key, CacheEntry(
                            key=cache_key,
                            value=result.model_dump() if hasattr(result, "model_dump") else result.__dict__,
                            raw_output=response.content,
                            created_at=time.time(),
                            model=request.fn.model,
                            prompt_hash=stable_hash(messages),
                            usage={
                                "prompt_tokens": metrics.usage.prompt_tokens,
                                "completion_tokens": metrics.usage.completion_tokens,
                                "total_tokens": metrics.usage.total_tokens,
                            },
                        ))
                        metrics.timings.cache_write_ms = (time.perf_counter() - t) * 1000

                    if budget is not None:
                        budget.record(metrics); _record(metrics)

                    result._metrics = metrics  # type: ignore[attr-defined]
                    return result

                except (BackendError, ParseError, ValidationError) as exc:
                    last_exc = exc
                    metrics.retries = attempt
                    metrics.error = f"{type(exc).__name__}: {exc}"
                    if debug_on and isinstance(exc, ValidationError):
                        metrics.response_snapshot = _snapshot_response(
                            getattr(exc, "raw_output", None)
                        )
                    if ctx and attempt_span:
                        ctx.end_span(attempt_span, "error")
                    if not is_retryable(exc, retry_policy, attempt + 1):
                        break
                    delay = retry_policy.delay_for(attempt + 1)
                    t = time.perf_counter()
                    sleep_or_yield(delay, is_async=False)
                    metrics.timings.retry_backoff_ms += (time.perf_counter() - t) * 1000
                    _log.warning("retry %d/%d after %.2fs: %s",
                                 attempt + 1, retry_policy.max_retries, delay,
                                 kv(error=str(exc), function=request.fn.name))
                except Exception as exc:
                    # Unknown exception from the backend — wrap it as a non-retryable
                    # BackendError so the user gets a uniform error type and the
                    # retry policy is consulted.
                    wrapped = BackendError(
                        f"unexpected error in backend {request.fn.backend_name!r}: {exc}",
                        backend=request.fn.backend_name,
                        raw=exc,
                    )
                    last_exc = wrapped
                    metrics.retries = attempt
                    metrics.error = f"{type(exc).__name__}: {exc}"
                    if ctx and attempt_span:
                        ctx.end_span(attempt_span, "error")
                    break

            metrics.latency_ms = (time.perf_counter() - t_start) * 1000
            metrics.successful = False
            if span:
                span.finish("error")
            if budget is not None:
                budget.record(metrics); _record(metrics)
            raise RetryExhaustedError(
                f"all {retry_policy.max_retries + 1} attempts failed for {request.fn.name}",
                attempts=metrics.attempts,
                last_exception=last_exc or RuntimeError("unknown failure"),
            )
        except Exception:
            if span and span.status == "ok":
                span.finish("error")
            raise

    # ------------------------------------------------------------------
    # Async implementation — mirrors sync, with asyncio.sleep
    # ------------------------------------------------------------------
    async def _run_async(self, request: ExecutionRequest) -> AgenticResult:
        from .budget import BudgetTracker, get_default_budget_tracker

        metrics = CallMetrics()
        metrics.function_name = request.fn.qualified_name
        metrics.backend = request.fn.backend_name
        metrics.model = request.fn.model

        ctx = get_current_trace()
        span = ctx.start_span("agentic_function.call", function=request.fn.name) if ctx else None
        t_start = time.perf_counter()
        agg = get_default_aggregator()

        def _record(metrics: CallMetrics) -> None:
            if agg is not None:
                try:
                    agg.record(metrics, function_name=request.fn.name)
                except Exception:
                    pass

        try:
            cfg = global_config()
            cache_enabled = request.cache_enabled if request.cache_enabled is not None \
                else (cfg.cache_enabled and request.fn.cache)
            retry_policy = request.fn.retry_policy
            debug_on = _is_debug(request.fn)
            budget: BudgetTracker | None = get_default_budget_tracker()

            t = time.perf_counter()
            messages = render_prompt(request.fn, request.args, request.kwargs,
                                     extra=request.extra_context)
            metrics.timings.prompt_render_ms = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            cache_key = self.cache.make_key(
                request.fn.qualified_name,
                request.fn.model,
                messages,
                request.args, request.kwargs,
                request.fn.output_schema.spec_hash,
            )
            metrics.timings.cache_lookup_ms += (time.perf_counter() - t) * 1000

            if debug_on:
                metrics.request_snapshot = _snapshot_messages(messages)

            if cache_enabled:
                t = time.perf_counter()
                cached = self.cache.get(cache_key)
                metrics.timings.cache_lookup_ms += (time.perf_counter() - t) * 1000
                if cached is not None:
                    metrics.cache_hit = True
                    metrics.usage = TokenUsage(
                        prompt_tokens=cached.usage.get("prompt_tokens", 0),
                        completion_tokens=cached.usage.get("completion_tokens", 0),
                        total_tokens=cached.usage.get("total_tokens", 0),
                    )
                    metrics.total_usage = metrics.usage
                    metrics.cost_usd = estimate_cost(request.fn.model,
                                                    metrics.usage.prompt_tokens,
                                                    metrics.usage.completion_tokens)
                    metrics.total_cost_usd = metrics.cost_usd
                    metrics.successful = True
                    metrics.latency_ms = (time.perf_counter() - t_start) * 1000
                    metrics.attempts = 1
                    if span:
                        span.set_attribute("cache.hit", True)
                        span.finish("cached")
                    if budget is not None:
                        budget.record(metrics); _record(metrics)
                    result = self._rehydrate_result(request.fn, cached)
                    result._metrics = metrics  # type: ignore[attr-defined]
                    return result

            last_exc: BaseException | None = None
            for attempt in range(retry_policy.max_retries + 1):
                metrics.attempts = attempt + 1
                attempt_span = ctx.start_span("agentic_function.attempt",
                                              attempt=attempt) if ctx else None
                try:
                    t = time.perf_counter()
                    response = await self._acall_backend(request, messages)
                    metrics.timings.backend_call_ms = (time.perf_counter() - t) * 1000

                    t = time.perf_counter()
                    result = self._parse_and_validate(request.fn, response, messages)
                    metrics.timings.parse_validate_ms = (time.perf_counter() - t) * 1000

                    last_usage = TokenUsage(
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        total_tokens=response.total_tokens,
                    )
                    if attempt == 0:
                        metrics.total_usage = last_usage
                    else:
                        metrics.total_usage = TokenUsage(
                            prompt_tokens=metrics.total_usage.prompt_tokens + last_usage.prompt_tokens,
                            completion_tokens=metrics.total_usage.completion_tokens + last_usage.completion_tokens,
                            total_tokens=metrics.total_usage.total_tokens + last_usage.total_tokens,
                        )
                    metrics.usage = last_usage
                    last_cost = estimate_cost(
                        request.fn.model,
                        response.prompt_tokens,
                        response.completion_tokens,
                    )
                    metrics.cost_usd = last_cost
                    metrics.total_cost_usd = (metrics.total_cost_usd or 0.0) + (last_cost or 0.0)

                    metrics.backend = getattr(self._backend_for_metrics(request),
                                              "name", metrics.backend) or metrics.backend
                    metrics.model = response.model or metrics.model
                    metrics.finish_reason = response.finish_reason

                    metrics.successful = True
                    metrics.latency_ms = (time.perf_counter() - t_start) * 1000

                    if debug_on:
                        metrics.response_snapshot = _snapshot_response(response.content)

                    if ctx and attempt_span:
                        attempt_span.set_attribute("usage", metrics.usage.__dict__)
                        ctx.end_span(attempt_span, "ok")
                    if span:
                        span.set_attribute("model", request.fn.model)
                        span.set_attribute("backend", metrics.backend)
                        span.set_attribute("usage.prompt_tokens", metrics.usage.prompt_tokens)
                        span.set_attribute("usage.completion_tokens", metrics.usage.completion_tokens)
                        span.set_attribute("cost_usd", metrics.cost_usd)
                        span.finish("ok")

                    if cache_enabled:
                        t = time.perf_counter()
                        self.cache.set(cache_key, CacheEntry(
                            key=cache_key,
                            value=result.model_dump() if hasattr(result, "model_dump") else result.__dict__,
                            raw_output=response.content,
                            created_at=time.time(),
                            model=request.fn.model,
                            prompt_hash=stable_hash(messages),
                            usage={
                                "prompt_tokens": metrics.usage.prompt_tokens,
                                "completion_tokens": metrics.usage.completion_tokens,
                                "total_tokens": metrics.usage.total_tokens,
                            },
                        ))
                        metrics.timings.cache_write_ms = (time.perf_counter() - t) * 1000

                    if budget is not None:
                        budget.record(metrics); _record(metrics)

                    result._metrics = metrics  # type: ignore[attr-defined]
                    return result
                except (BackendError, ParseError, ValidationError) as exc:
                    last_exc = exc
                    metrics.retries = attempt
                    metrics.error = f"{type(exc).__name__}: {exc}"
                    if debug_on and isinstance(exc, ValidationError):
                        metrics.response_snapshot = _snapshot_response(
                            getattr(exc, "raw_output", None)
                        )
                    if ctx and attempt_span:
                        ctx.end_span(attempt_span, "error")
                    if not is_retryable(exc, retry_policy, attempt + 1):
                        break
                    delay = retry_policy.delay_for(attempt + 1)
                    t = time.perf_counter()
                    await asyncio.sleep(delay)  # type: ignore[arg-type]
                    metrics.timings.retry_backoff_ms += (time.perf_counter() - t) * 1000
                    _log.warning("retry %d/%d after %.2fs: %s",
                                 attempt + 1, retry_policy.max_retries, delay,
                                 kv(error=str(exc), function=request.fn.name))
                except Exception as exc:
                    wrapped = BackendError(
                        f"unexpected error in backend {request.fn.backend_name!r}: {exc}",
                        backend=request.fn.backend_name,
                        raw=exc,
                    )
                    last_exc = wrapped
                    metrics.retries = attempt
                    metrics.error = f"{type(exc).__name__}: {exc}"
                    if ctx and attempt_span:
                        ctx.end_span(attempt_span, "error")
                    break

            metrics.latency_ms = (time.perf_counter() - t_start) * 1000
            metrics.successful = False
            if span:
                span.finish("error")
            if budget is not None:
                budget.record(metrics); _record(metrics)
            raise RetryExhaustedError(
                f"all {retry_policy.max_retries + 1} attempts failed for {request.fn.name}",
                attempts=metrics.attempts,
                last_exception=last_exc or RuntimeError("unknown failure"),
            )
        except Exception:
            if span and span.status == "ok":
                span.finish("error")
            raise

    # ------------------------------------------------------------------
    # Backend dispatch
    # ------------------------------------------------------------------
    def _backend_for_metrics(self, request: ExecutionRequest) -> LLMBackend:
        """Resolve the backend *as the executor sees it* — used to fill
        ``metrics.backend`` with the registry name (e.g. ``"openai"``)."""
        try:
            return request.fn.backend
        except Exception:
            class _Fallback:
                name = request.fn.backend_name
            return _Fallback()  # type: ignore[return-value]

    def _call_backend(self, request: ExecutionRequest, messages: list[dict[str, Any]]) -> LLMResponse:
        backend = request.fn.backend
        temperature = request.temperature if request.temperature is not None else request.fn.temperature
        max_tokens = request.max_tokens if request.max_tokens is not None else request.fn.max_tokens
        model = request.model or request.fn.model
        return backend.complete(
            model=model,
            messages=messages,
            output_schema=request.fn.output_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=request.fn.top_p,
            stop=request.fn.stop,
            on_chunk=request.on_chunk,
            timeout=request.fn.timeout,
        )

    async def _acall_backend(self, request: ExecutionRequest, messages: list[dict[str, Any]]) -> LLMResponse:
        backend = request.fn.backend
        temperature = request.temperature if request.temperature is not None else request.fn.temperature
        max_tokens = request.max_tokens if request.max_tokens is not None else request.fn.max_tokens
        model = request.model or request.fn.model
        ac = getattr(backend, "acomplete", None)
        if ac is not None and inspect.iscoroutinefunction(ac):
            return await ac(
                model=model,
                messages=messages,
                output_schema=request.fn.output_schema,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=request.fn.top_p,
                stop=request.fn.stop,
                on_chunk=request.on_chunk,
                timeout=request.fn.timeout,
            )
        return await asyncio.to_thread(
            backend.complete,
            model=model,
            messages=messages,
            output_schema=request.fn.output_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=request.fn.top_p,
            stop=request.fn.stop,
            on_chunk=request.on_chunk,
            timeout=request.fn.timeout,
        )

    # ------------------------------------------------------------------
    # Parse + validate
    # ------------------------------------------------------------------
    def _parse_and_validate(self, fn: AgenticFunction, response: LLMResponse,
                            messages: list[dict[str, Any]]) -> AgenticResult:
        schema = fn.output_schema
        parsed = _extract_json(response.content, schema)
        try:
            result = schema.coerce(parsed)
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError(
                f"failed to validate response against schema for {fn.name}: {exc}",
                errors=[{"error": str(exc)}],
                raw_output=parsed,
            ) from exc
        if schema.is_dynamic:
            assert isinstance(result, DynamicResult)
            result._raw_output = response.content
            result._messages = messages
            return result
        assert not isinstance(result, DynamicResult)
        result._raw_output = response.content
        result._messages = messages
        return result

    # ------------------------------------------------------------------
    # Cache rehydration
    # ------------------------------------------------------------------
    def _rehydrate_result(self, fn: AgenticFunction, entry: CacheEntry) -> AgenticResult:
        schema = fn.output_schema
        if schema.is_dynamic:
            assert schema.model_class is not None
            return schema.model_class.model_validate(entry.value)
        model_cls = schema.model_class
        assert model_cls is not None
        return model_cls.model_validate(entry.value)


# ----------------------------------------------------------------------
# JSON extraction helpers — robust to markdown fences, garbage around the JSON,
# etc. We try a few strategies in order.
# ----------------------------------------------------------------------
def _extract_json(content: Any, schema: SchemaSpec) -> Any:
    """Extract a JSON value from ``content``.

    Backends can return:
      - a dict directly (when structured output is supported natively)
      - a JSON string
      - a JSON string wrapped in markdown fences
      - arbitrary prose with a JSON blob somewhere inside
    """
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = "\n".join(text_parts)
        else:
            raise ParseError(f"unexpected response content type: {type(content).__name__}",
                             raw_output=content)

    text = content.strip()

    try:
        import json as _json
        return _json.loads(text)
    except Exception:
        pass

    if text.startswith("```"):
        first_nl = text.find("\n")
        last_fence = text.rfind("```")
        if first_nl != -1 and last_fence > first_nl:
            inner = text[first_nl + 1:last_fence].strip()
            try:
                import json as _json
                return _json.loads(inner)
            except Exception:
                text = inner

    candidate = _find_top_level_json(text)
    if candidate is not None:
        try:
            import json as _json
            return _json.loads(candidate)
        except Exception:
            pass

    raise ParseError("could not locate a JSON object/array in model output",
                     raw_output=content)


def _find_top_level_json(text: str) -> str | None:
    """Find the first balanced top-level JSON object or array."""
    opens = {"{": "}", "[": "]"}
    for i, ch in enumerate(text):
        if ch in opens:
            close = opens[ch]
            depth = 0
            in_str = False
            esc = False
            for j in range(i, len(text)):
                c = text[j]
                if esc:
                    esc = False
                    continue
                if c == "\\":
                    esc = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if c == ch:
                    depth += 1
                elif c == close:
                    depth -= 1
                    if depth == 0:
                        return text[i:j + 1]
    return None


def _stable_msg_keys(messages: list[dict[str, Any]]) -> list[str]:
    """Return a deterministic view of message keys for cache-key derivation."""
    keys: list[str] = []
    for m in messages:
        for k in sorted(m.keys()):
            keys.append(k)
    return keys