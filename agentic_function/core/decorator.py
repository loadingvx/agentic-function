"""The ``@agentic_function`` decorator.

Three call shapes are supported::

    @agentic_function
    def f(text: str) -> MyPydantic: ...

    @agentic_function(model="gpt-4o", output_schema={"x": int, "y": str})
    def f(text: str) -> AgenticResult: ...

    @agentic_function(model="...", output_schema=MyModel, temperature=0.5)
    def f(text: str) -> MyModel: ...
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

from ..backends.base import LLMBackend
from ..errors import ConfigError, SchemaError
from ..runtime.config import global_config
from ..runtime.executor import Executor
from ..runtime.retry import RetryPolicy, default_retry_policy
from ..types import SchemaDeclaration
from .function import AgenticFunction, _resolve_owner
from .schema import resolve_schema, schema_from_annotation


# Default executor is process-wide — easy to monkeypatch for tests.
_default_executor = Executor()


def get_default_executor() -> Executor:
    return _default_executor


def set_default_executor(executor: Executor) -> None:
    global _default_executor
    _default_executor = executor


def agentic_function(
    func_or_model: Any = None,
    *,
    model: str | None = None,
    output_schema: SchemaDeclaration | None = None,
    backend: str | LLMBackend | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    stop: list[str] | None = None,
    max_retries: int | None = None,
    retry_policy: RetryPolicy | None = None,
    cache: bool | None = None,
    timeout: float | None = None,
    include_schema_in_prompt: bool = True,
    few_shots: list[tuple[dict[str, Any], dict[str, Any]]] | None = None,
    prompt_template: str | None = None,
    system_template: str | None = None,
    description: str | None = None,
    executor: Executor | None = None,
    debug: bool = False,
) -> Any:
    """Decorator that turns a Python function into an agentic function.

    May be used bare (``@agentic_function``) or with kwargs
    (``@agentic_function(model="gpt-4o", output_schema={...})``).
    """
    # Bare-decorator form: @agentic_function  (no parens)
    if callable(func_or_model) and not isinstance(func_or_model, type):
        return _build(
            func=func_or_model,
            model=model,
            output_schema=output_schema,
            backend=backend,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            max_retries=max_retries,
            retry_policy=retry_policy,
            cache=cache,
            timeout=timeout,
            include_schema_in_prompt=include_schema_in_prompt,
            few_shots=few_shots,
            prompt_template=prompt_template,
            system_template=system_template,
            description=description,
            executor=executor,
            debug=debug,
        )

    # Parameterised form: @agentic_function(...)
    if func_or_model is not None and not callable(func_or_model):
        raise ConfigError(
            f"first positional argument to @agentic_function must be the "
            f"decorated function when used bare, or omitted entirely; got "
            f"{type(func_or_model).__name__}"
        )

    def decorator(func: Callable[..., Any]) -> AgenticFunction:
        return _build(
            func=func,
            model=model or func_or_model,
            output_schema=output_schema,
            backend=backend,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            max_retries=max_retries,
            retry_policy=retry_policy,
            cache=cache,
            timeout=timeout,
            include_schema_in_prompt=include_schema_in_prompt,
            few_shots=few_shots,
            prompt_template=prompt_template,
            system_template=system_template,
            description=description,
            executor=executor,
            debug=debug,
        )

    return decorator


def _build(
    *,
    func: Callable[..., Any],
    model: str | None,
    output_schema: SchemaDeclaration | None,
    backend: str | LLMBackend | None,
    temperature: float | None,
    max_tokens: int | None,
    top_p: float | None,
    stop: list[str] | None,
    max_retries: int | None,
    retry_policy: RetryPolicy | None,
    cache: bool | None,
    timeout: float | None,
    include_schema_in_prompt: bool,
    few_shots: list[tuple[dict[str, Any], dict[str, Any]]] | None,
    prompt_template: str | None,
    system_template: str | None,
    description: str | None,
    executor: Executor | None,
    debug: bool,
) -> AgenticFunction:
    cfg = global_config()

    # ---- model ----
    resolved_model = model or cfg.default_model

    # ---- schema ----
    if output_schema is None:
        # Try to infer from return annotation.
        sig = inspect.signature(func)
        try:
            spec = schema_from_annotation(sig.return_annotation,
                                          function_name=func.__name__,
                                          func=func)
        except SchemaError:
            raise ConfigError(
                f"could not infer output_schema for {func.__name__!r}: "
                "pass `output_schema=` explicitly or annotate the return type "
                "with a BaseModel subclass"
            )
    else:
        spec = resolve_schema(output_schema, function_name=func.__name__)

    # ---- backend ----
    # We deliberately store ``backend`` as ``None`` when the user didn't pass
    # one — letting ``AgenticFunction.backend`` fall back to whatever the
    # process-wide default backend is *at call time*. This means a user can
    # call ``set_default_backend(MockBackend())`` AFTER decorating and still
    # have it take effect.
    if backend is None:
        backend_to_store: Any = None
    elif isinstance(backend, str):
        backend_to_store = backend
    else:
        backend_to_store = backend

    # ---- retry ----
    if retry_policy is None:
        mr = max_retries if max_retries is not None else cfg.max_retries
        retry_policy = default_retry_policy(max_retries=mr)

    # ---- cache ----
    if cache is None:
        cache = bool(cfg.cache_enabled)

    # ---- signature ----
    sig = inspect.signature(func)
    if inspect.iscoroutinefunction(func):
        # We still allow decoration; user should call .acall. We don't try to
        # auto-await because that would surprise people using the descriptor.
        pass

    func_executor = executor or get_default_executor()

    return AgenticFunction(
        wrapped=func,
        name=func.__name__,
        docstring=inspect.getdoc(func) or "",
        signature=sig,
        model=resolved_model,
        backend=backend_to_store,
        output_schema=spec,
        temperature=temperature if temperature is not None else cfg.temperature,
        max_tokens=max_tokens if max_tokens is not None else cfg.max_tokens,
        top_p=top_p if top_p is not None else cfg.top_p,
        stop=stop,
        retry_policy=retry_policy,
        cache=cache,
        timeout=timeout if timeout is not None else cfg.request_timeout,
        include_schema_in_prompt=include_schema_in_prompt,
        few_shots=few_shots or [],
        prompt_template=prompt_template,
        system_template=system_template,
        description=description,
        executor=func_executor,
        qualified_name=_resolve_owner(func),
        debug=debug,
    )