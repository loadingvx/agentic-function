"""The descriptor object returned by ``@agentic_function``.

Implements Python's descriptor protocol so that::

    class MyClass:
        @agentic_function(...)
        def my_method(self, text: str) -> ...: ...

    obj = MyClass()
    obj.my_method("hello")  # → AgenticResult
    MyClass.my_method("hello")  # → AgenticResult
    await obj.my_method.acall("hello")  # → AgenticResult
    await MyClass.my_method.acall("hello")  # → AgenticResult

Calling through an instance preserves ``self`` — useful when the same agentic
function needs to know which instance invoked it (e.g. for per-instance config).
"""
from __future__ import annotations

import asyncio
import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable

from .schema import SchemaSpec, resolve_schema, schema_from_annotation
from .result import AgenticResult
from ..backends.base import LLMBackend

if TYPE_CHECKING:
    from ..runtime.executor import Executor, ExecutionRequest
    from ..runtime.retry import RetryPolicy
    from ..backends.base import get_backend


def _resolve_owner(func: Callable[..., Any]) -> str:
    """Return a stable qualified name for the function, including its class."""
    qualname = getattr(func, "__qualname__", func.__name__)
    module = getattr(func, "__module__", "?")
    return f"{module}.{qualname}"


class AgenticFunction:
    """Descriptor object created by ``@agentic_function``.

    Calling the descriptor (whether on a class or an instance) runs the
    function. Awaiting it via ``.acall`` runs the async path.
    """

    def __init__(
        self,
        *,
        wrapped: Callable[..., Any],
        name: str,
        docstring: str,
        signature: inspect.Signature,
        model: str,
        backend: str | LLMBackend,
        output_schema: SchemaSpec,
        temperature: float,
        max_tokens: int | None,
        top_p: float | None,
        stop: list[str] | None,
        retry_policy: RetryPolicy,
        cache: bool,
        timeout: float | None,
        include_schema_in_prompt: bool,
        few_shots: list[tuple[dict[str, Any], dict[str, Any]]],
        prompt_template: str | None,
        system_template: str | None,
        description: str | None,
        executor: "Executor",
        qualified_name: str,
        debug: bool = False,
    ) -> None:
        self.wrapped = wrapped
        self._name = name
        # Expose the agentic function's name as the standard dunder too —
        # useful for inspect.getmembers, logging, etc.
        self.__name__ = name
        # The docstring of the agentic function is the user-supplied one (not
        # the wrapped function's), so tools like ``inspect.getdoc`` see the
        # intended behaviour description.
        self.__doc__ = docstring
        # Preserve origin metadata from the wrapped function. NOTE: we
        # deliberately do NOT copy ``__dict__`` — that would replace the
        # instance dict and wipe every attribute we just set.
        try:
            self.__module__ = getattr(wrapped, "__module__", self.__class__.__module__)
            self.__qualname__ = getattr(wrapped, "__qualname__", name)
            self.__wrapped__ = wrapped
        except (AttributeError, TypeError):
            pass
        self.signature = signature
        self.model = model
        self.backend_name = backend if isinstance(backend, str) else type(backend).__name__
        self._backend_override = backend if not isinstance(backend, str) else None
        self.output_schema = output_schema
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stop = stop
        self.retry_policy = retry_policy
        self.cache = cache
        self.timeout = timeout
        self.include_schema_in_prompt = include_schema_in_prompt
        self.few_shots = few_shots
        self.prompt_template = prompt_template
        self.system_template = system_template
        self.description = description or docstring.splitlines()[0] if docstring else None
        self._executor_override = executor
        self.qualified_name = qualified_name
        self.debug = debug

    # ------------------------------------------------------------------
    # Lazy executor resolution: if the user passed an explicit executor at
    # decoration time, use it; otherwise pick up the current process default
    # at call time. This lets tests swap the executor after decoration.
    # ------------------------------------------------------------------
    @property
    def executor(self) -> "Executor":
        if self._executor_override is not None:
            return self._executor_override
        from ..core.decorator import get_default_executor
        return get_default_executor()

    @executor.setter
    def executor(self, value: "Executor") -> None:
        self._executor_override = value
    @property
    def backend(self) -> LLMBackend:
        if self._backend_override is not None:
            return self._backend_override
        from ..backends.base import get_default_backend, get_backend
        default = get_default_backend()
        if default is not None:
            return default
        # Fall back to a named backend (e.g. ``"openai"``) when no global
        # default was installed.
        return get_backend(self.backend_name)

    # ------------------------------------------------------------------
    # Descriptor protocol — supports both class-level and instance-level calls
    # ------------------------------------------------------------------
    def __get__(self, instance: Any, owner: type | None = None) -> "AgenticFunction":
        if instance is None:
            return self
        # Return a thin bound wrapper so we can pass ``instance`` into the
        # wrapped function if it wants ``self``.
        return _BoundAgenticFunction(self, instance)

    # ------------------------------------------------------------------
    # Sync call path
    # ------------------------------------------------------------------
    def __call__(self, *args: Any, _instance: Any | None = None, **kwargs: Any) -> AgenticResult:
        # Unwrap bound-method call.
        if _instance is not None:
            args = (_instance, *args)
        # Local import to break the core ↔ runtime cycle.
        from ..runtime.executor import ExecutionRequest
        req = ExecutionRequest(
            fn=self,
            args=args,
            kwargs=kwargs,
            is_async=False,
        )
        return self.executor.execute(req)

    # ------------------------------------------------------------------
    # Async call path
    # ------------------------------------------------------------------
    async def acall(self, *args: Any, _instance: Any | None = None, **kwargs: Any) -> AgenticResult:
        if _instance is not None:
            args = (_instance, *args)
        from ..runtime.executor import ExecutionRequest
        req = ExecutionRequest(
            fn=self,
            args=args,
            kwargs=kwargs,
            is_async=True,
        )
        return await self.executor.aexecute(req)

    # ------------------------------------------------------------------
    # Repr / str
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"<AgenticFunction name={self.name!r} model={self.model!r} "
            f"backend={self.backend_name!r}>"
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def docstring(self) -> str:
        """Alias for ``__doc__`` — useful in templating code."""
        return self.__doc__ or ""


class _BoundAgenticFunction:
    """Lightweight wrapper returned when accessing the descriptor on an instance.

    Forwards ``__call__`` and ``acall`` to the underlying descriptor, threading
    ``instance`` through so that signatures with ``self`` work correctly.
    """

    __slots__ = ("_fn", "_instance")

    def __init__(self, fn: AgenticFunction, instance: Any) -> None:
        self._fn = fn
        self._instance = instance

    def __call__(self, *args: Any, **kwargs: Any) -> AgenticResult:
        return self._fn(*args, _instance=self._instance, **kwargs)

    async def acall(self, *args: Any, **kwargs: Any) -> AgenticResult:
        return await self._fn.acall(*args, _instance=self._instance, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fn, name)