"""Process-wide registry of agentic functions.

Useful when you want to look up a function by name (e.g. in a tool-calling
Agent that decides which agentic function to call dynamically).
"""
from __future__ import annotations

from typing import Iterator

from ..core.function import AgenticFunction
from ..errors import RegistrationError


class FunctionRegistry:
    """A name → ``AgenticFunction`` mapping."""

    def __init__(self) -> None:
        self._store: dict[str, AgenticFunction] = {}

    def add(self, fn: AgenticFunction, *, name: str | None = None,
            allow_overwrite: bool = False) -> str:
        key = name or fn.qualified_name
        if key in self._store and not allow_overwrite:
            raise RegistrationError(
                f"function {key!r} already registered; pass allow_overwrite=True"
            )
        self._store[key] = fn
        return key

    def remove(self, name: str) -> None:
        self._store.pop(name, None)

    def get(self, name: str) -> AgenticFunction:
        try:
            return self._store[name]
        except KeyError as exc:
            raise RegistrationError(f"no function registered under {name!r}",
                                    ) from exc

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __iter__(self) -> Iterator[AgenticFunction]:
        return iter(self._store.values())

    def __len__(self) -> int:
        return len(self._store)

    def names(self) -> list[str]:
        return sorted(self._store)


_global = FunctionRegistry()


def get_global_registry() -> FunctionRegistry:
    return _global


def register(fn: AgenticFunction, *, name: str | None = None,
             allow_overwrite: bool = False) -> str:
    """Register an ``AgenticFunction`` in the global registry."""
    return _global.add(fn, name=name, allow_overwrite=allow_overwrite)


def get_function(name: str) -> AgenticFunction:
    return _global.get(name)