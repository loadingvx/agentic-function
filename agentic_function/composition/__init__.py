"""Composition: registry, tool conversion, pipeline helpers."""
from .registry import FunctionRegistry, get_global_registry, register, get_function
from .as_tool import as_openai_tool, as_anthropic_tool

__all__ = [
    "FunctionRegistry",
    "get_global_registry",
    "register",
    "get_function",
    "as_openai_tool",
    "as_anthropic_tool",
]