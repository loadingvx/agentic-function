"""Schema resolution and pydantic-model generation.

The user passes ``output_schema`` in one of three forms:

1. A pydantic ``BaseModel`` subclass — used directly.
2. A ``dict[str, type]`` — converted to a dynamically generated pydantic model.
3. ``None`` (or omitted) — the function's return-type annotation is used.

The ``SchemaSpec`` returned by ``resolve_schema`` wraps the resolved pydantic
model and exposes helpers used by:

* the executor (``coerce(parsed)``)
* the prompt builder (``to_json_schema()`` for the prompt instructions)
* the OpenAI tool converter (``to_tool_json_schema()``)
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from pydantic import BaseModel, Field, create_model

from ..errors import SchemaError
from ..types import SchemaDeclaration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_PYTHON_TYPE_TO_JSON = {
    str:    {"type": "string"},
    int:    {"type": "integer"},
    float:  {"type": "number"},
    bool:   {"type": "boolean"},
    list:   {"type": "array"},
    dict:   {"type": "object"},
}


def _annotation_to_json(annotation: Any) -> dict[str, Any]:
    """Best-effort JSON-schema fragment for a Python annotation."""
    origin = getattr(annotation, "__origin__", None)
    if annotation in _PYTHON_TYPE_TO_JSON:
        return _PYTHON_TYPE_TO_JSON[annotation]  # type: ignore[index]
    if origin in (list, tuple, set, frozenset):
        args = getattr(annotation, "__args__", ())
        if args:
            return {"type": "array", "items": _annotation_to_json(args[0])}
        return {"type": "array"}
    if origin in (dict, Mapping):
        args = getattr(annotation, "__args__", ())
        if len(args) == 2:
            return {"type": "object",
                    "additionalProperties": _annotation_to_json(args[1])}
        return {"type": "object"}
    if origin is type or origin is type(None):  # noqa: E721
        return {"type": "string"}
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation.model_json_schema()
    # Fallback — be permissive.
    return {"type": "string"}


# ---------------------------------------------------------------------------
# SchemaSpec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SchemaSpec:
    """Resolved output schema.

    Attributes:
        model_class: the pydantic model class used to validate output.
        is_dynamic:  ``True`` if the schema was declared as a plain dict and
                     we generated a model on the fly.
        spec_hash:   short stable hash of the schema — used in cache keys.
        declared_fields: the original ``dict[str, type]`` (empty if pydantic).
    """
    model_class: type[BaseModel]
    is_dynamic: bool
    spec_hash: str
    declared_fields: Mapping[str, type] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Coerce a parsed value (dict) into a validated model instance.
    # ------------------------------------------------------------------
    def coerce(self, parsed: Any) -> BaseModel:
        if isinstance(parsed, self.model_class):
            return parsed
        if not isinstance(parsed, dict):
            # Try to surface a useful error.
            raise SchemaError(
                f"expected dict, got {type(parsed).__name__}: {parsed!r}"
            )
        return self.model_class.model_validate(parsed)

    # ------------------------------------------------------------------
    # JSON Schema for prompt rendering & OpenAI tool calls.
    # ------------------------------------------------------------------
    def to_json_schema(self) -> dict[str, Any]:
        return self.model_class.model_json_schema()

    def json_schema_str(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_json_schema(), indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# resolve_schema
# ---------------------------------------------------------------------------
def _build_dynamic_model(name: str, fields: Mapping[str, type]) -> type[BaseModel]:
    """Generate a pydantic model from ``{field_name: python_type}``."""
    from .result import DynamicResult
    if not fields:
        raise SchemaError(
            "output_schema is empty — declare at least one field, or omit the "
            "argument to use the function's return-type annotation."
        )
    field_defs: dict[str, tuple[Any, Any]] = {}
    for fname, ftype in fields.items():
        if not isinstance(fname, str) or not fname:
            raise SchemaError(f"invalid field name: {fname!r}")
        field_defs[fname] = (ftype, Field(default=...))
    # Inherit from DynamicResult so the result carries the private attrs and
    # the fallback attribute lookup for ``extras``.
    return create_model(name, __base__=DynamicResult, **field_defs)  # type: ignore[call-overload]


def _hash_schema(model_class: type[BaseModel], declared: Mapping[str, type]) -> str:
    payload = {
        "class": f"{model_class.__module__}.{model_class.__qualname__}",
        "fields": sorted(
            (name, str(annotation))
            for name, annotation in model_class.model_fields.items()
        ),
        "declared": sorted((k, str(v)) for k, v in declared.items()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def resolve_schema(declaration: SchemaDeclaration | None,
                   *,
                   function_name: str = "AgenticFunctionOutput") -> SchemaSpec:
    """Resolve ``output_schema`` to a ``SchemaSpec``.

    ``declaration`` may be:
      * ``None`` — caller must have set up the schema elsewhere.
      * a ``BaseModel`` subclass.
      * a ``dict[str, type]`` mapping field name → Python type.
    """
    if declaration is None:
        raise SchemaError("output_schema must be provided (model class or dict)")

    if isinstance(declaration, type) and issubclass(declaration, BaseModel):
        return SchemaSpec(
            model_class=declaration,
            is_dynamic=False,
            spec_hash=_hash_schema(declaration, {}),
            declared_fields={},
        )

    if isinstance(declaration, dict):
        # Strip reserved internal fields from declared_fields — we own those.
        reserved = {"extras", "_raw_output", "_messages", "_metrics"}
        clean = {k: v for k, v in declaration.items() if k not in reserved}
        model = _build_dynamic_model(function_name + "Output", clean)
        return SchemaSpec(
            model_class=model,
            is_dynamic=True,
            spec_hash=_hash_schema(model, clean),
            declared_fields=clean,
        )

    raise SchemaError(
        f"output_schema must be a BaseModel subclass or dict[str, type], "
        f"got {type(declaration).__name__}"
    )


def schema_from_annotation(annotation: Any, *,
                            function_name: str = "AgenticFunctionOutput",
                            func: Callable[..., Any] | None = None) -> SchemaSpec:
    """Build a schema from a function's return annotation, if possible.

    If ``func`` is provided, ``typing.get_type_hints`` is used to evaluate
    forward references in the function's own scope (so locally-defined
    classes work).
    """
    # Try to evaluate the annotation in the function's scope first.
    if func is not None:
        # Method 1: typing.get_type_hints — handles globals + class scope.
        try:
            import typing
            hints = typing.get_type_hints(func)
            annotation = hints.get("return", annotation)
        except Exception:
            pass
        # Method 2: if it's still a string, try the wrapped function's local
        # scope via __wrapped__ chain. (Decorator stacking may obscure this.)
        if isinstance(annotation, str) and hasattr(func, "__wrapped__"):
            try:
                hints = typing.get_type_hints(func.__wrapped__)
                annotation = hints.get("return", annotation)
            except Exception:
                pass
        # Method 3: walk the closure cells of the wrapped function for a
        # variable named like the string.
        if isinstance(annotation, str):
            try:
                closure = getattr(func, "__wrapped__", func).__closure__ or ()
                for cell in closure:
                    val = cell.cell_contents
                    if hasattr(val, "__name__") and val.__name__ == annotation:
                        annotation = val
                        break
            except Exception:
                pass
    if annotation is inspect._empty:  # type: ignore[attr-defined]
        raise SchemaError(
            "no output_schema provided and function has no return annotation"
        )
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return SchemaSpec(
            model_class=annotation,
            is_dynamic=False,
            spec_hash=_hash_schema(annotation, {}),
            declared_fields={},
        )
    if hasattr(annotation, "__origin__") and annotation.__origin__ is dict:
        # dict[str, type] — but we don't have a static field map here.
        raise SchemaError(
            "return annotation is a generic dict; pass output_schema explicitly"
        )
    raise SchemaError(
        f"unsupported return annotation for output_schema inference: {annotation!r}"
    )