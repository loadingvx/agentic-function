"""Coercion helpers — turn "almost right" LLM output into the right type.

LLMs are wonderfully creative with types. Common cases we handle:

* Numbers returned as strings ("42" → 42)
* Booleans returned as "yes" / "no" / "true" / "false"
* Lists returned as comma-separated strings ("a, b, c" → ["a", "b", "c"])
* Nested dicts that are mostly right but have an extra field

We do **not** silently fix semantic errors — if the LLM says ``severity="critical"``
when we asked for ``"low"|"medium"|"high"``, we raise. Coercion is for
shape, not content.
"""
from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel

from ..errors import ValidationError


def coerce_value(value: Any, target_type: Any) -> Any:
    """Coerce ``value`` into ``target_type`` if possible.

    Returns ``value`` unchanged if no coercion is needed or sensible.
    Raises ``ValidationError`` if coercion would lose semantic meaning.
    """
    origin = getattr(target_type, "__origin__", None)
    if target_type is Any or target_type is None:
        return value
    if target_type is bool:
        return _coerce_bool(value)
    if target_type is int:
        return _coerce_int(value)
    if target_type is float:
        return _coerce_float(value)
    if target_type is str:
        return str(value) if value is not None else None  # type: ignore[arg-type]
    if origin in (list, tuple, set, frozenset):
        return _coerce_collection(value, target_type)
    if origin in (dict, Mapping):
        return _coerce_dict(value, target_type)
    if isinstance(target_type, type) and issubclass(target_type, BaseModel):
        if isinstance(value, target_type):
            return value
        if isinstance(value, Mapping):
            return target_type.model_validate(value)
        raise ValidationError(
            f"cannot coerce {type(value).__name__} to {target_type.__name__}",
            raw_output=value,
        )
    return value


def coerce_to_schema(value: Any, model_class: type[BaseModel]) -> BaseModel:
    """Validate ``value`` against ``model_class`` with coercion enabled."""
    if isinstance(value, model_class):
        return value
    if not isinstance(value, Mapping):
        raise ValidationError(
            f"expected mapping for {model_class.__name__}, got {type(value).__name__}",
            raw_output=value,
        )
    return model_class.model_validate(value)


# ---------------------------------------------------------------------------
# Type-specific coercers
# ---------------------------------------------------------------------------
def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise ValidationError(f"cannot coerce int {value} to bool", raw_output=value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "yes", "y", "1"):
            return True
        if s in ("false", "no", "n", "0", ""):
            return False
        raise ValidationError(f"cannot coerce string {value!r} to bool", raw_output=value)
    raise ValidationError(f"cannot coerce {type(value).__name__} to bool", raw_output=value)


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValidationError(f"non-integer float {value} cannot become int", raw_output=value)
    if isinstance(value, str):
        s = value.strip()
        try:
            return int(s)
        except ValueError:
            try:
                f = float(s)
                if f.is_integer():
                    return int(f)
            except ValueError:
                pass
            raise ValidationError(f"cannot coerce string {value!r} to int", raw_output=value)
    raise ValidationError(f"cannot coerce {type(value).__name__} to int", raw_output=value)


def _coerce_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError as exc:
            raise ValidationError(f"cannot coerce string {value!r} to float",
                                  raw_output=value) from exc
    raise ValidationError(f"cannot coerce {type(value).__name__} to float", raw_output=value)


def _coerce_collection(value: Any, target_type: Any) -> list[Any]:
    args = getattr(target_type, "__args__", ())
    elem_type = args[0] if args else Any
    if isinstance(value, (list, tuple, set, frozenset)):
        return [coerce_value(v, elem_type) for v in value]
    if isinstance(value, str):
        # "a, b, c" → ["a", "b", "c"]
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return [coerce_value(p, elem_type) for p in parts]
    raise ValidationError(
        f"cannot coerce {type(value).__name__} to collection",
        raw_output=value,
    )


def _coerce_dict(value: Any, target_type: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        args = getattr(target_type, "__args__", ())
        v_type = args[1] if len(args) == 2 else Any
        return {str(k): coerce_value(v, v_type) for k, v in value.items()}
    raise ValidationError(
        f"cannot coerce {type(value).__name__} to dict",
        raw_output=value,
    )