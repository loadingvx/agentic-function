"""Validation — pretty error messages and a one-shot ``validate`` helper."""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError as PydanticValidationError

from ..errors import ValidationError


def validate(value: Any, model_class: type[BaseModel]) -> BaseModel:
    """Validate ``value`` against ``model_class`` and raise a friendly error on failure."""
    try:
        if isinstance(value, model_class):
            return value
        return model_class.model_validate(value)
    except PydanticValidationError as exc:
        raise ValidationError(
            f"validation failed for {model_class.__name__}: {format_validation_errors(exc)}",
            errors=_pydantic_errors_to_dicts(exc),
            raw_output=value,
        ) from exc


def format_validation_errors(exc: PydanticValidationError | BaseException) -> str:
    """Render a Pydantic ValidationError as a readable string."""
    try:
        errors = exc.errors()  # type: ignore[attr-defined]
    except Exception:
        return str(exc)
    lines = []
    for err in errors:
        loc = ".".join(str(x) for x in err.get("loc", ()))
        msg = err.get("msg", "invalid")
        typ = err.get("type", "error")
        lines.append(f"  - {loc}: {msg} ({typ})")
    return "\n".join(lines) if lines else str(exc)


def _pydantic_errors_to_dicts(exc: PydanticValidationError) -> list[dict[str, Any]]:
    try:
        out = []
        for err in exc.errors():
            out.append({
                "loc": list(err.get("loc", ())),
                "msg": err.get("msg", ""),
                "type": err.get("type", ""),
            })
        return out
    except Exception:
        return [{"error": str(exc)}]