"""Schema resolution tests."""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from agentic_function.core.schema import resolve_schema, schema_from_annotation
from agentic_function.errors import SchemaError


def test_resolve_dict_schema():
    spec = resolve_schema({"label": str, "score": float})
    assert spec.is_dynamic
    # Auto-generated model has those fields
    assert "label" in spec.model_class.model_fields
    assert "score" in spec.model_class.model_fields


def test_resolve_pydantic_model():
    class Output(BaseModel):
        x: int
        y: str

    spec = resolve_schema(Output)
    assert not spec.is_dynamic
    assert spec.model_class is Output


def test_resolve_invalid():
    with pytest.raises(SchemaError):
        resolve_schema(42)
    with pytest.raises(SchemaError):
        resolve_schema({})  # empty dict


def test_schema_coerce_dict():
    spec = resolve_schema({"a": int, "b": str})
    m = spec.coerce({"a": 1, "b": "hello"})
    assert m.a == 1 and m.b == "hello"


def test_schema_coerce_validation_error():
    spec = resolve_schema({"a": int})
    with pytest.raises(Exception):
        spec.coerce({"a": "not an int"})


def test_schema_to_json_schema():
    spec = resolve_schema({"x": int, "y": list[str]})
    js = spec.to_json_schema()
    assert "properties" in js
    assert "x" in js["properties"]
    assert js["properties"]["x"]["type"] == "integer"


def test_schema_from_annotation_missing():
    import inspect
    with pytest.raises(SchemaError):
        schema_from_annotation(inspect._empty)


def test_schema_from_annotation_pydantic():
    class Out(BaseModel):
        z: float
    spec = schema_from_annotation(Out)
    assert spec.model_class is Out


def test_schema_hash_changes_with_fields():
    s1 = resolve_schema({"a": int})
    s2 = resolve_schema({"a": str})
    assert s1.spec_hash != s2.spec_hash