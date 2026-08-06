"""Coercion tests."""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from agentic_function.validation.coerce import coerce_value, coerce_to_schema


def test_coerce_bool_string():
    assert coerce_value("yes", bool) is True
    assert coerce_value("no", bool) is False
    assert coerce_value("true", bool) is True
    assert coerce_value("0", bool) is False


def test_coerce_int_string():
    assert coerce_value("42", int) == 42
    assert coerce_value("3.0", int) == 3


def test_coerce_float_string():
    assert coerce_value("3.14", float) == 3.14


def test_coerce_string_passes_through():
    assert coerce_value(42, str) == "42"
    assert coerce_value(None, str) is None


def test_coerce_list_from_csv():
    result = coerce_value("a,b,c", list[str])
    assert result == ["a", "b", "c"]


def test_coerce_dict_to_dict():
    result = coerce_value({"a": 1}, dict[str, int])
    assert result == {"a": 1}


def test_coerce_to_schema():
    class M(BaseModel):
        x: int
        y: str

    m = coerce_to_schema({"x": "1", "y": "hi"}, M)
    assert isinstance(m, M)
    assert m.x == 1 and m.y == "hi"


def test_coerce_invalid_raises():
    with pytest.raises(Exception):
        coerce_value("not_a_bool", bool)