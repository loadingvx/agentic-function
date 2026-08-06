"""Backend tests — primarily MockBackend since OpenAI requires network/keys."""
from __future__ import annotations

import json

import pytest

from agentic_function import MockBackend, register_backend, get_backend
from agentic_function.backends.base import LLMResponse
from agentic_function.core.schema import resolve_schema
from agentic_function.errors import BackendError, RegistrationError


def test_mock_register_catchall():
    backend = MockBackend()
    calls = []
    def handler(req):
        calls.append(req)
        return {"ok": True}
    backend.register(handler)
    schema = resolve_schema({"ok": bool})
    resp = backend.complete(model="m", messages=[{"role": "user", "content": "x"}],
                            output_schema=schema)
    assert isinstance(resp, LLMResponse)
    assert resp.content == {"ok": True}
    assert calls  # handler was invoked


def test_mock_register_with_signature():
    from agentic_function.utils.hashing import stable_hash

    backend = MockBackend()
    schema = resolve_schema({"v": int})
    msg_hash = stable_hash("x")[:16]
    sig = ("m", (("user", msg_hash),))
    backend.register(lambda req: {"v": 42}, signature=sig)
    out = backend.complete(model="m", messages=[{"role": "user", "content": "x"}],
                           output_schema=schema)
    assert out.content == {"v": 42}


def test_mock_register_with_schema():
    backend = MockBackend()
    schema = resolve_schema({"label": str, "score": float})
    backend.register_with_schema(schema)
    out = backend.complete(model="m", messages=[{"role": "user", "content": "x"}],
                           output_schema=schema)
    assert out.content == {"label": "mock_label", "score": 0.5}


def test_mock_force_string_output():
    backend = MockBackend()
    backend.force_string_output = True
    backend.register(lambda req: {"x": 1})
    schema = resolve_schema({"x": int})
    out = backend.complete(model="m", messages=[{"role": "user", "content": "x"}],
                           output_schema=schema)
    assert isinstance(out.content, str)
    parsed = json.loads(out.content)
    assert parsed == {"x": 1}


def test_mock_default_response():
    backend = MockBackend()
    backend.default_response = {"label": "default", "score": 0.1}
    schema = resolve_schema({"label": str, "score": float})
    out = backend.complete(model="m", messages=[{"role": "user", "content": "x"}],
                           output_schema=schema)
    assert out.content == {"label": "default", "score": 0.1}


def test_register_and_get_backend():
    backend = MockBackend()
    register_backend("test_backend", backend)
    assert get_backend("test_backend") is backend
    with pytest.raises(BackendError):
        get_backend("does_not_exist")


def test_register_invalid_name():
    with pytest.raises(RegistrationError):
        register_backend("", MockBackend())


def test_register_invalid_backend():
    class NotABackend:
        pass
    with pytest.raises(RegistrationError):
        register_backend("nope", NotABackend())