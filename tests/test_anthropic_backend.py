"""Tests for AnthropicBackend and MiniMax alias.

These tests do NOT hit the network — they only verify the request-building
and response-conversion logic. Real LLM calls are exercised by the
``examples/06_real_minimax.py`` script which is opt-in (it requires
``MINIMAX_CN_API_KEY`` to be set in the environment).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agentic_function.backends import anthropic_backend as _ab
from agentic_function.core.schema import resolve_schema
from agentic_function.errors import BackendError


def _spec(decl):
    """Shortcut: build a SchemaSpec from a dict declaration."""
    return resolve_schema(decl)


# Pull the module-private helpers so the test code stays concise.
AnthropicBackend = _ab.AnthropicBackend
_extract_text = _ab._extract_text
_extract_tool_input = _ab._extract_tool_input
_supports_tool_schema = _ab._supports_tool_schema
_response_from_anthropic = _ab._response_from_anthropic


def _messages_to_anthropic(msgs):
    """Call the instance method on a throwaway backend."""
    return AnthropicBackend(api_key="test")._messages_to_anthropic(msgs)


# ---------------------------------------------------------------------------
# _supports_tool_schema
# ---------------------------------------------------------------------------
class TestSupportsToolSchema:
    @pytest.mark.parametrize("model", [
        "claude-3-5-sonnet-latest",
        "claude-3-7-sonnet",
        "claude-opus-4-5",
        "claude-sonnet-4-20250514",
        "MiniMax-M3",
        "MiniMax-M2",
    ])
    def test_supported_models(self, model):
        assert _supports_tool_schema(model) is True

    @pytest.mark.parametrize("model", [
        "gpt-4o-mini",
        "llama-3.1-70b",
        "unknown-model",
    ])
    def test_unsupported_models(self, model):
        assert _supports_tool_schema(model) is False


# ---------------------------------------------------------------------------
# _messages_to_anthropic
# ---------------------------------------------------------------------------
class TestMessagesConversion:
    def test_system_prompts_are_extracted(self):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        system, anthropic = _messages_to_anthropic(msgs)
        assert system == "You are a helpful assistant."
        assert anthropic == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_multiple_system_messages_are_joined(self):
        msgs = [
            {"role": "system", "content": "rule one"},
            {"role": "system", "content": "rule two"},
            {"role": "user", "content": "ok"},
        ]
        system, _ = _messages_to_anthropic(msgs)
        assert "rule one" in system and "rule two" in system

    def test_no_system_returns_none(self):
        msgs = [{"role": "user", "content": "hi"}]
        system, anthropic = _messages_to_anthropic(msgs)
        assert system is None
        assert anthropic == [{"role": "user", "content": "hi"}]

    def test_unknown_role_falls_back_to_system_chunk(self):
        msgs = [
            {"role": "system", "content": "you are"},
            {"role": "function", "content": "tool output"},
            {"role": "user", "content": "hi"},
        ]
        system, anthropic = _messages_to_anthropic(msgs)
        assert "you are" in system
        assert "[function] tool output" in system
        # The user turn is preserved
        assert {"role": "user", "content": "hi"} in anthropic


# ---------------------------------------------------------------------------
# _extract_text / _extract_tool_input (with mock block objects)
# ---------------------------------------------------------------------------
def _make_block(type_: str, **kwargs):
    block = MagicMock()
    block.type = type_
    for k, v in kwargs.items():
        setattr(block, k, v)
    return block


class TestExtractText:
    def test_concatenates_text_blocks(self):
        response = MagicMock()
        response.content = [
            _make_block("text", text="hello "),
            _make_block("text", text="world"),
            _make_block("tool_use", input={"x": 1}),  # ignored
        ]
        assert _extract_text(response) == "hello world"

    def test_empty_response(self):
        response = MagicMock()
        response.content = []
        assert _extract_text(response) == ""


class TestExtractToolInput:
    def test_returns_first_tool_input(self):
        response = MagicMock()
        response.content = [
            _make_block("text", text="ignored"),
            _make_block("tool_use", input={"category": "positive", "score": 0.9}),
        ]
        assert _extract_tool_input(response) == {"category": "positive", "score": 0.9}

    def test_no_tool_use_returns_none(self):
        response = MagicMock()
        response.content = [_make_block("text", text="hello")]
        assert _extract_tool_input(response) is None


# ---------------------------------------------------------------------------
# _response_from_anthropic — full response conversion
# ---------------------------------------------------------------------------
def _make_response(*, content_blocks, model="claude-3-5-sonnet", stop_reason="end_turn",
                   input_tokens=42, output_tokens=13):
    r = MagicMock()
    r.content = content_blocks
    r.model = model
    r.stop_reason = stop_reason
    r.usage = MagicMock()
    r.usage.input_tokens = input_tokens
    r.usage.output_tokens = output_tokens
    return r


class TestResponseConversion:
    def test_tool_use_path_returns_dict(self):
        response = _make_response(
            content_blocks=[
                _make_block("text", text="thinking..."),
                _make_block(
                    "tool_use",
                    input={"label": "positive", "score": 0.95},
                ),
            ],
        )
        out = _response_from_anthropic(response)
        assert out.content == {"label": "positive", "score": 0.95}
        assert out.finish_reason == "end_turn"
        assert out.prompt_tokens == 42
        assert out.completion_tokens == 13
        assert out.total_tokens == 55
        assert out.model == "claude-3-5-sonnet"

    def test_text_json_path_parses(self):
        response = _make_response(
            content_blocks=[
                _make_block("text", text='{"label": "negative", "score": 0.1}'),
            ],
        )
        out = _response_from_anthropic(response)
        assert out.content == {"label": "negative", "score": 0.1}

    def test_text_with_markdown_fence_is_stripped(self):
        response = _make_response(
            content_blocks=[
                _make_block("text", text='```json\n{"label": "neutral"}\n```'),
            ],
        )
        out = _response_from_anthropic(response)
        assert out.content == {"label": "neutral"}

    def test_non_json_text_falls_back_to_raw_string(self):
        response = _make_response(
            content_blocks=[
                _make_block("text", text="I'm sorry, I can't help with that."),
            ],
        )
        out = _response_from_anthropic(response)
        # We return the raw text; downstream validation will reject it.
        assert out.content == "I'm sorry, I can't help with that."

    def test_tool_use_wins_over_text(self):
        response = _make_response(
            content_blocks=[
                _make_block("text", text='{"x": 1}'),  # would parse, but…
                _make_block("tool_use", input={"x": 2}),  # …tool wins
            ],
        )
        out = _response_from_anthropic(response)
        assert out.content == {"x": 2}


# ---------------------------------------------------------------------------
# AnthropicBackend end-to-end with mocked client
# ---------------------------------------------------------------------------
class TestAnthropicBackendE2E:
    def _patched_complete(self, backend, monkeypatch, fake_response):
        """Helper: patch anthropic.Anthropic to return ``fake_response``."""
        captured = {}

        class FakeAnthropic:
            def __init__(self, **kwargs):
                captured["client_kwargs"] = kwargs

            def messages(self):  # not used; placeholder
                return self

        # We patch the import path used by the backend module.
        monkeypatch.setattr(
            "anthropic.Anthropic",
            lambda **kwargs: FakeAnthropic(**kwargs),
        )

        # Direct-mock: replace the messages.create call.
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_response
        monkeypatch.setattr(
            "anthropic.Anthropic",
            lambda **kwargs: fake_client,
        )
        return captured, fake_client

    def test_complete_uses_tool_schema_for_claude(self, monkeypatch):
        schema = _spec({"label": str, "score": float})
        fake_response = _make_response(
            content_blocks=[
                _make_block("tool_use", input={"label": "positive", "score": 0.9}),
            ],
        )
        _, fake_client = self._patched_complete(None, monkeypatch, fake_response)

        backend = AnthropicBackend(api_key="test-key")
        out = backend.complete(
            model="claude-3-5-sonnet-latest",
            messages=[
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ],
            output_schema=schema,
            temperature=0.0,
        )
        # Verify the request was shaped correctly
        call_kwargs = fake_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-3-5-sonnet-latest"
        assert call_kwargs["system"].startswith("be terse")
        assert "Respond with ONLY a single JSON object" in call_kwargs["system"]
        assert call_kwargs["max_tokens"] == 1024  # default
        # Claude supports tool schemas → tools were attached
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == {
            "type": "tool", "name": "agentic_function_output"
        }
        # Response was normalised
        assert out.content == {"label": "positive", "score": 0.9}
        assert out.prompt_tokens == 42
        assert out.completion_tokens == 13

    def test_unknown_model_falls_back_to_json_only_prompt(self, monkeypatch):
        schema = _spec({"x": int})
        fake_response = _make_response(
            content_blocks=[_make_block("text", text='{"x": 7}')],
        )
        _, fake_client = self._patched_complete(None, monkeypatch, fake_response)

        backend = AnthropicBackend(api_key="test-key")
        backend.complete(
            model="llama-3.1-70b",
            messages=[{"role": "user", "content": "hi"}],
            output_schema=schema,
        )
        call_kwargs = fake_client.messages.create.call_args.kwargs
        # No tools because the model isn't recognised
        assert "tools" not in call_kwargs
        # But the JSON suffix is still in the system message
        assert "Respond with ONLY a single JSON object" in call_kwargs["system"]

    def test_default_system_is_injected(self, monkeypatch):
        schema = _spec({"x": int})
        fake_response = _make_response(
            content_blocks=[_make_block("text", text='{"x": 1}')],
        )
        _, fake_client = self._patched_complete(None, monkeypatch, fake_response)

        backend = AnthropicBackend(api_key="test-key")
        backend.complete(
            model="llama-3.1-70b",
            messages=[{"role": "user", "content": "hi"}],
            output_schema=schema,
        )
        call_kwargs = fake_client.messages.create.call_args.kwargs
        # Even with no explicit system message, one is added.
        assert "helpful assistant" in call_kwargs["system"]

    def test_sdk_missing_raises_backend_error(self, monkeypatch):
        # Force the SDK import to fail.
        import builtins
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("simulated missing anthropic")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        backend = AnthropicBackend(api_key="test-key")
        schema = _spec({"x": int})
        with pytest.raises(BackendError) as excinfo:
            backend.complete(
                model="claude-3-5-sonnet-latest",
                messages=[{"role": "user", "content": "hi"}],
                output_schema=schema,
            )
        assert "anthropic" in str(excinfo.value).lower()

    def test_base_url_and_timeout_propagate(self, monkeypatch):
        schema = _spec({"x": int})
        fake_response = _make_response(
            content_blocks=[_make_block("text", text='{"x": 1}')],
        )
        captured: dict[str, Any] = {}
        per_call_capture: dict[str, Any] = {}

        def fake_constructor(**kwargs):
            captured.update(kwargs)
            client = MagicMock()
            client.messages.create.side_effect = (
                lambda **call_kwargs: (
                    per_call_capture.update(call_kwargs) or fake_response
                )
            )
            return client

        monkeypatch.setattr("anthropic.Anthropic", fake_constructor)

        backend = AnthropicBackend(
            api_key="k", base_url="https://custom.example/v1",
            request_timeout=12.0,
        )
        backend.complete(
            model="claude-3-5-sonnet-latest",
            messages=[{"role": "user", "content": "hi"}],
            output_schema=schema,
        )
        assert captured["base_url"] == "https://custom.example/v1"
        # The per-call timeout must be passed through to messages.create.
        assert per_call_capture["timeout"] == 12.0


# ---------------------------------------------------------------------------
# Public-API smoke: ``known_backends`` includes "anthropic" and "minimax"
# ---------------------------------------------------------------------------
class TestPublicRegistry:
    def test_anthropic_and_minimax_registered(self):
        from agentic_function.backends import known_backends, get_backend
        names = known_backends()
        assert "anthropic" in names
        assert "minimax" in names
        assert "mock" in names
        assert "openai" in names

    def test_minimax_backend_default_url_and_model(self):
        import os
        os.environ.pop("MINIMAX_BASE_URL", None)
        from agentic_function.backends import get_backend
        b = get_backend("minimax")
        # base_url is the MiniMax Anthropic-compatible endpoint
        assert b.base_url == "https://api.minimaxi.com/anthropic"
        # default model is MiniMax-M3
        assert b.default_model == "MiniMax-M3"
        assert b.name == "minimax"