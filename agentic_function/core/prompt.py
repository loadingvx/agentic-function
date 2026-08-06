"""Prompt rendering — turn (function, args) into an OpenAI-style message list.

The function's docstring becomes the system prompt (with the schema appended).
The bound arguments are formatted into a user message.

We support three optional mechanisms the user can hook into:

* ``prompt_template`` — explicit Jinja2 template string for the user message.
* ``system_template`` — explicit Jinja2 template for the system message.
* ``FewShot`` — list of (input, output) exemplars appended after the system
  message as alternating user/assistant turns.

All Jinja2 usage is optional — if the user doesn't need templating, we just
substitute ``{arg_name}`` placeholders literally using ``str.format``.
"""
from __future__ import annotations

import inspect
from typing import Any, Mapping, Sequence

from ..types import PromptContext


# A few-shot exemplar is a pair of (input_dict, output_dict).
FewShot = tuple[Mapping[str, Any], Mapping[str, Any]]


def _safe_format(template: str, context: dict[str, Any]) -> str:
    """``str.format(**context)`` but tolerant of curly braces in template body.

    Unknown ``{placeholder}`` keys are kept as-is so they show up in the
    prompt as a hint to the user. ``KeyError`` is never raised — the worst
    case is a visible placeholder in the output.
    """
    # We replace any literal ``{`` / ``}`` in the template with doubled versions
    # so they survive .format() — except for our own placeholders.
    import re
    # Tokenise placeholders like ``{name}`` — simple identifier only.
    placeholder_re = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
    out: list[str] = []
    last = 0
    for m in placeholder_re.finditer(template):
        out.append(template[last:m.start()].replace("{", "{{").replace("}", "}}"))
        key = m.group(1)
        if key in context:
            out.append(str(context[key]))
        else:
            # Leave the placeholder intact — visible to the LLM as a hint.
            out.append(m.group(0))
        last = m.end()
    out.append(template[last:].replace("{", "{{").replace("}", "}}"))
    # Use SafeString-style format; ``str.format_map`` falls back to literal for
    # unknown keys when given a default-dict. We approximate that by passing
    # a dict with every identifier mapped to itself first, then real values.
    safe_map = {m.group(1): "{" + m.group(1) + "}" for m in placeholder_re.finditer(template)}
    safe_map.update(context)
    return "".join(out).format_map(safe_map)


def _bind_args(signature: inspect.Signature,
                args: tuple[Any, ...],
                kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a ``{name: value}`` dict for every parameter the function declared."""
    bound = signature.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    return {name: value for name, value in bound.arguments.items()
            # Skip **kwargs / *args spread entries — they're not real params.
            if not (isinstance(value, inspect.Parameter) and
                    value.kind in (inspect.Parameter.VAR_POSITIONAL,
                                   inspect.Parameter.VAR_KEYWORD))}


def _format_arg_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    if isinstance(value, (list, tuple, set)):
        return "\n".join(f"- {_format_arg_value(v)}" for v in value)
    if isinstance(value, dict):
        return "\n".join(f"- {k}: {_format_arg_value(v)}" for k, v in value.items())
    if hasattr(value, "model_dump"):
        try:
            return _format_arg_value(value.model_dump())
        except Exception:
            pass
    return str(value)


def render_prompt(
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    extra: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Render the message list for one invocation."""
    signature = fn.signature
    bound = _bind_args(signature, args, kwargs)

    # ---- system message ----
    system_parts: list[str] = []
    if fn.system_template:
        system_parts.append(_safe_format(fn.system_template, {**bound, **(extra or {})}))
    else:
        if fn.docstring:
            system_parts.append(fn.docstring.strip())
    if fn.output_schema is not None and fn.include_schema_in_prompt:
        schema_json = fn.output_schema.json_schema_str()
        system_parts.append(
            "\n# Output format\n"
            "You MUST respond with a JSON object that EXACTLY matches this schema:\n"
            f"```json\n{schema_json}\n```\n"
            "Do NOT include any prose, markdown fences, or commentary outside the JSON."
        )
    system_message = {"role": "system", "content": "\n\n".join(system_parts) if system_parts else ""}

    # ---- user message ----
    user_context = dict(bound)
    if extra:
        user_context.update(extra)
    if fn.prompt_template:
        user_content = _safe_format(fn.prompt_template, user_context)
    else:
        if not bound:
            # No parameters at all — use docstring as the prompt itself.
            user_content = fn.docstring.strip() if fn.docstring else ""
        else:
            lines = [f"# Arguments"]
            for name, value in bound.items():
                lines.append(f"- {name}: {_format_arg_value(value)}")
            user_content = "\n".join(lines)

    messages: list[dict[str, Any]] = [system_message, {"role": "user", "content": user_content}]

    # ---- few-shot exemplars ----
    for ex_input, ex_output in fn.few_shots:
        ex_user_ctx = {**bound, **(extra or {}), **ex_input}
        if fn.prompt_template:
            ex_user = _safe_format(fn.prompt_template, ex_user_ctx)
        else:
            ex_user = _format_exemplar_input(ex_input)
        ex_assistant = _format_exemplar_output(ex_output)
        messages.append({"role": "user", "content": ex_user})
        messages.append({"role": "assistant", "content": ex_assistant})

    return messages


def _format_exemplar_input(ex: Mapping[str, Any]) -> str:
    return "\n".join(f"- {k}: {_format_arg_value(v)}" for k, v in ex.items())


def _format_exemplar_output(ex: Mapping[str, Any]) -> str:
    import json as _json
    try:
        return _json.dumps(ex, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(ex)