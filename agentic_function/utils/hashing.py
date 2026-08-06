"""Stable hashing for cache keys and trace IDs.

We need hashes that:
1. Don't change across Python runs (so ``hash((1, "a"))`` is unsuitable — its
   seed is randomised).
2. Treat equivalent JSON structures as equal (``{"a": 1}`` == ``{'a': 1}``).
3. Handle non-JSON-serialisable objects gracefully (pydantic models, dataclasses,
   custom classes with ``__str__`` fallback).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert ``obj`` into something ``json.dumps`` accepts."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, Mapping):
        # Sort keys for determinism.
        return {str(k): _to_jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return _to_jsonable(obj.model_dump())
        except Exception:  # pragma: no cover — defensive
            pass
    # Dataclass-ish
    if hasattr(obj, "__dict__"):
        return _to_jsonable({k: v for k, v in vars(obj).items() if not k.startswith("_")})
    # Last resort — string repr. Won't be canonical across versions but stable
    # enough for cache-busting purposes within a session.
    return repr(obj)


def stable_hash(*parts: Any) -> str:
    """Return a SHA-256 hex digest of the stable JSON representation of ``parts``."""
    serialised = json.dumps([_to_jsonable(p) for p in parts], ensure_ascii=False,
                            sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def hash_inputs(*args: Any, **kwargs: Any) -> str:
    """Convenience: hash a function call's positional + keyword arguments."""
    return stable_hash(args, kwargs)