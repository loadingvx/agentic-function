"""Cache backends.

We ship two implementations:
- ``InMemoryCache`` — fast, per-process, lost on restart. Good default for tests
  and short-lived scripts.
- ``DiskCache`` — persists across runs in a directory you choose.

A ``NullCache`` is also exposed for users who want to disable caching
per-function (``cache=False``).
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..errors import CacheError
from ..utils.hashing import stable_hash


@dataclass(frozen=True)
class CacheEntry:
    """A single cached call's result."""
    key: str
    value: Any                          # the validated AgenticResult
    raw_output: Any                     # the raw LLM string/dict
    created_at: float
    model: str
    prompt_hash: str
    usage: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": _make_serialisable(self.value),
            "raw_output": _make_serialisable(self.raw_output),
            "created_at": self.created_at,
            "model": self.model,
            "prompt_hash": self.prompt_hash,
            "usage": self.usage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheEntry":
        return cls(
            key=data["key"],
            value=data["value"],
            raw_output=data["raw_output"],
            created_at=data["created_at"],
            model=data["model"],
            prompt_hash=data["prompt_hash"],
            usage=data["usage"],
        )


def _make_serialisable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return {"__pydantic__": True, "data": value.model_dump()}
        except Exception:  # pragma: no cover
            pass
    return value


def _restore(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__pydantic__"):
        # Re-hydrate lazily — the caller knows the target model.
        return value["data"]
    return value


class CacheBackend(ABC):
    """Abstract cache interface."""

    @abstractmethod
    def get(self, key: str) -> CacheEntry | None: ...

    @abstractmethod
    def set(self, key: str, entry: CacheEntry, *, ttl_seconds: float | None = None) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...

    def make_key(self, *parts: Any) -> str:
        return stable_hash(*parts)


class NullCache(CacheBackend):
    """Cache that stores nothing."""

    def get(self, key: str) -> CacheEntry | None:
        return None

    def set(self, key: str, entry: CacheEntry, *, ttl_seconds: float | None = None) -> None:
        return

    def clear(self) -> None:
        return


class InMemoryCache(CacheBackend):
    """Thread-safe in-process cache."""

    def __init__(self, *, max_size: int = 1024) -> None:
        self._store: dict[str, CacheEntry] = {}
        self._expiry: dict[str, float] = {}
        self._max_size = max_size
        self._lock = threading.RLock()

    def get(self, key: str) -> CacheEntry | None:
        with self._lock:
            exp = self._expiry.get(key)
            if exp is not None and exp < time.time():
                self._store.pop(key, None)
                self._expiry.pop(key, None)
                return None
            return self._store.get(key)

    def set(self, key: str, entry: CacheEntry, *, ttl_seconds: float | None = None) -> None:
        with self._lock:
            if len(self._store) >= self._max_size:
                # Simple FIFO eviction — remove the oldest key.
                oldest_key = next(iter(self._store))
                self._store.pop(oldest_key, None)
                self._expiry.pop(oldest_key, None)
            self._store[key] = entry
            if ttl_seconds is not None:
                self._expiry[key] = time.time() + ttl_seconds

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._expiry.clear()


class DiskCache(CacheBackend):
    """File-system cache. One JSON file per key.

    Suitable for development and small-scale caching. Not optimised for huge
    caches or concurrent access — use Redis for that.
    """

    def __init__(self, directory: str) -> None:
        self._dir = os.path.abspath(directory)
        os.makedirs(self._dir, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, key: str) -> str:
        return os.path.join(self._dir, f"{key}.json")

    def get(self, key: str) -> CacheEntry | None:
        with self._lock:
            path = self._path(key)
            if not os.path.exists(path):
                return None
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                entry = CacheEntry.from_dict(data)
                # Check TTL stored as ``expires_at``.
                exp = data.get("expires_at")
                if exp is not None and exp < time.time():
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    return None
                return entry
            except (json.JSONDecodeError, OSError, KeyError) as exc:
                raise CacheError(f"failed to read cache entry {key}: {exc}") from exc

    def set(self, key: str, entry: CacheEntry, *, ttl_seconds: float | None = None) -> None:
        with self._lock:
            data = entry.to_dict()
            if ttl_seconds is not None:
                data["expires_at"] = time.time() + ttl_seconds
            tmp_path = self._path(key) + ".tmp"
            try:
                with open(tmp_path, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False, default=str)
                os.replace(tmp_path, self._path(key))
            except OSError as exc:
                raise CacheError(f"failed to write cache entry {key}: {exc}") from exc

    def clear(self) -> None:
        with self._lock:
            for name in os.listdir(self._dir):
                if name.endswith(".json"):
                    try:
                        os.remove(os.path.join(self._dir, name))
                    except OSError:
                        pass


# ----------------------------------------------------------------------
# Process-wide singleton accessor
# ----------------------------------------------------------------------
_active_cache: CacheBackend = InMemoryCache()


def get_cache() -> CacheBackend:
    return _active_cache


def set_cache(cache: CacheBackend) -> None:
    global _active_cache
    _active_cache = cache


def default_cache() -> CacheBackend:
    """Return a sensible default cache based on global config."""
    from .config import global_config
    cfg = global_config()
    if not cfg.cache_enabled:
        return NullCache()
    if cfg.cache_dir:
        return DiskCache(cfg.cache_dir)
    return InMemoryCache()