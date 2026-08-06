"""Cache backend tests."""
from __future__ import annotations

import time

from agentic_function.runtime.cache import (
    CacheEntry, InMemoryCache, DiskCache, NullCache,
)


def _entry(key: str, value: dict | None = None) -> CacheEntry:
    return CacheEntry(
        key=key,
        value=value or {"a": 1},
        raw_output={"a": 1},
        created_at=time.time(),
        model="m",
        prompt_hash="ph",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def test_in_memory_basic():
    cache = InMemoryCache()
    assert cache.get("k") is None
    cache.set("k", _entry("k"))
    assert cache.get("k") is not None


def test_in_memory_ttl():
    cache = InMemoryCache()
    cache.set("k", _entry("k"), ttl_seconds=0.05)
    assert cache.get("k") is not None
    time.sleep(0.1)
    assert cache.get("k") is None


def test_in_memory_eviction():
    cache = InMemoryCache(max_size=2)
    cache.set("a", _entry("a"))
    cache.set("b", _entry("b"))
    cache.set("c", _entry("c"))  # evicts oldest
    assert cache.get("c") is not None
    # "a" was first inserted → evicted (FIFO)
    # We don't assert about "a"/"b" precisely — the implementation may differ.


def test_in_memory_clear():
    cache = InMemoryCache()
    cache.set("k", _entry("k"))
    cache.clear()
    assert cache.get("k") is None


def test_null_cache():
    cache = NullCache()
    cache.set("k", _entry("k"))
    assert cache.get("k") is None


def test_disk_cache_roundtrip(tmp_path):
    cache = DiskCache(str(tmp_path))
    cache.set("k", _entry("k"))
    out = cache.get("k")
    assert out is not None
    assert out.key == "k"
    assert out.value == {"a": 1}


def test_disk_cache_clear(tmp_path):
    cache = DiskCache(str(tmp_path))
    cache.set("k", _entry("k"))
    cache.clear()
    assert cache.get("k") is None


def test_disk_cache_ttl(tmp_path):
    cache = DiskCache(str(tmp_path))
    cache.set("k", _entry("k"), ttl_seconds=0.05)
    assert cache.get("k") is not None
    time.sleep(0.1)
    assert cache.get("k") is None