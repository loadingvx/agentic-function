"""Lightweight structured logging.

We avoid pulling in ``structlog`` to keep the dependency surface small. The
default format is plain ``key=value`` pairs — easy to grep, easy to parse.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any


_DEFAULT_LEVEL = os.environ.get("AGENTIC_FUNCTION_LOG_LEVEL", "WARNING").upper()
_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"


_configured = False


def configure_logging(level: str | int | None = None, fmt: str | None = None) -> None:
    """Idempotently configure the root ``agentic_function`` logger."""
    global _configured
    if _configured:
        return
    logger = logging.getLogger("agentic_function")
    if level is not None:
        logger.setLevel(level)
    else:
        logger.setLevel(_DEFAULT_LEVEL)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(fmt or _DEFAULT_FORMAT))
        logger.addHandler(handler)
    logger.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``agentic_function`` namespace."""
    configure_logging()
    return logging.getLogger(f"agentic_function.{name}" if not name.startswith("agentic_function")
                             else name)


def kv(**fields: Any) -> str:
    """Format keyword arguments as ``key=value`` pairs, used for log messages."""
    parts: list[str] = []
    for k, v in fields.items():
        if isinstance(v, str) and (" " in v or "\n" in v):
            parts.append(f'{k}="{v}"')
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)