"""Shared logging configuration for the GameProofer sample project."""

from __future__ import annotations

import logging
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once per process."""
    if getattr(configure_logging, "_configured", False):
        return

    logging.basicConfig(level=level, format=_LOG_FORMAT)
    configure_logging._configured = True  # type: ignore[attr-defined]


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-level logger with consistent formatting."""
    configure_logging()
    return logging.getLogger(name if name else "gameproofer")
