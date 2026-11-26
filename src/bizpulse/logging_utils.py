"""Logging configuration shared by the CLI, pipeline and dashboard."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Idempotently configure root logging for console use."""
    root = logging.getLogger()
    if not any(getattr(h, "_bizpulse", False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        handler._bizpulse = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
