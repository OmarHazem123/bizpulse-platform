"""Packaged default configuration (FX table and demo dataset mappings)."""

from __future__ import annotations

from importlib.resources import files
from importlib.resources.abc import Traversable


def resource(*parts: str) -> Traversable:
    """Return a handle to a file shipped inside ``bizpulse/resources``."""
    node: Traversable = files(__name__)
    for part in parts:
        node = node.joinpath(part)
    return node


def read_text(*parts: str) -> str:
    return resource(*parts).read_text(encoding="utf-8")


def builtin_mapping_names() -> list[str]:
    return sorted(
        entry.name.removesuffix(".yaml")
        for entry in resource("mappings").iterdir()
        if entry.name.endswith(".yaml")
    )
