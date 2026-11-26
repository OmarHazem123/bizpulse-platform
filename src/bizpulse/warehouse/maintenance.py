"""Warehouse maintenance: storage statistics and compaction.

DuckDB reuses freed blocks but never shrinks the file. Primary-key indexes are rewritten
at every checkpoint during loads, which leaves free blocks behind. :func:`compact` copies
the database into a fresh file (constraints and indexes included) and atomically swaps it
in, keeping the warehouse inside the project's 50 MB artifact budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class StorageStats:
    file_bytes: int
    used_blocks: int
    free_blocks: int

    @property
    def free_ratio(self) -> float:
        total = self.used_blocks + self.free_blocks
        return 0.0 if total == 0 else self.free_blocks / total

    @property
    def file_mb(self) -> float:
        return self.file_bytes / 1024 / 1024


def storage_stats(path: Path) -> StorageStats:
    con = duckdb.connect(str(path), read_only=True)
    try:
        row = con.execute("SELECT used_blocks, free_blocks FROM pragma_database_size()").fetchone()
    finally:
        con.close()
    used, free = (int(row[0]), int(row[1])) if row else (0, 0)
    return StorageStats(file_bytes=path.stat().st_size, used_blocks=used, free_blocks=free)


def _literal(path: Path) -> str:
    """SQL string literal for a filesystem path (single quotes doubled)."""
    return "'" + str(path).replace("'", "''") + "'"


def compact(path: Path) -> tuple[int, int]:
    """Rewrite *path* without free blocks. Returns ``(bytes_before, bytes_after)``.

    The caller must not hold an open connection to *path* (Windows file locks).
    """
    before = path.stat().st_size
    tmp = path.with_name(path.stem + ".compacting.duckdb")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(":memory:")
    try:
        # ATTACH does not accept bound parameters: inline escaped string literals.
        con.execute(f"ATTACH {_literal(path)} AS src (READ_ONLY)")
        con.execute(f"ATTACH {_literal(tmp)} AS dst")
        con.execute("COPY FROM DATABASE src TO dst")
        con.execute("DETACH dst")
        con.execute("DETACH src")
    except Exception:
        con.close()
        tmp.unlink(missing_ok=True)
        raise
    con.close()
    tmp.replace(path)
    return before, path.stat().st_size


def compact_if_fragmented(path: Path, threshold: float = 0.25) -> tuple[int, int] | None:
    """Compact when more than *threshold* of the blocks are free; else return ``None``."""
    if not path.exists():
        return None
    stats = storage_stats(path)
    if stats.free_ratio <= threshold:
        return None
    return compact(path)
