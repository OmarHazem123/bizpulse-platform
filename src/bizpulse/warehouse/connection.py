"""DuckDB connection factory.

Connections are always closed deterministically: on Windows an open handle locks the
database file, which breaks re-runs, temp-dir cleanup and concurrent readers.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from bizpulse.warehouse.schema import create_schema


@contextmanager
def connect(path: Path, *, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open the warehouse (creating the schema for writers) and always close it."""
    if read_only and not path.exists():
        raise FileNotFoundError(f"warehouse not found: {path.name} (run the pipeline first)")
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    try:
        if not read_only:
            create_schema(con)
        yield con
    finally:
        con.close()
