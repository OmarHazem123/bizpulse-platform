"""SHA-256 based keys for idempotent batch and row loading.

* ``file_sha256``: batch identity (same bytes -> same batch).
* ``transaction_key``: natural-key identity of a fact row within a dataset.
* ``entity_key``: surrogate keys for dimensions.
* ``row_hashes``: content identity of a normalized row, insensitive to source formatting
  (``"$1,299.00"`` and ``1299`` hash identically once normalized).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

_FIELD_SEP = "\x1f"
_NULL = "\x00"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def key64(text: str) -> int:
    """Signed 64-bit surrogate key: the first 8 bytes of SHA-256(*text*).

    Compact BIGINT keys keep DuckDB's primary-key indexes and joins small and fast.
    The collision probability is ~N^2 / 2^65 (about 3e-8 at one million rows).
    """
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big", signed=True)


def transaction_key(dataset: str, transaction_id: str) -> int:
    return key64(f"{dataset}|{transaction_id}")


def entity_key(dataset: str, kind: str, natural_id: str) -> int:
    return key64(f"{dataset}|{kind}|{natural_id}")


def canonical_token(value: Any) -> str:
    """Stable text form of one normalized value (floats rounded to 6 dp)."""
    if value is None:
        return _NULL
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if math.isnan(value):
            return _NULL
        return f"{round(value, 6):.6f}"
    if isinstance(value, int):
        return f"{float(value):.6f}"
    if isinstance(value, pd.Timestamp):
        return str(value.isoformat())
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def row_hash(values: Iterable[Any]) -> str:
    return sha256_text(_FIELD_SEP.join(canonical_token(v) for v in values))


def row_hashes(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Row-content hashes of *frame* over *columns* (in the given order)."""
    subset = frame.loc[:, list(columns)].astype(object)
    return [row_hash(row) for row in subset.itertuples(index=False, name=None)]
