from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from bizpulse.ingestion.hashing import (
    canonical_token,
    entity_key,
    key64,
    row_hash,
    row_hashes,
    sha256_file,
    transaction_key,
)


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    payload = b"bizpulse" * 300_000
    path.write_bytes(payload)
    assert sha256_file(path, chunk_size=4096) == hashlib.sha256(payload).hexdigest()


def test_key64_is_deterministic_signed_64bit_prefix_of_sha256() -> None:
    digest = hashlib.sha256(b"acme|T1").digest()
    expected = int.from_bytes(digest[:8], "big", signed=True)
    assert key64("acme|T1") == expected == transaction_key("acme", "T1")
    assert -(2**63) <= expected < 2**63


def test_keys_are_scoped_by_dataset_and_kind() -> None:
    assert transaction_key("a", "1") != transaction_key("b", "1")
    assert entity_key("a", "customer", "1") != entity_key("a", "product", "1")


def test_row_hash_is_format_insensitive_after_normalization() -> None:
    assert row_hash([1299, "x", None]) == row_hash([1299.0, "x", float("nan")])
    assert row_hash([0.1 + 0.2]) == row_hash([0.3])
    assert row_hash([1.0]) != row_hash([1.000002])
    assert len(row_hash(["a"])) == 64


def test_canonical_token_handles_timestamps_and_bools() -> None:
    ts = pd.Timestamp("2026-01-02 03:04:05")
    assert canonical_token(ts) == "2026-01-02T03:04:05"
    assert canonical_token(True) == "1"


def test_row_hashes_respect_column_order() -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    forward = row_hashes(frame, ["a", "b"])
    backward = row_hashes(frame, ["b", "a"])
    assert forward[0] != backward[0]
    assert forward == [row_hash([1, "x"]), row_hash([2, "y"])]
