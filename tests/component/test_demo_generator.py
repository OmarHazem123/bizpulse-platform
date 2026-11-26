from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from bizpulse.config import DEMO_FILES
from bizpulse.demo.common import WINDOW_END, WINDOW_START
from bizpulse.demo.generator import generate_dataset, generate_demo_data
from bizpulse.ingestion.readers import read_table


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.slow
def test_generation_is_byte_identical_for_the_same_seed(tmp_path: Path) -> None:
    first = generate_demo_data(tmp_path / "a", seed=42, scale=0.1)
    second = generate_demo_data(tmp_path / "b", seed=42, scale=0.1)
    for a, b in zip(first, second, strict=True):
        assert _digest(a.path) == _digest(b.path), a.dataset
        assert _digest(a.manifest_path) == _digest(b.manifest_path), a.dataset


def test_different_seed_gives_different_data(tmp_path: Path) -> None:
    a = generate_dataset("saas", tmp_path / "a", seed=42, scale=0.1)
    b = generate_dataset("saas", tmp_path / "b", seed=7, scale=0.1)
    assert _digest(a.path) != _digest(b.path)


def test_unknown_dataset_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown demo dataset"):
        generate_dataset("crypto", tmp_path)


@pytest.mark.parametrize("dataset", sorted(DEMO_FILES))
def test_demo_files_and_manifests(demo_raw: Path, dataset: str) -> None:
    manifest = json.loads((demo_raw / "manifests" / f"{dataset}.json").read_text(encoding="utf-8"))
    table = read_table(demo_raw / DEMO_FILES[dataset])
    assert manifest["rows_total"] == len(table.frame)
    assert manifest["window"] == {"start": str(WINDOW_START), "end": str(WINDOW_END)}
    assert manifest["rows_dirty"] == len(manifest["dirty_rows"])
    assert 0.005 < manifest["rows_dirty"] / manifest["rows_total"] < 0.03
    assert all(len(entry["codes"]) == 1 for entry in manifest["dirty_rows"])
    assert len(manifest["defect_counts"]) >= 16
    assert manifest["anomalies"], "every dataset carries injected anomalies"
    for anomaly in manifest["anomalies"]:
        start, end = pd.Timestamp(anomaly["start"]), pd.Timestamp(anomaly["end"])
        assert pd.Timestamp(WINDOW_START) <= start <= end <= pd.Timestamp(WINDOW_END)
