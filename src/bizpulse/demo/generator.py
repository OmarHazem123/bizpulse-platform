"""Generate the three deterministic demo datasets plus their ground-truth manifests."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bizpulse.config import DEMO_FILES
from bizpulse.demo import ecommerce, retail_fb, saas
from bizpulse.demo.common import (
    WINDOW_END,
    WINDOW_START,
    DemoTable,
    Manifest,
    inject_defects,
    write_csv,
    write_parquet,
    write_xlsx,
)
from bizpulse.ingestion.fx import FxTable

DEFAULT_SEED = 42
Simulator = Callable[[np.random.Generator, float, FxTable], DemoTable]
SIMULATORS: dict[str, Simulator] = {
    "ecommerce": ecommerce.generate,
    "saas": saas.generate,
    "retail_fb": retail_fb.generate,
}
_STREAM = {"ecommerce": 1, "saas": 2, "retail_fb": 3}  # independent RNG streams per dataset


@dataclass(frozen=True)
class GeneratedFile:
    dataset: str
    path: Path
    manifest_path: Path
    rows: int
    dirty_rows: int


def generate_dataset(
    dataset: str, out_dir: Path, *, seed: int = DEFAULT_SEED, scale: float = 1.0
) -> GeneratedFile:
    if dataset not in SIMULATORS:
        raise ValueError(f"unknown demo dataset {dataset!r}; choose from {sorted(SIMULATORS)}")
    rng = np.random.default_rng([seed, _STREAM[dataset]])
    table = SIMULATORS[dataset](rng, scale, FxTable.default())
    file_name = DEMO_FILES[dataset]
    path = out_dir / file_name
    if path.suffix == ".csv":
        dirty, entries = inject_defects(table, rng, blank="")
        write_csv(dirty, path)
        fmt = "csv"
    elif path.suffix == ".xlsx":
        dirty, entries = inject_defects(table, rng, blank=None)
        write_xlsx(dirty, path, sheet_name="Invoices")
        fmt = "excel"
    else:
        dirty, entries = inject_defects(table, rng, blank=None)
        write_parquet(dirty, path)
        fmt = "parquet"
    defect_counts = Counter(code for entry in entries for code in entry["codes"])
    manifest = Manifest(
        dataset=dataset,
        file=file_name,
        format=fmt,
        seed=seed,
        scale=scale,
        window={"start": WINDOW_START.isoformat(), "end": WINDOW_END.isoformat()},
        rows_total=len(dirty),
        rows_dirty=len(entries),
        rows_expected_valid=len(dirty) - len(entries),
        defect_counts=dict(sorted(defect_counts.items())),
        dirty_rows=entries,
        messy_valid=table.messy_valid,
        anomalies=[asdict(a) for a in table.anomalies],
    )
    manifest_path = out_dir / "manifests" / f"{dataset}.json"
    manifest.write(manifest_path)
    return GeneratedFile(dataset, path, manifest_path, len(dirty), len(entries))


def generate_demo_data(
    out_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    scale: float = 1.0,
    datasets: Iterable[str] | None = None,
) -> list[GeneratedFile]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        generate_dataset(name, out_dir, seed=seed, scale=scale) for name in (datasets or SIMULATORS)
    ]
