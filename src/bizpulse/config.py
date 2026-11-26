"""Runtime settings and filesystem layout.

All generated artifacts live under one data directory (default ``./data``), which can be
relocated with the ``BIZPULSE_DATA_DIR`` environment variable. Nothing in the code base
hard-codes an absolute path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_DATA_DIR = "BIZPULSE_DATA_DIR"
WAREHOUSE_FILENAME = "bizpulse.duckdb"

#: Demo datasets and the raw file each one is generated into / ingested from.
DEMO_FILES: dict[str, str] = {
    "ecommerce": "ecommerce_orders.csv",
    "saas": "saas_invoices.xlsx",
    "retail_fb": "retail_fb_pos.parquet",
}


@dataclass(frozen=True)
class Settings:
    """Resolved filesystem layout for one BizPulse installation."""

    data_dir: Path

    @classmethod
    def from_env(cls, data_dir: Path | str | None = None) -> Settings:
        """Build settings from an explicit directory, ``$BIZPULSE_DATA_DIR`` or ``./data``."""
        raw = data_dir if data_dir is not None else os.environ.get(ENV_DATA_DIR)
        base = Path(raw) if raw else Path.cwd() / "data"
        return cls(data_dir=base.resolve())

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def manifests_dir(self) -> Path:
        return self.raw_dir / "manifests"

    @property
    def warehouse_path(self) -> Path:
        return self.data_dir / "warehouse" / WAREHOUSE_FILENAME

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    def ensure_dirs(self) -> None:
        for path in (self.raw_dir, self.warehouse_path.parent, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)

    def relative(self, path: Path) -> str:
        """Render *path* relative to the data dir (never leak absolute local paths)."""
        try:
            return path.resolve().relative_to(self.data_dir).as_posix()
        except ValueError:
            return path.name
