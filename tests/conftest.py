"""Shared fixtures. Tests never touch the real ``./data`` directory."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from bizpulse.config import Settings
from bizpulse.demo.generator import generate_demo_data
from bizpulse.ingestion.fx import FxTable
from bizpulse.ingestion.mapping import MappingConfig

AS_OF = datetime(2026, 11, 26, 12, 0, 0)
DEMO_SCALE = 0.25


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    resolved = Settings.from_env(tmp_path / "data")
    resolved.ensure_dirs()
    return resolved


@pytest.fixture(scope="session")
def fx() -> FxTable:
    return FxTable.default()


@pytest.fixture(scope="session")
def demo_raw(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Demo files generated once per test session at reduced scale."""
    out = tmp_path_factory.mktemp("demo") / "raw"
    generate_demo_data(out, seed=42, scale=DEMO_SCALE)
    return out


@pytest.fixture
def basic_config() -> MappingConfig:
    return MappingConfig(dataset="acme", reporting_currency="USD")


def make_rows(*overrides: dict[str, Any]) -> pd.DataFrame:
    """A small, valid source frame in canonical header names; overrides patch rows."""
    base: dict[str, Any] = {
        "transaction_id": "T1",
        "order_id": "O1",
        "order_date": "2026-03-01 10:00:00",
        "customer_id": "C1",
        "product_id": "P1",
        "product_name": "Widget",
        "category": "Gadgets",
        "channel": "Web Shop",
        "region": "EU",
        "quantity": "2",
        "unit_price": "10.00",
        "discount_amount": "1.00",
        "unit_cost": "4.00",
        "fee_amount": "0.50",
        "currency": "USD",
        "status": "paid",
    }
    rows = []
    for i, patch in enumerate(overrides or ({},), start=1):
        row = {**base, "transaction_id": f"T{i}", "order_id": f"O{i}"}
        row.update(patch)
        rows.append(row)
    return pd.DataFrame(rows).astype(object)
