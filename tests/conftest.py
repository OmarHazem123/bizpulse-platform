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
SMALL_SCALE = 0.1


def pytest_configure(config: pytest.Config) -> None:
    """Keep pytest temp dirs inside the repo (git-ignored), off the system drive."""
    if config.option.basetemp is None:
        config.option.basetemp = str(config.rootpath / ".pytest_tmp")


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


@pytest.fixture(scope="session")
def demo_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """A small, fully loaded demo warehouse (all three datasets), shared read-only."""
    from bizpulse.pipeline import load_demo

    resolved = Settings.from_env(tmp_path_factory.mktemp("demo_wh") / "data")
    resolved.ensure_dirs()
    load_demo(resolved, scale=SMALL_SCALE)
    return resolved


@pytest.fixture
def basic_config() -> MappingConfig:
    return MappingConfig(dataset="acme", reporting_currency="USD")


def make_tx(*rows: dict[str, Any]) -> pd.DataFrame:
    """A transactions frame (analytics input) from partial row dicts."""
    from bizpulse.analytics.data import prepare

    records = []
    for i, row in enumerate(rows, start=1):
        gross = float(row.get("gross", 10.0))
        discount = float(row.get("discount", 0.0))
        records.append(
            {
                "order_ts": pd.Timestamp(row.get("ts", "2026-01-01 10:00")),
                "order_id": row.get("order_id", f"O{i}"),
                "transaction_id": row.get("transaction_id", f"T{i}"),
                "customer_id": row.get("customer_id", "C1"),
                "is_anonymous": row.get("is_anonymous", False),
                "product_id": row.get("product_id", "P1"),
                "product_name": row.get("product_name", row.get("product_id", "P1")),
                "category": row.get("category", "Cat"),
                "channel": row.get("channel", "web"),
                "region": row.get("region", "EU"),
                "status": row.get("status", "completed"),
                "quantity": float(row.get("quantity", 1.0)),
                "gross": gross,
                "discount": discount,
                "net": gross - discount,
                "cost": row.get("cost", float("nan")),
                "fee": float(row.get("fee", 0.0)),
            }
        )
    return prepare(pd.DataFrame(records))


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
