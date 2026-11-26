"""Load one dataset's transactions (facts + dimensions) into an analysis frame.

Connections are short-lived and read-only: the warehouse file can be swapped by
compaction, and Windows locks open files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bizpulse.warehouse.connection import connect
from bizpulse.warehouse.schema import DIM_CUSTOMERS, DIM_PRODUCTS, FACT_TRANSACTIONS, META_DATASETS

#: Columns of a transactions frame, the input of every analytics/ML function.
TX_COLUMNS: tuple[str, ...] = (
    "order_ts",
    "order_id",
    "transaction_id",
    "customer_id",
    "is_anonymous",
    "product_id",
    "product_name",
    "category",
    "channel",
    "region",
    "status",
    "quantity",
    "gross",
    "discount",
    "net",
    "cost",
    "fee",
)


@dataclass(frozen=True)
class DatasetInfo:
    dataset: str
    display_name: str
    domain: str
    currency: str
    timezone: str


def list_datasets(warehouse: Path) -> list[DatasetInfo]:
    if not warehouse.exists():
        return []
    with connect(warehouse, read_only=True) as con:
        rows = con.execute(
            f"""SELECT dataset, display_name, domain, reporting_currency, timezone
                FROM {META_DATASETS} ORDER BY dataset"""
        ).fetchall()
    return [DatasetInfo(*(str(v) for v in row)) for row in rows]


def load_transactions(warehouse: Path, dataset: str) -> tuple[DatasetInfo, pd.DataFrame]:
    """Return dataset metadata and its transactions frame (sorted by time)."""
    infos = {info.dataset: info for info in list_datasets(warehouse)}
    if dataset not in infos:
        raise KeyError(f"dataset {dataset!r} not found; available: {sorted(infos)}")
    with connect(warehouse, read_only=True) as con:
        frame = con.execute(
            f"""SELECT f.order_ts, f.order_id, f.transaction_id, c.customer_id, c.is_anonymous,
                       p.product_id, p.product_name, p.category, f.channel, f.region, f.status,
                       f.quantity, f.gross_amount AS gross, f.discount_amount AS discount,
                       f.net_amount AS net, f.cost_amount AS cost, f.fee_amount AS fee
                FROM {FACT_TRANSACTIONS} f
                JOIN {DIM_CUSTOMERS} c USING (customer_key)
                JOIN {DIM_PRODUCTS} p USING (product_key)
                WHERE f.dataset = ?
                ORDER BY f.order_ts, f.transaction_id""",
            [dataset],
        ).df()
    return infos[dataset], prepare(frame)


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes of a transactions frame (also used by test fixtures)."""
    out = frame.loc[:, list(TX_COLUMNS)].copy()
    out["order_ts"] = pd.to_datetime(out["order_ts"])
    for column in ("quantity", "gross", "discount", "net", "cost", "fee"):
        out[column] = out[column].astype("float64")
    out["is_anonymous"] = out["is_anonymous"].astype(bool)
    return out.reset_index(drop=True)


def as_of(tx: pd.DataFrame) -> pd.Timestamp:
    """Analysis reference date: the day after the last transaction (data-derived)."""
    return pd.Timestamp(tx["order_ts"].max()).normalize() + pd.Timedelta(1, unit="D")
