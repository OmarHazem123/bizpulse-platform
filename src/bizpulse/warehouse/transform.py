"""Turn validated canonical rows into fact rows in the reporting currency."""

from __future__ import annotations

import pandas as pd

from bizpulse.ingestion.fx import FxTable
from bizpulse.ingestion.hashing import entity_key
from bizpulse.ingestion.mapping import MappingConfig

FACT_COLUMNS: tuple[str, ...] = (
    "transaction_key",
    "dataset",
    "transaction_id",
    "order_id",
    "order_ts",
    "date_key",
    "customer_key",
    "product_key",
    "channel",
    "region",
    "status",
    "is_refund",
    "quantity",
    "unit_price",
    "gross_amount",
    "discount_amount",
    "net_amount",
    "cost_amount",
    "fee_amount",
    "currency_original",
    "fx_rate",
    "row_hash",
)
#: Extra staging columns used to build dimensions (not stored on the fact).
STAGING_EXTRA: tuple[str, ...] = (
    "customer_id",
    "product_id",
    "product_name",
    "category",
    "is_anonymous",
)


def build_fact_frame(valid: pd.DataFrame, config: MappingConfig, fx: FxTable) -> pd.DataFrame:
    """Compute keys and measures (TECHNICAL-REQUIREMENTS.md §5) for validated rows."""
    target = config.reporting_currency
    dataset = config.dataset
    rates = {code: fx.rate(code, target) for code in valid["currency"].dropna().unique()}
    fx_rate = valid["currency"].map(rates).astype("float64")
    quantity = valid["quantity"].astype("float64")
    unit_price = valid["unit_price"].astype("float64") * fx_rate
    gross = (quantity * unit_price).round(6)
    discount = (valid["discount_amount"].astype("float64") * fx_rate).round(6)
    cost = (quantity * valid["unit_cost"].astype("float64") * fx_rate).round(6)
    order_ts = pd.to_datetime(valid["order_date"])
    facts = pd.DataFrame(
        {
            "transaction_key": valid["transaction_key"],
            "dataset": dataset,
            "transaction_id": valid["transaction_id"],
            "order_id": valid["order_id"],
            "order_ts": order_ts,
            "date_key": order_ts.dt.strftime("%Y%m%d").astype("int64"),
            "customer_key": [entity_key(dataset, "customer", str(c)) for c in valid["customer_id"]],
            "product_key": [entity_key(dataset, "product", str(p)) for p in valid["product_id"]],
            "channel": valid["channel"],
            "region": valid["region"],
            "status": valid["status"],
            "is_refund": valid["status"].eq("refunded"),
            "quantity": quantity,
            "unit_price": unit_price.round(6),
            "gross_amount": gross,
            "discount_amount": discount,
            "net_amount": (gross - discount).round(6),
            "cost_amount": cost,
            "fee_amount": (valid["fee_amount"].astype("float64") * fx_rate).round(6),
            "currency_original": valid["currency"],
            "fx_rate": fx_rate,
            "row_hash": valid["row_hash"],
            "customer_id": valid["customer_id"],
            "product_id": valid["product_id"],
            "product_name": valid["product_name"],
            "category": valid["category"],
            "is_anonymous": valid["is_anonymous"].astype(bool),
        }
    )
    return facts.reset_index(drop=True)
