"""Canonical vocabulary shared by every layer (fields, statuses, reason codes)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal, get_args

CanonicalField = Literal[
    "transaction_id",
    "order_id",
    "order_date",
    "customer_id",
    "product_id",
    "product_name",
    "category",
    "channel",
    "region",
    "quantity",
    "unit_price",
    "line_total",
    "discount_amount",
    "unit_cost",
    "fee_amount",
    "currency",
    "status",
]
CANONICAL_FIELDS: Final[tuple[CanonicalField, ...]] = get_args(CanonicalField)

#: Fields that must be resolvable from the source (``unit_price`` may come from ``line_total``).
REQUIRED_FIELDS: Final[tuple[CanonicalField, ...]] = (
    "transaction_id",
    "order_date",
    "customer_id",
    "product_id",
    "quantity",
)
PRICE_FIELDS: Final[tuple[CanonicalField, ...]] = ("unit_price", "line_total")

AMOUNT_FIELDS: Final[tuple[CanonicalField, ...]] = (
    "unit_price",
    "line_total",
    "discount_amount",
    "unit_cost",
    "fee_amount",
)
ID_FIELDS: Final[tuple[CanonicalField, ...]] = (
    "transaction_id",
    "order_id",
    "customer_id",
    "product_id",
)
TEXT_FIELDS: Final[tuple[CanonicalField, ...]] = ("product_name", "category", "channel", "region")

Status = Literal["completed", "refunded", "cancelled"]
STATUSES: Final[tuple[Status, ...]] = get_args(Status)

Domain = Literal["ecommerce", "saas", "retail_fb", "custom"]

DEFAULT_CATEGORY: Final = "Uncategorized"
DEFAULT_DIMENSION: Final = "unknown"


@dataclass(frozen=True)
class QualityIssue:
    """One data-quality violation on one source row.

    ``code`` is ``"<field>.<rule>"`` (e.g. ``order_date.unparseable``) or ``row.<rule>``.
    """

    field: str
    code: str
    message: str
    value: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "code": self.code,
            "message": self.message,
            "value": self.value,
        }


#: Columns of the canonical (normalized) frame handed to the contract, in hash order.
NORMALIZED_COLUMNS: Final[tuple[str, ...]] = (
    "transaction_id",
    "order_id",
    "order_date",
    "customer_id",
    "product_id",
    "product_name",
    "category",
    "channel",
    "region",
    "quantity",
    "unit_price",
    "discount_amount",
    "unit_cost",
    "fee_amount",
    "currency",
    "status",
)
