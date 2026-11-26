"""Column mapping: explicit configuration plus auto-detection heuristics.

Resolution order for every canonical field (first hit wins):

1. ``composite_keys`` / ``columns`` from the :class:`MappingConfig` (explicit),
2. exact match of the normalized header against the field's synonym list,
3. fuzzy match (``difflib`` ratio >= :data:`FUZZY_THRESHOLD`),
4. content-based detection (only for ``order_date``: >= 90 % parseable values).

Auto-detection never maps one source column to two canonical fields, and equal-score
ties are reported as *ambiguous* instead of being guessed.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bizpulse import resources
from bizpulse.schema import (
    CANONICAL_FIELDS,
    PRICE_FIELDS,
    REQUIRED_FIELDS,
    CanonicalField,
    Domain,
    Status,
)

FUZZY_THRESHOLD = 0.88
CONTENT_DATE_THRESHOLD = 0.9
CONTENT_SAMPLE_SIZE = 200

MatchMethod = Literal["explicit", "composite", "synonym", "fuzzy", "content"]

SYNONYMS: dict[CanonicalField, tuple[str, ...]] = {
    "transaction_id": (
        "transactionid", "lineid", "lineitemid", "lineref", "lineuid", "txnid",
        "transid", "rowid", "saleid", "orderlineid", "itemlineid", "lineitemref",
    ),
    "order_id": (
        "orderid", "ordernumber", "orderno", "invoiceid", "invoiceno", "invoicenumber",
        "ticketid", "ticketno", "receiptid", "receiptno", "basketid", "checkid",
    ),
    "order_date": (
        "orderdate", "date", "transactiondate", "invoicedate", "createdat", "created",
        "timestamp", "ts", "datetime", "saledate", "purchasedate", "orderdatetime",
        "orderedat", "paidat", "billingdate",
    ),
    "customer_id": (
        "customerid", "customer", "clientid", "accountid", "userid", "buyerid",
        "loyaltyid", "memberid", "customerno", "custid", "customernumber",
    ),
    "product_id": (
        "productid", "sku", "itemcode", "itemid", "plancode", "productcode",
        "articleid", "variantid", "menuitemid", "itemsku",
    ),
    "product_name": (
        "productname", "producttitle", "itemname", "title", "planname", "description",
        "itemdescription", "product", "menuitem",
    ),
    "category": (
        "category", "productcategory", "producttype", "productfamily", "menucategory",
        "department", "itemcategory", "productgroup",
    ),
    "channel": (
        "channel", "saleschannel", "acquisitionchannel", "ordertype", "source",
        "orderchannel", "platform", "medium", "salessource",
    ),
    "region": (
        "region", "shipregion", "country", "store", "location", "storename", "market",
        "territory", "shipcountry", "branch", "outlet",
    ),
    "quantity": (
        "quantity", "qty", "units", "seats", "quantityordered", "unitssold",
        "licenses", "qtyordered",
    ),
    "unit_price": (
        "unitprice", "price", "priceperseat", "priceperunit", "itemprice", "listprice",
        "unitamount", "sellingprice", "rate",
    ),
    "line_total": (
        "linetotal", "total", "amount", "lineamount", "subtotal", "sales", "revenue",
        "totalprice", "extendedprice", "lineprice",
    ),
    "discount_amount": (
        "discount", "discountamount", "discountamt", "promodiscount", "promo",
        "markdown", "couponamount", "discounts",
    ),
    "unit_cost": (
        "unitcost", "cost", "costperseat", "foodcost", "costperunit", "itemcost",
        "costprice", "cogsperunit",
    ),
    "fee_amount": (
        "fee", "fees", "feeamount", "processingfee", "paymentfee", "platformfee", "commission",
        "transactionfee", "merchantfee",
    ),
    "currency": ("currency", "currencycode", "curr", "ccy", "currencyiso"),
    "status": (
        "status", "financialstatus", "invoicestatus", "orderstatus", "paymentstatus",
        "state", "transactionstatus", "linestatus",
    ),
}  # fmt: skip


class MappingError(ValueError):
    """The source columns cannot be mapped onto the canonical contract."""


def normalize_header(header: str) -> str:
    """``" Order Date "`` -> ``orderdate`` (accents and punctuation removed)."""
    ascii_text = unicodedata.normalize("NFKD", header).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", ascii_text.lower())


class ReadOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sheet_name: str | int | None = None
    delimiter: str | None = None
    encoding: str | None = None


class MappingConfig(BaseModel):
    """How one dataset's files map onto the canonical transaction contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str = Field(pattern=r"^[a-z][a-z0-9_]{1,48}$")
    display_name: str | None = None
    domain: Domain = "custom"
    reporting_currency: str = "USD"
    default_currency: str | None = None
    dayfirst: bool = False
    date_format: str | None = None
    negative_quantity_as_refund: bool = False
    anonymous_customer_ids: tuple[str, ...] = ()
    max_quarantine_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    columns: dict[CanonicalField, str] = Field(default_factory=dict)
    composite_keys: dict[CanonicalField, list[str]] = Field(default_factory=dict)
    status_values: dict[str, Status] = Field(default_factory=dict)
    read_options: ReadOptions = Field(default_factory=ReadOptions)

    @field_validator("reporting_currency", "default_currency")
    @classmethod
    def _upper_iso(cls, value: str | None) -> str | None:
        if value is None:
            return None
        upper = value.strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", upper):
            raise ValueError(f"invalid currency code {value!r}")
        return upper

    @field_validator("status_values")
    @classmethod
    def _lower_status_keys(cls, values: dict[str, Status]) -> dict[str, Status]:
        return {k.strip().lower(): v for k, v in values.items()}

    @model_validator(mode="after")
    def _no_double_explicit(self) -> MappingConfig:
        both = set(self.columns) & set(self.composite_keys)
        if both:
            raise ValueError(f"fields both in columns and composite_keys: {sorted(both)}")
        for name, parts in self.composite_keys.items():
            if len(parts) < 2:
                raise ValueError(f"composite key {name!r} needs at least two columns")
        return self

    @property
    def effective_default_currency(self) -> str:
        return self.default_currency or self.reporting_currency

    @property
    def label(self) -> str:
        return self.display_name or self.dataset

    @classmethod
    def from_file(cls, path: Path) -> MappingConfig:
        text = path.read_text(encoding="utf-8")
        data: Any = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
        if not isinstance(data, dict):
            raise MappingError(f"{path.name}: mapping must be a mapping/object at the top level")
        return cls.model_validate(data)

    @classmethod
    def builtin(cls, name: str) -> MappingConfig:
        return cls.model_validate(yaml.safe_load(resources.read_text("mappings", f"{name}.yaml")))


@dataclass(frozen=True)
class FieldMatch:
    field: CanonicalField
    sources: tuple[str, ...]
    method: MatchMethod
    score: float

    @property
    def source_label(self) -> str:
        return " + ".join(self.sources)


@dataclass(frozen=True)
class ResolvedMapping:
    matches: dict[CanonicalField, FieldMatch]
    unmapped_columns: list[str]
    ambiguous: dict[str, list[str]] = field(default_factory=dict)
    missing_required: list[str] = field(default_factory=list)

    def source_of(self, name: CanonicalField) -> FieldMatch | None:
        return self.matches.get(name)

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {
            name: {"source": m.source_label, "method": m.method, "score": round(m.score, 3)}
            for name, m in self.matches.items()
        }


def _similarity(header: str, name: CanonicalField) -> float:
    return max(SequenceMatcher(None, header, syn).ratio() for syn in SYNONYMS[name])


def _assign_greedy(
    candidates: list[tuple[float, CanonicalField, str]],
    method: MatchMethod,
    matches: dict[CanonicalField, FieldMatch],
    claimed: set[str],
    ambiguous: dict[str, list[str]],
) -> None:
    """Assign best-scoring (field, column) pairs; equal-score ties become ambiguous."""
    by_field: dict[CanonicalField, list[tuple[float, str]]] = {}
    for score, name, column in candidates:
        by_field.setdefault(name, []).append((score, column))
    order = CANONICAL_FIELDS.index
    for score, name, column in sorted(candidates, key=lambda c: (-c[0], order(c[1]), c[2])):
        if name in matches or column in claimed or name in ambiguous:
            continue
        rivals = [c for s, c in by_field[name] if s == score and c not in claimed]
        if len(rivals) > 1:
            ambiguous[name] = sorted(rivals)
            continue
        matches[name] = FieldMatch(name, (column,), method, score)
        claimed.add(column)


def resolve_mapping(
    columns: Sequence[str],
    config: MappingConfig,
    *,
    sample: pd.DataFrame | None = None,
    date_ratio: Callable[[pd.Series[Any]], float] | None = None,
) -> ResolvedMapping:
    """Resolve canonical fields onto *columns* of a source file.

    Raises :class:`MappingError` if an explicit mapping references a missing column.
    Missing *required* fields are reported (not raised) so callers can include them in
    the audit report before failing the run.
    """
    present = list(columns)
    present_set = set(present)
    matches: dict[CanonicalField, FieldMatch] = {}
    claimed: set[str] = set()
    ambiguous: dict[str, list[str]] = {}

    for name, parts in config.composite_keys.items():
        missing = [p for p in parts if p not in present_set]
        if missing:
            raise MappingError(f"composite key {name!r}: columns not found in file: {missing}")
        matches[name] = FieldMatch(name, tuple(parts), "composite", 1.0)
        claimed.update(parts)
    for name, source in config.columns.items():
        if source not in present_set:
            raise MappingError(
                f"mapped column {source!r} for field {name!r} not found; available: {present}"
            )
        matches[name] = FieldMatch(name, (source,), "explicit", 1.0)
        claimed.add(source)

    normalized = {col: normalize_header(col) for col in present}
    pending = [f for f in CANONICAL_FIELDS if f not in matches]

    exact = [
        (1.0, name, col)
        for name in pending
        for col in present
        if col not in claimed and normalized[col] in SYNONYMS[name]
    ]
    _assign_greedy(exact, "synonym", matches, claimed, ambiguous)

    pending = [f for f in CANONICAL_FIELDS if f not in matches and f not in ambiguous]
    fuzzy: list[tuple[float, CanonicalField, str]] = []
    for col in present:
        if col in claimed or not normalized[col]:
            continue
        scores = sorted(((_similarity(normalized[col], f), f) for f in pending), reverse=True)
        if not scores or scores[0][0] < FUZZY_THRESHOLD:
            continue
        if len(scores) > 1 and scores[1][0] == scores[0][0]:
            ambiguous[f"column:{col}"] = sorted(f for s, f in scores if s == scores[0][0])
            continue
        fuzzy.append((round(scores[0][0], 4), scores[0][1], col))
    _assign_greedy(fuzzy, "fuzzy", matches, claimed, ambiguous)

    if "order_date" not in matches and sample is not None and date_ratio is not None:
        ratios = [
            (round(date_ratio(sample[col].head(CONTENT_SAMPLE_SIZE)), 4), "order_date", col)
            for col in present
            if col not in claimed and col in sample.columns
        ]
        content: list[tuple[float, CanonicalField, str]] = [
            (r, "order_date", c) for r, _, c in ratios if r >= CONTENT_DATE_THRESHOLD
        ]
        _assign_greedy(content, "content", matches, claimed, ambiguous)

    missing_required: list[str] = [f for f in REQUIRED_FIELDS if f not in matches]
    if not any(p in matches for p in PRICE_FIELDS):
        missing_required.append("unit_price|line_total")
    return ResolvedMapping(
        matches=dict(sorted(matches.items(), key=lambda kv: CANONICAL_FIELDS.index(kv[0]))),
        unmapped_columns=[c for c in present if c not in claimed],
        ambiguous=ambiguous,
        missing_required=missing_required,
    )
