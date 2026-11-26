"""The canonical data contract every loaded transaction line must satisfy.

``TransactionRecord`` is the single source of truth for field constraints. Pydantic
errors are translated into stable reason codes (``<field>.<rule>``) by
:func:`issues_from_validation_error`, so quarantine reports are exact and machine-readable.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated, Any, Self, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from bizpulse.schema import QualityIssue, Status

MIN_ORDER_DATE = datetime(2000, 1, 1)  # noqa: DTZ001 - the warehouse stores naive UTC
MAX_FUTURE_TOLERANCE = timedelta(days=1)
MAX_QUANTITY = 1_000_000.0
DISCOUNT_TOLERANCE = 1e-6

NonEmptyId = Annotated[str, StringConstraints(min_length=1, max_length=128)]
EntityId = Annotated[str, StringConstraints(min_length=1, max_length=256)]
CurrencyCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]


class ContractContext(TypedDict):
    min_date: datetime
    max_date: datetime


def contract_context(as_of: datetime) -> ContractContext:
    """Validation window: ``[2000-01-01, as_of + 1 day]`` (naive UTC)."""
    naive = as_of.replace(tzinfo=None)
    return {"min_date": MIN_ORDER_DATE, "max_date": naive + MAX_FUTURE_TOLERANCE}


class TransactionRecord(BaseModel):
    """One normalized transaction line (see TECHNICAL-REQUIREMENTS.md §2)."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    transaction_id: NonEmptyId
    order_id: NonEmptyId
    order_date: datetime
    customer_id: EntityId
    product_id: EntityId
    product_name: str
    category: str
    channel: str
    region: str
    quantity: float = Field(gt=0, le=MAX_QUANTITY)
    unit_price: float = Field(ge=0)
    discount_amount: float = Field(ge=0)
    unit_cost: float | None = Field(default=None, ge=0)
    fee_amount: float = Field(ge=0)
    currency: CurrencyCode
    status: Status

    @field_validator("order_date")
    @classmethod
    def _date_in_window(cls, value: datetime, info: ValidationInfo) -> datetime:
        context: dict[str, Any] = info.context or {}
        low: datetime = context.get("min_date", MIN_ORDER_DATE)
        high: datetime | None = context.get("max_date")
        naive = value.replace(tzinfo=None)
        if naive < low or (high is not None and naive > high):
            raise PydanticCustomError(
                "out_of_range",
                "order date {value} is outside the allowed window [{low}, {high}]",
                {"value": naive.isoformat(), "low": low.date().isoformat(), "high": str(high)},
            )
        return value

    @model_validator(mode="after")
    def _discount_within_gross(self) -> Self:
        gross = self.quantity * self.unit_price
        if self.discount_amount > gross + DISCOUNT_TOLERANCE:
            raise PydanticCustomError(
                "discount_exceeds_gross",
                "discount {discount} exceeds gross amount {gross}",
                {"discount": round(self.discount_amount, 4), "gross": round(gross, 4)},
            )
        return self


_TYPE_TO_RULE: dict[str, str] = {
    "missing": "missing",
    "string_too_short": "missing",
    "string_too_long": "too_long",
    "greater_than": "non_positive",
    "greater_than_equal": "negative",
    "less_than_equal": "out_of_range",
    "out_of_range": "out_of_range",
    "literal_error": "invalid",
    "string_pattern_mismatch": "unsupported",
    "finite_number": "unparseable",
}
_MODEL_RULES: dict[str, tuple[str, str]] = {
    "discount_exceeds_gross": ("discount_amount", "discount_amount.exceeds_gross"),
}


def issues_from_validation_error(
    error: ValidationError, skip_fields: frozenset[str] = frozenset()
) -> list[QualityIssue]:
    """Translate a Pydantic ``ValidationError`` into reason-coded issues.

    Fields in *skip_fields* already failed normalization and are not reported twice.
    """
    issues: list[QualityIssue] = []
    for err in error.errors(include_url=False):
        err_type = str(err["type"])
        loc = err.get("loc", ())
        if err_type in _MODEL_RULES:
            field_name, code = _MODEL_RULES[err_type]
        else:
            field_name = str(loc[0]) if loc else "row"
            value = err.get("input")
            rule = "missing" if value is None else _TYPE_TO_RULE.get(err_type, "invalid")
            code = f"{field_name}.{rule}"
        if field_name in skip_fields:
            continue
        raw_input = err.get("input")
        shown = None if raw_input is None or isinstance(raw_input, dict) else str(raw_input)[:200]
        issues.append(QualityIssue(field_name, code, str(err["msg"]), shown))
    return issues


def check_record(record: dict[str, Any], context: ContractContext) -> TransactionRecord:
    """Validate one normalized record (raises ``ValidationError``)."""
    return TransactionRecord.model_validate(record, context=context)
