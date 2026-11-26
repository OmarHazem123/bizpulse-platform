from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from pydantic import ValidationError

from bizpulse.quality.contracts import (
    check_record,
    contract_context,
    issues_from_validation_error,
)
from tests.conftest import AS_OF

VALID: dict[str, Any] = {
    "transaction_id": "T1",
    "order_id": "O1",
    "order_date": datetime(2026, 3, 1, 10),
    "customer_id": "C1",
    "product_id": "P1",
    "product_name": "Widget",
    "category": "Gadgets",
    "channel": "web",
    "region": "EU",
    "quantity": 2.0,
    "unit_price": 10.0,
    "discount_amount": 1.0,
    "unit_cost": 4.0,
    "fee_amount": 0.5,
    "currency": "USD",
    "status": "completed",
}


def codes_for(patch: dict[str, Any], skip: frozenset[str] = frozenset()) -> list[str]:
    try:
        check_record({**VALID, **patch}, contract_context(AS_OF))
    except ValidationError as exc:
        return sorted(i.code for i in issues_from_validation_error(exc, skip))
    return []


def test_valid_record_passes() -> None:
    record = check_record(VALID, contract_context(AS_OF))
    assert record.quantity == 2.0
    assert codes_for({"unit_cost": None}) == []


@pytest.mark.parametrize(
    ("patch", "expected"),
    [
        ({"customer_id": None}, ["customer_id.missing"]),
        ({"customer_id": ""}, ["customer_id.missing"]),
        ({"transaction_id": "x" * 129}, ["transaction_id.too_long"]),
        ({"quantity": 0.0}, ["quantity.non_positive"]),
        ({"quantity": -2.0}, ["quantity.non_positive"]),
        ({"quantity": 2e6}, ["quantity.out_of_range"]),
        ({"unit_price": -1.0}, ["unit_price.negative"]),
        ({"unit_price": None}, ["unit_price.missing"]),
        ({"unit_cost": -3.0}, ["unit_cost.negative"]),
        ({"fee_amount": -0.1}, ["fee_amount.negative"]),
        ({"discount_amount": 25.0}, ["discount_amount.exceeds_gross"]),
        ({"order_date": datetime(2099, 1, 15)}, ["order_date.out_of_range"]),
        ({"order_date": datetime(1999, 12, 31)}, ["order_date.out_of_range"]),
        ({"order_date": None}, ["order_date.missing"]),
        ({"status": "pending"}, ["status.invalid"]),
        ({"currency": "usd"}, ["currency.unsupported"]),
    ],
)
def test_rule_violations_map_to_reason_codes(patch: dict[str, Any], expected: list[str]) -> None:
    assert codes_for(patch) == expected


def test_cross_field_rule_skipped_when_inputs_invalid() -> None:
    # quantity 0 makes gross 0, but only the quantity defect is reported.
    assert codes_for({"quantity": 0.0, "discount_amount": 5.0}) == ["quantity.non_positive"]


def test_already_reported_fields_are_not_duplicated() -> None:
    assert codes_for({"unit_price": None}, skip=frozenset({"unit_price"})) == []


def test_future_tolerance_is_one_day() -> None:
    assert codes_for({"order_date": datetime(2026, 11, 27, 11)}) == []
    assert codes_for({"order_date": datetime(2026, 11, 27, 13)}) == ["order_date.out_of_range"]
