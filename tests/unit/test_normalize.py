from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import pandas as pd
import pytest

from bizpulse.ingestion.fx import FxTable
from bizpulse.ingestion.mapping import MappingConfig, resolve_mapping
from bizpulse.ingestion.normalize import (
    is_blank,
    normalize_channel,
    normalize_currency_code,
    normalize_frame,
    normalize_id,
    parse_amount,
    parse_amounts,
    parse_dates,
)
from tests.conftest import make_rows


@pytest.mark.parametrize(
    ("raw", "value", "hint"),
    [
        ("$1,299.00", 1299.0, "USD"),
        ("46,29 €", 46.29, "EUR"),
        ("1.299,00", 1299.0, None),
        ("1.234.567", 1234567.0, None),
        ("(12.50)", -12.5, None),
        ("12.50-", -12.5, None),
        ("USD 49.99", 49.99, "USD"),
        ("C$45.10", 45.1, "CAD"),
        ("£39.37", 39.37, "GBP"),
        ("-15.00", -15.0, None),
        ("  7 ", 7.0, None),
        (12.5, 12.5, None),
        (3, 3.0, None),
    ],
)
def test_parse_amount_valid(raw: Any, value: float, hint: str | None) -> None:
    parsed = parse_amount(raw)
    assert not parsed.invalid
    assert parsed.value == pytest.approx(value)
    assert parsed.currency_hint == hint


@pytest.mark.parametrize("raw", ["12..50", "abc", "two", "1,2345", "1e3", True, float("inf")])
def test_parse_amount_invalid(raw: Any) -> None:
    assert parse_amount(raw).invalid


@pytest.mark.parametrize("raw", [None, "", "  ", "N/A", "null", "NaN", "-", float("nan")])
def test_blank_tokens(raw: Any) -> None:
    assert is_blank(raw)
    assert parse_amount(raw).value is None
    assert not parse_amount(raw).invalid


def test_parse_amounts_vectorized_matches_scalar() -> None:
    series = pd.Series(["1", "$2.50", "", "bad", 4.0], dtype=object)
    result = parse_amounts(series)
    assert result.values.tolist()[:2] == [1.0, 2.5]
    assert math.isnan(float(result.values.iloc[2]))
    assert result.invalid.tolist() == [False, False, False, True, False]
    assert result.hints.tolist()[1] == "USD"


def test_parse_dates_formats_and_timezones() -> None:
    series = pd.Series(
        [
            "2025-05-01 14:23:00",
            "2025-07-01T08:15:32+02:00",
            "2099-01-15 10:00:00",  # regression: must not inherit the +02:00 offset
            "2025-05-01T14:23:00Z",
            "05/01/2025 02:23 PM",
            datetime(2025, 6, 1, 10, 0),
            45789,
            "20250501",
            "not-a-date",
            "2025-13-45",
            "",
        ],
        dtype=object,
    )
    parsed = parse_dates(series, dayfirst=False)
    expected = [
        "2025-05-01 14:23:00",
        "2025-07-01 06:15:32",
        "2099-01-15 10:00:00",
        "2025-05-01 14:23:00",
        "2025-05-01 14:23:00",
        "2025-06-01 10:00:00",
        "2025-05-12 00:00:00",
        "2025-05-01 00:00:00",
    ]
    assert [str(v) for v in parsed.values.iloc[:8]] == expected
    assert parsed.invalid.tolist() == [False] * 8 + [True, True, False]
    assert pd.isna(parsed.values.iloc[10])


def test_parse_dates_dayfirst() -> None:
    series = pd.Series(["03/04/2026", "15-03-2026"], dtype=object)
    assert [str(v.date()) for v in parse_dates(series, dayfirst=True).values] == [
        "2026-04-03",
        "2026-03-15",
    ]
    assert str(parse_dates(series, dayfirst=False).values.iloc[0].date()) == "2026-03-04"


def test_scalar_normalizers() -> None:
    assert normalize_id(1001.0) == "1001"
    assert normalize_id(" C 01  ") == "C 01"
    assert normalize_id("1001.0") == "1001"
    assert normalize_id("") is None
    assert normalize_channel("Paid Search ") == "paid_search"
    assert normalize_currency_code(" eur ") == "EUR"
    assert normalize_currency_code("€") == "EUR"
    assert normalize_currency_code("EURO") == ""
    assert normalize_currency_code("") is None


def _normalize(frame: pd.DataFrame, config: MappingConfig, fx: FxTable) -> pd.DataFrame:
    resolved = resolve_mapping(list(frame.columns), config)
    batch = normalize_frame(frame, resolved, config, fx)
    out = batch.frame.copy()
    out["codes"] = [[i.code for i in row] for row in batch.issues]
    return out


def test_normalize_frame_defaults_and_currency_precedence(fx: FxTable) -> None:
    frame = make_rows(
        {},
        {"currency": "", "unit_price": "€10,00"},
        {"currency": "", "unit_price": "10"},
        {"currency": "GBP", "unit_price": "$10"},
        {"currency": "XYZ"},
        {"status": "Pending Review"},
        {"status": "", "category": "", "channel": "", "product_name": ""},
    )
    out = _normalize(frame, MappingConfig(dataset="acme", default_currency="CAD"), fx)
    assert out["currency"].tolist()[:4] == ["USD", "EUR", "CAD", "GBP"]
    assert out["codes"].tolist()[4] == ["currency.unsupported"]
    assert out["codes"].tolist()[5] == ["status.invalid"]
    last = out.iloc[6]
    assert (last["status"], last["category"], last["channel"], last["product_name"]) == (
        "completed",
        "Uncategorized",
        "unknown",
        "P1",
    )
    assert out["channel"].iloc[0] == "web_shop"


def test_normalize_frame_line_total_and_negative_refunds(fx: FxTable) -> None:
    frame = make_rows({"quantity": "-2"}).drop(columns=["unit_price"])
    frame["line_total"] = "-30.00"
    config = MappingConfig(dataset="acme", negative_quantity_as_refund=True)
    out = _normalize(frame, config, fx)
    assert out["quantity"].iloc[0] == 2.0
    assert out["status"].iloc[0] == "refunded"
    assert out["unit_price"].iloc[0] == pytest.approx(15.0)


def test_empty_row_has_exactly_one_issue(fx: FxTable) -> None:
    frame = make_rows({}, dict.fromkeys(make_rows().columns, ""))
    out = _normalize(frame, MappingConfig(dataset="acme"), fx)
    assert out["codes"].tolist() == [[], ["row.empty"]]
