from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from bizpulse.ingestion.mapping import (
    SYNONYMS,
    MappingConfig,
    MappingError,
    normalize_header,
    resolve_mapping,
)
from bizpulse.ingestion.normalize import date_parse_ratio
from bizpulse.resources import builtin_mapping_names


def test_synonyms_are_unique_across_fields() -> None:
    counts = Counter(s for synonyms in SYNONYMS.values() for s in synonyms)
    assert [s for s, n in counts.items() if n > 1] == []


@pytest.mark.parametrize(
    ("header", "expected"),
    [(" Order Date ", "orderdate"), ("Prix Unitaire (€)", "prixunitaire"), ("Qty.", "qty")],
)
def test_normalize_header(header: str, expected: str) -> None:
    assert normalize_header(header) == expected


def test_synonym_detection_maps_generic_headers() -> None:
    columns = ["Date", "Customer", "SKU", "Qty", "Amount", "Status", "Ticket No"]
    resolved = resolve_mapping(columns, MappingConfig(dataset="acme"))
    got = {k: (m.source_label, m.method) for k, m in resolved.matches.items()}
    assert got["order_date"] == ("Date", "synonym")
    assert got["customer_id"] == ("Customer", "synonym")
    assert got["line_total"] == ("Amount", "synonym")
    assert got["order_id"] == ("Ticket No", "synonym")
    assert resolved.missing_required == ["transaction_id"]


def test_fuzzy_match_tolerates_typos() -> None:
    resolved = resolve_mapping(["Aquisition Channel"], MappingConfig(dataset="acme"))
    match = resolved.matches["channel"]
    assert match.method == "fuzzy"
    assert match.score >= 0.88


def test_explicit_mapping_wins_and_is_never_double_claimed() -> None:
    config = MappingConfig(dataset="acme", columns={"customer_id": "Date"})
    resolved = resolve_mapping(["Date", "Created At"], config)
    assert resolved.matches["customer_id"].source_label == "Date"
    assert resolved.matches["order_date"].source_label == "Created At"
    sources = [s for m in resolved.matches.values() for s in m.sources]
    assert len(sources) == len(set(sources))


def test_equal_score_ties_are_reported_as_ambiguous() -> None:
    resolved = resolve_mapping(["Date", "Timestamp"], MappingConfig(dataset="acme"))
    assert "order_date" not in resolved.matches
    assert resolved.ambiguous["order_date"] == ["Date", "Timestamp"]
    assert "order_date" in resolved.missing_required


def test_content_based_date_detection_ignores_numeric_columns() -> None:
    sample = pd.DataFrame(
        {"col_a": ["1", "2", "3"], "when?": ["2026-01-01", "2026-01-02", "03/01/2026"]}
    ).astype(object)
    resolved = resolve_mapping(
        list(sample.columns),
        MappingConfig(dataset="acme"),
        sample=sample,
        date_ratio=date_parse_ratio,
    )
    assert resolved.matches["order_date"].source_label == "when?"
    assert resolved.matches["order_date"].method == "content"


def test_composite_keys_and_missing_columns() -> None:
    config = MappingConfig(dataset="acme", composite_keys={"transaction_id": ["ticket", "line"]})
    resolved = resolve_mapping(["ticket", "line"], config)
    assert resolved.matches["transaction_id"].sources == ("ticket", "line")
    with pytest.raises(MappingError, match="not found"):
        resolve_mapping(["ticket"], config)
    with pytest.raises(MappingError, match="not found"):
        resolve_mapping(["a"], MappingConfig(dataset="acme", columns={"quantity": "Qty"}))


def test_missing_price_is_reported() -> None:
    resolved = resolve_mapping(
        ["Line ID", "Date", "Customer", "SKU", "Qty"], MappingConfig(dataset="x1")
    )
    assert resolved.missing_required == ["unit_price|line_total"]


def test_config_validation() -> None:
    with pytest.raises(ValidationError):
        MappingConfig(dataset="Bad Name")
    with pytest.raises(ValidationError):
        MappingConfig(dataset="acme", reporting_currency="dollars")
    with pytest.raises(ValidationError):
        MappingConfig(dataset="acme", composite_keys={"transaction_id": ["only_one"]})
    config = MappingConfig(dataset="acme", status_values={" Payment Failed ": "cancelled"})
    assert config.status_values == {"payment failed": "cancelled"}
    assert config.effective_default_currency == "USD"


def test_config_from_yaml_and_json(tmp_path: Path) -> None:
    yaml_path = tmp_path / "m.yaml"
    yaml_path.write_text("dataset: acme\ncolumns:\n  quantity: Units\n", encoding="utf-8")
    json_path = tmp_path / "m.json"
    json_path.write_text('{"dataset": "acme", "dayfirst": true}', encoding="utf-8")
    assert MappingConfig.from_file(yaml_path).columns == {"quantity": "Units"}
    assert MappingConfig.from_file(json_path).dayfirst is True
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(MappingError):
        MappingConfig.from_file(bad)


@pytest.mark.parametrize("name", ["ecommerce", "saas", "retail_fb"])
def test_builtin_mappings_load(name: str) -> None:
    assert name in builtin_mapping_names()
    assert MappingConfig.builtin(name).dataset == name
