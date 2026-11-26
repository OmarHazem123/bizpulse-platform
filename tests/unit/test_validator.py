from __future__ import annotations

import pandas as pd

from bizpulse.ingestion.fx import FxTable
from bizpulse.ingestion.hashing import transaction_key
from bizpulse.ingestion.mapping import MappingConfig, resolve_mapping
from bizpulse.quality.validator import ValidationResult, validate_batch
from tests.conftest import AS_OF, make_rows


def _validate(frame: pd.DataFrame, config: MappingConfig, fx: FxTable) -> ValidationResult:
    resolved = resolve_mapping(list(frame.columns), config)
    return validate_batch(frame, resolved, config, fx, as_of=AS_OF)


def test_duplicates_exact_vs_conflicting(fx: FxTable) -> None:
    frame = make_rows({}, {}, {})
    frame = pd.concat([frame, frame.iloc[[0]], frame.iloc[[1]]], ignore_index=True)
    frame.loc[4, "quantity"] = "3"  # conflicting copy of row 2
    result = _validate(frame, MappingConfig(dataset="acme"), fx)
    assert [(q.source_row, q.codes) for q in result.quarantined] == [
        (4, ["row.duplicate"]),
        (5, ["transaction_id.conflicting_duplicate"]),
    ]
    assert "source row 1" in result.quarantined[0].issues[0].message
    assert result.valid["source_row"].tolist() == [1, 2, 3]


def test_reconciliation_and_outputs(fx: FxTable) -> None:
    frame = make_rows({}, {"quantity": "two"}, {"customer_id": "GUEST"}, {"order_date": "bad"})
    config = MappingConfig(dataset="acme", anonymous_customer_ids=("GUEST",))
    result = _validate(frame, config, fx)
    assert result.rows_read == len(result.valid) + len(result.quarantined) == 4
    assert result.reason_counts == {"order_date.unparseable": 1, "quantity.unparseable": 1}
    assert result.pass_rate == 0.5
    assert result.valid["is_anonymous"].tolist() == [False, True]
    assert result.valid["transaction_key"].iloc[0] == transaction_key("acme", "T1")
    assert result.valid["row_hash"].str.len().eq(64).all()
    bad = result.quarantined[0]
    assert bad.raw["quantity"] == "two"
    assert bad.issues[0].value == "two"


def test_every_defect_yields_exactly_one_reason(fx: FxTable) -> None:
    patches = [
        {"order_date": "yesterday"},
        {"order_date": "2099-01-15"},
        {"customer_id": ""},
        {"product_id": " "},
        {"quantity": "0"},
        {"unit_price": "-15.00"},
        {"unit_price": "N/A"},
        {"unit_cost": "-3.00"},
        {"discount_amount": "500"},
        {"currency": "XYZ"},
        {"status": "pending_review"},
    ]
    result = _validate(make_rows(*patches), MappingConfig(dataset="acme"), fx)
    assert len(result.quarantined) == len(patches)
    assert all(len(q.codes) == 1 for q in result.quarantined)
