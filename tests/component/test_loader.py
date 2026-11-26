from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from bizpulse.ingestion.fx import FxTable
from bizpulse.ingestion.mapping import MappingConfig, resolve_mapping
from bizpulse.quality.validator import validate_batch
from bizpulse.warehouse.connection import connect
from bizpulse.warehouse.loader import LoadError, load_facts
from bizpulse.warehouse.maintenance import compact, compact_if_fragmented, storage_stats
from bizpulse.warehouse.schema import orphan_counts, table_counts
from bizpulse.warehouse.transform import build_fact_frame
from tests.conftest import AS_OF, make_rows

CONFIG = MappingConfig(dataset="acme", reporting_currency="USD")
LOADED_AT = datetime(2026, 11, 26)


def _facts(frame: pd.DataFrame, fx: FxTable) -> pd.DataFrame:
    resolved = resolve_mapping(list(frame.columns), CONFIG)
    result = validate_batch(frame, resolved, CONFIG, fx, as_of=AS_OF)
    assert not result.quarantined
    return build_fact_frame(result.valid, CONFIG, fx)


def test_measures_in_reporting_currency(fx: FxTable) -> None:
    facts = _facts(make_rows({"currency": "EUR", "unit_cost": ""}), fx)
    row = facts.iloc[0]
    assert row["fx_rate"] == pytest.approx(1.08)
    assert row["gross_amount"] == pytest.approx(2 * 10 * 1.08)
    assert row["net_amount"] == pytest.approx(row["gross_amount"] - 1.08)
    assert pd.isna(row["cost_amount"])  # unknown cost stays NULL, never 0
    assert row["date_key"] == 20260301


def test_upsert_insert_update_unchanged(tmp_path: Path, fx: FxTable) -> None:
    db = tmp_path / "w.duckdb"
    first = _facts(make_rows({}, {}, {"status": "refunded"}), fx)
    with connect(db) as con:
        r1 = load_facts(con, first, run_id="r1", loaded_at=LOADED_AT)
        assert (r1.inserted, r1.updated, r1.unchanged) == (3, 0, 0)
        assert (r1.customers_new, r1.products_new, r1.dates_new) == (1, 1, 1)

        r2 = load_facts(con, first, run_id="r2", loaded_at=LOADED_AT)
        assert (r2.inserted, r2.updated, r2.unchanged) == (0, 0, 3)

        changed = _facts(
            make_rows({}, {"quantity": "5"}, {"status": "refunded"}, {"order_date": "2026-03-05"}),
            fx,
        )
        r3 = load_facts(con, changed, run_id="r3", loaded_at=LOADED_AT)
        assert (r3.inserted, r3.updated, r3.unchanged) == (1, 1, 2)
        assert r3.dates_new == 4  # 2026-03-02 .. 2026-03-05 fill the calendar gap

        rows = con.execute(
            "SELECT transaction_id, quantity, run_id, is_refund FROM fact_transactions ORDER BY 1"
        ).fetchall()
        assert rows == [
            ("T1", 2.0, "r1", False),
            ("T2", 5.0, "r3", False),
            ("T3", 2.0, "r1", True),
            ("T4", 2.0, "r3", False),
        ]
        dim = con.execute(
            "SELECT first_seen_date, last_seen_date, updated_run_id FROM dim_customers"
        ).fetchone()
        assert dim is not None
        assert (str(dim[0]), str(dim[1]), dim[2]) == ("2026-03-01", "2026-03-05", "r3")
        assert orphan_counts(con) == {"customer_key": 0, "product_key": 0, "date_key": 0}


def test_failed_load_rolls_back(tmp_path: Path, fx: FxTable) -> None:
    db = tmp_path / "w.duckdb"
    facts = _facts(make_rows({}, {}), fx)
    broken = facts.copy()
    broken.loc[1, "order_ts"] = pd.NaT  # violates NOT NULL inside the transaction
    with connect(db) as con:
        with pytest.raises(duckdb.ConstraintException):
            load_facts(con, broken, run_id="bad", loaded_at=LOADED_AT)
        counts = table_counts(con)
        assert counts["fact_transactions"] == counts["dim_customers"] == 0
        load_facts(con, facts, run_id="ok", loaded_at=LOADED_AT)
        assert table_counts(con)["fact_transactions"] == 2


def test_loader_preconditions(tmp_path: Path, fx: FxTable) -> None:
    facts = _facts(make_rows(), fx)
    with connect(tmp_path / "w.duckdb") as con:
        with pytest.raises(LoadError, match="duplicate"):
            load_facts(con, pd.concat([facts, facts]), run_id="x", loaded_at=LOADED_AT)
        with pytest.raises(LoadError, match="missing columns"):
            load_facts(con, facts.drop(columns=["row_hash"]), run_id="x", loaded_at=LOADED_AT)
        assert load_facts(con, facts.iloc[0:0], run_id="x", loaded_at=LOADED_AT).inserted == 0


def test_read_only_connection_requires_existing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError), connect(tmp_path / "none.duckdb", read_only=True):
        pass


def test_compaction_preserves_data_and_constraints(tmp_path: Path, fx: FxTable) -> None:
    db = tmp_path / "w.duckdb"
    rows = [{"customer_id": f"C{i % 50}", "product_id": f"P{i % 7}"} for i in range(400)]
    with connect(db) as con:
        for run in range(4):
            load_facts(con, _facts(make_rows(*rows), fx), run_id=f"r{run}", loaded_at=LOADED_AT)
            rows = [{**r, "quantity": str(run + 3)} for r in rows]
    stats = storage_stats(db)
    before, after = compact(db)
    assert after <= before
    assert storage_stats(db).free_blocks == 0
    assert stats.used_blocks > 0
    with connect(db) as con:
        assert table_counts(con)["fact_transactions"] == 400
        with pytest.raises(duckdb.ConstraintException):
            con.execute("INSERT INTO fact_transactions SELECT * FROM fact_transactions LIMIT 1")
    assert compact_if_fragmented(db, threshold=0.99) is None
    assert compact_if_fragmented(tmp_path / "missing.duckdb") is None
