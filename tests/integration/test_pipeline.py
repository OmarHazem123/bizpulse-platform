"""End-to-end: demo generator -> pipeline -> warehouse -> reports."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from bizpulse.config import DEMO_FILES, Settings
from bizpulse.ingestion.mapping import MappingConfig
from bizpulse.pipeline import PipelineResult, run_demo_pipelines, run_pipeline
from bizpulse.quality.report import QualityReport
from bizpulse.warehouse.connection import connect
from bizpulse.warehouse.schema import orphan_counts
from tests.conftest import AS_OF

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def loaded(
    demo_raw: Path, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[Settings, list[PipelineResult]]]:
    settings = Settings.from_env(tmp_path_factory.mktemp("loaded") / "data")
    settings.ensure_dirs()
    shutil.copytree(demo_raw, settings.raw_dir, dirs_exist_ok=True)
    yield settings, run_demo_pipelines(settings)


def _quarantine(settings: Settings, run_id: str) -> dict[int, list[str]]:
    with connect(settings.warehouse_path, read_only=True) as con:
        rows = con.execute(
            "SELECT source_row, reason_codes FROM meta_quarantine WHERE run_id = ?", [run_id]
        ).fetchall()
    return {int(r[0]): str(r[1]).split(",") for r in rows}


def test_all_demo_datasets_load(loaded: tuple[Settings, list[PipelineResult]]) -> None:
    _, results = loaded
    assert [r.dataset for r in results] == list(DEMO_FILES)
    assert all(r.status == "success_with_warnings" for r in results)


def test_quarantine_equals_manifest_exactly(loaded: tuple[Settings, list[PipelineResult]]) -> None:
    settings, results = loaded
    for result in results:
        manifest = json.loads(
            (settings.manifests_dir / f"{result.dataset}.json").read_text(encoding="utf-8")
        )
        expected = {e["source_row"]: e["codes"] for e in manifest["dirty_rows"]}
        assert _quarantine(settings, result.run_id) == expected, result.dataset
        rows = result.report.rows
        assert rows.read == manifest["rows_total"]
        assert rows.valid == manifest["rows_expected_valid"]  # messy-but-valid rows load
        assert rows.read == rows.valid + rows.quarantined
        assert rows.valid == rows.inserted + rows.updated + rows.unchanged


def test_star_schema_integrity(loaded: tuple[Settings, list[PipelineResult]]) -> None:
    settings, results = loaded
    with connect(settings.warehouse_path, read_only=True) as con:
        assert orphan_counts(con) == {"customer_key": 0, "product_key": 0, "date_key": 0}
        per_dataset = dict(
            con.execute("SELECT dataset, count(*) FROM fact_transactions GROUP BY 1").fetchall()
        )
        assert per_dataset == {r.dataset: r.report.rows.valid for r in results}
        currencies = dict(
            con.execute("SELECT dataset, reporting_currency FROM meta_datasets").fetchall()
        )
        assert currencies == {"ecommerce": "USD", "saas": "USD", "retail_fb": "EUR"}
        anonymous = con.execute(
            "SELECT count(*) FROM dim_customers WHERE is_anonymous AND customer_id = 'GUEST'"
        ).fetchone()
        assert anonymous == (1,)
        converted = con.execute(
            """SELECT count(*) FROM fact_transactions
               WHERE dataset = 'ecommerce' AND currency_original <> 'USD' AND fx_rate <> 1"""
        ).fetchone()
        assert converted is not None and converted[0] > 0
        runs = con.execute(
            "SELECT count(*) FROM meta_pipeline_runs WHERE status = 'success_with_warnings'"
        ).fetchone()
        assert runs == (3,)


def test_reports_are_written_without_local_paths(
    loaded: tuple[Settings, list[PipelineResult]],
) -> None:
    settings, results = loaded
    for result in results:
        report = QualityReport.model_validate_json(result.json_report.read_text(encoding="utf-8"))
        assert report.source_file == DEMO_FILES[result.dataset]
        assert report.reason_counts
        assert len(report.quarantine_sample) <= 25
        text = result.json_report.read_text(encoding="utf-8") + result.md_report.read_text(
            encoding="utf-8"
        )
        assert str(settings.data_dir) not in text
        assert "## Quarantine reasons" in result.md_report.read_text(encoding="utf-8")


def _fresh(tmp_path: Path, demo_raw: Path) -> Settings:
    settings = Settings.from_env(tmp_path / "data")
    settings.ensure_dirs()
    shutil.copy(demo_raw / DEMO_FILES["ecommerce"], settings.raw_dir)
    return settings


def test_batch_and_row_idempotency(tmp_path: Path, demo_raw: Path) -> None:
    settings = _fresh(tmp_path, demo_raw)
    config = MappingConfig.builtin("ecommerce")
    source = settings.raw_dir / DEMO_FILES["ecommerce"]
    first = run_pipeline(source, config, settings, as_of=AS_OF)
    n_valid = first.report.rows.valid

    again = run_pipeline(source, config, settings, as_of=AS_OF)
    assert again.status == "skipped"
    assert first.run_id in (again.report.skipped_reason or "")

    forced = run_pipeline(source, config, settings, force=True, as_of=AS_OF)
    assert (forced.report.rows.inserted, forced.report.rows.updated) == (0, 0)
    assert forced.report.rows.unchanged == n_valid

    # Change one valid row and append one new row (different file bytes -> not skipped).
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    quarantined = set(_quarantine(settings, first.run_id))
    target = next(i for i in range(len(frame)) if i + 1 not in quarantined)
    frame.at[target, "Qty"] = str(int(str(frame.at[target, "Qty"])) + 10)
    extra = frame.iloc[[target]].copy()
    extra["Line ID"] = "NEW-LINE-1"
    pd.concat([frame, extra]).to_csv(source, index=False, lineterminator="\n")
    changed = run_pipeline(source, config, settings, as_of=AS_OF)
    rows = changed.report.rows
    assert (rows.inserted, rows.updated, rows.unchanged) == (1, 1, n_valid - 1)


def test_quality_gate_blocks_wrong_mapping(tmp_path: Path, demo_raw: Path) -> None:
    settings = _fresh(tmp_path, demo_raw)
    wrong = MappingConfig.builtin("ecommerce").model_copy(
        update={
            "columns": {**MappingConfig.builtin("ecommerce").columns, "quantity": "Product Title"}
        }
    )
    result = run_pipeline(settings.raw_dir / DEMO_FILES["ecommerce"], wrong, settings)
    assert result.status == "failed"
    assert "quality gate" in (result.report.error or "")
    assert result.report.reason_counts["quantity.unparseable"] > 0
    with connect(settings.warehouse_path, read_only=True) as con:
        assert con.execute("SELECT count(*) FROM fact_transactions").fetchone() == (0,)
        status = con.execute("SELECT status FROM meta_pipeline_runs").fetchone()
        assert status == ("failed",)


def test_unmappable_and_unreadable_files_fail_cleanly(tmp_path: Path) -> None:
    settings = Settings.from_env(tmp_path / "data")
    bad = tmp_path / "notes.csv"
    bad.write_text("foo,bar\n1,2\n", encoding="utf-8")
    result = run_pipeline(bad, MappingConfig(dataset="acme"), settings)
    assert result.status == "failed"
    assert "required fields could not be mapped" in (result.report.error or "")
    assert result.md_report.exists()

    unsupported = tmp_path / "data.json"
    unsupported.write_text("{}", encoding="utf-8")
    failed = run_pipeline(unsupported, MappingConfig(dataset="acme"), settings)
    assert failed.status == "failed"
    assert "unsupported file type" in (failed.report.error or "")


def test_custom_file_with_auto_detected_columns(tmp_path: Path) -> None:
    settings = Settings.from_env(tmp_path / "data")
    path = tmp_path / "shop.csv"
    path.write_text(
        "Transaction ID;Date;Customer;SKU;Qty;Total;Status\n"
        "1;2026-01-05;c1;A;2;19,98;paid\n"
        "2;2026-01-06;c2;B;1;5,00;refunded\n"
        "3;2026-01-06;c2;B;1;5,00;cancelled\n",
        encoding="utf-8",
    )
    result = run_pipeline(path, MappingConfig(dataset="shop"), settings, as_of=AS_OF)
    assert result.status == "success"
    methods = {m.field: m.method for m in result.report.column_mapping}
    assert methods["line_total"] == "synonym"
    assert any("no currency column" in w for w in result.report.warnings)
    with duckdb.connect(str(settings.warehouse_path), read_only=True) as con:
        rows = con.execute(
            "SELECT unit_price, status FROM fact_transactions ORDER BY transaction_id"
        ).fetchall()
    assert rows == [(9.99, "completed"), (5.0, "refunded"), (5.0, "cancelled")]
