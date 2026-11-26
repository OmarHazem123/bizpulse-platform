from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from bizpulse.config import ENV_DATA_DIR, Settings
from bizpulse.ingestion.fx import FxTable
from bizpulse.quality.report import (
    QualityReport,
    RowCounts,
    render_markdown,
    sample_quarantine,
    write_reports,
)
from bizpulse.quality.validator import QuarantinedRow
from bizpulse.schema import QualityIssue


def _row(n: int, code: str) -> QuarantinedRow:
    return QuarantinedRow(n, (QualityIssue("f", code, "msg", "v|x"),), {"col": "v|x"})


def test_sample_prefers_one_row_per_reason() -> None:
    rows = [_row(i, "a.x") for i in range(1, 40)] + [_row(99, "b.y")]
    sample = sample_quarantine(rows, limit=5)
    assert [s.source_row for s in sample] == [1, 2, 3, 4, 99]


def test_report_roundtrip_and_markdown(tmp_path: Path) -> None:
    report = QualityReport(
        run_id="abc",
        dataset="acme",
        display_name="Acme",
        source_file="orders.csv",
        file_sha256="0" * 64,
        status="success_with_warnings",
        started_at=datetime(2026, 1, 1),
        reporting_currency="USD",
        rows=RowCounts(read=10, valid=9, quarantined=1, inserted=9),
        pass_rate=0.9,
        reason_counts={"a.x": 1},
        quarantine_sample=sample_quarantine([_row(3, "a.x")]),
        warnings=["careful"],
    )
    json_path, md_path = write_reports(report, tmp_path)
    loaded = QualityReport.model_validate(json.loads(json_path.read_text(encoding="utf-8")))
    assert loaded == report
    markdown = md_path.read_text(encoding="utf-8")
    for heading in ("Row reconciliation", "Quarantine reasons", "Warnings", "Quarantine sample"):
        assert f"## {heading}" in markdown
    assert "v\\|x" in markdown  # pipes escaped inside tables
    assert "\r" not in md_path.read_bytes().decode("utf-8")
    assert "90.00 %" in render_markdown(report)


def test_settings_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path / "elsewhere"))
    settings = Settings.from_env()
    assert settings.data_dir == (tmp_path / "elsewhere").resolve()
    assert settings.warehouse_path.name == "bizpulse.duckdb"
    assert settings.relative(settings.reports_dir / "x" / "r.md") == "reports/x/r.md"
    assert settings.relative(tmp_path / "outside.md") == "outside.md"  # never absolute


def test_fx_table() -> None:
    fx = FxTable.default()
    assert fx.rate("USD", "USD") == 1.0
    assert fx.rate("EUR", "USD") == pytest.approx(1.08)
    assert fx.rate("USD", "EUR") == pytest.approx(1 / 1.08)
    with pytest.raises(ValidationError):
        FxTable(as_of="x", rates_to_usd={"USDX": 1.0})
    with pytest.raises(ValidationError):
        FxTable(as_of="x", rates_to_usd={"USD": 0.0})
