"""Pipeline orchestration: read -> map -> validate -> load -> report, with run tracking.

Every invocation creates one ``meta_pipeline_runs`` row and one audit report, whatever
the outcome (``success``, ``success_with_warnings``, ``skipped`` or ``failed``).
"""

from __future__ import annotations

import json
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

from bizpulse import __version__
from bizpulse.config import DEMO_FILES, Settings
from bizpulse.ingestion.fx import FxTable
from bizpulse.ingestion.hashing import sha256_file
from bizpulse.ingestion.mapping import MappingConfig, MappingError, resolve_mapping
from bizpulse.ingestion.normalize import date_parse_ratio
from bizpulse.ingestion.readers import ReadError, read_table
from bizpulse.logging_utils import get_logger
from bizpulse.quality.report import (
    MappingEntry,
    QualityReport,
    RunStatus,
    sample_quarantine,
    write_reports,
)
from bizpulse.quality.validator import QuarantinedRow, validate_batch
from bizpulse.warehouse.connection import connect
from bizpulse.warehouse.loader import LoadError, load_facts
from bizpulse.warehouse.schema import META_DATASETS, META_QUARANTINE, META_RUNS
from bizpulse.warehouse.transform import build_fact_frame

log = get_logger(__name__)
SUCCESS_STATUSES: frozenset[str] = frozenset({"success", "success_with_warnings"})


class PipelineError(RuntimeError):
    """An expected, user-actionable pipeline failure."""


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    dataset: str
    status: RunStatus
    report: QualityReport
    json_report: Path
    md_report: Path

    @property
    def ok(self) -> bool:
        return self.status != "failed"


def utcnow() -> datetime:
    """Current time as naive UTC (the warehouse convention)."""
    return datetime.now(UTC).replace(tzinfo=None)


def _insert_run(con: duckdb.DuckDBPyConnection, report: QualityReport) -> None:
    con.execute(
        f"""INSERT INTO {META_RUNS}
            (run_id, dataset, source_file, file_sha256, status, started_at, code_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            report.run_id,
            report.dataset,
            report.source_file,
            report.file_sha256,
            report.status,
            report.started_at,
            __version__,
        ],
    )


def _finalize_run(con: duckdb.DuckDBPyConnection, report: QualityReport, report_path: str) -> None:
    rows = report.rows
    mapping_json = json.dumps([m.model_dump() for m in report.column_mapping])
    con.execute(
        f"""UPDATE {META_RUNS} SET
                source_format = ?, status = ?, finished_at = ?, duration_seconds = ?,
                rows_read = ?, rows_valid = ?, rows_quarantined = ?, rows_inserted = ?,
                rows_updated = ?, rows_unchanged = ?, pass_rate = ?, mapping_json = ?,
                report_path = ?, error_message = ?
            WHERE run_id = ?""",
        [
            report.source_format,
            report.status,
            report.finished_at,
            report.duration_seconds,
            rows.read,
            rows.valid,
            rows.quarantined,
            rows.inserted,
            rows.updated,
            rows.unchanged,
            report.pass_rate,
            mapping_json,
            report_path,
            report.error or report.skipped_reason,
            report.run_id,
        ],
    )


def _insert_quarantine(
    con: duckdb.DuckDBPyConnection, run_id: str, dataset: str, rows: list[QuarantinedRow]
) -> None:
    if not rows:
        return
    frame = pd.DataFrame(
        {
            "run_id": run_id,
            "dataset": dataset,
            "source_row": [r.source_row for r in rows],
            "reason_codes": [",".join(r.codes) for r in rows],
            "issues_json": [json.dumps([i.as_dict() for i in r.issues]) for r in rows],
            "raw_json": [json.dumps(r.raw, ensure_ascii=False) for r in rows],
        }
    )
    con.register("quarantine_view", frame)
    try:
        con.execute(f"INSERT INTO {META_QUARANTINE} SELECT * FROM quarantine_view")
    finally:
        con.unregister("quarantine_view")


def _upsert_dataset(con: duckdb.DuckDBPyConnection, config: MappingConfig, when: datetime) -> None:
    con.execute(
        f"""INSERT INTO {META_DATASETS} VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (dataset) DO UPDATE SET
                display_name = excluded.display_name,
                domain = excluded.domain,
                reporting_currency = excluded.reporting_currency,
                mapping_json = excluded.mapping_json,
                updated_at = excluded.updated_at""",
        [
            config.dataset,
            config.label,
            config.domain,
            config.reporting_currency,
            config.model_dump_json(),
            when,
        ],
    )


def _process(
    con: duckdb.DuckDBPyConnection,
    source: Path,
    config: MappingConfig,
    fx: FxTable,
    report: QualityReport,
    as_of: datetime,
) -> None:
    """Run the batch through every stage, filling *report* in place (raises on failure)."""
    options = config.read_options
    table = read_table(
        source,
        delimiter=options.delimiter,
        encoding=options.encoding,
        sheet_name=options.sheet_name,
    )
    report.source_format = table.source_format
    resolved = resolve_mapping(
        [str(c) for c in table.frame.columns],
        config,
        sample=table.frame.head(200),
        date_ratio=date_parse_ratio,
    )
    report.column_mapping = [
        MappingEntry(field=name, source=m.source_label, method=m.method, score=round(m.score, 3))
        for name, m in resolved.matches.items()
    ]
    report.unmapped_source_columns = resolved.unmapped_columns
    report.ambiguous_fields = resolved.ambiguous
    report.missing_required_fields = resolved.missing_required
    if resolved.ambiguous:
        report.warnings.append(
            f"ambiguous column matches left unmapped: {sorted(resolved.ambiguous)}"
        )
    if resolved.missing_required:
        raise PipelineError(
            f"required fields could not be mapped: {resolved.missing_required}. "
            f"Add them under `columns:` in the mapping config. "
            f"Unmapped source columns: {resolved.unmapped_columns}"
        )

    result = validate_batch(table.frame, resolved, config, fx, as_of=as_of)
    report.rows.read = result.rows_read
    report.rows.valid = len(result.valid)
    report.rows.quarantined = len(result.quarantined)
    report.pass_rate = round(result.pass_rate, 6)
    report.reason_counts = result.reason_counts
    report.field_completeness = result.stats.completeness
    report.date_parse_methods = result.stats.date_methods
    report.currency_sources = result.stats.currency_sources
    mix = Counter(str(c) for c in result.valid["currency"])
    report.currency_mix = dict(sorted(mix.items(), key=lambda kv: (-kv[1], kv[0])))
    report.converted_rows = sum(n for c, n in mix.items() if c != config.reporting_currency)
    report.quarantine_sample = sample_quarantine(result.quarantined)
    _insert_quarantine(con, report.run_id, config.dataset, result.quarantined)

    ratio = 1.0 - result.pass_rate
    if result.rows_read == 0:
        raise PipelineError("file contains no data rows")
    if ratio > config.max_quarantine_ratio:
        raise PipelineError(
            f"quality gate: {ratio:.1%} of rows were quarantined "
            f"(max {config.max_quarantine_ratio:.0%}); nothing was loaded. "
            "Check the column mapping and the top quarantine reasons."
        )

    cost_coverage = result.stats.completeness.get("unit_cost", 0.0)
    if cost_coverage < 1.0:
        report.warnings.append(
            f"unit cost known for {cost_coverage:.1%} of rows; margin KPIs use that subset"
        )
    defaulted = result.stats.currency_sources.get("dataset_default", 0)
    if defaulted and "currency" not in resolved.matches:
        report.warnings.append(
            f"no currency column: {defaulted:,} rows assumed {config.effective_default_currency}"
        )

    facts = build_fact_frame(result.valid, config, fx)
    load = load_facts(con, facts, run_id=report.run_id, loaded_at=report.started_at)
    report.rows.inserted = load.inserted
    report.rows.updated = load.updated
    report.rows.unchanged = load.unchanged
    _upsert_dataset(con, config, report.started_at)
    report.status = "success_with_warnings" if result.quarantined else "success"


def run_pipeline(
    source: Path,
    config: MappingConfig,
    settings: Settings,
    *,
    force: bool = False,
    as_of: datetime | None = None,
    fx: FxTable | None = None,
) -> PipelineResult:
    """Ingest *source* into the warehouse. Never raises for data problems."""
    settings.ensure_dirs()
    fx_table = fx or FxTable.default()
    started = utcnow()
    clock = time.perf_counter()
    report = QualityReport(
        run_id=uuid.uuid4().hex,
        dataset=config.dataset,
        display_name=config.label,
        source_file=source.name,
        file_sha256=sha256_file(source) if source.is_file() else "",
        status="running",
        started_at=started,
        reporting_currency=config.reporting_currency,
    )
    with connect(settings.warehouse_path) as con:
        _insert_run(con, report)
        previous = con.execute(
            f"""SELECT run_id FROM {META_RUNS}
                WHERE dataset = ? AND file_sha256 = ? AND status IN ('success',
                      'success_with_warnings')
                ORDER BY started_at DESC LIMIT 1""",
            [config.dataset, report.file_sha256],
        ).fetchone()
        if previous and not force:
            report.status = "skipped"
            report.skipped_reason = (
                f"identical file already loaded by run {previous[0]}; use --force to reload"
            )
        else:
            try:
                _process(con, source, config, fx_table, report, as_of or started)
            except (ReadError, MappingError, PipelineError, LoadError) as exc:
                report.status = "failed"
                report.error = str(exc)
            except Exception as exc:
                log.exception("unexpected pipeline failure for %s", config.dataset)
                report.status = "failed"
                report.error = f"unexpected error: {exc!r}"
        report.finished_at = utcnow()
        report.duration_seconds = round(time.perf_counter() - clock, 3)
        json_path, md_path = write_reports(report, settings.reports_dir)
        _finalize_run(con, report, settings.relative(json_path))
    log.info(
        "%s: %s (read=%d valid=%d quarantined=%d inserted=%d updated=%d unchanged=%d) in %.2fs",
        config.dataset,
        report.status,
        report.rows.read,
        report.rows.valid,
        report.rows.quarantined,
        report.rows.inserted,
        report.rows.updated,
        report.rows.unchanged,
        report.duration_seconds or 0.0,
    )
    return PipelineResult(
        run_id=report.run_id,
        dataset=config.dataset,
        status=report.status,
        report=report,
        json_report=json_path,
        md_report=md_path,
    )


def run_demo_pipelines(settings: Settings, *, force: bool = False) -> list[PipelineResult]:
    """Load the three bundled demo datasets from ``<data>/raw``."""
    results: list[PipelineResult] = []
    for dataset, file_name in DEMO_FILES.items():
        path = settings.raw_dir / file_name
        if not path.is_file():
            raise FileNotFoundError(
                f"demo file {file_name} not found: run `bizpulse generate-demo` first"
            )
        results.append(run_pipeline(path, MappingConfig.builtin(dataset), settings, force=force))
    return results
