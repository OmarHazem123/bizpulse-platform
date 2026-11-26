"""Data-quality audit reports (JSON for machines, Markdown for people)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from bizpulse.quality.validator import QuarantinedRow

RunStatus = Literal["running", "success", "success_with_warnings", "failed", "skipped"]
SAMPLE_SIZE = 25


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MappingEntry(_Model):
    field: str
    source: str
    method: str
    score: float


class IssueOut(_Model):
    field: str
    code: str
    message: str
    value: str | None = None


class QuarantineSample(_Model):
    source_row: int
    codes: list[str]
    issues: list[IssueOut]
    raw: dict[str, str | None]


class RowCounts(_Model):
    read: int = 0
    valid: int = 0
    quarantined: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0


class QualityReport(_Model):
    report_version: int = 1
    run_id: str
    dataset: str
    display_name: str
    source_file: str
    source_format: str | None = None
    file_sha256: str
    status: RunStatus
    started_at: datetime
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    reporting_currency: str
    rows: RowCounts = Field(default_factory=RowCounts)
    pass_rate: float | None = None
    column_mapping: list[MappingEntry] = Field(default_factory=list)
    unmapped_source_columns: list[str] = Field(default_factory=list)
    ambiguous_fields: dict[str, list[str]] = Field(default_factory=dict)
    missing_required_fields: list[str] = Field(default_factory=list)
    reason_counts: dict[str, int] = Field(default_factory=dict)
    field_completeness: dict[str, float] = Field(default_factory=dict)
    currency_mix: dict[str, int] = Field(default_factory=dict)
    currency_sources: dict[str, int] = Field(default_factory=dict)
    converted_rows: int = 0
    date_parse_methods: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    skipped_reason: str | None = None
    quarantine_sample: list[QuarantineSample] = Field(default_factory=list)


def sample_quarantine(
    rows: list[QuarantinedRow], limit: int = SAMPLE_SIZE
) -> list[QuarantineSample]:
    """A diverse sample: first row of each reason code, then fill in source order."""
    chosen: list[QuarantinedRow] = []
    seen_codes: set[str] = set()
    for row in rows:
        if not set(row.codes) <= seen_codes:
            chosen.append(row)
            seen_codes.update(row.codes)
    for row in rows:
        if len(chosen) >= limit:
            break
        if row not in chosen:
            chosen.append(row)
    chosen = sorted(chosen[:limit], key=lambda r: r.source_row)
    return [
        QuarantineSample(
            source_row=row.source_row,
            codes=row.codes,
            issues=[IssueOut(**issue.as_dict()) for issue in row.issues],
            raw=row.raw,
        )
        for row in chosen
    ]


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f} %"


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: QualityReport) -> str:
    """Human-readable audit report."""
    icon = {
        "success": "PASS",
        "success_with_warnings": "PASS WITH WARNINGS",
        "failed": "FAILED",
        "skipped": "SKIPPED",
        "running": "RUNNING",
    }[report.status]
    rows = report.rows
    lines = [
        f"# Data Quality Audit: {report.display_name}",
        "",
        f"**Status:** {icon} &nbsp;·&nbsp; **Pass rate:** {_fmt_pct(report.pass_rate)}"
        f" &nbsp;·&nbsp; **Run:** `{report.run_id}`",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Dataset | `{report.dataset}` |",
        f"| Source file | `{report.source_file}` ({report.source_format or 'unknown'}) |",
        f"| File SHA-256 | `{report.file_sha256[:16]}…` |",
        f"| Started (UTC) | {report.started_at.isoformat(timespec='seconds')} |",
        f"| Duration | {report.duration_seconds or 0:.2f} s |",
        f"| Reporting currency | {report.reporting_currency} |",
        "",
    ]
    if report.error:
        lines += ["## Error", "", f"> {report.error}", ""]
    if report.skipped_reason:
        lines += ["## Skipped", "", f"> {report.skipped_reason}", ""]
    lines += [
        "## Row reconciliation",
        "",
        "| Read | Valid | Quarantined | Inserted | Updated | Unchanged |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {rows.read:,} | {rows.valid:,} | {rows.quarantined:,} | {rows.inserted:,}"
        f" | {rows.updated:,} | {rows.unchanged:,} |",
        "",
    ]
    if report.warnings:
        lines += ["## Warnings", "", *[f"- {w}" for w in report.warnings], ""]
    if report.reason_counts:
        lines += ["## Quarantine reasons", "", "| Reason code | Rows |", "| --- | ---: |"]
        lines += [f"| `{code}` | {count:,} |" for code, count in report.reason_counts.items()]
        lines.append("")
    if report.column_mapping:
        lines += [
            "## Column mapping",
            "",
            "| Canonical field | Source column | Method | Score |",
            "| --- | --- | --- | ---: |",
        ]
        lines += [
            f"| `{m.field}` | {_cell(m.source)} | {m.method} | {m.score:.2f} |"
            for m in report.column_mapping
        ]
        lines.append("")
    if report.missing_required_fields:
        lines += [
            "**Missing required fields:** "
            + ", ".join(f"`{f}`" for f in report.missing_required_fields),
            "",
        ]
    if report.ambiguous_fields:
        lines += ["**Ambiguous (map explicitly):**", ""]
        lines += [f"- `{k}`: {', '.join(v)}" for k, v in report.ambiguous_fields.items()]
        lines.append("")
    if report.unmapped_source_columns:
        lines += [
            "**Unmapped source columns:** "
            + ", ".join(f"`{_cell(c)}`" for c in report.unmapped_source_columns),
            "",
        ]
    if report.field_completeness:
        lines += ["## Field completeness", "", "| Field | Filled |", "| --- | ---: |"]
        lines += [f"| `{k}` | {_fmt_pct(v)} |" for k, v in report.field_completeness.items()]
        lines.append("")
    if report.currency_mix or report.date_parse_methods:
        lines += ["## Normalization", ""]
        if report.currency_mix:
            mix = ", ".join(f"{k}: {v:,}" for k, v in report.currency_mix.items())
            lines.append(f"- Currency mix (valid rows): {mix}")
            lines.append(
                f"- Rows converted to {report.reporting_currency}: {report.converted_rows:,}"
            )
        if report.currency_sources:
            src = ", ".join(f"{k}: {v:,}" for k, v in report.currency_sources.items())
            lines.append(f"- Currency resolved from: {src}")
        if report.date_parse_methods:
            dates = ", ".join(f"{k}: {v:,}" for k, v in report.date_parse_methods.items())
            lines.append(f"- Date parsing methods: {dates}")
        lines.append("")
    if report.quarantine_sample:
        lines += [
            f"## Quarantine sample (first {len(report.quarantine_sample)})",
            "",
            "| Row | Reason codes | Detail |",
            "| ---: | --- | --- |",
        ]
        for sample in report.quarantine_sample:
            detail = "; ".join(
                f"{i.field}={i.value!r}: {i.message}" if i.value is not None else i.message
                for i in sample.issues
            )
            lines.append(f"| {sample.source_row} | {', '.join(sample.codes)} | {_cell(detail)} |")
        lines.append("")
    return "\n".join(lines)


def write_reports(report: QualityReport, reports_dir: Path) -> tuple[Path, Path]:
    """Write ``<reports_dir>/<dataset>/<run_id>.{json,md}`` and return both paths."""
    target = reports_dir / report.dataset
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / f"{report.run_id}.json"
    md_path = target / f"{report.run_id}.md"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    md_path.write_text(render_markdown(report) + "\n", encoding="utf-8", newline="\n")
    return json_path, md_path
