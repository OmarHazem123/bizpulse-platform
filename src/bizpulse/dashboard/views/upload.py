"""Tab 5: Company Data Upload & Quality Audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from bizpulse.config import Settings
from bizpulse.dashboard import charts, state
from bizpulse.dashboard.ingest import (
    NOT_MAPPED,
    UPLOAD_TYPES,
    build_config,
    dataset_name_error,
    explicit_columns,
    ingest,
    mapping_table,
    preview,
    staged_upload,
    suggest_dataset_name,
)
from bizpulse.dashboard.palette import Palette
from bizpulse.ingestion.mapping import MappingConfig, MappingError
from bizpulse.ingestion.readers import ReadError

DOMAINS: tuple[str, ...] = ("custom", "ecommerce", "saas", "retail_fb")
LAST_RUN = "upload_last_run"
STATUS_ICON: dict[str, str] = {
    "success": ":material/check_circle:",
    "success_with_warnings": ":material/warning:",
    "skipped": ":material/skip_next:",
    "failed": ":material/error:",
}


def _settings_form(file_name: str, existing: list[str]) -> dict[str, Any] | None:
    c1, c2, c3 = st.columns(3)
    dataset = c1.text_input(
        "Dataset key", suggest_dataset_name(file_name), key="up_dataset"
    ).strip()
    display = c2.text_input("Display name", "", key="up_display", placeholder=file_name)
    domain = c3.selectbox("Business type", DOMAINS, key="up_domain")
    c4, c5, c6 = st.columns(3)
    currency = c4.text_input("Reporting currency (ISO)", "USD", key="up_currency").strip()
    timezone = c5.text_input("Business timezone (IANA)", "UTC", key="up_tz").strip()
    dayfirst = c6.checkbox("Dates are day-first (31/01/2026)", key="up_dayfirst")
    problem = dataset_name_error(dataset)
    if problem:
        st.error(f"Dataset key: {problem}.")
        return None
    if dataset in existing:
        st.warning(
            f"Dataset **{dataset}** already exists: rows will be upserted into it (new rows "
            "inserted, changed rows updated). Use a new key for a separate dataset."
        )
    return {
        "dataset": dataset,
        "display_name": display or None,
        "domain": domain,
        "reporting_currency": currency,
        "timezone": timezone,
        "dayfirst": dayfirst,
    }


def _mapping_editor(detected: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    edited: pd.DataFrame = st.data_editor(
        detected,
        hide_index=True,
        disabled=["field", "requirement", "method", "score"],
        column_config={
            "source": st.column_config.SelectboxColumn(
                "source column", options=[NOT_MAPPED, *columns], required=False
            )
        },
        key="up_mapping",
    )
    return edited


def _show_result(outcome: dict[str, Any], pal: Palette) -> None:
    status = str(outcome["status"])
    report = outcome["report"]
    rows = report["rows"]
    message = {
        "success": "Loaded: every row passed the data contract.",
        "success_with_warnings": "Loaded: valid rows are in the warehouse; invalid rows were "
        "quarantined with exact reasons.",
        "skipped": f"Skipped: {report.get('skipped_reason')}",
        "failed": f"Failed, nothing loaded: {report.get('error')}",
    }[status]
    box = st.error if status == "failed" else (st.success if status == "success" else st.info)
    box(f"**{report['dataset']}**: {message}", icon=STATUS_ICON.get(status))
    if outcome.get("compact_error"):
        st.caption(f"Warehouse not compacted ({outcome['compact_error']}); data is intact.")
    cols = st.columns(6)
    rate = report.get("pass_rate")
    for col, (label, value) in zip(
        cols,
        (
            ("Rows read", f"{rows['read']:,}"),
            ("Pass rate", "n/a" if rate is None else f"{rate:.2%}"),
            ("Quarantined", f"{rows['quarantined']:,}"),
            ("Inserted", f"{rows['inserted']:,}"),
            ("Updated", f"{rows['updated']:,}"),
            ("Unchanged", f"{rows['unchanged']:,}"),
        ),
        strict=True,
    ):
        col.metric(label, value, border=True)
    if report.get("missing_required_fields"):
        st.warning(f"Required fields not mapped: {report['missing_required_fields']}")
    reasons = report.get("reason_counts") or {}
    if reasons:
        left, right = st.columns(2)
        with left:
            st.markdown("**Quarantine reasons**")
            st.plotly_chart(charts.counts_bars(reasons, pal, name="rows"), key="reasons")
        with right:
            st.markdown("**Sample of quarantined rows**")
            sample = pd.DataFrame(
                [
                    {
                        "source row": item["source_row"],
                        "reasons": ", ".join(item["codes"]),
                        "detail": "; ".join(i["message"] for i in item["issues"]),
                    }
                    for item in report.get("quarantine_sample") or []
                ]
            )
            st.dataframe(sample, hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download audit report (JSON)",
        outcome["json"],
        file_name=f"{report['dataset']}_{report['run_id']}.json",
        mime="application/json",
        key="dl_json",
    )
    d2.download_button(
        "Download audit report (Markdown)",
        outcome["markdown"],
        file_name=f"{report['dataset']}_{report['run_id']}.md",
        mime="text/markdown",
        key="dl_md",
    )


def _upload(settings: Settings, existing: list[str], pal: Palette) -> None:
    st.subheader("Load your company data")
    st.caption(
        "CSV, Excel (.xlsx) or Parquet with one row per order line. Columns are detected "
        "automatically (synonyms, fuzzy matching, date sniffing); adjust any mapping "
        "below. Invalid rows are quarantined with exact reasons; valid rows load. If more "
        "than half the rows fail, nothing loads (the column mapping is probably wrong)."
    )
    upload = st.file_uploader("Data file", type=list(UPLOAD_TYPES), key="up_file")
    if upload is None:
        return
    data = upload.getvalue()
    form = _settings_form(upload.name, existing)
    if form is None:
        return
    try:
        base_config = build_config(**form)
        with staged_upload(data, upload.name, settings) as path:
            found = preview(path, base_config)
    except (ReadError, MappingError, ValidationError) as exc:
        st.error(f"Cannot read this file: {exc}")
        return
    st.markdown(f"**Preview** (first {len(found.sample)} of the file's rows)")
    st.dataframe(found.sample, hide_index=True)
    st.markdown("**Column mapping** (auto-detected; change a source column to pin it)")
    detected = mapping_table(found.mapping)
    edited = _mapping_editor(detected, found.columns)
    if found.mapping.missing_required:
        st.warning(
            f"Not detected: {', '.join(found.mapping.missing_required)}. Pick their source "
            "columns above, or the run will fail."
        )
    if found.mapping.ambiguous:
        st.info(f"Ambiguous matches left for you to choose: {found.mapping.ambiguous}")
    if not st.button("Validate and load", type="primary", key="up_run"):
        return
    try:
        config: MappingConfig = build_config(**form, columns=explicit_columns(edited, detected))
    except ValidationError as exc:
        st.error(f"Invalid mapping: {exc}")
        return
    with (
        st.spinner("Validating rows and loading the warehouse ..."),
        staged_upload(data, upload.name, settings) as path,
    ):
        outcome = ingest(path, config, settings)
    result = outcome.result
    st.session_state[LAST_RUN] = {
        "status": result.status,
        "report": result.report.model_dump(mode="json"),
        "json": result.json_report.read_bytes(),
        "markdown": result.md_report.read_bytes(),
        "compact_error": outcome.compact_error,
    }
    state.clear()
    st.rerun()


def _history(settings: Settings, stamp: float) -> None:
    st.subheader("Pipeline run history (audit trail)")
    runs = state.runs(str(settings.warehouse_path), stamp)
    if runs.empty:
        st.caption("No pipeline runs yet.")
        return
    st.dataframe(runs.drop(columns=["report_path"]), hide_index=True)
    labels = [
        f"{r.started_at:%Y-%m-%d %H:%M:%S} | {r.dataset} | {r.status} | {r.source_file}"
        for r in runs.itertuples()
    ]
    pick = st.selectbox(
        "Open the audit report of run",
        range(len(runs)),
        format_func=lambda i: labels[i],
        key="run_pick",
    )
    relative = runs.iloc[int(pick)]["report_path"]
    if not isinstance(relative, str) or not relative:
        st.caption("This run has no report file.")
        return
    md_path = report_path_for(settings, str(Path(relative).with_suffix(".md")))
    if md_path is None:
        st.caption("Report file not found (it may have been cleaned up).")
        return
    with st.expander("Audit report"):
        st.markdown(md_path.read_text(encoding="utf-8"))


def render(settings: Settings, stamp: float, existing: list[str], pal: Palette) -> None:
    last = st.session_state.get(LAST_RUN)
    if last:
        _show_result(last, pal)
        if st.button("Load another file", key="up_reset"):
            del st.session_state[LAST_RUN]
            st.rerun()
        st.divider()
    else:
        _upload(settings, existing, pal)
        st.divider()
    _history(settings, stamp)


def report_path_for(settings: Settings, relative: str) -> Path | None:
    """Resolve a stored report path safely inside the data directory."""
    path = (settings.data_dir / relative).resolve()
    return path if settings.data_dir in path.parents and path.is_file() else None
