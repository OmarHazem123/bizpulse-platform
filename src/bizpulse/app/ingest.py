"""Upload -> preview -> ingest helpers behind the dashboard's upload tab (no Streamlit).

An uploaded file is staged under ``<data>/uploads/<uuid>/<name>`` (keeping its original
name for the audit report), ingested with the normal pipeline, and the staging directory
is removed afterwards, so uploads never accumulate on disk.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import duckdb
import pandas as pd

from bizpulse.config import Settings
from bizpulse.ingestion.mapping import MappingConfig, ResolvedMapping, resolve_mapping
from bizpulse.ingestion.normalize import date_parse_ratio
from bizpulse.ingestion.readers import SUPPORTED_EXTENSIONS, read_table
from bizpulse.pipeline import PipelineResult, compact_if_changed, run_pipeline
from bizpulse.schema import CANONICAL_FIELDS, PRICE_FIELDS, REQUIRED_FIELDS, CanonicalField

DATASET_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,48}$")
UPLOAD_TYPES: tuple[str, ...] = tuple(sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS))
PREVIEW_ROWS = 20
NOT_MAPPED = ""


def safe_filename(name: str) -> str:
    """The base name of an uploaded file, restricted to a safe character set."""
    base = Path(name.replace("\\", "/")).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return cleaned or "upload"


def suggest_dataset_name(file_name: str) -> str:
    """A valid dataset key derived from a file name (``Q3 Orders.csv`` -> ``q3_orders``)."""
    stem = re.sub(r"[^a-z0-9]+", "_", Path(file_name).stem.lower()).strip("_")
    if not stem or not stem[0].isalpha():
        stem = f"data_{stem}".rstrip("_")
    return stem[:49] if len(stem) >= 2 else "my_dataset"


def dataset_name_error(name: str) -> str | None:
    if DATASET_PATTERN.match(name):
        return None
    return "use 2-49 characters: lowercase letters, digits and underscores, starting with a letter"


@contextmanager
def staged_upload(data: bytes, file_name: str, settings: Settings) -> Iterator[Path]:
    """Write *data* to a private staging directory; remove it on exit."""
    folder = settings.data_dir / "uploads" / uuid.uuid4().hex
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / safe_filename(file_name)
    try:
        path.write_bytes(data)
        yield path
    finally:
        shutil.rmtree(folder, ignore_errors=True)


@dataclass(frozen=True)
class Preview:
    sample: pd.DataFrame
    columns: list[str]
    mapping: ResolvedMapping


def preview(path: Path, config: MappingConfig) -> Preview:
    """Read *path* and auto-detect its column mapping (raises ``ReadError``/``MappingError``)."""
    options = config.read_options
    table = read_table(
        path,
        delimiter=options.delimiter,
        encoding=options.encoding,
        sheet_name=options.sheet_name,
    )
    columns = [str(c) for c in table.frame.columns]
    mapping = resolve_mapping(
        columns, config, sample=table.frame.head(200), date_ratio=date_parse_ratio
    )
    return Preview(sample=table.frame.head(PREVIEW_ROWS), columns=columns, mapping=mapping)


def mapping_table(mapping: ResolvedMapping) -> pd.DataFrame:
    """One row per canonical field: detected source column, method, score, requirement."""
    rows: list[dict[str, Any]] = []
    for name in CANONICAL_FIELDS:
        match = mapping.source_of(name)
        required = name in REQUIRED_FIELDS
        rows.append(
            {
                "field": name,
                "requirement": "required"
                if required
                else ("one of" if name in PRICE_FIELDS else "optional"),
                "source": match.sources[0] if match and len(match.sources) == 1 else NOT_MAPPED,
                "method": match.method if match else "",
                "score": round(match.score, 3) if match else None,
            }
        )
    return pd.DataFrame(rows)


def explicit_columns(edited: pd.DataFrame, detected: pd.DataFrame) -> dict[CanonicalField, str]:
    """Fields whose source the user changed in the mapping editor (pinned explicitly)."""
    before = dict(zip(detected["field"], detected["source"], strict=True))
    pinned: dict[CanonicalField, str] = {}
    for name, source in zip(edited["field"], edited["source"], strict=True):
        value = "" if source is None or pd.isna(source) else str(source)
        if value and value != before.get(name, ""):
            pinned[cast(CanonicalField, name)] = value
    return pinned


def build_config(
    *,
    dataset: str,
    display_name: str | None,
    domain: str,
    reporting_currency: str,
    timezone: str,
    dayfirst: bool,
    columns: Mapping[CanonicalField, str] | None = None,
) -> MappingConfig:
    """Validated mapping config for an upload (raises ``pydantic.ValidationError``)."""
    return MappingConfig.model_validate(
        {
            "dataset": dataset,
            "display_name": display_name or None,
            "domain": domain,
            "reporting_currency": reporting_currency,
            "timezone": timezone,
            "dayfirst": dayfirst,
            "columns": dict(columns or {}),
        }
    )


def quarantine_frame(rows: Sequence[tuple[Any, ...]]) -> pd.DataFrame:
    """Quarantined rows as a table: source row, reason codes, messages, original values.

    *rows* are ``(source_row, reason_codes, issues_json, raw_json)`` tuples from
    ``meta_quarantine``. Original values keep the source file's own column names.
    """
    records: list[dict[str, Any]] = []
    for source_row, codes, issues_json, raw_json in rows:
        issues = json.loads(issues_json) if issues_json else []
        raw = json.loads(raw_json) if raw_json else {}
        records.append(
            {
                "source row": int(source_row),
                "reasons": str(codes),
                "details": "; ".join(
                    f"{i.get('field')}: {i.get('message')}" for i in issues if isinstance(i, dict)
                ),
                **{str(k): v for k, v in (raw.items() if isinstance(raw, dict) else [])},
            }
        )
    return pd.DataFrame(records)


@dataclass(frozen=True)
class IngestOutcome:
    result: PipelineResult
    compacted: tuple[int, int] | None
    compact_error: str | None


def ingest(path: Path, config: MappingConfig, settings: Settings) -> IngestOutcome:
    """Run the pipeline on a staged upload, then compact if data changed."""
    result = run_pipeline(path, config, settings)
    try:
        compacted = compact_if_changed([result], settings)
    except (OSError, duckdb.Error) as exc:
        return IngestOutcome(result, None, str(exc))
    return IngestOutcome(result, compacted, None)
