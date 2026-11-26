"""Batch validation: normalization -> contract -> in-batch dedupe -> quarantine.

Invariant (asserted by tests): ``rows_read == len(valid) + len(quarantined)``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import numpy as np
import pandas as pd
from pydantic import ValidationError

from bizpulse.ingestion.fx import FxTable
from bizpulse.ingestion.hashing import row_hashes, transaction_key
from bizpulse.ingestion.mapping import MappingConfig, ResolvedMapping
from bizpulse.ingestion.normalize import NormalizationStats, normalize_frame, raw_text
from bizpulse.quality.contracts import (
    check_record,
    contract_context,
    issues_from_validation_error,
)
from bizpulse.schema import NORMALIZED_COLUMNS, QualityIssue


@dataclass(frozen=True)
class QuarantinedRow:
    source_row: int  # 1-based data row number (header excluded)
    issues: tuple[QualityIssue, ...]
    raw: dict[str, str | None]

    @property
    def codes(self) -> list[str]:
        return sorted({issue.code for issue in self.issues})


@dataclass
class ValidationResult:
    valid: pd.DataFrame
    quarantined: list[QuarantinedRow]
    stats: NormalizationStats
    rows_read: int

    @property
    def reason_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for row in self.quarantined:
            counts.update(row.codes)
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    @property
    def pass_rate(self) -> float:
        return 1.0 if self.rows_read == 0 else len(self.valid) / self.rows_read


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Frame rows as dicts with NaN/NaT replaced by ``None``."""
    as_object = frame.astype(object)
    none: Any = None  # pandas-stubs does not accept None as `other`
    cleaned = as_object.where(pd.notna(as_object), none)
    return cast(list[dict[str, Any]], cleaned.to_dict(orient="records"))


def validate_batch(
    raw_frame: pd.DataFrame,
    mapping: ResolvedMapping,
    config: MappingConfig,
    fx: FxTable,
    *,
    as_of: datetime,
) -> ValidationResult:
    batch = normalize_frame(raw_frame, mapping, config, fx)
    frame = batch.frame
    issues = batch.issues
    context = contract_context(as_of)

    for pos, record in enumerate(_records(frame)):
        if batch.empty_rows.iat[pos]:
            continue
        try:
            check_record(record, context)
        except ValidationError as exc:
            already = frozenset(issue.field for issue in issues[pos])
            issues[pos].extend(issues_from_validation_error(exc, already))

    # In-batch dedupe among contract-valid rows (first occurrence wins).
    clean = np.array([not row_issues for row_issues in issues], dtype=bool)
    clean_positions: list[int] = np.flatnonzero(clean).tolist()
    hashes = [""] * len(frame)
    if clean_positions:
        digests = row_hashes(frame.iloc[clean_positions], NORMALIZED_COLUMNS)
        for pos, digest in zip(clean_positions, digests, strict=True):
            hashes[pos] = digest
    transaction_ids = frame["transaction_id"].tolist()
    first_row: dict[str, tuple[int, str]] = {}
    for pos in clean_positions:
        tid = str(transaction_ids[pos])
        digest = hashes[pos]
        if tid not in first_row:
            first_row[tid] = (pos + 1, digest)
            continue
        origin, origin_hash = first_row[tid]
        if digest == origin_hash:
            issue = QualityIssue(
                "row", "row.duplicate", f"exact duplicate of source row {origin}", tid
            )
        else:
            issue = QualityIssue(
                "transaction_id",
                "transaction_id.conflicting_duplicate",
                f"transaction_id already used by source row {origin} with different values",
                tid,
            )
        issues[pos].append(issue)

    is_valid = np.array([not row_issues for row_issues in issues], dtype=bool)
    raw_reset = raw_frame.reset_index(drop=True)
    columns = [str(c) for c in raw_reset.columns]
    raw_rows: list[list[Any]] = raw_reset.to_numpy(dtype=object).tolist()
    quarantined = [
        QuarantinedRow(
            source_row=pos + 1,
            issues=tuple(issues[pos]),
            raw={col: raw_text(raw_rows[pos][j]) for j, col in enumerate(columns)},
        )
        for pos in np.flatnonzero(~is_valid).tolist()
    ]

    valid = frame.loc[is_valid].copy()
    valid["source_row"] = valid.index + 1
    valid["row_hash"] = [hashes[pos] for pos in np.flatnonzero(is_valid).tolist()]
    valid["transaction_key"] = [
        transaction_key(config.dataset, str(tid)) for tid in valid["transaction_id"]
    ]
    anonymous = set(config.anonymous_customer_ids)
    valid["is_anonymous"] = valid["customer_id"].isin(anonymous)
    return ValidationResult(
        valid=valid.reset_index(drop=True),
        quarantined=quarantined,
        stats=batch.stats,
        rows_read=len(frame),
    )
