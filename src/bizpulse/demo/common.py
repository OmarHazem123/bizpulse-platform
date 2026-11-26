"""Shared machinery for the deterministic demo generators.

* A fixed 18-month window (never derived from today's date).
* Seasonality multipliers (trend x weekly x annual x events).
* A dirty-row injector that records the expected reason code of every defect in a
  manifest, which is the ground truth for the integration tests.
* Byte-deterministic writers for CSV, XLSX (zip timestamps and core properties
  pinned) and Parquet.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

WINDOW_START = date(2025, 5, 1)
WINDOW_END = date(2026, 10, 31)
FIXED_ZIP_TIME = (2026, 11, 1, 0, 0, 0)
FIXED_DOC_TIME = b"2026-11-01T00:00:00Z"

DefectCode = Literal[
    "order_date.unparseable",
    "order_date.out_of_range",
    "transaction_id.missing",
    "customer_id.missing",
    "product_id.missing",
    "quantity.non_positive",
    "quantity.unparseable",
    "unit_price.negative",
    "unit_price.unparseable",
    "unit_price.missing",
    "unit_cost.negative",
    "discount_amount.exceeds_gross",
    "currency.unsupported",
    "status.invalid",
    "row.empty",
    "row.duplicate",
    "transaction_id.conflicting_duplicate",
]


def window_days() -> pd.DatetimeIndex:
    return pd.date_range(WINDOW_START, WINDOW_END, freq="D")


def seasonal_multiplier(
    days: pd.DatetimeIndex,
    *,
    weekly: Sequence[float],
    monthly: Mapping[int, float],
    monthly_growth: float,
    events: Mapping[str, float] | None = None,
) -> npt.NDArray[np.float64]:
    """Trend x day-of-week x month-of-year x one-off event multipliers."""
    months_elapsed = (days.year - WINDOW_START.year) * 12 + (days.month - WINDOW_START.month)
    trend = (1.0 + monthly_growth) ** (months_elapsed + days.day / days.days_in_month)
    dow = np.asarray(weekly, dtype=np.float64)[days.dayofweek]
    moy = np.array([monthly[m] for m in days.month], dtype=np.float64)
    mult = np.asarray(trend, dtype=np.float64) * dow * moy
    for day, factor in (events or {}).items():
        mult[days == pd.Timestamp(day)] *= factor
    return mult


@dataclass(frozen=True)
class Anomaly:
    name: str
    kind: Literal["drop", "spike", "rate_spike"]
    metric: str
    start: str
    end: str
    expected_severity: Literal["CRITICAL", "WARNING"]
    description: str
    slice: dict[str, str] | None = None
    #: "day" anomalies are point/short events for daily detectors; "month" anomalies
    #: are slow drifts surfaced by cohort/retention analytics instead.
    granularity: Literal["day", "month"] = "day"


@dataclass
class DemoTable:
    """A generated dataset before dirty-row injection."""

    dataset: str
    frame: pd.DataFrame  # source-shaped, object dtype, messy-but-valid formatting
    columns: dict[str, str | tuple[str, ...]]  # canonical field -> source column(s)
    gross: npt.NDArray[np.float64]  # quantity x unit price (source currency) per row
    anomalies: list[Anomaly]
    messy_valid: dict[str, int] = field(default_factory=dict)
    has_currency_column: bool = True


@dataclass
class Manifest:
    dataset: str
    file: str
    format: str
    seed: int
    scale: float
    window: dict[str, str]
    rows_total: int
    rows_dirty: int
    rows_expected_valid: int
    defect_counts: dict[str, int]
    dirty_rows: list[dict[str, Any]]
    messy_valid: dict[str, int]
    anomalies: list[dict[str, Any]]

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(asdict(self), indent=2, sort_keys=False) + "\n"
        path.write_text(text, encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# dirty-row injection
# ---------------------------------------------------------------------------
_UNPARSEABLE_DATES = ("not-a-date", "2025-13-45", "31/31/2025", "yesterday")
_OUT_OF_RANGE_DATES = ("2099-01-15 10:00:00", "1999-12-31 23:00:00")


def inject_defects(
    table: DemoTable, rng: np.random.Generator, *, rate: float = 0.012, blank: Any = ""
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Insert one-defect dirty rows; return the dirty frame and manifest entries.

    Value defects mutate a row in place. Duplicates are inserted right after their
    (untouched) original, so "first occurrence wins" keeps the original.
    """
    frame = table.frame.copy()
    cols = table.columns
    codes: list[DefectCode] = [
        "order_date.unparseable",
        "order_date.out_of_range",
        "transaction_id.missing",
        "customer_id.missing",
        "product_id.missing",
        "quantity.non_positive",
        "quantity.unparseable",
        "unit_price.negative",
        "unit_price.unparseable",
        "unit_price.missing",
        "unit_cost.negative",
        "discount_amount.exceeds_gross",
        "status.invalid",
        "row.empty",
        "row.duplicate",
        "transaction_id.conflicting_duplicate",
    ]
    if table.has_currency_column:
        codes.append("currency.unsupported")
    n_rows = len(frame)
    n_dirty = max(len(codes), round(n_rows * rate))
    per_code = n_dirty // len(codes)
    picks = rng.choice(n_rows, size=per_code * len(codes), replace=False)
    assignments: dict[int, DefectCode] = {
        int(pos): codes[i // per_code] for i, pos in enumerate(picks.tolist())
    }

    def col(name: str) -> str:
        source = cols[name]
        return source if isinstance(source, str) else source[-1]

    mutators: dict[str, Callable[[int], None]] = {
        "order_date.unparseable": lambda i: _set(
            frame, i, col("order_date"), _UNPARSEABLE_DATES[i % len(_UNPARSEABLE_DATES)]
        ),
        "order_date.out_of_range": lambda i: _set(
            frame, i, col("order_date"), _OUT_OF_RANGE_DATES[i % 2]
        ),
        "transaction_id.missing": lambda i: _set(frame, i, col("transaction_id"), blank),
        "customer_id.missing": lambda i: _set(frame, i, col("customer_id"), blank),
        "product_id.missing": lambda i: _set(frame, i, col("product_id"), blank),
        "quantity.non_positive": lambda i: _set(frame, i, col("quantity"), ("0", "-2")[i % 2]),
        "quantity.unparseable": lambda i: _set(frame, i, col("quantity"), "two"),
        "unit_price.negative": lambda i: _set(frame, i, col("unit_price"), "-15.00"),
        "unit_price.unparseable": lambda i: _set(
            frame, i, col("unit_price"), ("12..50", "abc")[i % 2]
        ),
        "unit_price.missing": lambda i: _set(frame, i, col("unit_price"), blank),
        "unit_cost.negative": lambda i: _set(frame, i, col("unit_cost"), "-3.00"),
        "discount_amount.exceeds_gross": lambda i: _set(
            frame, i, col("discount_amount"), f"{table.gross[i] * 1.5 + 10:.2f}"
        ),
        "currency.unsupported": lambda i: _set(frame, i, col("currency"), "XYZ"),
        "status.invalid": lambda i: _set(frame, i, col("status"), "pending_review"),
        "row.empty": lambda i: _blank_row(frame, i, blank),
    }
    duplicates: dict[int, DefectCode] = {}
    for pos, code in sorted(assignments.items()):
        if code in ("row.duplicate", "transaction_id.conflicting_duplicate"):
            duplicates[pos] = code
        else:
            mutators[code](pos)

    # Final row order: each duplicate goes right after its original.
    positions: list[int] = []
    out_codes: list[DefectCode | None] = []
    for pos in range(n_rows):
        positions.append(pos)
        out_codes.append(None if pos in duplicates else assignments.get(pos))
        if pos in duplicates:
            positions.append(pos)
            out_codes.append(duplicates[pos])
    dirty = frame.iloc[positions].reset_index(drop=True)
    qty_idx = _col_index(dirty, col("quantity"))
    for out_pos, out_code in enumerate(out_codes):
        if out_code == "transaction_id.conflicting_duplicate":
            dirty.iat[out_pos, qty_idx] = _bump_quantity(dirty.iat[out_pos, qty_idx])
    manifest_rows = [
        {"source_row": out_pos + 1, "codes": [out_code]}
        for out_pos, out_code in enumerate(out_codes)
        if out_code is not None
    ]
    return dirty, manifest_rows


def _col_index(frame: pd.DataFrame, column: str) -> int:
    return [str(c) for c in frame.columns].index(column)


def _set(frame: pd.DataFrame, pos: int, column: str, value: Any) -> None:
    frame.iat[pos, _col_index(frame, column)] = value


def _blank_row(frame: pd.DataFrame, pos: int, blank: Any) -> None:
    for j in range(frame.shape[1]):
        frame.iat[pos, j] = blank


def _bump_quantity(value: Any) -> Any:
    if isinstance(value, int | np.integer):
        return int(value) + 1
    return str(int(float(str(value))) + 1)


# ---------------------------------------------------------------------------
# deterministic writers
# ---------------------------------------------------------------------------
def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


_CORE_TIME = re.compile(rb"(<dcterms:(created|modified)[^>]*>)[^<]*(</dcterms:\2>)")


def write_xlsx(frame: pd.DataFrame, path: Path, *, sheet_name: str) -> None:
    """Write an .xlsx whose bytes depend only on the data (pinned zip + doc timestamps)."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name=sheet_name)
    source = zipfile.ZipFile(io.BytesIO(buffer.getvalue()))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "docProps/core.xml":
                data = _CORE_TIME.sub(rb"\g<1>" + FIXED_DOC_TIME + rb"\g<3>", data)
            pinned = zipfile.ZipInfo(info.filename, date_time=FIXED_ZIP_TIME)
            pinned.compress_type = zipfile.ZIP_DEFLATED
            pinned.external_attr = info.external_attr
            target.writestr(pinned, data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out.getvalue())


def hour_weights(
    profile: Mapping[int, float],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    hours = np.array(sorted(profile), dtype=np.int64)
    weights = np.array([profile[h] for h in hours], dtype=np.float64)
    return hours, weights / weights.sum()
