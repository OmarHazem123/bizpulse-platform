"""Multi-format readers producing an untyped (object) frame.

Values are deliberately *not* coerced here: CSV/Parquet cells arrive as strings and Excel
cells as their native Python values, so the data contract can see the raw input and give
an exact reason when it is invalid.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

SourceFormat = Literal["csv", "excel", "parquet"]

SUPPORTED_EXTENSIONS: dict[str, SourceFormat] = {
    ".csv": "csv",
    ".txt": "csv",
    ".tsv": "csv",
    ".xlsx": "excel",
    ".xlsm": "excel",
    ".parquet": "parquet",
    ".pq": "parquet",
}
_DELIMITERS = (",", ";", "\t", "|")
_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
_BOM = chr(0xFEFF)


class ReadError(ValueError):
    """The file cannot be read as a table."""


@dataclass(frozen=True)
class RawTable:
    frame: pd.DataFrame
    source_format: SourceFormat
    file_name: str
    encoding: str | None = None
    delimiter: str | None = None
    sheet_name: str | None = None


def detect_format(path: Path) -> SourceFormat:
    if path.suffix.lower() == ".xls":
        raise ReadError("legacy .xls workbooks are not supported: save the file as .xlsx")
    fmt = SUPPORTED_EXTENSIONS.get(path.suffix.lower())
    if fmt is None:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ReadError(f"unsupported file type {path.suffix!r}; supported: {supported}")
    return fmt


def _decode(data: bytes, encoding: str | None) -> tuple[str, str]:
    candidates = (encoding,) if encoding else _ENCODINGS
    for candidate in candidates:
        try:
            return data.decode(candidate), candidate
        except (UnicodeDecodeError, LookupError):
            continue
    raise ReadError(f"could not decode file with encodings {candidates}")


def sniff_delimiter(header_line: str) -> str:
    """Pick the delimiter that splits the header into the most fields (quote-aware)."""
    best, best_count = ",", 1
    for delimiter in _DELIMITERS:
        fields = next(csv.reader([header_line], delimiter=delimiter), [])
        if len(fields) > best_count:
            best, best_count = delimiter, len(fields)
    return best


def _clean_headers(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = pd.Index([str(c).strip() for c in frame.columns])
    if frame.columns.duplicated().any():
        dupes = sorted({str(c) for c in frame.columns[frame.columns.duplicated()]})
        raise ReadError(f"duplicate column headers: {dupes}")
    return frame


def read_csv(path: Path, *, delimiter: str | None = None, encoding: str | None = None) -> RawTable:
    text, used_encoding = _decode(path.read_bytes(), encoding)
    if not text.strip():
        raise ReadError("file is empty")
    header_line = text.lstrip(_BOM).splitlines()[0]
    sep = delimiter or sniff_delimiter(header_line)
    # pandas silently renames duplicate headers (a, a.1): detect them on the raw header.
    raw_headers = [h.strip() for h in next(csv.reader([header_line], delimiter=sep), [])]
    dupes = sorted({h for h in raw_headers if raw_headers.count(h) > 1})
    if dupes:
        raise ReadError(f"duplicate column headers: {dupes}")
    frame = pd.read_csv(
        io.StringIO(text),
        sep=sep,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        skip_blank_lines=True,
    )
    return RawTable(
        frame=_clean_headers(frame.astype(object)),
        source_format="csv",
        file_name=path.name,
        encoding=used_encoding,
        delimiter=sep,
    )


def read_excel(path: Path, *, sheet_name: str | int | None = None) -> RawTable:
    target: str | int = 0 if sheet_name is None else sheet_name
    try:
        frame = pd.read_excel(path, sheet_name=target, dtype=object, engine="openpyxl")
    except ValueError as exc:
        raise ReadError(f"cannot read sheet {target!r}: {exc}") from exc
    sheet = str(target) if isinstance(target, str) else None
    return RawTable(
        frame=_clean_headers(frame.astype(object)),
        source_format="excel",
        file_name=path.name,
        sheet_name=sheet,
    )


def read_parquet(path: Path) -> RawTable:
    frame = pd.read_parquet(path)
    return RawTable(
        frame=_clean_headers(frame.astype(object)), source_format="parquet", file_name=path.name
    )


def read_table(
    path: Path,
    *,
    delimiter: str | None = None,
    encoding: str | None = None,
    sheet_name: str | int | None = None,
) -> RawTable:
    """Read any supported file into a :class:`RawTable`."""
    if not path.is_file():
        raise ReadError(f"file not found: {path.name}")
    fmt = detect_format(path)
    try:
        if fmt == "csv":
            table = read_csv(path, delimiter=delimiter, encoding=encoding)
        elif fmt == "excel":
            table = read_excel(path, sheet_name=sheet_name)
        else:
            table = read_parquet(path)
    except ReadError:
        raise
    except Exception as exc:  # pragma: no cover - library-specific parse failures
        raise ReadError(f"failed to read {path.name}: {exc}") from exc
    if table.frame.shape[1] == 0:
        raise ReadError("file has no columns")
    return table
