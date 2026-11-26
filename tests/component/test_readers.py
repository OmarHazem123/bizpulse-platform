from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bizpulse.ingestion.readers import ReadError, read_table, sniff_delimiter


@pytest.mark.parametrize("sep", [",", ";", "\t", "|"])
def test_csv_delimiter_sniffing(tmp_path: Path, sep: str) -> None:
    path = tmp_path / "x.csv"
    path.write_text(f"a{sep}b{sep}c\n1{sep}2{sep}3\n", encoding="utf-8")
    table = read_table(path)
    assert table.delimiter == sep
    assert table.frame.to_dict("records") == [{"a": "1", "b": "2", "c": "3"}]


def test_csv_quoted_header_with_commas() -> None:
    assert sniff_delimiter('"Amount, USD";"Qty";"Date"') == ";"


def test_csv_keeps_raw_strings_and_strips_headers(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_bytes(b"\xef\xbb\xbf Order Date ,Qty\n2026-01-01,007\n,\n")
    table = read_table(path)
    assert list(table.frame.columns) == ["Order Date", "Qty"]
    assert table.frame["Qty"].tolist() == ["007", ""]
    assert table.encoding == "utf-8-sig"


def test_csv_latin1_fallback(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_bytes("name,city\nJos\xe9,M\xfcnchen\n".encode("latin-1"))
    table = read_table(path)
    assert table.frame["name"].iloc[0] == "José"


def test_excel_and_parquet_roundtrip(tmp_path: Path) -> None:
    frame = pd.DataFrame({"id": [1, 2], "when": pd.to_datetime(["2026-01-01", "2026-01-02"])})
    xlsx, pq = tmp_path / "x.xlsx", tmp_path / "x.parquet"
    frame.to_excel(xlsx, index=False)
    frame.to_parquet(pq, index=False)
    excel = read_table(xlsx)
    parquet = read_table(pq)
    assert excel.source_format == "excel" and parquet.source_format == "parquet"
    assert excel.frame["id"].tolist() == [1, 2]
    assert str(parquet.frame["when"].iloc[1]) == "2026-01-02 00:00:00"


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("x.json", "{}", "unsupported file type"),
        ("x.csv", "", "empty"),
        ("x.csv", "a,a\n1,2\n", "duplicate column"),
    ],
)
def test_read_errors(tmp_path: Path, name: str, content: str, message: str) -> None:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ReadError, match=message):
        read_table(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ReadError, match="not found"):
        read_table(tmp_path / "nope.csv")
