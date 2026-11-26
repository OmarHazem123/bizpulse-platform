"""Value normalization: amounts, currencies, dates, identifiers, text and statuses.

Every parser separates three outcomes per cell: *blank* (missing), *parsed*, and
*invalid* (present but unparseable). Invalid cells become :class:`QualityIssue` objects
with exact reason codes. Nothing is silently dropped or coerced to a default.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from bizpulse.ingestion.fx import FxTable
from bizpulse.ingestion.mapping import MappingConfig, ResolvedMapping
from bizpulse.schema import (
    AMOUNT_FIELDS,
    DEFAULT_CATEGORY,
    DEFAULT_DIMENSION,
    NORMALIZED_COLUMNS,
    CanonicalField,
    QualityIssue,
    Status,
)

_NBSP = chr(0x00A0)
_NARROW_NBSP = chr(0x202F)
BLANK_TOKENS = frozenset({"", "nan", "nat", "null", "none", "n/a", "#n/a", "-"})

#: Multi-character symbols first so ``US$`` wins over ``$``.
CURRENCY_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("US$", "USD"),
    ("C$", "CAD"),
    ("A$", "AUD"),
    ("R$", "BRL"),
    ("E£", "EGP"),
    ("$", "USD"),
    ("€", "EUR"),
    ("£", "GBP"),
    ("¥", "JPY"),
    ("₹", "INR"),
)
_SYMBOL_TO_CODE = dict(CURRENCY_SYMBOLS)
_ISO_TOKEN = re.compile(r"(?<![A-Za-z])([A-Za-z]{3})(?![A-Za-z])")
#: ISO-4217 codes recognised inside amount strings (e.g. ``USD 49.99``). Arbitrary
#: three-letter tokens are NOT stripped, so ``ELE-001`` stays unparseable.
KNOWN_CURRENCY_CODES = frozenset({
    "USD", "EUR", "GBP", "CAD", "AUD", "NZD", "CHF", "JPY", "CNY", "INR", "SGD", "HKD",
    "AED", "SAR", "EGP", "ZAR", "BRL", "MXN", "SEK", "NOK", "DKK", "PLN", "TRY", "KRW",
    "TWD", "THB", "MYR", "IDR", "PHP", "VND", "ILS", "QAR", "KWD", "BHD", "OMR", "JOD",
    "MAD", "NGN", "KES", "CZK", "HUF", "RON", "BGN", "ISK", "RUB", "UAH", "ARS", "CLP",
    "COP", "PEN",
})  # fmt: skip
_NUMBER = re.compile(r"\d+(?:\.\d+)?|\.\d+")
_DECIMAL_COMMA = re.compile(r"\d+,\d{1,2}")
_THOUSANDS_COMMA = re.compile(r"\d{1,3}(?:,\d{3})+")
_THOUSANDS_DOT = re.compile(r"\d{1,3}(?:\.\d{3})+")
_EXCEL_EPOCH = pd.Timestamp("1899-12-30")
_EXCEL_SERIAL = re.compile(r"\d{1,5}(?:\.\d+)?")
_YYYYMMDD = re.compile(r"(19|20)\d{6}")
_TZ_SUFFIX = re.compile(r"(?:Z|UTC|GMT|[+-]\d{2}:?\d{2})$", re.IGNORECASE)

STATUS_SYNONYMS: dict[str, Status] = {
    **dict.fromkeys(
        (
            "paid", "complete", "completed", "fulfilled", "closed", "success", "succeeded",
            "settled", "captured", "delivered", "shipped", "ok", "sale", "done",
        ),
        "completed",
    ),
    **dict.fromkeys(
        ("refund", "refunded", "returned", "return", "chargeback", "reversed", "credited"),
        "refunded",
    ),
    **dict.fromkeys(
        (
            "cancelled", "canceled", "cancel", "void", "voided", "failed", "declined",
            "abandoned", "rejected",
        ),
        "cancelled",
    ),
}  # fmt: skip


# ---------------------------------------------------------------------------
# scalar helpers
# ---------------------------------------------------------------------------
def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in BLANK_TOKENS
    if isinstance(value, float | np.floating):
        return math.isnan(float(value))
    return value is pd.NaT


def blank_mask(series: pd.Series[Any]) -> pd.Series[bool]:
    return series.map(is_blank).astype(bool)


def raw_text(value: Any) -> str | None:
    """Printable form of a raw cell for reports (``None`` when blank)."""
    if is_blank(value):
        return None
    return str(value)[:200]


@dataclass(frozen=True)
class AmountParse:
    value: float | None
    currency_hint: str | None = None
    invalid: bool = False


_INVALID_AMOUNT = AmountParse(None, None, True)


def parse_amount(raw: Any) -> AmountParse:
    """Parse one monetary cell (see DESIGN.md §4 for the separator rules)."""
    if is_blank(raw):
        return AmountParse(None)
    if isinstance(raw, bool):
        return _INVALID_AMOUNT
    if isinstance(raw, int | float | np.integer | np.floating):
        number = float(raw)
        return AmountParse(number) if math.isfinite(number) else _INVALID_AMOUNT
    text = str(raw).strip()
    hint: str | None = None
    for symbol, code in CURRENCY_SYMBOLS:
        if symbol in text:
            hint, text = code, text.replace(symbol, "")
            break
    iso = _ISO_TOKEN.search(text)
    if iso and iso.group(1).upper() in KNOWN_CURRENCY_CODES:
        hint = hint or iso.group(1).upper()
        text = text[: iso.start()] + text[iso.end() :]
    for junk in (_NBSP, _NARROW_NBSP, " ", "'"):
        text = text.replace(junk, "")
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative, text = True, text[1:-1]
    if text.startswith("-"):
        negative, text = not negative, text[1:]
    elif text.startswith("+"):
        text = text[1:]
    if text.endswith("-"):
        negative, text = True, text[:-1]
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        if _DECIMAL_COMMA.fullmatch(text):
            text = text.replace(",", ".")
        elif _THOUSANDS_COMMA.fullmatch(text):
            text = text.replace(",", "")
        else:
            return _INVALID_AMOUNT
    elif text.count(".") > 1:
        if not _THOUSANDS_DOT.fullmatch(text):
            return _INVALID_AMOUNT
        text = text.replace(".", "")
    if not _NUMBER.fullmatch(text):
        return _INVALID_AMOUNT
    number = float(text)
    return AmountParse(-number if negative else number, hint)


def normalize_currency_code(raw: Any) -> str | None:
    """Return an upper-case 3-letter code, ``""`` when invalid, ``None`` when blank."""
    if is_blank(raw):
        return None
    text = str(raw).strip()
    if text in _SYMBOL_TO_CODE:
        return _SYMBOL_TO_CODE[text]
    upper = text.upper()
    return upper if re.fullmatch(r"[A-Z]{3}", upper) else ""


def normalize_id(raw: Any) -> str | None:
    if is_blank(raw):
        return None
    if isinstance(raw, bool):
        return str(raw)
    if isinstance(raw, int | np.integer):
        return str(int(raw))
    if isinstance(raw, float | np.floating):
        number = float(raw)
        return str(int(number)) if number.is_integer() else repr(number)
    text = re.sub(r"\s+", " ", str(raw).strip())
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text or None


def normalize_text(raw: Any) -> str | None:
    if is_blank(raw):
        return None
    text = re.sub(r"\s+", " ", str(raw).strip())
    return text or None


def normalize_channel(raw: Any) -> str | None:
    text = normalize_text(raw)
    if text is None:
        return None
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or None


def status_key(text: str) -> str:
    return re.sub(r"[\s\-]+", "_", text.strip().lower())


# ---------------------------------------------------------------------------
# vectorized parsers
# ---------------------------------------------------------------------------
@dataclass
class AmountColumn:
    values: pd.Series[float]
    hints: pd.Series[Any]
    invalid: pd.Series[bool]


def parse_amounts(series: pd.Series[Any]) -> AmountColumn:
    blank = blank_mask(series)
    fast = pd.to_numeric(series.where(~blank), errors="coerce").astype("float64")
    ok = fast.notna() & np.isfinite(fast)
    values = fast.where(ok)
    hints = pd.Series([None] * len(series), index=series.index, dtype=object)
    invalid = pd.Series(False, index=series.index)
    slow_idx = series.index[~blank & ~ok]
    for idx in slow_idx:
        parsed = parse_amount(series.at[idx])
        if parsed.invalid:
            invalid.at[idx] = True
        else:
            values.at[idx] = parsed.value
            hints.at[idx] = parsed.currency_hint
    return AmountColumn(values=values, hints=hints, invalid=invalid)


@dataclass
class DateColumn:
    values: pd.Series[Any]  # datetime64[ns], naive UTC
    invalid: pd.Series[bool]
    methods: Counter[str] = field(default_factory=Counter)


def _to_naive_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


def _from_number(number: float) -> pd.Timestamp | None:
    if 1 <= number < 80_000:
        return _EXCEL_EPOCH + pd.to_timedelta(number, unit="D")
    return None


def parse_dates(
    series: pd.Series[Any], *, dayfirst: bool = False, date_format: str | None = None
) -> DateColumn:
    """Parse a column of dates into naive UTC timestamps (see DESIGN.md §4)."""
    blank = blank_mask(series)
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    invalid = pd.Series(False, index=series.index)
    methods: Counter[str] = Counter()
    text_idx: list[Any] = []
    for idx in series.index[~blank]:
        value = series.at[idx]
        parsed: pd.Timestamp | None = None
        method = ""
        if isinstance(value, datetime | date | np.datetime64):
            parsed, method = _to_naive_utc(pd.Timestamp(value)), "native"
        elif isinstance(value, int | float | np.integer | np.floating) and not isinstance(
            value, bool
        ):
            parsed, method = _from_number(float(value)), "excel_serial"
        elif isinstance(value, str):
            stripped = value.strip()
            if _YYYYMMDD.fullmatch(stripped):
                maybe = pd.to_datetime(stripped, format="%Y%m%d", errors="coerce")
                parsed, method = (None if pd.isna(maybe) else maybe), "yyyymmdd"
            elif _EXCEL_SERIAL.fullmatch(stripped):
                parsed, method = _from_number(float(stripped)), "excel_serial"
            else:
                text_idx.append(idx)
                continue
        if parsed is None or pd.isna(parsed):
            invalid.at[idx] = True
        else:
            out.at[idx] = parsed
            methods[method] += 1
    if text_idx:
        texts = series.loc[text_idx].astype(str).str.strip()
        fmt = date_format or "ISO8601"
        first = _parse_text_dates(texts, fmt=fmt, dayfirst=dayfirst)
        done = first.notna()
        out.loc[first.index[done]] = first[done].to_numpy()
        methods["explicit_format" if date_format else "iso8601"] += int(done.sum())
        rest = texts[~done]
        if len(rest) and not date_format:
            second = _parse_text_dates(rest, fmt="mixed", dayfirst=dayfirst)
            ok = second.notna()
            out.loc[second.index[ok]] = second[ok].to_numpy()
            methods["mixed"] += int(ok.sum())
            invalid.loc[second.index[~ok]] = True
        elif len(rest):
            invalid.loc[rest.index] = True
    return DateColumn(values=out, invalid=invalid, methods=methods)


def _parse_text_dates(texts: pd.Series[str], *, fmt: str, dayfirst: bool) -> pd.Series[Any]:
    """Parse strings to naive UTC, parsing offset-aware and naive strings *separately*.

    pandas' vectorized ISO8601 parser can carry a UTC offset seen on one element over to
    later naive elements of the same array, shifting them by hours. Splitting the groups
    makes each parse homogeneous.
    """
    aware = texts.str.contains(_TZ_SUFFIX, regex=True)
    parts: list[pd.Series[Any]] = []
    for mask in (aware, ~aware):
        group = texts[mask]
        if group.empty:
            continue
        parsed = pd.to_datetime(group, format=fmt, dayfirst=dayfirst, errors="coerce", utc=True)
        parts.append(parsed.dt.tz_localize(None))
    if not parts:
        return pd.Series(pd.NaT, index=texts.index, dtype="datetime64[ns]")
    combined: pd.Series[Any] = pd.concat(parts)
    return combined.reindex(texts.index)


def _numeric_like(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float | np.integer | np.floating):
        return True
    return isinstance(value, str) and _EXCEL_SERIAL.fullmatch(value.strip()) is not None


def date_parse_ratio(series: pd.Series[Any]) -> float:
    """Share of non-blank values that parse as dates (content-based detection).

    Numeric-looking values never count as dates here: a ``Qty`` column of small integers
    would otherwise parse as Excel serial dates and be mistaken for ``order_date``.
    """
    non_blank = series[~blank_mask(series)]
    if non_blank.empty:
        return 0.0
    candidates = non_blank[~non_blank.map(_numeric_like).astype(bool)]
    if candidates.empty:
        return 0.0
    parsed = parse_dates(candidates)
    return float((~parsed.invalid).sum()) / len(non_blank)


# ---------------------------------------------------------------------------
# frame normalization
# ---------------------------------------------------------------------------
@dataclass
class NormalizationStats:
    date_methods: dict[str, int]
    currency_sources: dict[str, int]
    amount_symbol_cells: int
    completeness: dict[str, float]


@dataclass
class NormalizedBatch:
    """Canonical frame (positional index 0..n-1) plus per-row normalization issues."""

    frame: pd.DataFrame
    issues: list[list[QualityIssue]]
    empty_rows: pd.Series[bool]
    stats: NormalizationStats


def _issue_rows(
    issues: list[list[QualityIssue]],
    mask: pd.Series[bool],
    raw: pd.Series[Any] | None,
    field_name: str,
    code: str,
    message: str,
) -> None:
    for pos in np.flatnonzero(mask.to_numpy()).tolist():
        value = raw_text(raw.iat[pos]) if raw is not None else None
        issues[pos].append(QualityIssue(field_name, code, message, value))


def normalize_frame(
    frame: pd.DataFrame, mapping: ResolvedMapping, config: MappingConfig, fx: FxTable
) -> NormalizedBatch:
    """Map and normalize a raw frame into :data:`NORMALIZED_COLUMNS`."""
    raw = frame.reset_index(drop=True)
    n = len(raw)
    issues: list[list[QualityIssue]] = [[] for _ in range(n)]
    empty_rows = (
        raw.apply(lambda col: blank_mask(col)).all(axis=1).astype(bool)
        if raw.shape[1]
        else pd.Series(True, index=raw.index)
    )
    out = pd.DataFrame(index=raw.index)

    def source(name: CanonicalField) -> pd.Series[Any] | None:
        match = mapping.source_of(name)
        if match is None:
            return None
        if len(match.sources) == 1:
            return raw[match.sources[0]]
        parts = [raw[s].map(normalize_id) for s in match.sources]
        joined = parts[0].astype(object)
        for part in parts[1:]:
            joined = joined.combine(
                part, lambda a, b: None if a is None or b is None else f"{a}-{b}"
            )
        return joined

    # identifiers -----------------------------------------------------------
    id_fields: tuple[CanonicalField, ...] = ("transaction_id", "customer_id", "product_id")
    for name in id_fields:
        col = source(name)
        out[name] = col.map(normalize_id) if col is not None else None
    order_col = source("order_id")
    out["order_id"] = order_col.map(normalize_id) if order_col is not None else None
    out["order_id"] = out["order_id"].where(out["order_id"].notna(), out["transaction_id"])

    # dates -----------------------------------------------------------------
    date_col = source("order_date")
    date_methods: Counter[str] = Counter()
    if date_col is not None:
        dates = parse_dates(date_col, dayfirst=config.dayfirst, date_format=config.date_format)
        out["order_date"] = dates.values
        date_methods = dates.methods
        _issue_rows(
            issues, dates.invalid, date_col, "order_date", "order_date.unparseable",
            "value is not a recognizable date",
        )  # fmt: skip
    else:
        out["order_date"] = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

    # text dimensions -------------------------------------------------------
    name_col, cat_col = source("product_name"), source("category")
    chan_col, region_col = source("channel"), source("region")
    out["product_name"] = name_col.map(normalize_text) if name_col is not None else None
    out["product_name"] = out["product_name"].where(out["product_name"].notna(), out["product_id"])
    out["category"] = cat_col.map(normalize_text) if cat_col is not None else None
    out["category"] = out["category"].fillna(DEFAULT_CATEGORY)
    out["channel"] = chan_col.map(normalize_channel) if chan_col is not None else None
    out["channel"] = out["channel"].fillna(DEFAULT_DIMENSION)
    out["region"] = region_col.map(normalize_text) if region_col is not None else None
    out["region"] = out["region"].fillna(DEFAULT_DIMENSION)

    # amounts ---------------------------------------------------------------
    parsed: dict[str, AmountColumn] = {}
    symbol_cells = 0
    for name in AMOUNT_FIELDS:
        col = source(name)
        if col is None:
            continue
        amounts = parse_amounts(col)
        parsed[name] = amounts
        symbol_cells += int(amounts.hints.notna().sum())
        _issue_rows(
            issues, amounts.invalid, col, name, f"{name}.unparseable",
            "value is not a recognizable number",
        )  # fmt: skip

    qty_col = source("quantity")
    if qty_col is not None:
        qty = parse_amounts(qty_col)
        _issue_rows(
            issues, qty.invalid, qty_col, "quantity", "quantity.unparseable",
            "value is not a recognizable number",
        )  # fmt: skip
        quantity = qty.values
    else:
        quantity = pd.Series(np.nan, index=raw.index, dtype="float64")

    if "unit_price" in parsed:
        unit_price = parsed["unit_price"].values
    elif "line_total" in parsed:
        with np.errstate(divide="ignore", invalid="ignore"):
            unit_price = parsed["line_total"].values / quantity.where(quantity != 0)
    else:
        unit_price = pd.Series(np.nan, index=raw.index, dtype="float64")

    # status ------------------------------------------------------------------
    status_col = source("status")
    synonyms = {**STATUS_SYNONYMS, **{status_key(k): v for k, v in config.status_values.items()}}
    if status_col is not None:
        status_blank = blank_mask(status_col)
        keys = status_col.map(lambda v: None if is_blank(v) else status_key(str(v)))
        status = keys.map(lambda k: None if k is None else synonyms.get(k)).astype(object)
        bad_status = ~status_blank & status.isna()
        _issue_rows(
            issues, bad_status, status_col, "status", "status.invalid",
            f"unknown status; expected one of {sorted(set(synonyms))[:12]}...",
        )  # fmt: skip
        status = status.where(~status_blank, "completed")
    else:
        status = pd.Series(["completed"] * n, index=raw.index, dtype=object)

    if config.negative_quantity_as_refund:
        negative = quantity < 0
        quantity = quantity.abs().where(negative, quantity)
        status = status.where(~negative, "refunded")

    out["quantity"] = quantity
    out["unit_price"] = unit_price
    out["discount_amount"] = (
        parsed["discount_amount"].values.fillna(0.0) if "discount_amount" in parsed else 0.0
    )
    out["unit_cost"] = parsed["unit_cost"].values if "unit_cost" in parsed else np.nan
    out["fee_amount"] = parsed["fee_amount"].values.fillna(0.0) if "fee_amount" in parsed else 0.0
    # Keep invalid optional amounts as NaN-free zeros; their issue already quarantines the row.
    for name in ("discount_amount", "fee_amount"):
        if name in parsed:
            out[name] = out[name].where(~parsed[name].invalid, 0.0)

    # currency ----------------------------------------------------------------
    currency_sources: Counter[str] = Counter()
    hint = pd.Series([None] * n, index=raw.index, dtype=object)
    for name in ("unit_price", "line_total"):
        if name in parsed:
            hint = hint.where(hint.notna(), parsed[name].hints)
    cur_col = source("currency")
    if cur_col is not None:
        codes = cur_col.map(normalize_currency_code).astype(object)
    else:
        codes = pd.Series([None] * n, index=raw.index, dtype=object)
    from_column = codes.notna()
    from_hint = ~from_column & hint.notna()
    currency = codes.where(from_column, hint).fillna(config.effective_default_currency)
    currency_sources["column"] = int(from_column.sum())
    currency_sources["amount_symbol"] = int(from_hint.sum())
    currency_sources["dataset_default"] = int((~from_column & ~from_hint).sum())
    unsupported = currency.map(lambda c: not fx.supports(str(c))).astype(bool)
    _issue_rows(
        issues, unsupported, cur_col if cur_col is not None else currency, "currency",
        "currency.unsupported", "currency is not in the FX reference table",
    )  # fmt: skip
    out["currency"] = currency.mask(unsupported)
    out["status"] = status

    # row-level: empty rows carry exactly one issue -----------------------------
    for pos in np.flatnonzero(empty_rows.to_numpy()).tolist():
        issues[pos] = [QualityIssue("row", "row.empty", "every field in the row is blank")]

    non_empty = int((~empty_rows).sum())
    completeness: dict[str, float] = {}
    if non_empty:
        for column in NORMALIZED_COLUMNS:
            filled = out[column][~empty_rows].notna().mean()
            completeness[column] = round(float(filled), 4)

    stats = NormalizationStats(
        date_methods=dict(date_methods),
        currency_sources=dict(currency_sources),
        amount_symbol_cells=symbol_cells,
        completeness=completeness,
    )
    return NormalizedBatch(
        frame=out.loc[:, list(NORMALIZED_COLUMNS)],
        issues=issues,
        empty_rows=empty_rows,
        stats=stats,
    )
