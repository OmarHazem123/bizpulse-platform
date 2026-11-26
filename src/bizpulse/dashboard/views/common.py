"""Shared view context, formatting and empty/error states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
import streamlit as st

from bizpulse.analytics.data import DatasetInfo
from bizpulse.analytics.kpis import window
from bizpulse.config import Settings
from bizpulse.dashboard import state
from bizpulse.dashboard.palette import Palette
from bizpulse.pipeline import load_demo

Freq = Literal["D", "W", "M"]
FREQ_LABELS: dict[str, str] = {"D": "Day", "W": "Week", "M": "Month"}
CURRENCY_PREFIX: dict[str, str] = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}


@dataclass(frozen=True)
class Context:
    """Everything a tab needs: the selected dataset, filters and chart palette."""

    settings: Settings
    stamp: float
    info: DatasetInfo
    tx: pd.DataFrame  # full history (ML always uses this)
    start: pd.Timestamp  # selected range [start, end), applied to BI views
    end: pd.Timestamp
    freq: Freq
    pal: Palette

    @property
    def warehouse(self) -> str:
        return str(self.settings.warehouse_path)

    @property
    def key(self) -> tuple[str, float, str]:
        """Cache key for :mod:`bizpulse.dashboard.state` loaders."""
        return self.warehouse, self.stamp, self.info.dataset

    @property
    def prefix(self) -> str:
        return CURRENCY_PREFIX.get(self.info.currency, f"{self.info.currency} ")

    @property
    def tx_range(self) -> pd.DataFrame:
        return window(self.tx, self.start, self.end)

    @property
    def range_label(self) -> str:
        last = self.end - pd.Timedelta(1, unit="D")
        return f"{self.start:%Y-%m-%d} to {last:%Y-%m-%d}"

    def money(self, value: float | None, digits: int | None = None) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        places = digits if digits is not None else (2 if abs(value) < 1000 else 0)
        sign = "-" if value < 0 else ""
        return f"{sign}{self.prefix}{abs(value):,.{places}f}"


def pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:.{digits}%}"


def signed_pct(value: float | None, digits: int = 1) -> str | None:
    return None if value is None or pd.isna(value) else f"{value:+.{digits}%}"


def change(current: float | None, previous: float | None) -> float | None:
    """Relative change, or ``None`` when it is undefined."""
    if current is None or previous is None or pd.isna(current) or pd.isna(previous):
        return None
    return None if previous == 0 else (current - previous) / abs(previous)


def spark(values: pd.Series[float]) -> list[float]:
    """Sparkline points (gaps dropped)."""
    return [float(v) for v in values.dropna().to_numpy()]


def table_view(title: str, frame: pd.DataFrame, **kwargs: Any) -> None:
    """The accessible table behind a chart (every value readable without hovering)."""
    with st.expander(f"Table: {title}"):
        st.dataframe(frame, hide_index=kwargs.pop("hide_index", True), **kwargs)


def empty_state(settings: Settings, *, key: str) -> None:
    """Call-to-action when the warehouse has no data yet."""
    st.info(
        "**No data yet.** Load the three demo businesses (e-commerce, SaaS and a cafe "
        "chain; about 40 seconds), or upload your own CSV / Excel / Parquet file in the "
        "**Company Data Upload & Quality Audit** tab.",
        icon=":material/database:",
    )
    if st.button("Load demo data", key=f"demo-{key}", type="primary"):
        with st.spinner("Generating 18 months of demo data and running the pipelines ..."):
            load_demo(settings)
        state.clear()
        st.rerun()
    st.caption("CLI equivalent: `bizpulse generate-demo` then `bizpulse run-pipeline --all`.")


def failure(message: str, exc: Exception) -> None:
    """A friendly error instead of a stack trace."""
    st.warning(f"{message}: {exc}", icon=":material/warning:")
