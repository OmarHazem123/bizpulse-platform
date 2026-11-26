"""BizPulse Streamlit dashboard: ``bizpulse dashboard`` or ``tasks.py dashboard``.

The sidebar selects a dataset, a date range (applied to BI views) and a trend
granularity. Tabs render lazily: only the open tab runs, and heavy ML results are cached
per dataset version (see :mod:`bizpulse.dashboard.state`).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import duckdb
import pandas as pd
import streamlit as st

from bizpulse import __version__
from bizpulse.analytics.data import DatasetInfo
from bizpulse.config import Settings
from bizpulse.dashboard import state
from bizpulse.dashboard.palette import Palette, palette_for
from bizpulse.dashboard.views import customers, forecasts, overview, upload, whatif
from bizpulse.dashboard.views.common import FREQ_LABELS, Context, Freq, empty_state
from bizpulse.logging_utils import get_logger

log = get_logger(__name__)

OVERVIEW = "Executive Overview"
CUSTOMERS = "Customer & Cohort Intelligence"
FORECASTS = "ML Forecasts & Anomaly Alerts"
WHATIF = "What-If Growth Simulator"
UPLOAD = "Company Data Upload & Quality Audit"
TABS: tuple[str, ...] = (OVERVIEW, CUSTOMERS, FORECASTS, WHATIF, UPLOAD)
RENDERERS: dict[str, Callable[[Context], None]] = {
    OVERVIEW: overview.render,
    CUSTOMERS: customers.render,
    FORECASTS: forecasts.render,
    WHATIF: whatif.render,
}


def _theme_type() -> str | None:
    try:
        kind = st.context.theme.type
    except (AttributeError, KeyError, RuntimeError):
        return None
    return str(kind) if kind else None


def _sidebar(
    settings: Settings, stamp: float, infos: list[DatasetInfo], pal: Palette
) -> Context | None:
    with st.sidebar:
        st.title("BizPulse")
        st.caption(f"Decision intelligence platform | v{__version__}")
        if not infos:
            st.info("No datasets loaded yet.")
            return None
        by_name = {info.dataset: info for info in infos}
        dataset = st.selectbox(
            "Dataset",
            list(by_name),
            format_func=lambda name: by_name[name].display_name,
            key="dataset",
        )
        info, tx = state.transactions(str(settings.warehouse_path), stamp, dataset)
        if tx.empty:
            st.warning("This dataset has no transactions.")
            return None
        first = pd.Timestamp(tx["order_ts"].min()).date()
        last = pd.Timestamp(tx["order_ts"].max()).date()
        picked = st.date_input(
            "Date range",
            value=(first, last),
            min_value=first,
            max_value=last,
            key=f"range_{dataset}",
            help="Applies to KPIs, trends, Pareto, mix and alert lists. Customer segments, "
            "cohorts, forecasts, anomaly scoring and the simulator use the full history.",
        )
        bounds = list(picked) if isinstance(picked, tuple | list) else [picked]
        start = bounds[0] if len(bounds) == 2 and bounds[0] else first
        end = bounds[1] if len(bounds) == 2 and bounds[1] else last
        freq: Freq = st.radio(
            "Trend granularity",
            ["D", "W", "M"],
            index=1,
            format_func=lambda f: FREQ_LABELS[f],
            horizontal=True,
            key="freq",
        )
        st.caption(
            f"{info.domain} | {info.currency} | timezone {info.timezone}<br>"
            f"data {first} to {last} | {len(tx):,} transaction lines",
            unsafe_allow_html=True,
        )
        if st.button("Refresh data", key="refresh", icon=":material/refresh:"):
            state.clear()
            st.rerun()
    return Context(
        settings=settings,
        stamp=stamp,
        info=info,
        tx=tx,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end) + pd.Timedelta(1, unit="D"),
        freq=freq,
        pal=pal,
    )


def _render(name: str, action: Callable[[], None]) -> None:
    """Run one tab; show a friendly message instead of a stack trace on failure."""
    try:
        action()
    except Exception as exc:
        log.exception("dashboard tab %r failed", name)
        st.error(f"This view could not be rendered: {exc}", icon=":material/error:")


def main() -> None:
    st.set_page_config(page_title="BizPulse", page_icon=":material/monitoring:", layout="wide")
    settings = Settings.from_env()
    stamp = state.warehouse_stamp(settings)
    pal = palette_for(_theme_type())
    try:
        infos = state.datasets(str(settings.warehouse_path), stamp)
    except duckdb.Error as exc:
        st.error(f"The warehouse could not be opened: {exc}")
        infos = []
    ctx = _sidebar(settings, stamp, infos, pal)
    if ctx is not None:
        st.header(ctx.info.display_name)
    else:
        st.header("BizPulse decision intelligence")

    tabs = st.tabs(list(TABS), key="nav", on_change="rerun")
    for name, tab in zip(TABS, tabs, strict=True):
        with tab:
            if not tab.open:
                continue
            if name == UPLOAD:
                existing = [info.dataset for info in infos]
                _render(name, partial(upload.render, settings, stamp, existing, pal))
            elif ctx is None:
                empty_state(settings, key=name)
            else:
                _render(name, partial(RENDERERS[name], ctx))


if __name__ == "__main__":
    main()
