"""Cached, read-only data access for the dashboard.

Every loader takes ``(warehouse, stamp, dataset)``: *stamp* is the warehouse file's
modification time, so a pipeline run or a compaction (which swaps the file) invalidates
the cache automatically. Connections are opened per call and closed immediately; the
dashboard never holds the warehouse open. Heavy ML (forecast backtest, anomaly scoring)
always uses the full history, so it is cached once per dataset version.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from bizpulse.analytics.cohorts import CohortResult, cohort_analysis
from bizpulse.analytics.data import DatasetInfo, list_datasets, load_transactions
from bizpulse.analytics.rfm import rfm_table
from bizpulse.config import Settings
from bizpulse.ml.anomaly import AnomalyReport, detect_anomalies, granularity_for
from bizpulse.ml.forecasting import ForecastResult, daily_revenue, forecast_series
from bizpulse.ml.whatif import Baseline, compute_baseline
from bizpulse.warehouse.connection import connect
from bizpulse.warehouse.schema import META_RUNS


def warehouse_stamp(settings: Settings) -> float:
    """Modification time of the warehouse file (0.0 when there is none yet)."""
    path = settings.warehouse_path
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False, max_entries=16)
def datasets(warehouse: str, stamp: float) -> list[DatasetInfo]:
    return list_datasets(Path(warehouse))


@st.cache_data(show_spinner="Loading transactions ...", max_entries=6)
def transactions(warehouse: str, stamp: float, dataset: str) -> tuple[DatasetInfo, pd.DataFrame]:
    return load_transactions(Path(warehouse), dataset)


@st.cache_data(
    show_spinner="Training the forecast model (LightGBM, 6-fold time-series backtest) ...",
    max_entries=6,
)
def forecast(warehouse: str, stamp: float, dataset: str) -> ForecastResult:
    _, tx = transactions(warehouse, stamp, dataset)
    return forecast_series(daily_revenue(tx))


@st.cache_data(show_spinner="Scoring anomalies (z-score, IQR, IsolationForest) ...", max_entries=6)
def anomalies(warehouse: str, stamp: float, dataset: str) -> AnomalyReport:
    info, tx = transactions(warehouse, stamp, dataset)
    return detect_anomalies(tx, granularity_for(info.domain))


@st.cache_data(show_spinner="Segmenting customers ...", max_entries=6)
def rfm(warehouse: str, stamp: float, dataset: str) -> pd.DataFrame:
    _, tx = transactions(warehouse, stamp, dataset)
    return rfm_table(tx)


@st.cache_data(show_spinner="Building cohorts ...", max_entries=6)
def cohorts(warehouse: str, stamp: float, dataset: str) -> CohortResult:
    _, tx = transactions(warehouse, stamp, dataset)
    return cohort_analysis(tx)


@st.cache_data(show_spinner=False, max_entries=6)
def whatif_baseline(warehouse: str, stamp: float, dataset: str) -> Baseline:
    _, tx = transactions(warehouse, stamp, dataset)
    return compute_baseline(tx)


@st.cache_data(show_spinner=False, max_entries=8)
def runs(warehouse: str, stamp: float, limit: int = 50) -> pd.DataFrame:
    """Latest pipeline runs (audit trail) from ``meta_pipeline_runs``."""
    if not Path(warehouse).exists():
        return pd.DataFrame()
    with connect(Path(warehouse), read_only=True) as con:
        frame: pd.DataFrame = con.execute(
            f"""SELECT started_at, dataset, status, source_file, rows_read, rows_quarantined,
                       rows_inserted, rows_updated, rows_unchanged, pass_rate,
                       duration_seconds, report_path, run_id
                FROM {META_RUNS} ORDER BY started_at DESC LIMIT ?""",
            [limit],
        ).df()
    return frame


def clear() -> None:
    """Drop every cached result (after loading data from the dashboard)."""
    st.cache_data.clear()
