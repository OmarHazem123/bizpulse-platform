"""Pre-registered Phase 2 acceptance criteria on the FULL-scale demo data (TESTING-GUIDE §6)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from bizpulse.analytics.cohorts import cohort_analysis, ltv_at
from bizpulse.analytics.data import DatasetInfo, load_transactions
from bizpulse.analytics.rfm import rfm_table
from bizpulse.analytics.timeofday import peak_hours
from bizpulse.config import Settings
from bizpulse.demo.generator import generate_demo_data
from bizpulse.ml.anomaly import AnomalyReport, detect_anomalies
from bizpulse.ml.forecasting import ForecastResult, daily_revenue, forecast_series
from bizpulse.ml.whatif import Levers, compute_baseline, project
from bizpulse.pipeline import run_demo_pipelines

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_STATE: dict[str, Settings] = {}


@pytest.fixture(scope="module", autouse=True)
def full_scale(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Settings]:
    settings = Settings.from_env(tmp_path_factory.mktemp("full") / "data")
    settings.ensure_dirs()
    generate_demo_data(settings.raw_dir, seed=42, scale=1.0)
    run_demo_pipelines(settings)
    _STATE["settings"] = settings
    yield settings
    _STATE.clear()
    _load.cache_clear()
    _forecast.cache_clear()
    _anomalies.cache_clear()


@cache
def _load(dataset: str) -> tuple[DatasetInfo, pd.DataFrame]:
    return load_transactions(_STATE["settings"].warehouse_path, dataset)


@cache
def _forecast(dataset: str) -> ForecastResult:
    return forecast_series(daily_revenue(_load(dataset)[1]))


@cache
def _anomalies(dataset: str) -> AnomalyReport:
    return detect_anomalies(_load(dataset)[1])


def _manifest(dataset: str) -> dict[str, Any]:
    path = _STATE["settings"].manifests_dir / f"{dataset}.json"
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


# --------------------------------------------------------------------------- forecasting
@pytest.mark.parametrize("dataset", ["ecommerce", "retail_fb"])
def test_model_beats_seasonal_naive(dataset: str) -> None:
    ratio = _forecast(dataset).backtest.mae_ratio
    assert ratio is not None and ratio <= 1.1


@pytest.mark.parametrize("dataset", ["ecommerce", "retail_fb", "saas"])
def test_interval_coverage_and_order(dataset: str) -> None:
    result = _forecast(dataset)
    assert result.backtest.coverage80 is not None
    assert abs(result.backtest.coverage80 - 0.80) <= 0.15
    fc = result.forecast
    assert ((fc["lo95"] <= fc["lo80"]) & (fc["lo80"] <= fc["yhat"])).all()
    assert ((fc["yhat"] <= fc["hi80"]) & (fc["hi80"] <= fc["hi95"])).all()
    assert len(result.backtest.folds) == 6


# --------------------------------------------------------------------------- anomalies
DAILY_ANOMALIES = [
    ("ecommerce", "checkout_outage"),
    ("ecommerce", "social_viral_spike"),
    ("ecommerce", "apparel_refund_spike"),
    ("saas", "enterprise_deal"),
    pytest.param(
        "retail_fb",
        "riverside_closure",
        marks=pytest.mark.xfail(
            strict=True,
            reason="known limitation: one of three stores closing moves the TOTAL only "
            "~-30% (z ~ -2); needs per-segment monitoring (TECHNICAL-REQUIREMENTS §7.1)",
        ),
    ),
    ("retail_fb", "pos_outage"),
    ("retail_fb", "bogo_cold_drinks"),
]


def _window_alerts(dataset: str, name: str) -> tuple[dict[str, Any], list[Any]]:
    anomaly = next(a for a in _manifest(dataset)["anomalies"] if a["name"] == name)
    alerts = _anomalies(dataset).alerts_between(
        pd.Timestamp(anomaly["start"]), pd.Timestamp(anomaly["end"]), anomaly["metric"]
    )
    return anomaly, alerts


@pytest.mark.parametrize(("dataset", "name"), DAILY_ANOMALIES)
def test_daily_anomaly_recall_and_root_cause(dataset: str, name: str) -> None:
    anomaly, alerts = _window_alerts(dataset, name)
    assert anomaly["granularity"] == "day"
    assert alerts, f"no {anomaly['metric']} alert inside {anomaly['start']}..{anomaly['end']}"
    for dimension, expected in (anomaly["slice"] or {}).items():
        top = alerts[0].top_cause(dimension)
        assert top is not None and top.value.lower() == expected.lower(), dimension


def test_checkout_outage_is_critical() -> None:
    _, alerts = _window_alerts("ecommerce", "checkout_outage")
    assert alerts and all(a.severity == "CRITICAL" for a in alerts)
    assert len(alerts) == 3  # one per outage day


def test_month_granularity_anomalies_are_documented() -> None:
    monthly = [a for a in _manifest("saas")["anomalies"] if a["granularity"] == "month"]
    assert [a["name"] for a in monthly] == ["starter_churn_spike"]


# --------------------------------------------------------------------------- analytics
def test_time_of_day_uses_business_timezone() -> None:
    info, tx = _load("retail_fb")
    assert info.timezone == "Europe/Berlin"
    local = peak_hours(tx, info.timezone, top=3)
    assert {8, 13} <= set(local)  # café breakfast and lunch peaks, local time
    assert peak_hours(tx, "UTC", top=3) != local


@pytest.mark.parametrize("dataset", ["ecommerce", "saas", "retail_fb"])
def test_whatif_baseline_matches_trailing_actuals(dataset: str) -> None:
    baseline = compute_baseline(_load(dataset)[1])
    projected = float(project(baseline, Levers())["net_revenue"].iloc[0])
    assert projected == pytest.approx(baseline.actual_net_revenue, rel=0.15)


@pytest.mark.parametrize("dataset", ["ecommerce", "saas", "retail_fb"])
def test_segments_and_ltv_are_sane(dataset: str) -> None:
    _, tx = _load(dataset)
    rfm = rfm_table(tx)
    assert rfm["segment"].nunique() >= 5
    assert not rfm["customer_id"].isin(["GUEST"]).any()
    values = [v for v in ltv_at(cohort_analysis(tx)).values() if v is not None]
    assert values == sorted(values)  # cumulative LTV never decreases
