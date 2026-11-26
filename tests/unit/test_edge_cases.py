"""Edge cases from real uploads: missing optional columns, single slices, NaN in tables."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from bizpulse.analytics.data import NOT_PROVIDED, prepare
from bizpulse.analytics.kpis import breakdown, kpi_timeseries, pareto
from bizpulse.app import charts
from bizpulse.app.ingest import quarantine_frame
from bizpulse.app.palette import LIGHT
from bizpulse.app.views.overview import with_profit
from bizpulse.ml.anomaly import detect_anomalies
from tests.conftest import make_tx


def test_prepare_labels_missing_optional_dimensions() -> None:
    tx = make_tx({"product_id": "P1"}, {"product_id": "P2"})
    raw = tx.astype(object)
    raw.loc[:, ["category", "channel", "region"]] = None
    raw.loc[0, "product_name"] = None
    out = prepare(raw)
    assert set(out["category"]) == set(out["channel"]) == set(out["region"]) == {NOT_PROVIDED}
    assert out.loc[0, "product_name"] == "P1"  # falls back to the product id
    # Breakdowns and Pareto never mix missing values with names.
    assert list(breakdown(out, "channel").index) == [NOT_PROVIDED]
    assert pareto(out, by="category").items_total == 1


def test_single_valued_dimensions_are_not_monitored_as_slices() -> None:
    days = pd.date_range("2026-01-01", periods=120, freq="D")
    rows = [
        {"ts": f"{day:%Y-%m-%d} 12:00", "order_id": f"O{i}", "customer_id": f"C{i % 7}",
         "gross": 2000.0 if i == 110 else 100.0 + (i % 7), "region": "EU", "channel": "web"}
        for i, day in enumerate(days)
    ]  # fmt: skip
    report = detect_anomalies(make_tx(*rows), "D")
    assert report.alerts  # the spike is caught on the business total ...
    assert all(a.scope == "total" for a in report.alerts)  # ... and not duplicated per slice


def test_quarantine_frame_flattens_issues_and_raw_values() -> None:
    issues = [{"field": "unit_price", "code": "unit_price.unparseable", "message": "bad"}]
    raw = {"txn": "H1", "price": "inf"}
    table = quarantine_frame(
        [(7, "unit_price.unparseable", json.dumps(issues), json.dumps(raw)), (9, "x", "", "")]
    )
    assert list(table.columns[:3]) == ["source row", "reasons", "details"]
    assert table.loc[0, "details"] == "unit_price: bad"
    assert table.loc[0, "price"] == "inf"
    assert quarantine_frame([]).empty


def test_profit_trend_and_two_series_chart() -> None:
    tx = make_tx(
        {"ts": "2026-01-01", "gross": 100.0, "cost": 40.0, "fee": 5.0},
        {"ts": "2026-01-02", "gross": 50.0},  # no cost known: margin undefined that day
    )
    series = with_profit(kpi_timeseries(tx, "D"))
    assert series.loc[pd.Timestamp("2026-01-01"), "net_profit"] == 100.0 * 0.6 - 5.0
    assert np.isnan(series.loc[pd.Timestamp("2026-01-02"), "net_profit"])
    fig = charts.multi_trend(
        {"Net revenue": series["net_revenue"], "Net profit": series["net_profit"]}, LIGHT
    )
    assert [t.name for t in fig.data] == ["Net revenue", "Net profit"]
    assert [t.line.color for t in fig.data] == list(LIGHT.series[:2])


def test_pareto_figure_names_its_unit() -> None:
    table = pareto(make_tx({"category": "A"}, {"category": "B"}), by="category").table
    fig = charts.pareto_figure(table, 1, LIGHT, "categories")
    assert "categories" in fig.layout.annotations[0].text
    assert fig.layout.xaxis.title.text == "categories ranked by net revenue"
