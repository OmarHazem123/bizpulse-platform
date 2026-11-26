"""KPI formulas against a hand-computed fixture (TECHNICAL-REQUIREMENTS.md §6)."""

from __future__ import annotations

import pandas as pd
import pytest

from bizpulse.analytics.kpis import (
    breakdown,
    compute_kpis,
    kpi_timeseries,
    latest_growth,
    pareto,
)
from tests.conftest import make_tx

NAN = float("nan")


@pytest.fixture
def tx() -> pd.DataFrame:
    return make_tx(
        {"ts": "2026-01-05 09:00", "order_id": "O1", "customer_id": "C1", "gross": 100,
         "discount": 10, "cost": 40, "fee": 3},
        {"ts": "2026-01-05 09:00", "order_id": "O1", "customer_id": "C1", "product_id": "P2",
         "gross": 50, "cost": NAN, "fee": 1},
        {"ts": "2026-01-10 12:00", "order_id": "O2", "customer_id": "C2", "status": "refunded",
         "gross": 80, "cost": 30, "fee": 2},
        {"ts": "2026-02-03 15:00", "order_id": "O3", "customer_id": "C1", "status": "cancelled",
         "gross": 60, "cost": 20, "fee": 2},
        {"ts": "2026-02-07 18:00", "order_id": "O4", "customer_id": "GUEST", "is_anonymous": True,
         "product_id": "P3", "gross": 20, "cost": 5, "fee": 0.5},
    )  # fmt: skip


def test_kpis_match_hand_computation(tx: pd.DataFrame) -> None:
    k = compute_kpis(tx)
    assert k.gross_revenue == 250  # cancelled excluded
    assert k.discounts == 10
    assert k.refunds == 80
    assert k.net_revenue == 160 == k.gross_revenue - k.discounts - k.refunds
    assert k.cogs_known == 45
    assert k.cost_coverage == pytest.approx(110 / 160)
    assert k.gross_margin_pct == pytest.approx(65 / 110)
    assert k.cogs_estimated == pytest.approx(160 * (1 - 65 / 110))
    assert k.gross_profit == pytest.approx(160 * 65 / 110)
    assert k.fees == 6.5
    assert k.net_profit == pytest.approx(160 * 65 / 110 - 6.5)
    assert k.orders == 3
    assert k.aov == pytest.approx(240 / 3)
    assert k.refund_rate == pytest.approx(80 / 240)
    assert k.refund_rate_orders == pytest.approx(1 / 3)
    assert k.units == 4
    assert k.active_customers == 2  # anonymous GUEST excluded
    assert k.new_customers == 2


def test_window_and_new_customers_use_full_history(tx: pd.DataFrame) -> None:
    feb = compute_kpis(tx, start=pd.Timestamp("2026-02-01"), end=pd.Timestamp("2026-03-01"))
    assert (feb.net_revenue, feb.orders, feb.new_customers, feb.active_customers) == (20, 1, 0, 0)


def test_margin_is_none_without_any_cost() -> None:
    k = compute_kpis(make_tx({"gross": 10}))
    assert k.gross_margin_pct is None
    assert k.net_profit is None
    assert k.cost_coverage == 0.0


def test_empty_window_has_no_division_errors(tx: pd.DataFrame) -> None:
    k = compute_kpis(tx, start=pd.Timestamp("2030-01-01"))
    assert (k.net_revenue, k.orders, k.aov, k.refund_rate) == (0.0, 0, None, None)


def test_daily_series_is_gap_filled(tx: pd.DataFrame) -> None:
    daily = kpi_timeseries(tx, "D")
    assert len(daily) == 34  # 2026-01-05 .. 2026-02-07 inclusive
    assert daily.loc["2026-01-06", "net_revenue"] == 0
    assert daily.loc["2026-01-05", "orders"] == 1
    assert daily["net_revenue"].sum() == 160
    assert daily["complete"].all()


def test_monthly_series_and_completeness(tx: pd.DataFrame) -> None:
    monthly = kpi_timeseries(tx, "M")
    assert [str(p)[:7] for p in monthly.index] == ["2026-01", "2026-02"]
    assert monthly["complete"].tolist() == [False, False]
    assert monthly.loc["2026-01-01", "refund_rate"] == pytest.approx(80 / 220)
    assert monthly.loc["2026-01-01", "new_customers"] == 2
    assert latest_growth(monthly) is None  # no two complete months


def test_growth_uses_complete_periods_only() -> None:
    tx = make_tx(
        {"ts": "2026-01-01 08:00", "gross": 100},
        {"ts": "2026-02-15 08:00", "gross": 150},
        {"ts": "2026-03-31 08:00", "gross": 120},
    )
    growth = latest_growth(kpi_timeseries(tx, "M"))
    assert growth is not None
    assert str(growth.period.date()) == "2026-03-01"
    assert growth.pct == pytest.approx(-0.2)
    weekly = kpi_timeseries(tx, "W")
    assert weekly.index[0].dayofweek == 0  # ISO weeks start Monday
    assert not weekly["complete"].iloc[0]  # 2026-01-01 is a Thursday


def test_pareto_core_set() -> None:
    tx = make_tx(
        *[{"product_id": p, "gross": g} for p, g in [("A", 50), ("B", 30), ("C", 15), ("D", 5)]],
        {"product_id": "E", "gross": 99, "status": "cancelled"},
    )
    result = pareto(tx, by="product_id")
    assert result.items_total == 5
    assert result.items_for_80 == 2
    assert result.table["item"].tolist() == ["A", "B", "C", "D", "E"]
    assert result.table["in_core"].tolist() == [True, True, False, False, False]
    assert result.revenue_share_top_20pct == pytest.approx(0.5)


def test_breakdown_shares(tx: pd.DataFrame) -> None:
    mix = breakdown(tx, "product_id")
    assert mix.loc["P1", "net_revenue"] == 90
    assert mix["share"].sum() == pytest.approx(1.0)
