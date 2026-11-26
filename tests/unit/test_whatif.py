"""What-if simulator: hand-computed projection, identity, monotonicity, validation."""

from __future__ import annotations

import pandas as pd
import pytest

from bizpulse.ml.whatif import Baseline, Levers, compute_baseline, project, simulate
from tests.conftest import make_tx

BASE = Baseline(
    months=("2026-01",),
    active_customers=100.0,
    new_customers=20.0,
    retention=0.8,
    orders_per_customer=1.5,
    aov_gross=50.0,
    discount_rate=0.1,
    refund_rate=0.05,
    cost_per_order=20.0,
    cost_ratio=0.4,
    fee_rate=0.03,
    anonymous_net_revenue=1000.0,
    actual_net_revenue=7412.5,
)


def test_projection_matches_hand_computation() -> None:
    proj = project(BASE, Levers())
    first = proj.loc[1]
    assert first["active_customers"] == pytest.approx(100.0)  # steady state
    assert first["orders"] == pytest.approx(150.0)
    assert first["gross_revenue"] == pytest.approx(7500.0)
    assert first["discounts"] == pytest.approx(750.0)
    assert first["refunds"] == pytest.approx(337.5)
    assert first["net_revenue"] == pytest.approx(6412.5 + 1000.0)
    assert first["cogs"] == pytest.approx(150 * 20 + 1000 * 0.4)
    assert first["fees"] == pytest.approx(7412.5 * 0.03)
    assert first["contribution"] == pytest.approx(7412.5 - 3400 - 222.375)
    assert (proj["net_revenue"] == first["net_revenue"]).all()  # flat steady state


def test_zero_levers_reproduce_baseline_exactly() -> None:
    result = simulate(BASE, Levers(cac=25.0))
    pd.testing.assert_frame_equal(result.baseline_projection, result.scenario_projection)
    totals = result.totals()
    assert (totals["change"] == 0).all()
    assert result.baseline_units == result.scenario_units


def test_levers_move_outcomes_in_the_expected_direction() -> None:
    base = project(BASE, Levers())
    retained = project(BASE, Levers(retention_pp=0.05))
    assert retained["active_customers"].is_monotonic_increasing

    def last(frame: pd.DataFrame, column: str) -> float:
        return float(frame[column].iloc[-1])

    assert last(retained, "net_revenue") > last(base, "net_revenue")
    discounted = project(BASE, Levers(discount_pp=0.05))
    assert last(discounted, "contribution") < last(base, "contribution")
    assert last(discounted, "cogs") == last(base, "cogs")  # price does not change cost
    pricier = project(BASE, Levers(aov_pct=0.1))
    assert last(pricier, "contribution") > last(base, "contribution")
    acquired = project(BASE, Levers(new_customers_pct=0.5, cac=30.0))
    assert last(acquired, "marketing") == pytest.approx(30 * 30.0)


def test_unit_economics() -> None:
    result = simulate(BASE, Levers(cac=50.0))
    units = result.baseline_units
    share = 6412.5 / 7412.5
    margin = (7412.5 - 3400 - 222.375) * share / 100
    assert units.margin_per_customer_month == pytest.approx(margin)
    assert units.expected_lifetime_months == pytest.approx(5.0)
    assert units.ltv == pytest.approx(margin * 5)
    assert units.ltv_to_cac == pytest.approx(margin * 5 / 50)
    assert units.payback_months == pytest.approx(50 / margin)
    free = simulate(BASE, Levers()).baseline_units
    assert free.ltv_to_cac is None and free.payback_months is None


@pytest.mark.parametrize(
    "levers",
    [
        Levers(retention_pp=0.25),  # 0.8 + 0.25 > 0.99
        Levers(discount_pp=-0.2),  # 0.1 - 0.2 < 0
        Levers(aov_pct=-0.9),
        Levers(cac=-1.0),
    ],
)
def test_invalid_levers_rejected(levers: Levers) -> None:
    with pytest.raises(ValueError):
        project(BASE, levers)


def test_baseline_from_data_matches_actuals() -> None:
    rows = []
    for month in ("2026-01", "2026-02", "2026-03", "2026-04"):
        for i in range(6):
            rows.append(
                {"ts": f"{month}-{(i % 27) + 1:02d} 10:00", "customer_id": f"C{i}",
                 "order_id": f"{month}-{i}", "gross": 100, "discount": 10, "cost": 30}
            )  # fmt: skip
        rows.append({"ts": f"{month}-15 12:00", "customer_id": "G", "is_anonymous": True,
                     "order_id": f"{month}-g", "gross": 40, "cost": 10})  # fmt: skip
    rows.append({"ts": "2026-04-30 18:00", "customer_id": "C0", "order_id": "x", "gross": 1})
    baseline = compute_baseline(make_tx(*rows))
    assert baseline.months == ("2026-02", "2026-03", "2026-04")
    assert baseline.anonymous_net_revenue == pytest.approx(40.0)
    first = project(baseline, Levers()).loc[1, "net_revenue"]
    assert first == pytest.approx(baseline.actual_net_revenue)
