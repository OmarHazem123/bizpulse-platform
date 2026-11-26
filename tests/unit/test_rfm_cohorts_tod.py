from __future__ import annotations

import math

import pandas as pd
import pytest

from bizpulse.analytics.cohorts import (
    average_retention_curve,
    cohort_analysis,
    last_complete_month,
    ltv_at,
)
from bizpulse.analytics.rfm import (
    PLAYBOOKS,
    SEGMENTS,
    percentile_score,
    rfm_table,
    segment_for,
    segment_summary,
)
from bizpulse.analytics.timeofday import hourly_profile, peak_hours, to_local
from tests.conftest import make_tx

# --------------------------------------------------------------------------- RFM


def test_rfm_grid_is_a_complete_partition() -> None:
    cells = {(r, f): segment_for(r, f) for r in range(1, 6) for f in range(1, 6)}
    assert set(cells.values()) == set(SEGMENTS)
    assert cells[(5, 5)] == "Champions"
    assert cells[(3, 4)] == "Loyal Customers"
    assert cells[(5, 1)] == "New Customers"
    assert cells[(4, 2)] == cells[(3, 1)] == "Potential Loyalists"
    assert cells[(1, 5)] == "At Risk"
    assert cells[(2, 2)] == "Hibernating/Lost"
    assert set(PLAYBOOKS) == set(SEGMENTS)


def test_percentile_score_ties_share_lowest_score() -> None:
    scores = percentile_score(pd.Series([1.0, 1.0, 1.0, 1.0, 2.0, 5.0]))
    assert scores.tolist() == [1, 1, 1, 1, 4, 5]
    recency = percentile_score(pd.Series([1.0, 100.0]), higher_is_better=False)
    assert recency.tolist() == [3, 1]


def test_rfm_table_excludes_anonymous_and_cancelled() -> None:
    tx = make_tx(
        {"ts": "2026-03-01", "customer_id": "A", "order_id": "1", "gross": 100},
        {"ts": "2026-03-20", "customer_id": "A", "order_id": "2", "gross": 50},
        {"ts": "2026-03-21", "customer_id": "A", "order_id": "3", "gross": 70,
         "status": "refunded"},
        {"ts": "2026-01-01", "customer_id": "B", "order_id": "4", "gross": 10},
        {"ts": "2026-03-30", "customer_id": "B", "order_id": "5", "gross": 999,
         "status": "cancelled"},
        {"ts": "2026-03-31", "customer_id": "GUEST", "order_id": "6", "is_anonymous": True},
    )  # fmt: skip
    table = rfm_table(tx).set_index("customer_id")
    assert sorted(table.index) == ["A", "B"]
    assert table.loc["A", "frequency"] == 3  # refunded order still counts as an order
    assert table.loc["A", "monetary"] == 150  # completed net only
    assert table.loc["A", "recency_days"] == 11  # as_of = 2026-04-01 (data-derived)
    assert table.loc["B", "recency_days"] == 90
    r_scores = table["r_score"].to_dict()
    assert r_scores["A"] > r_scores["B"]
    summary = segment_summary(table.reset_index())
    assert summary["segment"].tolist() == list(SEGMENTS)
    assert summary["customers"].sum() == 2
    assert summary["share_revenue"].sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- cohorts


@pytest.fixture
def cohort_tx() -> pd.DataFrame:
    return make_tx(
        {"ts": "2026-01-10", "customer_id": "A", "order_id": "a1", "gross": 10},
        {"ts": "2026-02-10", "customer_id": "A", "order_id": "a2", "gross": 20},
        {"ts": "2026-03-10", "customer_id": "A", "order_id": "a3", "gross": 30},
        {"ts": "2026-01-20", "customer_id": "B", "order_id": "b1", "gross": 40},
        {"ts": "2026-02-05", "customer_id": "C", "order_id": "c1", "gross": 50},
        {"ts": "2026-03-31", "customer_id": "C", "order_id": "c2", "gross": 5},
        {"ts": "2026-01-02", "customer_id": "D", "order_id": "d0", "status": "cancelled"},
        {"ts": "2026-03-15", "customer_id": "D", "order_id": "d1", "gross": 7},
        {"ts": "2026-01-05", "customer_id": "GUEST", "is_anonymous": True, "order_id": "g"},
    )


def test_cohort_retention_and_censoring(cohort_tx: pd.DataFrame) -> None:
    result = cohort_analysis(cohort_tx)
    assert str(result.last_complete_month.date()) == "2026-03-01"
    sizes = {str(k)[:10]: int(v) for k, v in result.sizes.items()}
    assert sizes == {"2026-01-01": 2, "2026-02-01": 1, "2026-03-01": 1}  # D's cohort is March
    retention = result.retention.to_numpy()  # rows: Jan, Feb, Mar cohorts; cols: k = 0..2
    ltv = result.ltv.to_numpy()
    assert retention[0].tolist() == [1.0, 0.5, 0.5]
    assert retention[1, :2].tolist() == [1.0, 1.0]
    assert math.isnan(retention[1, 2])  # censored, not 0
    assert math.isnan(retention[2, 1])
    assert ltv[0].tolist() == [25.0, 35.0, 50.0]
    assert ltv[1, 1] == 55.0


def test_ltv_at_uses_only_fully_observed_cohorts(cohort_tx: pd.DataFrame) -> None:
    result = cohort_analysis(cohort_tx)
    values = ltv_at(result, months=(1, 2, 3))
    assert values[1] == pytest.approx((35 * 2 + 55 * 1) / 3)
    assert values[2] == pytest.approx(50.0)
    assert values[3] is None
    curve = average_retention_curve(result)
    assert curve.iloc[0] == 1.0
    assert curve.iloc[1] == pytest.approx((0.5 * 2 + 1.0 * 1) / 3)


def test_partial_last_month_is_not_complete() -> None:
    tx = make_tx({"ts": "2026-01-15"}, {"ts": "2026-03-14"})
    assert str(last_complete_month(tx).date()) == "2026-02-01"


# --------------------------------------------------------------------------- time of day


def test_to_local_handles_dst_transitions() -> None:
    utc = pd.Series(pd.to_datetime(["2026-03-28 10:00", "2026-03-30 10:00", "2026-10-26 10:00"]))
    local = to_local(utc, "Europe/Berlin")
    assert [ts.hour for ts in local] == [11, 12, 11]  # CET, CEST, CET
    assert to_local(utc, "UTC").tolist() == utc.tolist()


def test_hourly_profile_localizes_and_ranks() -> None:
    tx = make_tx(
        {"ts": "2026-07-01 06:30", "order_id": "1"},  # 08:30 Berlin (Wednesday)
        {"ts": "2026-07-01 06:45", "order_id": "2"},
        {"ts": "2026-07-01 11:00", "order_id": "3"},  # 13:00 Berlin
        {"ts": "2026-07-01 11:00", "order_id": "4", "status": "cancelled"},
    )
    profile = hourly_profile(tx, "Europe/Berlin")
    assert profile.loc[(2, 8), "orders"] == 2
    assert profile.loc[(2, 13), "orders"] == 1
    assert int(profile["orders"].sum()) == 3
    assert peak_hours(tx, "Europe/Berlin", top=2) == [8, 13]
    assert peak_hours(tx, "UTC", top=1) == [6]
