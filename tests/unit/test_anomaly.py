"""Anomaly detection on synthetic transactions with known injected events."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bizpulse.analytics.data import prepare
from bizpulse.ml.anomaly import detect_anomalies, score_metric

REGIONS = ("R1", "R2", "R3")


def synthetic_tx(
    days: int = 160,
    *,
    drop_day: int | None = None,
    spike_day: int | None = None,
    seed: int = 3,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weekly = np.array([1.0, 0.95, 0.95, 1.0, 1.1, 1.3, 1.2])
    start = pd.Timestamp("2026-01-05")
    rows: list[dict[str, object]] = []
    order = 0
    for d in range(days):
        day = start + pd.Timedelta(d, unit="D")
        n = int(rng.poisson(80 * weekly[day.dayofweek]))
        n_extra = 3 * n if d == spike_day else 0
        for i in range(n + n_extra):
            region = str(rng.choice(REGIONS, p=[0.45, 0.3, 0.25]))
            if d == drop_day and region != "R3":
                continue  # a large outage: only region R3 keeps selling
            order += 1
            gross = float(rng.lognormal(3.5, 0.3))
            rows.append(
                {
                    "order_ts": day + pd.Timedelta(int(rng.integers(0, 86_000)), unit="s"),
                    "order_id": f"O{order}",
                    "transaction_id": f"T{order}",
                    "customer_id": f"C{int(rng.integers(0, 3000))}",
                    "is_anonymous": False,
                    "product_id": "P1",
                    "product_name": "P1",
                    "category": "A" if rng.random() < 0.6 else "B",
                    "channel": "app" if i >= n else str(rng.choice(["web", "app"], p=[0.8, 0.2])),
                    "region": region,
                    "status": "refunded" if rng.random() < 0.03 else "completed",
                    "quantity": 1.0,
                    "gross": gross,
                    "discount": 0.0,
                    "net": gross,
                    "cost": gross * 0.5,
                    "fee": 0.0,
                }
            )
    return prepare(pd.DataFrame(rows))


def test_large_drop_is_critical_with_root_cause() -> None:
    tx = synthetic_tx(drop_day=120)
    day = pd.Timestamp("2026-01-05") + pd.Timedelta(120, unit="D")
    alerts = detect_anomalies(tx).alerts_between(day, day, "net_revenue")
    assert len(alerts) == 1
    alert = alerts[0]
    assert (alert.severity, alert.direction) == ("CRITICAL", "drop")
    assert alert.deviation < -0.6
    top = alert.top_cause("region")
    assert top is not None and top.value == "R1"  # biggest region lost the most revenue
    assert top.delta < 0


def test_spike_is_detected_and_attributed_to_channel() -> None:
    tx = synthetic_tx(spike_day=100)
    day = pd.Timestamp("2026-01-05") + pd.Timedelta(100, unit="D")
    alerts = detect_anomalies(tx).alerts_between(day, day, "orders")
    assert alerts and alerts[0].direction == "spike"
    top = alerts[0].top_cause("channel")
    assert top is not None and top.value == "app"


def test_quiet_series_false_alarm_rate_regression_guard() -> None:
    # Measured, not pre-registered: on quiet Poisson data the rules raise 6 volume alerts
    # in 200 days (1.5 % of metric-days), 4 of them CRITICAL because IsolationForest with
    # a fixed 2 % contamination always labels ~2 % of days and can upgrade marginal z-flags.
    # Reported as an open decision; this guard keeps the rate from getting worse.
    days = 200
    report = detect_anomalies(synthetic_tx(days=days, seed=11))
    volume_alerts = [a for a in report.alerts if a.metric in ("net_revenue", "orders")]
    assert len(volume_alerts) <= 0.02 * 2 * days


def test_scores_use_trailing_windows_only() -> None:
    index = pd.date_range("2026-01-01", periods=120, freq="D")
    rng = np.random.default_rng(5)
    series = pd.Series(1000 + rng.normal(0, 50, 120), index=index, dtype="float64")
    base = score_metric(series, weekday=True)
    poisoned = series.copy()
    poisoned.iloc[90:] = 1e9  # perturb the current day and everything after it
    changed = score_metric(poisoned, weekday=True)
    pd.testing.assert_frame_equal(base.iloc[:90], changed.iloc[:90])
    assert bool(changed["z_flag"].iloc[90])


def test_point_in_time_flags_are_stable_when_data_is_appended() -> None:
    # z / IQR flags use trailing windows only, so they never change when later data
    # arrives. Severity is NOT pinned: IsolationForest is fit on the whole period
    # (retrospective by design, see TECHNICAL-REQUIREMENTS section 7.1).
    full = synthetic_tx(days=150, drop_day=120)
    cut = pd.Timestamp("2026-01-05") + pd.Timedelta(130, unit="D")
    early = detect_anomalies(full.loc[full["order_ts"] < cut]).scores
    later = detect_anomalies(full).scores
    for metric in ("net_revenue", "orders"):
        for column in ("z", "z_flag", "iqr_flag"):
            pd.testing.assert_series_equal(
                early[(metric, column)], later[(metric, column)].loc[early.index]
            )


def test_warmup_and_volume_guard() -> None:
    report = detect_anomalies(synthetic_tx(days=40))
    scores = report.scores
    assert scores[("net_revenue", "z")].iloc[:28].isna().all()  # no history yet
    tiny = synthetic_tx(days=120)
    tiny = tiny.loc[tiny["order_ts"].dt.day != 15].copy()  # keep the volume shape
    low = tiny.groupby(tiny["order_ts"].dt.normalize()).head(5)  # <= 5 orders per day
    guarded = detect_anomalies(low).scores
    assert guarded[("refund_rate", "z")].isna().all()
    assert guarded[("aov", "z")].isna().all()


@pytest.mark.parametrize("metric", ["net_revenue", "orders", "aov", "refund_rate"])
def test_scores_have_expected_columns(metric: str) -> None:
    scores = detect_anomalies(synthetic_tx(days=60)).scores
    for column in ("value", "baseline", "dev", "z", "z_flag", "iqr_flag"):
        assert (metric, column) in scores.columns
