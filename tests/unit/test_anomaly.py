"""Anomaly detection on synthetic transactions with known injected events."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bizpulse.analytics.data import prepare
from bizpulse.ml.anomaly import (
    RULES,
    detect_anomalies,
    granularity_for,
    score_metric,
    severity_of,
)

REGIONS = ("R1", "R2", "R3")
START = pd.Timestamp("2026-01-05")


def synthetic_tx(
    days: int = 160,
    *,
    drop_day: int | None = None,
    drop_span: int = 1,
    spike_day: int | None = None,
    closed_region: tuple[int, str] | None = None,
    seed: int = 3,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weekly = np.array([1.0, 0.95, 0.95, 1.0, 1.1, 1.3, 1.2])
    rows: list[dict[str, object]] = []
    order = 0
    for d in range(days):
        day = START + pd.Timedelta(d, unit="D")
        n = int(rng.poisson(80 * weekly[day.dayofweek]))
        n_extra = 3 * n if d == spike_day else 0
        for i in range(n + n_extra):
            region = str(rng.choice(REGIONS, p=[0.45, 0.3, 0.25]))
            if drop_day is not None and drop_day <= d < drop_day + drop_span and region != "R3":
                continue  # a large outage: only region R3 keeps selling
            if closed_region is not None and d == closed_region[0] and region == closed_region[1]:
                continue  # one store closed for the day
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


def _day(offset: int) -> pd.Timestamp:
    return START + pd.Timedelta(offset, unit="D")


def test_large_drop_is_critical_with_root_cause() -> None:
    report = detect_anomalies(synthetic_tx(drop_day=120))
    alerts = report.alerts_between(_day(120), _day(120), "net_revenue", scope="total")
    assert len(alerts) == 1
    alert = alerts[0]
    assert (alert.severity, alert.direction) == ("CRITICAL", "drop")
    assert alert.deviation < -0.6
    top = alert.top_cause("region")
    assert top is not None and top.value == "R1"  # biggest region lost the most revenue
    assert top.delta < 0


def test_spike_is_detected_and_attributed_to_channel() -> None:
    report = detect_anomalies(synthetic_tx(spike_day=100))
    alerts = report.alerts_between(_day(100), _day(100), "orders", scope="total")
    assert alerts and alerts[0].direction == "spike"
    top = alerts[0].top_cause("channel")
    assert top is not None and top.value == "app"


def test_slice_monitoring_catches_a_localized_closure() -> None:
    report = detect_anomalies(synthetic_tx(closed_region=(120, "R2")))
    local = report.alerts_between(_day(120), _day(120), "net_revenue", scope="region=R2")
    assert len(local) == 1
    assert (local[0].severity, local[0].direction) == ("CRITICAL", "drop")
    assert local[0].value == 0.0


def test_noise_is_never_critical_and_false_alarms_are_rare() -> None:
    days = 200
    report = detect_anomalies(synthetic_tx(days=days, seed=11))
    assert [a for a in report.alerts if a.severity == "CRITICAL"] == []
    volume = [
        a for a in report.alerts if a.scope == "total" and a.metric in ("net_revenue", "orders")
    ]
    assert len(volume) <= 0.02 * 2 * days


def test_alerts_and_severities_are_immutable_when_data_is_appended() -> None:
    full = synthetic_tx(days=170, drop_day=120, spike_day=150)
    cut = _day(140)
    early = detect_anomalies(full.loc[full["order_ts"] < cut])
    later = detect_anomalies(full)
    assert early.alerts == [a for a in later.alerts if a.date < cut]  # incl. severity
    pd.testing.assert_frame_equal(early.scores, later.scores.loc[early.scores.index])


def test_scores_use_trailing_windows_only() -> None:
    index = pd.date_range("2026-01-01", periods=120, freq="D")
    rng = np.random.default_rng(5)
    series = pd.Series(1000 + rng.normal(0, 50, 120), index=index, dtype="float64")
    base = score_metric(series, volume=True)
    poisoned = series.copy()
    poisoned.iloc[90:] = 1e9  # perturb the current day and everything after it
    changed = score_metric(poisoned, volume=True)
    pd.testing.assert_frame_equal(base.iloc[:90], changed.iloc[:90])
    assert bool(changed["z_flag"].iloc[90])


def test_severity_calibration() -> None:
    assert severity_of(3.6, 0.4, True, False, True) == "WARNING"  # 2 methods are not enough
    assert severity_of(3.6, 0.4, True, True, True) == "CRITICAL"  # all three agree
    assert severity_of(5.2, 0.35, True, False, False) == "CRITICAL"
    assert severity_of(9.0, 0.10, True, True, True) == "WARNING"  # small effect size
    assert severity_of(2.0, 0.9, False, False, True) is None  # forest alone never alerts


def test_weekly_granularity_for_subscriptions() -> None:
    assert granularity_for("saas") == "W"
    assert granularity_for("retail_fb") == granularity_for("ecommerce") == "D"
    # A week-long incident; a single bad day is (by design) diluted in a weekly total.
    report = detect_anomalies(synthetic_tx(days=300, drop_day=252, drop_span=7), "W")
    assert report.granularity == "W"
    assert all(pd.Timestamp(str(d)).dayofweek == 0 for d in report.scores.index)
    hits = report.alerts_between(_day(252), _day(258), "net_revenue", scope="total")
    assert hits and hits[0].severity == "CRITICAL"
    assert hits[0].date.dayofweek == 0
    assert RULES["W"].warmup < RULES["D"].warmup


def test_warmup_and_volume_guard() -> None:
    report = detect_anomalies(synthetic_tx(days=40))
    assert report.scores[("net_revenue", "z")].iloc[:28].isna().all()  # no history yet
    tiny = synthetic_tx(days=120)
    low = tiny.groupby(tiny["order_ts"].dt.normalize()).head(5)  # <= 5 orders per day
    guarded = detect_anomalies(low).scores
    assert guarded[("refund_rate", "z")].isna().all()
    assert guarded[("aov", "z")].isna().all()


@pytest.mark.parametrize("metric", ["net_revenue", "orders", "aov", "refund_rate"])
def test_scores_have_expected_columns(metric: str) -> None:
    scores = detect_anomalies(synthetic_tx(days=60)).scores
    for column in ("value", "baseline", "dev", "z", "z_flag", "iqr_flag"):
        assert (metric, column) in scores.columns
