"""Grouping anomaly alerts into ranked incidents."""

from __future__ import annotations

import pandas as pd

from bizpulse.ml.anomaly import Alert
from bizpulse.ml.incidents import group_incidents


def _alert(
    day: str,
    *,
    metric: str = "net_revenue",
    severity: str = "WARNING",
    z: float = 4.0,
    scope: tuple[str, str] | None = None,
) -> Alert:
    return Alert(
        date=pd.Timestamp(day),
        metric=metric,
        value=1.0,
        baseline=2.0,
        deviation=-0.5,
        z=-z,
        methods=("zscore",),
        severity=severity,
        direction="drop",
        scope_dimension=scope[0] if scope else None,
        scope_value=scope[1] if scope else None,
    )


def test_adjacent_alerts_form_one_incident_and_gaps_split() -> None:
    outage = [
        _alert("2026-03-10", severity="CRITICAL", z=9.0),
        _alert("2026-03-10", metric="orders", severity="CRITICAL", z=8.0),
        _alert("2026-03-10", scope=("region", "North"), z=6.0),
        _alert("2026-03-11", scope=("channel", "web")),
        _alert("2026-03-12", metric="aov"),
    ]
    later = [_alert("2026-03-20", scope=("region", "South"))]
    incidents = group_incidents([*later, *outage], "D")
    assert len(incidents) == 2
    broad, local = incidents
    assert (broad.start, broad.end) == (pd.Timestamp("2026-03-10"), pd.Timestamp("2026-03-12"))
    assert broad.company_wide and not local.company_wide
    assert broad.severity == "CRITICAL"
    assert broad.metrics == ("net_revenue", "orders", "aov")
    assert broad.slices == ("channel=web", "region=North")
    assert broad.periods == 3 and len(broad.alerts) == 5
    # The headline is the most severe business-total alert with the largest |z|.
    assert broad.headline.scope == "total" and broad.headline.metric == "net_revenue"
    assert local.slices == ("region=South",)


def test_weekly_monitoring_merges_consecutive_weeks() -> None:
    weeks = [_alert("2026-03-02"), _alert("2026-03-09"), _alert("2026-03-23")]
    chronological = sorted(group_incidents(weeks, "W"), key=lambda i: i.start)
    assert [len(i.alerts) for i in chronological] == [2, 1]
    assert len(group_incidents(weeks, "D")) == 3


def test_ranking_severity_then_reach_then_size() -> None:
    alerts = [
        _alert("2026-01-05", scope=("region", "A"), z=20.0),  # local warning, huge z
        _alert("2026-02-05", z=4.0),  # company-wide warning
        _alert("2026-03-05", severity="CRITICAL", scope=("channel", "x"), z=6.0),
        _alert("2026-04-05", severity="CRITICAL", z=5.0),  # company-wide critical
    ]
    ranked = group_incidents(alerts, "D")
    assert [i.start.month for i in ranked] == [4, 3, 2, 1]


def test_no_alerts_no_incidents() -> None:
    assert group_incidents([], "D") == []
