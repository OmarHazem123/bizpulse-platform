"""Group anomaly alerts into incidents so broad events stay readable.

One real-world event usually raises several alerts. A checkout outage, for example,
can raise:

- business-total alerts on more than one metric (net revenue, orders, ...);
- alerts on every store / channel slice the outage touched;
- all of the above on consecutive periods.

Alerts whose periods are adjacent (a gap of at most one monitoring period) form one
incident. Incidents are ranked by:

1. severity;
2. company-wide before local;
3. magnitude;
4. recency.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from bizpulse.ml.anomaly import METRICS, Alert, Granularity

SEVERITY_RANK: dict[str, int] = {"CRITICAL": 0, "WARNING": 1}


def _alert_rank(alert: Alert) -> tuple[int, int, float]:
    """Most important first: severity, business-total scope, magnitude."""
    return (SEVERITY_RANK.get(alert.severity, 9), alert.scope != "total", -abs(alert.z))


@dataclass(frozen=True)
class Incident:
    alerts: tuple[Alert, ...]  # ordered most important first

    @property
    def start(self) -> pd.Timestamp:
        return min(a.date for a in self.alerts)

    @property
    def end(self) -> pd.Timestamp:
        return max(a.date for a in self.alerts)

    @property
    def headline(self) -> Alert:
        """The alert that best describes the incident."""
        return self.alerts[0]

    @property
    def severity(self) -> str:
        return self.headline.severity

    @property
    def company_wide(self) -> bool:
        """True when the business total itself moved (not just one store / channel)."""
        return any(a.scope == "total" for a in self.alerts)

    @property
    def metrics(self) -> tuple[str, ...]:
        found = {a.metric for a in self.alerts}
        return tuple(m for m in METRICS if m in found)

    @property
    def slices(self) -> tuple[str, ...]:
        return tuple(sorted({a.scope for a in self.alerts if a.scope != "total"}))

    @property
    def periods(self) -> int:
        return len({a.date for a in self.alerts})

    @property
    def max_abs_z(self) -> float:
        return max(abs(a.z) for a in self.alerts)

    def rank_key(self) -> tuple[int, int, float, float]:
        return (
            SEVERITY_RANK.get(self.severity, 9),
            0 if self.company_wide else 1,
            -self.max_abs_z,
            -self.end.timestamp(),
        )


def group_incidents(alerts: Sequence[Alert], granularity: Granularity) -> list[Incident]:
    """Merge alerts on adjacent periods into incidents, ranked most important first."""
    step = pd.Timedelta(7 if granularity == "W" else 1, unit="D")
    groups: list[list[Alert]] = []
    last: pd.Timestamp | None = None
    for alert in sorted(alerts, key=lambda a: (a.date, a.scope, a.metric)):
        if last is None or alert.date - last > step:
            groups.append([])
        groups[-1].append(alert)
        last = alert.date if last is None else max(last, alert.date)
    incidents = [Incident(tuple(sorted(group, key=_alert_rank))) for group in groups]
    return sorted(incidents, key=Incident.rank_key)
