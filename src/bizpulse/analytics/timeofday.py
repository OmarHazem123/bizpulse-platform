"""Hour-of-day / weekday profiles in the dataset's business timezone (TR-AN-03).

Warehouse timestamps are canonical naive UTC. They are localized here, at query time,
with DST-aware IANA rules. Stored data is never altered.
"""

from __future__ import annotations

import pandas as pd


def to_local(ts: pd.Series[pd.Timestamp], timezone: str) -> pd.Series[pd.Timestamp]:
    """Naive UTC -> naive local wall-clock time in *timezone*."""
    local: pd.Series[pd.Timestamp] = (
        ts.dt.tz_localize("UTC").dt.tz_convert(timezone).dt.tz_localize(None)
    )
    return local


def hourly_profile(tx: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Orders and net revenue by local weekday (0 = Monday) x hour (0-23)."""
    rows = tx.loc[tx["status"].ne("cancelled")]
    local = to_local(rows["order_ts"], timezone)
    frame = pd.DataFrame(
        {
            "weekday": local.dt.dayofweek,
            "hour": local.dt.hour,
            "order_id": rows["order_id"],
            "net": rows["net"].where(rows["status"].eq("completed"), 0.0),
        }
    )
    grouped = frame.groupby(["weekday", "hour"])
    out = pd.DataFrame(
        {"orders": grouped["order_id"].nunique(), "net_revenue": grouped["net"].sum()}
    )
    full = pd.MultiIndex.from_product([range(7), range(24)], names=["weekday", "hour"])
    return out.reindex(full, fill_value=0)


def peak_hours(tx: pd.DataFrame, timezone: str, top: int = 3) -> list[int]:
    """The *top* local hours by order count (ties broken by earlier hour)."""
    by_hour = hourly_profile(tx, timezone).groupby(level="hour")["orders"].sum()
    counts = {int(str(hour)): int(n) for hour, n in by_hour.items()}
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [hour for hour, _ in ranked[:top]]
