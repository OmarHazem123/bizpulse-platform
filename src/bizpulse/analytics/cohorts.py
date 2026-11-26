"""Monthly acquisition cohorts: retention matrix and empirical LTV (TR-AN-02).

* Cohort = month of a customer's first non-cancelled order (computed from facts).
* ``retention[c, k]`` = customers of cohort *c* active in month ``c + k`` / cohort size.
* ``ltv[c, k]`` = cumulative completed net revenue per cohort customer from the acquisition
  month through month ``c + k``.
* Right-censoring: cells after the last *complete* month of data are NaN (unknown), never 0.
  LTV@k only uses cohorts that have been observed for k full months after acquisition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CohortResult:
    sizes: pd.Series[int]  # cohort month -> customers
    active: pd.DataFrame  # cohort x months_since -> active customers (NaN censored)
    retention: pd.DataFrame  # fractions (NaN censored)
    ltv: pd.DataFrame  # cumulative net revenue per customer (NaN censored)
    last_complete_month: pd.Timestamp


def _month(ts: pd.Series[pd.Timestamp]) -> pd.Series[pd.Timestamp]:
    day = ts.dt.normalize()
    return day - pd.to_timedelta(day.dt.day - 1, unit="D")


def _months_between(
    later: pd.Series[pd.Timestamp], earlier: pd.Series[pd.Timestamp]
) -> pd.Series[int]:
    months: pd.Series[int] = (later.dt.year - earlier.dt.year) * 12 + (
        later.dt.month - earlier.dt.month
    )
    return months


def last_complete_month(tx: pd.DataFrame) -> pd.Timestamp:
    last_day = pd.Timestamp(tx["order_ts"].max()).normalize()
    month = last_day.replace(day=1)
    if (last_day + pd.Timedelta(1, unit="D")).month != last_day.month:
        return month  # data runs through month end
    return month - pd.offsets.MonthBegin(1)


def cohort_analysis(tx: pd.DataFrame) -> CohortResult:
    rows = tx.loc[tx["status"].ne("cancelled") & ~tx["is_anonymous"]].copy()
    final = last_complete_month(tx)
    if rows.empty:
        empty = pd.DataFrame()
        return CohortResult(pd.Series(dtype="int64"), empty, empty, empty, final)
    rows["month"] = _month(rows["order_ts"])
    cohort_of = rows.groupby("customer_id")["month"].min()
    rows["cohort"] = rows["customer_id"].map(cohort_of)
    rows["k"] = _months_between(rows["month"], rows["cohort"])
    rows["completed_net"] = rows["net"].where(rows["status"].eq("completed"), 0.0)

    sizes = cohort_of.value_counts().sort_index().astype("int64")
    sizes = sizes[sizes.index <= final]
    cohorts = sizes.index
    horizon = int(_months_between(pd.Series([final]), pd.Series([cohorts.min()])).iloc[0]) + 1
    columns = pd.RangeIndex(horizon, name="months_since")

    active = (
        rows.groupby(["cohort", "k"])["customer_id"].nunique().unstack(fill_value=0)
        .reindex(index=cohorts, columns=columns, fill_value=0).astype("float64")
    )  # fmt: skip
    revenue = (
        rows.groupby(["cohort", "k"])["completed_net"].sum().unstack(fill_value=0)
        .reindex(index=cohorts, columns=columns, fill_value=0.0)
    )  # fmt: skip
    observed = np.array(
        [
            [int(_months_between(pd.Series([final]), pd.Series([c])).iloc[0]) >= k for k in columns]
            for c in cohorts
        ]
    )
    active = active.where(observed)
    retention = active.div(sizes, axis=0)
    ltv = revenue.cumsum(axis=1).div(sizes, axis=0).where(observed)
    active.index.name = retention.index.name = ltv.index.name = "cohort"
    return CohortResult(sizes, active, retention, ltv, final)


def ltv_at(
    result: CohortResult, months: tuple[int, ...] = (1, 3, 6, 12)
) -> dict[int, float | None]:
    """Size-weighted LTV@k over cohorts observed for k full months after acquisition."""
    out: dict[int, float | None] = {}
    for k in months:
        if result.ltv.empty or k not in result.ltv.columns:
            out[k] = None
            continue
        values = result.ltv[k]
        mask = values.notna()
        weights = result.sizes[mask]
        out[k] = float((values[mask] * weights).sum() / weights.sum()) if weights.sum() else None
    return out


def average_retention_curve(result: CohortResult) -> pd.Series[float]:
    """Size-weighted retention by months since acquisition (observed cells only)."""
    if result.retention.empty:
        return pd.Series(dtype="float64")
    weighted = result.retention.mul(result.sizes, axis=0)
    weights = result.retention.notna().mul(result.sizes, axis=0)
    curve: pd.Series[float] = weighted.sum() / weights.sum().replace(0, np.nan)
    return curve.dropna()
