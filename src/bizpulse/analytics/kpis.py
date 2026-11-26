"""Executive KPIs, period series, growth, mix and Pareto (TECHNICAL-REQUIREMENTS.md §6).

Status sets: **S** = completed + refunded (non-cancelled), **C** = completed,
**R** = refunded. Cancelled rows never count.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

Freq = Literal["D", "W", "M"]


def _masks(tx: pd.DataFrame) -> tuple[pd.Series[bool], pd.Series[bool], pd.Series[bool]]:
    status = tx["status"]
    return status.ne("cancelled"), status.eq("completed"), status.eq("refunded")


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


@dataclass(frozen=True)
class KpiSummary:
    gross_revenue: float
    discounts: float
    refunds: float
    net_revenue: float
    cogs_known: float
    cost_coverage: float | None
    gross_margin_pct: float | None
    cogs_estimated: float | None
    gross_profit: float | None
    fees: float
    net_profit: float | None
    orders: int
    aov: float | None
    refund_rate: float | None
    refund_rate_orders: float | None
    units: float
    active_customers: int
    new_customers: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def first_orders(tx: pd.DataFrame) -> pd.Series[pd.Timestamp]:
    """First non-cancelled order timestamp per identified customer."""
    keep = tx["status"].ne("cancelled") & ~tx["is_anonymous"]
    firsts: pd.Series[pd.Timestamp] = tx.loc[keep].groupby("customer_id")["order_ts"].min()
    return firsts


def window(
    tx: pd.DataFrame, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Rows with ``start <= order_ts < end`` (either bound optional)."""
    mask = pd.Series(True, index=tx.index)
    if start is not None:
        mask &= tx["order_ts"] >= start
    if end is not None:
        mask &= tx["order_ts"] < end
    return tx.loc[mask]


def compute_kpis(
    tx: pd.DataFrame, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
) -> KpiSummary:
    """KPIs over ``[start, end)``. New customers use the full history for first orders."""
    part = window(tx, start, end)
    s, c, r = _masks(part)
    gross = float(part.loc[s, "gross"].sum())
    discounts = float(part.loc[s, "discount"].sum())
    refunds = float(part.loc[r, "net"].sum())
    net_revenue = float(part.loc[c, "net"].sum())
    known = part.loc[c & part["cost"].notna()]
    cogs_known = float(known["cost"].sum())
    net_known = float(known["net"].sum())
    margin = _ratio(net_known - cogs_known, net_known)
    cogs_est = None if margin is None else net_revenue * (1.0 - margin)
    gross_profit = None if cogs_est is None else net_revenue - cogs_est
    fees = float(part.loc[s, "fee"].sum())
    orders = int(part.loc[s, "order_id"].nunique())
    sold = gross - discounts
    firsts = first_orders(tx)
    new = firsts
    if start is not None:
        new = new[new >= start]
    if end is not None:
        new = new[new < end]
    return KpiSummary(
        gross_revenue=gross,
        discounts=discounts,
        refunds=refunds,
        net_revenue=net_revenue,
        cogs_known=cogs_known,
        cost_coverage=_ratio(net_known, net_revenue),
        gross_margin_pct=margin,
        cogs_estimated=cogs_est,
        gross_profit=gross_profit,
        fees=fees,
        net_profit=None if gross_profit is None else gross_profit - fees,
        orders=orders,
        aov=_ratio(sold, orders),
        refund_rate=_ratio(refunds, sold),
        refund_rate_orders=_ratio(float(part.loc[r, "order_id"].nunique()), orders),
        units=float(part.loc[s, "quantity"].sum()),
        active_customers=int(part.loc[s & ~part["is_anonymous"], "customer_id"].nunique()),
        new_customers=len(new),
    )


def period_start(ts: pd.Series[pd.Timestamp], freq: Freq) -> pd.Series[pd.Timestamp]:
    day = ts.dt.normalize()
    if freq == "D":
        return day
    if freq == "W":  # ISO weeks start on Monday
        return day - pd.to_timedelta(day.dt.dayofweek, unit="D")
    return day - pd.to_timedelta(day.dt.day - 1, unit="D")


def _next_period(start: pd.Timestamp, freq: Freq) -> pd.Timestamp:
    if freq == "D":
        return start + pd.Timedelta(1, unit="D")
    if freq == "W":
        return start + pd.Timedelta(7, unit="D")
    return start + pd.offsets.MonthBegin(1)


def kpi_timeseries(tx: pd.DataFrame, freq: Freq = "D") -> pd.DataFrame:
    """KPI table per period (index = period start), gap-filled, with a ``complete`` flag.

    A period is complete when the data covers every day of it.
    """
    s, c, r = _masks(tx)
    period = period_start(tx["order_ts"], freq)
    known = c & tx["cost"].notna()
    parts = pd.DataFrame(
        {
            "period": period,
            "gross_revenue": tx["gross"].where(s, 0.0),
            "discounts": tx["discount"].where(s, 0.0),
            "refunds": tx["net"].where(r, 0.0),
            "net_revenue": tx["net"].where(c, 0.0),
            "net_known": tx["net"].where(known, 0.0),
            "cogs_known": tx["cost"].where(known, 0.0),
            "fees": tx["fee"].where(s, 0.0),
        }
    )
    sums = parts.groupby("period").sum()
    orders = tx.loc[s].groupby(period[s])["order_id"].nunique()
    active = (
        tx.loc[s & ~tx["is_anonymous"]]
        .groupby(period[s & ~tx["is_anonymous"]])["customer_id"]
        .nunique()
    )
    firsts = first_orders(tx)
    new = period_start(firsts.reset_index(drop=True), freq).value_counts()

    first_day = pd.Timestamp(tx["order_ts"].min()).normalize()
    last_day = pd.Timestamp(tx["order_ts"].max()).normalize()
    starts = pd.date_range(period_start(pd.Series([first_day]), freq).iloc[0], last_day, freq="D")
    index = pd.DatetimeIndex(sorted(set(period_start(pd.Series(starts), freq))), name="period")
    out = sums.reindex(index, fill_value=0.0)
    out["orders"] = orders.reindex(index, fill_value=0).astype("int64")
    out["active_customers"] = active.reindex(index, fill_value=0).astype("int64")
    out["new_customers"] = new.reindex(index, fill_value=0).astype("int64")
    sold = out["gross_revenue"] - out["discounts"]
    out["aov"] = (sold / out["orders"].replace(0, np.nan)).astype("float64")
    out["refund_rate"] = (out["refunds"] / sold.replace(0.0, np.nan)).astype("float64")
    out["gross_margin_pct"] = (
        (out["net_known"] - out["cogs_known"]) / out["net_known"].replace(0.0, np.nan)
    ).astype("float64")
    out["complete"] = [
        p >= first_day and _next_period(p, freq) - pd.Timedelta(1, unit="D") <= last_day
        for p in out.index
    ]
    return out.drop(columns=["net_known"])


@dataclass(frozen=True)
class Growth:
    period: pd.Timestamp
    previous: pd.Timestamp
    value: float
    previous_value: float
    pct: float | None


def latest_growth(series: pd.DataFrame, metric: str = "net_revenue") -> Growth | None:
    """Growth of the last *complete* period vs the one before (MoM for ``M``, WoW for ``W``)."""
    complete = series.loc[series["complete"]]
    if len(complete) < 2:
        return None
    current, previous = complete.index[-1], complete.index[-2]
    value = float(complete[metric].iloc[-1])
    before = float(complete[metric].iloc[-2])
    return Growth(
        period=pd.Timestamp(current),
        previous=pd.Timestamp(previous),
        value=value,
        previous_value=before,
        pct=_ratio(value - before, before),
    )


def breakdown(tx: pd.DataFrame, by: str) -> pd.DataFrame:
    """Net revenue, orders and revenue share per value of dimension *by*."""
    s, c, _ = _masks(tx)
    revenue = tx.loc[c].groupby(by)["net"].sum()
    orders = tx.loc[s].groupby(by)["order_id"].nunique()
    out = pd.DataFrame({"net_revenue": revenue, "orders": orders}).fillna(0.0)
    total = float(out["net_revenue"].sum())
    out["share"] = out["net_revenue"] / total if total else 0.0
    return out.sort_values("net_revenue", ascending=False)


@dataclass(frozen=True)
class ParetoResult:
    table: pd.DataFrame  # item, net_revenue, share, cum_share, rank, in_core
    items_total: int
    items_for_80: int
    share_items_for_80: float
    revenue_share_top_20pct: float


def pareto(tx: pd.DataFrame, by: str = "product_name", threshold: float = 0.8) -> ParetoResult:
    """The smallest set of items reaching *threshold* of net revenue (ties broken by name)."""
    _, c, _ = _masks(tx)
    revenue = tx.loc[c].groupby(by)["net"].sum()
    items = sorted(tx[by].unique().tolist())
    table = pd.DataFrame({"item": items})
    table["net_revenue"] = table["item"].map(revenue).fillna(0.0).astype("float64")
    table = table.sort_values(["net_revenue", "item"], ascending=[False, True], kind="stable")
    table = table.reset_index(drop=True)
    total = float(table["net_revenue"].clip(lower=0.0).sum())
    table["share"] = table["net_revenue"] / total if total else 0.0
    table["cum_share"] = table["share"].cumsum()
    table["rank"] = np.arange(1, len(table) + 1)
    reached = np.flatnonzero(table["cum_share"].to_numpy() >= threshold - 1e-12)
    core = int(reached[0]) + 1 if len(reached) else len(table)
    table["in_core"] = table["rank"] <= core
    top_n = max(1, int(np.ceil(0.2 * len(table)))) if len(table) else 0
    top_share = float(table["share"].iloc[:top_n].sum()) if top_n else 0.0
    return ParetoResult(
        table=table,
        items_total=len(table),
        items_for_80=core,
        share_items_for_80=core / len(table) if len(table) else 0.0,
        revenue_share_top_20pct=top_share,
    )
