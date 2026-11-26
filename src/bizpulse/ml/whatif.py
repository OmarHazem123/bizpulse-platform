"""6-month What-If growth & unit-economics simulator (TR-ML-05).

Baseline: a **steady state** at the trailing 3-complete-month averages of the data.

* Active-base retention ``r = 1 - new / active``: the share of last month's active
  identified customers active again this month, including reactivations. It is defined
  so the zero-lever projection reproduces the baseline exactly.
* ``active_m = r_s * active_{m-1} + new_m``. Orders = active x frequency.
  Gross = orders x gross-AOV. Discounts, refunds and fees are rates of sales.
  COGS = orders x baseline cost per order (refund mix included): price/AOV levers lift
  margin and discount levers cut it. Fees scale with net revenue.
* Anonymous revenue (e.g. café guests) is a separate stream that responds to price
  levers (AOV, discount) but not to customer levers.
* Marketing spend = new customers x CAC (CAC is an explicit input: it is not in the data).

Unit economics: margin per active customer-month m, expected lifetime ``1 / (1 - r)``
months, so ``LTV = m / (1 - r)``, ``LTV:CAC`` and payback ``CAC / m`` months.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from bizpulse.analytics.kpis import first_orders, kpi_timeseries

MONTHS = 6
TRAILING_MONTHS = 3
MAX_RETENTION = 0.99
MAX_DISCOUNT = 0.9


@dataclass(frozen=True)
class Baseline:
    months: tuple[str, ...]
    active_customers: float
    new_customers: float
    retention: float
    orders_per_customer: float
    aov_gross: float
    discount_rate: float
    refund_rate: float
    cost_per_order: float | None
    cost_ratio: float | None
    fee_rate: float
    anonymous_net_revenue: float
    actual_net_revenue: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Levers:
    """Scenario levers (all relative to the baseline; zero = status quo)."""

    retention_pp: float = 0.0  # absolute change of monthly active-base retention
    aov_pct: float = 0.0  # relative change of gross AOV (price / basket size)
    discount_pp: float = 0.0  # absolute change of the discount rate
    new_customers_pct: float = 0.0  # relative change of acquisition volume
    frequency_pct: float = 0.0  # relative change of orders per active customer
    cac: float = 0.0  # acquisition cost per new customer (reporting currency)

    def validate(self, baseline: Baseline) -> None:
        checks = [
            (-0.5 <= self.retention_pp <= 0.5, "retention_pp must be within [-0.5, 0.5]"),
            (-0.5 <= self.aov_pct <= 1.0, "aov_pct must be within [-0.5, 1.0]"),
            (-0.5 <= self.discount_pp <= 0.5, "discount_pp must be within [-0.5, 0.5]"),
            (-1.0 <= self.new_customers_pct <= 3.0, "new_customers_pct must be within [-1, 3]"),
            (-0.5 <= self.frequency_pct <= 1.0, "frequency_pct must be within [-0.5, 1.0]"),
            (self.cac >= 0, "cac must be >= 0"),
            (
                0.0
                <= baseline.retention + self.retention_pp
                <= max(MAX_RETENTION, baseline.retention),
                f"resulting retention must be within [0, {MAX_RETENTION}]",
            ),
            (
                0.0 <= baseline.discount_rate + self.discount_pp <= MAX_DISCOUNT,
                f"resulting discount rate must be within [0, {MAX_DISCOUNT}]",
            ),
        ]
        errors = [message for ok, message in checks if not ok]
        if errors:
            raise ValueError("; ".join(errors))


def _safe(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_baseline(tx: pd.DataFrame, trailing: int = TRAILING_MONTHS) -> Baseline:
    """Trailing averages over the last *trailing* complete months of data."""
    monthly = kpi_timeseries(tx, "M")
    complete = monthly.loc[monthly["complete"]].tail(trailing)
    if complete.empty:
        raise ValueError("no complete month of data to build a baseline")
    start = pd.Timestamp(complete.index[0])
    end = pd.Timestamp(complete.index[-1]) + pd.offsets.MonthBegin(1)
    part = tx.loc[(tx["order_ts"] >= start) & (tx["order_ts"] < end)]
    s = part["status"].ne("cancelled")
    c = part["status"].eq("completed")
    r = part["status"].eq("refunded")
    ident = ~part["is_anonymous"]
    n = len(complete)

    active = float(complete["active_customers"].mean())
    firsts = first_orders(tx)
    new = float(((firsts >= start) & (firsts < end)).sum()) / n
    orders = float(part.loc[s & ident, "order_id"].nunique())
    gross = float(part.loc[s & ident, "gross"].sum())
    discounts = float(part.loc[s & ident, "discount"].sum())
    sold = gross - discounts
    refunds = float(part.loc[r & ident, "net"].sum())
    fees = float(part.loc[s, "fee"].sum())
    net_all = float(part.loc[c, "net"].sum())
    known = c & part["cost"].notna()
    cost_ratio = _safe(float(part.loc[known, "cost"].sum()), float(part.loc[known, "net"].sum()))
    net_ident_completed = float(part.loc[c & ident, "net"].sum())
    cost_per_order = cost_ratio * net_ident_completed / orders if known.any() and orders else None
    return Baseline(
        months=tuple(str(p)[:7] for p in complete.index),
        active_customers=active,
        new_customers=new,
        # Unclamped so the zero-lever projection is an exact steady state; the 0.99 cap
        # applies only to the customer-lifetime calculation.
        retention=min(1.0, max(0.0, 1.0 - _safe(new, active))),
        orders_per_customer=_safe(orders / n, active),
        aov_gross=_safe(gross, orders),
        discount_rate=_safe(discounts, gross),
        refund_rate=_safe(refunds, sold),
        cost_per_order=cost_per_order,
        cost_ratio=cost_ratio if known.any() else None,
        fee_rate=_safe(fees, net_all),
        anonymous_net_revenue=float(part.loc[c & ~ident, "net"].sum()) / n,
        actual_net_revenue=float(complete["net_revenue"].mean()),
    )


def project(baseline: Baseline, levers: Levers, months: int = MONTHS) -> pd.DataFrame:
    """Month-by-month projection under *levers* (the zero lever set gives the baseline)."""
    levers.validate(baseline)
    retention = baseline.retention + levers.retention_pp
    discount_rate = baseline.discount_rate + levers.discount_pp
    price_factor = (1 + levers.aov_pct) * _safe(1 - discount_rate, 1 - baseline.discount_rate)
    new = baseline.new_customers * (1 + levers.new_customers_pct)
    frequency = baseline.orders_per_customer * (1 + levers.frequency_pct)
    aov = baseline.aov_gross * (1 + levers.aov_pct)
    rows = []
    active = baseline.active_customers
    for month in range(1, months + 1):
        active = retention * active + new
        orders = active * frequency
        gross = orders * aov
        discounts = gross * discount_rate
        sold = gross - discounts
        refunds = sold * baseline.refund_rate
        anonymous = baseline.anonymous_net_revenue * price_factor
        net = sold - refunds + anonymous
        # Cost follows volume, not price: identified COGS per order (refund mix included)
        # and the anonymous stream's baseline cost, which price levers leave unchanged.
        cogs = 0.0
        if baseline.cost_per_order is not None and baseline.cost_ratio is not None:
            cogs = (
                orders * baseline.cost_per_order
                + baseline.anonymous_net_revenue * baseline.cost_ratio
            )
        fees = net * baseline.fee_rate
        marketing = new * levers.cac
        rows.append(
            {
                "month": month,
                "active_customers": active,
                "new_customers": new,
                "orders": orders,
                "gross_revenue": gross,
                "discounts": discounts,
                "refunds": refunds,
                "anonymous_net_revenue": anonymous,
                "net_revenue": net,
                "cogs": cogs,
                "fees": fees,
                "marketing": marketing,
                "contribution": net - cogs - fees - marketing,
            }
        )
    return pd.DataFrame(rows).set_index("month")


@dataclass(frozen=True)
class UnitEconomics:
    margin_per_customer_month: float
    expected_lifetime_months: float
    ltv: float
    cac: float
    ltv_to_cac: float | None
    payback_months: float | None


def unit_economics(projection: pd.DataFrame, retention: float, cac: float) -> UnitEconomics:
    """Steady-state unit economics from the last projected month."""
    last = projection.iloc[-1]
    identified_net = float(last["net_revenue"] - last["anonymous_net_revenue"])
    share = _safe(identified_net, float(last["net_revenue"]))
    margin = _safe(
        (float(last["net_revenue"] - last["cogs"] - last["fees"])) * share,
        float(last["active_customers"]),
    )
    lifetime = 1.0 / (1.0 - min(retention, MAX_RETENTION))
    ltv = margin * lifetime
    return UnitEconomics(
        margin_per_customer_month=margin,
        expected_lifetime_months=lifetime,
        ltv=ltv,
        cac=cac,
        ltv_to_cac=ltv / cac if cac > 0 else None,
        payback_months=cac / margin if cac > 0 and margin > 0 else None,
    )


@dataclass(frozen=True)
class WhatIfResult:
    baseline: Baseline
    levers: Levers
    baseline_projection: pd.DataFrame
    scenario_projection: pd.DataFrame
    baseline_units: UnitEconomics
    scenario_units: UnitEconomics

    def totals(self) -> pd.DataFrame:
        """6-month totals: baseline, scenario, absolute and relative change."""
        columns = ["net_revenue", "cogs", "fees", "marketing", "contribution", "orders"]
        base = self.baseline_projection[columns].sum()
        scen = self.scenario_projection[columns].sum()
        out = pd.DataFrame({"baseline": base, "scenario": scen})
        out["change"] = out["scenario"] - out["baseline"]
        out["change_pct"] = out["change"] / out["baseline"].where(out["baseline"] != 0)
        return out


def simulate(baseline: Baseline, levers: Levers, months: int = MONTHS) -> WhatIfResult:
    """Scenario vs status quo. Both use the same CAC so marketing is comparable."""
    status_quo = Levers(cac=levers.cac)
    base_proj = project(baseline, status_quo, months)
    scen_proj = project(baseline, levers, months)
    return WhatIfResult(
        baseline=baseline,
        levers=levers,
        baseline_projection=base_proj,
        scenario_projection=scen_proj,
        baseline_units=unit_economics(base_proj, baseline.retention, levers.cac),
        scenario_units=unit_economics(
            scen_proj, baseline.retention + levers.retention_pp, levers.cac
        ),
    )
