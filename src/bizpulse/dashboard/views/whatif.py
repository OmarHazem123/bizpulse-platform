"""Tab 4: What-If Growth & Unit Economics Simulator."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from bizpulse.dashboard import charts, state
from bizpulse.dashboard.views.common import Context, change, signed_pct, table_view
from bizpulse.ml.whatif import MAX_DISCOUNT, MAX_RETENTION, MONTHS, Baseline, Levers, simulate

SLIDER_PP = 20.0  # +/- percentage points for rate levers
TOTAL_LABELS: dict[str, str] = {
    "net_revenue": "Net revenue",
    "cogs": "COGS",
    "fees": "Fees",
    "marketing": "Marketing (new customers x CAC)",
    "contribution": "Contribution",
    "orders": "Orders",
}


def _bounds(current: float, upper_limit: float) -> tuple[float, float]:
    """Slider range in pp so ``current + lever`` stays within ``[0, upper_limit]``."""
    low = max(-SLIDER_PP, -round(current * 100, 1))
    high = min(SLIDER_PP, round((upper_limit - current) * 100, 1))
    return low, max(high, low)


def _levers(base: Baseline, dataset: str) -> Levers:
    st.markdown("**Scenario levers** (relative to the trailing 3-month baseline)")
    c1, c2, c3 = st.columns(3)
    r_low, r_high = _bounds(base.retention, max(MAX_RETENTION, base.retention))
    retention = c1.slider(
        "Monthly retention (pp)",
        r_low,
        r_high,
        0.0,
        0.5,
        help=f"Baseline {base.retention:.1%} of last month's active customers return.",
        key=f"wi_ret_{dataset}",
    )
    aov = c2.slider(
        "Average order value (%)",
        -30.0,
        50.0,
        0.0,
        1.0,
        help=f"Baseline gross AOV {base.aov_gross:,.2f} (price or basket size).",
        key=f"wi_aov_{dataset}",
    )
    d_low, d_high = _bounds(base.discount_rate, MAX_DISCOUNT)
    discount = c3.slider(
        "Discount rate (pp)",
        d_low,
        d_high,
        0.0,
        0.5,
        help=f"Baseline discount rate {base.discount_rate:.1%} of gross sales.",
        key=f"wi_disc_{dataset}",
    )
    c4, c5, c6 = st.columns(3)
    acquisition = c4.slider(
        "New customers per month (%)",
        -50.0,
        200.0,
        0.0,
        5.0,
        help=f"Baseline {base.new_customers:,.0f} new customers per month.",
        key=f"wi_new_{dataset}",
    )
    frequency = c5.slider(
        "Orders per active customer (%)",
        -30.0,
        50.0,
        0.0,
        1.0,
        help=f"Baseline {base.orders_per_customer:.2f} orders per active customer per month.",
        key=f"wi_freq_{dataset}",
    )
    cac = c6.number_input(
        "Acquisition cost per new customer (CAC)",
        min_value=0.0,
        value=0.0,
        step=5.0,
        help="Not in the data: enter your blended acquisition cost. It drives marketing "
        "spend, contribution, LTV:CAC and payback.",
        key=f"wi_cac_{dataset}",
    )
    return Levers(
        retention_pp=retention / 100,
        aov_pct=aov / 100,
        discount_pp=discount / 100,
        new_customers_pct=acquisition / 100,
        frequency_pct=frequency / 100,
        cac=float(cac),
    )


def _delta(value: float | None) -> str | None:
    """A signed % change, hidden when it rounds to zero (zero levers = no change)."""
    if value is None or pd.isna(value) or abs(value) < 5e-4:
        return None
    return signed_pct(value)


def _number(value: float | None, *, sign: bool = False) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:+,.0f}" if sign else f"{value:,.0f}"


def _ratio(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:,.2f}{suffix}"


def render(ctx: Context) -> None:
    st.subheader(f"{MONTHS}-month growth & unit-economics simulator")
    try:
        base = state.whatif_baseline(*ctx.key)
    except ValueError as exc:
        st.warning(f"Simulator unavailable: {exc}")
        return
    st.caption(
        f"Baseline = steady state at the average of {', '.join(base.months)}: "
        f"{base.active_customers:,.0f} active and {base.new_customers:,.0f} new identified "
        f"customers per month, retention {base.retention:.1%}, net revenue "
        f"{ctx.money(base.actual_net_revenue)} per month. With every lever at zero the "
        "projection equals the baseline."
    )
    levers = _levers(base, ctx.info.dataset)
    try:
        result = simulate(base, levers)
    except ValueError as exc:
        st.warning(f"Invalid scenario: {exc}")
        return
    totals = result.totals()
    if levers.cac == 0:
        st.caption("Enter a CAC above to include marketing spend, LTV:CAC and payback.")
    base_units, scen_units = result.baseline_units, result.scenario_units

    tiles = st.columns(4)
    for col, key, label in (
        (tiles[0], "net_revenue", f"{MONTHS}-month net revenue"),
        (tiles[1], "contribution", f"{MONTHS}-month contribution"),
    ):
        col.metric(
            label,
            ctx.money(float(totals["scenario"][key])),
            _delta(float(totals["change_pct"][key])),
            help=f"Scenario vs status quo ({ctx.money(float(totals['baseline'][key]))}). "
            "Contribution = net revenue - COGS - fees - marketing.",
            border=True,
        )
    tiles[2].metric(
        "LTV : CAC",
        _ratio(scen_units.ltv_to_cac, "x"),
        _delta(change(scen_units.ltv_to_cac, base_units.ltv_to_cac)),
        help=f"Status quo {_ratio(base_units.ltv_to_cac, 'x')}. LTV = monthly margin per "
        "active customer / (1 - retention).",
        border=True,
    )
    tiles[3].metric(
        "CAC payback",
        _ratio(scen_units.payback_months, " months"),
        _delta(change(scen_units.payback_months, base_units.payback_months)),
        delta_color="inverse",
        help=f"Status quo {_ratio(base_units.payback_months, ' months')}.",
        border=True,
    )

    left, right = st.columns(2)
    for col, column, title in (
        (left, "net_revenue", "Net revenue per month"),
        (right, "contribution", "Contribution per month"),
    ):
        with col:
            st.markdown(f"**{title}**")
            st.plotly_chart(
                charts.whatif_figure(
                    result.baseline_projection[column],
                    result.scenario_projection[column],
                    ctx.pal,
                    prefix=ctx.prefix,
                ),
                key=f"whatif_{column}",
            )

    left, right = st.columns(2)
    with left:
        st.markdown(f"**{MONTHS}-month totals**")
        shown = totals.rename(index=TOTAL_LABELS)
        st.dataframe(
            pd.DataFrame(
                {
                    "status quo": shown["baseline"].map(_number),
                    "scenario": shown["scenario"].map(_number),
                    "change": shown["change"].map(lambda v: _number(v, sign=True)),
                    "change %": shown["change_pct"].map(lambda v: signed_pct(v) or "n/a"),
                }
            )
        )
    with right:
        st.markdown("**Unit economics (steady state)**")
        rows = {
            "Margin per active customer-month": "margin_per_customer_month",
            "Expected lifetime (months)": "expected_lifetime_months",
            "LTV": "ltv",
            "CAC": "cac",
            "LTV : CAC": "ltv_to_cac",
            "Payback (months)": "payback_months",
        }
        units = pd.DataFrame(
            {
                "status quo": [getattr(base_units, attr) for attr in rows.values()],
                "scenario": [getattr(scen_units, attr) for attr in rows.values()],
            },
            index=list(rows),
        )
        st.dataframe(units.map(lambda v: _ratio(None if pd.isna(v) else float(v))))
    table_view(
        "month-by-month scenario projection", result.scenario_projection.round(2).reset_index()
    )
