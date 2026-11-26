"""Tab 2: Customer & Cohort Intelligence (RFM segments, playbooks, cohorts, LTV)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from bizpulse.analytics.cohorts import average_retention_curve, ltv_at
from bizpulse.analytics.rfm import PLAYBOOKS, SEGMENTS, segment_summary
from bizpulse.app import charts, state
from bizpulse.app.views.common import Context, pct, table_view

LTV_MONTHS: tuple[int, ...] = (1, 3, 6, 12)


def _month_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """String column labels (``m0``, ``m1`` ...) for table views."""
    out = frame.copy()
    out.columns = pd.Index([f"m{c}" for c in frame.columns])
    return out


def _segments(ctx: Context) -> None:
    rfm = state.rfm(*ctx.key)
    if rfm.empty:
        st.info("No identified customers in this dataset (all sales are anonymous).")
        return
    summary = segment_summary(rfm)
    st.subheader("RFM segments")
    st.caption(
        f"{len(rfm):,} identified customers scored 1-5 on recency, frequency and monetary "
        "value (quintiles) over the full history, as of the day after the last order."
    )
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(charts.segment_bars(summary, ctx.pal), key="segments")
        table = summary[
            [
                "segment",
                "customers",
                "share_customers",
                "net_revenue",
                "share_revenue",
                "avg_recency_days",
                "avg_frequency",
                "avg_monetary",
            ]
        ]
        table_view("segments", table)
    with right:
        segment = st.selectbox("Action playbook for", list(SEGMENTS), key="playbook_segment")
        playbook = PLAYBOOKS[segment]
        row = summary.loc[summary["segment"] == segment].iloc[0]
        st.markdown(f"**Priority {playbook.priority}: {playbook.goal}**")
        st.markdown("\n".join(f"- {action}" for action in playbook.actions))
        st.markdown(f"*Success metric:* {playbook.success_metric}")
        st.caption(
            f"{int(row['customers']):,} customers ({pct(row['share_customers'])}) holding "
            f"{pct(row['share_revenue'])} of identified net revenue."
        )
        members = rfm.loc[rfm["segment"] == segment].sort_values("monetary", ascending=False)
        st.download_button(
            f"Download {segment} customer list (CSV)",
            members.to_csv(index=False, lineterminator="\n").encode("utf-8"),
            file_name=f"{ctx.info.dataset}_{segment.lower().replace('/', '_').replace(' ', '_')}"
            ".csv",
            mime="text/csv",
            key="segment_download",
        )


def _cohorts(ctx: Context) -> None:
    result = state.cohorts(*ctx.key)
    if result.retention.empty:
        st.info("Not enough identified customers for cohort analysis.")
        return
    st.subheader("Monthly cohort retention")
    st.caption(
        "Share of each acquisition cohort active again N months after the first order. "
        f"Cells after {result.last_complete_month:%b %Y} (last complete month) are not "
        "yet observable and are left blank."
    )
    retention = result.retention.copy()
    retention.index = pd.Index([f"{pd.Timestamp(str(c)):%Y-%m}" for c in retention.index])
    later = retention.iloc[:, 1:]
    top = float(later.max().max()) if later.notna().to_numpy().any() else 1.0
    st.plotly_chart(
        charts.heatmap(
            retention,
            ctx.pal,
            x_title="months since first order",
            y_title="cohort",
            fmt=".0%",
            value_name="active",
            zmin=0.0,
            zmax=top,
            height=max(320, 22 * len(retention) + 80),
        ),
        key="cohort_heatmap",
    )
    table_view("cohort retention", _month_columns(retention).reset_index(names="cohort"))

    st.subheader("Customer lifetime value")
    ltv = ltv_at(result, LTV_MONTHS)
    curve = average_retention_curve(result)
    for col, months in zip(st.columns(len(LTV_MONTHS)), LTV_MONTHS, strict=True):
        retained = float(curve.loc[months]) if months in curve.index else None
        col.metric(
            f"LTV after {months} month{'s' if months > 1 else ''}",
            ctx.money(ltv[months], 2),
            help="Size-weighted cumulative net revenue per acquired customer over cohorts "
            f"observed for {months} full months. Retention at month {months}: "
            f"{pct(retained)}.",
            border=True,
        )
    st.plotly_chart(
        charts.ltv_figure(result.ltv, result.sizes, ctx.pal, ctx.prefix), key="ltv_curves"
    )
    ltv_table = result.ltv.copy()
    ltv_table.index = pd.Index([f"{pd.Timestamp(str(c)):%Y-%m}" for c in ltv_table.index])
    ltv_table = _month_columns(ltv_table)
    ltv_table.insert(0, "customers", result.sizes.to_numpy())
    table_view("cumulative LTV per customer by cohort", ltv_table.reset_index(names="cohort"))


def render(ctx: Context) -> None:
    _segments(ctx)
    _cohorts(ctx)
