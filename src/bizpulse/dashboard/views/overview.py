"""Tab 1: Executive Overview."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from bizpulse.analytics.kpis import (
    breakdown,
    compute_kpis,
    kpi_timeseries,
    latest_growth,
    pareto,
)
from bizpulse.analytics.timeofday import hourly_profile, peak_hours
from bizpulse.dashboard import charts, state
from bizpulse.dashboard.views.common import (
    FREQ_LABELS,
    Context,
    change,
    failure,
    pct,
    signed_pct,
    spark,
    table_view,
)

TREND_METRICS: dict[str, tuple[str, str]] = {
    # label -> (kpi_timeseries column, kind)
    "Net revenue": ("net_revenue", "money"),
    "Orders": ("orders", "count"),
    "Average order value": ("aov", "money"),
    "Gross margin %": ("gross_margin_pct", "pct"),
    "Refund rate": ("refund_rate", "pct"),
    "New customers": ("new_customers", "count"),
}


def _pp(current: float | None, previous: float | None) -> str | None:
    if current is None or previous is None or pd.isna(current) or pd.isna(previous):
        return None
    return f"{(current - previous) * 100:+.1f} pp"


def _kpi_tiles(ctx: Context, series: pd.DataFrame) -> None:
    cur = compute_kpis(ctx.tx, ctx.start, ctx.end)
    span = ctx.end - ctx.start
    first_day = pd.Timestamp(ctx.tx["order_ts"].min()).normalize()
    comparable = ctx.start - span >= first_day
    prev = compute_kpis(ctx.tx, ctx.start - span, ctx.start) if comparable else None
    vs = f"vs the previous {span.days} days" if comparable else "no comparable prior period"

    def delta(now: float | None, before: float | None) -> str | None:
        return signed_pct(change(now, before)) if prev is not None else None

    tiles = [
        (
            "Net revenue",
            ctx.money(cur.net_revenue),
            delta(cur.net_revenue, prev.net_revenue if prev else None),
            "normal",
            spark(series["net_revenue"]),
            "Completed sales after discounts, refunds excluded.",
        ),
        (
            "Gross margin",
            pct(cur.gross_margin_pct),
            _pp(cur.gross_margin_pct, prev.gross_margin_pct) if prev else None,
            "normal",
            spark(series["gross_margin_pct"]),
            f"From rows with a known cost (coverage {pct(cur.cost_coverage)}).",
        ),
        (
            "Net profit",
            ctx.money(cur.net_profit),
            delta(cur.net_profit, prev.net_profit if prev else None),
            "normal",
            None,
            "Gross profit minus payment / platform fees (COGS estimated from margin).",
        ),
        (
            "Orders",
            f"{cur.orders:,}",
            delta(float(cur.orders), float(prev.orders) if prev else None),
            "normal",
            spark(series["orders"].astype("float64")),
            "Distinct non-cancelled orders.",
        ),
        (
            "Average order value",
            ctx.money(cur.aov, 2),
            delta(cur.aov, prev.aov if prev else None),
            "normal",
            spark(series["aov"]),
            "(Gross - discounts) / orders.",
        ),
        (
            "Refund rate",
            pct(cur.refund_rate),
            _pp(cur.refund_rate, prev.refund_rate) if prev else None,
            "inverse",
            spark(series["refund_rate"]),
            "Refunded value / sold value. Lower is better.",
        ),
    ]
    rows = [tiles[:3], tiles[3:]]
    for row in rows:
        for col, (label, value, dlt, color, chart, help_text) in zip(
            st.columns(len(row)), row, strict=True
        ):
            col.metric(
                label,
                value,
                dlt,
                delta_color=color,  # type: ignore[arg-type]
                help=f"{help_text} Change {vs}.",
                chart_data=chart if chart and len(chart) > 1 else None,
                border=True,
            )

    monthly = kpi_timeseries(ctx.tx, "M")
    growth = latest_growth(monthly)
    left, right = st.columns(2)
    if growth is not None:
        left.metric(
            "MoM net revenue growth",
            signed_pct(growth.pct) or "n/a",
            help=(
                f"Last complete month {growth.period:%b %Y} "
                f"({ctx.money(growth.value)}) vs {growth.previous:%b %Y} "
                f"({ctx.money(growth.previous_value)}). Uses the full history."
            ),
            chart_data=spark(monthly.loc[monthly["complete"], "net_revenue"].tail(12)),
            chart_type="bar",
            border=True,
        )
    try:
        forecast = state.forecast(*ctx.key)
    except ValueError as exc:
        right.warning(f"Forecast unavailable: {exc}")
        return
    total = float(forecast.forecast["yhat"].sum())
    ratio = forecast.backtest.mae_ratio
    right.metric(
        "Forecast: next 30 days net revenue",
        ctx.money(total),
        f"{(ratio - 1):+.0%} error vs seasonal naive" if ratio is not None else None,
        delta_color="inverse",
        help="LightGBM forecast after the last day of data; the delta is the backtest MAE "
        "relative to a seasonal-naive baseline (negative = more accurate). Details in the "
        "ML Forecasts tab.",
        chart_data=spark(forecast.forecast["yhat"]),
        border=True,
    )


def _trend(ctx: Context, series: pd.DataFrame) -> None:
    label = st.selectbox("Trend metric", list(TREND_METRICS), key="trend_metric")
    column, kind = TREND_METRICS[label]
    prefix = ctx.prefix if kind == "money" else ""
    fmt = {"money": ",.0f", "count": ",.0f", "pct": ".1%"}[kind]
    values = series[column].astype("float64")
    st.plotly_chart(
        charts.trend_figure(
            values, ctx.pal, name=label, prefix=prefix, fmt=fmt, partial=~series["complete"]
        ),
        key="overview_trend",
    )
    table = series[[column, "complete"]].reset_index()
    table_view(f"{label} per {FREQ_LABELS[ctx.freq].lower()}", table)


def render(ctx: Context) -> None:
    part = ctx.tx_range
    if part.empty:
        st.warning("No transactions in the selected date range.")
        return
    series = kpi_timeseries(part, ctx.freq)
    st.caption(
        f"{ctx.range_label} | {len(part):,} transaction lines | "
        f"{FREQ_LABELS[ctx.freq].lower()}ly trends"
    )
    _kpi_tiles(ctx, series)

    st.subheader("Trend")
    _trend(ctx, series)

    left, right = st.columns(2)
    with left:
        st.subheader("Revenue concentration (Pareto)")
        result = pareto(part)
        st.caption(
            f"**{result.items_for_80} of {result.items_total}** products "
            f"({pct(result.share_items_for_80)}) generate 80% of net revenue; the top 20% "
            f"of products generate {pct(result.revenue_share_top_20pct)}."
        )
        st.plotly_chart(
            charts.pareto_figure(result.table, result.items_for_80, ctx.pal), key="pareto"
        )
        table_view("Pareto", result.table)
    with right:
        st.subheader("When customers buy")
        profile = hourly_profile(part, ctx.info.timezone)
        hours = ", ".join(f"{h:02d}:00" for h in peak_hours(part, ctx.info.timezone))
        st.caption(
            f"Orders by weekday and local hour ({ctx.info.timezone}); peak hours {hours}. "
            "Stored timestamps stay UTC."
        )
        st.plotly_chart(
            charts.hour_heatmap(profile, ctx.pal, ctx.info.timezone), key="hour_heatmap"
        )
        table_view("Orders by weekday and hour", profile.reset_index())

    st.subheader("Revenue mix")
    columns = st.columns(3)
    for col, (dimension, title) in zip(
        columns,
        (("channel", "Channel"), ("category", "Category"), ("region", "Region")),
        strict=True,
    ):
        with col:
            try:
                table = breakdown(part, dimension)
            except KeyError as exc:  # pragma: no cover - every frame has the columns
                failure(f"{title} breakdown unavailable", exc)
                continue
            st.markdown(f"**Net revenue by {title.lower()}**")
            st.plotly_chart(
                charts.ranked_bars(
                    table["net_revenue"], ctx.pal, prefix=ctx.prefix, name="net revenue"
                ),
                key=f"mix_{dimension}",
            )
            table_view(f"net revenue by {title.lower()}", table.reset_index())
