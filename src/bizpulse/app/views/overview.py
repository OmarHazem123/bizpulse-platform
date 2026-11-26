"""Tab 1: Executive Overview."""

from __future__ import annotations

from typing import Literal

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from bizpulse.analytics.kpis import (
    KpiSummary,
    breakdown,
    compute_kpis,
    kpi_timeseries,
    latest_growth,
    pareto,
)
from bizpulse.analytics.timeofday import hourly_profile, peak_hours
from bizpulse.app import charts, state
from bizpulse.app.views.common import (
    FREQ_LABELS,
    Context,
    change,
    pct,
    signed_pct,
    spark,
    table_view,
)
from bizpulse.ml.forecasting import ForecastResult
from bizpulse.reporting import build_report, render_markdown

DeltaColor = Literal["normal", "inverse"]
Kind = Literal["money", "count", "pct"]

#: Trend options: label -> (kpi_timeseries columns, value kind). Same unit per option.
TREND_METRICS: dict[str, tuple[tuple[str, ...], Kind]] = {
    "Net revenue & net profit": (("net_revenue", "net_profit"), "money"),
    "Gross revenue": (("gross_revenue",), "money"),
    "Orders": (("orders",), "count"),
    "Average order value": (("aov",), "money"),
    "Gross margin %": (("gross_margin_pct",), "pct"),
    "Refund rate": (("refund_rate",), "pct"),
    "New customers": (("new_customers",), "count"),
}
SERIES_LABELS: dict[str, str] = {
    "net_revenue": "Net revenue",
    "net_profit": "Net profit",
    "gross_revenue": "Gross revenue",
    "orders": "Orders",
    "aov": "Average order value",
    "gross_margin_pct": "Gross margin %",
    "refund_rate": "Refund rate",
    "new_customers": "New customers",
}
PARETO_BY: dict[str, tuple[str, str]] = {
    "Product": ("product_name", "products"),
    "Category": ("category", "categories"),
}


def with_profit(series: pd.DataFrame) -> pd.DataFrame:
    """Add per-period net profit, defined as in ``compute_kpis``.

    Net profit = net revenue x gross margin (from cost-known rows) - fees.
    """
    out = series.copy()
    out["net_profit"] = out["net_revenue"] * out["gross_margin_pct"] - out["fees"]
    return out


def _pp(current: float | None, previous: float | None) -> str | None:
    if current is None or previous is None or pd.isna(current) or pd.isna(previous):
        return None
    return f"{(current - previous) * 100:+.1f} pp"


def _tile(
    col: DeltaGenerator,
    label: str,
    value: str,
    delta: str | None,
    chart: list[float] | None,
    help_text: str,
    color: DeltaColor = "normal",
) -> None:
    col.metric(
        label,
        value,
        delta,
        delta_color=color,
        help=help_text,
        chart_data=chart if chart and len(chart) > 1 else None,
        border=True,
    )


def _kpi_tiles(ctx: Context, series: pd.DataFrame) -> None:
    cur = compute_kpis(ctx.tx, ctx.start, ctx.end)
    span = ctx.end - ctx.start
    first_day = pd.Timestamp(ctx.tx["order_ts"].min()).normalize()
    prev: KpiSummary | None = None
    if ctx.start - span >= first_day:
        prev = compute_kpis(ctx.tx, ctx.start - span, ctx.start)
    vs = f" Change vs. the previous {span.days} days." if prev else ""

    def rel(now: float | None, before: float | None) -> str | None:
        return signed_pct(change(now, before)) if prev is not None else None

    row1 = st.columns(4)
    _tile(
        row1[0],
        "Gross revenue",
        ctx.money(cur.gross_revenue),
        rel(cur.gross_revenue, prev.gross_revenue if prev else None),
        spark(series["gross_revenue"]),
        "List value of all non-cancelled sales, before discounts and refunds." + vs,
    )
    _tile(
        row1[1],
        "Net revenue",
        ctx.money(cur.net_revenue),
        rel(cur.net_revenue, prev.net_revenue if prev else None),
        spark(series["net_revenue"]),
        "Completed sales after discounts; refunds excluded." + vs,
    )
    _tile(
        row1[2],
        "Gross margin",
        pct(cur.gross_margin_pct),
        _pp(cur.gross_margin_pct, prev.gross_margin_pct) if prev else None,
        spark(series["gross_margin_pct"]),
        f"From rows with a known cost (coverage {pct(cur.cost_coverage)})." + vs,
    )
    _tile(
        row1[3],
        "Net profit",
        ctx.money(cur.net_profit),
        rel(cur.net_profit, prev.net_profit if prev else None),
        spark(series["net_profit"]),
        "Gross profit minus payment / platform fees (COGS estimated from the margin)." + vs,
    )
    row2 = st.columns(3)
    _tile(
        row2[0],
        "Orders",
        f"{cur.orders:,}",
        rel(float(cur.orders), float(prev.orders) if prev else None),
        spark(series["orders"].astype("float64")),
        "Distinct non-cancelled orders." + vs,
    )
    _tile(
        row2[1],
        "Average order value",
        ctx.money(cur.aov, 2),
        rel(cur.aov, prev.aov if prev else None),
        spark(series["aov"]),
        "(Gross - discounts) / orders." + vs,
    )
    _tile(
        row2[2],
        "Refund rate",
        pct(cur.refund_rate),
        _pp(cur.refund_rate, prev.refund_rate) if prev else None,
        spark(series["refund_rate"]),
        "Refunded value / sold value. Lower is better." + vs,
        color="inverse",
    )

    row3 = st.columns(3)
    for col, freq, label, name in (
        (row3[0], "M", "MoM net revenue growth", "month"),
        (row3[1], "W", "WoW net revenue growth", "week"),
    ):
        periods = kpi_timeseries(ctx.tx, "M" if freq == "M" else "W")
        growth = latest_growth(periods)
        if growth is None:
            _tile(col, label, "n/a", None, None, f"Needs two complete {name}s of data.")
            continue
        when = f"{growth.period:%Y-%m-%d}"
        _tile(
            col,
            label,
            signed_pct(growth.pct) or "n/a",
            None,
            spark(periods.loc[periods["complete"], "net_revenue"].tail(12)),
            f"Last complete {name} (starting {when}): {ctx.money(growth.value)} vs. "
            f"{ctx.money(growth.previous_value)} the {name} before. Uses the full history.",
        )
    forecast = _forecast(ctx)
    if forecast is None:
        _tile(row3[2], "Forecast: next 30 days", "n/a", None, None, "Needs more history.")
        return
    ratio = forecast.backtest.mae_ratio
    _tile(
        row3[2],
        "Forecast: next 30 days net revenue",
        ctx.money(float(forecast.forecast["yhat"].sum())),
        f"{(ratio - 1):+.0%} error vs seasonal naive" if ratio is not None else None,
        spark(forecast.forecast["yhat"]),
        "LightGBM forecast after the last day of data. The delta is the backtest MAE "
        "relative to a seasonal-naive baseline (negative = more accurate).",
        color="inverse",
    )


def _forecast(ctx: Context) -> ForecastResult | None:
    try:
        return state.forecast(*ctx.key)
    except ValueError:
        return None


def _trend(ctx: Context, series: pd.DataFrame) -> None:
    label = st.selectbox("Trend", list(TREND_METRICS), key="trend_metric")
    columns, kind = TREND_METRICS[label]
    prefix = ctx.prefix if kind == "money" else ""
    fmt = {"money": ",.0f", "count": ",.0f", "pct": ".1%"}[kind]
    if len(columns) == 1:
        figure = charts.trend_figure(
            series[columns[0]].astype("float64"),
            ctx.pal,
            name=label,
            prefix=prefix,
            fmt=fmt,
            partial=~series["complete"],
        )
    else:
        figure = charts.multi_trend(
            {SERIES_LABELS[c]: series[c].astype("float64") for c in columns},
            ctx.pal,
            prefix=prefix,
            fmt=fmt,
        )
    st.plotly_chart(figure, key="overview_trend")
    partial = int((~series["complete"]).sum())
    if partial:
        st.caption(
            f"{partial} partial {FREQ_LABELS[ctx.freq].lower()}(s) at the edges of the range "
            "cover only some days."
        )
    table = series[[*columns, "complete"]].reset_index()
    table_view(f"{label} per {FREQ_LABELS[ctx.freq].lower()}", table)


def _summary_download(ctx: Context) -> None:
    forecast = _forecast(ctx)
    info, tx = ctx.info, ctx.tx

    def build() -> str:
        report = build_report(info, tx, include_forecast=forecast is not None, forecast=forecast)
        return render_markdown(report)

    st.download_button(
        "Download executive summary (Markdown)",
        build,
        file_name=f"{info.dataset}_executive_summary.md",
        mime="text/markdown",
        key="exec_summary",
        icon=":material/download:",
        help="KPIs, growth, revenue concentration, segments with actions, the outlook and "
        "the top incidents over the full history (generated when you click).",
    )


def render(ctx: Context) -> None:
    part = ctx.tx_range
    if part.empty:
        st.warning("No transactions in the selected date range.")
        return
    series = with_profit(kpi_timeseries(part, ctx.freq))
    head, button = st.columns([3, 1])
    head.caption(
        f"{ctx.range_label} | {len(part):,} transaction lines | "
        f"{FREQ_LABELS[ctx.freq].lower()}ly trends"
    )
    with button:
        _summary_download(ctx)
    _kpi_tiles(ctx, series)

    st.subheader("Trend")
    _trend(ctx, series)

    left, right = st.columns(2)
    with left:
        st.subheader("Revenue concentration (Pareto 80/20)")
        by_label = st.radio(
            "Pareto by",
            list(PARETO_BY),
            horizontal=True,
            key="pareto_by",
            label_visibility="collapsed",
        )
        column, noun = PARETO_BY[by_label]
        result = pareto(part, by=column)
        st.caption(
            f"**{result.items_for_80} of {result.items_total}** {noun} "
            f"({pct(result.share_items_for_80)}) generate 80% of net revenue; the top 20% "
            f"of {noun} generate {pct(result.revenue_share_top_20pct)}."
        )
        st.plotly_chart(
            charts.pareto_figure(result.table, result.items_for_80, ctx.pal, noun),
            key="pareto",
        )
        table_view(f"Pareto by {by_label.lower()}", result.table)
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
        (("channel", "Channel"), ("category", "Category"), ("region", "Region / store")),
        strict=True,
    ):
        with col:
            table = breakdown(part, dimension)
            st.markdown(f"**Net revenue by {title.lower()}**")
            if table.empty:
                st.caption("No completed sales in this range.")
                continue
            st.plotly_chart(
                charts.ranked_bars(
                    table["net_revenue"], ctx.pal, prefix=ctx.prefix, name="net revenue"
                ),
                key=f"mix_{dimension}",
            )
            table_view(f"net revenue by {title.lower()}", table.reset_index())
