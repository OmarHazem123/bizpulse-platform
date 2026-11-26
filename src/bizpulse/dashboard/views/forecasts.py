"""Tab 3: ML Forecasts & Anomaly Alerts."""

from __future__ import annotations

from typing import Literal

import pandas as pd
import streamlit as st

from bizpulse.analytics.kpis import kpi_timeseries
from bizpulse.dashboard import charts, state
from bizpulse.dashboard.palette import severity_label
from bizpulse.dashboard.views.common import Context, pct, table_view
from bizpulse.ml.anomaly import METRICS, Alert

METRIC_LABELS: dict[str, str] = {
    "net_revenue": "Net revenue",
    "orders": "Orders",
    "aov": "Average order value",
    "refund_rate": "Refund rate",
}
ALL_SCOPES = "All alerts"
TOTAL_ONLY = "Business total only"
SLICES_ONLY = "Stores / channels only"


def _metric_format(ctx: Context, metric: str) -> tuple[str, str]:
    if metric == "refund_rate":
        return "", ".1%"
    if metric == "orders":
        return "", ",.0f"
    return ctx.prefix, ",.0f"


def _metrics_table(
    ctx: Context, model: dict[str, float | None], baseline: dict[str, float | None]
) -> pd.DataFrame:
    def show(name: str, value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.1%}" if name in {"mape", "smape"} else ctx.money(value, 0)

    labels = {"mae": "MAE", "rmse": "RMSE", "mape": "MAPE", "smape": "sMAPE"}
    return pd.DataFrame(
        [
            {
                "metric": label,
                "LightGBM": show(key, model.get(key)),
                "Seasonal naive": show(key, baseline.get(key)),
            }
            for key, label in labels.items()
        ]
    )


def _forecast(ctx: Context) -> None:
    st.subheader("30-day net revenue forecast")
    try:
        result = state.forecast(*ctx.key)
    except ValueError as exc:
        st.warning(f"Forecast unavailable: {exc}")
        return
    bt = result.backtest
    fc = result.forecast
    cols = st.columns(4)
    cols[0].metric(
        "Next 30 days (point forecast)",
        ctx.money(float(fc["yhat"].sum())),
        help="Sum of daily point forecasts. Daily 80% / 95% intervals are in the chart; "
        "they do not add up to an interval for the total.",
        border=True,
    )
    cols[1].metric(
        "Backtest MAE vs seasonal naive",
        "n/a" if bt.mae_ratio is None else f"{bt.mae_ratio:.2f}x",
        None if bt.mae_ratio is None else f"{bt.mae_ratio - 1:+.0%}",
        delta_color="inverse",
        help="Mean absolute error of the model divided by the seasonal-naive baseline's, "
        f"over {len(bt.folds)} rolling 30-day TimeSeriesSplit folds. Below 1 = better.",
        border=True,
    )
    cols[2].metric(
        "80% interval coverage",
        pct(bt.coverage80),
        help="Share of backtest days inside the 80% band, estimated leave-one-fold-out "
        "(nominal 80%).",
        border=True,
    )
    cols[3].metric(
        "95% interval coverage",
        pct(bt.coverage95),
        help="Share of backtest days inside the 95% band (nominal 95%).",
        border=True,
    )
    st.plotly_chart(
        charts.forecast_figure(result.history, fc, ctx.pal, prefix=ctx.prefix),
        key="forecast_chart",
    )
    st.caption(
        "Recursive LightGBM on the ratio to a trailing 28-day level (lags, calendar and "
        "rolling features; strictly past data only). Intervals come from out-of-fold log "
        "residuals per horizon bucket. Dashed = forecast; orange = seasonal-naive baseline."
    )
    table_view("daily forecast", fc.reset_index().round(2))

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Backtest error per fold (MAE per day)**")
        st.plotly_chart(charts.fold_bars(bt.folds, ctx.pal, ctx.prefix), key="fold_bars")
        table_view("backtest folds", bt.folds.round(2))
    with right:
        st.markdown("**Overall backtest metrics**")
        st.dataframe(_metrics_table(ctx, bt.model, bt.baseline), hide_index=True)
        st.markdown("**Top model features**")
        st.plotly_chart(
            charts.ranked_bars(
                result.feature_importance, ctx.pal, top=8, fmt=",.0f", name="splits", fold=False
            ),
            key="feature_importance",
        )


def _alert_rows(alerts: list[Alert], ctx: Context) -> pd.DataFrame:
    rows = []
    for alert in alerts:
        prefix, fmt = _metric_format(ctx, alert.metric)
        cause = alert.root_causes[0] if alert.root_causes else None
        value = f"{alert.value:.1%}" if fmt == ".1%" else f"{prefix}{alert.value:,.0f}"
        rows.append(
            {
                "date": alert.date.date(),
                "severity": severity_label(alert.severity),
                "metric": METRIC_LABELS[alert.metric],
                "scope": alert.scope,
                "direction": alert.direction,
                "value": value,
                "deviation": f"{alert.deviation:+.1%}",
                "z": round(alert.z, 1),
                "methods": ", ".join(alert.methods),
                "top cause": f"{cause.dimension}={cause.value}" if cause else "",
            }
        )
    return pd.DataFrame(rows)


def _series_for(ctx: Context, scope: str, metric: str, granularity: str) -> pd.Series[float]:
    report = state.anomalies(*ctx.key)
    if scope == "total":
        values: pd.Series[float] = report.scores[(metric, "value")].astype("float64")
        return values
    dimension, value = scope.split("=", 1)
    part = ctx.tx.loc[ctx.tx[dimension].astype(str) == value]
    freq: Literal["D", "W"] = "W" if granularity == "W" else "D"
    sliced: pd.Series[float] = kpi_timeseries(part, freq)[metric].astype("float64")
    return sliced


def _anomalies(ctx: Context) -> None:
    st.subheader("KPI anomaly alerts")
    report = state.anomalies(*ctx.key)
    weekly = report.granularity == "W"
    st.caption(
        f"{'Weekly' if weekly else 'Daily'} monitoring"
        + (" (subscription billing calendars make daily totals lumpy)" if weekly else "")
        + ". An alert needs a robust z-score or IQR breach vs. a seasonal baseline; "
        "IsolationForest is fitted on past periods only, so a stored severity never changes "
        "when new data arrives. CRITICAL = at least 30% off baseline and |z| >= 5 or all "
        "three methods agree. Store / channel slices are monitored on net revenue."
    )
    last = ctx.end - pd.Timedelta(1, unit="D")
    in_range = report.alerts_between(ctx.start, last)

    f1, f2, f3 = st.columns([2, 2, 3])
    scope_filter = f1.selectbox("Scope", [ALL_SCOPES, TOTAL_ONLY, SLICES_ONLY], key="scope")
    severities = f2.multiselect(
        "Severity", ["CRITICAL", "WARNING"], default=["CRITICAL", "WARNING"], key="severity"
    )
    metrics = f3.multiselect(
        "Metrics",
        list(METRICS),
        default=list(METRICS),
        format_func=lambda m: METRIC_LABELS[m],
        key="alert_metrics",
    )
    alerts = [
        a
        for a in in_range
        if a.severity in severities
        and a.metric in metrics
        and (scope_filter == ALL_SCOPES or (scope_filter == TOTAL_ONLY) == (a.scope == "total"))
    ]
    counts = {s: sum(1 for a in alerts if a.severity == s) for s in ("CRITICAL", "WARNING")}
    c1, c2 = st.columns(2)
    c1.metric(severity_label("CRITICAL"), counts["CRITICAL"], border=True)
    c2.metric(severity_label("WARNING"), counts["WARNING"], border=True)
    st.caption(f"Alerts dated {ctx.range_label} (sidebar date range).")

    scopes = ["total", *sorted({a.scope for a in report.alerts if a.scope != "total"})]
    t1, t2 = st.columns(2)
    scope = t1.selectbox("Timeline scope", scopes, key="timeline_scope")
    metric_options = list(METRICS) if scope == "total" else ["net_revenue"]
    metric = t2.selectbox(
        "Timeline metric", metric_options, format_func=lambda m: METRIC_LABELS[m], key="tl_metric"
    )
    values = _series_for(ctx, scope, metric, report.granularity)
    values = values.loc[(values.index >= ctx.start) & (values.index < ctx.end)]
    marks = [a for a in alerts if a.scope == scope and a.metric == metric]
    prefix, fmt = _metric_format(ctx, metric)
    st.plotly_chart(
        charts.anomaly_timeline(
            values, marks, ctx.pal, name=METRIC_LABELS[metric], prefix=prefix, fmt=fmt
        ),
        key="anomaly_timeline",
    )

    if not alerts:
        st.success("No alerts match the filters in the selected period.")
        return
    table = _alert_rows(alerts, ctx)
    st.dataframe(table, hide_index=True)
    st.markdown("**Root-cause drill-down**")
    labels = [
        f"{a.date:%Y-%m-%d} | {a.severity} | {METRIC_LABELS[a.metric]} | {a.scope} | "
        f"{a.deviation:+.0%}"
        for a in alerts
    ]
    choice = st.selectbox(
        "Alert", range(len(alerts)), format_func=lambda i: labels[i], key="alert_pick"
    )
    alert = alerts[int(choice)]
    if not alert.root_causes:
        st.info(
            f"This alert is already localized to **{alert.scope}**: the slice itself is the "
            "root cause."
            if alert.scope != "total"
            else "No single slice explains this change."
        )
        return
    quantity = {"orders": "orders", "refund_rate": "refunded value"}.get(
        alert.metric, "net revenue"
    )
    st.caption(
        f"Biggest contributors to the {alert.direction}, measured in {quantity} per slice "
        "(category, channel, region, customer type) vs. the same seasonal baseline."
    )
    st.plotly_chart(
        charts.root_cause_bars(alert, ctx.pal, "" if alert.metric == "orders" else ctx.prefix),
        key="root_causes",
    )
    table_view(
        "root causes",
        pd.DataFrame(
            [
                {
                    "dimension": c.dimension,
                    "value": c.value,
                    "actual": round(c.actual, 2),
                    "expected": round(c.expected, 2),
                    "change": round(c.delta, 2),
                    "share of change": None
                    if c.share_of_change is None
                    else round(c.share_of_change, 3),
                }
                for c in alert.root_causes
            ]
        ),
    )


def render(ctx: Context) -> None:
    _forecast(ctx)
    st.divider()
    _anomalies(ctx)
