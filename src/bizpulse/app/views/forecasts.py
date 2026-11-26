"""Tab 3: ML Forecasts & Anomaly Alerts."""

from __future__ import annotations

from typing import Literal

import pandas as pd
import streamlit as st

from bizpulse.analytics.kpis import kpi_timeseries
from bizpulse.app import charts, state
from bizpulse.app.palette import severity_label
from bizpulse.app.views.common import Context, pct, table_view
from bizpulse.ml.anomaly import METRICS, Alert
from bizpulse.ml.incidents import Incident, group_incidents

METRIC_LABELS: dict[str, str] = {
    "net_revenue": "Net revenue",
    "orders": "Orders",
    "aov": "Average order value",
    "refund_rate": "Refund rate",
}
SLICE_LABELS: dict[str, str] = {"region": "Stores / regions", "channel": "Channels"}


def _metric_format(ctx: Context, metric: str) -> tuple[str, str]:
    if metric == "refund_rate":
        return "", ".1%"
    if metric == "orders":
        return "", ",.0f"
    return ctx.prefix, ",.0f"


# --------------------------------------------------------------------------- forecast
def _metrics_table(
    ctx: Context, model: dict[str, float | None], baseline: dict[str, float | None]
) -> pd.DataFrame:
    def show(name: str, value: float | None) -> str:
        if value is None or pd.isna(value):
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
        st.warning(f"Forecast unavailable: {exc}. It needs several months of daily history.")
        return
    bt = result.backtest
    fc = result.forecast
    cols = st.columns(4)
    cols[0].metric(
        "Next 30 days (point forecast)",
        ctx.money(float(fc["yhat"].sum())),
        help="Sum of the daily point forecasts. The daily 80% / 95% intervals are in the "
        "chart; they do not add up to an interval for the total.",
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


# --------------------------------------------------------------------------- anomalies
def _value_text(ctx: Context, alert: Alert) -> str:
    prefix, fmt = _metric_format(ctx, alert.metric)
    return f"{alert.value:.1%}" if fmt == ".1%" else f"{prefix}{alert.value:,.0f}"


def _when(incident: Incident) -> str:
    if incident.start == incident.end:
        return f"{incident.start:%Y-%m-%d}"
    return f"{incident.start:%Y-%m-%d} to {incident.end:%Y-%m-%d}"


def incident_rows(incidents: list[Incident]) -> pd.DataFrame:
    """The severity-ranked incident feed as a table."""
    rows = []
    for rank, incident in enumerate(incidents, start=1):
        head = incident.headline
        cause = head.root_causes[0] if head.root_causes else None
        rows.append(
            {
                "rank": rank,
                "severity": severity_label(incident.severity),
                "when": _when(incident),
                "reach": "Company-wide" if incident.company_wide else "Local",
                "headline": f"{METRIC_LABELS[head.metric]} {head.direction} "
                f"{head.deviation:+.0%} ({head.scope})",
                "metrics": ", ".join(METRIC_LABELS[m] for m in incident.metrics),
                "stores / channels": ", ".join(incident.slices) or "-",
                "alerts": len(incident.alerts),
                "top cause": f"{cause.dimension}={cause.value}" if cause else "",
            }
        )
    return pd.DataFrame(rows)


def _alert_rows(alerts: tuple[Alert, ...] | list[Alert], ctx: Context) -> pd.DataFrame:
    rows = []
    for alert in alerts:
        cause = alert.root_causes[0] if alert.root_causes else None
        rows.append(
            {
                "date": alert.date.date(),
                "severity": severity_label(alert.severity),
                "metric": METRIC_LABELS[alert.metric],
                "scope": alert.scope,
                "direction": alert.direction,
                "value": _value_text(ctx, alert),
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


def _filters(ctx: Context, all_alerts: list[Alert]) -> list[Alert]:
    """Severity, metric, company-wide and store / channel filters (one row each)."""
    f1, f2 = st.columns([2, 3])
    severities = f1.multiselect(
        "Severity", ["CRITICAL", "WARNING"], default=["CRITICAL", "WARNING"], key="severity"
    )
    metrics = f2.multiselect(
        "Metrics",
        list(METRICS),
        default=list(METRICS),
        format_func=lambda m: METRIC_LABELS[m],
        key="alert_metrics",
    )
    slice_values: dict[str, list[str]] = {
        dim: sorted({a.scope_value or "" for a in all_alerts if a.scope_dimension == dim} - {""})
        for dim in SLICE_LABELS
    }
    g1, g2, g3 = st.columns([2, 3, 3])
    include_total = g1.toggle("Company-wide alerts", value=True, key="alert_total")
    picked: dict[str, set[str]] = {}
    for col, (dim, label) in zip((g2, g3), SLICE_LABELS.items(), strict=True):
        options = slice_values[dim]
        picked[dim] = set(
            col.multiselect(
                label,
                options,
                default=options,
                key=f"alert_{dim}",
                disabled=not options,
                placeholder="no slice alerts" if not options else "none selected",
            )
        )

    def keep(alert: Alert) -> bool:
        if alert.severity not in severities or alert.metric not in metrics:
            return False
        if alert.scope == "total":
            return include_total
        return (alert.scope_value or "") in picked.get(alert.scope_dimension or "", set())

    return [a for a in all_alerts if keep(a)]


def _drilldown(ctx: Context, incidents: list[Incident]) -> None:
    labels = [
        f"#{rank} {severity_label(i.severity)} | {_when(i)} | "
        f"{'company-wide' if i.company_wide else 'local'} | {len(i.alerts)} alerts"
        for rank, i in enumerate(incidents, start=1)
    ]
    pick = st.selectbox(
        "Incident", range(len(incidents)), format_func=lambda i: labels[i], key="incident_pick"
    )
    incident = incidents[int(pick)]
    st.dataframe(_alert_rows(incident.alerts, ctx), hide_index=True)
    with_causes = [a for a in incident.alerts if a.root_causes]
    if not with_causes:
        st.info(
            f"Localized to **{', '.join(incident.slices)}**: the store / channel itself is the "
            "root cause."
            if incident.slices
            else "No single slice explains this change."
        )
        return
    alert = with_causes[0]
    quantity = {"orders": "orders", "refund_rate": "refunded value"}.get(
        alert.metric, "net revenue"
    )
    st.caption(
        f"Root causes of the {alert.direction} in {METRIC_LABELS[alert.metric].lower()} on "
        f"{alert.date:%Y-%m-%d}, measured in {quantity} per slice (category, channel, region, "
        "customer type) vs. the same seasonal baseline."
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


def _timeline(ctx: Context, alerts: list[Alert], granularity: str, scopes: list[str]) -> None:
    t1, t2 = st.columns(2)
    scope = t1.selectbox("Timeline scope", scopes, key="timeline_scope")
    metric_options = list(METRICS) if scope == "total" else ["net_revenue"]
    metric = t2.selectbox(
        "Timeline metric", metric_options, format_func=lambda m: METRIC_LABELS[m], key="tl_metric"
    )
    values = _series_for(ctx, scope, metric, granularity)
    values = values.loc[(values.index >= ctx.start) & (values.index < ctx.end)]
    marks = [a for a in alerts if a.scope == scope and a.metric == metric]
    prefix, fmt = _metric_format(ctx, metric)
    st.plotly_chart(
        charts.anomaly_timeline(
            values, marks, ctx.pal, name=METRIC_LABELS[metric], prefix=prefix, fmt=fmt
        ),
        key="anomaly_timeline",
    )


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
        "three methods agree. Stores and channels are monitored on net revenue."
    )
    last = ctx.end - pd.Timedelta(1, unit="D")
    in_range = report.alerts_between(ctx.start, last)
    alerts = _filters(ctx, in_range)
    incidents = group_incidents(alerts, report.granularity)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Incidents",
        len(incidents),
        help="Alerts on adjacent periods are grouped into one incident, so a broad event "
        "(e.g. an outage hitting every store and several metrics) reads as one line.",
        border=True,
    )
    c2.metric("Company-wide incidents", sum(1 for i in incidents if i.company_wide), border=True)
    c3.metric(
        severity_label("CRITICAL"), sum(a.severity == "CRITICAL" for a in alerts), border=True
    )
    c4.metric(severity_label("WARNING"), sum(a.severity == "WARNING" for a in alerts), border=True)
    st.caption(f"Alerts dated {ctx.range_label} (sidebar date range).")

    scopes = ["total", *sorted({a.scope for a in report.alerts if a.scope != "total"})]
    _timeline(ctx, alerts, report.granularity, scopes)

    if not incidents:
        st.success("No alerts match the filters in the selected period.")
        return
    st.markdown("**Incident feed** (ranked by severity, then company-wide before local, then size)")
    st.dataframe(incident_rows(incidents), hide_index=True)
    st.markdown("**Incident drill-down**")
    _drilldown(ctx, incidents)


def render(ctx: Context) -> None:
    _forecast(ctx)
    st.divider()
    _anomalies(ctx)
