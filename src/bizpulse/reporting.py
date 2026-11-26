"""One-dataset decision report across BI and ML (CLI ``report`` / ``smoke``).

:func:`build_report` returns plain JSON-serializable data; :func:`render_text` formats it
for a terminal. The dashboard calls the analytics/ML modules directly because it needs
the frames for charts; this module is the headless equivalent.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from bizpulse.analytics.cohorts import average_retention_curve, cohort_analysis, ltv_at
from bizpulse.analytics.data import DatasetInfo, as_of
from bizpulse.analytics.kpis import (
    breakdown,
    compute_kpis,
    kpi_timeseries,
    latest_growth,
    pareto,
)
from bizpulse.analytics.rfm import PLAYBOOKS, rfm_table, segment_summary
from bizpulse.analytics.timeofday import peak_hours
from bizpulse.ml.anomaly import Alert, AnomalyReport, detect_anomalies, granularity_for
from bizpulse.ml.forecasting import ForecastResult, daily_revenue, forecast_series
from bizpulse.ml.incidents import Incident, group_incidents

LTV_MONTHS: tuple[int, ...] = (1, 3, 6, 12)
TOP_ITEMS = 5
RECENT_ALERT_DAYS = 30
TOP_INCIDENTS = 5


def jsonable(value: Any) -> Any:
    """Recursively convert pandas/numpy scalars, timestamps and NaN into JSON types."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp | datetime | date):
        return value.isoformat()
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating | float):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else number
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): v for k, v in row.items()} for row in frame.to_dict(orient="records")]


def _growth(series: pd.DataFrame) -> dict[str, Any] | None:
    growth = latest_growth(series)
    if growth is None:
        return None
    return {
        "period": growth.period,
        "previous": growth.previous,
        "value": growth.value,
        "previous_value": growth.previous_value,
        "pct": growth.pct,
    }


def _alert(alert: Alert) -> dict[str, Any]:
    cause = alert.root_causes[0] if alert.root_causes else None
    return {
        "date": alert.date,
        "metric": alert.metric,
        "scope": alert.scope,
        "severity": alert.severity,
        "direction": alert.direction,
        "value": alert.value,
        "baseline": alert.baseline,
        "deviation": alert.deviation,
        "z": alert.z,
        "methods": list(alert.methods),
        "top_cause": None
        if cause is None
        else {"dimension": cause.dimension, "value": cause.value, "delta": cause.delta},
    }


def _kpi_section(tx: pd.DataFrame) -> dict[str, Any]:
    monthly = kpi_timeseries(tx, "M")
    complete = monthly.loc[monthly["complete"]]
    last_month: dict[str, Any] | None = None
    if not complete.empty:
        start = pd.Timestamp(complete.index[-1])
        end = start + pd.offsets.MonthBegin(1)
        last_month = {"month": start.strftime("%Y-%m"), **compute_kpis(tx, start, end).as_dict()}
    return {
        "all_time": compute_kpis(tx).as_dict(),
        "last_complete_month": last_month,
        "mom_growth": _growth(monthly),
        "wow_growth": _growth(kpi_timeseries(tx, "W")),
    }


def _pareto_section(tx: pd.DataFrame) -> dict[str, Any]:
    result = pareto(tx)
    top = result.table.head(TOP_ITEMS)[["item", "net_revenue", "share", "cum_share"]]
    return {
        "items_total": result.items_total,
        "items_for_80": result.items_for_80,
        "share_items_for_80": result.share_items_for_80,
        "revenue_share_top_20pct": result.revenue_share_top_20pct,
        "top_items": _records(top),
    }


def _mix(tx: pd.DataFrame, by: str) -> list[dict[str, Any]]:
    table = breakdown(tx, by).head(TOP_ITEMS).reset_index().rename(columns={by: "value"})
    return _records(table)


def _customer_section(tx: pd.DataFrame) -> dict[str, Any]:
    summary = segment_summary(rfm_table(tx))
    segments = [
        {
            "segment": row["segment"],
            "customers": int(row["customers"]),
            "share_customers": row["share_customers"],
            "net_revenue": row["net_revenue"],
            "share_revenue": row["share_revenue"],
            "goal": row["goal"],
            "first_action": PLAYBOOKS[str(row["segment"])].actions[0],
        }
        for row in _records(summary)
    ]
    cohorts = cohort_analysis(tx)
    curve = average_retention_curve(cohorts)
    return {
        "rfm_segments": segments,
        "cohorts": len(cohorts.sizes),
        "ltv": {f"m{k}": v for k, v in ltv_at(cohorts, LTV_MONTHS).items()},
        "retention": {
            f"m{k}": float(curve.loc[k]) if k in curve.index else None for k in LTV_MONTHS
        },
    }


def _forecast_section(tx: pd.DataFrame, result: ForecastResult | None) -> dict[str, Any]:
    result = result if result is not None else forecast_series(daily_revenue(tx))
    bt = result.backtest
    frame = result.forecast.reset_index()
    return {
        "horizon_days": len(frame),
        "next_30d_point_total": float(result.forecast["yhat"].sum()),
        "next_30d_baseline_total": float(result.forecast["baseline"].sum()),
        "backtest": {
            "folds": len(bt.folds),
            "model": bt.model,
            "seasonal_naive": bt.baseline,
            "mae_ratio": bt.mae_ratio,
            "coverage80": bt.coverage80,
            "coverage95": bt.coverage95,
        },
        "daily": _records(frame),
    }


def _incident(incident: Incident) -> dict[str, Any]:
    head = incident.headline
    cause = head.root_causes[0] if head.root_causes else None
    return {
        "start": incident.start,
        "end": incident.end,
        "severity": incident.severity,
        "company_wide": incident.company_wide,
        "alerts": len(incident.alerts),
        "metrics": list(incident.metrics),
        "slices": list(incident.slices),
        "headline": {
            "metric": head.metric,
            "scope": head.scope,
            "direction": head.direction,
            "deviation": head.deviation,
        },
        "top_cause": None if cause is None else f"{cause.dimension}={cause.value}",
    }


def _anomaly_section(tx: pd.DataFrame, domain: str, report: AnomalyReport | None) -> dict[str, Any]:
    if report is None:
        report = detect_anomalies(tx, granularity_for(domain))
    counts: dict[str, int] = {}
    for alert in report.alerts:
        kind = "total" if alert.scope == "total" else "slice"
        key = f"{kind}_{alert.severity.lower()}"
        counts[key] = counts.get(key, 0) + 1
    end = pd.Timestamp(tx["order_ts"].max()).normalize()
    start = end - pd.Timedelta(RECENT_ALERT_DAYS - 1, unit="D")
    recent = report.alerts_between(start, end)
    incidents = group_incidents(report.alerts, report.granularity)
    return {
        "granularity": report.granularity,
        "alerts_total": len(report.alerts),
        "counts": dict(sorted(counts.items())),
        "recent_window": {"start": start, "end": end},
        "recent": [_alert(a) for a in recent],
        "incidents_total": len(incidents),
        "top_incidents": [_incident(i) for i in incidents[:TOP_INCIDENTS]],
    }


def build_report(
    info: DatasetInfo,
    tx: pd.DataFrame,
    *,
    include_forecast: bool = True,
    include_anomalies: bool = True,
    forecast: ForecastResult | None = None,
    anomalies: AnomalyReport | None = None,
) -> dict[str, Any]:
    """Every headline number for one dataset, as JSON-serializable data.

    Already computed *forecast* / *anomalies* results (e.g. cached by the dashboard) are
    reused instead of being recomputed.
    """
    if tx.empty:
        raise ValueError(f"dataset {info.dataset!r} has no transactions")
    report: dict[str, Any] = {
        "dataset": {
            "name": info.dataset,
            "display_name": info.display_name,
            "domain": info.domain,
            "currency": info.currency,
            "timezone": info.timezone,
        },
        "coverage": {
            "first_order": tx["order_ts"].min(),
            "last_order": tx["order_ts"].max(),
            "as_of": as_of(tx),
            "transactions": len(tx),
        },
        "kpis": _kpi_section(tx),
        "pareto": _pareto_section(tx),
        "mix": {"channel": _mix(tx, "channel"), "category": _mix(tx, "category")},
        "customers": _customer_section(tx),
        "peak_local_hours": peak_hours(tx, info.timezone),
    }
    if include_forecast:
        report["forecast"] = _forecast_section(tx, forecast)
    if include_anomalies:
        report["anomalies"] = _anomaly_section(tx, info.domain, anomalies)
    result: dict[str, Any] = jsonable(report)
    return result


# --------------------------------------------------------------------------- text
def _money(value: float | None, currency: str) -> str:
    if value is None:
        return "n/a"
    return f"{currency} {value:,.2f}" if abs(value) < 1000 else f"{currency} {value:,.0f}"


def _pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}%}"


def _signed_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1%}"


def render_text(report: dict[str, Any]) -> str:
    """A compact, human-readable rendering of :func:`build_report` output."""
    ds = report["dataset"]
    cur = ds["currency"]
    cov = report["coverage"]
    k = report["kpis"]["all_time"]
    lines = [
        f"{ds['display_name']} [{ds['name']}, {ds['domain']}, {cur}, tz {ds['timezone']}]",
        f"data: {cov['first_order'][:10]} -> {cov['last_order'][:10]}, "
        f"{cov['transactions']:,} transactions",
        "",
        "KPIs (all time)",
        f"  net revenue {_money(k['net_revenue'], cur)} | gross margin "
        f"{_pct(k['gross_margin_pct'])} | net profit {_money(k['net_profit'], cur)}",
        f"  orders {k['orders']:,} | AOV {_money(k['aov'], cur)} | refund rate "
        f"{_pct(k['refund_rate'])} | customers {k['active_customers']:,}",
    ]
    month = report["kpis"]["last_complete_month"]
    if month:
        lines.append(
            f"  last complete month {month['month']}: net {_money(month['net_revenue'], cur)}, "
            f"orders {month['orders']:,}"
        )
    for label, key in (("MoM", "mom_growth"), ("WoW", "wow_growth")):
        growth = report["kpis"][key]
        if growth:
            lines.append(
                f"  {label} net revenue growth {_signed_pct(growth['pct'])} "
                f"({growth['period'][:10]} vs {growth['previous'][:10]})"
            )
    p = report["pareto"]
    lines += [
        "",
        f"Pareto: {p['items_for_80']} of {p['items_total']} products "
        f"({_pct(p['share_items_for_80'])}) make 80% of net revenue; the top 20% make "
        f"{_pct(p['revenue_share_top_20pct'])}",
        "",
        "RFM segments (customers, revenue share -> first action)",
    ]
    for seg in report["customers"]["rfm_segments"]:
        lines.append(
            f"  {seg['segment']:<20} {seg['customers']:>6,}  {_pct(seg['share_revenue']):>6}"
            f"  -> {seg['first_action']}"
        )
    ltv = report["customers"]["ltv"]
    ret = report["customers"]["retention"]
    lines.append(
        "  LTV per customer: "
        + ", ".join(f"{m} {_money(ltv[m], cur)}" for m in ltv)
        + " | retention: "
        + ", ".join(f"{m} {_pct(ret[m])}" for m in ret)
    )
    hours = ", ".join(f"{h:02d}:00" for h in report["peak_local_hours"])
    lines.append(f"  peak local hours ({ds['timezone']}): {hours}")
    if "forecast" in report:
        f = report["forecast"]
        bt = f["backtest"]
        lines += [
            "",
            f"Forecast (next {f['horizon_days']} days): {_money(f['next_30d_point_total'], cur)} "
            f"(seasonal naive {_money(f['next_30d_baseline_total'], cur)})",
            f"  backtest {bt['folds']} folds: MAE {bt['model']['mae']:,.1f} vs naive "
            f"{bt['seasonal_naive']['mae']:,.1f} (ratio {bt['mae_ratio']:.2f}); "
            f"coverage 80% {_pct(bt['coverage80'])}, 95% {_pct(bt['coverage95'])}",
        ]
    if "anomalies" in report:
        a = report["anomalies"]
        counts = ", ".join(f"{key} {n}" for key, n in a["counts"].items()) or "none"
        window = a["recent_window"]
        lines += [
            "",
            f"Anomalies ({'weekly' if a['granularity'] == 'W' else 'daily'} monitoring): "
            f"{a['alerts_total']} alerts ({counts})",
            f"  last {RECENT_ALERT_DAYS} days ({window['start'][:10]} -> {window['end'][:10]}):",
        ]
        for alert in a["recent"] or []:
            cause = alert["top_cause"]
            why = f" <- {cause['dimension']}={cause['value']}" if cause else ""
            lines.append(
                f"    {alert['date'][:10]} {alert['severity']:<8} {alert['metric']:<12} "
                f"{alert['scope']:<22} {alert['direction']} {_signed_pct(alert['deviation'])}"
                f"{why}"
            )
        if not a["recent"]:
            lines.append("    no alerts")
        lines.append(
            f"  {a['incidents_total']} incidents after grouping adjacent alerts; most important:"
        )
        for inc in a["top_incidents"]:
            lines.append(f"    {_incident_line(inc)}")
    return "\n".join(lines)


def _incident_line(inc: dict[str, Any]) -> str:
    head = inc["headline"]
    if inc["start"] == inc["end"]:
        span = inc["start"][:10]
    else:
        span = f"{inc['start'][:10]} -> {inc['end'][:10]}"
    where = "company-wide" if inc["company_wide"] else "local: " + ", ".join(inc["slices"])
    cause = f", top cause {inc['top_cause']}" if inc["top_cause"] else ""
    return (
        f"{inc['severity']} {span} ({where}): {head['metric']} {head['direction']} "
        f"{_signed_pct(head['deviation'])}, {inc['alerts']} alert"
        f"{'' if inc['alerts'] == 1 else 's'}{cause}"
    )


def render_markdown(report: dict[str, Any]) -> str:
    """An executive summary in Markdown (dashboard download, ``report --markdown``)."""
    ds = report["dataset"]
    cur = ds["currency"]
    cov = report["coverage"]
    k = report["kpis"]["all_time"]
    out = [
        f"# Executive summary: {ds['display_name']}",
        "",
        f"Dataset `{ds['name']}` ({ds['domain']}, {cur}, timezone {ds['timezone']}). Data from "
        f"{cov['first_order'][:10]} to {cov['last_order'][:10]}, "
        f"{cov['transactions']:,} transaction lines.",
        "",
        "## Key figures (all time)",
        "",
        "| KPI | Value |",
        "| --- | ---: |",
        f"| Gross revenue | {_money(k['gross_revenue'], cur)} |",
        f"| Net revenue | {_money(k['net_revenue'], cur)} |",
        f"| Gross margin | {_pct(k['gross_margin_pct'])} |",
        f"| Net profit | {_money(k['net_profit'], cur)} |",
        f"| Orders | {k['orders']:,} |",
        f"| Average order value | {_money(k['aov'], cur)} |",
        f"| Refund rate | {_pct(k['refund_rate'])} |",
        f"| Active identified customers | {k['active_customers']:,} |",
    ]
    for label, key in (("Month over month", "mom_growth"), ("Week over week", "wow_growth")):
        growth = report["kpis"][key]
        if growth:
            out.append(
                f"| {label} net revenue growth ({growth['period'][:10]}) | "
                f"{_signed_pct(growth['pct'])} |"
            )
    p = report["pareto"]
    out += [
        "",
        "## Revenue concentration",
        "",
        f"{p['items_for_80']} of {p['items_total']} products ({_pct(p['share_items_for_80'])}) "
        "generate 80% of net revenue. Top products:",
        "",
        "| Product | Net revenue | Share |",
        "| --- | ---: | ---: |",
        *(
            f"| {item['item']} | {_money(item['net_revenue'], cur)} | {_pct(item['share'])} |"
            for item in p["top_items"]
        ),
        "",
        "## Customers",
        "",
        "| Segment | Customers | Revenue share | First action |",
        "| --- | ---: | ---: | --- |",
        *(
            f"| {seg['segment']} | {seg['customers']:,} | {_pct(seg['share_revenue'])} | "
            f"{seg['first_action']} |"
            for seg in report["customers"]["rfm_segments"]
        ),
        "",
        "LTV per acquired customer: "
        + ", ".join(
            f"{m[1:]} months {_money(v, cur)}" for m, v in report["customers"]["ltv"].items()
        )
        + ".",
    ]
    if "forecast" in report:
        f = report["forecast"]
        bt = f["backtest"]
        out += [
            "",
            "## Outlook",
            "",
            f"Forecast net revenue for the next {f['horizon_days']} days: "
            f"**{_money(f['next_30d_point_total'], cur)}** (seasonal naive "
            f"{_money(f['next_30d_baseline_total'], cur)}). Backtest MAE ratio vs. seasonal "
            f"naive {bt['mae_ratio']:.2f}; 80% interval coverage {_pct(bt['coverage80'])}.",
        ]
    if "anomalies" in report:
        a = report["anomalies"]
        out += ["", "## Incidents to review", ""]
        out += [f"- {_incident_line(inc)}" for inc in a["top_incidents"]] or ["- none"]
    out += ["", "_Generated by BizPulse._", ""]
    return "\n".join(out)
