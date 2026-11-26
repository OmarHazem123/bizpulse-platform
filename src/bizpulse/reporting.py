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
from bizpulse.ml.anomaly import Alert, detect_anomalies, granularity_for
from bizpulse.ml.forecasting import daily_revenue, forecast_series

LTV_MONTHS: tuple[int, ...] = (1, 3, 6, 12)
TOP_ITEMS = 5
RECENT_ALERT_DAYS = 30


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


def _forecast_section(tx: pd.DataFrame) -> dict[str, Any]:
    result = forecast_series(daily_revenue(tx))
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


def _anomaly_section(tx: pd.DataFrame, domain: str) -> dict[str, Any]:
    report = detect_anomalies(tx, granularity_for(domain))
    counts: dict[str, int] = {}
    for alert in report.alerts:
        kind = "total" if alert.scope == "total" else "slice"
        key = f"{kind}_{alert.severity.lower()}"
        counts[key] = counts.get(key, 0) + 1
    end = pd.Timestamp(tx["order_ts"].max()).normalize()
    start = end - pd.Timedelta(RECENT_ALERT_DAYS - 1, unit="D")
    recent = report.alerts_between(start, end)
    return {
        "granularity": report.granularity,
        "alerts_total": len(report.alerts),
        "counts": dict(sorted(counts.items())),
        "recent_window": {"start": start, "end": end},
        "recent": [_alert(a) for a in recent],
    }


def build_report(
    info: DatasetInfo,
    tx: pd.DataFrame,
    *,
    include_forecast: bool = True,
    include_anomalies: bool = True,
) -> dict[str, Any]:
    """Every headline number for one dataset, as JSON-serializable data."""
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
        report["forecast"] = _forecast_section(tx)
    if include_anomalies:
        report["anomalies"] = _anomaly_section(tx, info.domain)
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
    return "\n".join(lines)
