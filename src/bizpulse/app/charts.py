"""Pure Plotly figure builders (no Streamlit), styled by a :class:`Palette`.

Conventions (see ``dashboard/palette.py``): thin marks and 2 px lines, hairline grid,
one value axis per chart (never dual axes), one hue for ranked nominal bars, a legend
whenever two or more series share a plot, status colors only for alert severity, and a
sequential one-hue ramp for heatmaps. Every chart is paired with a table view in the UI.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go

from bizpulse.app.palette import Palette, severity_label
from bizpulse.ml.anomaly import Alert

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
OTHER = "Other"
WEEKDAYS: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def style(
    fig: go.Figure,
    pal: Palette,
    *,
    height: int = 320,
    legend: bool = True,
    hovermode: str = "closest",
) -> go.Figure:
    """Apply the shared chart chrome (transparent surface, recessive grid, top legend)."""
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 36 if legend else 12, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT, "color": pal.ink_secondary, "size": 12},
        hoverlabel={"font": {"family": FONT}},
        hovermode=hovermode,
        showlegend=legend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"color": pal.ink_secondary},
        },
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False, linecolor=pal.axis, ticks="outside", tickcolor=pal.axis
    )
    fig.update_yaxes(gridcolor=pal.grid, gridwidth=1, zeroline=False, linecolor=pal.axis)
    return fig


def _value(prefix: str, fmt: str, field: str = "y") -> str:
    return f"{prefix}%{{{field}:{fmt}}}"


# --------------------------------------------------------------------------- overview
def trend_figure(
    series: pd.Series[float],
    pal: Palette,
    *,
    name: str,
    prefix: str = "",
    fmt: str = ",.0f",
    partial: pd.Series[bool] | None = None,
) -> go.Figure:
    """One KPI over time (single series: the title names it, no legend)."""
    fig = go.Figure(
        go.Scatter(
            x=series.index,
            y=series.to_numpy(),
            mode="lines+markers" if len(series) <= 40 else "lines",
            line={"color": pal.series[0], "width": 2},
            marker={"size": 8, "color": pal.series[0], "line": {"color": pal.surface, "width": 2}},
            name=name,
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{name}: {_value(prefix, fmt)}<extra></extra>",
        )
    )
    if partial is not None and bool(partial.any()):
        part = series[partial.reindex(series.index, fill_value=False)]
        fig.add_trace(
            go.Scatter(
                x=part.index,
                y=part.to_numpy(),
                mode="markers",
                marker={"size": 10, "color": pal.surface, "line": {"color": pal.muted, "width": 2}},
                name="Partial period",
                hovertemplate="%{x|%Y-%m-%d}: partial period (data incomplete)<extra></extra>",
            )
        )
    fig.update_yaxes(tickformat=fmt, tickprefix=prefix, rangemode="tozero")
    return style(fig, pal, legend=bool(partial is not None and partial.any()))


def multi_trend(
    series: dict[str, pd.Series[float]],
    pal: Palette,
    *,
    prefix: str = "",
    fmt: str = ",.0f",
) -> go.Figure:
    """Up to four same-unit series on one axis (fixed slot order, legend always shown)."""
    fig = go.Figure()
    for slot, (name, values) in enumerate(series.items()):
        fig.add_trace(
            go.Scatter(
                x=values.index,
                y=values.to_numpy(),
                mode="lines+markers" if len(values) <= 40 else "lines",
                line={"color": pal.series[slot], "width": 2},
                marker={"size": 8, "line": {"color": pal.surface, "width": 2}},
                name=name,
                hovertemplate=f"{name}: {_value(prefix, fmt)}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line={"color": pal.axis, "width": 1})
    fig.update_yaxes(tickformat=fmt, tickprefix=prefix)
    return style(fig, pal, hovermode="x unified")


def pareto_figure(
    table: pd.DataFrame, items_for_80: int, pal: Palette, noun: str = "items"
) -> go.Figure:
    """Cumulative revenue share by item rank, with the 80 % line (one axis)."""
    fig = go.Figure(
        go.Scatter(
            x=table["rank"],
            y=table["cum_share"],
            customdata=table[["item", "share"]].to_numpy(),
            mode="lines",
            line={"color": pal.series[0], "width": 2},
            name="Cumulative share",
            hovertemplate=(
                "#%{x}: %{customdata[0]}<br>share %{customdata[1]:.1%}"
                "<br>cumulative %{y:.1%}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0.8, line={"color": pal.axis, "width": 1})
    fig.add_vline(x=items_for_80, line={"color": pal.axis, "width": 1})
    fig.add_annotation(
        x=items_for_80,
        y=0.8,
        text=f"{items_for_80} of {len(table)} {noun} = 80% of net revenue",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        xshift=6,
        yshift=-6,
        font={"color": pal.ink},
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
    fig.update_xaxes(title_text=f"{noun} ranked by net revenue")
    return style(fig, pal, legend=False)


def fold_other(values: pd.Series[float], top: int) -> pd.Series[float]:
    """Keep the *top* largest values and fold the rest into ``Other``."""
    ordered = values.sort_values(ascending=False)
    if len(ordered) <= top + 1:
        return ordered
    head = ordered.iloc[:top].copy()
    head.loc[OTHER] = float(ordered.iloc[top:].sum())
    return head


def ranked_bars(
    values: pd.Series[float],
    pal: Palette,
    *,
    top: int = 8,
    prefix: str = "",
    fmt: str = ",.0f",
    name: str = "value",
    fold: bool = True,
) -> go.Figure:
    """Ranked horizontal bars, one hue (nominal categories), largest on top.

    The tail beyond *top* is folded into ``Other`` (part-to-whole views) or, with
    ``fold=False``, simply left out (rankings such as feature importance).
    """
    ordered = values.astype("float64")
    data = fold_other(ordered, top) if fold else ordered.sort_values(ascending=False).head(top)
    labels = [str(label) for label in data.index]
    fig = go.Figure(
        go.Bar(
            x=data.to_numpy(),
            y=labels,
            orientation="h",
            marker={"color": pal.series[0], "cornerradius": 4},
            name=name,
            hovertemplate=f"%{{y}}<br>{name}: {_value(prefix, fmt, 'x')}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(
        showgrid=True,
        gridcolor=pal.grid,
        tickformat="~s" if fmt == ",.0f" else fmt,  # compact ticks; hover shows full values
        tickprefix=prefix,
        nticks=5,
    )
    fig.update_layout(bargap=0.3)
    return style(fig, pal, height=max(180, 34 * len(data) + 60), legend=False)


def heatmap(
    z: pd.DataFrame,
    pal: Palette,
    *,
    x_title: str,
    y_title: str,
    fmt: str,
    value_name: str,
    zmin: float | None = None,
    zmax: float | None = None,
    height: int = 360,
) -> go.Figure:
    """Sequential one-hue heatmap with 2 px surface gaps between cells."""
    steps = len(pal.sequential) - 1
    scale = [[i / steps, color] for i, color in enumerate(pal.sequential)]
    fig = go.Figure(
        go.Heatmap(
            z=z.to_numpy(dtype="float64"),
            x=[str(c) for c in z.columns],
            y=[str(i) for i in z.index],
            colorscale=scale,
            zmin=zmin,
            zmax=zmax,
            xgap=2,
            ygap=2,
            hoverongaps=False,
            colorbar={"thickness": 10, "outlinewidth": 0, "tickformat": fmt},
            hovertemplate=(
                f"{y_title}: %{{y}}<br>{x_title}: %{{x}}<br>{value_name}: %{{z:{fmt}}}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title_text=x_title, showline=False, ticks="")
    fig.update_yaxes(title_text=y_title, autorange="reversed", showgrid=False, showline=False)
    return style(fig, pal, height=height, legend=False)


def hour_heatmap(profile: pd.DataFrame, pal: Palette, timezone: str) -> go.Figure:
    """Orders by local weekday x hour (``hourly_profile`` output)."""
    grid = profile["orders"].unstack("hour").reindex(index=range(7), columns=range(24))
    grid.index = pd.Index(list(WEEKDAYS))
    grid.columns = pd.Index([f"{h:02d}" for h in range(24)])
    return heatmap(
        grid,
        pal,
        x_title=f"hour ({timezone})",
        y_title="weekday",
        fmt=",.0f",
        value_name="orders",
        height=300,
    )


# --------------------------------------------------------------------------- customers
def segment_bars(summary: pd.DataFrame, pal: Palette) -> go.Figure:
    """Share of customers vs share of revenue per RFM segment (one % axis)."""
    fig = go.Figure()
    for slot, (column, label) in enumerate(
        (("share_customers", "Share of customers"), ("share_revenue", "Share of net revenue"))
    ):
        fig.add_trace(
            go.Bar(
                x=summary[column],
                y=summary["segment"],
                orientation="h",
                name=label,
                marker={"color": pal.series[slot], "cornerradius": 4},
                hovertemplate=f"%{{y}}<br>{label}: %{{x:.1%}}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor=pal.grid, tickformat=".0%")
    return style(fig, pal, height=360)


def ltv_figure(ltv: pd.DataFrame, sizes: pd.Series[int], pal: Palette, prefix: str) -> go.Figure:
    """Cumulative net revenue per customer by cohort: cohorts in gray, the average in color."""
    fig = go.Figure()
    for position, (cohort, row) in enumerate(ltv.iterrows()):
        values = row.dropna()
        fig.add_trace(
            go.Scatter(
                x=values.index,
                y=values.to_numpy(),
                mode="lines",
                line={"color": pal.muted, "width": 1},
                opacity=0.55,
                name="Individual cohorts",
                legendgroup="cohorts",
                showlegend=position == 0,
                hovertemplate=(
                    f"cohort {pd.Timestamp(str(cohort)):%Y-%m}<br>month %{{x}}: "
                    f"{_value(prefix, ',.2f')}<extra></extra>"
                ),
            )
        )
    weights = ltv.notna().mul(sizes, axis=0)
    average = ltv.mul(sizes, axis=0).sum() / weights.sum().where(weights.sum() > 0)
    average = average.dropna()
    fig.add_trace(
        go.Scatter(
            x=average.index,
            y=average.to_numpy(),
            mode="lines+markers",
            line={"color": pal.series[0], "width": 2.5},
            marker={"size": 8, "line": {"color": pal.surface, "width": 2}},
            name="All cohorts (size-weighted)",
            hovertemplate=f"month %{{x}}: {_value(prefix, ',.2f')}<extra>average</extra>",
        )
    )
    last = int(max(ltv.columns)) if len(ltv.columns) else 1
    fig.update_xaxes(title_text="months since first order", dtick=1, range=[-0.3, last + 0.3])
    fig.update_yaxes(tickprefix=prefix, rangemode="tozero")
    return style(fig, pal, height=340)


# --------------------------------------------------------------------------- ML
def forecast_figure(
    history: pd.Series[float],
    forecast: pd.DataFrame,
    pal: Palette,
    *,
    days: int = 120,
    prefix: str = "",
) -> go.Figure:
    """Recent actuals, the forecast with 80 / 95 % bands and the seasonal-naive baseline."""
    recent = history.iloc[-days:]
    x = forecast.index
    fig = go.Figure()
    for upper, lower, fill, label in (
        ("hi95", "lo95", pal.band95, "95% interval"),
        ("hi80", "lo80", pal.band80, "80% interval"),
    ):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=forecast[upper],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=forecast[lower],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=fill,
                name=label,
                customdata=forecast[upper].to_numpy(),
                hovertemplate=(
                    f"{label}: {_value(prefix, ',.0f')} - {prefix}%{{customdata:,.0f}}"
                    "<extra></extra>"
                ),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent.to_numpy(),
            mode="lines",
            line={"color": pal.series[0], "width": 2},
            name="Actual",
            hovertemplate=f"actual: {_value(prefix, ',.0f')}<extra></extra>",
        )
    )
    bridge_x = [recent.index[-1], *x]
    bridge_y = [float(recent.iloc[-1]), *forecast["yhat"].to_numpy()]
    fig.add_trace(
        go.Scatter(
            x=bridge_x,
            y=bridge_y,
            mode="lines",
            line={"color": pal.series[0], "width": 2, "dash": "dash"},
            name="Forecast (LightGBM)",
            hovertemplate=f"forecast: {_value(prefix, ',.0f')}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast["baseline"],
            mode="lines",
            line={"color": pal.series[1], "width": 1.5},
            name="Seasonal naive",
            hovertemplate=f"seasonal naive: {_value(prefix, ',.0f')}<extra></extra>",
        )
    )
    fig.update_yaxes(tickprefix=prefix, rangemode="tozero")
    return style(fig, pal, height=380, hovermode="x unified")


def fold_bars(folds: pd.DataFrame, pal: Palette, prefix: str) -> go.Figure:
    """Backtest MAE per fold: model vs seasonal naive."""
    labels = [f"{pd.Timestamp(s):%b %d}" for s in folds["test_start"]]
    windows = [
        f"{pd.Timestamp(s):%Y-%m-%d} to {pd.Timestamp(e):%Y-%m-%d}"
        for s, e in zip(folds["test_start"], folds["test_end"], strict=True)
    ]
    fig = go.Figure()
    for slot, (column, label) in enumerate(
        (("model_mae", "LightGBM"), ("baseline_mae", "Seasonal naive"))
    ):
        fig.add_trace(
            go.Bar(
                x=labels,
                y=folds[column],
                name=label,
                marker={"color": pal.series[slot], "cornerradius": 4},
                customdata=windows,
                hovertemplate=(
                    f"test window %{{customdata}}<br>{label} MAE: {_value(prefix, ',.0f')}"
                    "<extra></extra>"
                ),
            )
        )
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(tickprefix=prefix, title_text="MAE per day")
    fig.update_xaxes(title_text="fold test window start (30 days each)")
    return style(fig, pal, height=300)


def anomaly_timeline(
    values: pd.Series[float],
    alerts: Sequence[Alert],
    pal: Palette,
    *,
    name: str,
    prefix: str = "",
    fmt: str = ",.0f",
) -> go.Figure:
    """The monitored series (context, gray) with alerts marked by severity."""
    fig = go.Figure(
        go.Scatter(
            x=values.index,
            y=values.to_numpy(),
            mode="lines",
            line={"color": pal.muted, "width": 1.5},
            name=name,
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{name}: {_value(prefix, fmt)}<extra></extra>",
        )
    )
    for severity, color, symbol, size in (
        ("WARNING", pal.warning, "circle", 9),
        ("CRITICAL", pal.critical, "diamond", 12),
    ):
        hits = [a for a in alerts if a.severity == severity]
        if not hits:
            continue
        fig.add_trace(
            go.Scatter(
                x=[a.date for a in hits],
                y=[a.value for a in hits],
                mode="markers",
                marker={
                    "color": color,
                    "symbol": symbol,
                    "size": size,
                    "line": {"color": pal.surface, "width": 2},
                },
                name=severity_label(severity),
                customdata=[[a.deviation, a.z, a.scope] for a in hits],
                hovertemplate=(
                    f"%{{x|%Y-%m-%d}} {severity}<br>{name}: {_value(prefix, fmt)}"
                    "<br>deviation %{customdata[0]:+.1%} (z %{customdata[1]:.1f})"
                    "<br>scope %{customdata[2]}<extra></extra>"
                ),
            )
        )
    fig.update_yaxes(tickformat=fmt, tickprefix=prefix, rangemode="tozero")
    return style(fig, pal, height=340)


def root_cause_bars(alert: Alert, pal: Palette, prefix: str) -> go.Figure:
    """Largest contributions to an alert's change, grouped by dimension."""
    causes = sorted(alert.root_causes, key=lambda c: (c.dimension, -abs(c.delta)))
    labels = [f"{c.dimension}: {c.value}" for c in causes]
    fig = go.Figure(
        go.Bar(
            x=[c.delta for c in causes],
            y=labels,
            orientation="h",
            marker={"color": pal.series[0], "cornerradius": 4},
            customdata=[
                [c.actual, c.expected, 0.0 if c.share_of_change is None else c.share_of_change]
                for c in causes
            ],
            name="change vs expected",
            hovertemplate=(
                f"%{{y}}<br>actual {prefix}%{{customdata[0]:,.1f}} vs expected "
                f"{prefix}%{{customdata[1]:,.1f}}<br>change {_value(prefix, ',.1f', 'x')} "
                "(%{customdata[2]:.0%} of the total change)<extra></extra>"
            ),
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor=pal.grid, tickprefix=prefix, zeroline=True)
    fig.update_layout(bargap=0.3)
    return style(fig, pal, height=max(200, 30 * len(causes) + 60), legend=False)


# --------------------------------------------------------------------------- what-if
def whatif_figure(
    baseline: pd.Series[float], scenario: pd.Series[float], pal: Palette, *, prefix: str
) -> go.Figure:
    """Month-by-month status quo (gray) vs scenario (accent) for one measure."""
    fig = go.Figure()
    for series, label, color, width in (
        (baseline, "Status quo", pal.muted, 2.0),
        (scenario, "Scenario", pal.series[0], 2.5),
    ):
        fig.add_trace(
            go.Scatter(
                x=[f"M{m}" for m in series.index],
                y=series.to_numpy(),
                mode="lines+markers",
                line={"color": color, "width": width},
                marker={"size": 8, "line": {"color": pal.surface, "width": 2}},
                name=label,
                hovertemplate=f"{label}: {_value(prefix, ',.0f')}<extra></extra>",
            )
        )
    fig.update_yaxes(tickprefix=prefix, tickformat="~s", rangemode="tozero")
    return style(fig, pal, height=300, hovermode="x unified")


def counts_bars(counts: Mapping[str, int], pal: Palette, *, name: str) -> go.Figure:
    """Ranked bars for a small count table (e.g. quarantine reasons)."""
    series = pd.Series({str(k): float(v) for k, v in counts.items()}, dtype="float64")
    return ranked_bars(series, pal, top=10, fmt=",.0f", name=name)
