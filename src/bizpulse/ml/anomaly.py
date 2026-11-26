"""KPI anomaly detection with severity and root-cause attribution (TR-ML-04, §7.1).

Rules (TECHNICAL-REQUIREMENTS.md §7.1, incl. amendments A1 and A2):

* Periods are days (``D``), or ISO weeks (``W``) for subscription domains, whose
  billing calendars make daily revenue structurally lumpy.
* Baselines exclude the current period. Daily revenue and orders use the median of the
  same weekday over the previous 4 weeks. Weekly metrics and daily rate metrics use
  the trailing median.
* Two-tailed scoring (A1): spikes on ``dev = (x - b) / b``, drops on
  ``log1p(x) - log1p(b)``. Robust z (median / MAD) over trailing history only.
* Methods: ``|z| >= 3.5``, a 3 x IQR fence, and an **expanding-window IsolationForest**
  (A2). It is refit every ``refit_every`` periods on strictly earlier periods only, and
  flags a period whose score falls below the 1st percentile of past scores.
* Severity (A2): CRITICAL iff ``|dev| >= 30 %`` and (``|z| >= 5`` or all three methods
  agree). Otherwise WARNING. Severity of a past period never changes when data is
  appended.
* Slice monitoring (A2): net revenue per store/region and per channel is scored with
  the same rules, so a localized incident is caught even when the total barely moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import IsolationForest

from bizpulse.analytics.kpis import first_orders, kpi_timeseries, period_start

Granularity = Literal["D", "W"]
FloatArray = npt.NDArray[np.float64]

METRICS: tuple[str, ...] = ("net_revenue", "orders", "aov", "refund_rate")
VOLUME_METRICS = frozenset({"net_revenue", "orders"})
RATE_METRICS = frozenset({"aov", "refund_rate"})
Z_THRESHOLD = 3.5
CRITICAL_Z = 5.0
IQR_K = 3.0
CRITICAL_MIN_DEV = 0.30
IFOREST_QUANTILE = 0.01
SEED = 42
ROOT_CAUSE_DIMENSIONS: tuple[str, ...] = ("category", "channel", "region", "customer_type")
SLICE_DIMENSIONS: tuple[str, ...] = ("region", "channel")
MIN_SLICE_SHARE = 0.05


@dataclass(frozen=True)
class Rules:
    """Per-granularity windows (counted in periods)."""

    volume_lags: tuple[int, ...]  # periods back used for the volume baseline
    rate_window: int  # trailing periods for rate baselines
    history: int  # trailing periods for the robust z / IQR scale
    warmup: int  # minimum scored history
    refit_every: int  # IsolationForest refit cadence
    min_train: int  # minimum IsolationForest training periods
    min_orders: int  # volume guard for rate metrics
    min_slice_orders: float  # trailing-median orders a slice needs to be monitored


RULES: dict[Granularity, Rules] = {
    "D": Rules((7, 14, 21, 28), 28, 56, 28, 28, 56, 20, 10.0),
    "W": Rules((1, 2, 3, 4), 4, 26, 8, 4, 16, 50, 30.0),
}


def granularity_for(domain: str) -> Granularity:
    """Subscription businesses are monitored weekly (billing calendars), others daily."""
    return "W" if domain == "saas" else "D"


@dataclass(frozen=True)
class RootCause:
    dimension: str
    value: str
    actual: float
    expected: float
    delta: float
    share_of_change: float | None


@dataclass(frozen=True)
class Alert:
    date: pd.Timestamp
    metric: str
    value: float
    baseline: float
    deviation: float
    z: float
    methods: tuple[str, ...]
    severity: str
    direction: str
    root_causes: tuple[RootCause, ...] = field(default=())
    scope_dimension: str | None = None
    scope_value: str | None = None

    @property
    def scope(self) -> str:
        if self.scope_dimension is None:
            return "total"
        return f"{self.scope_dimension}={self.scope_value}"

    def top_cause(self, dimension: str) -> RootCause | None:
        return next((c for c in self.root_causes if c.dimension == dimension), None)


@dataclass
class AnomalyReport:
    granularity: Granularity
    scores: pd.DataFrame  # MultiIndex columns (metric, field)
    alerts: list[Alert]

    def alerts_between(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        metric: str | None = None,
        *,
        scope: str | None = None,
    ) -> list[Alert]:
        """Alerts whose period overlaps ``[start, end]`` (weekly periods start on Monday)."""
        if self.granularity == "W":
            start = start - pd.Timedelta(int(start.dayofweek), unit="D")
        return [
            a
            for a in self.alerts
            if start <= a.date <= end
            and (metric is None or a.metric == metric)
            and (scope is None or a.scope == scope)
        ]


# --------------------------------------------------------------------------- scoring
def _baseline(values: FloatArray, t: int, lags: tuple[int, ...] | None, window: int) -> float:
    if lags is not None:
        picks = [values[t - k] for k in lags if t - k >= 0]
        need = max(2, len(lags) - 1)
    else:
        picks = list(values[max(0, t - window) : t])
        need = max(2, window // 4)
    clean = [v for v in picks if not np.isnan(v)]
    return float(np.median(clean)) if len(clean) >= need else float("nan")


def score_metric(
    series: pd.Series[float],
    *,
    volume: bool,
    rules: Rules = RULES["D"],
    valid: pd.Series[bool] | None = None,
) -> pd.DataFrame:
    """Baseline, deviation, robust z and method flags per period (trailing windows only)."""
    values = np.array(series.to_numpy(), dtype=np.float64)
    if valid is not None:
        values[~valid.to_numpy(dtype=bool)] = np.nan
    n = len(values)
    lags = rules.volume_lags if volume else None
    baseline = np.asarray(
        [_baseline(values, t, lags, rules.rate_window) for t in range(n)], dtype=np.float64
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.where(baseline > 0, (values - baseline) / baseline, np.nan)
    if volume:
        log_dev = np.log1p(np.maximum(values, 0.0)) - np.log1p(np.maximum(baseline, 0.0))
        log_dev[np.isnan(dev)] = np.nan
    else:
        log_dev = dev.copy()
    z = np.full(n, np.nan)
    z_flag = np.zeros(n, dtype=bool)
    iqr_flag = np.zeros(n, dtype=bool)
    for t in range(n):
        if np.isnan(dev[t]):
            continue
        stat = dev if dev[t] >= 0 else log_dev  # A1: two-tailed statistic
        hist = stat[max(0, t - rules.history) : t]
        hist = hist[~np.isnan(hist)]
        if len(hist) < rules.warmup:
            continue
        center = float(np.median(hist))
        mad = float(np.median(np.abs(hist - center)))
        scale = 1.4826 * mad if mad > 0 else float(np.std(hist)) or 1e-9
        z[t] = (stat[t] - center) / scale
        z_flag[t] = abs(z[t]) >= Z_THRESHOLD
        q1, q3 = np.percentile(hist, [25, 75])
        spread = q3 - q1
        iqr_flag[t] = bool(stat[t] < q1 - IQR_K * spread or stat[t] > q3 + IQR_K * spread)
    return pd.DataFrame(
        {
            "value": series.to_numpy(dtype=np.float64),
            "baseline": baseline,
            "dev": dev,
            "z": z,
            "z_flag": z_flag,
            "iqr_flag": iqr_flag,
        },
        index=series.index,
    )


def _trailing_median(series: pd.Series[float], rules: Rules) -> pd.Series[float]:
    window = rules.rate_window if rules.volume_lags[0] == 1 else 28
    trailing: pd.Series[float] = (
        series.shift(1).rolling(window, min_periods=max(2, window // 4)).median()
    )
    return trailing


def _volume_guard(orders: pd.Series[float], rules: Rules) -> pd.Series[bool]:
    """Rate metrics are scored only on periods with enough orders (no look-ahead)."""
    trailing = _trailing_median(orders, rules).fillna(0.0)
    return (orders >= rules.min_orders) & (orders >= 0.25 * trailing)


def expanding_iforest(z_matrix: FloatArray, rules: Rules) -> npt.NDArray[np.bool_]:
    """IsolationForest flags where period t is scored by a model fit on periods < refit <= t.

    Refit points sit on a fixed cadence from the series start, so appending data never
    changes the model, score or threshold used for an earlier period.
    """
    n = len(z_matrix)
    flags = np.zeros(n, dtype=bool)
    usable = ~np.isnan(z_matrix).all(axis=1)
    features = np.nan_to_num(z_matrix)
    for refit in range(rules.min_train, n, rules.refit_every):
        train = np.flatnonzero(usable[:refit])
        if len(train) < rules.min_train // 2:
            continue
        forest = IsolationForest(n_estimators=100, random_state=SEED)
        forest.fit(features[train])
        threshold = float(np.quantile(forest.score_samples(features[train]), IFOREST_QUANTILE))
        block = np.arange(refit, min(n, refit + rules.refit_every))
        block = block[usable[block]]
        if len(block):
            flags[block] = forest.score_samples(features[block]) < threshold
    return flags


def severity_of(z: float, dev: float, z_hit: bool, iqr_hit: bool, forest_hit: bool) -> str | None:
    """A2 calibration: noise cannot become CRITICAL through a single corroborating method."""
    if not (z_hit or iqr_hit):
        return None
    strong = abs(z) >= CRITICAL_Z or (z_hit and iqr_hit and forest_hit)
    return "CRITICAL" if strong and abs(dev) >= CRITICAL_MIN_DEV else "WARNING"


# --------------------------------------------------------------------------- root cause
def _with_customer_type(tx: pd.DataFrame) -> pd.DataFrame:
    firsts = first_orders(tx)
    out = tx.copy()
    first_ts = out["customer_id"].map(firsts)
    is_new = out["order_ts"].dt.normalize() == first_ts.dt.normalize()
    out["customer_type"] = np.where(
        out["is_anonymous"], "anonymous", np.where(is_new, "new", "returning")
    )
    return out


def _slice_table(
    tx: pd.DataFrame, dimension: str, metric: str, granularity: Granularity
) -> pd.DataFrame:
    """Period x slice values of the attributable quantity for *metric*."""
    if metric == "orders":
        rows = tx.loc[tx["status"].ne("cancelled")]
    elif metric == "refund_rate":
        rows = tx.loc[tx["status"].eq("refunded")]
    else:  # net_revenue and aov are attributed through revenue
        rows = tx.loc[tx["status"].eq("completed")]
    period = period_start(rows["order_ts"], granularity)
    if metric == "orders":
        table = rows.groupby([period, rows[dimension]])["order_id"].nunique()
    else:
        table = rows.groupby([period, rows[dimension]])["net"].sum()
    return table.unstack(fill_value=0).astype("float64")


def _root_causes(
    slices: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    metric: str,
    direction: str,
    rules: Rules,
    step: pd.Timedelta,
) -> tuple[RootCause, ...]:
    causes: list[RootCause] = []
    if metric in RATE_METRICS or rules.volume_lags[0] == 1:
        lags: tuple[int, ...] = tuple(range(1, rules.rate_window + 1))
    else:
        lags = rules.volume_lags
    history_days = [date - step * k for k in lags]
    for dimension, wide in slices.items():
        index = pd.DatetimeIndex(wide.index)
        actual_row = wide.reindex([date], fill_value=0.0).iloc[0]
        history = wide.loc[index.isin(history_days)].reindex(history_days, fill_value=0.0)
        expected = history.median() if len(history) else pd.Series(0.0, index=wide.columns)
        delta = (actual_row - expected).astype("float64")
        total = float(delta.sum())
        ordered = delta.sort_values(ascending=direction == "drop")
        for value, change in ordered.items():
            if (direction == "drop" and change >= 0) or (direction == "spike" and change <= 0):
                break
            causes.append(
                RootCause(
                    dimension=dimension,
                    value=str(value),
                    actual=float(actual_row.get(value, 0.0)),
                    expected=float(expected.get(value, 0.0)),
                    delta=float(change),
                    share_of_change=float(change) / total if total else None,
                )
            )
            if sum(1 for c in causes if c.dimension == dimension) >= 3:
                break
    return tuple(causes)


# --------------------------------------------------------------------------- detection
def _periods(tx: pd.DataFrame, granularity: Granularity) -> pd.DataFrame:
    table = kpi_timeseries(tx, granularity)
    if granularity == "W":
        table = table.loc[table["complete"]]
    return table


def _slice_alerts(
    tx: pd.DataFrame, index: pd.DatetimeIndex, granularity: Granularity, rules: Rules
) -> list[Alert]:
    """Net-revenue monitoring per region/store and per channel (A2)."""
    alerts: list[Alert] = []
    completed = tx.loc[tx["status"].eq("completed")]
    non_cancelled = tx.loc[tx["status"].ne("cancelled")]
    total_revenue = (
        completed.groupby(period_start(completed["order_ts"], granularity))["net"]
        .sum()
        .reindex(index, fill_value=0.0)
        .astype("float64")
    )
    total_trailing = _trailing_median(total_revenue, rules)
    for dimension in SLICE_DIMENSIONS:
        if tx[dimension].nunique(dropna=True) < 2:
            continue  # a single slice is the total itself: monitoring it would duplicate alerts
        revenue = _slice_table(tx, dimension, "net_revenue", granularity).reindex(
            index, fill_value=0.0
        )
        orders = (
            non_cancelled.groupby(
                [period_start(non_cancelled["order_ts"], granularity), non_cancelled[dimension]]
            )["order_id"]
            .nunique()
            .unstack(fill_value=0)
            .reindex(index, fill_value=0)
            .astype("float64")
        )
        if revenue.shape[1] < 2:
            continue  # a single slice is the total
        for value in sorted(map(str, revenue.columns)):
            slice_revenue = revenue[value].astype("float64")
            slice_orders = (
                orders[value].astype("float64")
                if value in orders.columns
                else pd.Series(0.0, index=index)
            )
            eligible = (_trailing_median(slice_orders, rules) >= rules.min_slice_orders) & (
                _trailing_median(slice_revenue, rules) >= MIN_SLICE_SHARE * total_trailing
            )
            scores = score_metric(slice_revenue, volume=True, rules=rules, valid=eligible)
            for date, row in scores.iterrows():
                z_hit, iqr_hit = bool(row["z_flag"]), bool(row["iqr_flag"])
                z, dev = float(row["z"]), float(row["dev"])
                severity = severity_of(z, dev, z_hit, iqr_hit, False)
                if severity is None:
                    continue
                alerts.append(
                    Alert(
                        date=pd.Timestamp(str(date)),
                        metric="net_revenue",
                        value=float(row["value"]),
                        baseline=float(row["baseline"]),
                        deviation=dev,
                        z=z,
                        methods=tuple(n for n, h in (("zscore", z_hit), ("iqr", iqr_hit)) if h),
                        severity=severity,
                        direction="spike" if dev > 0 else "drop",
                        scope_dimension=dimension,
                        scope_value=value,
                    )
                )
    return alerts


def detect_anomalies(
    tx: pd.DataFrame, granularity: Granularity = "D", *, slices: bool = True
) -> AnomalyReport:
    rules = RULES[granularity]
    table = _periods(tx, granularity)
    index = pd.DatetimeIndex(table.index)
    step = pd.Timedelta(7 if granularity == "W" else 1, unit="D")
    guard = _volume_guard(table["orders"].astype("float64"), rules)
    frames: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        frames[metric] = score_metric(
            table[metric].astype("float64"),
            volume=metric in VOLUME_METRICS,
            rules=rules,
            valid=guard if metric in RATE_METRICS else None,
        )
    scores = pd.concat(frames, axis=1)
    z_matrix = np.column_stack([frames[m]["z"].to_numpy(dtype=np.float64) for m in METRICS])
    forest_flags = expanding_iforest(z_matrix, rules)
    scores[("all", "iforest_flag")] = forest_flags

    enriched = _with_customer_type(tx)
    alerts: list[Alert] = []
    slice_cache: dict[str, dict[str, pd.DataFrame]] = {}
    for metric in METRICS:
        frame = frames[metric]
        for pos, (date, row) in enumerate(frame.iterrows()):
            z_hit, iqr_hit = bool(row["z_flag"]), bool(row["iqr_flag"])
            forest_hit = bool(forest_flags[pos])
            z, dev = float(row["z"]), float(row["dev"])
            severity = severity_of(z, dev, z_hit, iqr_hit, forest_hit)
            if severity is None:
                continue
            methods = tuple(
                name
                for name, hit in (("zscore", z_hit), ("iqr", iqr_hit), ("iforest", forest_hit))
                if hit
            )
            direction = "spike" if dev > 0 else "drop"
            if metric not in slice_cache:
                slice_cache[metric] = {
                    dim: _slice_table(enriched, dim, metric, granularity)
                    for dim in ROOT_CAUSE_DIMENSIONS
                }
            when = pd.Timestamp(str(date))
            alerts.append(
                Alert(
                    date=when,
                    metric=metric,
                    value=float(row["value"]),
                    baseline=float(row["baseline"]),
                    deviation=dev,
                    z=z,
                    methods=methods,
                    severity=severity,
                    direction=direction,
                    root_causes=_root_causes(
                        slice_cache[metric], when, metric, direction, rules, step
                    ),
                )
            )
    if slices:
        alerts.extend(_slice_alerts(tx, index, granularity, rules))
    alerts.sort(key=lambda a: (a.date, METRICS.index(a.metric), a.scope))
    return AnomalyReport(granularity=granularity, scores=scores, alerts=alerts)
