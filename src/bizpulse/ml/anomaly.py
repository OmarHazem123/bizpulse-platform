"""KPI anomaly detection with severity and root-cause attribution (TR-ML-04, §7.1).

Rules (pre-registered in TECHNICAL-REQUIREMENTS.md §7.1):

* Baselines exclude the current day. Revenue and orders use the median of the same
  weekday over the previous 4 weeks. AOV and refund rate use the median of the previous
  28 days.
* ``dev = (x - baseline) / baseline`` (reported). The robust z-score (median / MAD) is
  computed over the trailing 56 days, excluding the current day. Volume metrics are
  scored two-tailed (amendment A1): spikes on ``dev``, drops on
  ``log1p(x) - log1p(baseline)``. Rate metrics are scored on ``dev``.
* Methods: ``|z| >= 3.5``, a 3 x IQR fence, and IsolationForest (corroboration only).
* CRITICAL: ``|z| >= 6`` or (>= 2 methods agree and ``|dev| >= 30 %``). WARNING otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import IsolationForest

from bizpulse.analytics.kpis import first_orders, kpi_timeseries

METRICS: tuple[str, ...] = ("net_revenue", "orders", "aov", "refund_rate")
WEEKDAY_METRICS = frozenset({"net_revenue", "orders"})
RATE_METRICS = frozenset({"aov", "refund_rate"})
Z_THRESHOLD = 3.5
CRITICAL_Z = 6.0
IQR_K = 3.0
CRITICAL_MIN_DEV = 0.30
HISTORY_WINDOW = 56
WARMUP = 28
MIN_ORDERS = 20
MIN_ORDERS_SHARE = 0.25
IFOREST_CONTAMINATION = 0.02
SEED = 42
ROOT_CAUSE_DIMENSIONS: tuple[str, ...] = ("category", "channel", "region", "customer_type")


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

    def top_cause(self, dimension: str) -> RootCause | None:
        return next((c for c in self.root_causes if c.dimension == dimension), None)


@dataclass
class AnomalyReport:
    scores: pd.DataFrame  # MultiIndex columns (metric, field)
    alerts: list[Alert]

    def alerts_between(
        self, start: pd.Timestamp, end: pd.Timestamp, metric: str | None = None
    ) -> list[Alert]:
        return [
            a
            for a in self.alerts
            if start <= a.date <= end and (metric is None or a.metric == metric)
        ]


# --------------------------------------------------------------------------- scoring
def _baseline(values: npt.NDArray[np.float64], t: int, weekday: bool) -> float:
    if weekday:
        picks = [values[t - k] for k in (7, 14, 21, 28) if t - k >= 0]
    else:
        picks = list(values[max(0, t - 28) : t])
    clean = [v for v in picks if not np.isnan(v)]
    return float(np.median(clean)) if len(clean) >= (3 if weekday else 7) else float("nan")


def score_metric(
    series: pd.Series[float], *, weekday: bool, valid: pd.Series[bool] | None = None
) -> pd.DataFrame:
    """Baseline, deviation, robust z and method flags per day (trailing windows only)."""
    values = np.array(series.to_numpy(), dtype=np.float64)
    if valid is not None:
        values[~valid.to_numpy()] = np.nan
    n = len(values)
    baseline = np.full(n, np.nan)
    dev = np.full(n, np.nan)
    for t in range(n):
        b = _baseline(values, t, weekday)
        baseline[t] = b
        if not np.isnan(values[t]) and not np.isnan(b) and b > 0:
            dev[t] = (values[t] - b) / b
    # Amendment A1 (two-tailed): daily volume noise is right-skewed. The ratio deviation is
    # the natural unbounded statistic for spikes and the log deviation for drops (a ratio
    # is bounded at -100 %, so a total collapse could not reach |z| >= 3.5). Each tail is
    # scored against its own statistic's robust scale. Rate metrics stay linear.
    if weekday:
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
        stat = dev if dev[t] >= 0 else log_dev
        hist = stat[max(0, t - HISTORY_WINDOW) : t]
        hist = hist[~np.isnan(hist)]
        if len(hist) < WARMUP:
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


def _volume_guard(orders: pd.Series[float]) -> pd.Series[bool]:
    """Rate metrics are scored only on days with enough orders (trailing median, no look-ahead)."""
    trailing = orders.shift(1).rolling(28, min_periods=7).median()
    return (orders >= MIN_ORDERS) & (orders >= MIN_ORDERS_SHARE * trailing.fillna(0.0))


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


def _slice_daily(tx: pd.DataFrame, dimension: str, metric: str) -> pd.DataFrame:
    """Daily x slice values of the attributable quantity for *metric*."""
    day = tx["order_ts"].dt.normalize()
    if metric == "orders":
        rows = tx.loc[tx["status"].ne("cancelled")]
        table = rows.groupby([day[rows.index], rows[dimension]])["order_id"].nunique()
    elif metric == "refund_rate":
        rows = tx.loc[tx["status"].eq("refunded")]
        table = rows.groupby([day[rows.index], rows[dimension]])["net"].sum()
    else:  # net_revenue and aov are attributed through revenue
        rows = tx.loc[tx["status"].eq("completed")]
        table = rows.groupby([day[rows.index], rows[dimension]])["net"].sum()
    wide = table.unstack(fill_value=0)
    return wide.astype("float64")


def _root_causes(
    slices: dict[str, pd.DataFrame], date: pd.Timestamp, metric: str, direction: str
) -> tuple[RootCause, ...]:
    causes: list[RootCause] = []
    for dimension, wide in slices.items():
        index = pd.DatetimeIndex(wide.index)
        actual_row = wide.reindex([date], fill_value=0.0).iloc[0]
        if metric in RATE_METRICS:
            history = wide.loc[(index < date) & (index >= date - pd.Timedelta(28, unit="D"))]
        else:
            same_days = [date - pd.Timedelta(7 * k, unit="D") for k in (1, 2, 3, 4)]
            history = wide.loc[index.isin(same_days)]
            history = history.reindex(same_days, fill_value=0.0)
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
def detect_anomalies(tx: pd.DataFrame) -> AnomalyReport:
    daily = kpi_timeseries(tx, "D")
    guard = _volume_guard(daily["orders"].astype("float64"))
    frames: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        frames[metric] = score_metric(
            daily[metric].astype("float64"),
            weekday=metric in WEEKDAY_METRICS,
            valid=guard if metric in RATE_METRICS else None,
        )
    scores = pd.concat(frames, axis=1)

    # IsolationForest corroboration on the multivariate robust-z vector.
    z_matrix = np.column_stack([frames[m]["z"].to_numpy() for m in METRICS])
    scored = ~np.isnan(z_matrix).all(axis=1)
    iforest = np.zeros(len(daily), dtype=bool)
    if scored.sum() >= 50:
        forest = IsolationForest(
            contamination=IFOREST_CONTAMINATION, random_state=SEED, n_estimators=200
        )
        labels = forest.fit_predict(np.nan_to_num(z_matrix[scored]))
        iforest[np.flatnonzero(scored)] = labels == -1
    scores[("all", "iforest_flag")] = iforest

    enriched = _with_customer_type(tx)
    alerts: list[Alert] = []
    slice_cache: dict[str, dict[str, pd.DataFrame]] = {}
    for metric in METRICS:
        frame = frames[metric]
        for pos, (date, row) in enumerate(frame.iterrows()):
            z_hit, iqr_hit = bool(row["z_flag"]), bool(row["iqr_flag"])
            if not (z_hit or iqr_hit):
                continue
            methods = tuple(
                name
                for name, hit in (("zscore", z_hit), ("iqr", iqr_hit), ("iforest", iforest[pos]))
                if hit
            )
            z, dev = float(row["z"]), float(row["dev"])
            critical = abs(z) >= CRITICAL_Z or (len(methods) >= 2 and abs(dev) >= CRITICAL_MIN_DEV)
            direction = "spike" if dev > 0 else "drop"
            if metric not in slice_cache:
                slice_cache[metric] = {
                    dim: _slice_daily(enriched, dim, metric) for dim in ROOT_CAUSE_DIMENSIONS
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
                    severity="CRITICAL" if critical else "WARNING",
                    direction=direction,
                    root_causes=_root_causes(slice_cache[metric], when, metric, direction),
                )
            )
    alerts.sort(key=lambda a: (a.date, METRICS.index(a.metric)))
    return AnomalyReport(scores=scores, alerts=alerts)
