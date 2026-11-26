"""Leak-free daily net-revenue forecasting (TR-ML-01..03).

Design: **leakage is impossible by construction.**

* :func:`features_for` receives only the history strictly *before* the target date.
  Training rows and forecast steps are both built with it, so no code path can read a
  value at or after the date being predicted.
* Multi-step forecasts are **recursive**. Each step appends the model's own prediction
  to the history, never an actual value.
* Backtests use ``TimeSeriesSplit`` with fixed 30-day test folds. Each fold's model is
  fitted on that fold's training prefix only.

Model: LightGBM predicts the ratio ``y_t / level_t``, where ``level_t`` is the trailing
28-day mean known at *t*. The result is rescaled by the level. Trees cannot extrapolate
a trend, but they can model a ratio to a moving level.

Limitation (documented): with ~18 months of history and lags <= 28 days, the model has
no memory of annual events such as Black Friday, so those are under-forecast.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit

FloatArray = npt.NDArray[np.float64]

HORIZON = 30
N_SPLITS = 6
LAGS: tuple[int, ...] = (1, 2, 3, 7, 14, 21, 28)
LEVEL_WINDOW = 28
MIN_HISTORY = 56  # days of history a training row / forecast origin needs
MIN_TRAIN_DAYS = 120
LEVEL_EPS = 1.0
HORIZON_BUCKETS: tuple[tuple[int, int], ...] = ((1, 7), (8, 14), (15, 21), (22, 30))
QUANTILES = {"lo95": 0.025, "lo80": 0.10, "hi80": 0.90, "hi95": 0.975}
SEED = 42

FEATURE_NAMES: tuple[str, ...] = (
    *(f"lag_{k}" for k in LAGS),
    "mean_7",
    "mean_14",
    "std_7",
    "same_weekday_mean_4w",
    "momentum_28_vs_prev_28",
    "dow",
    "dom",
    "month",
    "is_weekend",
    "is_month_start",
    "is_month_end",
)


# --------------------------------------------------------------------------- data
def daily_revenue(tx: pd.DataFrame) -> pd.Series[float]:
    """Completed net revenue per day over a continuous daily index (gaps = 0)."""
    completed = tx.loc[tx["status"].eq("completed")]
    daily = completed.groupby(completed["order_ts"].dt.normalize())["net"].sum()
    start = pd.Timestamp(tx["order_ts"].min()).normalize()
    end = pd.Timestamp(tx["order_ts"].max()).normalize()
    index = pd.date_range(start, end, freq="D", name="date")
    series: pd.Series[float] = daily.reindex(index, fill_value=0.0).astype("float64")
    series.name = "net_revenue"
    return series


# --------------------------------------------------------------------------- features
def _level(history: FloatArray) -> float:
    return float(history[-LEVEL_WINDOW:].mean()) + LEVEL_EPS


def features_for(history: FloatArray, date: pd.Timestamp) -> FloatArray:
    """Feature vector for *date* using only ``history`` (= values strictly before it)."""
    if len(history) < MIN_HISTORY:
        raise ValueError(f"need >= {MIN_HISTORY} days of history, got {len(history)}")
    level = _level(history)
    lags = [history[-k] / level for k in LAGS]
    previous = float(history[-2 * LEVEL_WINDOW : -LEVEL_WINDOW].mean()) + LEVEL_EPS
    same_weekday = history[[-7, -14, -21, -28]].mean() / level
    next_day = date + pd.Timedelta(1, unit="D")
    row = [
        *lags,
        history[-7:].mean() / level,
        history[-14:].mean() / level,
        history[-7:].std() / level,
        same_weekday,
        level / previous,
        float(date.dayofweek),
        float(date.day),
        float(date.month),
        float(date.dayofweek >= 5),
        float(date.day == 1),
        float(next_day.month != date.month),
    ]
    return np.asarray(row, dtype=np.float64)


def training_frame(
    values: FloatArray, dates: pd.DatetimeIndex
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Supervised rows ``(X, ratio_target, level)`` for every t with enough history."""
    rows, targets, levels = [], [], []
    for t in range(MIN_HISTORY, len(values)):
        history = values[:t]
        level = _level(history)
        rows.append(features_for(history, dates[t]))
        targets.append(values[t] / level)
        levels.append(level)
    return (
        np.vstack(rows) if rows else np.empty((0, len(FEATURE_NAMES))),
        np.asarray(targets, dtype=np.float64),
        np.asarray(levels, dtype=np.float64),
    )


# --------------------------------------------------------------------------- model
def make_model() -> Any:
    return LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=15,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=SEED,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        verbose=-1,
    )


def fit_model(values: FloatArray, dates: pd.DatetimeIndex) -> Any:
    """Fit on a training prefix (instrumented by tests to prove fold isolation)."""
    features, target, _ = training_frame(values, dates)
    if len(target) < 30:
        raise ValueError("not enough history to train a forecasting model")
    model = make_model()
    model.fit(pd.DataFrame(features, columns=list(FEATURE_NAMES)), target)
    return model


def recursive_forecast(
    model: Any, history: FloatArray, start: pd.Timestamp, horizon: int = HORIZON
) -> FloatArray:
    """Predict ``horizon`` days from ``start``, feeding back predictions (never actuals)."""
    extended = list(map(float, history))
    out: list[float] = []
    for step in range(horizon):
        date = start + pd.Timedelta(step, unit="D")
        current = np.asarray(extended, dtype=np.float64)
        row = pd.DataFrame([features_for(current, date)], columns=list(FEATURE_NAMES))
        ratio = float(model.predict(row)[0])
        value = max(0.0, ratio * _level(current))
        out.append(value)
        extended.append(value)
    return np.asarray(out, dtype=np.float64)


def seasonal_naive(history: FloatArray, horizon: int = HORIZON, season: int = 7) -> FloatArray:
    """Repeat the last observed season: y_hat(T+h) = y(T+h - season * ceil(h / season))."""
    last = history[-season:]
    return np.asarray([last[h % season] for h in range(horizon)], dtype=np.float64)


# --------------------------------------------------------------------------- metrics
def metrics(actual: FloatArray, predicted: FloatArray) -> dict[str, float | None]:
    """MAE, RMSE, MAPE (non-zero actuals only) and sMAPE (0/0 counts as 0)."""
    error = actual - predicted
    nonzero = actual != 0
    denominator = np.abs(actual) + np.abs(predicted)
    smape_terms = np.divide(
        2 * np.abs(error), denominator, out=np.zeros_like(error), where=denominator != 0
    )
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(math.sqrt(np.mean(error**2))),
        "mape": float(np.mean(np.abs(error[nonzero] / actual[nonzero]))) if nonzero.any() else None,
        "smape": float(np.mean(smape_terms)),
    }


# --------------------------------------------------------------------------- intervals
def _bucket(step: int) -> int:
    for idx, (low, high) in enumerate(HORIZON_BUCKETS):
        if low <= step <= high:
            return idx
    return len(HORIZON_BUCKETS) - 1


def interval_quantiles(residuals: list[tuple[int, float]]) -> dict[int, dict[str, float]]:
    """Per-horizon-bucket log-residual quantiles, monotone and bracketing 0."""
    table: dict[int, dict[str, float]] = {}
    for idx in range(len(HORIZON_BUCKETS)):
        values = np.asarray([r for step, r in residuals if _bucket(step) == idx])
        if values.size == 0:
            values = np.asarray([r for _, r in residuals]) if residuals else np.zeros(1)
        q = {name: float(np.quantile(values, p)) for name, p in QUANTILES.items()}
        q["lo80"] = min(q["lo80"], 0.0)
        q["hi80"] = max(q["hi80"], 0.0)
        q["lo95"] = min(q["lo95"], q["lo80"])
        q["hi95"] = max(q["hi95"], q["hi80"])
        table[idx] = q
    return table


def apply_intervals(
    point: FloatArray, quantiles: dict[int, dict[str, float]]
) -> dict[str, FloatArray]:
    """Multiplicative (log-space) bands around *point*, clipped at 0."""
    bands: dict[str, list[float]] = {name: [] for name in QUANTILES}
    for step, value in enumerate(point, start=1):
        q = quantiles[_bucket(step)]
        for name in QUANTILES:
            bands[name].append(max(0.0, float(np.expm1(np.log1p(value) + q[name]))))
    return {name: np.asarray(v, dtype=np.float64) for name, v in bands.items()}


# --------------------------------------------------------------------------- backtest
@dataclass
class BacktestResult:
    folds: pd.DataFrame
    model: dict[str, float | None]
    baseline: dict[str, float | None]
    coverage80: float | None
    coverage95: float | None
    residuals: list[tuple[int, float]] = field(default_factory=list)

    @property
    def mae_ratio(self) -> float | None:
        model_mae, base_mae = self.model["mae"], self.baseline["mae"]
        if model_mae is None or not base_mae:
            return None
        return model_mae / base_mae


Fitter = Callable[[FloatArray, pd.DatetimeIndex], Any]


def backtest(
    series: pd.Series[float],
    *,
    n_splits: int = N_SPLITS,
    horizon: int = HORIZON,
    fitter: Fitter | None = None,
) -> BacktestResult:
    """Rolling-origin evaluation with ``TimeSeriesSplit`` (fixed ``horizon``-day folds)."""
    fit = fitter or fit_model
    values = np.asarray(series.to_numpy(), dtype=np.float64)
    dates = pd.DatetimeIndex(series.index)
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=horizon)
    rows: list[dict[str, Any]] = []
    actual_all, model_all, base_all = [], [], []
    per_fold: list[tuple[FloatArray, FloatArray, list[tuple[int, float]]]] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(values)):
        if len(train_idx) < MIN_TRAIN_DAYS:
            continue
        train = values[train_idx]
        model = fit(train, dates[train_idx])
        start = dates[test_idx[0]]
        predicted = recursive_forecast(model, train, start, horizon)
        baseline = seasonal_naive(train, horizon)
        actual = values[test_idx]
        fold_residuals = [
            (step, float(np.log1p(a) - np.log1p(p)))
            for step, (a, p) in enumerate(zip(actual, predicted, strict=True), start=1)
        ]
        per_fold.append((actual, predicted, fold_residuals))
        model_metrics = metrics(actual, predicted)
        base_metrics = metrics(actual, baseline)
        rows.append(
            {
                "fold": fold,
                "train_end": dates[train_idx[-1]],
                "test_start": start,
                "test_end": dates[test_idx[-1]],
                "model_mae": model_metrics["mae"],
                "baseline_mae": base_metrics["mae"],
                "model_smape": model_metrics["smape"],
                "baseline_smape": base_metrics["smape"],
            }
        )
        actual_all.append(actual)
        model_all.append(predicted)
        base_all.append(baseline)
    if not rows:
        raise ValueError("series too short for backtesting")
    actual_cat = np.concatenate(actual_all)
    residuals = [r for _, _, fold_res in per_fold for r in fold_res]

    # Leave-one-fold-out calibration: coverage is measured on residuals that did NOT
    # set the quantiles, so it is an honest out-of-fold estimate.
    hits80, hits95, total = 0, 0, 0
    for idx, (actual, predicted, _) in enumerate(per_fold):
        others = [r for j, (_, _, res) in enumerate(per_fold) if j != idx for r in res]
        if not others:
            continue
        bands = apply_intervals(predicted, interval_quantiles(others))
        hits80 += int(((actual >= bands["lo80"]) & (actual <= bands["hi80"])).sum())
        hits95 += int(((actual >= bands["lo95"]) & (actual <= bands["hi95"])).sum())
        total += len(actual)
    return BacktestResult(
        folds=pd.DataFrame(rows),
        model=metrics(actual_cat, np.concatenate(model_all)),
        baseline=metrics(actual_cat, np.concatenate(base_all)),
        coverage80=hits80 / total if total else None,
        coverage95=hits95 / total if total else None,
        residuals=residuals,
    )


# --------------------------------------------------------------------------- forecast
@dataclass
class ForecastResult:
    history: pd.Series[float]
    forecast: pd.DataFrame  # index = date: yhat, lo80, hi80, lo95, hi95, baseline
    backtest: BacktestResult
    feature_importance: pd.Series[float]


def forecast_series(
    series: pd.Series[float],
    *,
    horizon: int = HORIZON,
    cutoff: pd.Timestamp | None = None,
    n_splits: int = N_SPLITS,
) -> ForecastResult:
    """Backtest, then forecast *horizon* days after the last observation before *cutoff*.

    Everything at or after *cutoff* is discarded up front, so no downstream code can see it.
    """
    history = series if cutoff is None else series.loc[series.index < cutoff]
    history = history.astype("float64")
    result = backtest(history, n_splits=n_splits, horizon=horizon)
    values = np.asarray(history.to_numpy(), dtype=np.float64)
    dates = pd.DatetimeIndex(history.index)
    model = fit_model(values, dates)
    start = dates[-1] + pd.Timedelta(1, unit="D")
    point = recursive_forecast(model, values, start, horizon)
    bands = apply_intervals(point, interval_quantiles(result.residuals))
    index = pd.date_range(start, periods=horizon, freq="D", name="date")
    frame = pd.DataFrame(
        {
            "yhat": point,
            "lo80": np.minimum(bands["lo80"], point),
            "hi80": np.maximum(bands["hi80"], point),
            "lo95": np.minimum(bands["lo95"], np.minimum(bands["lo80"], point)),
            "hi95": np.maximum(bands["hi95"], np.maximum(bands["hi80"], point)),
            "baseline": seasonal_naive(values, horizon),
        },
        index=index,
    )
    importance = pd.Series(
        np.asarray(model.feature_importances_, dtype=np.float64),
        index=list(FEATURE_NAMES),
        name="importance",
    ).sort_values(ascending=False)
    return ForecastResult(
        history=history, forecast=frame, backtest=result, feature_importance=importance
    )
