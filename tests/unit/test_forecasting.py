"""Forecasting: strict temporal / leakage guarantees, metrics, intervals, determinism."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from bizpulse.ml.forecasting import (
    FEATURE_NAMES,
    MIN_HISTORY,
    FloatArray,
    backtest,
    daily_revenue,
    features_for,
    fit_model,
    forecast_series,
    interval_quantiles,
    metrics,
    seasonal_naive,
    training_frame,
)
from tests.conftest import make_tx


def synthetic(n: int = 320, seed: int = 0) -> pd.Series[float]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=n, freq="D")
    weekly = np.array([1.0, 0.9, 0.95, 1.0, 1.1, 1.4, 1.3])[index.dayofweek]
    values = 1000 * 1.002 ** np.arange(n) * weekly * rng.lognormal(0.0, 0.08, n)
    return pd.Series(values, index=index, dtype="float64")


@pytest.fixture(scope="module")
def series() -> pd.Series[float]:
    return synthetic()


def test_features_require_history_and_have_stable_shape(series: pd.Series[float]) -> None:
    values = np.asarray(series.to_numpy(), dtype=np.float64)
    with pytest.raises(ValueError, match="history"):
        features_for(values[: MIN_HISTORY - 1], pd.Timestamp("2025-03-01"))
    row = features_for(values[:100], series.index[100])
    assert row.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(row).all()


def test_features_at_t_never_see_t_or_later(series: pd.Series[float]) -> None:
    values = np.asarray(series.to_numpy(), dtype=np.float64)
    dates = pd.DatetimeIndex(series.index)
    cut = 150
    poisoned = values.copy()
    poisoned[cut:] = 1e12  # perturb y at and after t
    clean_x, _, _ = training_frame(values, dates)
    poisoned_x, _, _ = training_frame(poisoned, dates)
    rows_up_to_cut = cut - MIN_HISTORY + 1  # rows for t = MIN_HISTORY .. cut (inclusive)
    np.testing.assert_array_equal(clean_x[:rows_up_to_cut], poisoned_x[:rows_up_to_cut])
    assert not np.array_equal(clean_x[rows_up_to_cut], poisoned_x[rows_up_to_cut])


@pytest.mark.parametrize("garbage", [float("nan"), 1e9, -5.0])
def test_forecast_ignores_actuals_after_cutoff(series: pd.Series[float], garbage: float) -> None:
    cutoff = series.index[260]
    reference = forecast_series(series.loc[series.index < cutoff], n_splits=3)
    poisoned = series.copy()
    poisoned.loc[poisoned.index >= cutoff] = garbage
    result = forecast_series(poisoned, cutoff=cutoff, n_splits=3)
    pd.testing.assert_frame_equal(reference.forecast, result.forecast)
    assert reference.backtest.model == result.backtest.model
    assert result.forecast.index[0] == cutoff


def test_backtest_folds_train_strictly_before_test(series: pd.Series[float]) -> None:
    seen: list[tuple[int, pd.Timestamp]] = []

    def spy(values: FloatArray, dates: pd.DatetimeIndex) -> Any:
        seen.append((len(values), pd.Timestamp(dates[-1])))
        return fit_model(values, dates)

    result = backtest(series, n_splits=4, horizon=30, fitter=spy)
    folds = result.folds
    assert len(folds) == len(seen) == 4
    for (n_train, train_end), (_, fold) in zip(seen, folds.iterrows(), strict=True):
        assert train_end == fold["train_end"]
        assert train_end < fold["test_start"]
        assert fold["test_start"] == train_end + pd.Timedelta(1, unit="D")
        # the training data is exactly the prefix before the test window
        assert n_train == series.index.get_loc(fold["test_start"])
    assert folds["test_end"].is_monotonic_increasing


def test_seasonal_naive_repeats_last_week() -> None:
    history = np.arange(1, 15, dtype=np.float64)
    assert seasonal_naive(history, 10).tolist() == [8, 9, 10, 11, 12, 13, 14, 8, 9, 10]


def test_metrics_handle_zero_actuals() -> None:
    actual = np.asarray([0.0, 2.0, 4.0])
    predicted = np.asarray([0.0, 1.0, 5.0])
    result = metrics(actual, predicted)
    assert result["mae"] == pytest.approx(2 / 3)
    assert result["rmse"] == pytest.approx(math.sqrt(2 / 3))
    assert result["mape"] == pytest.approx((0.5 + 0.25) / 2)  # zero actual excluded
    assert result["smape"] == pytest.approx((0 + 2 / 3 + 2 / 9) / 3)  # 0/0 counts as 0
    assert metrics(np.zeros(2), np.zeros(2))["mape"] is None


def test_interval_quantiles_are_monotone_and_bracket_zero() -> None:
    table = interval_quantiles([(1, 0.1), (2, 0.2), (3, 0.3), (10, -0.4), (25, 0.5)])
    for q in table.values():
        assert q["lo95"] <= q["lo80"] <= 0.0 <= q["hi80"] <= q["hi95"]


def test_forecast_intervals_ordered_and_deterministic(series: pd.Series[float]) -> None:
    first = forecast_series(series, n_splits=4)
    second = forecast_series(series, n_splits=4)
    pd.testing.assert_frame_equal(first.forecast, second.forecast)
    fc = first.forecast
    assert len(fc) == 30
    assert (fc["lo95"] >= 0).all()
    assert (fc["lo95"] <= fc["lo80"]).all()
    assert (fc["lo80"] <= fc["yhat"]).all()
    assert (fc["yhat"] <= fc["hi80"]).all()
    assert (fc["hi80"] <= fc["hi95"]).all()
    bt = first.backtest
    assert bt.coverage80 is not None and 0.0 <= bt.coverage80 <= 1.0
    assert bt.coverage95 is not None and bt.coverage95 >= bt.coverage80
    assert set(first.feature_importance.index) == set(FEATURE_NAMES)


def test_daily_revenue_gap_fills_and_counts_completed_only() -> None:
    tx = make_tx(
        {"ts": "2026-01-01 10:00", "gross": 10},
        {"ts": "2026-01-03 10:00", "gross": 20},
        {"ts": "2026-01-03 11:00", "gross": 99, "status": "refunded"},
        {"ts": "2026-01-03 12:00", "gross": 99, "status": "cancelled"},
    )
    daily = daily_revenue(tx)
    assert daily.tolist() == [10.0, 0.0, 20.0]
    assert pd.DatetimeIndex(daily.index).freqstr == "D"


def test_too_short_series_is_rejected() -> None:
    with pytest.raises(ValueError):
        backtest(synthetic(n=100), n_splits=4)
