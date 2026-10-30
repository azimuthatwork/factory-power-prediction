"""Quick smoke test for refactored modules."""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from config import SEED, HISTORY_DAYS, EXOG_FEATURES, TARGET_LABELS
from data.generator import generate_factory_data, make_future_schedule
from features.calendar import calendar_features
from features.tabular import build_tabular_matrix
from models.random_forest import RandomForestModel, model_metrics
from evaluation.metrics import seasonal_naive_forecast, format_metrics_table


def test_data_generation():
    df = generate_factory_data(days=7, seed=SEED)
    assert len(df) == 7 * 24
    assert "feeder_kw" in df.columns
    assert "timestamp" in df.columns
    print("✓ Data generation works")


def test_calendar_features():
    idx = pd.date_range("2025-01-06", periods=48, freq="h")
    feats = calendar_features(idx)
    assert len(feats) == 48
    assert all(c in feats.columns for c in EXOG_FEATURES)
    print("✓ Calendar features work")


def test_tabular_matrix():
    df = generate_factory_data(days=10, seed=SEED)
    X, y, pos = build_tabular_matrix(df, "feeder_kw")
    assert X.shape[0] == y.shape[0] == len(pos)
    assert X.shape[1] == 6 + len(EXOG_FEATURES)  # 6 lags/means + exog
    print("✓ Tabular matrix building works")


def test_random_forest():
    df = generate_factory_data(days=14, seed=SEED)
    split = len(df) - 7 * 24
    model = RandomForestModel()
    model.fit(df, "feeder_kw")
    backtest = model.backtest(df, "feeder_kw", split)
    assert "metrics" in backtest
    assert "interval" in backtest
    print("✓ Random Forest model works")


def test_metrics():
    actual = np.array([10.0, 20.0, 30.0])
    pred = np.array([11.0, 19.0, 32.0])
    m = model_metrics(actual, pred)
    assert "mae" in m and "rmse" in m
    print("✓ Metrics computation works")


def test_seasonal_naive():
    df = generate_factory_data(days=14, seed=SEED)
    split = len(df) - 7 * 24
    baseline_preds, actual_test = seasonal_naive_forecast(df, "feeder_kw", split)
    assert len(baseline_preds) == len(actual_test) == 7 * 24
    print("✓ Seasonal naive baseline works")


def test_format_metrics():
    baseline = {"mae": 1.234, "rmse": 2.345}
    rf = {"mae": 0.987, "rmse": 1.876}
    df = format_metrics_table(baseline, rf)
    assert len(df) == 2
    assert list(df.columns) == ["Model", "MAE (kW)", "RMSE (kW)"]
    print("✓ Metrics formatting works")


if __name__ == "__main__":
    test_data_generation()
    test_calendar_features()
    test_tabular_matrix()
    test_random_forest()
    test_metrics()
    test_seasonal_naive()
    test_format_metrics()
    print("\n✅ All smoke tests passed!")
