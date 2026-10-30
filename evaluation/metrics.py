"""Evaluation metrics and backtesting utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def model_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute MAE and RMSE."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def seasonal_naive_forecast(df: pd.DataFrame, target: str, split: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate seasonal naive baseline (same hour last week).

    Parameters
    ----------
    df : pd.DataFrame
        Full historical data.
    target : str
        Target column name.
    split : int
        Train/test split index.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (baseline_predictions, actual_values) for test period.
    """
    y = df[target].to_numpy(dtype=float)
    baseline_preds = y[split - 168:len(df) - 168]
    actual_test = y[split:]
    return baseline_preds, actual_test


def format_metrics_table(
    baseline_metrics: dict[str, float],
    rf_metrics: dict[str, float],
    transformer_metrics: dict[str, float] | None = None
) -> pd.DataFrame:
    """Format metrics for display in Streamlit."""
    rows = [
        {"Model": "Seasonal naïve (same hour last week)", "MAE (kW)": round(baseline_metrics["mae"], 2), "RMSE (kW)": round(baseline_metrics["rmse"], 2)},
        {"Model": "Random Forest", "MAE (kW)": round(rf_metrics["mae"], 2), "RMSE (kW)": round(rf_metrics["rmse"], 2)},
    ]
    if transformer_metrics is not None:
        rows.append({"Model": "Tiny temporal Transformer", "MAE (kW)": round(transformer_metrics["mae"], 2), "RMSE (kW)": round(transformer_metrics["rmse"], 2)})
    return pd.DataFrame(rows)
