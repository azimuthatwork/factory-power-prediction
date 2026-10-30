"""Tabular feature engineering for Random Forest forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import EXOG_FEATURES


def tabular_row(values: np.ndarray, exog: pd.Series, i: int) -> list[float]:
    """
    Build a single tabular feature row for time index i.

    Parameters
    ----------
    values : np.ndarray
        Observed target values (history + any recursive forecasts).
    exog : pd.Series
        Exogenous features for current time step.
    i : int
        Current time index in values array.

    Returns
    -------
    list[float]
        Feature vector: [lag1, lag2, lag3, lag24, lag48, lag168,
                         mean_3h, mean_24h, mean_168h, exog...]
    """
    lag = lambda n: float(values[i - n])
    return [
        lag(1), lag(2), lag(3), lag(24), lag(48), lag(168),
        float(np.mean(values[max(0, i - 3):i])),
        float(np.mean(values[max(0, i - 24):i])),
        float(np.mean(values[max(0, i - 168):i])),
        *[float(exog[c]) for c in EXOG_FEATURES]
    ]


def build_tabular_matrix(df: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build supervised learning matrix from time series data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with target column and EXOG_FEATURES.
    target : str
        Name of target column.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (X, y, positions) where X is feature matrix, y is target, positions are row indices.
    """
    y = df[target].to_numpy(dtype=float)
    xs, ys, positions = [], [], []
    for i in range(168, len(df)):
        xs.append(tabular_row(y, df.iloc[i], i))
        ys.append(y[i])
        positions.append(i)
    return np.asarray(xs), np.asarray(ys), np.asarray(positions, dtype=int)
