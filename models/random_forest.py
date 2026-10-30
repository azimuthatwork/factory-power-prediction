"""Random Forest forecasting model implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import (
    SEED, RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF, RF_N_JOBS
)
from features.tabular import build_tabular_matrix, tabular_row
from models.base import ForecastModel


class RandomForestModel(ForecastModel):
    """Random Forest forecaster with recursive multi-step prediction."""

    def __init__(
        self,
        n_estimators: int = RF_N_ESTIMATORS,
        max_depth: int = RF_MAX_DEPTH,
        min_samples_leaf: int = RF_MIN_SAMPLES_LEAF,
        n_jobs: int = RF_N_JOBS,
        random_state: int = SEED,
    ):
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf
        self._n_jobs = n_jobs
        self._random_state = random_state
        self._model: RandomForestRegressor | None = None
        self._interval: tuple[float, float] = (-8.0, 8.0)

    @property
    def name(self) -> str:
        return "Random Forest (fast CPU demo)"

    def fit(self, df: pd.DataFrame, target: str) -> "RandomForestModel":
        """Fit Random Forest on all available history."""
        X, Y, _ = build_tabular_matrix(df, target)
        self._model = RandomForestRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            min_samples_leaf=self._min_samples_leaf,
            n_jobs=self._n_jobs,
            random_state=self._random_state,
        )
        self._model.fit(X, Y)
        return self

    def predict(self, history: pd.DataFrame, target: str, future: pd.DataFrame) -> np.ndarray:
        """Recursive multi-step forecast."""
        if self._model is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return recursive_rf_forecast(self._model, history, target, future)

    def backtest(self, df: pd.DataFrame, target: str, split: int) -> dict[str, Any]:
        """Run backtest on held-out period."""
        X, Y, POS = build_tabular_matrix(df, target)
        train_mask = POS < split
        test_mask = POS >= split

        model = RandomForestRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            min_samples_leaf=self._min_samples_leaf,
            n_jobs=self._n_jobs,
            random_state=self._random_state,
        )
        model.fit(X[train_mask], Y[train_mask])
        preds = model.predict(X[test_mask])
        actual = Y[test_mask]

        metrics = model_metrics(actual, preds)
        interval = np.quantile(actual - preds, [0.10, 0.90]).tolist()
        self._interval = (float(interval[0]), float(interval[1]))

        return {
            "metrics": metrics,
            "interval": self._interval,
            "predictions": preds,
            "actual": actual,
        }

    @property
    def interval(self) -> tuple[float, float]:
        """Residual quantile interval from backtest."""
        return self._interval


def recursive_rf_forecast(
    model: RandomForestRegressor,
    history: pd.DataFrame,
    target: str,
    future: pd.DataFrame
) -> np.ndarray:
    """Generate recursive forecast using fitted Random Forest."""
    vals = history[target].to_numpy(dtype=float).tolist()
    preds = []
    for j in range(len(future)):
        i = len(vals)
        row = np.asarray(tabular_row(np.asarray(vals, dtype=float), future.iloc[j], i), dtype=float).reshape(1, -1)
        p = float(model.predict(row)[0])
        preds.append(p)
        vals.append(p)
    return np.asarray(preds)


def model_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute MAE and RMSE."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }
