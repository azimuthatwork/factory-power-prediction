"""Base model interface for forecasting models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class ForecastModel(ABC):
    """Abstract base class for forecasting models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        pass

    @abstractmethod
    def fit(self, df: pd.DataFrame, target: str) -> "ForecastModel":
        """Fit model on historical data."""
        pass

    @abstractmethod
    def predict(self, history: pd.DataFrame, target: str, future: pd.DataFrame) -> np.ndarray:
        """Generate forecast for future horizon."""
        pass

    @abstractmethod
    def backtest(self, df: pd.DataFrame, target: str, split: int) -> dict[str, Any]:
        """Run backtest on held-out period and return metrics."""
        pass
