"""Calendar feature engineering for factory load forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import EXOG_FEATURES


def calendar_features(
    index: pd.DatetimeIndex,
    production_multiplier: float = 1.0,
    force_maintenance: bool = False
) -> pd.DataFrame:
    """
    Generate calendar-based exogenous features for a given datetime index.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Timestamps for which to generate features.
    production_multiplier : float
        Multiplier for planned production intensity (0.5 to 1.2).
    force_maintenance : bool
        If True, simulate maintenance in first 4 hours (for scenario testing).

    Returns
    -------
    pd.DataFrame
        DataFrame with calendar features aligned to index.
    """
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    is_weekday = (dow < 5).astype(float)
    shift1 = ((hour >= 6) & (hour < 14) & (dow < 5)).astype(float)
    shift2 = ((hour >= 14) & (hour < 22) & (dow < 5)).astype(float)
    night_shift = ((hour >= 22) & (dow < 5) | (hour < 6) & (dow < 5)).astype(float)

    # Planned maintenance is a calendar-known flag; synthetic schedule only.
    maintenance = (((dow == 6) & (hour >= 8) & (hour < 12))).astype(float)

    if force_maintenance:
        # Scenario: a planned stop during the first four forecast hours.
        n = min(4, len(maintenance))
        maintenance[:n] = 1.0
        shift1[:n] = 0.0
        shift2[:n] = 0.0
        night_shift[:n] = 0.0

    prod = (0.10 + 0.84 * shift1 + 0.72 * shift2 + 0.22 * night_shift)
    # Reduced planned production on scheduled maintenance hours.
    prod = np.where(maintenance > 0, 0.12, prod)
    prod = np.clip(prod * float(production_multiplier), 0.03, 1.15)

    # Synthetic ambient-temperature profile. In an actual deployment this column
    # would come from a weather forecast or a separately forecast sensor.
    day_index = np.arange(len(index), dtype=float) + (index[0].dayofyear * 24)
    ambient = (29.0 + 3.0 * np.sin(2 * np.pi * (hour - 8) / 24)
               + 1.2 * np.sin(2 * np.pi * day_index / (365.25 * 24)))

    return pd.DataFrame({
        "timestamp": index,
        "ambient_temp_c": ambient.astype(float),
        "production_plan": prod.astype(float),
        "shift_1": shift1.astype(float),
        "shift_2": shift2.astype(float),
        "night_shift": night_shift.astype(float),
        "maintenance": maintenance.astype(float),
        "is_weekday": is_weekday.astype(float),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
    }, index=index)[EXOG_FEATURES + ["timestamp"]]
