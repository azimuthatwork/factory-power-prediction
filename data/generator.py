"""Synthetic factory electrical load data generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    SEED, HISTORY_DAYS, EXOG_FEATURES, TARGET_LABELS
)
from features.calendar import calendar_features


def generate_factory_data(days: int = HISTORY_DAYS, seed: int = SEED) -> pd.DataFrame:
    """
    Generate synthetic hourly factory electrical load data.

    Parameters
    ----------
    days : int
        Number of days of historical data to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with timestamp index and columns for each load component,
        total feeder load, power factor, and production units.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-06 00:00:00", periods=days * 24, freq="h")
    df = calendar_features(timestamps)
    production = df["production_plan"].to_numpy()
    ambient = df["ambient_temp_c"].to_numpy()
    hour = timestamps.hour.to_numpy()

    # Correlated disturbances create more realistic patterns than white noise alone.
    noise = rng.normal(0, 2.5, len(df))
    drift = np.cumsum(rng.normal(0, 0.12, len(df)))
    start_up = (((hour == 6) | (hour == 14)) & (production > 0.4)).astype(float)

    df["compressor_kw"] = np.maximum(4, 8 + 23 * production + 0.28 * np.maximum(ambient - 25, 0)
                                     + rng.normal(0, 1.8, len(df)))
    df["cnc_line_kw"] = np.maximum(3, 7 + 52 * production + 8 * start_up
                                   + rng.normal(0, 3.4, len(df)))
    df["industrial_heater_kw"] = np.maximum(2, 5 + 30 * production
                                            + 0.35 * (ambient - 28)
                                            + rng.normal(0, 2.2, len(df)))
    df["hvac_kw"] = np.maximum(4, 8 + 1.25 * np.maximum(ambient - 25, 0)
                               + 1.8 * np.sin(2 * np.pi * (hour - 7) / 24)
                               + rng.normal(0, 1.0, len(df)))
    df["auxiliary_kw"] = np.maximum(3, 5 + 2 * df["is_weekday"].to_numpy()
                                    + rng.normal(0, 0.7, len(df)))
    df["feeder_kw"] = (df["compressor_kw"] + df["cnc_line_kw"]
                       + df["industrial_heater_kw"] + df["hvac_kw"]
                       + df["auxiliary_kw"] + noise + drift)
    df["feeder_kw"] = np.maximum(df["feeder_kw"], 20)

    # Approximate power factor as a sensor-like supporting signal.
    df["power_factor"] = np.clip(0.94 - 0.025 * (production > 0.8)
                                  + rng.normal(0, 0.008, len(df)), 0.82, 0.99)
    df["production_units"] = np.maximum(0, np.round(12 * production + rng.normal(0, 1.2, len(df))))

    return df


def make_future_schedule(
    last_timestamp: pd.Timestamp,
    multiplier: float,
    force_maintenance: bool
) -> pd.DataFrame:
    """
    Generate future calendar features for forecasting horizon.

    Parameters
    ----------
    last_timestamp : pd.Timestamp
        Last timestamp in historical data.
    multiplier : float
        Production intensity multiplier for forecast period.
    force_maintenance : bool
        Whether to simulate maintenance in first 4 forecast hours.

    Returns
    -------
    pd.DataFrame
        Future calendar features for HORIZON hours.
    """
    from config import HORIZON
    future_index = pd.date_range(last_timestamp + pd.Timedelta(hours=1), periods=HORIZON, freq="h")
    return calendar_features(future_index, production_multiplier=multiplier, force_maintenance=force_maintenance)
