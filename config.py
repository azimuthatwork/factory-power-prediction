"""Centralized configuration for FactoryPower AI demo."""

from __future__ import annotations

# App metadata
APP_TITLE = "FactoryPower AI"
APP_ICON = "⚡"

# Reproducibility
SEED = 17

# Data generation
HISTORY_DAYS = 60
HORIZON = 24
LOOKBACK = 72

# Model defaults
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 12
RF_MIN_SAMPLES_LEAF = 3
RF_N_JOBS = 2

TRANSFORMER_D_MODEL = 16
TRANSFORMER_NHEAD = 2
TRANSFORMER_DIM_FEEDFORWARD = 32
TRANSFORMER_DROPOUT = 0.1
TRANSFORMER_NUM_LAYERS = 1
TRANSFORMER_LR = 0.003
TRANSFORMER_BATCH_SIZE = 32
TRANSFORMER_EPOCHS = 5
TRANSFORMER_GRAD_CLIP = 1.0

# Evaluation
BACKTEST_DAYS = 7
QUANTILE_LOW = 0.10
QUANTILE_HIGH = 0.90

# Feature columns
EXOG_FEATURES = [
    "ambient_temp_c", "production_plan", "shift_1", "shift_2",
    "night_shift", "maintenance", "is_weekday", "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
]

TARGET_LABELS = {
    "feeder_kw": "Main feeder demand (kW)",
    "compressor_kw": "Air compressor (kW)",
    "cnc_line_kw": "CNC line (kW)",
    "industrial_heater_kw": "Industrial heater (kW)",
}

# UI
MODEL_CHOICES_BASE = ["Random Forest (fast CPU demo)"]
MODEL_CHOICE_TRANSFORMER = "Tiny temporal Transformer (CPU, educational)"

# Safety disclaimer
SAFETY_DISCLAIMER = (
    "Demo data is synthetic. This application does not connect to a real meter, "
    "PLC, SCADA, or factory control system."
)
