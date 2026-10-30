from __future__ import annotations

from datetime import timedelta
import math
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

APP_TITLE = "FactoryPower AI"
SEED = 17
HISTORY_DAYS = 60
HORIZON = 24
LOOKBACK = 72

st.set_page_config(page_title=APP_TITLE, page_icon="⚡", layout="wide")


# ---------------------------- Synthetic factory data ----------------------------
def calendar_features(index: pd.DatetimeIndex, production_multiplier: float = 1.0,
                      force_maintenance: bool = False) -> pd.DataFrame:
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
        maintenance[:min(4, len(maintenance))] = 1.0
        shift1[:min(4, len(shift1))] = 0.0
        shift2[:min(4, len(shift2))] = 0.0
        night_shift[:min(4, len(night_shift))] = 0.0
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
    })


@st.cache_data(show_spinner=False)
def generate_factory_data(days: int = HISTORY_DAYS, seed: int = SEED) -> pd.DataFrame:
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


EXOG_FEATURES = ["ambient_temp_c", "production_plan", "shift_1", "shift_2",
                 "night_shift", "maintenance", "is_weekday", "hour_sin", "hour_cos",
                 "dow_sin", "dow_cos"]
TARGET_LABELS = {
    "feeder_kw": "Main feeder demand (kW)",
    "compressor_kw": "Air compressor (kW)",
    "cnc_line_kw": "CNC line (kW)",
    "industrial_heater_kw": "Industrial heater (kW)",
}


def tabular_row(values: np.ndarray, exog: pd.Series, i: int) -> list[float]:
    # values holds observed target values, followed by recursively forecast values.
    lag = lambda n: float(values[i - n])
    return [lag(1), lag(2), lag(3), lag(24), lag(48), lag(168),
            float(np.mean(values[max(0, i - 3):i])),
            float(np.mean(values[max(0, i - 24):i])),
            float(np.mean(values[max(0, i - 168):i])),
            *[float(exog[c]) for c in EXOG_FEATURES]]


def build_tabular_matrix(df: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = df[target].to_numpy(dtype=float)
    xs, ys, positions = [], [], []
    for i in range(168, len(df)):
        xs.append(tabular_row(y, df.iloc[i], i))
        ys.append(y[i])
        positions.append(i)
    return np.asarray(xs), np.asarray(ys), np.asarray(positions, dtype=int)


def make_future_schedule(last_timestamp: pd.Timestamp, multiplier: float,
                         force_maintenance: bool) -> pd.DataFrame:
    future_index = pd.date_range(last_timestamp + pd.Timedelta(hours=1), periods=HORIZON, freq="h")
    return calendar_features(future_index, production_multiplier=multiplier,
                             force_maintenance=force_maintenance)


def recursive_rf_forecast(model: RandomForestRegressor, history: pd.DataFrame,
                           target: str, future: pd.DataFrame) -> np.ndarray:
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
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


@st.cache_resource(show_spinner="Training a small CPU temporal Transformer and checking a holdout window...")
def train_temporal_transformer(target: str, n_rows: int, lookback: int = LOOKBACK,
                               horizon: int = HORIZON, epochs: int = 5):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed. Install requirements-transformer.txt to enable this mode.")
    torch.set_num_threads(min(2, max(1, torch.get_num_threads())))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    df = generate_factory_data(days=n_rows // 24)
    split = len(df) - 7 * 24
    target_mean = float(df.iloc[:split][target].mean())
    target_std = float(df.iloc[:split][target].std() + 1e-6)
    exog_mean = df.iloc[:split][EXOG_FEATURES].mean().to_numpy(dtype=float)
    exog_std = (df.iloc[:split][EXOG_FEATURES].std().to_numpy(dtype=float) + 1e-6)
    exog_raw = df[EXOG_FEATURES].to_numpy(dtype=float)
    exog_z = (exog_raw - exog_mean) / exog_std
    y_raw = df[target].to_numpy(dtype=float)
    y_z = (y_raw - target_mean) / target_std
    past = np.column_stack([y_z, exog_z]).astype(np.float32)
    future_cov = exog_z.astype(np.float32)

    class TinyTemporalTransformer(nn.Module):
        def __init__(self, past_dim: int, future_dim: int, d_model: int = 16):
            super().__init__()
            self.proj = nn.Linear(past_dim, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=2, dim_feedforward=32,
                dropout=0.1, batch_first=True, activation="gelu"
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.future_proj = nn.Linear(future_dim, d_model)
            self.head = nn.Sequential(
                nn.Linear(d_model * 2, 32), nn.GELU(), nn.Linear(32, 1)
            )
            self.position = nn.Parameter(torch.zeros(1, lookback, d_model))
            nn.init.normal_(self.position, mean=0.0, std=0.02)

        def forward(self, x_past, x_future):
            encoded = self.encoder(self.proj(x_past) + self.position)
            context = encoded[:, -1, :].unsqueeze(1).expand(-1, x_future.shape[1], -1)
            future_emb = self.future_proj(x_future)
            return self.head(torch.cat([context, future_emb], dim=-1)).squeeze(-1)

    def windows(starts):
        xp, xf, yy = [], [], []
        for t in starts:
            xp.append(past[t - lookback:t])
            xf.append(future_cov[t:t + horizon])
            yy.append(y_z[t:t + horizon])
        if not xp:
            return (np.empty((0, lookback, past.shape[1]), np.float32),
                    np.empty((0, horizon, future_cov.shape[1]), np.float32),
                    np.empty((0, horizon), np.float32))
        return np.asarray(xp, np.float32), np.asarray(xf, np.float32), np.asarray(yy, np.float32)

    train_starts = list(range(lookback, split - horizon + 1, 2))
    val_starts = list(range(split, len(df) - horizon + 1, 3))
    xpt, xft, yt = windows(train_starts)
    xpv, xfv, yv = windows(val_starts)
    device = torch.device("cpu")
    model = TinyTemporalTransformer(past.shape[1], future_cov.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(torch.from_numpy(xpt), torch.from_numpy(xft), torch.from_numpy(yt))
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    model.train()
    for _ in range(epochs):
        for bx, bf, by in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(bx, bf)
            loss = loss_fn(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        val_pred_z = model(torch.from_numpy(xpv), torch.from_numpy(xfv)).cpu().numpy()
    val_pred = val_pred_z * target_std + target_mean
    val_actual = yv * target_std + target_mean
    resid = (val_actual - val_pred).reshape(-1)
    q_low, q_high = np.quantile(resid, [0.10, 0.90]).tolist() if len(resid) else (-8.0, 8.0)
    metrics = model_metrics(val_actual.reshape(-1), val_pred.reshape(-1))

    # Fine-tune the same tiny model on all available windows for the forward forecast.
    all_starts = list(range(lookback, len(df) - horizon + 1, 2))
    xpa, xfa, ya = windows(all_starts)
    ds_all = TensorDataset(torch.from_numpy(xpa), torch.from_numpy(xfa), torch.from_numpy(ya))
    loader_all = DataLoader(ds_all, batch_size=32, shuffle=True)
    model.train()
    for _ in range(max(2, epochs // 2)):
        for bx, bf, by in loader_all:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(bx, bf), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    pack = {
        "model": model,
        "target_mean": target_mean,
        "target_std": target_std,
        "exog_mean": exog_mean,
        "exog_std": exog_std,
        "lookback": lookback,
        "q_low": float(q_low),
        "q_high": float(q_high),
        "metrics": metrics,
        "model_type": "Tiny temporal Transformer (CPU)",
    }
    return pack


def predict_transformer(pack: dict, history: pd.DataFrame, target: str,
                         future: pd.DataFrame) -> np.ndarray:
    exog_hist = (history[EXOG_FEATURES].to_numpy(dtype=float) - pack["exog_mean"]) / pack["exog_std"]
    y_hist = (history[target].to_numpy(dtype=float) - pack["target_mean"]) / pack["target_std"]
    past = np.column_stack([y_hist, exog_hist]).astype(np.float32)[-pack["lookback"]:]
    future_exog = ((future[EXOG_FEATURES].to_numpy(dtype=float) - pack["exog_mean"]) / pack["exog_std"]).astype(np.float32)
    with torch.no_grad():
        pred_z = pack["model"](torch.from_numpy(past[None, :, :]), torch.from_numpy(future_exog[None, :, :])).cpu().numpy()[0]
    return pred_z * pack["target_std"] + pack["target_mean"]


# ---------------------------- Sidebar and dashboard ----------------------------
st.title("⚡ FactoryPower AI")
st.subheader("Electrical load forecasting for a manufacturing line")
st.info("Demo data is synthetic. This application does not connect to a real meter, PLC, SCADA, or factory control system.")

with st.sidebar:
    st.header("Demo controls")
    target = st.selectbox("Electrical load to forecast", list(TARGET_LABELS),
                          format_func=lambda x: TARGET_LABELS[x])
    model_choices = ["Random Forest (fast CPU demo)"]
    if TORCH_AVAILABLE:
        model_choices.append("Tiny temporal Transformer (CPU, educational)")
    model_choice = st.selectbox("Forecast model", model_choices)
    production_multiplier = st.slider("Planned production intensity", 0.5, 1.2, 1.0, 0.05,
                                       help="Scenario input for the next 24 hours.")
    force_maintenance = st.checkbox("Simulate maintenance in first 4 forecast hours", value=False)
    epochs = st.slider("Transformer training epochs", 3, 8, 5) if "Transformer" in model_choice else 5
    st.caption("CPU-friendly data: 60 days of hourly simulated readings. No GPU required.")

df = generate_factory_data()
split = len(df) - 7 * 24
future = make_future_schedule(df["timestamp"].iloc[-1], production_multiplier, force_maintenance)

# Baseline: last week's corresponding hour. It is easy to explain and difficult to beat
# when the series has strong repeating weekly patterns.
y = df[target].to_numpy(dtype=float)
baseline_preds = y[split - 168:len(df) - 168]
actual_test = y[split:]
baseline_metrics = model_metrics(actual_test, baseline_preds) if len(baseline_preds) == len(actual_test) else {"mae": float("nan"), "rmse": float("nan")}

# Fit RF for backtest metrics, then refit on all history to create a forward forecast.
X, Y, POS = build_tabular_matrix(df, target)
train_mask = POS < split
test_mask = POS >= split
rf_eval = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=3,
                                n_jobs=2, random_state=SEED)
rf_eval.fit(X[train_mask], Y[train_mask])
rf_test = rf_eval.predict(X[test_mask])
rf_test_actual = Y[test_mask]
rf_metrics = model_metrics(rf_test_actual, rf_test)
rf_interval = np.quantile(rf_test_actual - rf_test, [0.10, 0.90]).tolist()
rf_final = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=3,
                                n_jobs=2, random_state=SEED)
rf_final.fit(X, Y)

transformer_pack = None
if "Transformer" in model_choice:
    if st.button("Train / load temporal Transformer", type="primary"):
        st.session_state["train_transformer_requested"] = True
    if st.session_state.get("train_transformer_requested", False):
        transformer_pack = train_temporal_transformer(target, len(df), epochs=epochs)
    else:
        st.warning("Click **Train / load temporal Transformer** to run the small CPU training demo.")

# KPIs and model selection
if "Transformer" in model_choice and transformer_pack is not None:
    forecast = predict_transformer(transformer_pack, df, target, future)
    err_low, err_high = transformer_pack["q_low"], transformer_pack["q_high"]
    selected_metrics = transformer_pack["metrics"]
    active_model = "Tiny temporal Transformer (CPU)"
else:
    forecast = recursive_rf_forecast(rf_final, df, target, future)
    err_low, err_high = rf_interval
    selected_metrics = rf_metrics
    active_model = "Random Forest (fast CPU demo)"

forecast = np.maximum(0, forecast)
low_band = np.maximum(0, forecast + err_low)
high_band = np.maximum(low_band, forecast + err_high)
peak_idx = int(np.argmax(forecast))
peak_time = future["timestamp"].iloc[peak_idx]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest measured load", f"{df[target].iloc[-1]:.1f} kW")
c2.metric("Next 24h average forecast", f"{forecast.mean():.1f} kW")
c3.metric("Forecast peak", f"{forecast.max():.1f} kW", help="Point forecast, not a guaranteed maximum.")
c4.metric("Estimated peak time", peak_time.strftime("%a %H:%M"))

st.markdown(f"### Forecast view — {TARGET_LABELS[target]}")
hist_view = df.tail(7 * 24)
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist_view["timestamp"], y=hist_view[target], mode="lines",
                         name="Observed history", line=dict(color="#1f77b4", width=2)))
# Show a bridge point so the forecast visually connects to the last observation.
forecast_x = [df["timestamp"].iloc[-1], *future["timestamp"].tolist()]
forecast_y = [float(df[target].iloc[-1]), *forecast.tolist()]
fig.add_trace(go.Scatter(x=forecast_x, y=forecast_y, mode="lines+markers", name="Forecast",
                         line=dict(color="#e67e22", width=3, dash="dash"), marker=dict(size=5)))
fig.add_trace(go.Scatter(x=list(future["timestamp"]) + list(future["timestamp"][::-1]),
                         y=list(high_band) + list(low_band[::-1]), fill="toself",
                         fillcolor="rgba(230,126,34,0.16)", line=dict(color="rgba(255,255,255,0)"),
                         name="Illustrative residual range", hoverinfo="skip"))
fig.update_layout(height=440, margin=dict(l=10, r=10, t=25, b=10),
                  xaxis_title="Time", yaxis_title="Electrical demand (kW)",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.18)")
fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.18)")
st.plotly_chart(fig, use_container_width=True)
st.caption("The shaded band is based on historical validation residuals; it is an illustrative uncertainty range, not a formally calibrated prediction interval.")

left, right = st.columns([1, 1])
with left:
    st.markdown("### Backtest: last 7 days")
    metric_rows = [
        {"Model": "Seasonal naïve (same hour last week)", "MAE (kW)": round(baseline_metrics["mae"], 2), "RMSE (kW)": round(baseline_metrics["rmse"], 2)},
        {"Model": "Random Forest", "MAE (kW)": round(rf_metrics["mae"], 2), "RMSE (kW)": round(rf_metrics["rmse"], 2)},
    ]
    if transformer_pack is not None:
        metric_rows.append({"Model": "Tiny temporal Transformer", "MAE (kW)": round(transformer_pack["metrics"]["mae"], 2), "RMSE (kW)": round(transformer_pack["metrics"]["rmse"], 2)})
    st.dataframe(pd.DataFrame(metric_rows), hide_index=True, use_container_width=True)
    st.caption("MAE = average absolute error. RMSE penalizes larger errors more heavily. Backtests use historical readings; actual performance on real equipment may differ.")
with right:
    st.markdown("### Electrical engineer's checklist")
    st.write("- Does the predicted peak approach the feeder or transformer limit?")
    st.write("- Are shift start-up peaks plausible?")
    st.write("- Is the planned maintenance period reflected in the scenario?")
    st.write("- Do we have enough meter history and trustworthy timestamps?")
    st.write("- Is the forecast used for advisory planning only, not automatic control?")

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download synthetic factory meter data (CSV)", data=csv,
                   file_name="synthetic_factory_power_hourly.csv", mime="text/csv")

with st.expander("How the demo works", expanded=False):
    st.markdown("""
    **Forecast target:** hourly electrical demand in kW for the selected feeder/load.

    **Inputs:** lagged demand, recent rolling averages, ambient-temperature proxy, hour/day cycles, shift schedule, production plan, and maintenance calendar.

    **Baseline:** same hour one week earlier.

    **Fast model:** Random Forest using lag features. This is a practical CPU benchmark, not a Transformer.

    **Optional model:** a deliberately small PyTorch temporal Transformer encoder trained on the synthetic series. It is an educational Transformer example, **not the full Temporal Fusion Transformer (TFT) architecture**.

    **Real-factory adaptation:** replace synthetic data with timestamped meter readings and known future schedule fields; validate with rolling-origin backtests; align sampling intervals and units; check sensor quality, missing data, and meter calibration.

    **Safety:** do not connect this demo to PLC/SCADA control loops or use forecasts to trip/protect electrical equipment. Protective functions must remain with certified protection systems.
    """)
