from __future__ import annotations

import streamlit as st

from config import APP_TITLE, APP_ICON, HISTORY_DAYS, HORIZON, SEED
from data.generator import generate_factory_data, make_future_schedule
from models.random_forest import RandomForestModel
from models.transformer import TransformerModel, TORCH_AVAILABLE
from evaluation.metrics import model_metrics, seasonal_naive_forecast, format_metrics_table
from ui.components import (
    render_header, render_sidebar, render_kpis, render_forecast_chart,
    render_backtest_table, render_checklist, render_download_button, render_how_it_works
)


st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


def main() -> None:
    render_header()

    # Sidebar controls
    target, model_choice, production_multiplier, force_maintenance, epochs = render_sidebar(TORCH_AVAILABLE)

    # Load data
    df = generate_factory_data(days=HISTORY_DAYS, seed=SEED)
    split = len(df) - 7 * 24
    future = make_future_schedule(df["timestamp"].iloc[-1], production_multiplier, force_maintenance)

    # Baseline: seasonal naive (same hour last week)
    baseline_preds, actual_test = seasonal_naive_forecast(df, target, split)
    baseline_metrics = model_metrics(actual_test, baseline_preds) if len(baseline_preds) == len(actual_test) else {"mae": float("nan"), "rmse": float("nan")}

    # Random Forest model
    rf_model = RandomForestModel()
    rf_model.fit(df, target)
    rf_backtest = rf_model.backtest(df, target, split)
    rf_metrics = rf_backtest["metrics"]
    rf_interval = rf_backtest["interval"]

    # Transformer model (optional)
    transformer_model = None
    transformer_metrics = None
    if "Transformer" in model_choice:
        if st.button("Train / load temporal Transformer", type="primary"):
            st.session_state["train_transformer_requested"] = True
        if st.session_state.get("train_transformer_requested", False):
            transformer_model = TransformerModel(epochs=epochs)
            transformer_model.fit(df, target)
            transformer_metrics = transformer_model.backtest(df, target, split)["metrics"]
        else:
            st.warning("Click **Train / load temporal Transformer** to run the small CPU training demo.")

    # Select active model for forecast
    if "Transformer" in model_choice and transformer_model is not None:
        forecast = transformer_model.predict(df, target, future)
        err_low, err_high = transformer_model.interval
        active_model = transformer_model.name
    else:
        forecast = rf_model.predict(df, target, future)
        err_low, err_high = rf_interval
        active_model = rf_model.name

    # Post-process forecast
    forecast = np.maximum(0, forecast)
    low_band = np.maximum(0, forecast + err_low)
    high_band = np.maximum(low_band, forecast + err_high)
    peak_idx = int(np.argmax(forecast))
    peak_time = future["timestamp"].iloc[peak_idx]

    # KPIs
    render_kpis(
        latest_load=df[target].iloc[-1],
        avg_forecast=forecast.mean(),
        peak_forecast=forecast.max(),
        peak_time_str=peak_time.strftime("%a %H:%M")
    )

    # Compute baseline for future period (seasonal naive: same hour last week)
    # We need 7 days (168 hours) of history before the future period starts
    baseline_future = None
    if len(df) >= 168:
        baseline_future = df[target].iloc[-168:].values  # last 168 hours = same hours 1 week ago
        if len(baseline_future) > len(future):
            baseline_future = baseline_future[-len(future):]
        elif len(baseline_future) < len(future):
            # Pad if needed (shouldn't happen with 24h forecast and 60 days history)
            baseline_future = np.pad(baseline_future, (0, len(future) - len(baseline_future)), mode='edge')

    # Forecast chart
    st.markdown(f"### Forecast view — {target}")
    hist_view = df.tail(7 * 24)
    render_forecast_chart(
        hist_view=hist_view,
        target=target,
        future=future,
        forecast=forecast,
        low_band=low_band,
        high_band=high_band,
        last_timestamp=df["timestamp"].iloc[-1],
        last_value=df[target].iloc[-1],
        baseline=baseline_future,
        baseline_label="Seasonal Naive (Baseline)"
    )

    # Backtest and checklist
    left, right = st.columns([1, 1])
    with left:
        render_backtest_table(baseline_metrics, rf_metrics, transformer_metrics)
    with right:
        render_checklist()

    # Download and methodology
    render_download_button(df)
    render_how_it_works()


if __name__ == "__main__":
    import numpy as np  # noqa: F401
    main()
