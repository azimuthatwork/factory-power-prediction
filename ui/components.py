"""Streamlit UI components for FactoryPower AI dashboard."""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from config import APP_TITLE, APP_ICON, SAFETY_DISCLAIMER, TARGET_LABELS, MODEL_CHOICES_BASE, MODEL_CHOICE_TRANSFORMER


def render_header() -> None:
    """Render app title and disclaimer."""
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.subheader("Electrical load forecasting for a manufacturing line")
    st.info(SAFETY_DISCLAIMER)


def render_sidebar(
    torch_available: bool,
    default_epochs: int = 5
) -> tuple[str, str, float, bool, int]:
    """
    Render sidebar controls and return user selections.

    Returns
    -------
    tuple
        (target, model_choice, production_multiplier, force_maintenance, epochs)
    """
    with st.sidebar:
        st.header("Demo controls")
        target = st.selectbox(
            "Electrical load to forecast",
            list(TARGET_LABELS),
            format_func=lambda x: TARGET_LABELS[x]
        )
        model_choices = MODEL_CHOICES_BASE.copy()
        if torch_available:
            model_choices.append(MODEL_CHOICE_TRANSFORMER)
        model_choice = st.selectbox("Forecast model", model_choices)
        production_multiplier = st.slider(
            "Planned production intensity", 0.5, 1.2, 1.0, 0.05,
            help="Scenario input for the next 24 hours."
        )
        force_maintenance = st.checkbox(
            "Simulate maintenance in first 4 forecast hours", value=False
        )
        epochs = st.slider("Transformer training epochs", 3, 8, default_epochs) if "Transformer" in model_choice else default_epochs
        st.caption("CPU-friendly data: 60 days of hourly simulated readings. No GPU required.")
    return target, model_choice, production_multiplier, force_maintenance, epochs


def render_kpis(
    latest_load: float,
    avg_forecast: float,
    peak_forecast: float,
    peak_time_str: str
) -> None:
    """Render KPI metric cards."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest measured load", f"{latest_load:.1f} kW")
    c2.metric("Next 24h average forecast", f"{avg_forecast:.1f} kW")
    c3.metric("Forecast peak", f"{peak_forecast:.1f} kW", help="Point forecast, not a guaranteed maximum.")
    c4.metric("Estimated peak time", peak_time_str)


def render_forecast_chart(
    hist_view: pd.DataFrame,
    target: str,
    future: pd.DataFrame,
    forecast: np.ndarray,
    low_band: np.ndarray,
    high_band: np.ndarray,
    last_timestamp: pd.Timestamp,
    last_value: float
) -> None:
    """Render forecast chart with history, forecast, and uncertainty band."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_view["timestamp"], y=hist_view[target], mode="lines",
        name="Observed history", line=dict(color="#1f77b4", width=2)
    ))
    # Bridge point so forecast visually connects to last observation
    forecast_x = [last_timestamp, *future["timestamp"].tolist()]
    forecast_y = [float(last_value), *forecast.tolist()]
    fig.add_trace(go.Scatter(
        x=forecast_x, y=forecast_y, mode="lines+markers", name="Forecast",
        line=dict(color="#e67e22", width=3, dash="dash"), marker=dict(size=5)
    ))
    fig.add_trace(go.Scatter(
        x=list(future["timestamp"]) + list(future["timestamp"][::-1]),
        y=list(high_band) + list(low_band[::-1]), fill="toself",
        fillcolor="rgba(230,126,34,0.16)", line=dict(color="rgba(255,255,255,0)"),
        name="Illustrative residual range", hoverinfo="skip"
    ))
    fig.update_layout(
        height=440, margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Time", yaxis_title="Electrical demand (kW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The shaded band is based on historical validation residuals; "
        "it is an illustrative uncertainty range, not a formally calibrated prediction interval."
    )


def render_backtest_table(
    baseline_metrics: dict[str, float],
    rf_metrics: dict[str, float],
    transformer_metrics: dict[str, float] | None = None
) -> None:
    """Render backtest metrics comparison table."""
    from evaluation.metrics import format_metrics_table
    st.markdown("### Backtest: last 7 days")
    df_metrics = format_metrics_table(baseline_metrics, rf_metrics, transformer_metrics)
    st.dataframe(df_metrics, hide_index=True, use_container_width=True)
    st.caption(
        "MAE = average absolute error. RMSE penalizes larger errors more heavily. "
        "Backtests use historical readings; actual performance on real equipment may differ."
    )


def render_checklist() -> None:
    """Render electrical engineer's checklist."""
    st.markdown("### Electrical engineer's checklist")
    st.write("- Does the predicted peak approach the feeder or transformer limit?")
    st.write("- Are shift start-up peaks plausible?")
    st.write("- Is the planned maintenance period reflected in the scenario?")
    st.write("- Do we have enough meter history and trustworthy timestamps?")
    st.write("- Is the forecast used for advisory planning only, not automatic control?")


def render_download_button(df: pd.DataFrame) -> None:
    """Render CSV download button for synthetic data."""
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download synthetic factory meter data (CSV)",
        data=csv,
        file_name="synthetic_factory_power_hourly.csv",
        mime="text/csv"
    )


def render_how_it_works() -> None:
    """Render expander with methodology explanation."""
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
