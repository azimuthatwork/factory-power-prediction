# FactoryPower AI — CPU-friendly factory electrical load forecasting demo

A visual Show & Share demo for electrical engineers, software developers, designers, and product managers. It forecasts the next 24 hours of electrical demand for a synthetic manufacturing line.

> **Important:** all data is synthetic. This demo is not connected to any real energy meter, PLC, SCADA system, or factory control network. It is for learning and discussion, not operational control or protection.

## Demo story

An electrical engineer wants to understand tomorrow's electrical demand for a factory feeder that supplies an air compressor, CNC line, industrial heater, HVAC, and auxiliary loads. The dashboard shows recent demand, a 24-hour forecast, an illustrative uncertainty band, and a simple backtest comparison.

The audience can change the planned production intensity and simulate a maintenance stop in the first four forecast hours. This creates a discussion across roles:

- **Electrical engineer:** Would the predicted peak approach feeder or transformer capacity? Are shift-start peaks plausible?
- **Developer:** Are there enough good readings? Is the forecast better than a simple baseline?
- **Designer:** How should uncertainty be displayed so it does not look like a guarantee?
- **PM:** Which decision is the forecast meant to improve, and what is the cost of a false alarm vs. missed peak?

## Requirements

- Python 3.10 or newer recommended
- Works on CPU; no GPU required
- Internet required for the initial package installation, unless packages are already available locally

## Run the reliable fast demo

### Linux / macOS

```bash
cd factorypower-ai-demo
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Windows PowerShell

```powershell
cd factorypower-ai-demo
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

The default Random Forest forecast, the seasonal baseline, data generator, chart, and backtest do not require PyTorch.

## Enable the educational temporal Transformer

The app also includes a deliberately small temporal Transformer encoder that trains on CPU. It is intended to illustrate sequence learning, not to reproduce the complete Temporal Fusion Transformer (TFT) architecture.

Install PyTorch using the CPU-appropriate command for your Python/platform from the official selector: https://pytorch.org/get-started/locally/

Then install the optional dependencies:

```bash
pip install -r requirements-transformer.txt
streamlit run app.py
```

In the sidebar, select **Tiny temporal Transformer (CPU, educational)** and click **Train / load temporal Transformer**. The first run trains on synthetic data; Streamlit caches the trained model during the app session. Training speed depends on CPU and Python/PyTorch build. If the live training takes too long, select Random Forest and use the prepared charts to keep the session moving.

## What the app simulates

- Hourly feeder load, in kW
- Air compressor demand
- CNC line demand
- Industrial heater demand
- HVAC and auxiliary demand
- Ambient-temperature proxy
- Production shift schedule and a maintenance calendar
- A power-factor sensor-like signal and production units for discussion

The synthetic data intentionally contains daily/weekly schedules, temperature effects, start-up peaks, and noise. It is suitable for a demonstration, not for estimating real plant performance.

## Model and evaluation notes

- The baseline repeats demand from the same hour one week earlier.
- The Random Forest uses lag features and schedule/temperature inputs; the model is refit on all synthetic history for the forward forecast.
- The optional temporal Transformer uses a short history window and known future calendar/schedule features to predict a 24-hour horizon.
- Backtest metrics show rolling one-step estimates over a held-out final week; they are not a guarantee of multi-step future accuracy.
- The displayed residual range is illustrative and should not be described as a calibrated confidence interval.
- No model is assumed superior before comparison against the baseline.

## How to use the demo in a 45–60 minute Show & Share

1. Ask the room to guess tomorrow's peak demand (5 min).
2. Explain historical demand vs. known future schedule (7 min).
3. Show the chart and compare the forecast against the same-hour-last-week baseline (10–12 min).
4. Change production intensity or simulate maintenance, then discuss the change (5 min).
5. Split participants into teams. Ask them to recommend a response to a possible demand peak while considering limits, uncertainty, and data quality (13 min).
6. Each team pitches for one minute; score business objective, uncertainty, UX, baseline evaluation, and operational risk (9 min).
7. Close by asking what additional real plant data is needed (4 min).

## Suggested real-world data schema

When adapting this demo, start with an offline CSV—not direct control-system connectivity—with fields such as:

```text
timestamp, feeder_kw, compressor_kw, cnc_line_kw, industrial_heater_kw,
ambient_temp_c, production_plan, shift, maintenance_flag
```

Use consistent timezone handling, document meter sampling intervals, handle missing/duplicate timestamps, and ensure future inputs really are known at the forecast origin. Use rolling-origin evaluation to avoid look-ahead leakage.

## Electrical safety note

This is advisory analytics only. Do not use it for protection trips, electrical interlocks, PLC control, or autonomous switching. Those functions require engineered and validated protection/control systems.
