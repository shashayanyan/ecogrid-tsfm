import os
# Force Hugging Face to use the locally cached model and skip corporate SSL network checks
os.environ["HF_HUB_OFFLINE"] = "1"

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

# Ensure Streamlit can find our src module
sys.path.append(str(Path.cwd()))
from src.data_loader import EnergyDataLoader
from src.forecaster import TSFMForecaster

# --- PAGE CONFIG ---
st.set_page_config(page_title="EcoGrid AI: TSFM Anomaly Detection", layout="wide")
st.title("EcoGrid-TSFM: Smart Building Energy Forecasting")
st.markdown("""
*Powered by Amazon Chronos Foundation Model.* This dashboard provides zero-shot load forecasting and automated predictive maintenance 
for commercial microgrids.
""")

# --- CACHING FOR PERFORMANCE ---
# We use st.cache_resource so we only load the 185MB model into memory ONCE
@st.cache_resource
def load_model():
    return TSFMForecaster(model_id="amazon/chronos-t5-small")

# We use st.cache_data so we don't reload the CSV on every button click
@st.cache_data
def load_data():
    loader = EnergyDataLoader('data/electricity_cleaned.csv')
    return loader.load_and_clean(target_building_id='Fox_education_Ollie')

# Load assets
try:
    forecaster = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Error loading assets. Ensure data exists in data/ folder. Error: {e}")
    st.stop()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Microgrid Controls")

# Let the user pick an experiment date
available_dates = df.index.date
selected_date = st.sidebar.date_input(
    "Select Forecast Start Date", 
    value=pd.to_datetime('2016-09-15').date(),
    min_value=available_dates[0],
    max_value=available_dates[-5]
)

# The "Killer Feature": A toggle to simulate an equipment failure
simulate_fault = st.sidebar.toggle("⚠️ Simulate VFD Equipment Fault", value=False)
st.sidebar.markdown("*(Simulates a cooling fan failing to spin down at night, causing a 35% load spike).*")

# --- DATA SLICING & PREDICTION ---
experiment_datetime = pd.to_datetime(f"{selected_date} 23:00:00")
context_length = 24 * 7 # 1 week of context
prediction_length = 48  # 48 hours ahead

# Slice context and actual future
historical_data = df.loc[:experiment_datetime].tail(context_length)
actual_future = df.loc[experiment_datetime:].iloc[1:prediction_length + 1]

# Run zero-shot forecast
with st.spinner("Chronos is generating the 48-hour probabilistic forecast..."):
    forecast_df = forecaster.predict(historical_data['energy_kwh'], prediction_length=prediction_length)

# Apply fault if toggled
display_actual = actual_future.copy()
if simulate_fault:
    # Spike the energy during the night hours (e.g., hours 26 to 30 into the future)
    display_actual.iloc[26:30, 0] = display_actual.iloc[26:30, 0] * 1.35

# Detect Anomalies
anomalies = display_actual[display_actual['energy_kwh'] > forecast_df['forecast_high_p90']]

# --- METRICS DASHBOARD ---
col1, col2, col3 = st.columns(3)
col1.metric("Predicted 48h Load", f"{forecast_df['forecast_median'].sum():.0f} kWh")
col2.metric("Actual 48h Load", f"{display_actual['energy_kwh'].sum():.0f} kWh", 
            delta=f"{(display_actual['energy_kwh'].sum() - forecast_df['forecast_median'].sum()):.0f} kWh" if simulate_fault else "Normal",
            delta_color="inverse")
col3.metric("Detected Faults", len(anomalies), 
            delta="Requires Maintenance!" if len(anomalies) > 0 else "System Healthy",
            delta_color="inverse")

# --- PLOTLY VISUALIZATION ---
fig = go.Figure()

# Plot History
fig.add_trace(go.Scatter(
    x=historical_data.tail(24*2).index, 
    y=historical_data.tail(24*2)['energy_kwh'],
    mode='lines', name='Historical Context', line=dict(color='blue')
))

# Plot Forecast Median & Interval
fig.add_trace(go.Scatter(
    x=forecast_df.index, y=forecast_df['forecast_median'],
    mode='lines', name='Chronos Forecast (Median)', line=dict(color='orange')
))
fig.add_trace(go.Scatter(
    x=list(forecast_df.index) + list(forecast_df.index)[::-1],
    y=list(forecast_df['forecast_high_p90']) + list(forecast_df['forecast_low_p10'])[::-1],
    fill='toself', fillcolor='rgba(255, 165, 0, 0.2)', line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip", name='80% Confidence Interval'
))

# Plot Actuals (Healthy or Faulty)
fig.add_trace(go.Scatter(
    x=display_actual.index, y=display_actual['energy_kwh'],
    mode='lines', name='Actual Meter Reading', line=dict(color='red' if simulate_fault else 'green')
))

# Highlight Anomalies
if not anomalies.empty:
    fig.add_trace(go.Scatter(
        x=anomalies.index, y=anomalies['energy_kwh'],
        mode='markers', name='Anomaly Detected', marker=dict(color='red', size=12, symbol='x')
    ))

fig.update_layout(height=500, template='plotly_white', hovermode="x unified")
st.plotly_chart(fig, width='stretch')