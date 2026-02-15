
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Tractor Sales Forecasting App")

st.title("🚜 Tractor Sales Forecasting")
st.write("Forecast for 2015 using Holt-Winters and ARIMA models.")

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    data_path = 'Tractor-Sales.csv' # Assuming the CSV is in the same directory as app.py
    df = pd.read_csv(data_path)
    df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='%b-%y')
    df1 = df.set_index('Month-Year')
    df1.index.freq = 'MS' # Set frequency for statsmodels
    return df1

df1 = load_data()

st.subheader("Original Data")
st.write(df1.head())

# --- MAPE Calculation Function ---
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# --- Holt-Winters Forecasting ---
st.subheader("Holt-Winters Forecasting")

# Split data for evaluation
test_size = 12
train_df_hw = df1[:-test_size]
test_df_hw = df1[-test_size:]

# Train HW model on training data
hw_model_eval = ExponentialSmoothing(train_df_hw['Number of Tractor Sold'],
                                   seasonal_periods=12,
                                   trend='add',
                                   seasonal='add').fit()

hw_test_predictions = hw_model_eval.forecast(steps=len(test_df_hw))
mape_hw = calculate_mape(test_df_hw['Number of Tractor Sold'], hw_test_predictions)
st.write(f"Holt-Winters MAPE on the test set: **{mape_hw:.2f}%**")

# Retrain HW model on full data for 2015 forecast
full_hw_model = ExponentialSmoothing(df1['Number of Tractor Sold'],
                                     seasonal_periods=12,
                                     trend='add',
                                     seasonal='add').fit()
forecast_2015_hw = full_hw_model.forecast(steps=12)

# Plot Holt-Winters Results
fig_hw = go.Figure()
fig_hw.add_trace(go.Scatter(x=df1.index, y=df1['Number of Tractor Sold'], mode='lines', name='Historical Data'))
fig_hw.add_trace(go.Scatter(x=hw_test_predictions.index, y=hw_test_predictions, mode='lines', name='HW Test Predictions', line=dict(dash='dash')))
fig_hw.add_trace(go.Scatter(x=forecast_2015_hw.index, y=forecast_2015_hw, mode='lines', name='HW 2015 Forecast', line=dict(color='green')))
fig_hw.update_layout(title='Holt-Winters: History, Test Predictions, and 2015 Forecast',
                     xaxis_title='Month-Year',
                     yaxis_title='Number of Tractor Sold',
                     hovermode='x unified')
st.plotly_chart(fig_hw, use_container_width=True)


# --- ARIMA Forecasting ---
st.subheader("ARIMA Forecasting")

# Split data for evaluation
train_df_arima = df1[:-test_size]
test_df_arima = df1[-test_size:]
train_df_arima.index.freq = 'MS'

# Train ARIMA model on training data
arima_model_eval = ARIMA(train_df_arima['Number of Tractor Sold'],
                             order=(1, 1, 1),
                             seasonal_order=(1, 1, 1, 12),
                             trend='n').fit()

arima_test_predictions = arima_model_eval.forecast(steps=len(test_df_arima))
mape_arima = calculate_mape(test_df_arima['Number of Tractor Sold'], arima_test_predictions)
st.write(f"ARIMA MAPE on the test set: **{mape_arima:.2f}%**")

# Retrain ARIMA model on full data for 2015 forecast
full_arima_model = ARIMA(df1['Number of Tractor Sold'],
                         order=(1, 1, 1),
                         seasonal_order=(1, 1, 1, 12),
                         trend='n').fit()
forecast_2015_arima = full_arima_model.forecast(steps=12)

# Plot ARIMA Results
fig_arima = go.Figure()
fig_arima.add_trace(go.Scatter(x=df1.index, y=df1['Number of Tractor Sold'], mode='lines', name='Historical Data'))
fig_arima.add_trace(go.Scatter(x=arima_test_predictions.index, y=arima_test_predictions, mode='lines', name='ARIMA Test Predictions', line=dict(dash='dash')))
fig_arima.add_trace(go.Scatter(x=forecast_2015_arima.index, y=forecast_2015_arima, mode='lines', name='ARIMA 2015 Forecast', line=dict(color='orange')))
fig_arima.update_layout(title='ARIMA: History, Test Predictions, and 2015 Forecast',
                        xaxis_title='Month-Year',
                        yaxis_title='Number of Tractor Sold',
                        hovermode='x unified')
st.plotly_chart(fig_arima, use_container_width=True)

st.subheader("Conclusion")
st.write(f"Holt-Winters MAPE: **{mape_hw:.2f}%**")
st.write(f"ARIMA MAPE: **{mape_arima:.2f}%**")
st.write("The Holt-Winters model achieved a slightly lower MAPE, indicating potentially better predictive performance for this dataset.")
