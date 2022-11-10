# Build a streamlit app that predicts future stock prices

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fbprophet import Prophet
from fbprophet.plot import plot_plotly

# Set the page layout to wide
st.set_page_config(layout="wide")

# Set the title
st.title("Stock Price Prediction App")

# Set the subheader
st.markdown("""
This app predicts the **future stock prices**!
""")
st.sidebar.header('User Input Parameters')

# Function to get the user input


def get_input():
    start_date = st.sidebar.text_input("Start Date", "2015-01-01")
    end_date = st.sidebar.text_input("End Date", "2020-12-31")
    stock_symbol = st.sidebar.text_input("Stock Symbol", "GOOG")
    return start_date, end_date, stock_symbol

# Function to get the company name


def get_company_name(symbol):
    if symbol == 'GOOG':
        return 'Google'
    elif symbol == 'AAPL':
        return 'Apple'
    elif symbol == 'MSFT':
        return 'Microsoft'
    elif symbol == 'GME':
        return 'Gamestop'
    else:
        'None'

# Function to get the proper company data and the proper timeframe


def get_data(symbol, start, end):
    # Load the data
    if symbol.upper() == 'GME':
        df = yf.download(symbol, start, end)
    else:
        df = yf.download(symbol, start, end)
    # Get the company name
    company_name = get_company_name(symbol.upper())
    # Rename the columns
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                   "Close": "close", "Adj Close": "adj_close", "Volume": "volume"})
    return df

# Function to plot the close price of the stock


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name="stock_close"))
    fig.layout.update(title_text="Time Series Data",
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Function to predict future prices


def predict_prices(df, days):
    df_train = df[['Date', 'close']]
    df_train = df_train.rename(columns={"Date": "ds", "close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    return forecast

# Function to plot the predicted prices


def plot_prediction(df, days):
    forecast = predict_prices(df, days)
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    fig2 = m.plot_components(forecast)
    st.write(fig2)


# Get the user input
start, end, symbol = get_input()

# Get the data
df = get_data(symbol, start, end)

# Plot the raw data
st.header('Raw Data')
plot_raw_data()

# Predict future prices
st.header('Forecast Data')

# Number of days to predict
days = st.slider('Days of prediction:', 1, 365)
st.write('You selected:', days, 'days')

# Show and plot the data
st.subheader('Forecast data')
forecast_data = predict_prices(df, days)
st.write(forecast_data.tail())

# Plot the forecast data
st.subheader('Forecast data')
plot_prediction(df, days)

# Show the data
st.subheader('Forecast components')
fig2 = m.plot_components(forecast_data)
st.write(fig2)
