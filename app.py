# Build a streamlit app that uses data from yfinance and uses fbprophet to predict stock prices

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

st.title('Stock Price Prediction')

st.write("""
# Stock Price Prediction
Shown are the stock **closing price** and **volume** of Google!
""")

# Define the ticker symbol
tickerSymbol = 'GOOGL'

# Get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

# Get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-5-31')

st.write("""
## Closing Price
""")
st.line_chart(tickerDf.Close)
st.write("""
## Volume Price
""")
st.line_chart(tickerDf.Volume)

# Predict future prices
df_train = tickerDf[['Close']]
df_train = df_train.reset_index()
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

st.write("""
## Predicted Closing Price
""")
fig1 = m.plot(forecast)
st.pyplot(fig1)

st.write("""
## Predicted Closing Price Components
""")
fig2 = m.plot_components(forecast)
st.pyplot(fig2)
