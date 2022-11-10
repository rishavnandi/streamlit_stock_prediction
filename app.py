# Build a streamlit app that uses data from yfinance and uses prophet to predict the future stock price

import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.title('Stock Price Prediction App')

# Create a sidebar
# Add a subheader in the sidebar
st.sidebar.subheader('User Input Features')

# Create a function to get the user input


def get_input():
    start_date = st.sidebar.text_input('Start Date', '2010-01-01')
    end_date = st.sidebar.text_input('End Date', '2021-01-01')
    stock_symbol = st.sidebar.text_input('Stock Symbol', 'AMZN')
    return start_date, end_date, stock_symbol

# Create a function to get the company name


def get_company_name(symbol):
    if symbol == 'AMZN':
        return 'Amazon'
    elif symbol == 'GOOG':
        return 'Google'
    elif symbol == 'MSFT':
        return 'Microsoft'
    elif symbol == 'FB':
        return 'Facebook'
    else:
        'None'

# Create a function to get the proper company data and the proper timeframe from the user input


def get_data(symbol, start, end):
    # Load the data
    if symbol.upper() == 'AMZN':
        df = yf.download(symbol, start, end)
    else:
        df = yf.download(symbol, start, end)
    # Get the date, the open price, the high, the low and the close price
    df = df[['Open', 'High', 'Low', 'Close']]
    # Rename the columns
    df.columns = ['open', 'high', 'low', 'close']
    # Reset the index
    df.reset_index(inplace=True)
    return df

# Create a function to predict the future stock price


def predict_price(df, symbol):
    # Create a new dataframe
    new_df = df[['ds', 'y']]
    # Rename the columns
    new_df = new_df.rename(columns={'ds': 'ds', 'y': 'y'})
    # Create and train the model
    m = Prophet()
    m.fit(new_df)
    # Create a dataframe to hold the dates and the future price
    future = m.make_future_dataframe(periods=365)
    # Predict the future price
    forecast = m.predict(future)
    # Create and show the plot
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    # Show the forecast data
    st.write(forecast.tail())
    # Show the forecast components
    fig2 = m.plot_components(forecast)
    st.write(fig2)


# Get the user input
start, end, symbol = get_input()

# Get the data
df = get_data(symbol, start, end)

# Get the company name
company_name = get_company_name(symbol.upper())

# Display the close price
st.header(company_name + ' Close Price\n')
st.line_chart(df['close'])

# Display the data
st.header('Data')
st.write(df)

# Display the statistics
st.header('Data Statistics')
st.write(df.describe())

# Predict the future price
st.header('Predict Future Price')
predict_price(df, symbol)
