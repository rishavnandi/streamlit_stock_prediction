# Build a streamlit app that predicts stock prices

import streamlit as st
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

# Set page layout to wide
st.set_page_config(layout="wide")

st.title('Stock Price Prediction App')

# Create a sidebar header
st.sidebar.header('User Input Parameters')

# Create a function to get the user input


def get_input():
    start_date = st.sidebar.text_input("Start Date", "2015-01-01")
    end_date = st.sidebar.text_input("End Date", "2020-12-31")
    stock_symbol = st.sidebar.text_input("Stock Symbol", "GOOG")
    return start_date, end_date, stock_symbol

# Create a function to get the company name


def get_company_name(symbol):
    if symbol == 'GOOG':
        return 'Google'
    elif symbol == 'MSFT':
        return 'Microsoft'
    elif symbol == 'AMZN':
        return 'Amazon'
    elif symbol == 'FB':
        return 'Facebook'
    elif symbol == 'AAPL':
        return 'Apple'
    elif symbol == 'TSLA':
        return 'Tesla'
    elif symbol == 'SPY':
        return 'S&P 500'
    elif symbol == 'VTI':
        return 'Vanguard Technology ETF'
    else:
        'None'

# Create a function to get the proper company data and the proper timeframe from the user start date to the end date


@st.cache
def get_data(symbol, start, end):
    # Load the data
    if symbol.upper() == 'GOOG':
        df = yf.download(symbol, start, end)
    else:
        # Set the ticker symbol
        tickerSymbol = symbol.upper()
        # Get the data for this ticker
        tickerData = yf.Ticker(tickerSymbol)
        # Get the historical prices for this ticker
        df = tickerData.history(period='1d', start=start, end=end)
    # Return the dataframe
    return df

# Create a function to plot the closing price of the stock


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="stock_close"))
    fig.layout.update(title_text="Time Series Data",
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Create a function to predict stock prices using Facebook Prophet


def predict_prices(df, symbol):
    # Rename the columns
    df = df.reset_index()
    df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    # Instantiate the model
    model = Prophet()
    # Fit the model
    model.fit(df)
    # Create a dataframe to hold the future dates
    future_dates = model.make_future_dataframe(periods=365)
    # Make predictions
    prediction = model.predict(future_dates)
    # Create a dataframe to hold the predictions
    df = prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    # Rename the columns
    df.rename(columns={'ds': 'Date', 'yhat': 'Predictions'}, inplace=True)
    # Set the date as the index
    df.set_index('Date', inplace=True)
    # Plot the predictions
    fig1 = plot_plotly(model, prediction)
    st.plotly_chart(fig1)
    # Show the data
    st.write(df)


# Get the user input
start, end, symbol = get_input()

# Get the data
df = get_data(symbol, start, end)

# Get the company name
company_name = get_company_name(symbol.upper())

# Display the closing price
st.header(company_name + ' Close Price\n')
st.line_chart(df['Close'])

# Display the volume
st.header(company_name + ' Volume\n')
st.line_chart(df['Volume'])

# Show the raw data
if st.button('Show Raw Data'):
    st.write(df)

# Predict stock prices
st.header('Predict Stock Prices')
st.write('Predict closing prices for the next 365 days')
predict_prices(df, symbol)

# Plot the raw data
plot_raw_data()
