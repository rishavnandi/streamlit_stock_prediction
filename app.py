import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

with st.sidebar:
    st.title("Stock Price Prediction App")
    START = st.date_input("Start Date", value=date(2012, 1, 1))
    stocks = ("AAPL", "GOOGL", "MSFT", "AMZN", "TSLA",
              "META", "VGT", "SPY", "QQQ", "NVDA", "ADBE",)
    selected_stock = st.selectbox("Select dataset for prediction", stocks)
    n_years = st.slider("Years of prediction:", 1, 10)

TODAY = date.today().strftime("%Y-%m-%d")
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.info("Loading data...")
data = load_data(selected_stock)
data_load_state.success("Loading data...done!")

st.subheader('Raw data')
st.write(data.tail())


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],
                  y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'],
                  y=data['Close'], name="stock_close"))
    fig.layout.update(title_text="Time Series data with Rangeslider",
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.subheader(f"Forecast plot for {n_years} years")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.subheader("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

if st.button("Show raw data"):
    plot_raw_data()
