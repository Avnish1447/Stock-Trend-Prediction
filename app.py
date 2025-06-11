import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="LSTM Stock Data Explorer", layout="wide")

st.title("LSTM Stock Data Explorer")

# Sidebar inputs
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Ticker (US stocks only)", value="TSLA")
api_key = st.sidebar.text_input("Alpha Vantage API Key", value="GTFJMDMRP40HCBL8Q")
download_button = st.sidebar.button("Download & Process Data")

@st.cache_data(show_spinner=True)
def fetch_stock_data(ticker, api_key):
    base_url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': ticker,
        'outputsize': 'full',
        'datatype': 'json',
        'apikey': api_key
    }
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if 'Time Series (Daily)' not in data:
                raise ValueError(f"API error: {data.get('Note') or data.get('Error Message') or 'Unknown error'}")
            raw_data = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(raw_data, orient='index').rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df.astype(float)
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Date'}, inplace=True)
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            return df
        except Exception as e:
            time.sleep(5)
    raise RuntimeError("All download attempts failed.")

if download_button:
    try:
        df = fetch_stock_data(ticker, api_key)
        st.success(f"Data for {ticker} downloaded successfully! Shape: {df.shape}")
        st.dataframe(df.head())

        # Data integrity checks
        st.write("**Missing values per column:**")
        st.write(df.isnull().sum())

        # Compute moving averages
        ma100 = df['Close'].rolling(window=100).mean()
        ma200 = df['Close'].rolling(window=200).mean()

        # Plot Close with 100-day & 200-day MA
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Date'], df['Close'], label='Close Price')
        ax.plot(df['Date'], ma100, 'r', label='100-Day MA')
        ax.plot(df['Date'], ma200, 'g', label='200-Day MA')
        ax.set_title(f"{ticker} Close Price with 100-Day and 200-Day Moving Averages")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Split data
        train_size = int(len(df) * 0.70)
        data_training = pd.DataFrame(df['Close'][0:train_size])
        data_testing = pd.DataFrame(df['Close'][train_size:])

        st.write(f"Training data shape: {data_training.shape}")
        st.write(f"Testing data shape: {data_testing.shape}")
        st.write("**First five rows of testing data:**")
        st.dataframe(data_testing.head())

        # Normalize
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)
        st.write("**First 5 rows of normalized training data:**")
        st.write(data_training_array[:5])

        # Prepare LSTM sequences (example for user, not actual model training here)
        x_train = []
        y_train = []
        for i in range(100, data_training_array.shape[0]):
            x_train.append(data_training_array[i-100:i])
            y_train.append(data_training_array[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        st.write(f"x_train shape: {x_train.shape}")
        st.write(f"y_train shape: {y_train.shape}")

        st.success("Data processing complete. Ready for LSTM modeling!")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Enter a ticker and API key, then click 'Download & Process Data'.")

st.markdown(
    """
    ---
    **Note**: This app demonstrates data download, preprocessing, and visualization steps for LSTM-based stock prediction. 
    Model training and prediction are not included in this demo.
    """
)
