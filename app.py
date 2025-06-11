import pandas as pd
import numpy as np
import time
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import requests
import io

# Alpha Vantage API setup
API_KEY = "GTFJMDMRP40HCBL8q"

@st.cache_data(show_spinner=True, ttl=3600)
def fetch_stock_data_alpha_vantage(ticker):
    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
        f"&symbol={ticker}&outputsize=full&datatype=csv&apikey={API_KEY}"
    )

    max_retries = 3
for attempt in range(max_retries):
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'Time Series (Daily)' not in data:
            raise ValueError(f"Invalid API response: {data.get('Note') or data.get('Error Message') or 'Unknown error'}")

        # Parse the JSON into a DataFrame
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

        # Reorder columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"✅ Data saved to {csv_path}")
        break

    except Exception as e:
        print(f"⚠️ Attempt {attempt + 1} failed: {e}")
        time.sleep(5)
else:
    raise RuntimeError("❌ All download attempts failed.")

def plot_close_price(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Date'], df['Close'], label='Close Price')
    ax.set_title(f'{ticker} Closing Price vs Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True)
    ax.legend()
    return fig

# Streamlit app starts here
st.title("📈 Stock Trend Prediction")

ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
start = pd.to_datetime('2000-01-01')
end = pd.to_datetime('2024-12-31')

if ticker:
    try:
        df = fetch_stock_data_alpha_vantage(ticker)
        df = df[(df['Date'] >= start) & (df['Date'] <= end)]

        st.subheader(f'Data Summary for {ticker} ({start.date()} to {end.date()})')
        st.write(df.describe())

        st.subheader('📉 Closing Price vs Time')
        fig = plot_close_price(df)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error fetching data: {e}")

# Plot with 100-Day Moving Average
st.subheader('📊 Closing Price with 100-Day Moving Average')
ma100 = df['Close'].rolling(window=100).mean()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Date'], df['Close'], label='Close Price')
ax.plot(df['Date'], ma100, 'r', label='100-Day MA')
ax.set_title(f'{ticker} Close Price with 100-Day Moving Average')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Plot with 200-Day Moving Average
st.subheader('📊 Closing Price with 200-Day Moving Average')
ma200 = df['Close'].rolling(window=200).mean()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Date'], df['Close'], label='Close Price')
ax.plot(df['Date'], ma200, 'g', label='200-Day MA')
ax.set_title(f'{ticker} Close Price with 200-Day Moving Average')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Combined MA plot
st.subheader('📊 Closing Price with 100 & 200-Day Moving Averages')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Date'], df['Close'], label='Close Price')
ax.plot(df['Date'], ma100, 'r', label='100-Day MA')
ax.plot(df['Date'], ma200, 'g', label='200-Day MA')
ax.set_title(f'{ticker} Close Price with 100 & 200-Day MAs')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Prepare data for LSTM model
train_size = int(len(df) * 0.70)
data_training = pd.DataFrame(df['Close'][0:train_size])
data_testing = pd.DataFrame(df['Close'][train_size:])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Load trained model
model = load_model('keras_model.h5')

# Prepare testing sequences
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Predict and inverse scale
y_predicted = model.predict(x_test)
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final visualization
st.subheader('🔮 Predicted vs Original Stock Price')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.title(f'{ticker} Stock Price: Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
st.pyplot(fig2)
