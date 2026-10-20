import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf

# Page setup
st.set_page_config(
    page_title="📈 Stock Trend Prediction & Analysis",
    page_icon="📈",
    layout="wide"
)

# Custom header styling
st.markdown("""
    <style>
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            border-left: 5px solid #4B8BBE;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Stock Trend Prediction App")
st.markdown("Forecast stock price trends and analyze historical moving averages using a deep learning **LSTM** model powered by **Yahoo Finance**.")

# Sidebar Configuration
st.sidebar.header("⚙️ Stock Selection")

ticker = st.sidebar.text_input("Stock Ticker Symbol", value="AAPL").upper().strip()

today = datetime.date.today()
default_start = today - datetime.timedelta(days=365 * 10)  # Default 10 years
col_start, col_end = st.sidebar.columns(2)
start_date = col_start.date_input("Start Date", value=default_start)
end_date = col_end.date_input("End Date", value=today)


class ONNXModelWrapper:
    def __init__(self, session):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

    def predict(self, x, verbose=0):
        return self.session.run([self.output_name], {self.input_name: np.array(x, dtype=np.float32)})[0]


@st.cache_resource
def get_lstm_model():
    """Load and cache the pre-trained LSTM model (preferring fast ONNX, falling back to Keras)."""
    # 1. Try ONNX model first (< 1MB, fast inference)
    if os.path.exists("model.onnx"):
        try:
            import onnxruntime as ort
            session = ort.InferenceSession("model.onnx")
            return ONNXModelWrapper(session)
        except Exception as e:
            pass

    # 2. Fallback to Keras model
    try:
        from tensorflow.keras.models import load_model
        model = load_model("keras_model.h5")
        return model
    except Exception as e:
        st.error(f"Failed to load model ('model.onnx' or 'keras_model.h5'): {e}")
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_stock_data(symbol: str, start: str, end: str):
    """Fetch historical stock data using Yahoo Finance."""
    ticker_obj = yf.Ticker(symbol)
    df = ticker_obj.history(start=start, end=end)
    if df.empty:
        raise ValueError(f"No price data found for ticker '{symbol}'. Please verify the symbol or date range.")
    df.reset_index(inplace=True)
    # Ensure timezone naive datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    elif 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.sort_values(by='Date', inplace=True)
    return df


def render_dashboard():
    if not ticker:
        st.info("Please enter a stock ticker in the sidebar to begin.")
        return

    # 1. Fetch Data
    df = None
    with st.spinner(f"Fetching historical data for {ticker}..."):
        try:
            df = fetch_stock_data(ticker, str(start_date), str(end_date))
        except Exception as err:
            st.error(f"⚠️ Error fetching stock data: {err}")
            return

    if df is None or len(df) < 150:
        st.error(f"Insufficient data returned ({len(df) if df is not None else 0} records). At least 150 trading days are required to compute moving averages and evaluate the 100-day LSTM model.")
        return

    # Calculate Moving Averages
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    # Summary Metrics Row
    latest_close = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    price_change = latest_close - prev_close
    pct_change = (price_change / prev_close) * 100
    high_52w = df['Close'].tail(252).max()
    low_52w = df['Close'].tail(252).min()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Close", f"${latest_close:,.2f}", f"{price_change:+,.2f} ({pct_change:+.2f}%)")
    col2.metric("52-Week High", f"${high_52w:,.2f}")
    col3.metric("52-Week Low", f"${low_52w:,.2f}")
    col4.metric("Total Records", f"{len(df):,} days")

    st.markdown("---")

    # Tabs for Visualizations
    tab_charts, tab_prediction, tab_data = st.tabs(["📊 Interactive Charts", "🔮 LSTM Predictions", "📋 Raw Data"])

    with tab_charts:
        st.subheader(f"📈 {ticker} Price & Moving Average Analysis")

        # Interactive Plotly Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price', line=dict(color='#1f77b4', width=2)))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA100'], name='100-Day MA', line=dict(color='#d62728', width=1.5, dash='dot')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA200'], name='200-Day MA', line=dict(color='#2ca02c', width=1.5, dash='dash')))

        fig.update_layout(
            title=f"{ticker} Historical Close Price with 100 & 200 Day Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View Static Matplotlib Chart"):
            fig_mpl, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df['Date'], df['Close'], label='Close Price', color='#1f77b4', linewidth=1.5)
            ax.plot(df['Date'], df['MA100'], label='100-Day MA', color='red', linestyle='--', linewidth=1.2)
            ax.plot(df['Date'], df['MA200'], label='200-Day MA', color='green', linestyle=':', linewidth=1.2)
            ax.set_title(f"{ticker} Close Price with 100 & 200-Day MAs")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig_mpl)

    with tab_prediction:
        st.subheader("🧠 Deep Learning Trend Forecast (LSTM)")

        model = get_lstm_model()
        if model is None:
            return

        with st.spinner("Processing test dataset and evaluating LSTM predictions..."):
            # 70% Training / 30% Testing split strictly chronologically
            train_size = int(len(df) * 0.70)
            closes = df['Close'].values

            x_test = []
            curr_prices = []
            actual_prices = []
            test_dates = []

            for i in range(train_size, len(df)):
                window = closes[i - 100:i]
                curr_p = float(window[-1])
                if curr_p <= 0:
                    continue
                # Stationary local window normalization (zero data leakage)
                norm_w = (window / curr_p) - 1.0
                x_test.append(norm_w.reshape(100, 1))
                curr_prices.append(curr_p)
                actual_prices.append(float(closes[i]))
                test_dates.append(df['Date'].iloc[i])

            if len(x_test) == 0:
                st.warning("Insufficient test sequences generated. Try extending the date range.")
                return

            x_test = np.array(x_test, dtype=np.float32)
            curr_prices = np.array(curr_prices)
            actual_prices = np.array(actual_prices)

            # Generate predictions (returns relative to anchor close)
            pred_returns = model.predict(x_test, verbose=0).flatten()
            y_predicted_inv = curr_prices * (1.0 + pred_returns)
            y_test_inv = actual_prices

            # Performance Metrics
            rmse = float(np.sqrt(mean_squared_error(y_test_inv, y_predicted_inv)))
            mae = float(mean_absolute_error(y_test_inv, y_predicted_inv))
            mape = float(np.mean(np.abs((y_test_inv - y_predicted_inv) / y_test_inv)) * 100)
            dir_acc = float(np.mean(np.sign(y_test_inv - curr_prices) == np.sign(y_predicted_inv - curr_prices)) * 100)

            # Next-Day Forecast projection from latest market observation
            last_100 = closes[-100:]
            latest_close = float(last_100[-1])
            last_norm = ((last_100 / latest_close) - 1.0).reshape(1, 100, 1).astype(np.float32)
            next_day_ret = float(model.predict(last_norm, verbose=0)[0, 0])
            next_day_pred = float(latest_close * (1.0 + next_day_ret))
            next_day_change = next_day_pred - latest_close
            next_day_pct = next_day_ret * 100

            # Calculate exact next trading day & 5-day horizon
            latest_dt = df['Date'].iloc[-1]
            next_trading_day = pd.bdate_range(latest_dt, periods=2)[1].strftime("%Y-%m-%d")
            trend_5d_date = pd.bdate_range(latest_dt, periods=6)[-1].strftime("%Y-%m-%d")

            ma20 = float(np.mean(closes[-20:]))
            ma20_ratio = (latest_close / ma20) - 1.0
            vol_20 = float(np.std(np.diff(closes[-20:]) / closes[-20:-1]))
            trend_5d_ret = float(np.clip(5.0 * next_day_ret + 0.3 * ma20_ratio, -0.15, 0.15))
            trend_5d_pred = float(latest_close * (1.0 + trend_5d_ret))
            trend_5d_change = trend_5d_pred - latest_close
            trend_5d_pct = float(trend_5d_ret * 100)

            z_score = abs(next_day_ret) / (vol_20 + 1e-6)
            if z_score >= 0.08 and np.sign(next_day_ret) == np.sign(ma20_ratio):
                conviction = "HIGH"
            elif z_score >= 0.04:
                conviction = "MODERATE"
            else:
                conviction = "BALANCED"

        # Forecast KPI Cards
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric(f"Next Day ({next_trading_day})", f"${next_day_pred:,.2f}", f"{next_day_change:+,.2f} ({next_day_pct:+.2f}%)")
        kpi2.metric(f"5-Day Trend ({trend_5d_date})", f"${trend_5d_pred:,.2f}", f"{trend_5d_change:+,.2f} ({trend_5d_pct:+.2f}%)")
        kpi3.metric("Conviction", conviction)
        kpi4.metric("RMSE / MAE", f"${rmse:.2f} / ${mae:.2f}")
        kpi5.metric("1D Hit Rate", f"{dir_acc:.1f}%")

        # Interactive Comparison Plot
        pred_fig = go.Figure()
        pred_fig.add_trace(go.Scatter(
            x=test_dates,
            y=y_test_inv,
            name='Actual Price',
            line=dict(color='#1f77b4', width=2)
        ))
        pred_fig.add_trace(go.Scatter(
            x=test_dates,
            y=y_predicted_inv,
            name='Predicted Price (LSTM)',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))

        pred_fig.update_layout(
            title=f"{ticker} Model Evaluation: Actual vs LSTM Predicted Price",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=520
        )
        st.plotly_chart(pred_fig, use_container_width=True)

        with st.expander("View Static Matplotlib Prediction Chart"):
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.plot(test_dates, y_test_inv, label='Actual Price', color='blue', linewidth=1.5)
            ax2.plot(test_dates, y_predicted_inv, label='Predicted Price', color='red', linestyle='--', linewidth=1.5)
            ax2.set_title(f"{ticker} Stock Price: Actual vs Predicted")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Price (USD)")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            st.pyplot(fig2)

    with tab_data:
        st.subheader(f"Historical Data Summary for {ticker}")
        st.write(df.describe())
        st.dataframe(df.sort_values(by='Date', ascending=False), use_container_width=True)


if __name__ == '__main__':
    render_dashboard()
