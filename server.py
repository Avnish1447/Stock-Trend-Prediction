import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request, send_from_directory
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model

app = Flask(__name__, static_folder="website")

# Preload model in memory
print("Loading LSTM model (keras_model.h5)...")
model = None
try:
    model = load_model("keras_model.h5")
    print("LSTM Model loaded successfully into memory!")
except Exception as e:
    print(f"Warning: Could not load keras_model.h5: {e}")


@app.route("/")
def index():
    return send_from_directory("website", "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("website", path)


@app.route("/api/predict", methods=["GET", "POST"])
def predict_stock():
    if model is None:
        return jsonify({"error": "LSTM model not loaded"}), 500

    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker and request.is_json:
        ticker = request.json.get("ticker", "").strip().upper()
    if not ticker:
        ticker = "AAPL"

    try:
        ticker_obj = yf.Ticker(ticker)
        # Fetch 5 years of daily data
        df = ticker_obj.history(period="5y")
        if df.empty or len(df) < 150:
            return jsonify({"error": f"Insufficient price data found for ticker '{ticker}' (at least 150 trading days required)."}), 400

        df.reset_index(inplace=True)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        elif "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()
        df.sort_values(by="Date", inplace=True)

        # Calculate Moving Averages
        df["MA100"] = df["Close"].rolling(window=100).mean()
        df["MA200"] = df["Close"].rolling(window=200).mean()

        # Split 70% train / 30% test
        train_size = int(len(df) * 0.70)
        data_training = pd.DataFrame(df["Close"].iloc[0:train_size])
        data_testing = pd.DataFrame(df["Close"].iloc[train_size:])

        # Fit scaler ONLY on training data to prevent data leakage
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data_training)

        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.transform(final_df)

        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        y_predicted = model.predict(x_test, verbose=0)
        y_pred_inv = scaler.inverse_transform(y_predicted).flatten()
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        test_dates = df["Date"].iloc[train_size:].dt.strftime("%Y-%m-%d").tolist()

        # Performance metrics
        rmse = float(np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)))
        mae = float(mean_absolute_error(y_test_inv, y_pred_inv))
        mape = float(np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100)

        # Next-Day Forecast projection
        last_100_days = df["Close"].tail(100).values.reshape(-1, 1)
        next_scaled = model.predict(scaler.transform(last_100_days).reshape(1, 100, 1), verbose=0)
        next_day_pred = float(scaler.inverse_transform(next_scaled)[0, 0])

        latest_close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        price_change = latest_close - prev_close
        pct_change = (price_change / prev_close) * 100

        next_day_change = next_day_pred - latest_close
        next_day_pct = (next_day_change / latest_close) * 100

        high_52w = float(df["Close"].tail(252).max())
        low_52w = float(df["Close"].tail(252).min())

        # Build comparison timeseries for plotting (last 200 points for responsiveness)
        step = max(1, len(test_dates) // 150)
        test_comparison = []
        for i in range(0, len(test_dates), step):
            test_comparison.append({
                "date": test_dates[i],
                "actual": round(float(y_test_inv[i]), 2),
                "predicted": round(float(y_pred_inv[i]), 2)
            })

        # Recent history for moving average overlay
        recent_df = df.tail(150)
        recent_history = []
        for _, row in recent_df.iterrows():
            recent_history.append({
                "date": row["Date"].strftime("%Y-%m-%d"),
                "close": round(float(row["Close"]), 2),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "ma100": round(float(row["MA100"]), 2) if pd.notnull(row["MA100"]) else None,
                "ma200": round(float(row["MA200"]), 2) if pd.notnull(row["MA200"]) else None,
            })

        return jsonify({
            "ticker": ticker,
            "latest_close": round(latest_close, 2),
            "prev_close": round(prev_close, 2),
            "price_change": round(price_change, 2),
            "pct_change": round(pct_change, 2),
            "next_day_pred": round(next_day_pred, 2),
            "next_day_change": round(next_day_change, 2),
            "next_day_pct": round(next_day_pct, 2),
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "mape": round(mape, 2),
            "total_records": len(df),
            "test_points": len(test_comparison),
            "test_comparison": test_comparison,
            "recent_history": recent_history
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=False)
