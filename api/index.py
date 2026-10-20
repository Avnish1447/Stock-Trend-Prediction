import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import onnxruntime as ort

app = Flask(__name__)

# Locate and load model.onnx
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_candidates = [
    os.path.join(BASE_DIR, "model.onnx"),
    os.path.join(BASE_DIR, "..", "model.onnx"),
    "model.onnx"
]

session = None
input_name = None
output_name = None

for path in model_candidates:
    if os.path.exists(path):
        try:
            session = ort.InferenceSession(path)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            print(f"Loaded ONNX model from: {path}")
            break
        except Exception as e:
            print(f"Failed to load ONNX model from {path}: {e}")

if session is None:
    print("Warning: Could not locate model.onnx")


POPULAR_COMPANIES = {
    "AAPL": "Apple Inc.",
    "NVDA": "NVIDIA Corporation",
    "TSLA": "Tesla, Inc.",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com, Inc.",
    "GOOGL": "Alphabet Inc.",
    "GOOG": "Alphabet Inc.",
    "META": "Meta Platforms, Inc.",
    "NFLX": "Netflix, Inc.",
    "AMD": "Advanced Micro Devices, Inc.",
    "INTC": "Intel Corporation",
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust",
    "COIN": "Coinbase Global, Inc.",
    "PLTR": "Palantir Technologies Inc.",
    "BABA": "Alibaba Group Holding Ltd.",
    "DIS": "The Walt Disney Company",
    "JPM": "JPMorgan Chase & Co.",
    "V": "Visa Inc.",
    "WMT": "Walmart Inc.",
    "BA": "The Boeing Company",
    "CRM": "Salesforce, Inc.",
    "UBER": "Uber Technologies, Inc."
}

def resolve_company_name(ticker, ticker_obj=None):
    if ticker in POPULAR_COMPANIES:
        return POPULAR_COMPANIES[ticker]
    if ticker_obj is not None:
        try:
            name = ticker_obj.info.get("shortName") or ticker_obj.info.get("longName")
            if name:
                return name
        except Exception:
            pass
    return ticker


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>", methods=["GET", "POST", "OPTIONS"])
def handler(path):
    if request.method == "OPTIONS":
        return "", 200

    if session is None:
        return jsonify({"error": "ONNX Model session not initialized"}), 500

    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker and request.is_json:
        ticker = request.json.get("ticker", "").strip().upper()
    if not ticker:
        ticker = "AAPL"

    try:
        import time
        df = pd.DataFrame()
        # Fetch 5 years of daily data with retries
        for attempt in range(3):
            try:
                ticker_obj = yf.Ticker(ticker)
                df = ticker_obj.history(period="5y")
                if not df.empty and len(df) >= 150:
                    break
            except Exception:
                pass
            time.sleep(1)

        if df.empty or len(df) < 150:
            return jsonify({
                "error": f"Insufficient price data found for ticker '{ticker}' (at least 150 trading days required)."
            }), 400

        df.reset_index(inplace=True)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        elif "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()
        df.drop_duplicates(subset=["Date"], inplace=True)
        df.sort_values(by="Date", inplace=True)

        # Sanity check: valid OHLC relationships
        df = df[(df["High"] >= df["Low"]) & (df["Close"] > 0)].reset_index(drop=True)

        # Calculate Moving Averages
        df["MA100"] = df["Close"].rolling(window=100).mean()
        df["MA200"] = df["Close"].rolling(window=200).mean()

        # Split 70% train / 30% test strictly chronologically
        train_size = int(len(df) * 0.70)
        closes = df["Close"].values

        x_test = []
        curr_prices = []
        actual_prices = []
        test_dates = []
        test_ma100 = []
        test_ma200 = []

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
            test_dates.append(df["Date"].iloc[i].strftime("%Y-%m-%d"))
            test_ma100.append(float(df["MA100"].iloc[i]) if pd.notnull(df["MA100"].iloc[i]) else None)
            test_ma200.append(float(df["MA200"].iloc[i]) if pd.notnull(df["MA200"].iloc[i]) else None)

        if len(x_test) == 0:
            return jsonify({"error": "Unable to generate valid sequence windows for evaluation."}), 400

        x_test = np.array(x_test, dtype=np.float32)
        curr_prices = np.array(curr_prices)
        actual_prices = np.array(actual_prices)

        # ONNX Inference
        pred_returns = session.run([output_name], {input_name: x_test})[0].flatten()
        pred_prices = curr_prices * (1.0 + pred_returns)

        # Performance metrics (1-day horizon)
        rmse = float(np.sqrt(mean_squared_error(actual_prices, pred_prices)))
        mae = float(mean_absolute_error(actual_prices, pred_prices))
        mape = float(np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100)
        dir_acc = float(np.mean(np.sign(actual_prices - curr_prices) == np.sign(pred_prices - curr_prices)) * 100)

        # Multi-day 5-day weekly horizon backtest
        if len(actual_prices) > 10:
            actual_5d = closes[train_size + 5:]
            curr_5d = curr_prices[:-5]
            pred_5d_proxy = pred_prices[:-5]
            # Empirical 5-day directional hit rate on holdout test set
            dir_acc_5d = float(np.mean(np.sign(actual_5d - curr_5d) == np.sign(pred_5d_proxy - curr_5d)) * 100)
        else:
            dir_acc_5d = dir_acc

        # Next-Day Forecast projection from latest market observation
        last_100 = closes[-100:]
        latest_close = float(last_100[-1])
        last_norm = ((last_100 / latest_close) - 1.0).reshape(1, 100, 1).astype(np.float32)

        next_day_ret = float(session.run([output_name], {input_name: last_norm})[0][0, 0])
        next_day_pred = float(latest_close * (1.0 + next_day_ret))
        prev_close = float(df["Close"].iloc[-2])
        price_change = latest_close - prev_close
        pct_change = (price_change / prev_close) * 100

        next_day_change = next_day_pred - latest_close
        next_day_pct = next_day_ret * 100

        # Exact trading day calendar dates
        latest_dt = df["Date"].iloc[-1]
        next_trading_day = pd.bdate_range(latest_dt, periods=2)[1].strftime("%Y-%m-%d")
        trend_5d_target_date = pd.bdate_range(latest_dt, periods=6)[-1].strftime("%Y-%m-%d")

        # 5-Day Weekly Trend Projection & Conviction Scoring
        ma20 = float(np.mean(closes[-20:]))
        ma20_ratio = (latest_close / ma20) - 1.0
        vol_20 = float(np.std(np.diff(closes[-20:]) / closes[-20:-1]))

        trend_5d_ret = float(np.clip(5.0 * next_day_ret + 0.3 * ma20_ratio, -0.15, 0.15))
        trend_5d_pred = float(latest_close * (1.0 + trend_5d_ret))
        trend_5d_change = trend_5d_pred - latest_close
        trend_5d_pct = float(trend_5d_ret * 100)
        trend_5d_dir = "BULLISH" if trend_5d_ret > 0.005 else ("BEARISH" if trend_5d_ret < -0.005 else "NEUTRAL")

        # Signal-to-noise conviction index
        z_score = abs(next_day_ret) / (vol_20 + 1e-6)
        if z_score >= 0.08 and np.sign(next_day_ret) == np.sign(ma20_ratio):
            conviction_level = "HIGH CONVICTION"
        elif z_score >= 0.04:
            conviction_level = "MODERATE CONVICTION"
        else:
            conviction_level = "BALANCED / LOW NOISE"

        high_52w = float(df["Close"].tail(252).max())
        low_52w = float(df["Close"].tail(252).min())

        # Build comparison timeseries for plotting (capped at ~150 points for performance)
        step = max(1, len(test_dates) // 150)
        test_comparison = []
        for i in range(0, len(test_dates), step):
            test_comparison.append({
                "date": test_dates[i],
                "actual": round(float(actual_prices[i]), 2),
                "predicted": round(float(pred_prices[i]), 2),
                "ma100": round(float(test_ma100[i]), 2) if test_ma100[i] is not None else None,
                "ma200": round(float(test_ma200[i]), 2) if test_ma200[i] is not None else None,
            })

        # Recent history for moving average overlay (preserved for API compatibility)
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

        company_name = resolve_company_name(ticker, ticker_obj)
        logo_url = f"https://assets.parqet.com/logos/symbol/{ticker}"

        return jsonify({
            "ticker": ticker,
            "company_name": company_name,
            "logo_url": logo_url,
            "latest_close": round(latest_close, 2),
            "prev_close": round(prev_close, 2),
            "price_change": round(price_change, 2),
            "pct_change": round(pct_change, 2),
            "next_day_pred": round(next_day_pred, 2),
            "next_day_change": round(next_day_change, 2),
            "next_day_pct": round(next_day_pct, 2),
            "next_trading_day": next_trading_day,
            "trend_5d_pred": round(trend_5d_pred, 2),
            "trend_5d_change": round(trend_5d_change, 2),
            "trend_5d_pct": round(trend_5d_pct, 2),
            "trend_5d_dir": trend_5d_dir,
            "trend_5d_target_date": trend_5d_target_date,
            "conviction_level": conviction_level,
            "directional_accuracy_5d": round(dir_acc_5d, 2),
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "mape": round(mape, 2),
            "directional_accuracy": round(dir_acc, 2),
            "total_records": len(df),
            "test_points": len(test_comparison),
            "test_comparison": test_comparison,
            "recent_history": recent_history
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=3000, debug=True)
