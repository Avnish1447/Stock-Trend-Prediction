import os
import shutil
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tf2onnx
import onnx
import onnxruntime as ort

# Ensure deterministic execution
np.random.seed(42)
tf.random.set_seed(42)

def train_and_export():
    print("=" * 60)
    print("STOCK TREND PREDICTION — REPRODUCIBLE MODEL TRAINING")
    print("=" * 60)
    
    # 1. Multi-Asset Training Universe (diversified market representation)
    universe = [
        'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'META',
        'SPY', 'QQQ', 'JPM', 'V', 'WMT', 'PG', 'XOM'
    ]
    
    train_seqs = []
    train_targs = []
    val_seqs = []
    val_targs = []
    
    # Strict chronological partition:
    # Historical Training: 2018-01-01 to 2023-12-31
    # Historical Validation: 2024-01-01 to 2024-12-31
    # Final Test: 2025-01-01 to present (kept strictly untouched during training)
    
    print("\n[Step 1/5] Fetching historical data and creating stationary causal sequences...")
    for ticker in universe:
        try:
            print(f"  Processing {ticker}...")
            df = yf.Ticker(ticker).history(start="2018-01-01", end="2024-12-31")
            if df.empty or len(df) < 150:
                continue
            df.reset_index(inplace=True)
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            
            # Data quality checks
            df = df.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'])
            df = df[(df['High'] >= df['Low']) & (df['Close'] > 0)]
            df.sort_values(by='Date', inplace=True)
            
            closes = df['Close'].values
            dates = df['Date'].values
            
            for i in range(100, len(closes) - 1):
                window = closes[i-100:i]
                target_next = closes[i]
                target_date = dates[i]
                
                curr_p = window[-1]
                if curr_p <= 0:
                    continue
                
                # Stationary local window normalization (zero leakage)
                norm_w = (window / curr_p) - 1.0
                norm_t = (target_next / curr_p) - 1.0
                
                # Filter anomalies (splits, data glitches > 25%)
                if np.abs(norm_t) > 0.25:
                    continue
                
                if target_date < pd.Timestamp("2024-01-01"):
                    train_seqs.append(norm_w.reshape(100, 1))
                    train_targs.append(norm_t)
                else:
                    val_seqs.append(norm_w.reshape(100, 1))
                    val_targs.append(norm_t)
        except Exception as e:
            print(f"  Warning: failed on {ticker}: {e}")
            
    X_train = np.array(train_seqs, dtype=np.float32)
    y_train = np.array(train_targs, dtype=np.float32)
    X_val = np.array(val_seqs, dtype=np.float32)
    y_val = np.array(val_targs, dtype=np.float32)
    
    print(f"\n[Step 2/5] Prepared Dataset Shapes:")
    print(f"  Train: {X_train.shape[0]} sequences")
    print(f"  Validation: {X_val.shape[0]} sequences")
    
    # 2. Build LSTM Model
    print("\n[Step 3/5] Defining LSTM architecture...")
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(100, 1), name="lstm_layer_1"),
        Dropout(0.2, name="dropout_1"),
        LSTM(64, return_sequences=False, name="lstm_layer_2"),
        Dropout(0.2, name="dropout_2"),
        Dense(32, activation='relu', name="dense_1"),
        Dense(1, activation='linear', name="dense_output")
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae']
    )
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
    ]
    
    print("\n[Step 4/5] Training LSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )
    
    # 3. Model Persistence (Keras and ONNX)
    print("\n[Step 5/5] Exporting model to Keras HDF5 and lightweight ONNX...")
    keras_path = "keras_model.h5"
    model.save(keras_path)
    print(f"  Saved Keras model to: {keras_path} ({os.path.getsize(keras_path):,} bytes)")
    
    # Convert to ONNX using tf.function wrapper
    onnx_path = "model.onnx"
    api_onnx_path = os.path.join("api", "model.onnx")
    
    @tf.function
    def serve_model(x):
        return model(x)
        
    onnx_model, _ = tf2onnx.convert.from_function(
        serve_model,
        input_signature=[tf.TensorSpec([None, 100, 1], tf.float32, name="input")],
        opset=13,
        output_path=onnx_path
    )
    
    # Verify ONNX model integrity
    onnx.checker.check_model(onnx_path)
    print(f"  Saved root ONNX model to: {onnx_path} ({os.path.getsize(onnx_path):,} bytes)")
    
    # Sync with api/model.onnx
    shutil.copyfile(onnx_path, api_onnx_path)
    print(f"  Synced ONNX model to: {api_onnx_path} ({os.path.getsize(api_onnx_path):,} bytes)")
    
    # Quick sanity check with ONNX Runtime
    session = ort.InferenceSession(onnx_path)
    test_dummy = np.zeros((1, 100, 1), dtype=np.float32)
    dummy_out = session.run(None, {session.get_inputs()[0].name: test_dummy})[0]
    print(f"  ONNX Inference Verification: dummy zero input -> output: {dummy_out[0, 0]:.6f}")
    
    print("\nModel training and ONNX conversion complete successfully!")

if __name__ == '__main__':
    train_and_export()
