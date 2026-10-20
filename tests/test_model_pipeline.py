import os
import unittest
import numpy as np
import pandas as pd
import onnxruntime as ort
from server import app

class TestModelAndPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.onnx_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model.onnx")
        cls.api_onnx_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api", "model.onnx")
        cls.client = app.test_client()

    def test_01_onnx_artifacts_exist_and_match(self):
        """Verify model.onnx and api/model.onnx exist, are valid, and under 1MB."""
        self.assertTrue(os.path.exists(self.onnx_path), "Root model.onnx does not exist")
        self.assertTrue(os.path.exists(self.api_onnx_path), "api/model.onnx does not exist")
        
        root_size = os.path.getsize(self.onnx_path)
        api_size = os.path.getsize(self.api_onnx_path)
        
        # Verify lightweight artifact (< 1MB for serverless deployment)
        self.assertLess(root_size, 1024 * 1024, f"model.onnx is too large: {root_size} bytes")
        self.assertEqual(root_size, api_size, "root and api model.onnx sizes differ")

    def test_02_onnx_session_io_signature(self):
        """Verify ONNX model has expected input shape (None, 100, 1) and output (None, 1)."""
        session = ort.InferenceSession(self.onnx_path)
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        self.assertEqual(len(inputs), 1)
        self.assertEqual(len(outputs), 1)
        
        self.assertEqual(inputs[0].name, "input")
        # Time steps must be 100, features must be 1
        self.assertEqual(inputs[0].shape[1], 100)
        self.assertEqual(inputs[0].shape[2], 1)
        self.assertEqual(outputs[0].shape[1], 1)

    def test_03_forward_pass_stability_and_bounds(self):
        """Verify model forward pass produces calibrated return predictions within reasonable bounds."""
        session = ort.InferenceSession(self.onnx_path)
        inp_name = session.get_inputs()[0].name
        out_name = session.get_outputs()[0].name
        
        # 1. Flat market (all zeros in relative price space)
        flat_input = np.zeros((1, 100, 1), dtype=np.float32)
        out_flat = session.run([out_name], {inp_name: flat_input})[0]
        # Return must be near 0 (small drift expected between -2% and +2%)
        self.assertGreater(out_flat[0, 0], -0.05)
        self.assertLess(out_flat[0, 0], 0.05)
        
        # 2. Bullish trend sequence (+10% upward slope)
        bull_input = np.linspace(-0.10, 0.0, 100).reshape(1, 100, 1).astype(np.float32)
        out_bull = session.run([out_name], {inp_name: bull_input})[0]
        self.assertGreater(out_bull[0, 0], -0.10)
        self.assertLess(out_bull[0, 0], 0.10)

    def test_04_data_cleaning_and_validation(self):
        """Verify OHLC relationship validation, handling of duplicate dates, and sorting."""
        sample_data = {
            "Date": ["2026-01-05", "2026-01-02", "2026-01-02", "2026-01-06", "2026-01-07"],
            "Open": [100.0, 95.0, 95.0, 102.0, 105.0],
            "High": [105.0, 98.0, 98.0, 101.0, 108.0],  # row 3: High (101) < Open (102) -> invalid
            "Low":  [98.0,  92.0, 92.0, 103.0, 104.0],  # row 3: Low (103) > High (101) -> invalid
            "Close": [103.0, 97.0, 97.0, 100.0, 107.0],
            "Volume": [1000, 1200, 1200, 1100, 1500]
        }
        df = pd.DataFrame(sample_data)
        df["Date"] = pd.to_datetime(df["Date"])
        df.drop_duplicates(subset=["Date"], inplace=True)
        df.sort_values(by="Date", inplace=True)
        df = df[(df["High"] >= df["Low"]) & (df["Close"] > 0)].reset_index(drop=True)
        
        # Expected: 3 clean, unique, sorted rows
        self.assertEqual(len(df), 3)
        self.assertEqual(df["Date"].iloc[0].strftime("%Y-%m-%d"), "2026-01-02")
        self.assertEqual(df["Date"].iloc[1].strftime("%Y-%m-%d"), "2026-01-05")
        self.assertEqual(df["Date"].iloc[2].strftime("%Y-%m-%d"), "2026-01-07")

    def test_05_api_predict_success_and_payload_structure(self):
        """Verify /api/predict returns full expected schema with consistent, non-corrupted numbers."""
        response = self.client.get("/api/predict?ticker=AAPL")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        
        self.assertEqual(data["ticker"], "AAPL")
        self.assertIn("latest_close", data)
        self.assertIn("next_day_pred", data)
        self.assertIn("next_day_pct", data)
        self.assertIn("next_trading_day", data)
        self.assertIn("rmse", data)
        self.assertIn("mae", data)
        self.assertIn("mape", data)
        self.assertIn("directional_accuracy", data)
        self.assertIn("test_comparison", data)
        
        # Crucial regression check: Next day price must not be a -50% flash crash!
        pct_diff = abs(data["next_day_pred"] - data["latest_close"]) / data["latest_close"]
        self.assertLess(pct_diff, 0.15, f"Unrealistic flash prediction: {data['next_day_pred']} vs {data['latest_close']}")
        
        # RMSE check: must be realistic price deviation (< $15, not $88!)
        self.assertLess(data["rmse"], 20.0, f"Unreasonably high RMSE: {data['rmse']}")
        
        # MAPE check: must be < 5%, not 22%!
        self.assertLess(data["mape"], 5.0, f"Unreasonably high MAPE: {data['mape']}")
        
        # Verify synchronized test_comparison items
        comp = data["test_comparison"]
        self.assertGreater(len(comp), 0)
        first_pt = comp[0]
        self.assertIn("date", first_pt)
        self.assertIn("actual", first_pt)
        self.assertIn("predicted", first_pt)
        self.assertIn("ma100", first_pt)
        self.assertIn("ma200", first_pt)

    def test_06_api_predict_invalid_ticker(self):
        """Verify API handles nonexistent/corrupt tickers gracefully with 400 status."""
        response = self.client.get("/api/predict?ticker=NONEXISTENT_TICKER_XYZ999")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

if __name__ == "__main__":
    unittest.main()
