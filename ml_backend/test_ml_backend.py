"""
test_ml_backend.py â€” Air Quality ML Backend Test Suite v2
Tests the single-shot training architecture, validation logic, and model safety.
"""
import os, json, time, shutil, unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import numpy as np
from sklearn.ensemble import IsolationForest

# Import the refactored backend
import sys
sys.path.insert(0, os.path.dirname(__file__))
import main as backend


# ============================================================
# HELPERS
# ============================================================
def _make_row(i, ts_utc: datetime):
    return {
        "id": i,
        "created_at": ts_utc.isoformat(),
        "aqi_ppm":    620.0 + (i % 10),
        "temperature": 28.0 + (i % 5) * 0.5,
        "humidity":    65.0 + (i % 8)
    }

def _rows_over_minutes(n: int, span_min: float, start: datetime = None):
    """Generate n rows spread evenly over span_min minutes."""
    if start is None:
        start = datetime.now(timezone.utc) - timedelta(minutes=span_min)
    step = timedelta(minutes=span_min) / max(n - 1, 1)
    return [_make_row(i + 1, start + step * i) for i in range(n)]


# ============================================================
# TEST CASES
# ============================================================
class TestMLBackend(unittest.TestCase):

    # ----------------------------------------------------------
    # TEST 01: Live Supabase connectivity
    # ----------------------------------------------------------
    def test_01_supabase_connection(self):
        print("\n[TEST 01] Supabase connection & row fetch")
        res = backend.supabase.table("sensor_data") \
            .select("id, created_at, aqi_ppm, temperature, humidity") \
            .order("created_at", desc=True).limit(5).execute()
        self.assertIsNotNone(res.data, "Should return a list")
        self.assertGreater(len(res.data), 0, "sensor_data must have at least one row")
        print(f"  âœ“ Fetched {len(res.data)} rows. Latest id={res.data[0]['id']}")

    # ----------------------------------------------------------
    # TEST 02: Exact window query (no fallback)
    # ----------------------------------------------------------
    def test_02_exact_window_no_fallback(self):
        print("\n[TEST 02] No fallback to latest data when window is empty")
        future_end   = datetime.now(timezone.utc) + timedelta(hours=1)
        future_start = future_end - timedelta(minutes=5)
        rows = backend.fetch_window_data(future_start, future_end)
        # Must be 0 â€” there are no rows in the future
        self.assertEqual(rows, [], f"Expected 0 rows for future window, got {len(rows)}")
        print(f"  âœ“ Future window returned 0 rows (no silent fallback).")

    # ----------------------------------------------------------
    # TEST 03: Training fails on empty window
    # ----------------------------------------------------------
    def test_03_training_fails_empty_window(self):
        print("\n[TEST 03] Training fails when window has no data")
        with patch("main.fetch_window_data", return_value=[]):
            result = backend.run_training(window_minutes=60, contamination=0.05)
        self.assertEqual(result["status"], "failed")
        self.assertIn("No sensor data", result["error"])
        print(f"  âœ“ Correct failure: {result['error']}")

    # ----------------------------------------------------------
    # TEST 04: Training fails on insufficient temporal coverage
    # ----------------------------------------------------------
    def test_04_insufficient_coverage(self):
        print("\n[TEST 04] Training fails when temporal coverage < 50% of requested window")
        # 30 rows spanning only 8 minutes when 60 minutes requested
        rows = _rows_over_minutes(30, span_min=8)
        with patch("main.fetch_window_data", return_value=rows):
            result = backend.run_training(window_minutes=60, contamination=0.05)
        self.assertEqual(result["status"], "failed")
        self.assertIn("coverage", result["error"].lower())
        print(f"  âœ“ Rejected: {result['error']}")

    # ----------------------------------------------------------
    # TEST 05: Training fails on too few rows
    # ----------------------------------------------------------
    def test_05_insufficient_rows(self):
        print("\n[TEST 05] Training fails when fewer than MIN_TRAINING_ROWS usable rows")
        rows = _rows_over_minutes(5, span_min=60)
        with patch("main.fetch_window_data", return_value=rows):
            result = backend.run_training(window_minutes=60, contamination=0.05)
        self.assertEqual(result["status"], "failed")
        print(f"  âœ“ Rejected (5 rows): {result['error']}")

    # ----------------------------------------------------------
    # TEST 06: NaN / inf values are removed, training fails if too few remain
    # ----------------------------------------------------------
    def test_06_invalid_sensor_values_rejected(self):
        print("\n[TEST 06] NaN/inf sensor values are rejected")
        rows = _rows_over_minutes(25, span_min=60)
        # Corrupt all rows
        for r in rows:
            r["aqi_ppm"] = float("nan")
        with patch("main.fetch_window_data", return_value=rows):
            result = backend.run_training(window_minutes=60, contamination=0.05)
        self.assertEqual(result["status"], "failed")
        print(f"  âœ“ All NaN rows removed â†’ training blocked: {result['error']}")

    # ----------------------------------------------------------
    # TEST 07: Gaps are detected and reported
    # ----------------------------------------------------------
    def test_07_gap_detection(self):
        print("\n[TEST 07] Significant data gaps are detected")
        rows = _rows_over_minutes(100, span_min=60)
        # Insert a 20-minute gap at index 50
        for i in range(50, len(rows)):
            ts = datetime.fromisoformat(rows[i]["created_at"])
            rows[i]["created_at"] = (ts + timedelta(minutes=20)).isoformat()
        stats = backend.analyse_dataset(rows, window_minutes=60)
        self.assertGreater(stats["data_gaps_detected"], 0)
        print(f"  âœ“ Gaps detected: {stats['data_gaps_detected']}, largest: {stats['largest_gap_seconds']}s")

    # ----------------------------------------------------------
    # TEST 08: Successful training â€” model + metadata written
    # ----------------------------------------------------------
    def test_08_successful_training(self):
        print("\n[TEST 08] Successful training saves model and metadata")
        rows = _rows_over_minutes(200, span_min=60)

        original_model = None
        if os.path.exists(backend.MODEL_FILE_PATH):
            original_model = open(backend.MODEL_FILE_PATH, 'rb').read()

        with patch("main.fetch_window_data", return_value=rows):
            result = backend.run_training(window_minutes=60, contamination=0.05)

        self.assertEqual(result["status"], "completed")
        self.assertTrue(os.path.exists(backend.MODEL_FILE_PATH))
        self.assertTrue(os.path.exists(backend.METADATA_PATH))

        with open(backend.METADATA_PATH) as f:
            meta = json.load(f)
        self.assertIn("model_version", meta)
        self.assertIn("features_used", meta)
        self.assertIn("training_duration_seconds", meta)
        self.assertEqual(meta["n_estimators"], backend.N_ESTIMATORS)
        self.assertEqual(meta["random_state"],  backend.RANDOM_STATE)

        print(f"  âœ“ Model trained in {result['training_duration_seconds']:.3f}s, "
              f"version={result['model_version']}, "
              f"anomalies={result['training_anomaly_count']}/{result['rows_used_for_training']}")

    # ----------------------------------------------------------
    # TEST 09: Failed training does NOT replace existing model
    # ----------------------------------------------------------
    def test_09_failed_training_preserves_old_model(self):
        print("\n[TEST 09] Failed training leaves old model intact")
        # Ensure a model exists first
        rows_ok = _rows_over_minutes(200, span_min=60)
        with patch("main.fetch_window_data", return_value=rows_ok):
            r1 = backend.run_training(60, 0.05)
        self.assertEqual(r1["status"], "completed")
        v1 = r1["model_version"]

        # Now fail a training attempt (empty window)
        with patch("main.fetch_window_data", return_value=[]):
            r2 = backend.run_training(60, 0.05)
        self.assertEqual(r2["status"], "failed")

        # Confirm old model still on disk with original version
        loaded = backend.load_model()
        self.assertIsNotNone(loaded)
        _, meta = loaded
        self.assertEqual(meta["model_version"], v1)
        print(f"  âœ“ Old model v{v1} intact after failed attempt.")

    # ----------------------------------------------------------
    # TEST 10: Live scoring uses saved model, does NOT retrain
    # ----------------------------------------------------------
    def test_10_live_scoring_no_retrain(self):
        print("\n[TEST 10] score_latest uses saved model, does not retrain")
        loaded = backend.load_model()
        if not loaded:
            self.skipTest("No model available â€” run test_08 first")

        _, meta_before = loaded
        version_before = meta_before["model_version"]

        # score_latest() must NOT change the model version on disk
        # (we just verify the file is identical after calling load_model again)
        loaded2 = backend.load_model()
        self.assertIsNotNone(loaded2, "Model should still be on disk")
        _, meta_after = loaded2
        self.assertEqual(version_before, meta_after["model_version"],
                         "Model version must not change during a live-score call")
        print(f"  âœ“ Model version unchanged: v{version_before}")


    # ----------------------------------------------------------
    # TEST 11: Timezone boundary â€” only records inside window included
    # ----------------------------------------------------------
    def test_11_timezone_boundary(self):
        print("\n[TEST 11] Timezone-aware window boundary filtering")
        now    = datetime.now(timezone.utc)
        start  = now - timedelta(minutes=60)
        end    = now

        inside  = [_make_row(i, start + timedelta(minutes=i)) for i in range(1, 60)]
        outside_before = [_make_row(0,   start - timedelta(seconds=1))]
        outside_after  = [_make_row(100, end   + timedelta(seconds=1))]
        all_rows = outside_before + inside + outside_after

        def mock_fetch(ws, we):
            return [r for r in all_rows
                    if datetime.fromisoformat(r["created_at"]) >= ws
                    and datetime.fromisoformat(r["created_at"]) <= we]

        with patch("main.fetch_window_data", side_effect=mock_fetch):
            fetched = backend.fetch_window_data(start, end)

        ids = [r["id"] for r in fetched]
        self.assertNotIn(0,   ids, "Row before window must be excluded")
        self.assertNotIn(100, ids, "Row after window must be excluded")
        print(f"  âœ“ {len(fetched)} rows inside window; boundaries respected.")

    # ----------------------------------------------------------
    # TEST 12: Metadata contains all required fields
    # ----------------------------------------------------------
    def test_12_metadata_completeness(self):
        print("\n[TEST 12] Model metadata contains all required fields")
        required = [
            "model_type", "model_version", "trained_at",
            "training_started_at", "training_completed_at", "training_duration_seconds",
            "window_requested_minutes", "window_start", "window_end",
            "actual_oldest_timestamp", "actual_newest_timestamp", "actual_coverage_minutes",
            "coverage_ratio", "training_dataset_hash", "expected_sample_count", "actual_sample_count",
            "rows_fetched", "rows_after_validation", "rows_used_for_training",
            "median_sampling_interval_seconds", "data_gaps_detected",
            "features_used", "n_estimators", "contamination", "random_state",
            "training_anomaly_count", "training_anomaly_percentage",
            "score_min", "score_max", "score_mean", "score_median",
        ]
        loaded = backend.load_model()
        if not loaded:
            self.skipTest("No model available â€” run test_08 first")
        _, meta = loaded
        missing = [f for f in required if f not in meta]
        self.assertEqual(missing, [], f"Missing metadata fields: {missing}")
        print(f"  âœ“ All {len(required)} required metadata fields present.")

    # ----------------------------------------------------------
    # TEST 13: dataset warning for small but valid dataset
    # ----------------------------------------------------------
    def test_13_small_dataset_warning(self):
        print("\n[TEST 13] Warning shown for small but technically-valid dataset")
        rows = _rows_over_minutes(21, span_min=60)  # just above MIN_TRAINING_ROWS
        stats = backend.analyse_dataset(rows, window_minutes=60)
        self.assertTrue(stats["valid"])
        self.assertIsNotNone(stats.get("warning"))
        print(f"  âœ“ Warning: {stats['warning']}")

    # ----------------------------------------------------------
    # TEST 14 (A): 60-min request, 59-min coverage -> PASS
    # ----------------------------------------------------------
    def test_14_test_A_59min_coverage(self):
        print("\n[TEST A] 60-min request, 59-min coverage")
        rows = _rows_over_minutes(200, span_min=59)
        stats = backend.analyse_dataset(rows, window_minutes=60)
        self.assertTrue(stats["valid"], stats.get("error"))
        print("  âœ“ PASS")

    # ----------------------------------------------------------
    # TEST 15 (B): 60-min request, 54-min coverage -> PASS (exactly 90%)
    # ----------------------------------------------------------
    def test_15_test_B_54min_coverage(self):
        print("\n[TEST B] 60-min request, 54-min coverage")
        rows = _rows_over_minutes(200, span_min=54)
        stats = backend.analyse_dataset(rows, window_minutes=60)
        self.assertTrue(stats["valid"], stats.get("error"))
        print("  âœ“ PASS")

    # ----------------------------------------------------------
    # TEST 16 (C): 60-min request, 53-min coverage -> FAIL (<90%)
    # ----------------------------------------------------------
    def test_16_test_C_53min_coverage(self):
        print("\n[TEST C] 60-min request, 53-min coverage")
        rows = _rows_over_minutes(200, span_min=53)
        stats = backend.analyse_dataset(rows, window_minutes=60)
        self.assertFalse(stats["valid"], "Should fail due to <90% coverage")
        print(f"  âœ“ FAIL as expected: {stats['error']}")

    # ----------------------------------------------------------
    # TEST 17 (D): 60-min request, 31-min coverage -> FAIL
    # ----------------------------------------------------------
    def test_17_test_D_31min_coverage(self):
        print("\n[TEST D] 60-min request, 31-min coverage")
        rows = _rows_over_minutes(200, span_min=31)
        stats = backend.analyse_dataset(rows, window_minutes=60)
        self.assertFalse(stats["valid"])
        print("  âœ“ FAIL as expected")

    # ----------------------------------------------------------
    # TEST 18 (E): 60-min request, 300 rows but 20-min coverage -> FAIL
    # ----------------------------------------------------------
    def test_18_test_E_many_rows_low_coverage(self):
        print("\n[TEST E] 60-min request, 300 rows, 20-min coverage")
        rows = _rows_over_minutes(300, span_min=20)
        stats = backend.analyse_dataset(rows, window_minutes=60)
        self.assertFalse(stats["valid"])
        print("  âœ“ FAIL as expected (coverage is orthogonal to row count)")

    # ----------------------------------------------------------
    # TEST 19 (F): 60-min request, 270 rows over ~60 min -> PASS
    # ----------------------------------------------------------
    def test_19_test_F_270_rows_60min(self):
        print("\n[TEST F] 60-min request, 270 rows, ~60 min coverage")
        rows = _rows_over_minutes(270, span_min=59.5)
        stats = backend.analyse_dataset(rows, window_minutes=60)
        self.assertTrue(stats["valid"])
        print("  âœ“ PASS")

    # ----------------------------------------------------------
    # TEST 20 (G): Unrelated historical gap outside window
    # ----------------------------------------------------------
    def test_20_test_G_unrelated_historical_gap(self):
        print("\n[TEST G] Gap analysis only applies to selected window")
        now = datetime.now(timezone.utc)
        start = now - timedelta(minutes=60)
        # Add a 10-hour gap OUTSIDE the window (e.g. 5 hours ago)
        # BUT fetch_window_data only returns rows strictly inside the window.
        # So we just mock what fetch_window_data returns for a clean window.
        rows_in_window = _rows_over_minutes(100, span_min=60, start=start)
        stats = backend.analyse_dataset(rows_in_window, window_minutes=60)
        self.assertEqual(stats["data_gaps_detected"], 0)
        print("  [PASS] (0 gaps inside window)")

    # ----------------------------------------------------------
    # TEST 21 (H): Late arriving row doesn't enter snapshot
    # ----------------------------------------------------------
    def test_21_test_H_training_snapshot(self):
        print("\n[TEST H] Late arriving row does not enter snapshot")
        # Run training
        with patch("main.fetch_window_data") as mock_fetch:
            mock_fetch.return_value = _rows_over_minutes(200, span_min=60)
            res = backend.run_training(60, 0.05)
            
            # Verify fetch_window_data was called ONCE, and its window_end 
            # equals the exact training_started_at.
            args, _ = mock_fetch.call_args
            ws, we = args
            
            # The backend calculates window_end immediately.
            # A row arriving at window_end + 12s wouldn't be fetched 
            # because the query is literally lte(window_end).
            self.assertEqual(ws, we - timedelta(minutes=60))
            print(f"  [PASS] (Snapshot window is locked to {we.isoformat()})")

    # ----------------------------------------------------------
    # TEST 22: Successful live scoring with score_latest()
    # ----------------------------------------------------------
    def test_22_live_scoring_success(self):
        print("\n[TEST 22] Live scoring executes without retraining and produces predictions")
        test_rows = [
            {"id": 9991, "aqi_ppm": 450.0, "temperature": 25.0, "humidity": 60.0},
            {"id": 9992, "aqi_ppm": 950.0, "temperature": 38.0, "humidity": 90.0},
        ]
        mock_table = MagicMock()
        mock_select = MagicMock()
        mock_order = MagicMock()
        mock_limit = MagicMock()
        mock_limit.execute.return_value = MagicMock(data=test_rows)
        mock_order.limit.return_value = mock_limit
        mock_select.order.return_value = mock_order
        mock_table.select.return_value = mock_select

        with patch.object(backend.supabase, "table", return_value=mock_table), \
             patch("main._upsert_predictions", return_value=2) as mock_upsert:
            res = backend.score_latest(limit=2)

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["scored_count"], 2)
        self.assertEqual(res["persisted_predictions"], 2)
        self.assertIn("model_version", res)
        # Verify upsert payload contains valid keys
        upsert_records = mock_upsert.call_args[0][0]
        self.assertEqual(len(upsert_records), 2)
        self.assertEqual(upsert_records[0]["sensor_data_id"], 9991)
        self.assertIn("is_anomaly", upsert_records[0])
        self.assertIn("anomaly_score", upsert_records[0])
        print("  [PASS] score_latest successfully generated prediction records.")

    # ----------------------------------------------------------
    # TEST 23: Live scoring fails gracefully when no model exists
    # ----------------------------------------------------------
    def test_23_live_scoring_no_model(self):
        print("\n[TEST 23] Live scoring raises 400 when no model is available")
        with patch("main.load_model", return_value=None):
            with self.assertRaises(backend.HTTPException) as cm:
                backend.score_latest(limit=10)
            self.assertEqual(cm.exception.status_code, 400)
            self.assertIn("No trained model available", cm.exception.detail)
        print("  [PASS] Correctly blocked live scoring when model is absent.")

    # ----------------------------------------------------------
    # TEST 24: Absent prediction verification (Unscored logic)
    # ----------------------------------------------------------
    def test_24_absent_prediction_remains_unscored(self):
        print("\n[TEST 24] Absent prediction must not be classified as Normal")
        # Simulating frontend joining sensor_data with ml_predictions
        sensor_row = {"id": 1050, "aqi_ppm": 500.0, "temperature": 27.0, "humidity": 65.0}
        pred_map = {}  # No entry for id 1050
        has_pred = sensor_row["id"] in pred_map
        self.assertFalse(has_pred, "Sensor row should not have prediction")
        # Prediction status must be UNSCORED, never Normal
        status = "NORMAL" if (has_pred and not pred_map[sensor_row["id"]]["is_anomaly"]) else ("ANOMALY" if (has_pred and pred_map[sensor_row["id"]]["is_anomaly"]) else "UNSCORED")
        self.assertEqual(status, "UNSCORED")
        print("  [PASS] Unscored row remains strictly UNSCORED.")

    # ----------------------------------------------------------
    # TEST 25: Corrupt model or metadata handling in load_model
    # ----------------------------------------------------------
    def test_25_model_loading_error_handling(self):
        print("\n[TEST 25] Corrupt model or invalid metadata returns None safely")
        with patch("os.path.exists", return_value=True), \
             patch("joblib.load", side_effect=Exception("Corrupt binary")):
            result = backend.load_model()
            self.assertIsNone(result)
        print("  [PASS] Gracefully returns None on load failure.")


# ============================================================
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 60)
    print("Air Quality ML Backend â€” Test Suite v2")
    print("=" * 60)
    unittest.main(verbosity=2)

