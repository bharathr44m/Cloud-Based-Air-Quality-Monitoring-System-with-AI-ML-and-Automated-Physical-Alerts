import os
import time
import logging
import asyncio
import json
import shutil
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from supabase import create_client, Client

# ====================================================================
# CONFIGURATION & PATHS
# ====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml_backend")

MODEL_STORE_DIR  = os.path.join(BASE_DIR, "model_store")
MODEL_FILE_PATH  = os.path.join(MODEL_STORE_DIR, "isolation_forest.joblib")
METADATA_PATH    = os.path.join(MODEL_STORE_DIR, "model_metadata.json")
MODEL_FILE_TMP   = MODEL_FILE_PATH + ".tmp"
METADATA_TMP     = METADATA_PATH   + ".tmp"
HISTORY_PATH     = os.path.join(MODEL_STORE_DIR, "training_history.json")
os.makedirs(MODEL_STORE_DIR, exist_ok=True)

SUPABASE_URL         = os.getenv("SUPABASE_URL", "https://ppvtqsyhiuabvxjtxtyp.supabase.co")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# ---- Configurable thresholds ----------------------------------------
MIN_TRAINING_ROWS        = 20     # Hard minimum; training is blocked below this
RECOMMENDED_ROWS         = 200    # Below this a data-quality warning is shown
SUPABASE_PAGE_SIZE       = 1000   # Rows per Supabase request (pagination)
GAP_THRESHOLD_MULTIPLIER = 5      # Gaps > median * this factor are flagged
MIN_COVERAGE_RATIO       = 0.90   # Dataset must cover at least 90% of requested window

# ---- Model parameters (deterministic) --------------------------------
N_ESTIMATORS  = 100
CONTAMINATION = 0.05
RANDOM_STATE  = 42

FEATURE_COLS  = ["aqi_ppm", "temperature", "humidity"]
# NOTE: aqi_ppm is a linear approximation from the MQ135 ADC value via a
#       voltage-divider + empirical curve (see readMQ135PPM in ESP32 firmware).
#       It is NOT a scientifically calibrated ppm concentration.
#       The database column name is preserved for compatibility.
#       Dashboard label should read "Gas Sensor (est.)" not "true ppm".

VALID_WINDOW_MINUTES = [15, 30, 60, 120, 360, 720, 1440]

# ====================================================================

if not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_SERVICE_KEY missing from ml_backend/.env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ====================================================================
# FASTAPI APP
# ====================================================================
app = FastAPI(
    title="Air Quality ML Anomaly Backend",
    description="Isolation Forest anomaly detection — single-shot training, live prediction",
    version="2.1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================================================================
# TRAINING STATE MACHINE
# ====================================================================
class MLState:
    def __init__(self):
        self.phase: str = "idle"
        self.training_active: bool = False
        self.training_started_at: Optional[float] = None
        self.training_completed_at: Optional[float] = None
        self.training_duration_s: Optional[float] = None
        self.last_result: Optional[Dict[str, Any]] = None
        self.task: Optional[asyncio.Task] = None

ml_state = MLState()

# ====================================================================
# REQUEST MODEL
# ====================================================================
class TrainRequest(BaseModel):
    window_minutes: int = 60
    contamination: float = 0.05

# ====================================================================
# MODEL PERSISTENCE
# ====================================================================
def save_model_atomic(model: IsolationForest, metadata: Dict[str, Any]) -> None:
    joblib.dump(model, MODEL_FILE_TMP)
    with open(METADATA_TMP, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    os.replace(MODEL_FILE_TMP, MODEL_FILE_PATH)
    os.replace(METADATA_TMP, METADATA_PATH)
    logger.info("Model saved atomically to model_store/")

    history: List[Dict] = []
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH) as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append({k: metadata[k] for k in [
        "model_version", "trained_at", "training_duration_seconds",
        "window_requested_minutes", "window_start", "window_end",
        "rows_used_for_training", "training_anomaly_count",
        "training_anomaly_percentage", "training_dataset_hash"
    ] if k in metadata})
    with open(HISTORY_PATH, "w") as f:
        json.dump(history[-50:], f, indent=2, default=str)

def load_model() -> Optional[tuple]:
    if not os.path.exists(MODEL_FILE_PATH) or not os.path.exists(METADATA_PATH):
        return None
    try:
        model = joblib.load(MODEL_FILE_PATH)
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        return model, meta
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

# ====================================================================
# SUPABASE FETCH
# ====================================================================
def fetch_window_data(window_start: datetime, window_end: datetime) -> List[Dict]:
    rows: List[Dict] = []
    offset = 0
    start_iso = window_start.isoformat()
    end_iso   = window_end.isoformat()

    while True:
        res = (
            supabase.table("sensor_data")
            .select("id, created_at, aqi_ppm, temperature, humidity")
            .gte("created_at", start_iso)
            .lte("created_at", end_iso)
            .order("created_at", desc=False)
            .range(offset, offset + SUPABASE_PAGE_SIZE - 1)
            .execute()
        )
        batch = res.data or []
        rows.extend(batch)
        if len(batch) < SUPABASE_PAGE_SIZE:
            break
        offset += SUPABASE_PAGE_SIZE

    return rows

# ====================================================================
# DATASET VALIDATION & STATISTICS
# ====================================================================
import hashlib

def analyse_dataset(rows: List[Dict], window_minutes: int) -> Dict[str, Any]:
    rows_fetched = len(rows)
    result: Dict[str, Any] = {
        "rows_fetched": rows_fetched,
        "rows_after_validation": 0,
        "rows_removed": 0,
        "removal_reasons": [],
        "valid": False,
        "error": None,
        "warning": None,
    }

    if rows_fetched == 0:
        result["error"] = (
            f"No sensor data found in the selected {window_minutes}-minute window. "
            "Connect the ESP32 hardware or select a different time window."
        )
        return result

    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df = df.sort_values("created_at").reset_index(drop=True)

    t_first = df["created_at"].iloc[0]
    t_last  = df["created_at"].iloc[-1]
    actual_coverage_min = (t_last - t_first).total_seconds() / 60.0
    coverage_ratio = actual_coverage_min / window_minutes if window_minutes else 1.0

    result["actual_oldest_timestamp"] = t_first.isoformat()
    result["actual_newest_timestamp"] = t_last.isoformat()
    result["actual_coverage_minutes"] = round(actual_coverage_min, 2)
    result["coverage_ratio"] = round(coverage_ratio, 4)

    min_required_coverage = window_minutes * MIN_COVERAGE_RATIO
    if round(actual_coverage_min, 2) < min_required_coverage:
        result["error"] = (
            f"Insufficient temporal coverage: requested {window_minutes} min but only "
            f"{actual_coverage_min:.1f} min of valid data available in the window "
            f"({coverage_ratio*100:.1f}%). At least {MIN_COVERAGE_RATIO*100:.1f}% "
            f"({min_required_coverage:.1f} min) coverage is required."
        )
        return result

    before_clean = len(df)
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_mask = df[FEATURE_COLS].isnull().any(axis=1) | np.isinf(df[FEATURE_COLS].values).any(axis=1)
    removed_invalid = int(bad_mask.sum())
    if removed_invalid:
        result["removal_reasons"].append(f"{removed_invalid} rows with NaN/inf sensor values")
    df = df[~bad_mask].reset_index(drop=True)

    dup_mask = df["created_at"].duplicated(keep="first")
    removed_dups = int(dup_mask.sum())
    if removed_dups:
        result["removal_reasons"].append(f"{removed_dups} duplicate timestamps")
    df = df[~dup_mask].reset_index(drop=True)

    rows_clean = len(df)
    result["rows_after_validation"] = rows_clean
    result["rows_removed"] = before_clean - rows_clean

    if rows_clean < MIN_TRAINING_ROWS:
        result["error"] = (
            f"Only {rows_clean} usable rows after cleaning "
            f"(minimum required: {MIN_TRAINING_ROWS})."
        )
        return result

    if rows_clean >= 2:
        deltas = df["created_at"].diff().dropna().dt.total_seconds()
        median_s = float(deltas.median())
        min_s    = float(deltas.min())
        max_s    = float(deltas.max())
        gap_threshold = median_s * GAP_THRESHOLD_MULTIPLIER
        gaps = deltas[deltas > gap_threshold]
        result["median_sampling_interval_seconds"] = round(median_s, 1)
        result["minimum_sampling_interval_seconds"] = round(min_s, 1)
        result["maximum_sampling_interval_seconds"] = round(max_s, 1)
        result["data_gaps_detected"] = int(len(gaps))
        result["largest_gap_seconds"] = round(float(gaps.max()), 1) if len(gaps) else 0.0
        result["expected_sample_count"] = int((window_minutes * 60) / median_s) if median_s > 0 else rows_clean
    else:
        result["median_sampling_interval_seconds"] = None
        result["minimum_sampling_interval_seconds"] = None
        result["maximum_sampling_interval_seconds"] = None
        result["data_gaps_detected"] = 0
        result["largest_gap_seconds"] = 0.0
        result["expected_sample_count"] = rows_clean

    result["actual_sample_count"] = rows_clean

    if rows_clean < RECOMMENDED_ROWS:
        result["warning"] = (
            f"Dataset is technically sufficient ({rows_clean} rows, {coverage_ratio*100:.1f}% coverage) "
            f"but relatively small for robust anomaly modelling. Recommended: {RECOMMENDED_ROWS}+ rows."
        )

    # Compute deterministic hash of the cleaned dataset
    hash_payload = "".join(f"{r['id']}_{r['created_at'].isoformat()}" for _, r in df.iterrows())
    result["training_dataset_hash"] = hashlib.sha256(hash_payload.encode('utf-8')).hexdigest()

    result["valid"] = True
    result["df_clean"] = df
    return result


# ====================================================================
# CORE TRAINING FUNCTION
# ====================================================================
def run_training(window_minutes: int, contamination: float, phase_callback: Optional[Any] = None) -> Dict[str, Any]:
    # Exact window boundary is captured ONCE
    training_started_at = datetime.now(timezone.utc)
    t0 = time.time()
    window_end   = training_started_at
    window_start = window_end - timedelta(minutes=window_minutes)

    logger.info(
        f"[TRAIN] window={window_minutes}m  "
        f"{window_start.isoformat()} → {window_end.isoformat()}"
    )

    if phase_callback:
        phase_callback("fetching_data")
    rows = fetch_window_data(window_start, window_end)
    logger.info(f"[TRAIN] Fetched {len(rows)} rows from Supabase")

    if phase_callback:
        phase_callback("validating_data")
    analysis = analyse_dataset(rows, window_minutes)

    if not analysis["valid"]:
        duration = round(time.time() - t0, 3)
        logger.warning(f"[TRAIN] Validation failed: {analysis['error']}")
        return {
            "status": "failed",
            "error": analysis["error"],
            "window_requested_minutes": window_minutes,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "rows_fetched": analysis["rows_fetched"],
            "rows_after_validation": analysis.get("rows_after_validation", 0),
            "training_duration_seconds": duration,
        }

    df = analysis.pop("df_clean")
    rows_used = analysis["rows_after_validation"]

    X = df[FEATURE_COLS].values
    logger.info(f"[TRAIN] Fitting IsolationForest on {rows_used} rows, contamination={contamination}")

    if phase_callback:
        phase_callback("training")
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=contamination,
        random_state=RANDOM_STATE
    )
    model.fit(X)

    predictions   = model.predict(X)
    scores        = model.decision_function(X)

    anom_count    = int((predictions == -1).sum())
    anom_pct      = round(anom_count / rows_used * 100, 2)
    training_completed_at = datetime.now(timezone.utc)
    duration      = round(time.time() - t0, 3)

    model_version = training_completed_at.strftime("%Y%m%dT%H%M%SZ")

    metadata = {
        "model_type":                "IsolationForest",
        "model_version":             model_version,
        "trained_at":                training_completed_at.isoformat(),
        "training_started_at":       training_started_at.isoformat(),
        "training_completed_at":     training_completed_at.isoformat(),
        "training_duration_seconds": duration,

        "window_requested_minutes":  window_minutes,
        "window_start":              window_start.isoformat(),
        "window_end":                window_end.isoformat(),

        "training_dataset_start":    analysis["actual_oldest_timestamp"],
        "training_dataset_end":      analysis["actual_newest_timestamp"],
        "actual_oldest_timestamp":   analysis["actual_oldest_timestamp"],
        "actual_newest_timestamp":   analysis["actual_newest_timestamp"],
        "actual_coverage_minutes":   analysis["actual_coverage_minutes"],
        "coverage_ratio":            analysis["coverage_ratio"],
        "training_dataset_hash":     analysis["training_dataset_hash"],

        "rows_fetched":              analysis["rows_fetched"],
        "rows_after_validation":     rows_used,
        "rows_used_for_training":    rows_used,
        "expected_sample_count":     analysis.get("expected_sample_count"),
        "actual_sample_count":       analysis["actual_sample_count"],
        "rows_removed":              analysis["rows_removed"],
        "removal_reasons":           analysis["removal_reasons"],

        "median_sampling_interval_seconds":  analysis.get("median_sampling_interval_seconds"),
        "minimum_sampling_interval_seconds": analysis.get("minimum_sampling_interval_seconds"),
        "maximum_sampling_interval_seconds": analysis.get("maximum_sampling_interval_seconds"),
        "data_gaps_detected":        analysis.get("data_gaps_detected", 0),
        "largest_gap_seconds":       analysis.get("largest_gap_seconds", 0.0),

        "features_used":             FEATURE_COLS,
        "feature_note":              (
            "aqi_ppm is a linear approximation from MQ135 ADC via voltage-divider "
            "empirical curve. It is NOT a scientifically calibrated ppm concentration."
        ),

        "n_estimators":              N_ESTIMATORS,
        "contamination":             contamination,
        "random_state":              RANDOM_STATE,

        "training_anomaly_count":    anom_count,
        "training_anomaly_percentage": anom_pct,
        "unsupervised_note":         (
            "Isolation Forest is unsupervised. No accuracy/precision/recall "
            "metrics are computed without labelled ground-truth data."
        ),

        "score_min":    round(float(scores.min()), 6),
        "score_max":    round(float(scores.max()), 6),
        "score_mean":   round(float(scores.mean()), 6),
        "score_median": round(float(np.median(scores)), 6),

        "warning": analysis.get("warning"),
    }

    if phase_callback:
        phase_callback("saving_model")
    save_model_atomic(model, metadata)
    logger.info(f"[TRAIN] Completed in {duration}s. Anomalies: {anom_count}/{rows_used} ({anom_pct}%)")

    # NOTE: Training no longer upserts its own anomalies into `ml_predictions`.
    # `ml_predictions` is strictly reserved for LIVE PREDICTION runs via /score/latest.

    response = {
        "status":  "completed",
        "model_type": "IsolationForest",
        "model_version": model_version,

        "window_requested_minutes": window_minutes,
        "window_start": window_start.isoformat(),
        "window_end":   window_end.isoformat(),

        "actual_oldest_timestamp":  metadata["actual_oldest_timestamp"],
        "actual_newest_timestamp":  metadata["actual_newest_timestamp"],
        "actual_coverage_minutes":  metadata["actual_coverage_minutes"],
        "coverage_ratio":           metadata["coverage_ratio"],
        "training_dataset_hash":    metadata["training_dataset_hash"],

        "rows_fetched":            analysis["rows_fetched"],
        "rows_after_validation":   rows_used,
        "rows_used_for_training":  rows_used,
        "expected_sample_count":   metadata["expected_sample_count"],
        "actual_sample_count":     metadata["actual_sample_count"],

        "median_sampling_interval_seconds": metadata["median_sampling_interval_seconds"],
        "minimum_sampling_interval_seconds": metadata["minimum_sampling_interval_seconds"],
        "maximum_sampling_interval_seconds": metadata["maximum_sampling_interval_seconds"],
        "data_gaps_detected":      metadata["data_gaps_detected"],
        "largest_gap_seconds":     metadata["largest_gap_seconds"],

        "features_used":    FEATURE_COLS,
        "feature_note":     metadata["feature_note"],

        "training_duration_seconds": duration,

        "n_estimators":   N_ESTIMATORS,
        "contamination":  contamination,
        "random_state":   RANDOM_STATE,

        "training_anomaly_count":      anom_count,
        "training_anomaly_percentage": anom_pct,
        "unsupervised_note":           metadata["unsupervised_note"],

        "score_min":    metadata["score_min"],
        "score_max":    metadata["score_max"],
        "score_mean":   metadata["score_mean"],
        "score_median": metadata["score_median"],

        "warning": analysis.get("warning"),
        "error": None,
    }
    return response


# ====================================================================
# PREDICTIONS PERSISTENCE
# ====================================================================
def _upsert_predictions(records: List[Dict]) -> int:
    if not records:
        return 0
    try:
        res = supabase.table("ml_predictions").upsert(
            records, on_conflict="sensor_data_id"
        ).execute()
        return len(res.data) if res.data else 0
    except Exception as e:
        logger.warning(f"Could not persist predictions: {e}")
        return 0


# ====================================================================
# ASYNC WRAPPER (single-shot — no repeating loop)
# ====================================================================
async def run_training_task(window_minutes: int, contamination: float):
    try:
        ml_state.phase = "fetching_data"
        def _set_phase(p: str):
            ml_state.phase = p
        result = await asyncio.to_thread(run_training, window_minutes, contamination, _set_phase)
        ml_state.last_result = result
        ml_state.training_completed_at = time.time()
        ml_state.training_duration_s = round(
            ml_state.training_completed_at - ml_state.training_started_at, 3
        )
        ml_state.phase = "completed" if result["status"] == "completed" else "failed"
    except asyncio.CancelledError:
        ml_state.phase = "failed"
        ml_state.last_result = {"status": "failed", "error": "Training was cancelled."}
    except Exception as e:
        logger.error(f"Training task error: {e}", exc_info=True)
        ml_state.phase = "failed"
        ml_state.last_result = {"status": "failed", "error": str(e)}
    finally:
        ml_state.training_active = False


# ====================================================================
# FASTAPI ENDPOINTS
# ====================================================================

@app.get("/")
def root():
    p = os.path.join(os.path.dirname(BASE_DIR), "dashboard", "index.html")
    if os.path.exists(p):
        return FileResponse(p)
    return {"service": "Air Quality ML Backend", "status": "online"}


@app.get("/health")
def health():
    loaded = load_model()
    meta = loaded[1] if loaded else None
    return {
        "status": "online",
        "service": "Air Quality ML Anomaly Backend v2",
        "training_active": ml_state.training_active,
        "training_phase": ml_state.phase,
        "model_loaded": loaded is not None,
        "model_version": meta.get("model_version") if meta else None,
        "model_trained_at": meta.get("trained_at") if meta else None,
    }


@app.post("/train/start")
async def start_training(req: TrainRequest):
    """Single-shot training job. Returns immediately; poll /train/status."""

    if ml_state.training_active:
        return {
            "message": "Training already in progress.",
            "training_phase": ml_state.phase,
            "training_active": True,
        }

    # Validate window
    if req.window_minutes not in VALID_WINDOW_MINUTES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid window_minutes={req.window_minutes}. "
                   f"Allowed: {VALID_WINDOW_MINUTES}"
        )
    if not (0.001 <= req.contamination <= 0.5):
        raise HTTPException(
            status_code=400,
            detail="contamination must be between 0.001 and 0.5"
        )

    # Reset state
    ml_state.training_active = True
    ml_state.training_started_at = time.time()
    ml_state.training_completed_at = None
    ml_state.training_duration_s = None
    ml_state.phase = "fetching_data"
    ml_state.last_result = None

    ml_state.task = asyncio.create_task(
        run_training_task(req.window_minutes, req.contamination)
    )

    return {
        "message": "Training started.",
        "training_active": True,
        "training_phase": ml_state.phase,
        "window_minutes": req.window_minutes,
    }


@app.get("/train/status")
def get_status():
    elapsed = None
    if ml_state.training_started_at:
        if ml_state.training_active:
            elapsed = round(time.time() - ml_state.training_started_at, 1)
        elif ml_state.training_duration_s is not None:
            elapsed = ml_state.training_duration_s

    loaded = load_model()
    active_meta = loaded[1] if loaded else None

    return {
        "training_active": ml_state.training_active,
        "training_phase": ml_state.phase,
        "elapsed_seconds": elapsed,
        "last_result": ml_state.last_result,
        "model_available": loaded is not None,
        "active_model_version": active_meta.get("model_version") if active_meta else None,
        "active_model_trained_at": active_meta.get("trained_at") if active_meta else None,
    }


@app.get("/train/history")
def get_history():
    if not os.path.exists(HISTORY_PATH):
        return {"history": []}
    with open(HISTORY_PATH) as f:
        return {"history": json.load(f)}


class ScoreRequest(BaseModel):
    limit: Optional[int] = 100


@app.post("/score/latest")
def score_latest(limit: Optional[int] = 100, req: Optional[ScoreRequest] = None):
    """Score the most recent sensor records using the saved model."""
    target_limit = req.limit if (req and req.limit is not None) else (limit or 100)

    loaded = load_model()
    if not loaded:
        raise HTTPException(status_code=400, detail="No trained model available. Please train an Isolation Forest model first.")
    model, meta = loaded
    feature_cols = meta.get("features_used", FEATURE_COLS)
    model_version = meta.get("model_version", "unknown")

    res = (
        supabase.table("sensor_data")
        .select("id, aqi_ppm, temperature, humidity")
        .order("created_at", desc=True)
        .limit(target_limit)
        .execute()
    )
    data = res.data or []
    if not data:
        return {"status": "success", "message": "No sensor readings found.", "scored_count": 0, "persisted_predictions": 0, "anomaly_count": 0}

    df = pd.DataFrame(data)
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    X = df[feature_cols].values
    preds  = model.predict(X)
    scores = model.decision_function(X)

    records = []
    for i, row in enumerate(data):
        records.append({
            "sensor_data_id": row["id"],
            "is_anomaly":     bool(preds[i] == -1),
            "anomaly_score":  round(float(scores[i]), 6),
            "model_version":  model_version,
        })

    persisted = _upsert_predictions(records)
    anom = int((np.array(preds) == -1).sum())

    return {
        "status": "success",
        "scored_count": len(data),
        "persisted_predictions": persisted,
        "anomaly_count": anom,
        "model_version": model_version,
        "score_min": round(float(scores.min()), 6) if len(scores) else None,
        "score_max": round(float(scores.max()), 6) if len(scores) else None,
        "score_mean": round(float(scores.mean()), 6) if len(scores) else None,
    }
