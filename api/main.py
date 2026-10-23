"""CSV signal exploration; classification requires a compatible trained public-data bundle."""

from contextlib import asynccontextmanager
from io import StringIO
import os
from pathlib import Path
from threading import Lock
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from src.signal_processing.filters import preprocess_ecg, normalize_signal
from src.signal_processing.peak_detection import (
    detect_r_peaks,
    calculate_heart_rate,
    calculate_rr_intervals,
)
from src.signal_processing.features import (
    calculate_hrv_time_domain,
    calculate_hrv_frequency_domain,
)
from src.models.cnn_model import ArrhythmiaClassifier
from src.models.contract import extract_beat

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 4 * 1024 * 1024


def read_signal(file):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "CSV exceeds the 4 MiB limit")
    try:
        df = pd.read_csv(StringIO(data.decode("utf-8")))
        columns = [c for c in ["signal", "ecg", "value", "amplitude"] if c in df.columns]
        if not columns and len(df.columns) != 1:
            raise ValueError("Use one signal column or name it signal/ecg/value/amplitude")
        values = pd.to_numeric(
            df[columns[0] if columns else df.columns[0]], errors="raise"
        ).to_numpy(dtype=float)
        if not len(values) or not np.all(np.isfinite(values)):
            raise ValueError("Signal must be nonempty and finite")
        return values
    except (
        ValueError,
        IndexError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ):
        raise HTTPException(
            400, "Invalid CSV: provide finite numeric signal samples with a header"
        ) from None


def create_app(path: Path):
    lock = Lock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.classifier = ArrhythmiaClassifier.load_model(path) if path.exists() else None
        try:
            yield
        finally:
            app.state.classifier = None

    app = FastAPI(title="BioSignal Analyzer API", version="1.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            x.strip()
            for x in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
            if x.strip()
        ],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/")
    def root():
        return {
            "service": "BioSignal Analyzer",
            "scope": "CSV ECG exploration; coarse beat classifier optional",
        }

    @app.get("/health")
    def health():
        ready = app.state.classifier is not None
        return {
            "status": "ready",
            "signal_analysis_available": True,
            "model_loaded": ready,
            "classification_available": ready,
        }

    @app.post("/analyze")
    def analyze(file: UploadFile = File(...), sampling_rate: float = Query(360, ge=128, le=2000)):
        values = read_signal(file)
        if len(values) < 64:
            raise HTTPException(400, "Need at least 64 samples for filtering")
        filtered = preprocess_ecg(values, sampling_rate, notch_freq=60)
        peaks = detect_r_peaks(filtered, sampling_rate)
        mean_hr, _ = calculate_heart_rate(peaks, sampling_rate)
        rr = calculate_rr_intervals(peaks, sampling_rate)
        predictions = []
        classifier = app.state.classifier
        if classifier is not None and sampling_rate == 360:
            with lock:
                for i, peak in enumerate(peaks[:10]):
                    segment = extract_beat(filtered, peak)
                    if segment is None:
                        continue
                    label, confidence, probabilities = classifier.predict(segment)
                    predictions.append(
                        {
                            "beat_number": i + 1,
                            "predicted_class": label,
                            "confidence": confidence,
                            "probabilities": probabilities,
                        }
                    )
        return {
            "success": True,
            "duration": len(values) / sampling_rate,
            "sampling_rate": sampling_rate,
            "analysis": {
                "mean_heart_rate": float(mean_hr),
                "num_peaks": len(peaks),
                "hrv_metrics": {
                    **calculate_hrv_time_domain(rr),
                    **calculate_hrv_frequency_domain(rr),
                },
                "signal_quality": "not clinically assessed",
            },
            "predictions": predictions,
            "classification_available": classifier is not None and sampling_rate == 360,
            "message": "Signal exploration completed; classification requires a compatible 360 Hz trained model",
        }

    @app.post("/predict")
    def predict(
        file: UploadFile = File(...),
        sampling_rate: float = Query(360, ge=128, le=2000),
        prefiltered: bool = False,
    ):
        classifier = app.state.classifier
        if classifier is None:
            raise HTTPException(
                503,
                "Compatible trained model unavailable; signal analysis is still available",
            )
        if sampling_rate != 360 or not prefiltered:
            raise HTTPException(
                400,
                "Provide a 360 Hz beat extracted after record-level contract filtering; use /analyze for raw signals",
            )
        values = read_signal(file)
        if len(values) != 187:
            raise HTTPException(
                400, "Expected exactly 187 samples; no silent truncation or padding"
            )
        with lock:
            label, confidence, probabilities = classifier.predict(normalize_signal(values))
        return {
            "success": True,
            "predicted_class": label,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    @app.get("/model/info")
    def info():
        classifier = app.state.classifier
        if classifier is None:
            raise HTTPException(503, "Compatible trained model unavailable")
        return classifier.metadata

    return app


app = create_app(
    Path(os.getenv("BIOSIGNAL_MODEL_PATH", ROOT / "data/models/ecg_arrhythmia_classifier.keras"))
)
