"""
FastAPI backend for ECG analysis API endpoints.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from io import StringIO
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signal_processing.filters import preprocess_ecg, normalize_signal
from src.signal_processing.peak_detection import detect_r_peaks, calculate_heart_rate, calculate_rr_intervals
from src.signal_processing.features import calculate_hrv_time_domain, calculate_hrv_frequency_domain
from src.models.cnn_model import ArrhythmiaClassifier


app = FastAPI(
    title="BioSignal Analyzer API",
    description="API for ECG signal analysis and arrhythmia detection",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model = None


@app.on_event("startup")
async def load_model_on_startup():
    global model
    model_path = "data/models/ecg_arrhythmia_classifier.h5"

    if os.path.exists(model_path):
        model = ArrhythmiaClassifier(model_path=model_path)
    else:
        model = ArrhythmiaClassifier()


# Pydantic models
class AnalysisResult(BaseModel):
    mean_heart_rate: float
    num_peaks: int
    hrv_metrics: Dict[str, float]
    signal_quality: str


class PredictionResult(BaseModel):
    beat_number: int
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]


class ECGAnalysisResponse(BaseModel):
    success: bool
    duration: float
    sampling_rate: float
    analysis: AnalysisResult
    predictions: List[PredictionResult]
    message: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "BioSignal Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/analyze": "Analyze ECG signal (POST)",
            "/predict": "Predict arrhythmia (POST)",
            "/docs": "API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/analyze", response_model=ECGAnalysisResponse)
async def analyze_ecg(
    file: UploadFile = File(...),
    sampling_rate: Optional[float] = 360.0
):
    """
    Analyze ECG signal from uploaded file.

    Args:
        file: CSV file with ECG signal
        sampling_rate: Sampling frequency in Hz (default: 360)

    Returns:
        Complete analysis results including HRV and predictions
    """
    try:
        # Read file
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))

        # Extract signal
        signal_col = None
        for col in ['signal', 'ecg', 'value', 'amplitude', df.columns[0]]:
            if col in df.columns:
                signal_col = col
                break

        if signal_col is None:
            signal_col = df.columns[0]

        signal = df[signal_col].values
        fs = sampling_rate

        # Preprocess
        filtered_signal = preprocess_ecg(signal, fs)
        normalized_signal = normalize_signal(filtered_signal)

        # Peak detection
        peaks = detect_r_peaks(filtered_signal, fs)
        mean_hr, _ = calculate_heart_rate(peaks, fs)
        rr_intervals = calculate_rr_intervals(peaks, fs)

        # HRV analysis
        hrv_time = calculate_hrv_time_domain(rr_intervals)
        hrv_freq = calculate_hrv_frequency_domain(rr_intervals)

        # Combine HRV metrics
        hrv_metrics = {**hrv_time, **hrv_freq}

        # Signal quality assessment
        if len(peaks) < 5:
            quality = "poor"
        elif len(peaks) < 20:
            quality = "fair"
        else:
            quality = "good"

        # Predictions
        predictions = []
        segment_length = 187
        half_length = segment_length // 2

        for i, peak in enumerate(peaks[:10]):
            start = max(0, peak - half_length)
            end = min(len(filtered_signal), peak + half_length)

            if end - start == segment_length:
                segment = normalized_signal[start:end]
                pred_class, confidence, probs = model.predict(segment)

                predictions.append(PredictionResult(
                    beat_number=i + 1,
                    predicted_class=pred_class,
                    confidence=confidence,
                    probabilities=probs
                ))

        # Create response
        analysis = AnalysisResult(
            mean_heart_rate=float(mean_hr),
            num_peaks=len(peaks),
            hrv_metrics=hrv_metrics,
            signal_quality=quality
        )

        response = ECGAnalysisResponse(
            success=True,
            duration=len(signal) / fs,
            sampling_rate=fs,
            analysis=analysis,
            predictions=predictions,
            message="Analysis completed successfully"
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")


@app.post("/predict")
async def predict_arrhythmia(
    file: UploadFile = File(...),
    sampling_rate: Optional[float] = 360.0
):
    """
    Predict arrhythmia from ECG segment.

    Args:
        file: CSV file with single ECG beat (187 samples)
        sampling_rate: Sampling frequency in Hz

    Returns:
        Prediction results
    """
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))

        # Get signal
        signal_col = df.columns[0]
        signal = df[signal_col].values

        # Normalize
        signal = normalize_signal(signal)

        # Ensure correct length
        if len(signal) != 187:
            # Resample or pad
            if len(signal) > 187:
                signal = signal[:187]
            else:
                signal = np.pad(signal, (0, 187 - len(signal)), mode='edge')

        # Predict
        pred_class, confidence, probs = model.predict(signal)

        return {
            "success": True,
            "predicted_class": pred_class,
            "confidence": confidence,
            "probabilities": probs
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        return {"error": "Model not loaded"}

    return {
        "model_type": "1D CNN",
        "classes": model.CLASS_NAMES,
        "input_shape": "(187, 1)",
        "architecture": "Conv1D + BatchNorm + MaxPooling + Dense"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
