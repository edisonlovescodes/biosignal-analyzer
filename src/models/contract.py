"""Shared coarse beat-label, filtering and segmentation contract (not an AF rhythm task)."""

import numpy as np
from src.signal_processing.filters import normalize_signal, preprocess_ecg

PROTOCOL = "mitbih-coarse-beats-v1"
CLASS_NAMES = ["Normal/Conduction", "PAC", "PVC", "Fusion", "Other"]
ANNOTATION_MAP = {
    **dict.fromkeys(["N", "L", "R"], CLASS_NAMES[0]),
    **dict.fromkeys(["A", "a", "J", "S"], "PAC"),
    "V": "PVC",
    "F": "Fusion",
    "f": "Fusion",
    "/": "Other",
    "Q": "Other",
    "?": "Other",
}
# Both fusion annotations are coarsely grouped; F is ventricular/normal, f paced/normal.
PREPROCESSING = {
    "sampling_rate": 360,
    "lead": 0,
    "segment_length": 187,
    "bandpass_hz": [0.5, 40],
    "notch_hz": 60,
    "normalization": "per-beat z-score",
}


def extract_beat(filtered_signal, peak, segment_length=187):
    before = segment_length // 2
    start = int(peak) - before
    end = start + segment_length
    if start < 0 or end > len(filtered_signal):
        return None
    return normalize_signal(np.asarray(filtered_signal[start:end], dtype=float))


def filter_record(values, sampling_rate):
    if sampling_rate != 360:
        raise ValueError(
            "Classifier contract requires a 360 Hz lead-0 record; resampling is not implemented"
        )
    return preprocess_ecg(values, sampling_rate, apply_notch=True, notch_freq=60)


def subject_id(record):
    # PhysioNet documents these records as the same subject.
    return "201-202" if str(record) in {"201", "202"} else str(record)
