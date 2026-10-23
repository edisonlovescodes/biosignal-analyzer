"""
Signal filtering functions for ECG/EEG preprocessing.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional


def bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to remove noise outside the frequency range.

    Args:
        data: Input signal
        lowcut: Low frequency cutoff (Hz)
        highcut: High frequency cutoff (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order

    Returns:
        Filtered signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = signal.butter(order, [low, high], btype="band")
    filtered = signal.filtfilt(b, a, data)

    return filtered


def notch_filter(
    data: np.ndarray, notch_freq: float, fs: float, quality: float = 30.0
) -> np.ndarray:
    """
    Apply notch filter to remove powerline interference (50/60 Hz).

    Args:
        data: Input signal
        notch_freq: Frequency to remove (Hz)
        fs: Sampling frequency (Hz)
        quality: Quality factor

    Returns:
        Filtered signal
    """
    nyquist = 0.5 * fs
    if not 0 < notch_freq < nyquist or quality <= 0:
        raise ValueError("Notch frequency must be below Nyquist and quality positive")
    b, a = signal.iirnotch(notch_freq, quality, fs=fs)
    filtered = signal.filtfilt(b, a, data)

    return filtered


def remove_baseline_wander(data: np.ndarray, fs: float, cutoff: float = 0.5) -> np.ndarray:
    """
    Remove baseline wander using high-pass filter.

    Args:
        data: Input signal
        fs: Sampling frequency (Hz)
        cutoff: Cutoff frequency (Hz)

    Returns:
        Signal with baseline removed
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    b, a = signal.butter(4, normal_cutoff, btype="high")
    filtered = signal.filtfilt(b, a, data)

    return filtered


def preprocess_ecg(data: np.ndarray, fs: float, *, notch_freq: float | None) -> np.ndarray:
    """Remove baseline, bandpass 0.5–40 Hz, then optionally reject one mains frequency."""
    filtered = bandpass_filter(remove_baseline_wander(data, fs), 0.5, 40, fs)
    if notch_freq is None:
        return filtered
    return notch_filter(filtered, notch_freq, fs)


def normalize_signal(data: np.ndarray) -> np.ndarray:
    """
    Normalize signal to zero mean and unit variance.

    Args:
        data: Input signal

    Returns:
        Normalized signal
    """
    mean = np.mean(data)
    std = np.std(data)

    if std == 0:
        return data - mean

    return (data - mean) / std
