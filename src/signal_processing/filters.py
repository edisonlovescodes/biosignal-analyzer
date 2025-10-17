"""
Signal filtering functions for ECG/EEG preprocessing.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional


def bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4
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

    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data)

    return filtered


def notch_filter(
    data: np.ndarray,
    notch_freq: float,
    fs: float,
    quality: float = 30.0
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
    freq = notch_freq / nyquist

    b, a = signal.iirnotch(freq, quality, fs)
    filtered = signal.filtfilt(b, a, data)

    return filtered


def remove_baseline_wander(
    data: np.ndarray,
    fs: float,
    cutoff: float = 0.5
) -> np.ndarray:
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

    b, a = signal.butter(4, normal_cutoff, btype='high')
    filtered = signal.filtfilt(b, a, data)

    return filtered


def preprocess_ecg(
    data: np.ndarray,
    fs: float,
    apply_notch: bool = True,
    notch_freq: float = 60.0
) -> np.ndarray:
    """
    Complete ECG preprocessing pipeline.

    Applies:
    1. Baseline wander removal
    2. Bandpass filter (0.5-40 Hz for ECG)
    3. Notch filter for powerline interference (optional)

    Args:
        data: Raw ECG signal
        fs: Sampling frequency (Hz)
        apply_notch: Whether to apply notch filter
        notch_freq: Powerline frequency (50 or 60 Hz)

    Returns:
        Preprocessed ECG signal
    """
    # Remove baseline wander
    signal_clean = remove_baseline_wander(data, fs)

    # Bandpass filter
    signal_clean = bandpass_filter(signal_clean, 0.5, 40.0, fs)

    # Notch filter for powerline interference
    if apply_notch:
        signal_clean = notch_filter(signal_clean, notch_freq, fs)

    return signal_clean


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
