"""
Feature extraction for ECG signal analysis.
"""

import numpy as np
from scipy import signal, stats
from typing import Dict, Tuple


def calculate_hrv_time_domain(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Calculate time-domain HRV (Heart Rate Variability) features.

    Args:
        rr_intervals: RR intervals in milliseconds

    Returns:
        Dictionary of HRV metrics
    """
    if len(rr_intervals) < 2:
        return {
            'mean_rr': 0.0,
            'sdnn': 0.0,
            'rmssd': 0.0,
            'pnn50': 0.0
        }

    # SDNN: Standard deviation of RR intervals
    sdnn = np.std(rr_intervals)

    # RMSSD: Root mean square of successive differences
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))

    # pNN50: Percentage of successive RR intervals that differ by more than 50ms
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = (nn50 / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0.0

    return {
        'mean_rr': float(np.mean(rr_intervals)),
        'sdnn': float(sdnn),
        'rmssd': float(rmssd),
        'pnn50': float(pnn50)
    }


def calculate_hrv_frequency_domain(
    rr_intervals: np.ndarray,
    fs_resample: float = 4.0
) -> Dict[str, float]:
    """
    Calculate frequency-domain HRV features using power spectral density.

    Args:
        rr_intervals: RR intervals in milliseconds
        fs_resample: Resampling frequency (Hz)

    Returns:
        Dictionary of frequency-domain HRV metrics
    """
    if len(rr_intervals) < 10:
        return {
            'vlf_power': 0.0,
            'lf_power': 0.0,
            'hf_power': 0.0,
            'lf_hf_ratio': 0.0
        }

    # Interpolate RR intervals to create evenly sampled signal
    time = np.cumsum(rr_intervals) / 1000  # Convert to seconds
    time_interp = np.arange(0, time[-1], 1/fs_resample)

    rr_interp = np.interp(time_interp, time[:-1], rr_intervals[:-1])

    # Calculate power spectral density
    freqs, psd = signal.welch(rr_interp, fs=fs_resample, nperseg=256)

    # Define frequency bands
    vlf_band = (0.003, 0.04)  # Very low frequency
    lf_band = (0.04, 0.15)     # Low frequency
    hf_band = (0.15, 0.4)      # High frequency

    # Calculate power in each band
    vlf_power = np.trapz(psd[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])])
    lf_power = np.trapz(psd[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
    hf_power = np.trapz(psd[(freqs >= hf_band[0]) & (freqs < hf_band[1])])

    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0.0

    return {
        'vlf_power': float(vlf_power),
        'lf_power': float(lf_power),
        'hf_power': float(hf_power),
        'lf_hf_ratio': float(lf_hf_ratio)
    }


def compute_frequency_spectrum(
    ecg_signal: np.ndarray,
    fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute frequency spectrum using FFT.

    Args:
        ecg_signal: ECG signal
        fs: Sampling frequency (Hz)

    Returns:
        Tuple of (frequencies, magnitudes)
    """
    n = len(ecg_signal)
    fft_vals = np.fft.rfft(ecg_signal)
    fft_freq = np.fft.rfftfreq(n, 1/fs)
    fft_mag = np.abs(fft_vals)

    return fft_freq, fft_mag


def extract_statistical_features(signal_data: np.ndarray) -> Dict[str, float]:
    """
    Extract basic statistical features from signal.

    Args:
        signal_data: Input signal

    Returns:
        Dictionary of statistical features
    """
    return {
        'mean': float(np.mean(signal_data)),
        'std': float(np.std(signal_data)),
        'min': float(np.min(signal_data)),
        'max': float(np.max(signal_data)),
        'median': float(np.median(signal_data)),
        'skewness': float(stats.skew(signal_data)),
        'kurtosis': float(stats.kurtosis(signal_data))
    }


def extract_all_features(
    ecg_signal: np.ndarray,
    rr_intervals: np.ndarray,
    fs: float
) -> Dict[str, float]:
    """
    Extract all features from ECG signal.

    Args:
        ecg_signal: Preprocessed ECG signal
        rr_intervals: RR intervals in milliseconds
        fs: Sampling frequency (Hz)

    Returns:
        Dictionary containing all extracted features
    """
    features = {}

    # Time-domain HRV
    features.update(calculate_hrv_time_domain(rr_intervals))

    # Frequency-domain HRV
    features.update(calculate_hrv_frequency_domain(rr_intervals))

    # Statistical features
    stat_features = extract_statistical_features(ecg_signal)
    features.update({f'signal_{k}': v for k, v in stat_features.items()})

    return features
