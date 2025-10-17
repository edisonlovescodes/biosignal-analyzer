"""
Peak detection algorithms for R-peak identification in ECG signals.
"""

import numpy as np
from scipy import signal
from typing import Tuple, List


def detect_r_peaks(
    ecg_signal: np.ndarray,
    fs: float,
    method: str = 'pantompkins'
) -> np.ndarray:
    """
    Detect R-peaks in ECG signal using Pan-Tompkins algorithm.

    Args:
        ecg_signal: Preprocessed ECG signal
        fs: Sampling frequency (Hz)
        method: Detection method ('pantompkins' or 'simple')

    Returns:
        Array of R-peak indices
    """
    if method == 'pantompkins':
        return _pan_tompkins(ecg_signal, fs)
    else:
        return _simple_peak_detection(ecg_signal, fs)


def _pan_tompkins(ecg_signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Pan-Tompkins algorithm for QRS detection.

    Steps:
    1. Bandpass filter (5-15 Hz)
    2. Derivative
    3. Squaring
    4. Moving window integration
    5. Adaptive thresholding
    """
    # Bandpass filter
    nyquist = 0.5 * fs
    b, a = signal.butter(4, [5/nyquist, 15/nyquist], btype='band')
    filtered = signal.filtfilt(b, a, ecg_signal)

    # Derivative
    derivative = np.diff(filtered)

    # Squaring
    squared = derivative ** 2

    # Moving window integration
    window_size = int(0.15 * fs)  # 150ms window
    integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')

    # Find peaks
    min_distance = int(0.2 * fs)  # Minimum 200ms between peaks (300 bpm max)
    peaks, _ = signal.find_peaks(
        integrated,
        distance=min_distance,
        prominence=np.max(integrated) * 0.3
    )

    return peaks


def _simple_peak_detection(ecg_signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Simple peak detection using scipy.
    """
    min_distance = int(0.2 * fs)
    peaks, _ = signal.find_peaks(
        ecg_signal,
        distance=min_distance,
        prominence=np.std(ecg_signal)
    )

    return peaks


def calculate_rr_intervals(peaks: np.ndarray, fs: float) -> np.ndarray:
    """
    Calculate RR intervals from R-peaks.

    Args:
        peaks: R-peak indices
        fs: Sampling frequency (Hz)

    Returns:
        RR intervals in milliseconds
    """
    if len(peaks) < 2:
        return np.array([])

    rr_intervals = np.diff(peaks) / fs * 1000  # Convert to ms
    return rr_intervals


def calculate_heart_rate(peaks: np.ndarray, fs: float) -> Tuple[float, np.ndarray]:
    """
    Calculate heart rate from R-peaks.

    Args:
        peaks: R-peak indices
        fs: Sampling frequency (Hz)

    Returns:
        Tuple of (mean heart rate in bpm, instantaneous heart rates)
    """
    rr_intervals = calculate_rr_intervals(peaks, fs)

    if len(rr_intervals) == 0:
        return 0.0, np.array([])

    # Convert RR intervals (ms) to heart rate (bpm)
    heart_rates = 60000 / rr_intervals
    mean_hr = np.mean(heart_rates)

    return mean_hr, heart_rates


def segment_heartbeats(
    ecg_signal: np.ndarray,
    peaks: np.ndarray,
    fs: float,
    window_size: float = 0.6
) -> List[np.ndarray]:
    """
    Segment individual heartbeats around R-peaks.

    Args:
        ecg_signal: ECG signal
        peaks: R-peak indices
        fs: Sampling frequency (Hz)
        window_size: Window size in seconds (centered on R-peak)

    Returns:
        List of segmented heartbeats
    """
    half_window = int(window_size * fs / 2)
    beats = []

    for peak in peaks:
        start = max(0, peak - half_window)
        end = min(len(ecg_signal), peak + half_window)

        if end - start == 2 * half_window:  # Only keep complete beats
            beats.append(ecg_signal[start:end])

    return beats
