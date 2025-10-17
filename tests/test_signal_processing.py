"""
Unit tests for signal processing module.
"""

import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signal_processing.filters import (
    bandpass_filter, notch_filter, remove_baseline_wander,
    preprocess_ecg, normalize_signal
)
from src.signal_processing.peak_detection import (
    detect_r_peaks, calculate_rr_intervals, calculate_heart_rate
)
from src.signal_processing.features import (
    calculate_hrv_time_domain, calculate_hrv_frequency_domain,
    compute_frequency_spectrum, extract_statistical_features
)


class TestFilters:
    """Test signal filtering functions."""

    def test_bandpass_filter(self):
        # Create test signal
        fs = 360
        t = np.linspace(0, 1, fs)
        signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave

        filtered = bandpass_filter(signal, 5, 15, fs)

        assert len(filtered) == len(signal)
        assert not np.array_equal(filtered, signal)

    def test_notch_filter(self):
        fs = 360
        t = np.linspace(0, 1, fs)
        signal = np.sin(2 * np.pi * 60 * t)  # 60 Hz (powerline)

        filtered = notch_filter(signal, 60, fs)

        assert len(filtered) == len(signal)
        # Power at 60 Hz should be reduced
        assert np.max(np.abs(filtered)) < np.max(np.abs(signal))

    def test_remove_baseline_wander(self):
        fs = 360
        t = np.linspace(0, 10, fs * 10)
        baseline = 0.5 * np.sin(2 * np.pi * 0.2 * t)  # Slow drift
        signal = np.sin(2 * np.pi * 5 * t) + baseline

        filtered = remove_baseline_wander(signal, fs)

        assert len(filtered) == len(signal)
        assert np.mean(np.abs(filtered)) < np.mean(np.abs(signal))

    def test_preprocess_ecg(self):
        fs = 360
        t = np.linspace(0, 10, fs * 10)
        signal = np.random.randn(len(t))

        processed = preprocess_ecg(signal, fs)

        assert len(processed) == len(signal)
        assert isinstance(processed, np.ndarray)

    def test_normalize_signal(self):
        signal = np.random.randn(1000) * 5 + 10

        normalized = normalize_signal(signal)

        assert len(normalized) == len(signal)
        assert abs(np.mean(normalized)) < 1e-10  # Mean should be ~0
        assert abs(np.std(normalized) - 1.0) < 1e-10  # Std should be ~1


class TestPeakDetection:
    """Test peak detection functions."""

    def test_detect_r_peaks(self):
        # Create synthetic ECG
        fs = 360
        duration = 10
        hr = 75  # bpm
        t = np.linspace(0, duration, fs * duration)

        signal = np.zeros_like(t)
        for beat_time in np.arange(0, duration, 60/hr):
            beat_idx = int(beat_time * fs)
            if beat_idx < len(signal) - 50:
                signal[beat_idx:beat_idx+20] += np.exp(-np.linspace(0, 3, 20))

        peaks = detect_r_peaks(signal, fs)

        assert len(peaks) > 0
        assert len(peaks) < len(signal)
        # Should detect roughly the correct number of beats
        expected_beats = duration * hr / 60
        assert abs(len(peaks) - expected_beats) < 5

    def test_calculate_rr_intervals(self):
        peaks = np.array([100, 460, 820, 1180])  # ~360 samples apart at 360 Hz = 1s = 60 bpm
        fs = 360

        rr_intervals = calculate_rr_intervals(peaks, fs)

        assert len(rr_intervals) == len(peaks) - 1
        assert all(rr > 0 for rr in rr_intervals)
        # Should be ~1000 ms
        assert all(900 < rr < 1100 for rr in rr_intervals)

    def test_calculate_heart_rate(self):
        peaks = np.array([100, 460, 820, 1180])
        fs = 360

        mean_hr, heart_rates = calculate_heart_rate(peaks, fs)

        assert mean_hr > 0
        assert len(heart_rates) == len(peaks) - 1
        # Should be ~60 bpm
        assert 55 < mean_hr < 65


class TestFeatures:
    """Test feature extraction functions."""

    def test_calculate_hrv_time_domain(self):
        # Regular RR intervals
        rr_intervals = np.array([1000, 1020, 980, 1000, 990])

        hrv = calculate_hrv_time_domain(rr_intervals)

        assert 'mean_rr' in hrv
        assert 'sdnn' in hrv
        assert 'rmssd' in hrv
        assert 'pnn50' in hrv

        assert hrv['mean_rr'] > 0
        assert hrv['sdnn'] >= 0
        assert hrv['rmssd'] >= 0
        assert 0 <= hrv['pnn50'] <= 100

    def test_calculate_hrv_frequency_domain(self):
        # Create varying RR intervals
        rr_intervals = 1000 + 50 * np.sin(np.linspace(0, 4*np.pi, 100))

        hrv_freq = calculate_hrv_frequency_domain(rr_intervals)

        assert 'vlf_power' in hrv_freq
        assert 'lf_power' in hrv_freq
        assert 'hf_power' in hrv_freq
        assert 'lf_hf_ratio' in hrv_freq

        assert all(v >= 0 for v in hrv_freq.values())

    def test_compute_frequency_spectrum(self):
        fs = 360
        t = np.linspace(0, 1, fs)
        signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz signal

        freqs, mags = compute_frequency_spectrum(signal, fs)

        assert len(freqs) == len(mags)
        assert all(f >= 0 for f in freqs)
        assert all(m >= 0 for m in mags)

        # Peak should be around 10 Hz
        peak_freq = freqs[np.argmax(mags)]
        assert 9 < peak_freq < 11

    def test_extract_statistical_features(self):
        signal = np.random.randn(1000)

        features = extract_statistical_features(signal)

        assert 'mean' in features
        assert 'std' in features
        assert 'min' in features
        assert 'max' in features
        assert 'median' in features
        assert 'skewness' in features
        assert 'kurtosis' in features

        assert features['min'] <= features['median'] <= features['max']


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_signal(self):
        signal = np.array([])
        fs = 360

        # Should not crash
        with pytest.raises((ValueError, IndexError)):
            bandpass_filter(signal, 0.5, 40, fs)

    def test_very_short_signal(self):
        signal = np.array([1, 2, 3])
        fs = 360

        peaks = detect_r_peaks(signal, fs)
        assert len(peaks) == 0

    def test_no_peaks(self):
        signal = np.ones(1000)  # Flat signal
        fs = 360

        peaks = detect_r_peaks(signal, fs)
        rr_intervals = calculate_rr_intervals(peaks, fs)

        assert len(peaks) == 0
        assert len(rr_intervals) == 0

    def test_single_peak(self):
        peaks = np.array([100])
        fs = 360

        rr_intervals = calculate_rr_intervals(peaks, fs)
        mean_hr, heart_rates = calculate_heart_rate(peaks, fs)

        assert len(rr_intervals) == 0
        assert mean_hr == 0
        assert len(heart_rates) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
