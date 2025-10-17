"""
Visualization functions for ECG signals and analysis results.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple


def plot_ecg_signal(
    signal: np.ndarray,
    fs: float,
    peaks: Optional[np.ndarray] = None,
    title: str = "ECG Signal",
    height: int = 400
) -> go.Figure:
    """
    Plot ECG signal with optional R-peak markers.

    Args:
        signal: ECG signal
        fs: Sampling frequency (Hz)
        peaks: R-peak indices (optional)
        title: Plot title
        height: Plot height in pixels

    Returns:
        Plotly figure
    """
    time = np.arange(len(signal)) / fs

    fig = go.Figure()

    # Plot signal
    fig.add_trace(go.Scatter(
        x=time,
        y=signal,
        mode='lines',
        name='ECG',
        line=dict(color='#2E86AB', width=1.5)
    ))

    # Plot R-peaks if provided
    if peaks is not None and len(peaks) > 0:
        fig.add_trace(go.Scatter(
            x=time[peaks],
            y=signal[peaks],
            mode='markers',
            name='R-peaks',
            marker=dict(color='#A23B72', size=8, symbol='x')
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (mV)",
        height=height,
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def plot_comparison(
    original: np.ndarray,
    filtered: np.ndarray,
    fs: float,
    title: str = "Signal Comparison"
) -> go.Figure:
    """
    Plot original and filtered signals for comparison.

    Args:
        original: Original signal
        filtered: Filtered signal
        fs: Sampling frequency (Hz)
        title: Plot title

    Returns:
        Plotly figure
    """
    time = np.arange(len(original)) / fs

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Original Signal', 'Filtered Signal'),
        vertical_spacing=0.12
    )

    # Original
    fig.add_trace(
        go.Scatter(x=time, y=original, mode='lines', name='Original',
                   line=dict(color='#8B8B8B')),
        row=1, col=1
    )

    # Filtered
    fig.add_trace(
        go.Scatter(x=time, y=filtered, mode='lines', name='Filtered',
                   line=dict(color='#2E86AB')),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude (mV)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude (mV)", row=2, col=1)

    fig.update_layout(
        title_text=title,
        height=600,
        showlegend=False,
        template='plotly_white'
    )

    return fig


def plot_frequency_spectrum(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    title: str = "Frequency Spectrum"
) -> go.Figure:
    """
    Plot frequency spectrum.

    Args:
        frequencies: Frequency values (Hz)
        magnitudes: Magnitude values
        title: Plot title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=frequencies,
        y=magnitudes,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#F18F01', width=2)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        height=400,
        template='plotly_white'
    )

    # Limit x-axis to meaningful range
    fig.update_xaxes(range=[0, 50])

    return fig


def plot_heart_rate(
    peaks: np.ndarray,
    fs: float,
    heart_rates: np.ndarray
) -> go.Figure:
    """
    Plot instantaneous heart rate over time.

    Args:
        peaks: R-peak indices
        fs: Sampling frequency (Hz)
        heart_rates: Instantaneous heart rates (bpm)

    Returns:
        Plotly figure
    """
    if len(peaks) < 2:
        return go.Figure()

    time = peaks[1:] / fs  # Time of each RR interval

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time,
        y=heart_rates,
        mode='lines+markers',
        name='Heart Rate',
        line=dict(color='#C73E1D', width=2),
        marker=dict(size=6)
    ))

    # Add mean line
    mean_hr = np.mean(heart_rates)
    fig.add_hline(
        y=mean_hr,
        line_dash="dash",
        line_color="#666",
        annotation_text=f"Mean: {mean_hr:.1f} bpm"
    )

    fig.update_layout(
        title="Instantaneous Heart Rate",
        xaxis_title="Time (s)",
        yaxis_title="Heart Rate (bpm)",
        height=400,
        template='plotly_white'
    )

    return fig


def plot_rr_intervals(rr_intervals: np.ndarray) -> go.Figure:
    """
    Plot RR interval tachogram.

    Args:
        rr_intervals: RR intervals in milliseconds

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.arange(len(rr_intervals)),
        y=rr_intervals,
        mode='lines+markers',
        name='RR Intervals',
        line=dict(color='#6A4C93', width=2),
        marker=dict(size=4)
    ))

    fig.update_layout(
        title="RR Interval Tachogram",
        xaxis_title="Beat Number",
        yaxis_title="RR Interval (ms)",
        height=400,
        template='plotly_white'
    )

    return fig


def plot_poincare(rr_intervals: np.ndarray) -> go.Figure:
    """
    Create Poincaré plot for HRV analysis.

    Args:
        rr_intervals: RR intervals in milliseconds

    Returns:
        Plotly figure
    """
    if len(rr_intervals) < 2:
        return go.Figure()

    rr_n = rr_intervals[:-1]
    rr_n1 = rr_intervals[1:]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rr_n,
        y=rr_n1,
        mode='markers',
        marker=dict(
            color='#2E86AB',
            size=6,
            opacity=0.6
        ),
        name='RR(n) vs RR(n+1)'
    ))

    # Add identity line
    min_val = min(rr_n.min(), rr_n1.min())
    max_val = max(rr_n.max(), rr_n1.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='#666', dash='dash'),
        name='Identity Line'
    ))

    fig.update_layout(
        title="Poincaré Plot",
        xaxis_title="RR(n) (ms)",
        yaxis_title="RR(n+1) (ms)",
        height=500,
        width=550,
        template='plotly_white'
    )

    return fig


def plot_hrv_features(hrv_metrics: Dict[str, float]) -> go.Figure:
    """
    Plot HRV metrics as bar chart.

    Args:
        hrv_metrics: Dictionary of HRV features

    Returns:
        Plotly figure
    """
    # Separate time and frequency domain metrics
    time_domain = {k: v for k, v in hrv_metrics.items()
                   if k in ['mean_rr', 'sdnn', 'rmssd', 'pnn50']}
    freq_domain = {k: v for k, v in hrv_metrics.items()
                   if k in ['vlf_power', 'lf_power', 'hf_power', 'lf_hf_ratio']}

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Time Domain HRV', 'Frequency Domain HRV')
    )

    # Time domain
    if time_domain:
        fig.add_trace(
            go.Bar(
                x=list(time_domain.keys()),
                y=list(time_domain.values()),
                marker_color='#2E86AB',
                name='Time Domain'
            ),
            row=1, col=1
        )

    # Frequency domain
    if freq_domain:
        fig.add_trace(
            go.Bar(
                x=list(freq_domain.keys()),
                y=list(freq_domain.values()),
                marker_color='#F18F01',
                name='Frequency Domain'
            ),
            row=1, col=2
        )

    fig.update_layout(
        height=400,
        showlegend=False,
        template='plotly_white'
    )

    return fig


def plot_prediction_probabilities(probabilities: Dict[str, float]) -> go.Figure:
    """
    Plot prediction probabilities as horizontal bar chart.

    Args:
        probabilities: Dictionary of class probabilities

    Returns:
        Plotly figure
    """
    classes = list(probabilities.keys())
    probs = list(probabilities.values())

    # Sort by probability
    sorted_indices = np.argsort(probs)[::-1]
    classes = [classes[i] for i in sorted_indices]
    probs = [probs[i] for i in sorted_indices]

    # Color the highest probability differently
    colors = ['#A23B72' if i == 0 else '#2E86AB' for i in range(len(classes))]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=probs,
        y=classes,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{p:.1%}' for p in probs],
        textposition='auto'
    ))

    fig.update_layout(
        title="Arrhythmia Classification Probabilities",
        xaxis_title="Probability",
        yaxis_title="Class",
        height=350,
        template='plotly_white'
    )

    fig.update_xaxes(range=[0, 1])

    return fig


def create_dashboard_figure(
    signal: np.ndarray,
    filtered: np.ndarray,
    peaks: np.ndarray,
    fs: float,
    freqs: np.ndarray,
    mags: np.ndarray,
    heart_rates: np.ndarray
) -> go.Figure:
    """
    Create comprehensive dashboard with multiple subplots.

    Args:
        signal: Original signal
        filtered: Filtered signal
        peaks: R-peak indices
        fs: Sampling frequency
        freqs: Frequency values
        mags: Magnitude values
        heart_rates: Heart rate values

    Returns:
        Plotly figure with subplots
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'ECG Signal with R-peaks',
            'Frequency Spectrum',
            'Instantaneous Heart Rate',
            'Signal Comparison'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    time = np.arange(len(signal)) / fs

    # ECG with peaks
    fig.add_trace(
        go.Scatter(x=time, y=filtered, mode='lines', name='ECG',
                   line=dict(color='#2E86AB')),
        row=1, col=1
    )
    if len(peaks) > 0:
        fig.add_trace(
            go.Scatter(x=time[peaks], y=filtered[peaks], mode='markers',
                       name='R-peaks', marker=dict(color='#A23B72', size=6)),
            row=1, col=1
        )

    # Frequency spectrum
    fig.add_trace(
        go.Scatter(x=freqs, y=mags, mode='lines', fill='tozeroy',
                   name='Spectrum', line=dict(color='#F18F01')),
        row=1, col=2
    )

    # Heart rate
    if len(peaks) > 1:
        hr_time = peaks[1:] / fs
        fig.add_trace(
            go.Scatter(x=hr_time, y=heart_rates, mode='lines+markers',
                       name='HR', line=dict(color='#C73E1D')),
            row=2, col=1
        )

    # Signal comparison
    fig.add_trace(
        go.Scatter(x=time, y=signal, mode='lines', name='Original',
                   line=dict(color='#8B8B8B')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=time, y=filtered, mode='lines', name='Filtered',
                   line=dict(color='#2E86AB')),
        row=2, col=2
    )

    fig.update_layout(height=800, showlegend=False, template='plotly_white')

    return fig
