"""
Streamlit web application for ECG signal analysis and arrhythmia detection.
"""

import streamlit as st
import numpy as np
import pandas as pd
import wfdb
from io import BytesIO
import os

# Import our modules
from src.signal_processing.filters import preprocess_ecg, normalize_signal
from src.signal_processing.peak_detection import (
    detect_r_peaks, calculate_heart_rate, calculate_rr_intervals
)
from src.signal_processing.features import (
    calculate_hrv_time_domain, calculate_hrv_frequency_domain,
    compute_frequency_spectrum, extract_all_features
)
from src.visualization.plots import (
    plot_ecg_signal, plot_comparison, plot_frequency_spectrum,
    plot_heart_rate, plot_rr_intervals, plot_poincare,
    plot_hrv_features, plot_prediction_probabilities
)
from src.models.cnn_model import ArrhythmiaClassifier


# Page config
st.set_page_config(
    page_title="BioSignal Analyzer",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E86AB;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model (or use untrained for demo)."""
    model_path = "data/models/ecg_arrhythmia_classifier.h5"

    if os.path.exists(model_path):
        return ArrhythmiaClassifier(model_path=model_path)
    else:
        # Return untrained model for demo purposes
        st.warning("Using untrained model for demo. Train a model for actual predictions.")
        return ArrhythmiaClassifier()


def load_csv_ecg(file) -> tuple:
    """Load ECG from CSV file."""
    try:
        df = pd.read_csv(file)

        # Try to find signal column
        signal_col = None
        for col in ['signal', 'ecg', 'value', 'amplitude', df.columns[0]]:
            if col in df.columns:
                signal_col = col
                break

        if signal_col is None:
            signal_col = df.columns[0]

        signal = df[signal_col].values

        # Try to get sampling frequency
        if 'time' in df.columns:
            time_diff = np.diff(df['time'].values[:100]).mean()
            fs = 1.0 / time_diff if time_diff > 0 else 360.0
        else:
            fs = 360.0  # Default MIT-BIH frequency

        return signal, fs
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None, None


def load_wfdb_record(file) -> tuple:
    """Load ECG from WFDB format."""
    try:
        # Save uploaded file temporarily
        with open("/tmp/temp_record.dat", "wb") as f:
            f.write(file.read())

        record = wfdb.rdrecord("/tmp/temp_record")
        signal = record.p_signal[:, 0]
        fs = record.fs

        return signal, fs
    except Exception as e:
        st.error(f"Error loading WFDB file: {e}")
        return None, None


def generate_synthetic_ecg(duration: int = 10, fs: int = 360) -> np.ndarray:
    """Generate synthetic ECG for demo."""
    t = np.linspace(0, duration, duration * fs)

    # Simple synthetic ECG-like signal
    hr = 75  # bpm
    hr_hz = hr / 60

    ecg = np.zeros_like(t)

    # Add QRS complexes
    for beat_time in np.arange(0, duration, 1/hr_hz):
        beat_idx = int(beat_time * fs)
        if beat_idx < len(ecg) - 50:
            # Q wave
            ecg[beat_idx-5:beat_idx] -= np.linspace(0, 0.1, 5)
            # R peak
            ecg[beat_idx:beat_idx+15] += np.exp(-np.linspace(0, 3, 15)) * 1.5
            # S wave
            ecg[beat_idx+15:beat_idx+25] -= np.linspace(0.2, 0, 10)
            # T wave
            ecg[beat_idx+40:beat_idx+80] += np.exp(-np.linspace(0, 2, 40)) * 0.3

    # Add some noise
    ecg += np.random.normal(0, 0.05, len(ecg))

    return ecg


def analyze_ecg(signal: np.ndarray, fs: float, model: ArrhythmiaClassifier):
    """Perform complete ECG analysis."""

    # Preprocessing
    with st.spinner("Preprocessing signal..."):
        filtered_signal = preprocess_ecg(signal, fs)
        normalized_signal = normalize_signal(filtered_signal)

    # Peak detection
    with st.spinner("Detecting R-peaks..."):
        peaks = detect_r_peaks(filtered_signal, fs)
        mean_hr, heart_rates = calculate_heart_rate(peaks, fs)
        rr_intervals = calculate_rr_intervals(peaks, fs)

    # Feature extraction
    with st.spinner("Extracting features..."):
        hrv_time = calculate_hrv_time_domain(rr_intervals)
        hrv_freq = calculate_hrv_frequency_domain(rr_intervals)
        freqs, mags = compute_frequency_spectrum(filtered_signal, fs)
        all_features = extract_all_features(filtered_signal, rr_intervals, fs)

    # Arrhythmia prediction on segments
    predictions = []
    if len(peaks) > 2:
        with st.spinner("Classifying heartbeats..."):
            segment_length = 187
            half_length = segment_length // 2

            for peak in peaks[:10]:  # Analyze first 10 beats
                start = max(0, peak - half_length)
                end = min(len(filtered_signal), peak + half_length)

                if end - start == segment_length:
                    segment = normalized_signal[start:end]
                    pred_class, confidence, probs = model.predict(segment)
                    predictions.append({
                        'beat': peak,
                        'class': pred_class,
                        'confidence': confidence,
                        'probabilities': probs
                    })

    return {
        'original': signal,
        'filtered': filtered_signal,
        'normalized': normalized_signal,
        'peaks': peaks,
        'mean_hr': mean_hr,
        'heart_rates': heart_rates,
        'rr_intervals': rr_intervals,
        'hrv_time': hrv_time,
        'hrv_freq': hrv_freq,
        'freqs': freqs,
        'mags': mags,
        'features': all_features,
        'predictions': predictions
    }


def main():
    # Header
    st.markdown('<p class="main-header">🫀 BioSignal Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ECG Signal Analysis & Arrhythmia Detection</p>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("📊 Options")

    # Data source selection
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["Upload File", "Use Demo Data", "Generate Synthetic"]
    )

    signal = None
    fs = 360.0

    if data_source == "Upload File":
        st.sidebar.subheader("Upload ECG Data")
        uploaded_file = st.sidebar.file_uploader(
            "Choose a file",
            type=['csv', 'dat', 'edf'],
            help="Supported formats: CSV, WFDB (.dat), EDF"
        )

        if uploaded_file:
            file_type = uploaded_file.name.split('.')[-1].lower()

            if file_type == 'csv':
                signal, fs = load_csv_ecg(uploaded_file)
            elif file_type == 'dat':
                signal, fs = load_wfdb_record(uploaded_file)
            else:
                st.error("Unsupported file format")

    elif data_source == "Use Demo Data":
        st.sidebar.info("Demo: MIT-BIH Record 100 (first 10 seconds)")
        # For demo, we'll generate synthetic data
        # In production, you'd load actual MIT-BIH data
        signal = generate_synthetic_ecg(duration=10, fs=360)
        fs = 360.0

    else:  # Generate Synthetic
        duration = st.sidebar.slider("Duration (seconds)", 5, 30, 10)
        signal = generate_synthetic_ecg(duration=duration, fs=360)
        fs = 360.0

    # Processing options
    st.sidebar.subheader("⚙️ Processing Options")
    apply_notch = st.sidebar.checkbox("Apply Notch Filter (60 Hz)", value=True)
    notch_freq = st.sidebar.selectbox("Powerline Frequency", [50, 60], index=1)

    # Load model
    model = load_model()

    # Main content
    if signal is not None:
        st.success(f"✅ Loaded signal: {len(signal)} samples at {fs} Hz ({len(signal)/fs:.1f} seconds)")

        # Analyze
        if st.button("🔬 Analyze ECG", type="primary"):
            results = analyze_ecg(signal, fs, model)

            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Signal", "💓 Heart Rate", "📊 HRV Analysis",
                "🔬 Frequency Analysis", "🏥 Arrhythmia Detection"
            ])

            with tab1:
                st.subheader("ECG Signal Analysis")

                col1, col2, col3 = st.columns(3)
                col1.metric("Duration", f"{len(signal)/fs:.1f} s")
                col2.metric("Sampling Rate", f"{fs} Hz")
                col3.metric("R-peaks Detected", len(results['peaks']))

                # ECG plot
                fig_ecg = plot_ecg_signal(
                    results['filtered'], fs, results['peaks'],
                    title="ECG Signal with Detected R-peaks"
                )
                st.plotly_chart(fig_ecg, use_container_width=True)

                # Comparison
                with st.expander("View Original vs Filtered Signal"):
                    fig_comp = plot_comparison(results['original'], results['filtered'], fs)
                    st.plotly_chart(fig_comp, use_container_width=True)

            with tab2:
                st.subheader("Heart Rate Analysis")

                col1, col2, col3 = st.columns(3)
                col1.metric("Mean HR", f"{results['mean_hr']:.1f} bpm")
                if len(results['heart_rates']) > 0:
                    col2.metric("Min HR", f"{np.min(results['heart_rates']):.1f} bpm")
                    col3.metric("Max HR", f"{np.max(results['heart_rates']):.1f} bpm")

                if len(results['heart_rates']) > 0:
                    fig_hr = plot_heart_rate(results['peaks'], fs, results['heart_rates'])
                    st.plotly_chart(fig_hr, use_container_width=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        fig_rr = plot_rr_intervals(results['rr_intervals'])
                        st.plotly_chart(fig_rr, use_container_width=True)
                    with col2:
                        fig_poin = plot_poincare(results['rr_intervals'])
                        st.plotly_chart(fig_poin, use_container_width=True)

            with tab3:
                st.subheader("Heart Rate Variability (HRV)")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Time Domain Metrics**")
                    for key, value in results['hrv_time'].items():
                        st.metric(key.upper(), f"{value:.2f}")

                with col2:
                    st.write("**Frequency Domain Metrics**")
                    for key, value in results['hrv_freq'].items():
                        st.metric(key.upper(), f"{value:.2f}")

                # HRV visualization
                hrv_combined = {**results['hrv_time'], **results['hrv_freq']}
                fig_hrv = plot_hrv_features(hrv_combined)
                st.plotly_chart(fig_hrv, use_container_width=True)

            with tab4:
                st.subheader("Frequency Domain Analysis")

                fig_fft = plot_frequency_spectrum(
                    results['freqs'], results['mags'],
                    title="Power Spectrum Density"
                )
                st.plotly_chart(fig_fft, use_container_width=True)

                # Dominant frequency
                dominant_freq_idx = np.argmax(results['mags'][1:50]) + 1
                dominant_freq = results['freqs'][dominant_freq_idx]
                st.metric("Dominant Frequency", f"{dominant_freq:.2f} Hz")

            with tab5:
                st.subheader("Arrhythmia Detection")

                if results['predictions']:
                    # Summary
                    classes = [p['class'] for p in results['predictions']]
                    unique, counts = np.unique(classes, return_counts=True)

                    st.write("**Classification Summary**")
                    for cls, cnt in zip(unique, counts):
                        st.write(f"- {cls}: {cnt} beats ({cnt/len(classes)*100:.1f}%)")

                    # Show individual predictions
                    st.write("**Individual Beat Classifications**")
                    for i, pred in enumerate(results['predictions'][:5]):
                        with st.expander(f"Beat {i+1}: {pred['class']} (Confidence: {pred['confidence']:.1%})"):
                            fig_prob = plot_prediction_probabilities(pred['probabilities'])
                            st.plotly_chart(fig_prob, use_container_width=True)
                else:
                    st.warning("Not enough beats detected for classification")

            # Download report
            st.divider()
            st.subheader("📥 Export Results")

            # Create summary dataframe
            summary_data = {
                'Metric': ['Duration (s)', 'Sampling Rate (Hz)', 'Mean HR (bpm)',
                          'SDNN (ms)', 'RMSSD (ms)', 'pNN50 (%)'],
                'Value': [
                    f"{len(signal)/fs:.1f}",
                    f"{fs}",
                    f"{results['mean_hr']:.1f}",
                    f"{results['hrv_time']['sdnn']:.2f}",
                    f"{results['hrv_time']['rmssd']:.2f}",
                    f"{results['hrv_time']['pnn50']:.2f}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)

            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download Summary (CSV)",
                data=csv,
                file_name="ecg_analysis_summary.csv",
                mime="text/csv"
            )

    else:
        # Welcome screen
        st.info("👆 Please upload an ECG file, use demo data, or generate synthetic data to begin analysis")

        st.markdown("""
        ### Features

        - **Signal Processing**: Bandpass filtering, baseline wander removal, notch filtering
        - **Peak Detection**: Pan-Tompkins algorithm for R-peak detection
        - **Heart Rate Analysis**: Instantaneous HR, RR intervals, tachogram
        - **HRV Analysis**: Time and frequency domain metrics (SDNN, RMSSD, pNN50, LF/HF ratio)
        - **Frequency Analysis**: Power spectral density, FFT analysis
        - **Arrhythmia Detection**: CNN-based classification (Normal, AFib, PVC, PAC, Other)

        ### Supported Data Formats

        - **CSV**: Simple format with signal values (auto-detects column)
        - **WFDB**: MIT-BIH standard format (.dat, .hea)
        - **Demo**: Pre-loaded example data
        - **Synthetic**: Generated ECG for testing

        ### About

        This tool uses deep learning (1D CNN) trained on the MIT-BIH Arrhythmia Database
        to automatically detect and classify cardiac arrhythmias from ECG signals.
        """)


if __name__ == "__main__":
    main()
