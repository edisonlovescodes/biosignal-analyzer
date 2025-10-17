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

# Custom CSS - Modern, professional design
st.markdown("""
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main container */
    .main {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        background-attachment: fixed;
    }

    .block-container {
        padding: 2rem 3rem;
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        margin: 2rem auto;
        max-width: 1400px;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        text-align: center;
    }

    .sub-header {
        font-size: 1.3rem;
        color: #64748b;
        margin-bottom: 3rem;
        text-align: center;
        font-weight: 400;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #dc2626 0%, #991b1b 100%);
        padding: 2rem 1rem;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.3rem 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateX(5px);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        color: white;
        border: none;
        padding: 0.75rem 2.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(220, 38, 38, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(220, 38, 38, 0.6);
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #dc2626;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
        border-bottom: 2px solid #e2e8f0;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        background: transparent;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
        color: #64748b;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        color: white !important;
    }

    /* Success message */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid #10b981;
        background: #ecfdf5;
        padding: 1rem;
    }

    /* Info box */
    .stInfo {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.1) 0%, rgba(153, 27, 27, 0.1) 100%);
        border-left: 4px solid #dc2626;
        border-radius: 12px;
        padding: 1.5rem;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #f8fafc;
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #dc2626;
        background: rgba(220, 38, 38, 0.05);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 10px;
        font-weight: 600;
        color: #334155;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
    }

    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #dc2626 0%, #991b1b 100%);
    }

    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
    }

    /* Subheader styling */
    h2, h3 {
        color: #1e293b;
        font-weight: 700;
        margin-top: 2rem;
    }

    /* Features list */
    ul {
        padding-left: 1.5rem;
    }

    li {
        margin: 0.5rem 0;
        color: #475569;
    }

    /* Code blocks */
    code {
        background: #f1f5f9;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        color: #dc2626;
        font-weight: 500;
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
    # Header with gradient and modern styling
    st.markdown('''
        <div style="text-align: center; margin-bottom: 3rem;">
            <h1 class="main-header">🫀 BioSignal Analyzer</h1>
            <p class="sub-header">Advanced ECG Analysis & AI-Powered Arrhythmia Detection</p>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1.5rem;">
                <span style="background: rgba(220, 38, 38, 0.1); padding: 0.5rem 1.5rem; border-radius: 20px; color: #dc2626; font-weight: 600;">
                    🔬 Signal Processing
                </span>
                <span style="background: rgba(220, 38, 38, 0.1); padding: 0.5rem 1.5rem; border-radius: 20px; color: #dc2626; font-weight: 600;">
                    🤖 Deep Learning
                </span>
                <span style="background: rgba(220, 38, 38, 0.1); padding: 0.5rem 1.5rem; border-radius: 20px; color: #dc2626; font-weight: 600;">
                    📊 Real-Time Analysis
                </span>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    # Sidebar with modern styling
    with st.sidebar:
        st.markdown('''
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="color: white; font-size: 1.8rem; margin-bottom: 0.5rem;">⚙️ Controls</h2>
                <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">Configure your analysis</p>
            </div>
        ''', unsafe_allow_html=True)

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
        # Success message with modern styling
        st.markdown(f'''
            <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
                        border-left: 4px solid #10b981;
                        border-radius: 12px;
                        padding: 1.2rem;
                        margin: 2rem 0;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <span style="font-size: 2rem;">✅</span>
                    <div>
                        <strong style="color: #059669; font-size: 1.1rem;">Signal Loaded Successfully</strong>
                        <p style="margin: 0.3rem 0 0 0; color: #047857;">
                            {len(signal):,} samples • {fs} Hz • {len(signal)/fs:.1f} seconds duration
                        </p>
                    </div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        # Analyze button with enhanced styling
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔬 Analyze ECG Signal", type="primary", use_container_width=True):
            results = analyze_ecg(signal, fs, model)

            # Display results in modern tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Signal Visualization",
                "💓 Heart Rate Metrics",
                "📊 HRV Analysis",
                "🔬 Frequency Domain",
                "🏥 AI Detection"
            ])

            with tab1:
                st.markdown("### 📈 ECG Signal Analysis")
                st.markdown("<br>", unsafe_allow_html=True)

                # Metric cards with better styling
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Duration", f"{len(signal)/fs:.1f} sec")
                with col2:
                    st.metric("📊 Sampling Rate", f"{fs:.0f} Hz")
                with col3:
                    st.metric("💓 R-peaks Found", f"{len(results['peaks'])}")

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
        # Modern welcome screen
        st.markdown('''
            <div style="background: linear-gradient(135deg, rgba(220, 38, 38, 0.08) 0%, rgba(153, 27, 27, 0.08) 100%);
                        border-radius: 16px;
                        padding: 3rem 2rem;
                        text-align: center;
                        margin: 2rem 0;">
                <h2 style="color: #dc2626; margin-bottom: 1rem; font-size: 2rem;">
                    👋 Welcome to BioSignal Analyzer
                </h2>
                <p style="color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;">
                    Upload an ECG file, use demo data, or generate synthetic signals to get started
                </p>
                <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
                    <span style="background: white; padding: 0.8rem 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                        📁 Upload Files
                    </span>
                    <span style="background: white; padding: 0.8rem 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                        📊 Use Demo Data
                    </span>
                    <span style="background: white; padding: 0.8rem 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                        ⚡ Generate Synthetic
                    </span>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        # Feature cards in columns
        st.markdown("### ✨ Key Features")
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('''
                <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); height: 100%;">
                    <div style="font-size: 2.5rem; margin-bottom: 1rem;">🔬</div>
                    <h4 style="color: #dc2626; margin-bottom: 0.8rem;">Signal Processing</h4>
                    <p style="color: #64748b; font-size: 0.9rem; line-height: 1.6;">
                        Advanced filtering techniques including bandpass, notch, and baseline wander removal
                    </p>
                </div>
            ''', unsafe_allow_html=True)

        with col2:
            st.markdown('''
                <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); height: 100%;">
                    <div style="font-size: 2.5rem; margin-bottom: 1rem;">💓</div>
                    <h4 style="color: #dc2626; margin-bottom: 0.8rem;">HRV Analysis</h4>
                    <p style="color: #64748b; font-size: 0.9rem; line-height: 1.6;">
                        Complete HRV metrics including SDNN, RMSSD, pNN50, and frequency domain analysis
                    </p>
                </div>
            ''', unsafe_allow_html=True)

        with col3:
            st.markdown('''
                <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); height: 100%;">
                    <div style="font-size: 2.5rem; margin-bottom: 1rem;">🤖</div>
                    <h4 style="color: #dc2626; margin-bottom: 0.8rem;">AI Detection</h4>
                    <p style="color: #64748b; font-size: 0.9rem; line-height: 1.6;">
                        Deep learning CNN trained on MIT-BIH database for arrhythmia classification
                    </p>
                </div>
            ''', unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        # Supported formats
        st.markdown("### 📁 Supported Formats")
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('''
                <div style="background: #f8fafc; padding: 1.2rem; border-radius: 10px; border-left: 4px solid #dc2626;">
                    <h5 style="color: #334155; margin-bottom: 0.8rem;">📄 File Formats</h5>
                    <ul style="margin: 0; padding-left: 1.2rem; color: #64748b;">
                        <li>CSV (auto-detects columns)</li>
                        <li>WFDB (.dat, .hea files)</li>
                        <li>EDF (European Data Format)</li>
                    </ul>
                </div>
            ''', unsafe_allow_html=True)

        with col2:
            st.markdown('''
                <div style="background: #f8fafc; padding: 1.2rem; border-radius: 10px; border-left: 4px solid #991b1b;">
                    <h5 style="color: #334155; margin-bottom: 0.8rem;">🎯 Analysis Types</h5>
                    <ul style="margin: 0; padding-left: 1.2rem; color: #64748b;">
                        <li>R-peak detection (Pan-Tompkins)</li>
                        <li>Heart rate variability</li>
                        <li>Arrhythmia classification</li>
                    </ul>
                </div>
            ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
