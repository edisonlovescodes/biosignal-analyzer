"""A waveform-first ECG exploration workspace. No synthetic classifier output."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
import hashlib
import html

import numpy as np
import pandas as pd
import streamlit as st
import wfdb

if TYPE_CHECKING:
    from src.models.cnn_model import ArrhythmiaClassifier
from src.models.contract import extract_beat
from src.signal_processing.filters import preprocess_ecg
from src.signal_processing.peak_detection import (
    detect_r_peaks,
    calculate_heart_rate,
    calculate_rr_intervals,
)
from src.signal_processing.features import (
    calculate_hrv_time_domain,
    calculate_hrv_frequency_domain,
    compute_frequency_spectrum,
)
from src.visualization.plots import (
    plot_ecg_signal,
    plot_comparison,
    plot_heart_rate,
    plot_rr_intervals,
    plot_poincare,
    plot_frequency_spectrum,
    plot_prediction_probabilities,
)

ROOT = Path(__file__).resolve().parent
BLUE = "#244AD2"
SOURCE = "https://github.com/edisonlovescodes/biosignal-analyzer"

st.set_page_config(
    page_title="BioSignal — ECG exploration",
    page_icon="〰",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(f'<style>{(ROOT / "styles.css").read_text()}</style>', unsafe_allow_html=True)


@dataclass(frozen=True)
class Study:
    signal: np.ndarray
    fs: float
    name: str
    kind: Literal["local", "synthetic"]


@st.cache_resource
def load_model():
    path = ROOT / "data/models/ecg_arrhythmia_classifier.keras"
    if not path.exists():
        return None
    from src.models.cnn_model import ArrhythmiaClassifier  # TensorFlow is slow to import

    return ArrhythmiaClassifier.load_model(path)


def load_csv_ecg(file):
    """Finite numeric samples; optional time column provides the sampling rate."""
    if file.size > 4 * 1024 * 1024:
        raise ValueError("CSV must be smaller than 4 MiB.")
    frame = pd.read_csv(file)
    candidates = [c for c in ["signal", "ecg", "value", "amplitude"] if c in frame]
    if not candidates:
        candidates = [c for c in frame if c != "time"]
        if len(candidates) != 1:
            raise ValueError("Name the signal column signal, ecg, value, or amplitude.")
    values = pd.to_numeric(frame[candidates[0]], errors="raise").to_numpy(dtype=float)
    if len(values) < 64 or not np.all(np.isfinite(values)):
        raise ValueError("Provide at least 64 finite numeric samples.")
    fs = 360.0
    if "time" in frame:
        times = pd.to_numeric(frame["time"], errors="raise").to_numpy(dtype=float)
        differences = np.diff(times)
        if not np.all(np.isfinite(differences)) or np.any(differences <= 0):
            raise ValueError("Time values must be finite and strictly increasing, in seconds.")
        fs = 1 / np.mean(differences)
    if not 128 <= fs <= 2000:
        raise ValueError("This filtering workspace supports 128–2000 Hz.")
    return values, float(fs)


@st.cache_data
def generate_synthetic_ecg(duration: int):
    """Deterministic illustrative waveform, not patient data."""
    fs = 360
    values = np.zeros(int(duration * fs))
    for time in np.arange(0, duration, 0.8):
        i = int(time * fs)
        if 5 <= i < len(values) - 80:
            values[i - 5 : i] -= np.linspace(0, 0.1, 5)
            values[i : i + 15] += np.exp(-np.linspace(0, 3, 15)) * 1.5
            values[i + 15 : i + 25] -= np.linspace(0.2, 0, 10)
            values[i + 40 : i + 80] += np.exp(-np.linspace(0, 2, 40)) * 0.3
    return values + np.random.default_rng(42).normal(0, 0.05, len(values))


def analyze_ecg(study: Study, model: ArrhythmiaClassifier | None, notch_freq: int | None):
    signal, fs = study.signal, study.fs
    filtered = preprocess_ecg(signal, fs, notch_freq=notch_freq)
    peaks = detect_r_peaks(filtered, fs)
    mean_hr, heart_rates = calculate_heart_rate(peaks, fs)
    rr = calculate_rr_intervals(peaks, fs)
    frequencies, magnitudes = compute_frequency_spectrum(filtered, fs)
    predictions = []
    if model is not None and fs == 360 and notch_freq == 60:
        for peak in peaks[:10]:
            beat = extract_beat(filtered, peak)
            if beat is not None:
                label, confidence, probabilities = model.predict(beat)
                predictions.append(
                    {
                        "beat": int(peak),
                        "class": label,
                        "confidence": confidence,
                        "probabilities": probabilities,
                    }
                )
    return {
        "original": signal,
        "filtered": filtered,
        "peaks": peaks,
        "mean_hr": mean_hr,
        "heart_rates": heart_rates,
        "rr_intervals": rr,
        "hrv_time": calculate_hrv_time_domain(rr),
        "hrv_freq": calculate_hrv_frequency_domain(rr),
        "freqs": frequencies,
        "mags": magnitudes,
        "predictions": predictions,
    }


def style_plot(figure, height=340):
    figure.update_layout(
        font={"family": "IBM Plex Sans, sans-serif", "color": "#3B465C", "size": 12},
        title=None,
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin={"l": 12, "r": 16, "t": 16, "b": 26},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hoverlabel={"bgcolor": "#152B6B", "font_color": "#FFFFFF"},
    )
    figure.update_xaxes(
        gridcolor="#E5EAF6",
        zeroline=False,
        showline=True,
        linecolor="#D8E0F2",
        tickfont={"size": 10},
    )
    figure.update_yaxes(gridcolor="#E5EAF6", zeroline=False, tickfont={"size": 10})
    for trace in figure.data:
        if trace.type == "scatter":
            if trace.mode and "markers" in trace.mode and trace.name == "R-peaks":
                trace.marker.color = "#D36B3E"
            elif trace.name == "Original":
                trace.line.color = "#A3AEC7"
            else:
                trace.line.color = BLUE
    return figure


def display_plot(figure, height=340):
    st.plotly_chart(
        style_plot(figure, height),
        width="stretch",
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )


def main():
    st.markdown(
        f"""<div class="bio-masthead"><a class="bio-wordmark" href="/">bio<span>signal</span><svg viewBox="0 0 42 22" fill="none" aria-hidden="true"><path d="M1 12h10l4-8 7 17 6-15 4 6h9" /></svg></a><div class="bio-navigation"><span>SIGNAL LAB / 03</span><a href="{SOURCE}" target="_blank" rel="noreferrer">Source code ↗</a></div></div>""",
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-title"><span>LAB CONTROLS</span><h2>1. Choose a signal</h2></div>',
            unsafe_allow_html=True,
        )
        data_source = st.radio(
            "Signal source",
            ["Use Demo Data", "Generate Synthetic", "Upload File"],
            index=0,
        )
        study = None
        match data_source:
            case "Upload File":
                uploaded = st.file_uploader(
                    "ECG file",
                    type=["csv"],
                    help="CSV with numeric samples and a header. An optional time column should use seconds.",
                )
                if uploaded is not None:
                    try:
                        values, fs = load_csv_ecg(uploaded)
                        study = Study(values, fs, uploaded.name, "local")
                    except (ValueError, pd.errors.ParserError) as error:
                        st.error(f"Could not read the signal: {error}")
                else:
                    st.caption(
                        "Choose a CSV with a numeric signal column. No patient data is needed for a demonstration."
                    )
            case "Use Demo Data":
                path = ROOT / "data/mit-bih/100"
                if path.with_suffix(".hea").exists():
                    record = wfdb.rdrecord(str(path))
                    fs = float(record.fs)
                    study = Study(
                        record.p_signal[: int(30 * fs), 0],
                        fs,
                        "MIT-BIH · record 100",
                        "local",
                    )
                else:
                    study = Study(
                        generate_synthetic_ecg(10),
                        360,
                        "Synthetic fallback · 10 seconds",
                        "synthetic",
                    )
                    st.caption(
                        "Local record 100 is unavailable. Showing a clearly labeled synthetic waveform."
                    )
            case "Generate Synthetic":
                duration = st.slider("Duration (seconds)", 5, 30, 10)
                study = Study(
                    generate_synthetic_ecg(duration),
                    360,
                    f"Synthetic study · {duration} seconds",
                    "synthetic",
                )
                st.caption(
                    "An illustrative ECG-like waveform. Generated locally, with a fixed seed."
                )
            case _:
                raise AssertionError(f"Unknown signal source: {data_source}")
        st.divider()
        st.markdown('<span class="control-label">FILTERING</span>', unsafe_allow_html=True)
        apply_notch = st.checkbox("Apply notch filter", value=True)
        notch_freq = st.selectbox(
            "Powerline frequency (Hz)", [50, 60], index=1, disabled=not apply_notch
        )
        st.caption("Baseline removal + 0.5–40 Hz bandpass. Optional powerline rejection.")
        run = st.button("Run analysis", type="primary", width="stretch", disabled=study is None)
        st.markdown(
            '<div class="sidebar-foot">ECG EXPLORATION<br />Not for medical decisions.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        """<section class="bio-hero"><div><p>BIOSIGNAL · ECG (HEART SIGNAL) ANALYZER</p><h1>Heart rate from<br />a raw ECG<span class="bio-dot">.</span></h1></div><div class="bio-hero-note"><span>WHAT THIS DOES</span><p>Cleans a heart recording and finds every beat.</p><small>How to use: 1. In the left panel, choose demo data (a real public recording), a synthetic signal, or upload a CSV. 2. Click Run analysis. 3. Use the tabs to see heart rate, heart-rate variability and frequencies.</small></div></section>""",
        unsafe_allow_html=True,
    )
    if study is None:
        st.info("Load a CSV or choose a synthetic study in the lab controls to begin.")
        return
    signal, fs, name = study.signal, study.fs, study.name
    match study.kind:
        case "synthetic":
            synthetic = True
        case "local":
            synthetic = False
        case _:
            raise AssertionError(f"Unknown study kind: {study.kind}")
    notch = notch_freq if apply_notch else None
    signature = hashlib.sha256(signal.tobytes()).hexdigest() + str((fs, notch))
    saved = st.session_state.get("analysis")
    if saved is not None and saved["signature"] != signature:
        del st.session_state["analysis"]
    if run:
        with st.spinner("Processing the signal…"):
            st.session_state["analysis"] = {
                "signature": signature,
                "result": analyze_ecg(study, load_model(), notch),
            }
    saved = st.session_state.get("analysis")
    results = saved["result"] if saved is not None else None
    st.markdown(
        f'<div class="study-bar"><span><i></i> {html.escape(name)}</span><span>{"SYNTHETIC / ILLUSTRATIVE" if synthetic else "LOCAL FILE / EXPLORATORY"}</span></div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    labels = [
        ("Duration", f"{len(signal)/fs:.1f}", "SECONDS"),
        ("Sampling rate", f"{fs:.0f}", "HZ"),
        (
            "R-peaks" if results else "Samples",
            str(len(results["peaks"]) if results else len(signal)),
            "DETECTED" if results else "IN THE RECORD",
        ),
        (
            "Mean heart rate" if results else "Study mode",
            (f"{results['mean_hr']:.1f}" if results else ("Synthetic" if synthetic else "Local")),
            "BPM" if results else "INPUT SOURCE",
        ),
    ]
    for column, (label, value, unit) in zip(columns, labels):
        with column:
            st.markdown(
                f'<div class="study-stat"><span>{label}</span><strong>{value}</strong><small>{unit}</small></div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="trace-header"><span>01 / THE WAVEFORM</span><span>TIME DOMAIN · INPUT AMPLITUDE UNITS</span></div>',
        unsafe_allow_html=True,
    )
    if results is None:
        display_plot(plot_ecg_signal(signal, fs, title="", height=360), 360)
        st.markdown(
            '<div class="trace-caption"><span>RAW SIGNAL / PREVIEW</span><p>Choose your filtering settings, then select <b>Run analysis</b> to reveal R-peaks and derived features.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        tabs = st.tabs(["Waveform", "Heart rate", "HRV", "Spectrum", "Beat classification"])
        with tabs[0]:
            display_plot(
                plot_ecg_signal(results["filtered"], fs, results["peaks"], title=""),
                360,
            )
            st.caption(
                "Filtered signal with detected R-peaks. Peak positions are exploratory estimates, not clinical annotations."
            )
            with st.expander("Compare raw and filtered signals"):
                display_plot(plot_comparison(signal, results["filtered"], fs), 480)
        with tabs[1]:
            if len(results["peaks"]) > 1:
                display_plot(plot_heart_rate(results["peaks"], fs, results["heart_rates"]))
                cols = st.columns(2)
                with cols[0]:
                    display_plot(plot_rr_intervals(results["rr_intervals"]), 300)
                with cols[1]:
                    display_plot(plot_poincare(results["rr_intervals"]), 300)
            else:
                st.info("At least two R-peaks are needed to estimate heart rate.")
        with tabs[2]:
            st.markdown("### Between the beats")
            st.caption(
                "Short recordings provide exploratory HRV estimates. A zero/insufficient-data output does not establish physiologic normality."
            )
            metrics = {**results["hrv_time"], **results["hrv_freq"]}
            units = {
                "mean_rr": "ms",
                "sdnn": "ms",
                "rmssd": "ms",
                "pnn50": "%",
                "vlf_power": "ms²",
                "lf_power": "ms²",
                "hf_power": "ms²",
                "lf_hf_ratio": "ratio",
            }
            table = pd.DataFrame(
                [
                    {
                        "Metric": key.upper().replace("_", " "),
                        "Value": round(value, 3),
                        "Unit": units[key],
                    }
                    for key, value in metrics.items()
                ]
            )
            st.dataframe(table, hide_index=True, width="stretch")
        with tabs[3]:
            display_plot(plot_frequency_spectrum(results["freqs"], results["mags"]))
            st.caption(
                "Amplitude spectrum of the filtered signal. Explore the chart with zoom and hover."
            )
        with tabs[4]:
            if results["predictions"]:
                st.caption(
                    "Coarse beat classifications from the supplied compatible model. Biological and clinical performance require separate evaluation."
                )
                for prediction in results["predictions"]:
                    with st.expander(
                        f"Beat at sample {prediction['beat']} · {prediction['class']}"
                    ):
                        display_plot(
                            plot_prediction_probabilities(prediction["probabilities"]),
                            280,
                        )
            else:
                st.info(
                    "No compatible trained classifier is available for these settings. The signal-processing views remain fully usable."
                )
        summary = pd.DataFrame(
            [
                {"Metric": "duration_seconds", "Value": len(signal) / fs},
                {"Metric": "sampling_rate_hz", "Value": fs},
                {"Metric": "r_peaks", "Value": len(results["peaks"])},
                {"Metric": "mean_heart_rate_bpm", "Value": results["mean_hr"]},
                *[
                    {"Metric": key, "Value": value}
                    for key, value in {
                        **results["hrv_time"],
                        **results["hrv_freq"],
                    }.items()
                ],
            ]
        )
        st.download_button(
            "Download analysis · CSV",
            summary.to_csv(index=False),
            "biosignal-summary.csv",
            "text/csv",
        )
    st.markdown(
        """<div class="bio-bottom"><div><span>HOW IT WORKS</span><h2>Filter, find beats,<br />measure.</h2></div><p>An ECG is the heart&#39;s electrical signal. The app removes slow drift and power-line hum, finds each heartbeat peak (the R-peak), and measures the time between beats to get heart rate and heart-rate variability.<br /><br />The demo recording is MIT-BIH record 100 from PhysioNet (ODC-BY). Beat classification is not available yet: it needs a trained model. Not for medical decisions.</p></div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<footer class="bio-footer"><span>BIOSIGNAL / A SIGNAL EXPLORATION TOOL</span><span>BY EDISON ESSIEN</span></footer>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
