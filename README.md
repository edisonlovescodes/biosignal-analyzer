# BioSignal Analyzer

A Python ECG exploration application with Streamlit/Plotly views, NumPy/SciPy filtering and HRV features, CNN/LSTM builders, and a separate FastAPI CSV interface.

**Signal exploration works without a classifier.** Classification requires a trained public-data artifact with the current label/preprocessing metadata and checksum. Random weights and synthetic-test models are never silently served.

## What changed

- Corrected the 60 Hz notch filter's frequency units and added an attenuation/preservation test.
- Unified complete 187-sample extraction: 93 samples before the peak, 94 starting at/after it.
- Corrected fusion-beat labels and persisted a fixed class order.
- Added subject-grouped train/validation/test splitting; records 201/202 stay together.
- Unified record filtering and per-beat normalization across training and classification.
- Fixed short/flat-signal peak detection and HRV PSD integration using frequency coordinates.
- Added model readiness, artifact checksums, CSV bounds and fail-closed prediction routes.
- Narrowed uploads to implemented CSV support and wired the UI filter controls.

The coarse classes are **Normal/Conduction, PAC, PVC, Fusion, Other**. This is a custom beat-label task, not an atrial-fibrillation rhythm detector. MIT-BIH F and f are ventricular/normal and paced/normal fusion annotations, respectively. See [PhysioNet's annotation guide](https://physionet.org/physiobank/database/html/mitdbdir/intro.htm).

## Run locally

Python 3.12, from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 and choose **Generate Synthetic**, or upload a CSV with a numeric `signal`, `ecg`, `value`, or `amplitude` column. Single-column CSVs are also supported; include a header. The UI defaults to 360 Hz. Synthetic signals demonstrate processing/visualization only. If the local MIT-BIH demo record is missing, the fallback is visibly labeled synthetic.

The Streamlit upload interface does not claim EDF or standalone WFDB `.dat` support. WFDB training reads complete local records with their headers/annotations separately.

## API

```sh
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Documentation: http://127.0.0.1:8000/docs.

- `/health`: signal analysis availability and a separate `model_loaded` flag.
- `/analyze`: bounded numeric CSV exploration (at least 64 samples, 4 MiB maximum); no fabricated classifier output if a model is absent.
- `/predict`: returns 503 without a compatible model. With a model, requires exactly 187 samples, `sampling_rate=360`, and `prefiltered=true`; it does not silently pad/truncate. For raw records use `/analyze`.
- `/model/info`: compatible artifact metadata; 503 when unavailable.

Classification uses the declared 360 Hz, lead-0, record-level 0.5–40 Hz bandpass/60 Hz notch contract and per-beat z-scoring. Other UI filter settings show signal features only. Resampling and lead equivalence are not implemented. `BIOSIGNAL_MODEL_PATH` configures the API artifact; Streamlit uses the default path below. `CORS_ORIGINS` configures separate-host origins.

## Train with local public data

Acquire [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/) separately and follow its published terms. No download or full real-data training is performed automatically:

```sh
python -m src.models.train_model --data /path/to/local/WFDB/records --epochs 10
```

The default output is `data/models/ecg_arrhythmia_classifier.keras` plus a JSON sidecar containing class order, filtering contract, trained/public status, model/data checksums, subject splits, beat counts, per-class test metrics and confusion matrix. At least five subjects are required. Classification only accepts compatible metadata and dimensions; old artifacts require retraining under the corrected contract. Synthetic model tests deliberately opt into untrained weights and cannot be mistaken for this public-data artifact.

No tracked trained public-data artifact or reproduced clinical accuracy is claimed. The earlier 95%, 109,117-beat, 48-record training and class-balancing assertions have been removed. Correct code alone does not establish classifier quality; a new public-data run and separate peak-detection evaluation remain necessary.

## Tests and containers

```sh
pip install -r requirements-dev.txt
python -m pytest tests -q
```

The suite checks scientific/runtime regressions and architecture/training wiring with synthetic data. CI runs the same suite. Docker uses Python 3.12 and a standard-library health probe:

```sh
docker build -t biosignal-analyzer .
docker run --rm -p 127.0.0.1:8501:8501 biosignal-analyzer
```

Container execution is a separate check from local Python tests. See [CLAIMS.md](CLAIMS.md) and [CHANGELOG.md](CHANGELOG.md).

## Interface and type

The interface uses self-hosted typefaces distributed by [Google Fonts](https://github.com/google/fonts). Original SIL Open Font License files are included in `static/fonts/`. No hosted font request is needed to render the page.
