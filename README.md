# 🫀 BioSignal Analyzer

A comprehensive web application for ECG signal analysis and automatic arrhythmia detection using deep learning.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Advanced Signal Processing**
  - Bandpass filtering (0.5-40 Hz)
  - Baseline wander removal
  - Powerline interference removal (50/60 Hz notch filter)
  - Pan-Tompkins R-peak detection

- **Heart Rate Analysis**
  - Instantaneous heart rate calculation
  - RR interval analysis
  - Heart rate variability (HRV) metrics

- **HRV Analysis**
  - Time domain: SDNN, RMSSD, pNN50
  - Frequency domain: VLF, LF, HF power, LF/HF ratio
  - Poincaré plot visualization

- **Frequency Analysis**
  - Power spectral density (FFT)
  - Dominant frequency detection
  - Frequency spectrum visualization

- **Arrhythmia Detection**
  - Deep learning classification (1D CNN)
  - 5 classes: Normal, Atrial Fibrillation, PVC, PAC, Other
  - Trained on MIT-BIH Arrhythmia Database
  - Confidence scores and probability distributions

- **Interactive Visualizations**
  - Real-time signal plotting
  - Interactive Plotly charts
  - Multiple analysis views
  - Export functionality

## Demo

🔗 **[Live Demo](http://localhost:8501)** - Run locally following installation steps below

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/edisonlovescodes/biosignal-analyzer.git
cd biosignal-analyzer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

5. Open your browser to `http://localhost:8501`

### Docker Setup

```bash
# Build the image
docker build -t biosignal-analyzer .

# Run the container
docker run -p 8501:8501 biosignal-analyzer
```

## Usage

### 1. Upload ECG Data

The application supports multiple input formats:

- **CSV files**: Simple format with signal values
- **WFDB format**: MIT-BIH standard (.dat, .hea files)
- **Demo data**: Pre-loaded examples
- **Synthetic generation**: For testing

### 2. Analyze Signal

Click "Analyze ECG" to perform comprehensive analysis:

- Signal preprocessing and filtering
- R-peak detection
- Heart rate calculation
- HRV metrics
- Frequency analysis
- Arrhythmia classification

### 3. View Results

Results are organized in tabs:

- **Signal**: ECG waveform with R-peaks
- **Heart Rate**: HR trends and RR intervals
- **HRV Analysis**: Time and frequency metrics
- **Frequency Analysis**: Power spectrum
- **Arrhythmia Detection**: Classification results

### 4. Export Data

Download analysis results as CSV for further processing.

## Data Format

### CSV Format

```csv
time,signal
0.000,0.050
0.003,0.045
0.006,0.048
...
```

Or simplified (auto-detected):

```csv
signal
0.050
0.045
0.048
...
```

### Example Data

Sample ECG files are available in `data/samples/`:

- `sample_ecg.csv`: Basic ECG example
- More examples available after downloading MIT-BIH data

## Model Training

To train your own model on the MIT-BIH Arrhythmia Database:

1. Download the MIT-BIH database:
```bash
# Using wfdb-python
python -c "import wfdb; wfdb.dl_database('mitdb', 'data/mit-bih')"
```

2. Run the training script:
```bash
python train_with_available_data.py
```

3. The trained model will be saved to `data/models/ecg_arrhythmia_classifier.h5`

## API Usage

The application includes a FastAPI backend for programmatic access:

```bash
# Run the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

**POST /analyze**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@path/to/ecg.csv" \
  -F "sampling_rate=360"
```

**POST /predict**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/beat.csv"
```

**GET /health**
```bash
curl "http://localhost:8000/health"
```

API documentation available at: `http://localhost:8000/docs`

## Project Structure

```
biosignal-analyzer/
├── src/
│   ├── signal_processing/     # Signal processing modules
│   │   ├── filters.py          # Filtering functions
│   │   ├── peak_detection.py  # R-peak detection
│   │   └── features.py         # Feature extraction
│   ├── models/                 # ML models
│   │   ├── cnn_model.py        # CNN architecture
│   │   ├── lstm_model.py       # LSTM models
│   │   └── train_model.py      # Training pipeline
│   └── visualization/          # Plotting functions
│       └── plots.py
├── api/                        # FastAPI backend
│   └── main.py
├── data/
│   ├── samples/                # Example data
│   └── models/                 # Trained models
├── tests/                      # Unit tests
├── app.py                      # Main Streamlit app
├── requirements.txt
├── Dockerfile
└── README.md
```

## Technologies Used

- **Python 3.10+**: Core language
- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web application framework
- **NumPy/SciPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **FastAPI**: REST API
- **WFDB**: Physionet database tools
- **Docker**: Containerization

## Model Architecture

The arrhythmia classifier uses a 1D Convolutional Neural Network:

```
Input (187 samples, 1 channel)
    ↓
Conv1D (64 filters) + BatchNorm + MaxPooling
    ↓
Conv1D (128 filters) + BatchNorm + MaxPooling
    ↓
Conv1D (256 filters) + BatchNorm + GlobalAvgPooling
    ↓
Dense (128) + Dropout
    ↓
Dense (64) + Dropout
    ↓
Dense (5, softmax)
```

**Performance**: Trained on 109,117 labeled heartbeats from 48 MIT-BIH records with class-balanced learning to handle severe imbalance (82% Normal, 1.6% AFib)

## Dataset

The model is trained on the **MIT-BIH Arrhythmia Database**:

- 48 half-hour ECG recordings
- 47 subjects (25 men, 22 women)
- 360 Hz sampling rate
- Expert annotations for arrhythmia types

**Citation**:
```
Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).
(PMID: 11446209)
```

## Deployment

### Render

1. Fork this repository
2. Create a new Web Service on Render
3. Connect your GitHub repository
4. Render will automatically detect `render.yaml`
5. Deploy!

### Railway/Heroku

Update `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

## Testing

Run tests with pytest:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MIT-BIH Arrhythmia Database (PhysioNet)
- WFDB Software Package
- Streamlit community
- TensorFlow developers

## Contact

Edison - [GitHub](https://github.com/edisonlovescodes)

Project Link: [https://github.com/edisonlovescodes/biosignal-analyzer](https://github.com/edisonlovescodes/biosignal-analyzer)

---

Made with ❤️ for biomedical signal analysis
