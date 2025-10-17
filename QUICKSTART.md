# Quick Start Guide

Get the BioSignal Analyzer running in 5 minutes!

## Option 1: Fastest Way (Using Scripts)

### macOS/Linux
```bash
./run.sh
```

### Windows
```bash
run.bat
```

That's it! The app will open at http://localhost:8501

## Option 2: Manual Setup

### Step 1: Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Install packages
pip install -r requirements.txt
```

### Step 2: Run the App
```bash
streamlit run app.py
```

## Option 3: Docker

```bash
docker build -t biosignal-analyzer .
docker run -p 8501:8501 biosignal-analyzer
```

Visit: http://localhost:8501

## Using the App

### 1. Upload Data
- Click "Upload File" in sidebar
- Choose CSV, DAT, or EDF file
- Or use "Demo Data" to try it out

### 2. Analyze
- Click "Analyze ECG" button
- View results in tabs:
  - Signal visualization
  - Heart rate analysis
  - HRV metrics
  - Frequency analysis
  - Arrhythmia detection

### 3. Export
- Download analysis summary as CSV
- Use results in your research

## Running Tests

```bash
pytest
```

## Running the API

```bash
# macOS/Linux
./run.sh api

# Windows
run.bat api

# Or manually
uvicorn api.main:app --reload
```

API will be at: http://localhost:8000/docs

## Sample API Request

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@data/samples/sample_ecg.csv" \
  -F "sampling_rate=360"
```

## Training Your Own Model

```bash
# macOS/Linux
./run.sh train

# Windows
run.bat train

# Or manually
python src/models/train_model.py
```

Note: You need MIT-BIH dataset first. See SETUP.md for details.

## Troubleshooting

**App won't start?**
- Make sure Python 3.10+ is installed
- Try: `pip install --upgrade pip setuptools wheel`
- Check port 8501 is not in use

**Import errors?**
- Activate virtual environment first
- Reinstall: `pip install -r requirements.txt --force-reinstall`

**Model errors?**
- The app works without a trained model (demo mode)
- For real predictions, train a model first

## What's Next?

- Read [README.md](README.md) for full documentation
- Check [SETUP.md](SETUP.md) for deployment
- Explore the code in `src/`
- Customize the model architecture
- Add your own features

## Key Features to Try

1. **Upload your own ECG data** - CSV format with signal values
2. **Interactive visualization** - Zoom, pan, explore the signal
3. **Real-time analysis** - Get instant HRV and HR metrics
4. **AI predictions** - Deep learning arrhythmia classification
5. **Export results** - Download analysis for reports

Enjoy analyzing biosignals! 🫀
