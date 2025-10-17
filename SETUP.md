# Setup Guide

This guide will help you get the BioSignal Analyzer up and running on your local machine or deploy it to the cloud.

## Table of Contents

- [Local Development Setup](#local-development-setup)
- [Training Your Own Model](#training-your-own-model)
- [Running Tests](#running-tests)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## Local Development Setup

### 1. Prerequisites

Make sure you have the following installed:

- Python 3.10 or higher
- pip (Python package manager)
- Git
- (Optional) Docker for containerized deployment

### 2. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/biosignal-analyzer.git
cd biosignal-analyzer

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Application

```bash
# Start Streamlit app
streamlit run app.py

# Or run the API server
uvicorn api.main:app --reload --port 8000
```

The application will be available at:
- Streamlit UI: http://localhost:8501
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Training Your Own Model

### 1. Download MIT-BIH Dataset

```python
import wfdb

# Download MIT-BIH Arrhythmia Database
wfdb.dl_database('mitdb', 'data/mit-bih-arrhythmia-database-1.0.0')
```

Or download manually from PhysioNet:
https://physionet.org/content/mitdb/1.0.0/

### 2. Train the Model

```bash
python src/models/train_model.py
```

Training parameters:
- Epochs: 50 (with early stopping)
- Batch size: 32
- Optimizer: Adam
- Learning rate: 0.001 (with decay)

The trained model will be saved to `data/models/ecg_arrhythmia_classifier.h5`

### 3. Evaluate Model

```python
from src.models.cnn_model import ArrhythmiaClassifier

# Load trained model
classifier = ArrhythmiaClassifier('data/models/ecg_arrhythmia_classifier.h5')

# Test on sample
import numpy as np
sample = np.random.randn(187)  # Your ECG segment
pred_class, confidence, probs = classifier.predict(sample)

print(f"Prediction: {pred_class} ({confidence:.2%} confidence)")
```

## Running Tests

### Unit Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_signal_processing.py -v

# Run specific test
pytest tests/test_models.py::TestArrhythmiaClassifier::test_prediction -v
```

### Manual Testing

```bash
# Test signal processing
python -c "
from src.signal_processing.filters import preprocess_ecg
import numpy as np

signal = np.random.randn(3600)
filtered = preprocess_ecg(signal, fs=360)
print(f'Filtered signal shape: {filtered.shape}')
"

# Test model prediction
python -c "
from src.models.cnn_model import ArrhythmiaClassifier
import numpy as np

model = ArrhythmiaClassifier()
segment = np.random.randn(187)
pred, conf, probs = model.predict(segment)
print(f'Prediction: {pred}, Confidence: {conf:.2%}')
"
```

## Deployment

### Deploy to Render

1. **Fork the repository** to your GitHub account

2. **Create a new Web Service** on [Render](https://render.com)

3. **Connect your repository**
   - Select "biosignal-analyzer" repository
   - Render will auto-detect `render.yaml`

4. **Configure environment variables** (if needed)
   ```
   PYTHON_VERSION=3.10
   ```

5. **Deploy!**
   - Render will build the Docker image
   - Application will be live in a few minutes

### Deploy to Railway

1. **Install Railway CLI** (optional)
   ```bash
   npm i -g @railway/cli
   ```

2. **Create Procfile**
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

3. **Deploy**
   ```bash
   railway login
   railway init
   railway up
   ```

### Deploy with Docker

```bash
# Build image
docker build -t biosignal-analyzer .

# Run locally
docker run -p 8501:8501 biosignal-analyzer

# Push to registry
docker tag biosignal-analyzer yourusername/biosignal-analyzer
docker push yourusername/biosignal-analyzer

# Deploy to any Docker host
docker pull yourusername/biosignal-analyzer
docker run -d -p 80:8501 yourusername/biosignal-analyzer
```

### Deploy to Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Go to** [share.streamlit.io](https://share.streamlit.io)

3. **Deploy your app**
   - Connect GitHub repository
   - Select branch: `main`
   - Main file: `app.py`
   - Click "Deploy"

## Troubleshooting

### Common Issues

#### 1. TensorFlow Installation Issues

**Problem**: TensorFlow fails to install

**Solution**:
```bash
# On macOS with Apple Silicon
pip install tensorflow-macos
pip install tensorflow-metal

# On Windows/Linux
pip install tensorflow==2.15.0
```

#### 2. Missing Model File

**Problem**: "Model not found" error

**Solution**:
- The app works without a trained model (uses untrained for demo)
- To use real predictions, train a model first:
  ```bash
  python src/models/train_model.py
  ```

#### 3. Port Already in Use

**Problem**: "Address already in use" error

**Solution**:
```bash
# Use different port
streamlit run app.py --server.port=8502

# Or kill existing process
lsof -ti:8501 | xargs kill
```

#### 4. Import Errors

**Problem**: "Module not found" errors

**Solution**:
```bash
# Make sure you're in the project root
cd biosignal-analyzer

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### 5. Memory Issues

**Problem**: Out of memory during training

**Solution**:
- Reduce batch size in `train_model.py`
- Use fewer training samples
- Close other applications

### Performance Optimization

#### Speed up inference:

```python
# Use batch prediction for multiple samples
classifier = ArrhythmiaClassifier(model_path)
predictions = classifier.predict_batch(segments)
```

#### Reduce memory usage:

```python
# Clear TensorFlow session
from tensorflow.keras import backend as K
K.clear_session()
```

## Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [MIT-BIH Database Info](https://physionet.org/content/mitdb/1.0.0/)
- [ECG Signal Processing](https://en.wikipedia.org/wiki/Electrocardiography)

## Getting Help

- Open an issue on GitHub
- Check existing issues and discussions
- Review the API documentation at `/docs`

## Next Steps

- Customize the model architecture
- Add more arrhythmia classes
- Integrate with medical devices
- Build mobile application
- Add real-time monitoring
