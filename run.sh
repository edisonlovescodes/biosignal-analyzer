#!/bin/bash

# BioSignal Analyzer - Quick Start Script

echo "🫀 BioSignal Analyzer - Starting..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/installed" ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    touch venv/installed
fi

# Check which mode to run
if [ "$1" == "api" ]; then
    echo "Starting FastAPI server..."
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
elif [ "$1" == "test" ]; then
    echo "Running tests..."
    pytest tests/ -v
elif [ "$1" == "train" ]; then
    echo "Training model..."
    python src/models/train_model.py
else
    echo "Starting Streamlit app..."
    streamlit run app.py
fi
