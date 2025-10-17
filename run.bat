@echo off
REM BioSignal Analyzer - Quick Start Script (Windows)

echo 🫀 BioSignal Analyzer - Starting...

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies if needed
if not exist "venv\installed" (
    echo Installing dependencies...
    pip install --upgrade pip
    pip install -r requirements.txt
    echo. > venv\installed
)

REM Check which mode to run
if "%1"=="api" (
    echo Starting FastAPI server...
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
) else if "%1"=="test" (
    echo Running tests...
    pytest tests/ -v
) else if "%1"=="train" (
    echo Training model...
    python src\models\train_model.py
) else (
    echo Starting Streamlit app...
    streamlit run app.py
)
