@echo off
REM ============================================================
REM  HRAssistant-Agent - one-click launcher (Windows)
REM  Creates a venv, installs deps, then runs the Streamlit app.
REM ============================================================
setlocal

cd /d "%~dp0"

echo ============================================================
echo   HRAssistant-Agent
echo ============================================================

REM --- Python check ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

REM --- Virtual environment ---
if not exist venv (
    echo [setup] Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat

REM --- Dependencies (install only if streamlit is missing) ---
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [setup] Installing dependencies ^(first run, this may take a few minutes^)...
    python -m pip install --upgrade pip >nul
    pip install -r requirements.txt
)

REM --- API key check ---
if not exist .env (
    if exist .env.example (
        echo [setup] No .env found - creating one from .env.example
        copy .env.example .env >nul
        echo [ACTION REQUIRED] Open .env and set OPENROUTER_API_KEY, then re-run this script.
        echo Get a free key at https://openrouter.ai/keys
        pause
        exit /b 1
    )
)

echo.
echo [run] Starting Streamlit at http://localhost:8501  (Ctrl+C to stop)
echo.
streamlit run app.py

endlocal
