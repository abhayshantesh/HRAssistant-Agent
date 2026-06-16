#!/usr/bin/env bash
# ============================================================
#  HRAssistant-Agent - one-click launcher (macOS / Linux)
#  Creates a venv, installs deps, then runs the Streamlit app.
# ============================================================
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "  HRAssistant-Agent"
echo "============================================================"

# --- Python check ---
if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] Python 3 not found. Install Python 3.10+ from https://www.python.org/"
    exit 1
fi

# --- Virtual environment ---
if [ ! -d venv ]; then
    echo "[setup] Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# --- Dependencies (install only if streamlit is missing) ---
if ! python -c "import streamlit" >/dev/null 2>&1; then
    echo "[setup] Installing dependencies (first run, this may take a few minutes)..."
    python -m pip install --upgrade pip >/dev/null
    pip install -r requirements.txt
fi

# --- API key check ---
if [ ! -f .env ] && [ -f .env.example ]; then
    echo "[setup] No .env found - creating one from .env.example"
    cp .env.example .env
    echo "[ACTION REQUIRED] Edit .env and set OPENROUTER_API_KEY, then re-run this script."
    echo "Get a free key at https://openrouter.ai/keys"
    exit 1
fi

echo ""
echo "[run] Starting Streamlit at http://localhost:8501  (Ctrl+C to stop)"
echo ""
streamlit run app.py
