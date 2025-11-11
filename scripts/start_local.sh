#!/usr/bin/env bash
set -euo pipefail

# Start backend (FastAPI)
( cd "$(dirname "$0")/../backend" && uvicorn main:app --host 127.0.0.1 --port 8000 ) &

# Give backend a moment
sleep 2

# Start frontend (Streamlit)
cd "$(dirname "$0")/../frontend"
streamlit run app_frontend.py
