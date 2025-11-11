@echo off
start cmd /k "cd /d %~dp0\..\backend && uvicorn main:app --host 127.0.0.1 --port 8000"
timeout /t 2 > nul
start cmd /k "cd /d %~dp0\..\frontend && streamlit run app_frontend.py"
