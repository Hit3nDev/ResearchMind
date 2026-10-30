@echo off
REM LitLens one-click launcher: sets up Python deps on first run, then opens the app.
cd /d "%~dp0"
if not exist .venv (
    echo [LitLens] First run: creating virtual environment...
    python -m venv .venv || (echo Python 3.10+ is required: https://www.python.org/downloads/ & pause & exit /b 1)
)
echo [LitLens] Checking dependencies...
.venv\Scripts\python -m pip install -q --disable-pip-version-check -r backend\requirements.txt
echo [LitLens] Starting on http://localhost:8000  (close this window to stop)
start "" cmd /c "timeout /t 8 >nul & start http://localhost:8000"
cd backend
..\.venv\Scripts\python -m uvicorn app.main:app --port 8000
pause
