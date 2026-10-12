@echo off
title ASR Tool - Automatic Speech Recognition Studio
color 0b
echo ==============================================================
echo       AUTOMATIC SPEECH RECOGNITION (ASR) STUDIO
echo       Powered by OpenAI Whisper + Python Flask
echo ==============================================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [*] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Python not found in PATH. Please install Python 3.10+.
        pause
        exit /b 1
    )
)

echo [*] Activating virtual environment...
call venv\Scripts\activate.bat

echo [*] Checking and installing dependencies...
python -m pip install -r requirements.txt --quiet

echo [*] Opening browser to http://127.0.0.1:5000 ...
start "" "http://127.0.0.1:5000"

echo [*] Starting Flask ASR Server (Press Ctrl+C to stop)...
python app.py --model base --port 5000
pause
