# ASR Tool - PowerShell Launch Script
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "      AUTOMATIC SPEECH RECOGNITION (ASR) STUDIO" -ForegroundColor White
Write-Host "      Powered by OpenAI Whisper + Python Flask" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $PSScriptRoot

if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "[*] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

Write-Host "[*] Activating virtual environment..." -ForegroundColor Green
& ".\venv\Scripts\Activate.ps1"

Write-Host "[*] Installing/verifying requirements..." -ForegroundColor Yellow
.\venv\Scripts\python.exe -m pip install -r requirements.txt --quiet

Write-Host "[*] Opening http://127.0.0.1:5000 in your browser..." -ForegroundColor Cyan
Start-Process "http://127.0.0.1:5000"

Write-Host "[*] Starting ASR Web Server on http://127.0.0.1:5000 (Ctrl+C to quit)..." -ForegroundColor Green
.\venv\Scripts\python.exe app.py --model base --port 5000
