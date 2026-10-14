# Real-time Sign Language Detection - Run script
Set-Location $PSScriptRoot

# Activate venv if present
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    & ".\venv\Scripts\Activate.ps1"
}

# Run real-time detection
python realtime.py