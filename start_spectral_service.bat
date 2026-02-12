@echo off
REM Quick Start Script for Spectral Service
REM This script starts the FastAPI service on port 8001

echo ========================================
echo  Starting Spectral Service
echo ========================================
echo.

cd "Spectral Service"

echo Checking Python installation...
python --version
echo.

echo Starting Spectral Service on port 8001...
echo Press CTRL+C to stop the service
echo.
echo Service URL: http://localhost:8001
echo Health check: http://localhost:8001/
echo API endpoint: http://localhost:8001/analyze_spectrum
echo.

uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
