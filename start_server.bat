@echo off
REM QIS Dashboard Server Startup Script
REM Sets proxy and starts Flask app

echo Starting QIS Dashboard Server...

REM Set proxy environment variables
set HTTP_PROXY=http://127.0.0.1:7890
set HTTPS_PROXY=http://127.0.0.1:7890
set NO_PROXY=localhost,127.0.0.1

echo [INFO] Proxy configured: %HTTP_PROXY%

REM Activate virtual environment and run app
call .venv\Scripts\activate.bat
echo [INFO] Virtual environment activated

echo [INFO] Starting Flask server...
python app.py

pause
