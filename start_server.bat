@echo off
REM QIS Dashboard Server Startup Script
REM Sets proxy and starts Flask app

echo Starting QIS Dashboard Server...

REM Set proxy environment variables
set HTTP_PROXY=http://127.0.0.1:7890
set HTTPS_PROXY=http://127.0.0.1:7890
set NO_PROXY=localhost,127.0.0.1

echo [INFO] Proxy configured: %HTTP_PROXY%

REM Use Python 3.7 directly (required for Wind / WindPy DLL compatibility)
set PYTHON37=C:\Users\wutong6\AppData\Local\Programs\Python\Python37\python.exe

if not exist "%PYTHON37%" (
    echo [WARN] Python 3.7 not found at %PYTHON37%, falling back to venv Python
    call .venv\Scripts\activate.bat
    set PYTHON37=python
)

echo [INFO] Starting Flask server...
"%PYTHON37%" app.py

pause
