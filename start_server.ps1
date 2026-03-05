#!/usr/bin/env pwsh
# QIS Dashboard Server Startup Script
# Sets proxy and starts Flask app

Write-Host "Starting QIS Dashboard Server..." -ForegroundColor Green

# Set proxy environment variables
$env:HTTP_PROXY = "http://127.0.0.1:7890"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
$env:NO_PROXY = "localhost,127.0.0.1"

Write-Host "[INFO] Proxy configured: $env:HTTP_PROXY" -ForegroundColor Cyan

# Use Python 3.7 directly — required for Wind / WindPy DLL compatibility
$python37 = "C:\Users\wutong6\AppData\Local\Programs\Python\Python37\python.exe"
if (Test-Path $python37) {
    Write-Host "[INFO] Using Python 3.7 (Wind compatible): $python37" -ForegroundColor Cyan
    $pythonExe = $python37
} else {
    Write-Host "[WARN] Python 3.7 not found, falling back to venv Python (Wind may not work)" -ForegroundColor Yellow
    & "$PSScriptRoot\.venv\Scripts\Activate.ps1"
    $pythonExe = "python"
}

# Start Flask server
Write-Host "[INFO] Starting Flask server..." -ForegroundColor Cyan
& $pythonExe app.py
