#!/usr/bin/env pwsh
# QIS Dashboard Server Startup Script
# Sets proxy and starts Flask app

Write-Host "Starting QIS Dashboard Server..." -ForegroundColor Green

# Set proxy environment variables
$env:HTTP_PROXY = "http://127.0.0.1:7890"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
$env:NO_PROXY = "localhost,127.0.0.1"

Write-Host "[INFO] Proxy configured: $env:HTTP_PROXY" -ForegroundColor Cyan

# Activate virtual environment
& "$PSScriptRoot\.venv\Scripts\Activate.ps1"
Write-Host "[INFO] Virtual environment activated" -ForegroundColor Cyan

# Start Flask server
Write-Host "[INFO] Starting Flask server..." -ForegroundColor Cyan
& python app.py
