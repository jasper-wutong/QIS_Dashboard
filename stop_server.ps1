#!/usr/bin/env pwsh
# QIS Dashboard Server Stop Script
# Kills Flask app process

Write-Host "Stopping QIS Dashboard Server..." -ForegroundColor Yellow

# Find and kill Python processes running app.py
$processes = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*app.py*"
}

if ($processes) {
    $processes | ForEach-Object {
        Write-Host "[INFO] Stopping process $($_.Id)..." -ForegroundColor Cyan
        Stop-Process -Id $_.Id -Force
    }
    Write-Host "[INFO] Server stopped" -ForegroundColor Green
} else {
    Write-Host "[WARN] No running Flask server found" -ForegroundColor Yellow
}
