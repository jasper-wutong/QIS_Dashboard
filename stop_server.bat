@echo off
REM QIS Dashboard Server Stop Script
REM Kills Flask app process

echo Stopping QIS Dashboard Server...

REM Kill Python processes running app.py
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO TABLE ^| findstr python') do (
    wmic process where "ProcessId=%%i AND CommandLine like '%%app.py%%'" delete 2>nul
)

echo [INFO] Server stopped

pause
