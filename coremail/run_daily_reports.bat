@echo off
REM 每日邮件报告下载脚本
REM 下载当日的期货研报（概要版和详细版）

cd /d D:\QIS_DASHBOARD\coremail

set LOGFILE=D:\QIS_DASHBOARD\coremail\daily_reports.log

echo ======================================== >> "%LOGFILE%"
echo 开始下载: %date% %time% >> "%LOGFILE%"
echo ======================================== >> "%LOGFILE%"

echo [1/2] 下载概要版报告... >> "%LOGFILE%"
python download_general_reports.py >> "%LOGFILE%" 2>&1

echo [2/2] 下载详细版报告... >> "%LOGFILE%"
python download_detailed_reports.py >> "%LOGFILE%" 2>&1

echo 完成: %date% %time% >> "%LOGFILE%"
echo. >> "%LOGFILE%"
