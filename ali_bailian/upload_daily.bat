@echo off
REM 每日百炼RAG上传脚本
REM 自动上传当日下载的期货研报（仅概要版 general reports）到阿里百炼知识库

cd /d D:\QIS_DASHBOARD\ali_bailian

set LOGFILE=D:\QIS_DASHBOARD\logs\upload_daily.log

REM 创建日志目录（如果不存在）
if not exist "D:\QIS_DASHBOARD\logs" mkdir "D:\QIS_DASHBOARD\logs"

echo ======================================== >> "%LOGFILE%"
echo 开始上传: %date% %time% >> "%LOGFILE%"
echo ======================================== >> "%LOGFILE%"

REM 激活虚拟环境并执行上传
call D:\QIS_DASHBOARD\.venv\Scripts\activate.bat

echo [上传百炼] 开始上传今日 General Reports... >> "%LOGFILE%"
python upload_daily.py >> "%LOGFILE%" 2>&1

echo 完成: %date% %time% >> "%LOGFILE%"
echo. >> "%LOGFILE%"

REM 如果运行失败，记录错误
if errorlevel 1 (
    echo [错误] 上传失败，退出码: %errorlevel% >> "%LOGFILE%"
)
