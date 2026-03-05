# ============================================================
# QIS Dashboard - 自动下载商品期货报告定时任务设置脚本
# ============================================================
# 功能: 创建 Windows 定时任务,每日 17:00 自动下载 general reports
# 作者: QIS Dashboard Team
# 日期: 2026-02-13
# ============================================================

# 检查管理员权限
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host ""
    Write-Host "❌ 错误: 需要管理员权限运行此脚本" -ForegroundColor Red
    Write-Host ""
    Write-Host "请右键 PowerShell 图标，选择 '以管理员身份运行'，然后再执行此脚本" -ForegroundColor Yellow
    Write-Host ""
    Pause
    exit 1
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  QIS Dashboard - 设置下载报告定时任务" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# 配置参数
$TaskName = "QIS_Download_Daily_Reports"
$TaskDescription = "每日 17:00 自动下载商品期货 general reports"
$ScriptPath = "D:\QIS_DASHBOARD\coremail\run_daily_reports.bat"
$TriggerTime = "17:00"
$LogFile = "D:\QIS_DASHBOARD\coremail\daily_reports.log"

# 检查批处理文件是否存在
if (-not (Test-Path $ScriptPath)) {
    Write-Host "❌ 错误: 找不到批处理文件 $ScriptPath" -ForegroundColor Red
    Write-Host ""
    Pause
    exit 1
}

Write-Host "配置信息:" -ForegroundColor Yellow
Write-Host "  任务名称: $TaskName" -ForegroundColor White
Write-Host "  执行时间: 每天 $TriggerTime" -ForegroundColor White
Write-Host "  执行脚本: $ScriptPath" -ForegroundColor White
Write-Host "  日志文件: $LogFile" -ForegroundColor White
Write-Host ""

# 检查任务是否已存在
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "⚠ 警告: 定时任务 '$TaskName' 已存在" -ForegroundColor Yellow
    $overwrite = Read-Host "是否删除现有任务并重新创建? [y/N]"
    
    if ($overwrite -eq 'y' -or $overwrite -eq 'Y') {
        Write-Host ""
        Write-Host "正在删除现有任务..." -ForegroundColor Cyan
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "✓ 已删除现有任务" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "取消操作" -ForegroundColor Yellow
        Write-Host ""
        Pause
        exit 0
    }
}

Write-Host ""
Write-Host "开始创建定时任务..." -ForegroundColor Cyan

try {
    # 创建触发器 (每天 17:00)
    $trigger = New-ScheduledTaskTrigger -Daily -At $TriggerTime

    # 创建操作 (执行批处理文件)
    $action = New-ScheduledTaskAction -Execute $ScriptPath

    # 创建任务设置
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RunOnlyIfNetworkAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Hours 2)

    # 注册定时任务 (当前用户)
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Description $TaskDescription `
        -Trigger $trigger `
        -Action $action `
        -Settings $settings `
        -User $env:USERNAME `
        -RunLevel Highest | Out-Null

    Write-Host ""
    Write-Host "✓ 定时任务创建成功!" -ForegroundColor Green
    Write-Host ""
    Write-Host "任务详情:" -ForegroundColor Yellow
    Write-Host "  - 任务名称: $TaskName" -ForegroundColor White
    Write-Host "  - 执行时间: 每天 $TriggerTime" -ForegroundColor White
    Write-Host "  - 执行用户: $env:USERNAME" -ForegroundColor White
    Write-Host "  - 日志文件: $LogFile" -ForegroundColor White
    Write-Host ""
    Write-Host "下一步操作:" -ForegroundColor Yellow
    Write-Host "  1. 打开 '任务计划程序' 确认任务已创建" -ForegroundColor White
    Write-Host "  2. 右键任务可'立即运行'测试" -ForegroundColor White
    Write-Host "  3. 查看日志: Get-Content D:\QIS_DASHBOARD\coremail\daily_reports.log -Tail 50" -ForegroundColor White
    Write-Host ""

    # 询问是否立即测试
    $testNow = Read-Host "是否立即运行一次测试? [y/N]"
    if ($testNow -eq 'y' -or $testNow -eq 'Y') {
        Write-Host ""
        Write-Host "开始测试运行..." -ForegroundColor Cyan
        Start-ScheduledTask -TaskName $TaskName
        Start-Sleep -Seconds 2
        Write-Host "✓ 任务已触发,请查看日志文件确认结果" -ForegroundColor Green
    }

} catch {
    Write-Host ""
    Write-Host "❌ 创建任务失败: $_" -ForegroundColor Red
    Write-Host ""
}

Write-Host ""
Write-Host "使用说明:" -ForegroundColor Cyan
Write-Host "  - 查看任务: Get-ScheduledTask -TaskName '$TaskName'" -ForegroundColor White
Write-Host "  - 手动运行: Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor White
Write-Host "  - 删除任务: Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false" -ForegroundColor White
Write-Host "  - 查看日志: Get-Content D:\QIS_DASHBOARD\coremail\daily_reports.log -Tail 50" -ForegroundColor White
Write-Host ""
