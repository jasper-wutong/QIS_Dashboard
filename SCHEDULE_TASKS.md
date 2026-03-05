# QIS Dashboard 每日自动化任务配置

## ⏰ 定时任务时间表

### 1. 邮件报告下载
- **时间**: 每天 **17:00**
- **脚本**: `D:\QIS_DASHBOARD\coremail\run_daily_reports.bat`
- **功能**: 从邮箱下载当日期货研报（general + detailed）
- **耗时**: 约 10-15 分钟

### 2. 百炼知识库上传
- **时间**: 每天 **18:00**
- **脚本**: `D:\QIS_DASHBOARD\ali_bailian\upload_daily.bat`
- **功能**: 上传当日 general reports 到阿里百炼RAG
- **耗时**: 约 15-30 分钟（取决于文件数量）

---

## 🚀 快速设置（二选一）

### 方案A：一键自动设置（推荐）

#### 1. 设置下载任务
打开任务计划程序创建下载任务，或运行：
```powershell
$action = New-ScheduledTaskAction -Execute "D:\QIS_DASHBOARD\coremail\run_daily_reports.bat" -WorkingDirectory "D:\QIS_DASHBOARD\coremail"
$trigger = New-ScheduledTaskTrigger -Daily -At "17:00"
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
Register-ScheduledTask -TaskName "DailyReportDownload" -Description "每日下载期货研报" -Action $action -Trigger $trigger -Principal $principal -Force
```

#### 2. 设置上传任务（以管理员身份运行）
```powershell
cd D:\QIS_DASHBOARD\ali_bailian
.\setup_scheduled_task.ps1
```

### 方案B：手动通过任务计划程序设置

1. 打开"任务计划程序" (Win+R → `taskschd.msc`)
2. 创建两个基本任务：
   - **任务1**: 报告下载，每天17:00执行 `run_daily_reports.bat`
   - **任务2**: 百炼上传，每天18:00执行 `upload_daily.bat`

---

## 📊 流程图

```
17:00 ─┐
       │ 开始下载报告
       │ ├─ 下载 General Reports
       │ └─ 下载 Detailed Reports
       │
17:15 ─┤ 下载完成
       │
18:00 ─┐
       │ 开始上传百炼
       │ └─ 仅上传 General Reports
       │
18:30 ─┘ 上传完成
```

---

## 📝 日志位置

- **下载日志**: `D:\QIS_DASHBOARD\coremail\daily_reports.log`
- **上传日志**: `D:\QIS_DASHBOARD\logs\upload_daily.log`

### 查看日志命令
```powershell
# 查看下载日志
Get-Content D:\QIS_DASHBOARD\coremail\daily_reports.log -Tail 50

# 查看上传日志
Get-Content D:\QIS_DASHBOARD\logs\upload_daily.log -Tail 50
```

---

## ✅ 验证任务是否设置成功

```powershell
# 查看所有QIS相关定时任务
Get-ScheduledTask | Where-Object {$_.TaskName -like "*Report*" -or $_.TaskName -like "*Bailian*"}

# 手动测试下载
cd D:\QIS_DASHBOARD\coremail
.\run_daily_reports.bat

# 手动测试上传
cd D:\QIS_DASHBOARD\ali_bailian
.\upload_daily.bat
```

---

## ⚠️ 注意事项

1. **网络要求**:
   - 下载任务需要访问邮箱服务器
   - 上传任务需要访问阿里云API

2. **时间选择原因**:
   - 17:00 开始下载，避开交易时段
   - 18:00 上传，确保下载完成且有充足时间

3. **权限要求**:
   - 任务需要在用户登录时执行
   - 上传任务建议设置为"最高权限"

4. **环境变量**:
   - 确保 `.env` 文件配置完整
   - 虚拟环境路径正确 (`D:\QIS_DASHBOARD\.venv`)

---

## 🔧 故障排查

### 任务未执行
- 确认电脑在设定时间处于开机状态
- 检查任务计划程序中任务状态是否"就绪"
- 查看任务属性→条件→取消勾选"只有在使用交流电源时才启动此任务"

### 下载失败
- 检查邮箱配置 (`coremail/param.json`)
- 查看下载日志确认错误信息
- 测试网络连接

### 上传失败
- 检查 `.env` 配置的API密钥
- 查看上传日志确认错误
- 确认知识库ID正确

---

## 📚 相关文档

- 下载详细说明: `D:\QIS_DASHBOARD\coremail\README.md`
- 上传详细说明: `D:\QIS_DASHBOARD\ali_bailian\README_DAILY_UPLOAD.md`
- Dashboard使用: `D:\QIS_DASHBOARD\README.md`
