# 阿里百炼每日上传说明

## 功能说明

每日自动上传当天下载的期货研报（仅 General Reports）到阿里百炼RAG知识库。

## 文件说明

- **upload_daily.py** - Python上传脚本（仅上传 general reports，不上传 detailed reports）
- **upload_daily.bat** - Windows批处理文件，用于定时任务

## 使用方式

### 1. 手动执行

**方式一：直接运行批处理文件**
```bash
cd D:\QIS_DASHBOARD\ali_bailian
upload_daily.bat
```

**方式二：命令行执行Python脚本**
```bash
# 上传今天的文件
python upload_daily.py

# 上传指定日期的文件
python upload_daily.py --date 2026-02-12

# 试运行（不实际上传）
python upload_daily.py --dry-run
```

### 2. 设置Windows定时任务

#### 🚀 方法A：使用一键设置脚本（最简单）

右键点击 `setup_scheduled_task.ps1`，选择"**以管理员身份运行**"，按提示操作即可。

或在管理员PowerShell中执行：
```powershell
cd D:\QIS_DASHBOARD\ali_bailian
.\setup_scheduled_task.ps1
```

#### 方法B：使用任务计划程序（手动）

1. 打开"任务计划程序" (taskschd.msc)
2. 点击"创建基本任务"
3. 填写信息：
   - **名称**: `阿里百炼研报上传`
   - **说明**: `每日自动上传期货研报到百炼知识库`
4. 触发器选择**每天**，设置时间（建议在邮件报告下载完成后，如 **18:00**）
5. 操作选择**启动程序**：
   - **程序/脚本**: `D:\QIS_DASHBOARD\ali_bailian\upload_daily.bat`
   - **起始于**: `D:\QIS_DASHBOARD\ali_bailian`
6. 完成

#### 方法C：使用PowerShell命令

```powershell
# 创建每日18:00执行的定时任务
$action = New-ScheduledTaskAction -Execute "D:\QIS_DASHBOARD\ali_bailian\upload_daily.bat" -WorkingDirectory "D:\QIS_DASHBOARD\ali_bailian"
$trigger = New-ScheduledTaskTrigger -Daily -At "18:00"
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType Interactive
Register-ScheduledTask -Action $action -Trigger $trigger -Principal $principal -TaskName "BailianReportUpload" -Description "每日上传期货研报到阿里百炼知识库"
```

### 3. 建议的定时任务流程

```
17:00 - 邮件报告下载 (run_daily_reports.bat)
         ↓
17:30 - 等待下载完成
         ↓
18:00 - 上传到百炼 (upload_daily.bat)
```

## 日志查看

日志文件位置：`D:\QIS_DASHBOARD\logs\upload_daily.log`

查看最近的上传记录：
```powershell
Get-Content D:\QIS_DASHBOARD\logs\upload_daily.log -Tail 50
```

## 注意事项

1. **只上传 general reports** - detailed reports 已被禁用
2. **自动跳过重复文件** - 使用智能去重机制
3. **需要网络连接** - 调用阿里云API
4. **环境变量配置** - 确保 `.env` 文件配置正确：
   - `DASHSCOPE_API_KEY`
   - `ALIBABA_CLOUD_ACCESS_KEY_ID`
   - `ALIBABA_CLOUD_ACCESS_KEY_SECRET`
   - `WORKSPACE_ID`
   - `INDEX_ID`

## 故障排查

### 上传失败
- 检查日志文件：`D:\QIS_DASHBOARD\logs\upload_daily.log`
- 检查网络连接
- 确认 `.env` 配置

### 文件未上传
- 确认文件已下载到：`D:\QIS_DASHBOARD\memory\commodity_general_reports\`
- 检查文件格式是否支持 (.pdf, .docx, .txt, .md)
- 查看上传历史：`D:\QIS_DASHBOARD\ali_bailian\.upload_history.json`

## 相关脚本

- **下载报告**: `D:\QIS_DASHBOARD\coremail\run_daily_reports.bat`
- **上传到百炼**: `D:\QIS_DASHBOARD\ali_bailian\upload_daily.bat`
- **启动Dashboard**: `D:\QIS_DASHBOARD\start_server.bat`
