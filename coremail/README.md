# Coremail 邮件系统工具

用于从 CICC Coremail 邮件系统下载研报的工具集。

## 文件说明

| 文件 | 说明 |
|------|------|
| `coremail_utils.py` | 底层 Coremail API 处理器 |
| `coremail_helper.py` | 高级邮件助手类（搜索、发送、下载） |
| `download_futures_reports.py` | 下载期货研报（概要版） |
| `download_detailed_reports.py` | 下载期货研报（完整版） |
| `param.json.example` | 配置文件模板 |
| `test_send_email.py` | 邮件发送测试脚本 |

## 快速开始

### 1. 配置账号

```powershell
# 复制配置模板
Copy-Item param.json.example param.json

# 编辑 param.json，填入您的邮箱和密码
```

### 2. 安装依赖

```powershell
pip install selenium requests
```

### 3. 测试邮件发送

```powershell
python test_send_email.py
```

### 4. 下载研报

```powershell
# 下载概要版研报
python download_futures_reports.py

# 下载完整版研报
python download_detailed_reports.py
```

## 配置说明

编辑 `param.json`：

```json
{
  "users": {
    "me": {
      "uid": "your.name@cicc.com.cn",
      "password": "your_password"
    }
  }
}
```

## 自定义配置

在各脚本文件顶部的配置区域可以修改：

- `SAVE_DIR`: PDF 保存目录
- `EMAIL_FOLDER`: 邮件文件夹名称
- `EMAIL_PATTERN`: 邮件主题搜索关键字
- `EMAIL_LIMIT`: 处理邮件数量限制

## 注意事项

- `param.json` 包含密码，已添加到 `.gitignore`，不会被提交到 Git
- 首次使用请确保已安装 Chrome 浏览器（用于 Selenium）
- 下载的 PDF 文件默认保存在 `downloads/` 目录下

## CoremailHelper 使用示例

```python
from coremail_helper import CoremailHelper

# 初始化（使用 param.json 配置）
helper = CoremailHelper()

# 搜索邮件
emails = helper.search_email(
    folder='收件箱.Research', 
    pattern='关键字',
    limit=10
)

# 发送邮件
helper.send_mail(
    recipients='someone@cicc.com.cn',
    subject='测试邮件',
    body='<h1>邮件内容</h1>',
    auto=True
)

# 下载附件
helper.download_email_all_attachments(email_id, folder=r'C:\Downloads')
```
