# 阿里云百炼 RAG 知识库工具使用指南

## 快速开始

### 1. 配置环境变量

复制 `.env.example` 为 `.env` 并填写你的配置信息：

```bash
cp .env.example .env
```

然后编辑 `.env` 文件，填写以下信息：
- `ALIBABA_CLOUD_ACCESS_KEY_ID`: 你的阿里云 AccessKey ID
- `ALIBABA_CLOUD_ACCESS_KEY_SECRET`: 你的阿里云 AccessKey Secret
- `WORKSPACE_ID`: 你的百炼业务空间 ID
- `INDEX_ID`: 你的知识库 ID（可选）

### 2. 安装依赖

项目根目录的 `requirements.txt` 已包含所需依赖：
- `alibabacloud_bailian20231229` - 百炼 SDK
- `alibabacloud_tea_openapi` - 阿里云 OpenAPI SDK
- `python-dotenv` - 环境变量管理
- `requests` - HTTP 请求库

### 3. 使用工具

#### 方式一：使用 upload_data_v2.py（推荐）

**上传单个文件：**
```bash
python ali_bailian/upload_data_v2.py --file /path/to/document.pdf
```

**批量上传文件夹：**
```bash
python ali_bailian/upload_data_v2.py --folder /path/to/pdfs/
```

**指定知识库 ID：**
```bash
python ali_bailian/upload_data_v2.py --file doc.pdf --index-id YOUR_INDEX_ID
```

**添加标签：**
```bash
python ali_bailian/upload_data_v2.py --file doc.pdf --tag 重要 --tag 财报
```

**跳过等待解析和索引：**
```bash
python ali_bailian/upload_data_v2.py --file doc.pdf --skip-parse-wait --skip-index-wait
```

#### 方式二：使用 upload_data.py（原始版本）

```bash
python ali_bailian/upload_data.py --file /path/to/document.pdf --index-id YOUR_INDEX_ID
```

#### 方式三：使用 bailian_SDK.py（SDK 封装）

查看该文件了解如何在代码中使用封装好的 SDK 功能。

## 支持的文件格式

- 文档：PDF, DOCX, DOC, TXT, Markdown
- 演示文稿：PPTX, PPT
- 电子表格：XLSX, XLS
- 网页：HTML
- 图片：PNG, JPG, JPEG, BMP, GIF

## 详细文档

请参考 [RAG Info.md](RAG%20Info.md) 获取完整的 API 使用说明和开发指南。

## 常用命令参数

### upload_data_v2.py 参数

| 参数 | 说明 |
|-----|------|
| `--file, -f` | 上传单个文件的路径 |
| `--folder, -d` | 批量上传文件夹路径 |
| `--index-id` | 知识库 ID |
| `--workspace-id` | 业务空间 ID |
| `--tag` | 文件标签（可多次使用） |
| `--no-recursive` | 不递归搜索子文件夹 |
| `--skip-parse-wait` | 不等待文件解析完成 |
| `--skip-index-wait` | 不等待索引任务完成 |

## 注意事项

1. 确保你的阿里云账号有 `AliyunBailianDataFullAccess` 权限
2. 每个业务空间最多支持 10 万个文件
3. 文件上传后需要等待解析完成才能被检索
4. 索引任务在高峰时段可能需要数小时
