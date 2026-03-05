# 阿里云百炼 RAG 知识库开发指南

> 本文档整理自阿里云百炼官方文档，提供完整的知识库 API 使用参考。

---

## 目录

1. [前置准备](#1-前置准备)
2. [环境配置](#2-环境配置)
3. [创建知识库](#3-创建知识库)
4. [检索知识库](#4-检索知识库)
5. [更新知识库](#5-更新知识库)
6. [管理知识库](#6-管理知识库)
7. [常见问题](#7-常见问题)
8. [计费说明](#8-计费说明)
9. [完整示例代码](#9-完整示例代码)

---

## 1. 前置准备

### 1.1 权限要求

| 账号类型 | 权限要求 | 操作范围 |
|---------|---------|---------|
| 主账号 | 无需额外配置 | 可操作所有业务空间下的知识库 |
| 子账号 | 需获取 `AliyunBailianDataFullAccess` 策略 | 只能操作已加入业务空间中的知识库 |

### 1.2 SDK 安装

```bash
pip install alibabacloud_bailian20231229
```

### 1.3 支持的文件格式

PDF、DOCX、DOC、TXT、Markdown、PPTX、PPT、XLSX、XLS、HTML、PNG、JPG、JPEG、BMP、GIF

### 1.4 限制说明

- 每个业务空间最多支持 **10万个文件**
- 索引任务在请求高峰时段可能需要数小时

---

## 2. 环境配置

### 2.1 环境变量设置

```bash
# Linux/macOS
export ALIBABA_CLOUD_ACCESS_KEY_ID='您的阿里云访问密钥ID'
export ALIBABA_CLOUD_ACCESS_KEY_SECRET='您的阿里云访问密钥密码'
export WORKSPACE_ID='您的阿里云百炼业务空间ID'
```

### 2.2 接入地址

| 网络类型 | 云类型 | 地域 | 接入地址 |
|---------|-------|------|---------|
| 公网 | 公有云 | 北京 | `bailian.cn-beijing.aliyuncs.com` |
| 公网 | 金融云 | 上海 | `bailian.cn-shanghai-finance-1.aliyuncs.com` |
| VPC | 公有云 | 北京 | `bailian-vpc.cn-beijing.aliyuncs.com` |
| VPC | 金融云 | 上海 | `bailian-vpc.cn-shanghai-finance-1.aliyuncs.com` |

> **注意**：VPC 接入不支持跨地域访问

---

## 3. 创建知识库

### 3.1 流程概览

```
初始化客户端 → 申请上传租约 → 上传文件 → 添加文件到类目 
    → 等待文件解析 → 初始化知识库 → 提交索引任务 → 等待索引完成
```

### 3.2 初始化客户端

```python
from alibabacloud_bailian20231229.client import Client as bailian20231229Client
from alibabacloud_tea_openapi import models as open_api_models
import os

def create_client() -> bailian20231229Client:
    """创建并配置客户端"""
    config = open_api_models.Config(
        access_key_id=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_ID'),
        access_key_secret=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
    )
    config.endpoint = 'bailian.cn-beijing.aliyuncs.com'
    return bailian20231229Client(config)
```

### 3.3 上传文件

#### 3.3.1 申请文件上传租约

```python
from alibabacloud_bailian20231229 import models as bailian_20231229_models
from alibabacloud_tea_util import models as util_models

def apply_lease(client, category_id, file_name, file_md5, file_size, workspace_id):
    """申请文件上传租约"""
    request = bailian_20231229_models.ApplyFileUploadLeaseRequest(
        file_name=file_name,
        md_5=file_md5,
        size_in_bytes=file_size,
    )
    runtime = util_models.RuntimeOptions()
    return client.apply_file_upload_lease_with_options(
        category_id, workspace_id, request, {}, runtime
    )
```

**参数说明**：

| 参数 | 说明 |
|-----|------|
| `category_id` | 类目ID，使用默认类目传 `default` |
| `file_name` | 文件名（含后缀），必须与实际文件名一致 |
| `file_md5` | 文件的 MD5 值 |
| `file_size` | 文件大小（字节） |

**返回值**：
- `Data.FileUploadLeaseId` - 租约ID
- `Data.Param.Url` - 临时上传URL
- `Data.Param.Headers` - 上传请求头

#### 3.3.2 工具函数

```python
import hashlib
import os

def calculate_md5(file_path: str) -> str:
    """计算文件的MD5值"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def get_file_size(file_path: str) -> int:
    """获取文件大小（字节）"""
    return os.path.getsize(file_path)
```

#### 3.3.3 上传文件到临时存储

```python
import requests

def upload_file(pre_signed_url, headers, file_path):
    """将文件上传到阿里云百炼服务（使用二进制方式）"""
    with open(file_path, 'rb') as f:
        file_content = f.read()
    upload_headers = {
        "X-bailian-extra": headers["X-bailian-extra"],
        "Content-Type": headers["Content-Type"]
    }
    response = requests.put(pre_signed_url, data=file_content, headers=upload_headers)
    response.raise_for_status()
```

> **注意**：该 URL 为预签名 URL，不支持 FormData 方式上传

#### 3.3.4 添加文件到类目

```python
def add_file(client, lease_id, parser, category_id, workspace_id):
    """将文件添加到指定类目"""
    request = bailian_20231229_models.AddFileRequest(
        lease_id=lease_id,
        parser=parser,  # 使用 'DASHSCOPE_DOCMIND'
        category_id=category_id,
    )
    runtime = util_models.RuntimeOptions()
    return client.add_file_with_options(workspace_id, request, {}, runtime)
```

#### 3.3.5 查询文件解析状态

```python
def describe_file(client, workspace_id, file_id):
    """获取文件的基本信息和解析状态"""
    runtime = util_models.RuntimeOptions()
    return client.describe_file_with_options(workspace_id, file_id, {}, runtime)
```

**文件状态说明**：

| 状态 | 说明 |
|-----|------|
| `INIT` | 文件待解析 |
| `PARSING` | 文件解析中 |
| `PARSE_SUCCESS` | 文件解析完成，可导入知识库 |

### 3.4 创建知识库

#### 3.4.1 初始化知识库

```python
def create_index(client, workspace_id, file_id, name, structure_type, source_type, sink_type):
    """初始化知识库"""
    request = bailian_20231229_models.CreateIndexRequest(
        structure_type=structure_type,  # 'unstructured'
        name=name,
        source_type=source_type,        # 'DATA_CENTER_FILE'
        sink_type=sink_type,            # 'DEFAULT' 或 'BUILT_IN'
        document_ids=[file_id]
    )
    runtime = util_models.RuntimeOptions()
    return client.create_index_with_options(workspace_id, request, {}, runtime)
```

**参数说明**：

| 参数 | 值 | 说明 |
|-----|-----|------|
| `structure_type` | `unstructured` | 知识库数据类型 |
| `source_type` | `DATA_CENTER_FILE` | 数据来源类型 |
| `sink_type` | `DEFAULT` 或 `BUILT_IN` | 向量存储类型 |

#### 3.4.2 提交索引任务

```python
def submit_index(client, workspace_id, index_id):
    """提交索引任务，启动知识库索引构建"""
    request = bailian_20231229_models.SubmitIndexJobRequest(index_id=index_id)
    runtime = util_models.RuntimeOptions()
    return client.submit_index_job_with_options(workspace_id, request, {}, runtime)
```

#### 3.4.3 等待索引任务完成

```python
def get_index_job_status(client, workspace_id, job_id, index_id):
    """查询索引任务状态"""
    request = bailian_20231229_models.GetIndexJobStatusRequest(
        index_id=index_id,
        job_id=job_id
    )
    runtime = util_models.RuntimeOptions()
    return client.get_index_job_status_with_options(workspace_id, request, {}, runtime)
```

**任务状态**：当 `Data.Status` 为 `COMPLETED` 时，表示知识库创建完成。

---

## 4. 检索知识库

### 4.1 检索方式

| 方式 | 说明 |
|-----|------|
| 阿里云百炼应用 | 通过 `rag_options` 传入知识库ID，模型结合检索结果生成回答 |
| 阿里云API（Retrieve） | 直接返回文本切片 |

### 4.2 使用 API 检索

```python
def retrieve_index(client, workspace_id, index_id, query):
    """在指定知识库中检索信息"""
    request = bailian_20231229_models.RetrieveRequest(
        index_id=index_id,
        query=query
    )
    runtime = util_models.RuntimeOptions()
    return client.retrieve_with_options(workspace_id, request, {}, runtime)
```

> **提示**：可通过 `SearchFilters` 设置检索条件（如标签筛选）排除干扰信息

---

## 5. 更新知识库

### 5.1 更新流程

```
上传更新后的文件 → 追加文件至知识库 → 删除旧文件
```

### 5.2 追加文件至知识库

```python
def submit_index_add_documents_job(client, workspace_id, index_id, file_id, source_type):
    """向知识库追加导入已解析的文件"""
    request = bailian_20231229_models.SubmitIndexAddDocumentsJobRequest(
        index_id=index_id,
        document_ids=[file_id],
        source_type=source_type  # 'DATA_CENTER_FILE'
    )
    runtime = util_models.RuntimeOptions()
    return client.submit_index_add_documents_job_with_options(
        workspace_id, request, {}, runtime
    )
```

> **注意**：任务完成前请勿重复提交

### 5.3 删除旧文件

```python
def delete_index_document(client, workspace_id, index_id, file_id):
    """从知识库中永久删除文件"""
    request = bailian_20231229_models.DeleteIndexDocumentRequest(
        index_id=index_id,
        document_ids=[file_id]
    )
    runtime = util_models.RuntimeOptions()
    return client.delete_index_document_with_options(
        workspace_id, request, {}, runtime
    )
```

> **限制**：仅能删除状态为 `INSERT_ERROR` 或 `FINISH` 的文件

### 5.4 自动更新/同步

使用对象存储 OSS 管理文件，通过函数计算 FC 监听文件变更事件，自动同步更新至知识库。

---

## 6. 管理知识库

### 6.1 查看知识库

```python
def list_indices(client, workspace_id):
    """获取业务空间下的知识库列表"""
    request = bailian_20231229_models.ListIndicesRequest()
    runtime = util_models.RuntimeOptions()
    return client.list_indices_with_options(workspace_id, request, {}, runtime)
```

### 6.2 删除知识库

```python
def delete_index(client, workspace_id, index_id):
    """永久删除指定知识库"""
    request = bailian_20231229_models.DeleteIndexRequest(index_id=index_id)
    runtime = util_models.RuntimeOptions()
    return client.delete_index_with_options(workspace_id, request, {}, runtime)
```

> **注意**：删除前需解除该知识库关联的所有应用，本操作不会删除已添加至类目中的文件

---

## 7. 常见问题

### Q1: 新建的知识库里没有内容？

**原因**：未成功执行提交索引任务

**解决**：调用 `CreateIndex` 后需调用 `SubmitIndexJob` 接口

### Q2: 报错 `Access your uploaded file failed`

**原因**：未成功执行上传文件到临时存储

**解决**：确认上传步骤成功执行后，再调用 `AddFile` 接口

### Q3: 报错 `Access denied: Either you are not authorized...`

**可能原因**：
1. 服务接入地址有误
2. `WorkspaceId` 值不正确
3. 不是该业务空间的成员

### Q4: 报错 `Specified access key is not found or invalid`

**原因**：`access_key_id` 或 `access_key_secret` 值不正确，或已被禁用

---

## 8. 计费说明

知识库采用**按量付费（后付费）**，按小时统计用量并自动扣费。

| 计费项 | 说明 |
|-------|------|
| 规格费用 | 标准版/旗舰版知识库的实际运行时长费用 |
| 模型调用费用 | 创建、更新或检索知识库时调用向量(embedding)、排序(rerank)模型的费用 |

> 请确保账户余额充足，以免因欠费导致服务中断

---

## 9. 完整示例代码

### 9.1 创建知识库完整示例

```python
import hashlib
import os
import time
import requests
from alibabacloud_bailian20231229 import models as bailian_20231229_models
from alibabacloud_bailian20231229.client import Client as bailian20231229Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models


def check_environment_variables():
    """检查必要的环境变量"""
    required_vars = {
        'ALIBABA_CLOUD_ACCESS_KEY_ID': '阿里云访问密钥ID',
        'ALIBABA_CLOUD_ACCESS_KEY_SECRET': '阿里云访问密钥密码',
        'WORKSPACE_ID': '阿里云百炼业务空间ID'
    }
    missing_vars = []
    for var, description in required_vars.items():
        if not os.environ.get(var):
            missing_vars.append(var)
            print(f"错误：请设置 {var} 环境变量 ({description})")
    return len(missing_vars) == 0


def calculate_md5(file_path: str) -> str:
    """计算文件的MD5值"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def get_file_size(file_path: str) -> int:
    """获取文件大小（字节）"""
    return os.path.getsize(file_path)


def create_client() -> bailian20231229Client:
    """创建并配置客户端"""
    config = open_api_models.Config(
        access_key_id=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_ID'),
        access_key_secret=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
    )
    config.endpoint = 'bailian.cn-beijing.aliyuncs.com'
    return bailian20231229Client(config)


def apply_lease(client, category_id, file_name, file_md5, file_size, workspace_id):
    """申请文件上传租约"""
    request = bailian_20231229_models.ApplyFileUploadLeaseRequest(
        file_name=file_name,
        md_5=file_md5,
        size_in_bytes=file_size,
    )
    runtime = util_models.RuntimeOptions()
    return client.apply_file_upload_lease_with_options(
        category_id, workspace_id, request, {}, runtime
    )


def upload_file(pre_signed_url, headers, file_path):
    """上传文件到临时存储"""
    with open(file_path, 'rb') as f:
        file_content = f.read()
    upload_headers = {
        "X-bailian-extra": headers["X-bailian-extra"],
        "Content-Type": headers["Content-Type"]
    }
    response = requests.put(pre_signed_url, data=file_content, headers=upload_headers)
    response.raise_for_status()


def add_file(client, lease_id, parser, category_id, workspace_id):
    """添加文件到类目"""
    request = bailian_20231229_models.AddFileRequest(
        lease_id=lease_id,
        parser=parser,
        category_id=category_id,
    )
    runtime = util_models.RuntimeOptions()
    return client.add_file_with_options(workspace_id, request, {}, runtime)


def describe_file(client, workspace_id, file_id):
    """获取文件信息"""
    runtime = util_models.RuntimeOptions()
    return client.describe_file_with_options(workspace_id, file_id, {}, runtime)


def create_index(client, workspace_id, file_id, name, structure_type, source_type, sink_type):
    """初始化知识库"""
    request = bailian_20231229_models.CreateIndexRequest(
        structure_type=structure_type,
        name=name,
        source_type=source_type,
        sink_type=sink_type,
        document_ids=[file_id]
    )
    runtime = util_models.RuntimeOptions()
    return client.create_index_with_options(workspace_id, request, {}, runtime)


def submit_index(client, workspace_id, index_id):
    """提交索引任务"""
    request = bailian_20231229_models.SubmitIndexJobRequest(index_id=index_id)
    runtime = util_models.RuntimeOptions()
    return client.submit_index_job_with_options(workspace_id, request, {}, runtime)


def get_index_job_status(client, workspace_id, job_id, index_id):
    """查询索引任务状态"""
    request = bailian_20231229_models.GetIndexJobStatusRequest(
        index_id=index_id,
        job_id=job_id
    )
    runtime = util_models.RuntimeOptions()
    return client.get_index_job_status_with_options(workspace_id, request, {}, runtime)


def create_knowledge_base(file_path: str, workspace_id: str, name: str):
    """创建知识库的完整流程"""
    category_id = 'default'
    parser = 'DASHSCOPE_DOCMIND'
    source_type = 'DATA_CENTER_FILE'
    structure_type = 'unstructured'
    sink_type = 'DEFAULT'
    
    try:
        # 步骤1：初始化客户端
        print("步骤1：初始化Client")
        client = create_client()
        
        # 步骤2：准备文件信息
        print("步骤2：准备文件信息")
        file_name = os.path.basename(file_path)
        file_md5 = calculate_md5(file_path)
        file_size = get_file_size(file_path)
        
        # 步骤3：申请上传租约
        print("步骤3：申请上传租约")
        lease_response = apply_lease(client, category_id, file_name, file_md5, file_size, workspace_id)
        lease_id = lease_response.body.data.file_upload_lease_id
        upload_url = lease_response.body.data.param.url
        upload_headers = lease_response.body.data.param.headers
        
        # 步骤4：上传文件
        print("步骤4：上传文件")
        upload_file(upload_url, upload_headers, file_path)
        
        # 步骤5：添加文件到类目
        print("步骤5：添加文件到类目")
        add_response = add_file(client, lease_id, parser, category_id, workspace_id)
        file_id = add_response.body.data.file_id
        
        # 步骤6：等待文件解析完成
        print("步骤6：等待文件解析完成")
        while True:
            describe_response = describe_file(client, workspace_id, file_id)
            status = describe_response.body.data.status
            print(f"  当前状态：{status}")
            if status == 'PARSE_SUCCESS':
                print("  文件解析完成！")
                break
            elif status in ['INIT', 'PARSING']:
                time.sleep(5)
            else:
                print(f"  未知状态：{status}")
                return None
        
        # 步骤7：初始化知识库
        print("步骤7：初始化知识库")
        index_response = create_index(client, workspace_id, file_id, name, 
                                      structure_type, source_type, sink_type)
        index_id = index_response.body.data.id
        
        # 步骤8：提交索引任务
        print("步骤8：提交索引任务")
        submit_response = submit_index(client, workspace_id, index_id)
        job_id = submit_response.body.data.id
        
        # 步骤9：等待索引完成
        print("步骤9：等待索引完成")
        while True:
            status_response = get_index_job_status(client, workspace_id, job_id, index_id)
            status = status_response.body.data.status
            print(f"  当前状态：{status}")
            if status == 'COMPLETED':
                break
            time.sleep(5)
        
        print(f"知识库创建成功！ID: {index_id}")
        return index_id
        
    except Exception as e:
        print(f"发生错误：{e}")
        return None


if __name__ == '__main__':
    if not check_environment_variables():
        print("环境变量校验未通过。")
    else:
        file_path = input("请输入文件路径：")
        kb_name = input("请输入知识库名称：")
        workspace_id = os.environ.get('WORKSPACE_ID')
        create_knowledge_base(file_path, workspace_id, kb_name)
```

### 9.2 检索知识库示例

```python
import os
from alibabacloud_bailian20231229 import models as bailian_20231229_models
from alibabacloud_bailian20231229.client import Client as bailian20231229Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient


def create_client() -> bailian20231229Client:
    config = open_api_models.Config(
        access_key_id=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_ID'),
        access_key_secret=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
    )
    config.endpoint = 'bailian.cn-beijing.aliyuncs.com'
    return bailian20231229Client(config)


def retrieve_index(client, workspace_id, index_id, query):
    """检索知识库"""
    request = bailian_20231229_models.RetrieveRequest(
        index_id=index_id,
        query=query
    )
    runtime = util_models.RuntimeOptions()
    return client.retrieve_with_options(workspace_id, request, {}, runtime)


if __name__ == '__main__':
    client = create_client()
    workspace_id = os.environ.get('WORKSPACE_ID')
    index_id = input("请输入知识库ID：")
    query = input("请输入检索内容：")
    
    resp = retrieve_index(client, workspace_id, index_id, query)
    print(UtilClient.to_jsonstring(resp.body))
```

### 9.3 更新知识库示例

```python
import hashlib
import os
import time
import requests
from alibabacloud_bailian20231229 import models as bailian_20231229_models
from alibabacloud_bailian20231229.client import Client as bailian20231229Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models


def create_client() -> bailian20231229Client:
    config = open_api_models.Config(
        access_key_id=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_ID'),
        access_key_secret=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
    )
    config.endpoint = 'bailian.cn-beijing.aliyuncs.com'
    return bailian20231229Client(config)


def calculate_md5(file_path: str) -> str:
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def update_knowledge_base(file_path: str, workspace_id: str, index_id: str, old_file_id: str):
    """更新知识库"""
    category_id = 'default'
    parser = 'DASHSCOPE_DOCMIND'
    source_type = 'DATA_CENTER_FILE'
    
    try:
        client = create_client()
        
        # 1. 上传新文件
        file_name = os.path.basename(file_path)
        file_md5 = calculate_md5(file_path)
        file_size = os.path.getsize(file_path)
        
        lease_request = bailian_20231229_models.ApplyFileUploadLeaseRequest(
            file_name=file_name, md_5=file_md5, size_in_bytes=file_size
        )
        runtime = util_models.RuntimeOptions()
        lease_response = client.apply_file_upload_lease_with_options(
            category_id, workspace_id, lease_request, {}, runtime
        )
        
        lease_id = lease_response.body.data.file_upload_lease_id
        upload_url = lease_response.body.data.param.url
        upload_headers = lease_response.body.data.param.headers
        
        with open(file_path, 'rb') as f:
            requests.put(upload_url, data=f.read(), headers={
                "X-bailian-extra": upload_headers["X-bailian-extra"],
                "Content-Type": upload_headers["Content-Type"]
            })
        
        add_request = bailian_20231229_models.AddFileRequest(
            lease_id=lease_id, parser=parser, category_id=category_id
        )
        add_response = client.add_file_with_options(workspace_id, add_request, {}, runtime)
        file_id = add_response.body.data.file_id
        
        # 2. 等待文件解析
        while True:
            resp = client.describe_file_with_options(workspace_id, file_id, {}, runtime)
            if resp.body.data.status == 'PARSE_SUCCESS':
                break
            time.sleep(5)
        
        # 3. 追加新文件
        add_doc_request = bailian_20231229_models.SubmitIndexAddDocumentsJobRequest(
            index_id=index_id, document_ids=[file_id], source_type=source_type
        )
        add_doc_response = client.submit_index_add_documents_job_with_options(
            workspace_id, add_doc_request, {}, runtime
        )
        job_id = add_doc_response.body.data.id
        
        # 4. 等待追加完成
        while True:
            status_request = bailian_20231229_models.GetIndexJobStatusRequest(
                index_id=index_id, job_id=job_id
            )
            status_resp = client.get_index_job_status_with_options(
                workspace_id, status_request, {}, runtime
            )
            if status_resp.body.data.status == 'COMPLETED':
                break
            time.sleep(5)
        
        # 5. 删除旧文件
        delete_request = bailian_20231229_models.DeleteIndexDocumentRequest(
            index_id=index_id, document_ids=[old_file_id]
        )
        client.delete_index_document_with_options(workspace_id, delete_request, {}, runtime)
        
        print("知识库更新成功！")
        return index_id
        
    except Exception as e:
        print(f"发生错误：{e}")
        return None


if __name__ == '__main__':
    workspace_id = os.environ.get('WORKSPACE_ID')
    file_path = input("请输入新文件路径：")
    index_id = input("请输入知识库ID：")
    old_file_id = input("请输入旧文件ID：")
    update_knowledge_base(file_path, workspace_id, index_id, old_file_id)
```

### 9.4 管理知识库示例

```python
import os
from alibabacloud_bailian20231229 import models as bailian_20231229_models
from alibabacloud_bailian20231229.client import Client as bailian20231229Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient


def create_client() -> bailian20231229Client:
    config = open_api_models.Config(
        access_key_id=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_ID'),
        access_key_secret=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
    )
    config.endpoint = 'bailian.cn-beijing.aliyuncs.com'
    return bailian20231229Client(config)


def list_indices(client, workspace_id):
    """查看知识库列表"""
    request = bailian_20231229_models.ListIndicesRequest()
    runtime = util_models.RuntimeOptions()
    return client.list_indices_with_options(workspace_id, request, {}, runtime)


def delete_index(client, workspace_id, index_id):
    """删除知识库"""
    request = bailian_20231229_models.DeleteIndexRequest(index_id=index_id)
    runtime = util_models.RuntimeOptions()
    return client.delete_index_with_options(workspace_id, request, {}, runtime)


if __name__ == '__main__':
    client = create_client()
    workspace_id = os.environ.get('WORKSPACE_ID')
    
    option = input("选择操作：\n1. 查看知识库\n2. 删除知识库\n请输入(1/2)：")
    
    if option == '1':
        resp = list_indices(client, workspace_id)
        print(UtilClient.to_jsonstring(resp.body.data))
    elif option == '2':
        index_id = input("请输入知识库ID：")
        confirm = input(f"确定删除 {index_id}？(y/n)：")
        if confirm.lower() == 'y':
            resp = delete_index(client, workspace_id, index_id)
            print("删除成功！" if resp.body.status == 200 else "删除失败")
```

---

## 参考资料

- [阿里云百炼官方文档](https://help.aliyun.com/product/610396.html)
- [API 概览（知识库）](https://help.aliyun.com/document_detail/2789956.html)
- [知识库计费说明](https://help.aliyun.com/document_detail/2712195.html)
- [错误码参考](https://error-center.aliyun.com/)
