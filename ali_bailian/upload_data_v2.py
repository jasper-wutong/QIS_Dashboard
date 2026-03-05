"""
阿里云百炼 RAG 知识库文件上传工具 v2

支持：
- 单个 PDF 文件上传
- 批量上传整个文件夹中的 PDF
- 上传到已有知识库

使用方式：
    # 上传单个文件
    python upload_data_v2.py --file /path/to/document.pdf --index-id YOUR_INDEX_ID

    # 批量上传文件夹
    python upload_data_v2.py --folder /path/to/pdfs/ --index-id YOUR_INDEX_ID

环境变量配置（可在 .env 文件中设置）：
    ALIBABA_CLOUD_ACCESS_KEY_ID=你的AccessKeyID
    ALIBABA_CLOUD_ACCESS_KEY_SECRET=你的AccessKeySecret
    WORKSPACE_ID=你的业务空间ID
    INDEX_ID=你的知识库ID（可选，也可通过命令行传入）
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 设置输出编码为 UTF-8，避免 Windows 批处理中的 GBK 编码错误
if sys.stdout.encoding != 'utf-8':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass
from typing import Dict, List, Optional

import requests
from alibabacloud_bailian20231229 import models as bailian_models
from alibabacloud_bailian20231229.client import Client as BailianClient
from alibabacloud_tea_openapi import models as open_api_models

# 从项目根目录加载 .env 文件
try:
    from dotenv import load_dotenv
    from pathlib import Path
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    pass

# ============== 配置区域 ==============
# 支持的文件格式
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt', '.md', '.pptx', '.ppt', 
                        '.xlsx', '.xls', '.html', '.png', '.jpg', '.jpeg', '.bmp', '.gif'}

# 默认配置
DEFAULT_ENDPOINT = 'bailian.cn-beijing.aliyuncs.com'
DEFAULT_REGION = 'cn-beijing'
DEFAULT_PARSER = 'DASHSCOPE_DOCMIND'
DEFAULT_CATEGORY_ID = 'default'
DEFAULT_CATEGORY_TYPE = 'UNSTRUCTURED'

# 上传历史记录文件路径
UPLOAD_HISTORY_FILE = Path(__file__).parent / '.upload_history.json'

# ============== 上传历史记录管理 ==============

class UploadHistory:
    """管理文件上传历史记录，避免重复上传"""
    
    def __init__(self, history_file: Path = UPLOAD_HISTORY_FILE):
        self.history_file = history_file
        self.records: Dict[str, Dict] = {}  # key: md5, value: {file_id, filename, upload_time, index_id}
        self._load()
    
    def _load(self):
        """加载历史记录"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.records = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.records = {}
    
    def _save(self):
        """保存历史记录"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)
    
    def is_uploaded(self, md5: str, index_id: str) -> bool:
        """检查文件是否已上传到指定知识库"""
        if md5 in self.records:
            record = self.records[md5]
            # 检查是否上传到同一个知识库
            return record.get('index_id') == index_id
        return False
    
    def get_record(self, md5: str) -> Optional[Dict]:
        """获取上传记录"""
        return self.records.get(md5)
    
    def add_record(self, md5: str, file_id: str, filename: str, index_id: str, job_id: str = None):
        """添加上传记录"""
        self.records[md5] = {
            'file_id': file_id,
            'filename': filename,
            'index_id': index_id,
            'job_id': job_id,
            'upload_time': datetime.now().isoformat(),
        }
        self._save()
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_files': len(self.records),
            'index_ids': list(set(r.get('index_id') for r in self.records.values())),
        }


# ============== 工具函数 ==============

def log(msg: str, level: str = "INFO"):
    """打印日志"""
    print(f"[{level}] {msg}")


def calculate_md5(file_path: str) -> str:
    """计算文件 MD5"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def truncate_filename(filename: str, max_length: int = 120) -> str:
    """
    截断文件名以符合API限制（最大128字符，预留一些余量）
    保留扩展名，截断中间部分
    """
    if len(filename) <= max_length:
        return filename
    
    # 分离文件名和扩展名
    name, ext = os.path.splitext(filename)
    
    # 计算可用长度（保留扩展名 + "..." 标记）
    available = max_length - len(ext) - 3  # 3 for "..."
    if available < 10:
        available = 10  # 至少保留10个字符
    
    # 截断文件名
    truncated = name[:available] + "..." + ext
    log(f"文件名过长，已截断: {filename[:50]}... -> {truncated}", "WARN")
    return truncated


def ensure_api_success(api_name: str, response) -> None:
    """检查 API 响应是否成功"""
    body = getattr(response, "body", None)
    if body is None:
        raise RuntimeError(f"{api_name} 失败: 响应体为空")
    
    success = getattr(body, "success", False)
    if isinstance(success, str):
        success = success.lower() == "true"
    
    if success:
        return
    
    code = getattr(body, "code", "unknown")
    message = getattr(body, "message", "unknown")
    request_id = getattr(body, "request_id", "unknown")
    raise RuntimeError(f"{api_name} 失败: code={code}, message={message}, request_id={request_id}")


# ============== 百炼客户端 ==============

class BailianUploader:
    """百炼文件上传器"""
    
    def __init__(
        self,
        access_key_id: str,
        access_key_secret: str,
        workspace_id: str,
        index_id: str,
        endpoint: str = DEFAULT_ENDPOINT,
        region_id: str = DEFAULT_REGION,
        category_id: str = DEFAULT_CATEGORY_ID,
        parser: str = DEFAULT_PARSER,
    ):
        self.workspace_id = workspace_id
        self.index_id = index_id
        self.category_id = category_id
        self.parser = parser
        
        # 初始化客户端
        config = open_api_models.Config(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            region_id=region_id,
            endpoint=endpoint,
        )
        self.client = BailianClient(config)
    
    def apply_upload_lease(self, file_path: str):
        """申请文件上传租约"""
        file_name = os.path.basename(file_path)
        # 截断过长的文件名（API限制128字符）
        file_name = truncate_filename(file_name)
        file_size = os.path.getsize(file_path)
        file_md5 = calculate_md5(file_path)
        
        request = bailian_models.ApplyFileUploadLeaseRequest(
            category_type=DEFAULT_CATEGORY_TYPE,
            file_name=file_name,
            md_5=file_md5,
            size_in_bytes=str(file_size),
        )
        
        response = self.client.apply_file_upload_lease(
            self.category_id, self.workspace_id, request
        )
        ensure_api_success("ApplyFileUploadLease", response)
        
        data = response.body.data
        if not data or not data.file_upload_lease_id:
            raise RuntimeError("申请上传租约失败: 返回数据为空")
        
        return data
    
    def upload_to_oss(self, file_path: str, lease_data, max_retries: int = 3) -> None:
        """上传文件到临时存储（带重试机制）"""
        from urllib.parse import quote
        
        headers = {}
        if lease_data.param.headers:
            for k, v in lease_data.param.headers.items():
                if v is None:
                    continue
                key = str(k)
                val = str(v)
                # 对非 ASCII 字符进行 URL 编码，避免 requests 的 latin-1 编码错误
                try:
                    val.encode('latin-1')
                except UnicodeEncodeError:
                    val = quote(val, safe='')
                headers[key] = val
            headers.pop("Host", None)
            headers.pop("host", None)
        
        last_error = None
        for attempt in range(max_retries):
            try:
                with open(file_path, "rb") as f:
                    response = requests.request(
                        method=lease_data.param.method.upper(),
                        url=lease_data.param.url,
                        data=f,
                        headers=headers,
                        timeout=600,  # 增加超时时间
                    )
                
                if response.status_code < 200 or response.status_code >= 300:
                    raise RuntimeError(f"文件上传失败: HTTP {response.status_code}")
                return  # 成功
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    log(f"上传失败，重试 {attempt + 2}/{max_retries}...", "WARN")
                    time.sleep(2 ** attempt)  # 指数退避
        
        raise last_error
    
    def add_file(self, lease_id: str, tags: Optional[List[str]] = None) -> str:
        """将文件添加到数据中心"""
        request = bailian_models.AddFileRequest(
            category_id=self.category_id,
            category_type=DEFAULT_CATEGORY_TYPE,
            lease_id=lease_id,
            parser=self.parser,
            tags=tags or [],
        )
        
        response = self.client.add_file(self.workspace_id, request)
        ensure_api_success("AddFile", response)
        
        file_id = response.body.data.file_id
        if not file_id:
            raise RuntimeError("添加文件失败: FileId 为空")
        
        return file_id
    
    def wait_for_parse(self, file_id: str, timeout: int = 600, interval: int = 5) -> None:
        """等待文件解析完成"""
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            response = self.client.describe_file(self.workspace_id, file_id)
            ensure_api_success("DescribeFile", response)
            
            status = response.body.data.status
            log(f"解析状态: {status}")
            
            if status == "PARSE_SUCCESS":
                return
            if status == "PARSE_FAILED":
                raise RuntimeError("文件解析失败")
            
            time.sleep(interval)
        
        raise TimeoutError("等待文件解析超时")
    
    def submit_to_index(self, file_id: str) -> str:
        """提交文件到知识库索引"""
        request = bailian_models.SubmitIndexAddDocumentsJobRequest(
            index_id=self.index_id,
            source_type="DATA_CENTER_FILE",
            document_ids=[file_id],
        )
        
        response = self.client.submit_index_add_documents_job(self.workspace_id, request)
        ensure_api_success("SubmitIndexAddDocumentsJob", response)
        
        job_id = response.body.data.id
        if not job_id:
            raise RuntimeError("提交索引任务失败: JobId 为空")
        
        return job_id
    
    def wait_for_index(self, job_id: str, timeout: int = 1200, interval: int = 5) -> None:
        """等待索引任务完成"""
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            request = bailian_models.GetIndexJobStatusRequest(
                index_id=self.index_id,
                job_id=job_id,
                page_number=1,
                page_size=20,
            )
            response = self.client.get_index_job_status(self.workspace_id, request)
            ensure_api_success("GetIndexJobStatus", response)
            
            status = response.body.data.status
            log(f"索引状态: {status}")
            
            if status == "COMPLETED":
                return
            if status == "FAILED":
                raise RuntimeError("索引任务失败")
            
            time.sleep(interval)
        
        raise TimeoutError("等待索引任务超时")
    
    def upload_single_file(
        self, 
        file_path: str, 
        wait_parse: bool = True,
        wait_index: bool = True,
        tags: Optional[List[str]] = None,
        force: bool = False,
        history: UploadHistory = None,
    ) -> Dict:
        """上传单个文件的完整流程"""
        file_name = os.path.basename(file_path)
        file_md5 = calculate_md5(file_path)
        
        # 检查是否已上传（除非强制上传）
        if not force and history and history.is_uploaded(file_md5, self.index_id):
            record = history.get_record(file_md5)
            log(f"⏭️ 跳过已上传文件: {file_name} (FileId: {record.get('file_id', 'unknown')})")
            return {
                'file_id': record.get('file_id'),
                'job_id': record.get('job_id'),
                'file_name': file_name,
                'skipped': True,
            }
        
        log(f"开始上传: {file_name}")
        
        # 步骤1: 申请上传租约
        log("申请上传租约...")
        lease_data = self.apply_upload_lease(file_path)
        
        # 步骤2: 上传文件
        log("上传文件到临时存储...")
        self.upload_to_oss(file_path, lease_data)
        
        # 步骤3: 添加文件到数据中心
        log("注册文件到数据中心...")
        file_id = self.add_file(lease_data.file_upload_lease_id, tags)
        log(f"文件注册成功, FileId: {file_id}")
        
        # 步骤4: 等待解析
        if wait_parse:
            log("等待文件解析...")
            self.wait_for_parse(file_id)
            log("文件解析完成")
        
        # 步骤5: 提交到知识库
        log("提交到知识库索引...")
        job_id = self.submit_to_index(file_id)
        log(f"索引任务已提交, JobId: {job_id}")
        
        # 步骤6: 等待索引完成
        if wait_index:
            log("等待索引任务完成...")
            self.wait_for_index(job_id)
            log("索引任务完成")
        
        # 记录上传历史
        if history:
            history.add_record(file_md5, file_id, file_name, self.index_id, job_id)
        
        log(f"✓ 文件上传成功: {file_name}")
        return {"file_id": file_id, "job_id": job_id, "file_name": file_name, "skipped": False}
    
    def upload_folder(
        self,
        folder_path: str,
        recursive: bool = True,
        wait_parse: bool = True,
        wait_index: bool = True,
        tags: Optional[List[str]] = None,
        force: bool = False,
        history: UploadHistory = None,
    ) -> List[Dict]:
        """批量上传文件夹中的文件"""
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")
        
        # 收集所有支持的文件
        files = []
        pattern = "**/*" if recursive else "*"
        for f in folder.glob(pattern):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(str(f))
        
        if not files:
            log("未找到支持的文件", "WARN")
            return []
        
        log(f"找到 {len(files)} 个文件待上传")
        
        results = []
        success_count = 0
        skip_count = 0
        fail_count = 0
        
        try:
            for i, file_path in enumerate(files, 1):
                log(f"\n[{i}/{len(files)}] 处理文件: {os.path.basename(file_path)}")
                try:
                    result = self.upload_single_file(
                        file_path, 
                        wait_parse=wait_parse,
                        wait_index=wait_index,
                        tags=tags,
                        force=force,
                        history=history,
                    )
                    results.append(result)
                    if result.get('skipped'):
                        skip_count += 1
                    else:
                        success_count += 1
                except KeyboardInterrupt:
                    raise  # 让外层处理
                except Exception as e:
                    log(f"✗ 上传失败: {e}", "ERROR")
                    fail_count += 1
                    results.append({"file_name": os.path.basename(file_path), "error": str(e)})
        except KeyboardInterrupt:
            log(f"\n⚠️ 用户中断，已完成 {success_count} 个文件", "WARN")
        
        log(f"\n========== 上传完成 ==========")
        log(f"新上传: {success_count}, 跳过(已存在): {skip_count}, 失败: {fail_count}")
        
        return results


# ============== 主程序 ==============

def get_env_or_arg(arg_value: Optional[str], env_name: str, required: bool = True) -> Optional[str]:
    """获取参数值，优先使用命令行参数，其次环境变量"""
    value = arg_value or os.getenv(env_name)
    if required and not value:
        raise ValueError(f"缺少必需参数: {env_name}，请通过命令行或环境变量提供")
    return value


def main():
    parser = argparse.ArgumentParser(
        description="阿里云百炼 RAG 知识库文件上传工具 v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 上传单个文件
  python upload_data_v2.py --file /path/to/document.pdf

  # 批量上传文件夹
  python upload_data_v2.py --folder /path/to/pdfs/

  # 指定知识库ID
  python upload_data_v2.py --file doc.pdf --index-id xxxx

环境变量:
  ALIBABA_CLOUD_ACCESS_KEY_ID     AccessKey ID
  ALIBABA_CLOUD_ACCESS_KEY_SECRET AccessKey Secret
  WORKSPACE_ID                    业务空间 ID
  INDEX_ID                        知识库 ID
        """
    )
    
    # 文件/文件夹参数（二选一）
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", "-f", help="上传单个文件的路径")
    group.add_argument("--folder", "-d", help="批量上传文件夹路径")
    
    # 认证参数
    parser.add_argument("--access-key-id", help="AccessKey ID")
    parser.add_argument("--access-key-secret", help="AccessKey Secret")
    parser.add_argument("--workspace-id", help="业务空间 ID")
    parser.add_argument("--index-id", help="知识库 ID")
    
    # 可选参数
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help=f"API 端点 (默认: {DEFAULT_ENDPOINT})")
    parser.add_argument("--region", default=DEFAULT_REGION, help=f"区域 (默认: {DEFAULT_REGION})")
    parser.add_argument("--category-id", default=DEFAULT_CATEGORY_ID, help="类目 ID (默认: default)")
    parser.add_argument("--parser", default=DEFAULT_PARSER, help=f"解析器 (默认: {DEFAULT_PARSER})")
    parser.add_argument("--tag", action="append", default=[], help="文件标签 (可多次使用)")
    parser.add_argument("--no-recursive", action="store_true", help="不递归搜索子文件夹")
    parser.add_argument("--skip-parse-wait", action="store_true", help="不等待文件解析完成")
    parser.add_argument("--skip-index-wait", action="store_true", help="不等待索引任务完成")
    parser.add_argument("--force", action="store_true", help="强制重新上传（忽略已上传记录）")
    
    args = parser.parse_args()
    
    # 获取必需参数
    try:
        access_key_id = get_env_or_arg(args.access_key_id, "ALIBABA_CLOUD_ACCESS_KEY_ID")
        access_key_secret = get_env_or_arg(args.access_key_secret, "ALIBABA_CLOUD_ACCESS_KEY_SECRET")
        workspace_id = get_env_or_arg(args.workspace_id, "WORKSPACE_ID")
        index_id = get_env_or_arg(args.index_id, "INDEX_ID")
    except ValueError as e:
        log(str(e), "ERROR")
        sys.exit(1)
    
    # 创建上传器
    uploader = BailianUploader(
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        workspace_id=workspace_id,
        index_id=index_id,
        endpoint=args.endpoint,
        region_id=args.region,
        category_id=args.category_id,
        parser=args.parser,
    )
    
    # 初始化上传历史记录
    history = UploadHistory()
    stats = history.get_stats()
    log(f"已有 {stats['total_files']} 个文件上传记录")
    
    try:
        if args.file:
            # 上传单个文件
            file_path = os.path.abspath(args.file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            uploader.upload_single_file(
                file_path,
                wait_parse=not args.skip_parse_wait,
                wait_index=not args.skip_index_wait,
                tags=args.tag,
                force=args.force,
                history=history,
            )
        else:
            # 批量上传文件夹
            uploader.upload_folder(
                args.folder,
                recursive=not args.no_recursive,
                wait_parse=not args.skip_parse_wait,
                wait_index=not args.skip_index_wait,
                tags=args.tag,
                force=args.force,
                history=history,
            )
        
        log("全部完成！")
        
    except Exception as e:
        log(f"执行失败: {e}", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
