"""
阿里云百炼 RAG 知识库智能上传管理系统

功能：
- 云端文件列表同步：获取知识库已有文件，避免重复上传
- 智能文件名处理：截断时嵌入MD5前缀保证唯一性
- 自动去重：本地历史 + 云端检查双重验证
- 删除重复文件：检测并删除云端重复文件
- 自动同步：扫描memory文件夹，自动上传新文件

使用方式：
    # 同步上传（自动检测新文件并上传）
    python upload_manager.py sync
    
    # 列出云端文件
    python upload_manager.py list
    
    # 检测重复文件
    python upload_manager.py check-duplicates
    
    # 清理重复文件
    python upload_manager.py clean-duplicates --no-dry-run
    
    # 强制刷新云端缓存
    python upload_manager.py refresh
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 设置输出编码为 UTF-8，避免 Windows 批处理中的 GBK 编码错误
if sys.stdout.encoding != 'utf-8':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

import requests
from alibabacloud_bailian20231229 import models as bailian_models
from alibabacloud_bailian20231229.client import Client as BailianClient
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models

# 从项目根目录加载 .env 文件
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    pass


# ============== 配置区域 ==============

SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt', '.md', '.pptx', '.ppt', 
                        '.xlsx', '.xls', '.html', '.png', '.jpg', '.jpeg', '.bmp', '.gif'}

DEFAULT_ENDPOINT = 'bailian.cn-beijing.aliyuncs.com'
DEFAULT_REGION = 'cn-beijing'
DEFAULT_PARSER = 'DASHSCOPE_DOCMIND'
DEFAULT_CATEGORY_ID = 'default'
DEFAULT_CATEGORY_TYPE = 'UNSTRUCTURED'

# 文件名最大长度（API限制128）
MAX_FILENAME_LENGTH = 128
MD5_PREFIX_LENGTH = 10  # 使用MD5前10位作为唯一标识

# 数据文件路径
DATA_DIR = Path(__file__).parent
UPLOAD_HISTORY_FILE = DATA_DIR / '.upload_history.json'
CLOUD_CACHE_FILE = DATA_DIR / '.cloud_files_cache.json'

# Memory 文件夹路径
MEMORY_DIR = Path(__file__).parent.parent / 'memory'


# ============== 工具函数 ==============

def log(msg: str, level: str = "INFO"):
    """打印日志（处理Windows批处理环境的编码问题）"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_msg = f"[{timestamp}] [{level}] {msg}"
    try:
        print(log_msg)
    except UnicodeEncodeError:
        # Windows批处理环境下，移除emoji和特殊字符
        import re
        safe_msg = re.sub(r'[^\x00-\x7F\u4e00-\u9fa5]+', '', msg)  # 只保留ASCII和中文
        print(f"[{timestamp}] [{level}] {safe_msg}")


def calculate_md5(file_path: str) -> str:
    """计算文件 MD5"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def generate_unique_filename(original_name: str, md5: str, max_length: int = MAX_FILENAME_LENGTH) -> str:
    """
    生成唯一文件名，在截断时嵌入MD5前缀保证唯一性
    
    策略：
    - 文件名不超过限制 → 直接使用
    - 超过限制 → [MD5前缀]原始名称截断...扩展名
    
    例如：
    - 原始: "很长很长的文件名.pdf" (超长)
    - 结果: "[a1b2c3d4e5]很长的文件名...pdf"
    """
    if len(original_name) <= max_length:
        return original_name
    
    name, ext = os.path.splitext(original_name)
    md5_prefix = f"[{md5[:MD5_PREFIX_LENGTH]}]"
    
    # 计算可用长度: 总长度 - MD5前缀 - 扩展名 - "..."
    available = max_length - len(md5_prefix) - len(ext) - 3
    if available < 10:
        available = 10
    
    return f"{md5_prefix}{name[:available]}...{ext}"


def extract_md5_from_filename(filename: str) -> Optional[str]:
    """
    从文件名中提取MD5前缀
    例如: "[a1b2c3d4e5]文件名....pdf" -> "a1b2c3d4e5"
    """
    match = re.match(r'^\[([a-f0-9]{10})\]', filename, re.IGNORECASE)
    return match.group(1) if match else None


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


# ============== 本地记录管理 ==============

class LocalRecordManager:
    """管理本地上传历史和云端缓存"""
    
    def __init__(self):
        self.upload_history: Dict[str, Dict] = {}  # key: md5
        self.cloud_cache: Dict[str, Dict] = {}  # key: document_id
        self.cloud_cache_time: Optional[datetime] = None
        self._load()
    
    def _load(self):
        """加载本地记录"""
        if UPLOAD_HISTORY_FILE.exists():
            try:
                with open(UPLOAD_HISTORY_FILE, 'r', encoding='utf-8') as f:
                    self.upload_history = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.upload_history = {}
        
        if CLOUD_CACHE_FILE.exists():
            try:
                with open(CLOUD_CACHE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cloud_cache = data.get('files', {})
                    cache_time = data.get('cache_time')
                    if cache_time:
                        self.cloud_cache_time = datetime.fromisoformat(cache_time)
            except (json.JSONDecodeError, IOError):
                self.cloud_cache = {}
    
    def save_upload_history(self):
        """保存上传历史"""
        with open(UPLOAD_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.upload_history, f, ensure_ascii=False, indent=2)
    
    def save_cloud_cache(self):
        """保存云端缓存"""
        with open(CLOUD_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                'files': self.cloud_cache,
                'cache_time': datetime.now().isoformat(),
            }, f, ensure_ascii=False, indent=2)
    
    def add_upload_record(self, md5: str, file_id: str, document_id: str, 
                          original_filename: str, cloud_filename: str, 
                          index_id: str, file_size: int):
        """添加上传记录"""
        self.upload_history[md5] = {
            'file_id': file_id,
            'document_id': document_id,
            'original_filename': original_filename,
            'cloud_filename': cloud_filename,
            'index_id': index_id,
            'file_size': file_size,
            'upload_time': datetime.now().isoformat(),
        }
        self.save_upload_history()
    
    def is_uploaded(self, md5: str, index_id: str) -> bool:
        """检查文件是否已上传"""
        if md5 in self.upload_history:
            return self.upload_history[md5].get('index_id') == index_id
        return False
    
    def get_upload_record(self, md5: str) -> Optional[Dict]:
        """获取上传记录"""
        return self.upload_history.get(md5)
    
    def update_cloud_cache(self, documents: List[Dict]):
        """更新云端文件缓存"""
        self.cloud_cache = {doc['document_id']: doc for doc in documents}
        self.cloud_cache_time = datetime.now()
        self.save_cloud_cache()
    
    def is_cache_valid(self, max_age_hours: int = 1) -> bool:
        """检查缓存是否有效"""
        if not self.cloud_cache_time:
            return False
        return datetime.now() - self.cloud_cache_time < timedelta(hours=max_age_hours)
    
    def find_in_cloud_by_md5(self, md5_prefix: str) -> List[Dict]:
        """通过MD5前缀在云端缓存中查找文件"""
        results = []
        for doc in self.cloud_cache.values():
            filename = doc.get('name', '')
            extracted_md5 = extract_md5_from_filename(filename)
            if extracted_md5 and extracted_md5.lower() == md5_prefix.lower():
                results.append(doc)
        return results
    
    def find_duplicates(self) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
        """
        查找重复文件
        返回: (基于MD5前缀的重复, 基于完全相同文件名的重复)
        """
        md5_groups: Dict[str, List[Dict]] = {}
        name_groups: Dict[str, List[Dict]] = {}
        
        for doc in self.cloud_cache.values():
            filename = doc.get('name', '')
            
            # 按MD5前缀分组
            md5_prefix = extract_md5_from_filename(filename)
            if md5_prefix:
                if md5_prefix not in md5_groups:
                    md5_groups[md5_prefix] = []
                md5_groups[md5_prefix].append(doc)
            
            # 按完整文件名分组
            if filename:
                if filename not in name_groups:
                    name_groups[filename] = []
                name_groups[filename].append(doc)
        
        md5_duplicates = {k: v for k, v in md5_groups.items() if len(v) > 1}
        name_duplicates = {k: v for k, v in name_groups.items() if len(v) > 1}
        
        return md5_duplicates, name_duplicates
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'local_records': len(self.upload_history),
            'cloud_files': len(self.cloud_cache),
            'cache_time': self.cloud_cache_time.isoformat() if self.cloud_cache_time else None,
        }


# ============== 云端操作管理 ==============

class CloudManager:
    """管理云端知识库操作"""
    
    def __init__(self, access_key_id: str, access_key_secret: str, 
                 workspace_id: str, index_id: str,
                 endpoint: str = DEFAULT_ENDPOINT, region_id: str = DEFAULT_REGION):
        self.workspace_id = workspace_id
        self.index_id = index_id
        
        config = open_api_models.Config(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            region_id=region_id,
            endpoint=endpoint,
        )
        self.client = BailianClient(config)
        self.runtime = util_models.RuntimeOptions()
    
    def list_index_documents(self, page_size: int = 100) -> List[Dict]:
        """获取知识库中的所有文档列表"""
        all_documents = []
        page_number = 1
        
        while True:
            try:
                request = bailian_models.ListIndexDocumentsRequest(
                    index_id=self.index_id,
                    page_number=page_number,
                    page_size=page_size,
                )
                response = self.client.list_index_documents_with_options(
                    self.workspace_id, request, {}, self.runtime
                )
                ensure_api_success("ListIndexDocuments", response)
                
                data = response.body.data
                if not data or not data.documents:
                    break
                
                for doc in data.documents:
                    all_documents.append({
                        'document_id': doc.id,
                        'name': doc.name,
                        'size': doc.size,
                        'status': doc.status,
                        'doc_type': getattr(doc, 'doc_type', None),
                    })
                
                total = data.total_count or 0
                if page_number * page_size >= total:
                    break
                
                page_number += 1
                
            except Exception as e:
                log(f"获取文档列表失败: {e}", "ERROR")
                break
        
        return all_documents
    
    def delete_document(self, document_id: str) -> bool:
        """从知识库删除文档"""
        try:
            request = bailian_models.DeleteIndexDocumentRequest(
                index_id=self.index_id,
                document_ids=[document_id],
            )
            response = self.client.delete_index_document_with_options(
                self.workspace_id, request, {}, self.runtime
            )
            ensure_api_success("DeleteIndexDocument", response)
            return True
        except Exception as e:
            log(f"删除文档失败: {e}", "ERROR")
            return False
    
    def apply_upload_lease(self, file_path: str, cloud_filename: str) -> Dict:
        """申请文件上传租约"""
        file_size = os.path.getsize(file_path)
        file_md5 = calculate_md5(file_path)
        
        request = bailian_models.ApplyFileUploadLeaseRequest(
            category_type=DEFAULT_CATEGORY_TYPE,
            file_name=cloud_filename,
            md_5=file_md5,
            size_in_bytes=str(file_size),
        )
        
        response = self.client.apply_file_upload_lease_with_options(
            DEFAULT_CATEGORY_ID, self.workspace_id, request, {}, self.runtime
        )
        ensure_api_success("ApplyFileUploadLease", response)
        
        data = response.body.data
        return {
            'lease_id': data.file_upload_lease_id,
            'url': data.param.url,
            'method': data.param.method,
            'headers': data.param.headers,
        }
    
    def upload_to_oss(self, file_path: str, lease_data: Dict, max_retries: int = 3) -> None:
        """上传文件到临时存储"""
        from urllib.parse import quote
        
        headers = {}
        if lease_data.get('headers'):
            for k, v in lease_data['headers'].items():
                if v is None:
                    continue
                key = str(k)
                val = str(v)
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
                        method=lease_data['method'].upper(),
                        url=lease_data['url'],
                        data=f,
                        headers=headers,
                        timeout=600,
                    )
                
                if response.status_code < 200 or response.status_code >= 300:
                    raise RuntimeError(f"HTTP {response.status_code}")
                return
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    log(f"上传失败，重试 {attempt + 2}/{max_retries}...", "WARN")
                    time.sleep(2 ** attempt)
        
        raise RuntimeError(f"文件上传失败: {last_error}")
    
    def add_file_to_datacenter(self, lease_id: str, tags: List[str] = None) -> str:
        """将文件添加到数据中心"""
        request = bailian_models.AddFileRequest(
            category_id=DEFAULT_CATEGORY_ID,
            category_type=DEFAULT_CATEGORY_TYPE,
            lease_id=lease_id,
            parser=DEFAULT_PARSER,
            tags=tags or [],
        )
        
        response = self.client.add_file_with_options(
            self.workspace_id, request, {}, self.runtime
        )
        ensure_api_success("AddFile", response)
        
        return response.body.data.file_id
    
    def wait_for_parse(self, file_id: str, timeout: int = 600, interval: int = 5) -> None:
        """等待文件解析完成"""
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            response = self.client.describe_file(self.workspace_id, file_id)
            ensure_api_success("DescribeFile", response)
            
            status = response.body.data.status
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
        
        response = self.client.submit_index_add_documents_job_with_options(
            self.workspace_id, request, {}, self.runtime
        )
        ensure_api_success("SubmitIndexAddDocumentsJob", response)
        
        return response.body.data.id
    
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
            response = self.client.get_index_job_status_with_options(
                self.workspace_id, request, {}, self.runtime
            )
            ensure_api_success("GetIndexJobStatus", response)
            
            status = response.body.data.status
            if status == "COMPLETED":
                return
            if status == "FAILED":
                raise RuntimeError("索引任务失败")
            
            time.sleep(interval)
        
        raise TimeoutError("等待索引任务超时")


# ============== 智能上传管理器 ==============

class SmartUploadManager:
    """智能上传管理器 - 整合本地和云端管理"""
    
    def __init__(self, access_key_id: str, access_key_secret: str,
                 workspace_id: str, index_id: str):
        self.index_id = index_id
        self.local = LocalRecordManager()
        self.cloud = CloudManager(access_key_id, access_key_secret, 
                                   workspace_id, index_id)
    
    def refresh_cloud_cache(self) -> int:
        """刷新云端文件缓存"""
        log("正在获取云端文件列表...")
        documents = self.cloud.list_index_documents()
        self.local.update_cloud_cache(documents)
        log(f"已缓存 {len(documents)} 个云端文件")
        return len(documents)
    
    def ensure_cache_valid(self):
        """确保缓存有效，必要时刷新"""
        if not self.local.is_cache_valid():
            self.refresh_cloud_cache()
    
    def is_file_in_cloud(self, file_path: str) -> Tuple[bool, Optional[Dict]]:
        """
        检查文件是否已在云端
        返回: (是否存在, 云端文件信息)
        """
        self.ensure_cache_valid()
        
        md5 = calculate_md5(file_path)
        md5_prefix = md5[:MD5_PREFIX_LENGTH]
        original_name = os.path.basename(file_path)
        
        # 1. 检查本地上传记录（基于完整MD5）
        local_record = self.local.get_upload_record(md5)
        if local_record and local_record.get('index_id') == self.index_id:
            return True, local_record
        
        # 2. 在云端缓存中查找（通过MD5前缀）
        cloud_matches = self.local.find_in_cloud_by_md5(md5_prefix)
        if cloud_matches:
            return True, cloud_matches[0]
        
        # 3. 检查原始文件名是否存在（针对早期上传的文件）
        for doc in self.local.cloud_cache.values():
            if doc.get('name') == original_name:
                return True, doc
        
        return False, None
    
    def upload_file(self, file_path: str, wait_parse: bool = True, 
                    wait_index: bool = True, force: bool = False) -> Dict:
        """
        智能上传单个文件
        """
        original_name = os.path.basename(file_path)
        file_md5 = calculate_md5(file_path)
        file_size = os.path.getsize(file_path)
        
        # 生成云端文件名（带MD5前缀以保证唯一性）
        cloud_filename = generate_unique_filename(original_name, file_md5)
        
        # 检查是否已上传
        if not force:
            exists, existing_doc = self.is_file_in_cloud(file_path)
            if exists:
                log(f"⏭️ 跳过已存在: {original_name}")
                return {
                    'status': 'skipped',
                    'reason': 'already_exists',
                    'original_name': original_name,
                    'cloud_name': existing_doc.get('name') if existing_doc else None,
                    'document_id': existing_doc.get('document_id') if existing_doc else None,
                }
        
        log(f"📤 上传: {original_name}")
        if cloud_filename != original_name:
            log(f"   云端名称: {cloud_filename}")
        
        try:
            # 申请上传租约
            lease_data = self.cloud.apply_upload_lease(file_path, cloud_filename)
            
            # 上传到临时存储
            log("   上传到临时存储...")
            self.cloud.upload_to_oss(file_path, lease_data)
            
            # 添加到数据中心
            log("   注册到数据中心...")
            file_id = self.cloud.add_file_to_datacenter(lease_data['lease_id'])
            
            # 等待解析
            if wait_parse:
                log("   等待解析...")
                self.cloud.wait_for_parse(file_id)
            
            # 提交到索引
            log("   提交到索引...")
            job_id = self.cloud.submit_to_index(file_id)
            
            # 等待索引完成
            if wait_index:
                log("   等待索引...")
                self.cloud.wait_for_index(job_id)
            
            # 记录上传信息
            self.local.add_upload_record(
                md5=file_md5,
                file_id=file_id,
                document_id=file_id,
                original_filename=original_name,
                cloud_filename=cloud_filename,
                index_id=self.index_id,
                file_size=file_size,
            )
            
            log(f"✅ 上传成功: {original_name}")
            
            return {
                'status': 'uploaded',
                'original_name': original_name,
                'cloud_name': cloud_filename,
                'file_id': file_id,
                'md5': file_md5,
            }
            
        except Exception as e:
            log(f"❌ 上传失败: {original_name} - {e}", "ERROR")
            return {
                'status': 'failed',
                'original_name': original_name,
                'error': str(e),
            }
    
    def sync_folder(self, folder_path: Path, wait_parse: bool = True,
                    wait_index: bool = True, force: bool = False) -> Dict:
        """
        同步文件夹到知识库
        """
        if not folder_path.exists():
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")
        
        # 收集所有支持的文件
        files = []
        for f in folder_path.rglob("*"):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(f)
        
        log(f"📁 扫描到 {len(files)} 个文件")
        
        # 刷新云端缓存
        self.refresh_cloud_cache()
        
        results = {
            'uploaded': [],
            'skipped': [],
            'failed': [],
        }
        
        for i, file_path in enumerate(files, 1):
            log(f"\n[{i}/{len(files)}] 处理: {file_path.name}")
            
            try:
                result = self.upload_file(
                    str(file_path),
                    wait_parse=wait_parse,
                    wait_index=wait_index,
                    force=force,
                )
                
                if result['status'] == 'uploaded':
                    results['uploaded'].append(result)
                elif result['status'] == 'skipped':
                    results['skipped'].append(result)
                else:
                    results['failed'].append(result)
                    
            except KeyboardInterrupt:
                log("\n⚠️ 用户中断", "WARN")
                break
            except Exception as e:
                log(f"❌ 错误: {e}", "ERROR")
                results['failed'].append({
                    'original_name': file_path.name,
                    'error': str(e),
                })
        
        # 打印汇总
        log(f"\n{'='*50}")
        log(f"📊 同步完成:")
        log(f"   新上传: {len(results['uploaded'])}")
        log(f"   已跳过: {len(results['skipped'])}")
        log(f"   失败: {len(results['failed'])}")
        
        return results
    
    def find_duplicates(self) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
        """查找云端重复文件"""
        self.ensure_cache_valid()
        return self.local.find_duplicates()
    
    def clean_duplicates(self, dry_run: bool = True, clean_name_dups: bool = True) -> List[Dict]:
        """
        清理重复文件（保留最早上传的）
        dry_run: 如果为True，只显示要删除的文件，不实际删除
        clean_name_dups: 是否清理同名重复文件
        """
        md5_duplicates, name_duplicates = self.find_duplicates()
        
        if not md5_duplicates and not name_duplicates:
            log("✅ 没有发现重复文件")
            return []
        
        to_delete = []
        
        # 处理MD5前缀重复
        if md5_duplicates:
            log(f"\n🔍 发现 {len(md5_duplicates)} 组 MD5 重复文件:")
            for md5_prefix, docs in md5_duplicates.items():
                sorted_docs = sorted(docs, key=lambda x: x.get('document_id', ''))
                keep = sorted_docs[0]
                delete_list = sorted_docs[1:]
                
                log(f"\n  [{md5_prefix}] ({len(docs)} 个):")
                log(f"   保留: {keep.get('name')}")
                
                for doc in delete_list:
                    log(f"   删除: {doc.get('name')}")
                    to_delete.append(doc)
        
        # 处理同名重复
        if name_duplicates and clean_name_dups:
            log(f"\n🔍 发现 {len(name_duplicates)} 组同名重复文件:")
            for filename, docs in name_duplicates.items():
                # 排除已被MD5重复标记的
                remaining = [d for d in docs if d not in to_delete and d != docs[0]]
                if not remaining:
                    continue
                
                sorted_docs = sorted(docs, key=lambda x: x.get('document_id', ''))
                keep = sorted_docs[0]
                delete_list = sorted_docs[1:]
                
                log(f"\n  \"{filename}\" ({len(docs)} 个):")
                log(f"   保留: ID={keep.get('document_id')}")
                
                for doc in delete_list:
                    if doc not in to_delete:
                        log(f"   删除: ID={doc.get('document_id')}")
                        to_delete.append(doc)
        
        if not to_delete:
            log("✅ 没有需要清理的重复文件")
            return []
        
        if dry_run:
            log(f"\n⚠️ 试运行模式，共 {len(to_delete)} 个文件待删除")
            log("使用 --no-dry-run 执行实际删除")
        else:
            log(f"\n🗑️ 开始删除 {len(to_delete)} 个重复文件...")
            for doc in to_delete:
                if self.cloud.delete_document(doc['document_id']):
                    log(f"   ✅ 已删除: {doc.get('name')} (ID: {doc.get('document_id')})")
                else:
                    log(f"   ❌ 删除失败: {doc.get('name')}")
            
            # 刷新缓存
            self.refresh_cloud_cache()
        
        return to_delete
    
    def list_cloud_files(self) -> List[Dict]:
        """列出云端所有文件"""
        self.refresh_cloud_cache()
        return list(self.local.cloud_cache.values())


# ============== 命令行接口 ==============

def get_config():
    """获取配置"""
    return {
        'access_key_id': os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID'),
        'access_key_secret': os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
        'workspace_id': os.getenv('WORKSPACE_ID'),
        'index_id': os.getenv('INDEX_ID'),
    }


def cmd_sync(args):
    """同步命令"""
    config = get_config()
    manager = SmartUploadManager(**config)
    
    folder = Path(args.folder) if args.folder else MEMORY_DIR
    
    manager.sync_folder(
        folder,
        wait_parse=not args.skip_parse,
        wait_index=not args.skip_index,
        force=args.force,
    )


def cmd_list(args):
    """列出云端文件"""
    config = get_config()
    manager = SmartUploadManager(**config)
    
    files = manager.list_cloud_files()
    
    log(f"\n📚 云端共 {len(files)} 个文件:\n")
    for i, doc in enumerate(files, 1):
        status = doc.get('status', 'UNKNOWN')
        status_icon = '✅' if status == 'FINISH' else '⏳' if status == 'RUNNING' else '❓'
        size_kb = (doc.get('size') or 0) / 1024
        print(f"{i:3}. {status_icon} {doc.get('name', 'N/A')} ({size_kb:.1f} KB)")


def cmd_check_duplicates(args):
    """检查重复文件"""
    config = get_config()
    manager = SmartUploadManager(**config)
    
    manager.refresh_cloud_cache()
    md5_duplicates, name_duplicates = manager.find_duplicates()
    
    if not md5_duplicates and not name_duplicates:
        log("✅ 没有发现重复文件")
        return
    
    # MD5前缀重复
    if md5_duplicates:
        total = sum(len(v) for v in md5_duplicates.values())
        log(f"\n🔍 基于 MD5 前缀发现 {len(md5_duplicates)} 组重复，共 {total} 个文件:")
        for md5_prefix, docs in md5_duplicates.items():
            log(f"\n  [{md5_prefix}] ({len(docs)} 个):")
            for doc in docs:
                print(f"     - {doc.get('name')} (ID: {doc.get('document_id')[:12]}...)")
    
    # 同名重复
    if name_duplicates:
        total = sum(len(v) for v in name_duplicates.values())
        log(f"\n🔍 基于完全相同文件名发现 {len(name_duplicates)} 组重复，共 {total} 个文件:")
        for filename, docs in name_duplicates.items():
            log(f"\n  \"{filename}\" ({len(docs)} 个):")
            for doc in docs:
                print(f"     - ID: {doc.get('document_id')}")


def cmd_clean_duplicates(args):
    """清理重复文件"""
    config = get_config()
    manager = SmartUploadManager(**config)
    
    manager.clean_duplicates(dry_run=args.dry_run)


def cmd_refresh(args):
    """刷新云端缓存"""
    config = get_config()
    manager = SmartUploadManager(**config)
    manager.refresh_cloud_cache()


def cmd_status(args):
    """显示状态"""
    config = get_config()
    manager = SmartUploadManager(**config)
    
    stats = manager.local.get_stats()
    log(f"\n📊 上传管理状态:")
    log(f"   本地上传记录: {stats['local_records']} 个")
    log(f"   云端文件缓存: {stats['cloud_files']} 个")
    log(f"   缓存更新时间: {stats['cache_time'] or '未缓存'}")


def main():
    parser = argparse.ArgumentParser(
        description="阿里云百炼 RAG 知识库智能上传管理系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python upload_manager.py sync                 # 同步 memory 文件夹
  python upload_manager.py sync -f /path/to    # 同步指定文件夹
  python upload_manager.py list                 # 列出云端文件
  python upload_manager.py check-duplicates     # 检查重复文件
  python upload_manager.py clean-duplicates     # 清理重复（试运行）
  python upload_manager.py clean-duplicates --no-dry-run  # 实际删除
  python upload_manager.py refresh              # 刷新云端缓存
  python upload_manager.py status               # 显示状态
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # sync 命令
    sync_parser = subparsers.add_parser('sync', help='同步文件夹到知识库')
    sync_parser.add_argument('--folder', '-f', help=f'文件夹路径 (默认: {MEMORY_DIR})')
    sync_parser.add_argument('--force', action='store_true', help='强制重新上传所有文件')
    sync_parser.add_argument('--skip-parse', action='store_true', help='不等待解析完成')
    sync_parser.add_argument('--skip-index', action='store_true', help='不等待索引完成')
    sync_parser.set_defaults(func=cmd_sync)
    
    # list 命令
    list_parser = subparsers.add_parser('list', help='列出云端所有文件')
    list_parser.set_defaults(func=cmd_list)
    
    # check-duplicates 命令
    check_parser = subparsers.add_parser('check-duplicates', help='检查重复文件')
    check_parser.set_defaults(func=cmd_check_duplicates)
    
    # clean-duplicates 命令
    clean_parser = subparsers.add_parser('clean-duplicates', help='清理重复文件')
    clean_parser.add_argument('--no-dry-run', dest='dry_run', action='store_false', 
                              help='实际执行删除（默认为试运行）')
    clean_parser.set_defaults(func=cmd_clean_duplicates, dry_run=True)
    
    # refresh 命令
    refresh_parser = subparsers.add_parser('refresh', help='刷新云端文件缓存')
    refresh_parser.set_defaults(func=cmd_refresh)
    
    # status 命令
    status_parser = subparsers.add_parser('status', help='显示状态')
    status_parser.set_defaults(func=cmd_status)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 检查配置
    config = get_config()
    missing = [k for k, v in config.items() if not v]
    if missing:
        log(f"缺少配置: {', '.join(missing)}", "ERROR")
        log("请在 .env 文件中设置这些环境变量")
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
