#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中金研究院报告获取模块 (CICC Research)

通过浏览器 Cookies 认证访问 https://www.research.cicc.com
用户需要:
  1. 在浏览器中手动登录 (需关闭 hkproxy)
  2. 导出 Cookies 保存到 cicc_cookies.json
  3. 本模块使用这些 Cookies 访问研究报告 API

Cookies 获取方式:
  - Chrome 扩展: EditThisCookie / Cookie-Editor
  - 浏览器 DevTools → Application → Cookies → 复制全部
  - 或通过本模块提供的前端界面粘贴 Cookies
"""

import json
import os
import re
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import requests

logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────────────────────────────────

BASE_URL = "https://www.research.cicc.com"
COOKIES_FILE = os.path.join(os.path.dirname(__file__), "cicc_cookies.json")
CACHE_TTL = 300  # 缓存 5 分钟

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": f"{BASE_URL}/zh_CN/reportList",
    "Origin": BASE_URL,
    "sec-ch-ua": '"Chromium";v="131", "Google Chrome";v="131", "Not_A Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-site": "same-origin",
    "sec-fetch-mode": "cors",
    "sec-fetch-dest": "empty",
}

# 已验证的 CICC Research API 路径 (real API discovered from JS chunks)
# GET 端点 (pageNum, pageSize 参数)
API_HOT_REPORTS = "/reports/api/hot/fetchHotReport"         # 热门研报 (分页)
API_BROWSING = "/reports/api/fetchReportBrowsing"           # 最新研报 (分页)
API_FOCUS_LIST = "/focus/api/fetchFocusReportList"          # 晨会焦点系列
API_FOCUS_LATEST = "/focus/api/fetchLastFocusReport"        # 最新晨会焦点
API_RECOMMEND = "/feed/api/v3/reports-activities/recommendList"  # 推荐研报
API_HOME_HOT = "/reports/api/hot/getHomeHotReport"          # 首页热门
API_DETAIL = "/reports/api/v3/detail"                       # 报告详情
# POST 端点 (portalCategoryId+industriesIds+size+page，经 JS 反向工程验证)
API_PAGE = "/reports/api/v3/page"                           # 分类全量研报列表


def _get_x_time() -> str:
    """生成 X-Time 鉴权头，与 JS 源码中 g() 函数逻辑一致。

    JS 原文 (time.e1655a35.js):
        g=()=>{const u=Math.floor(Date.now()/10),n=u%97,
               s=String(n).padStart(2,"0");return String(u)+s}
    """
    u = int(time.time() * 1000) // 10  # Math.floor(Date.now() / 10)
    n = u % 97
    return str(u) + str(n).zfill(2)

# ── 报告分类映射 (从 portalCategoryIds 字段反向工程得出) ────────────────────────
# 顶级分类 ID 与中文名对照（网站 UI 一致）
CATEGORY_MAP: Dict[int, str] = {
    51: "宏观经济",
    52: "市场策略",
    53: "行业研究",
    54: "公司研究",
    55: "大宗商品",
    57: "晨会焦点",
    65: "ESG",
    66: "全球外汇",
    68: "全球研究",
}
# 子分类 → 父分类映射（用于向上归类过滤）
_SUB_TO_PARENT: Dict[int, int] = {
    # 宏观经济 (51)
    101: 51, 102: 51, 103: 51, 106: 51,
    # 市场策略 (52)
    151: 52, 152: 52, 153: 52, 155: 52,
    # 行业研究 (53)
    202: 53, 1058: 53,
    # 公司研究 (54)
    252: 54,
    # 大宗商品 (55)
    721: 55, 722: 55, 723: 55, 724: 55, 725: 55,
    3104: 55, 3109: 55, 3209: 55, 3309: 55, 3409: 55, 3501: 55,
    # 晨会焦点 (57)
    401: 57,
    # ESG (65)
    704: 65, 676: 65,
    # 全球外汇 (66)
    689: 66, 694: 66,
    2001: 66, 2002: 66, 2003: 66, 2004: 66,
    2101: 66, 2102: 66, 2103: 66, 2104: 66,
    # 全球研究 (68)
    717: 68,
}


def get_categories() -> List[dict]:
    """返回可用分类列表（含 id 和 name）。"""
    return [{"id": k, "name": v} for k, v in CATEGORY_MAP.items()]


def _fetch_page_api(
    cookies: dict,
    category_id: int = 0,
    page: int = 1,
    page_size: int = 50,
    keyword: str = "",
) -> dict:
    """POST /reports/api/v3/page — 分类全量研报列表。

    关键发现：
    - portalCategoryId 必须是字符串，不能是整数
    - industriesIds 必须是空字符串 ""，不能是 []
    - X-Time 头必须按 JS 公式计算

    Returns:
        {"ok": bool, "content": [...], "total": int, "total_pages": int,
         "page": int, "page_size": int, "error": str|None}
    """
    payload = {
        "portalCategoryId": str(category_id) if category_id else "",
        "industriesIds": "",
        "size": page_size,
        "page": page,
        "input": keyword or "",
        "currencyIds": "",
        "commodityType": "",
        "authorId": "",
        "secCode": "",
        "minPageCount": "",
        "maxPageCount": "",
    }
    headers = {
        **HEADERS,
        "Content-Type": "application/json;charset=UTF-8",
        "X-Time": _get_x_time(),
        "Referer": f"{BASE_URL}/zh_CN/reportList?entrance_source=Header-Menu",
    }
    try:
        sess = requests.Session()
        sess.trust_env = False
        sess.proxies = {}
        sess.cookies.update(cookies)
        r = sess.post(BASE_URL + API_PAGE, json=payload, headers=headers, timeout=20)
        if r.status_code != 200:
            return {"ok": False, "content": [], "total": 0, "total_pages": 0,
                    "page": page, "page_size": page_size,
                    "error": f"HTTP {r.status_code}: {r.text[:200]}"}
        d = r.json()
        if d.get("code") != 0:
            return {"ok": False, "content": [], "total": 0, "total_pages": 0,
                    "page": page, "page_size": page_size,
                    "error": f"API code={d.get('code')}: {d.get('desc','')[:200]}"}
        data = d.get("data", {})
        return {
            "ok": True,
            "content": data.get("content", []),
            "total": data.get("totalElements", 0),
            "total_pages": data.get("totalPages", 1),
            "page": page,
            "page_size": page_size,
            "error": None,
        }
    except Exception as e:
        logger.error("_fetch_page_api failed: %s", e)
        return {"ok": False, "content": [], "total": 0, "total_pages": 0,
                "page": page, "page_size": page_size, "error": str(e)}


def _parse_page_api_item(item: dict) -> Optional[dict]:
    """Parse a single item from /reports/api/v3/page response."""
    report_id = str(item.get("reportId") or item.get("id") or "")
    title = str(item.get("title") or "")
    if not title or not report_id:
        return None

    # Authors: list of {id, name} dicts
    analysts = item.get("analysts", [])
    if isinstance(analysts, list):
        authors = ", ".join(
            a.get("name", a.get("analystName", "")) for a in analysts if isinstance(a, dict)
        ).strip(", ")
    else:
        authors = str(analysts)

    # Date
    pub_time = item.get("publishTime", "")
    if pub_time and isinstance(pub_time, str) and "T" in pub_time:
        date_val = pub_time[:10]  # "2026-03-16T06:11:10Z" → "2026-03-16"
    elif isinstance(pub_time, (int, float)) and pub_time > 1e12:
        date_val = datetime.fromtimestamp(pub_time / 1000).strftime("%Y-%m-%d")
    else:
        date_val = str(pub_time)[:10] if pub_time else ""

    abstract_val = str(item.get("reportAbstract") or item.get("summary") or "")[:800]

    portal_ids_raw = item.get("portalCategoryIds", [])
    portal_ids = [int(x) for x in portal_ids_raw if str(x).isdigit()] if isinstance(portal_ids_raw, list) else []

    return {
        "id": report_id,
        "title": title,
        "authors": authors,
        "date": date_val,
        "category": str(item.get("reportType") or ""),
        "abstract": abstract_val,
        "url": f"{BASE_URL}/zh_CN/report/detail/{report_id}",
        "portal_category_ids": portal_ids,
    }


# 旧式候选路径 (保留兼容，未验证)
API_CANDIDATES = [
    API_HOT_REPORTS,
    API_BROWSING,
    API_FOCUS_LIST,
]

DETAIL_CANDIDATES = [
    API_DETAIL,
    "/reports/api/v3/detail",
    "/frontend/report/detail",
]

# ── 内部缓存 ──────────────────────────────────────────────────────────────────

_cache: Dict[str, Any] = {}
_cache_time: Dict[str, float] = {}
_discovered_list_api: Optional[str] = None
_discovered_detail_api: Optional[str] = None


# ── Cookie 管理 ───────────────────────────────────────────────────────────────

def load_cookies() -> Dict[str, str]:
    """从本地文件加载 cookies。支持两种格式:
    1. EditThisCookie 导出的列表格式: [{"name":"x","value":"y",...}, ...]
    2. 简单 dict 格式: {"cookie_name": "cookie_value", ...}
    """
    if not os.path.exists(COOKIES_FILE):
        return {}
    try:
        with open(COOKIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 格式 1: EditThisCookie 数组
        if isinstance(data, list):
            cookies = {}
            for item in data:
                name = item.get("name", "")
                value = item.get("value", "")
                if name:
                    cookies[name] = value
            return cookies

        # 格式 2: 简易 dict
        if isinstance(data, dict):
            return {k: str(v) for k, v in data.items()}

        return {}
    except Exception as e:
        logger.error("加载 cookies 失败: %s", e)
        return {}


def save_cookies(cookies_input) -> Dict[str, str]:
    """保存 cookies 到本地文件。
    
    接受:
      - dict: {"name": "value", ...}
      - list: EditThisCookie 格式 [{"name":"x","value":"y",...}, ...]
      - str:  浏览器复制的 cookie 字符串 "name1=val1; name2=val2; ..."
    """
    if isinstance(cookies_input, str):
        # 解析 cookie 字符串
        cookies = {}
        for part in cookies_input.split(";"):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                cookies[k.strip()] = v.strip()
        data_to_save = cookies
    elif isinstance(cookies_input, list):
        # EditThisCookie 格式，直接保存
        data_to_save = cookies_input
        cookies = {item.get("name", ""): item.get("value", "") for item in cookies_input if item.get("name")}
    elif isinstance(cookies_input, dict):
        data_to_save = cookies_input
        cookies = cookies_input
    else:
        raise ValueError(f"不支持的 cookies 格式: {type(cookies_input)}")

    with open(COOKIES_FILE, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)

    # 清除缓存
    _cache.clear()
    _cache_time.clear()
    global _discovered_list_api, _discovered_detail_api
    _discovered_list_api = None
    _discovered_detail_api = None

    logger.info("已保存 %d 个 cookies 到 %s", len(cookies), COOKIES_FILE)
    return cookies


def get_cookies_status() -> Dict[str, Any]:
    """返回 cookies 状态信息。"""
    cookies = load_cookies()
    exists = os.path.exists(COOKIES_FILE)
    mtime = None
    if exists:
        mtime = datetime.fromtimestamp(os.path.getmtime(COOKIES_FILE)).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "has_cookies": len(cookies) > 0,
        "cookie_count": len(cookies),
        "file_exists": exists,
        "file_path": COOKIES_FILE,
        "last_updated": mtime,
        "cookie_names": list(cookies.keys())[:20],  # 最多显示 20 个名称
    }


# ── HTTP 请求 (禁用代理，直连 CICC) ─────────────────────────────────────────

def _make_session(cookies: Dict[str, str]) -> requests.Session:
    """创建禁用代理的 requests Session。"""
    s = requests.Session()
    s.headers.update(HEADERS)
    s.cookies.update(cookies)
    # 关键: 完全禁用系统代理，直连 CICC (trust_env=False 才能真正绕过 hkproxy)
    s.trust_env = False
    s.proxies = {"http": None, "https": None}
    s.verify = True
    return s


def _get(url: str, params: dict = None, cookies: dict = None) -> Optional[dict]:
    """GET 请求，返回 JSON 或 None。"""
    if cookies is None:
        cookies = load_cookies()
    if not cookies:
        return None

    sess = _make_session(cookies)
    try:
        resp = sess.get(url, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        logger.warning("HTTP 错误 %s: %s", url, e)
        return None
    except Exception as e:
        logger.warning("请求失败 %s: %s", url, e)
        return None


def _post(url: str, payload: dict = None, cookies: dict = None) -> Optional[dict]:
    """POST 请求，返回 JSON 或 None。"""
    if cookies is None:
        cookies = load_cookies()
    if not cookies:
        return None

    sess = _make_session(cookies)
    sess.headers["Content-Type"] = "application/json"
    try:
        resp = sess.post(url, json=payload or {}, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        logger.warning("HTTP POST 错误 %s: %s", url, e)
        return None
    except Exception as e:
        logger.warning("POST 请求失败 %s: %s", url, e)
        return None


# ── API 端点自动探测 ──────────────────────────────────────────────────────────

def _discover_list_api(cookies: dict) -> Optional[str]:
    """尝试多个候选 URL，找到第一个返回有效 JSON 的报告列表 API。"""
    global _discovered_list_api
    if _discovered_list_api:
        return _discovered_list_api

    for path in API_CANDIDATES:
        url = BASE_URL + path
        logger.info("探测报告列表 API: %s", url)
        # 尝试 GET
        result = _get(url, params={"pageNum": 1, "pageSize": 5}, cookies=cookies)
        if result and _looks_like_report_list(result):
            _discovered_list_api = path
            logger.info("发现报告列表 API: %s (GET)", path)
            return path

        # 尝试 POST
        result = _post(url, payload={"pageNum": 1, "pageSize": 5}, cookies=cookies)
        if result and _looks_like_report_list(result):
            _discovered_list_api = path
            logger.info("发现报告列表 API: %s (POST)", path)
            return path

    logger.warning("未能自动发现报告列表 API")
    return None


def _discover_detail_api(cookies: dict) -> Optional[str]:
    """探测报告详情 API。"""
    global _discovered_detail_api
    if _discovered_detail_api:
        return _discovered_detail_api

    for path in DETAIL_CANDIDATES:
        url = BASE_URL + path
        # 简单检查端点是否存在 (用一个假 ID)
        result = _get(url, params={"id": "test"}, cookies=cookies)
        if result is not None:  # 能返回 JSON 就记录
            _discovered_detail_api = path
            logger.info("发现报告详情 API: %s", path)
            return path

    return None


def _looks_like_report_list(data: Any) -> bool:
    """启发式判断返回数据是否像报告列表。"""
    if not isinstance(data, dict):
        return False

    # 常见模式: {"code":0, "data": {"list":[...], "total":N}}
    # 或: {"code":"200", "data": [...]}
    # 或: {"success":true, "result": {"records": [...]}}
    
    # 递归查找包含列表的部分
    def find_list(obj, depth=0):
        if depth > 3:
            return False
        if isinstance(obj, list) and len(obj) > 0:
            # 检查列表中的元素是否像报告
            item = obj[0]
            if isinstance(item, dict):
                keys_lower = {k.lower() for k in item.keys()}
                report_keys = {"title", "id", "name", "date", "author", "publishdate",
                               "createtime", "reportid", "reporttype", "abstract", "summary"}
                if keys_lower & report_keys:
                    return True
            return False
        if isinstance(obj, dict):
            for v in obj.values():
                if find_list(v, depth + 1):
                    return True
        return False

    return find_list(data)


# ── 全源聚合（最新研报）────────────────────────────────────────────────────────

def _fetch_aggregated_latest(cookies: dict) -> List[dict]:
    """从 hot、focus（多页）、recommend 三个来源聚合全量报告，按日期降序去重。

    策略：
    - Hot:      请求第 1 页 (最多 20 条)
    - Focus:    请求前 3 页焦点组 (每页 20 个组，展开子报告，可得约 100+ 篇)
    - Recommend:请求第 1 页 (18 条)
    全部按 reportId 去重后按 publishTime 倒序排列。
    """
    seen: Dict[str, dict] = {}  # reportId -> parsed report dict

    def _add_item(item: dict, src: str):
        """解析并加入 seen 字典，按 reportId 去重，保留日期较新的。"""
        parsed = _parse_single_report(item)
        if not parsed:
            return
        rid = parsed["id"]
        existing = seen.get(rid)
        if existing is None or (parsed["date"] or "") > (existing["date"] or ""):
            parsed["_src"] = src
            seen[rid] = parsed

    # ── Hot (page 1 only, API 分页实际上不工作) ─────────────────────────────
    try:
        hot_resp = _get(BASE_URL + API_HOT_REPORTS,
                        params={"pageNum": 1, "pageSize": 20}, cookies=cookies)
        if hot_resp and hot_resp.get("code") == 0:
            for item in hot_resp.get("data", {}).get("content", []):
                _add_item(item, "hot")
    except Exception as e:
        logger.warning("聚合 hot 源失败: %s", e)

    # ── Focus (前 3 页焦点组，每组展开子报告) ────────────────────────────────
    try:
        # 先不带参数拿第 1 页（已经验证返回 content 列表）
        focus_resp = _get(BASE_URL + API_FOCUS_LIST, cookies=cookies)
        if focus_resp and focus_resp.get("code") == 0:
            total_pages = focus_resp.get("data", {}).get("totalPages", 1)
            _process_focus_page(focus_resp, seen)
            # 额外拉取第 2、3 页（带参数）
            for pg in range(2, min(total_pages + 1, 4)):
                fp = _get(BASE_URL + API_FOCUS_LIST,
                          params={"pageNum": pg, "pageSize": 20}, cookies=cookies)
                if fp and fp.get("code") == 0:
                    _process_focus_page(fp, seen)
    except Exception as e:
        logger.warning("聚合 focus 源失败: %s", e)

    # ── Recommend ────────────────────────────────────────────────────────────
    try:
        rec_resp = _get(BASE_URL + API_RECOMMEND, cookies=cookies)
        if rec_resp and rec_resp.get("code") == 0:
            data = rec_resp.get("data", [])
            items = data if isinstance(data, list) else data.get("content", [])
            for item in items:
                _add_item(item, "rec")
    except Exception as e:
        logger.warning("聚合 recommend 源失败: %s", e)

    # 按日期降序排列
    sorted_list = sorted(
        seen.values(),
        key=lambda r: r.get("date") or "",
        reverse=True,
    )
    return sorted_list


def _process_focus_page(focus_resp: dict, seen: Dict[str, dict]):
    """展开 focus 页面中的每个焦点组，将子报告加入 seen 字典。"""
    content = focus_resp.get("data", {}).get("content", [])
    for focus_item in content:
        sub_reports = focus_item.get("reportList", [])
        targets = sub_reports if sub_reports else [focus_item]
        for item in targets:
            parsed = _parse_single_report(item)
            if not parsed:
                continue
            rid = parsed["id"]
            if rid not in seen or (parsed["date"] or "") > (seen[rid].get("date") or ""):
                parsed["_src"] = "focus"
                seen[rid] = parsed


# ── 报告列表获取 ──────────────────────────────────────────────────────────────

def fetch_report_list(
    page: int = 1,
    page_size: int = 20,
    keyword: str = "",
    category: str = "",
    source: str = "hot",
    category_id: int = 0,
) -> Dict[str, Any]:
    """获取研究报告列表。

    Args:
        source: 数据来源:
            "hot"     - 热门研报 (/reports/api/hot/fetchHotReport)
            "latest"  - 全部研报 (POST /reports/api/v3/page，支持分类 + 关键字)
            "focus"   - 晨会焦点 (/focus/api/fetchFocusReportList)
            "recommend" - 推荐研报 (/feed/api/v3/reports-activities/recommendList)
        category_id: 分类 ID (见 CATEGORY_MAP)。0 = 全部，对 latest 源有效（服务端过滤）。
        keyword: 关键字过滤，对 latest 源通过服务端 input 字段传递。
    
    Returns:
        {
            "ok": bool,
            "reports": [{"id", "title", "authors", "date", "category", "abstract", "url"}, ...],
            "total": int,
            "page": int,
            "page_size": int,
            "source": str,
            "error": str or None,
        }
    """
    cookies = load_cookies()
    if not cookies:
        return {
            "ok": False,
            "reports": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
            "source": source,
            "error": "未配置 Cookies。请先登录 CICC Research 并导出 Cookies。",
        }

    # 缓存检查
    cache_key = f"list_{source}_{page}_{page_size}_{keyword}_{category_id}"
    if cache_key in _cache and time.time() - _cache_time.get(cache_key, 0) < CACHE_TTL:
        return _cache[cache_key]

    # ── 路由到对应 API ────────────────────────────────────────────────────────
    if source == "latest":
        # 全量分页模式：使用 POST /reports/api/v3/page（支持服务端分类 + 关键字过滤）
        api_result = _fetch_page_api(
            cookies=cookies,
            category_id=category_id,
            page=page,
            page_size=page_size,
            keyword=keyword,
        )
        if not api_result["ok"]:
            # 降级到旧聚合方式（当 session 失效时）
            logger.warning("page API 失败，降级到聚合模式: %s", api_result["error"])
            pool_key = "aggregated_latest_pool"
            if pool_key in _cache and time.time() - _cache_time.get(pool_key, 0) < CACHE_TTL:
                all_reports = _cache[pool_key]
            else:
                all_reports = _fetch_aggregated_latest(cookies)
                _cache[pool_key] = all_reports
                _cache_time[pool_key] = time.time()
            if category_id and category_id in CATEGORY_MAP:
                def _matches_category(r: dict) -> bool:
                    ids = set(r.get("portal_category_ids", []))
                    if category_id in ids:
                        return True
                    for sub_id in ids:
                        if _SUB_TO_PARENT.get(sub_id) == category_id:
                            return True
                    return False
                all_reports = [r for r in all_reports if _matches_category(r)]
            if keyword:
                kw_lower = keyword.lower()
                all_reports = [r for r in all_reports
                               if kw_lower in r.get("title", "").lower()
                               or kw_lower in r.get("abstract", "").lower()
                               or kw_lower in r.get("authors", "").lower()]
            total = len(all_reports)
            start = (page - 1) * page_size
            reports = all_reports[start:start + page_size]
        else:
            # 解析 page API 的 content 列表
            parsed = [_parse_page_api_item(item) for item in api_result["content"]]
            reports = [r for r in parsed if r is not None]
            total = api_result["total"]

        result = {
            "ok": len(reports) > 0 or total > 0,
            "reports": reports,
            "total": total,
            "page": page,
            "page_size": page_size,
            "source": source,
            "error": None if reports else "返回数据为空，可能需要刷新 Cookies。",
        }
        _cache[cache_key] = result
        _cache_time[cache_key] = time.time()
        return result

    elif source == "focus":
        raw = _get(BASE_URL + API_FOCUS_LIST, cookies=cookies)
        # Focus response: content items have nested reportList — flatten them
        if raw and raw.get("code") == 0:
            content = raw.get("data", {}).get("content", [])
            flat_list = []
            for focus_item in content:
                sub_reports = focus_item.get("reportList", [])
                if sub_reports:
                    flat_list.extend(sub_reports)
                elif focus_item.get("title"):
                    flat_list.append(focus_item)
            data = {"code": 0, "data": {"content": flat_list, "total": len(flat_list)}}
        else:
            data = raw
    elif source == "recommend":
        data = _get(BASE_URL + API_RECOMMEND, cookies=cookies)
    else:  # default: hot
        params = {"pageNum": page, "pageSize": page_size}
        if keyword:
            params["keyword"] = keyword
        data = _get(BASE_URL + API_HOT_REPORTS, params=params, cookies=cookies)

    if not data:
        # 降级：尝试另一个已知端点
        fallback = _get(
            BASE_URL + API_HOME_HOT, cookies=cookies,
        )
        if fallback:
            data = fallback

    if not data:
        return {
            "ok": False,
            "reports": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
            "source": source,
            "error": "请求报告列表失败，Cookies 可能已过期或网络异常。",
        }

    # 解析响应
    reports, total = _parse_report_list(data)

    # 关键字客户端过滤 (对于不支持 keyword 参数的端点)
    if keyword and reports:
        kw_lower = keyword.lower()
        reports = [r for r in reports if kw_lower in r.get("title", "").lower()
                   or kw_lower in r.get("abstract", "").lower()
                   or kw_lower in r.get("authors", "").lower()]

    result = {
        "ok": len(reports) > 0 or total > 0,
        "reports": reports,
        "total": total or len(reports),
        "page": page,
        "page_size": page_size,
        "source": source,
        "error": None if reports else "返回数据为空，可能需要刷新 Cookies。",
    }

    _cache[cache_key] = result
    _cache_time[cache_key] = time.time()
    return result


def _try_frontend_page(cookies: dict, keyword: str = "") -> Dict[str, Any]:
    """直接请求前端页面，尝试从 HTML 中提取数据 (SSR 模式)。"""
    url = f"{BASE_URL}/zh_CN/reportList"
    if keyword:
        url = f"{BASE_URL}/zh_CN/search?keyWord={keyword}"

    sess = _make_session(cookies)
    try:
        resp = sess.get(url, timeout=20)
        resp.raise_for_status()
        html = resp.text

        # 查找内嵌的 JSON 数据 (Vue/Nuxt SSR 常见模式)
        patterns = [
            r'window\.__NUXT__\s*=\s*(\{.+?\});?\s*</script>',
            r'window\.__INITIAL_STATE__\s*=\s*(\{.+?\});?\s*</script>',
            r'<script[^>]*>\s*var\s+\w+\s*=\s*(\{.+?"report.+?\});?\s*</script>',
        ]
        for pat in patterns:
            m = re.search(pat, html, re.DOTALL)
            if m:
                try:
                    embedded_data = json.loads(m.group(1))
                    reports, total = _parse_report_list(embedded_data)
                    if reports:
                        return {
                            "ok": True,
                            "reports": reports,
                            "total": total,
                            "page": 1,
                            "page_size": len(reports),
                            "error": None,
                            "api_discovered": False,
                        }
                except json.JSONDecodeError:
                    continue

        # 检查是否需要登录
        if "login" in html.lower() and ("token" in html.lower() or "sign" in html.lower()):
            return {
                "ok": False,
                "reports": [],
                "total": 0,
                "page": 1,
                "page_size": 0,
                "error": "页面提示需要登录，Cookies 已过期。请重新登录并更新 Cookies。",
                "api_discovered": False,
            }

        return {
            "ok": False, "reports": [], "total": 0, "page": 1, "page_size": 0,
            "error": "无法从页面 HTML 中提取报告数据。", "api_discovered": False,
        }
    except Exception as e:
        return {
            "ok": False, "reports": [], "total": 0, "page": 1, "page_size": 0,
            "error": f"请求前端页面失败: {e}", "api_discovered": False,
        }


def _parse_report_list(data: Any) -> tuple:
    """从 API 响应中解析报告列表。返回 (reports_list, total_count)。"""
    reports = []
    total = 0

    if not isinstance(data, dict):
        return reports, total

    # 递归查找报告列表
    def extract(obj, depth=0):
        nonlocal total
        if depth > 4:
            return []
        if isinstance(obj, list):
            result = []
            for item in obj:
                if isinstance(item, dict):
                    parsed = _parse_single_report(item)
                    if parsed:
                        result.append(parsed)
            if result:
                return result
        if isinstance(obj, dict):
            # 检查 total
            for key in ("total", "totalCount", "totalRecords", "count"):
                if key in obj and isinstance(obj[key], (int, float)):
                    total = max(total, int(obj[key]))
            # 递归
            for k, v in obj.items():
                found = extract(v, depth + 1)
                if found:
                    return found
        return []

    reports = extract(data)
    if not total:
        total = len(reports)
    return reports, total


def _parse_single_report(item: dict) -> Optional[dict]:
    """解析单条报告记录。"""
    # 通用字段映射
    # reportId (actual report ID) > id (hot-report record ID) > articleId
    id_keys = ["reportId", "report_id", "id", "articleId", "article_id", "docId"]
    title_keys = ["title", "reportTitle", "report_title", "name", "articleTitle"]
    author_keys = ["authors", "author", "authorName", "analyst", "analysts"]
    date_keys = ["publishDate", "publish_date", "date", "createTime", "create_time",
                 "publishTime", "updateTime", "releaseDate", "genDate"]
    category_keys = ["category", "type", "reportType", "report_type", "typeName", "categoryName"]
    abstract_keys = ["abstract", "summary", "description", "content", "brief", "digest"]

    def _find(keys):
        for k in keys:
            if k in item and item[k] is not None and item[k] != "":
                return item[k]
        return ""

    report_id = str(_find(id_keys))
    title = str(_find(title_keys))

    if not title or not report_id:
        return None

    # 处理 authors: 可能是 list of dict [{name, ...}], list of str, 或 str
    authors_raw = _find(author_keys)
    if isinstance(authors_raw, list):
        authors = ", ".join(
            str(a.get("name", a.get("analystName", a)) if isinstance(a, dict) else a)
            for a in authors_raw
        )
    else:
        authors = str(authors_raw)

    date_val = _find(date_keys)
    if isinstance(date_val, (int, float)) and date_val > 1e12:
        date_val = datetime.fromtimestamp(date_val / 1000).strftime("%Y-%m-%d")
    elif isinstance(date_val, (int, float)) and date_val > 1e9:
        date_val = datetime.fromtimestamp(date_val).strftime("%Y-%m-%d")
    else:
        date_val = str(date_val)[:10] if date_val else ""

    abstract_val = str(_find(abstract_keys))
    if len(abstract_val) > 800:
        abstract_val = abstract_val[:800] + "..."

    # "reportList" nested items (focus reports contain a nested reportList)
    # If item has 'reportList', gather titles from it for display
    nested_sub = ""
    if "reportList" in item and isinstance(item["reportList"], list):
        sub_titles = [r.get("title", "") for r in item["reportList"][:3] if r.get("title")]
        if sub_titles:
            nested_sub = "；".join(sub_titles)

    # 保留原始分类 ID 列表，供分类过滤使用
    portal_ids_raw = item.get("portalCategoryIds", [])
    if isinstance(portal_ids_raw, list):
        portal_ids = [int(x) for x in portal_ids_raw if str(x).isdigit()]
    else:
        portal_ids = []

    return {
        "id": report_id,
        "title": title,
        "authors": authors,
        "date": date_val,
        "category": str(_find(category_keys)),
        "abstract": abstract_val or nested_sub,
        "url": f"{BASE_URL}/zh_CN/report/detail/{report_id}",
        "portal_category_ids": portal_ids,
    }


# ── 报告详情获取 ──────────────────────────────────────────────────────────────

def fetch_report_detail(report_id: str) -> Dict[str, Any]:
    """获取单篇研究报告详情。

    report_id 可以是:
      - rawId (reportId 字段, 例如 '1213452') — 优先使用
      - id (内部 id 字段, 例如 '386039')
    """
    cookies = load_cookies()
    if not cookies:
        return {"ok": False, "error": "未配置 Cookies。", "report": None}

    # 缓存
    cache_key = f"detail_{report_id}"
    if cache_key in _cache and time.time() - _cache_time.get(cache_key, 0) < CACHE_TTL:
        return _cache[cache_key]

    result_data = None
    sess = _make_session(cookies)
    xt = _get_x_time()
    url = BASE_URL + API_DETAIL

    # 正确参数: rawId (对应 reportId 字段) 或 id (对应 internal id)
    # 注意: 旧代码误用了 reportId/id，实际 API 接受 rawId 或 id(内部)
    for param_name in ("rawId", "id"):
        try:
            resp = sess.get(
                url,
                params={param_name: report_id},
                headers={
                    "X-Time": xt,
                    "Accept": "application/json, text/plain, */*",
                    "Referer": f"{BASE_URL}/zh_CN/report/detail/{report_id}",
                },
                timeout=20,
            )
            if resp.status_code == 200:
                j = resp.json()
                if j.get("code") == 0:
                    result_data = j
                    break
        except Exception as e:
            logger.warning("fetch_report_detail %s=%s failed: %s", param_name, report_id, e)

    if not result_data:
        detail_url = f"{BASE_URL}/zh_CN/report/detail/{report_id}"
        result = {
            "ok": False,
            "report": {
                "id": report_id,
                "title": f"报告 #{report_id}",
                "authors": "",
                "date": "",
                "content": "",
                "abstract": "无法获取详情，可能需要刷新 Cookies。",
                "url": detail_url,
                "signature_url": "",
            },
            "error": "获取报告详情失败，请检查 Cookies 是否有效。",
            "redirect_url": detail_url,
        }
        _cache[cache_key] = result
        _cache_time[cache_key] = time.time()
        return result

    # 解析详情数据
    report = _parse_report_detail(result_data, report_id)
    result = {
        "ok": report is not None,
        "report": report,
        "error": None if report else "解析报告详情失败。",
    }

    _cache[cache_key] = result
    _cache_time[cache_key] = time.time()
    return result


def _parse_report_detail(data: Any, report_id: str) -> Optional[dict]:
    """解析报告详情响应 (v3 detail API)。"""
    if not isinstance(data, dict):
        return None

    # v3 API 响应格式: {"code": 0, "data": {...}}
    detail = data.get("data", data)
    if not isinstance(detail, dict):
        return None

    def _g(*keys):
        for k in keys:
            v = detail.get(k)
            if v is not None and v != "":
                return v
        return ""

    # 提取 HTML 摘要并转为纯文本
    summary_html = str(_g("summary", "reportContent", "content", "body") or "")
    plain_summary = str(_g("plainSummary", "plain_summary") or "")
    if not plain_summary and summary_html:
        plain_summary = re.sub(r'<[^>]+>', ' ', summary_html)
        plain_summary = re.sub(r'\s+', ' ', plain_summary).strip()

    # authors
    author_names = str(_g("authorNames", "authors", "author") or "")

    # date
    pub_time = str(_g("publishTime", "publishDate", "date") or "")
    date_val = pub_time[:10] if pub_time else ""

    # signatureUrl — 有效期内可直接下载 PDF（需 X-Time 头）
    signature_url = str(_g("signatureUrl", "signature_url") or "")

    # pdf pages
    pdf_pages = detail.get("pdfPages", 0)

    # categories
    cats = detail.get("categories", [])
    cat_names = ", ".join(c.get("name", "") for c in cats if isinstance(c, dict)) if cats else ""

    raw_id = str(detail.get("rawId", "") or report_id)
    internal_id = str(detail.get("id", "") or "")

    return {
        "id": raw_id,
        "internal_id": internal_id,
        "title": str(_g("title", "reportTitle") or f"报告 #{raw_id}"),
        "authors": author_names,
        "date": date_val,
        "content": plain_summary[:8000] if plain_summary else "",
        "content_html": summary_html[:15000] if summary_html else "",
        "abstract": plain_summary[:1000] if plain_summary else "",
        "url": f"{BASE_URL}/zh_CN/report/detail/{raw_id}",
        "signature_url": signature_url,
        "pdf_pages": pdf_pages,
        "category": cat_names,
    }


# ── PDF 下载 ──────────────────────────────────────────────────────────────────

def fetch_report_pdf(report_id: str, out_dir: str = "") -> Dict[str, Any]:
    """下载研究报告 PDF。

    Args:
        report_id: 报告的 rawId (reportId 字段值，如 '1213452')
        out_dir:   保存目录，默认为当前目录下 memory/Research/

    Returns:
        {
            "ok": bool,
            "file_path": str,   # 保存路径
            "file_size": int,   # 文件大小 (bytes)
            "title": str,
            "error": str | None,
        }
    """
    cookies = load_cookies()
    if not cookies:
        return {"ok": False, "file_path": "", "file_size": 0,
                "title": "", "error": "未配置 Cookies。"}

    if not out_dir:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "memory", "Research")
    os.makedirs(out_dir, exist_ok=True)

    # 1. 获取最新 signatureUrl（有时效性，需即时获取再下载）
    sess = _make_session(cookies)
    xt = _get_x_time()
    try:
        resp = sess.get(
            BASE_URL + API_DETAIL,
            params={"rawId": report_id},
            headers={
                "X-Time": xt,
                "Accept": "application/json, text/plain, */*",
                "Referer": f"{BASE_URL}/zh_CN/report/detail/{report_id}",
            },
            timeout=20,
        )
        if resp.status_code != 200 or resp.json().get("code") != 0:
            return {"ok": False, "file_path": "", "file_size": 0, "title": "",
                    "error": f"获取报告详情失败 HTTP {resp.status_code}"}
        detail_json = resp.json()["data"]
    except Exception as e:
        return {"ok": False, "file_path": "", "file_size": 0,
                "title": "", "error": f"请求详情 API 失败: {e}"}

    signature_url = detail_json.get("signatureUrl", "")
    title = detail_json.get("title", f"report_{report_id}")

    if not signature_url:
        return {"ok": False, "file_path": "", "file_size": 0,
                "title": title, "error": "未获取到 PDF 下载链接 (signatureUrl 为空)"}

    # 2. 下载 PDF（必须带 X-Time 头）
    safe_title = re.sub(r'[\\/:*?"<>|]', '_', title)[:80]
    out_path = os.path.join(out_dir, f"{safe_title}.pdf")

    try:
        xt2 = _get_x_time()
        dl = sess.get(
            signature_url,
            headers={
                "X-Time": xt2,
                "Accept": "application/pdf,application/octet-stream,*/*",
                "Referer": f"{BASE_URL}/zh_CN/report/detail/{report_id}",
            },
            timeout=120,
            stream=True,
        )
        if dl.status_code != 200:
            return {"ok": False, "file_path": "", "file_size": 0, "title": title,
                    "error": f"PDF 下载失败 HTTP {dl.status_code}: {dl.text[:200]}"}

        with open(out_path, "wb") as f:
            total = 0
            for chunk in dl.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)

        # 验证 PDF
        with open(out_path, "rb") as f:
            header = f.read(5)
        if not header.startswith(b"%PDF"):
            os.remove(out_path)
            return {"ok": False, "file_path": "", "file_size": 0, "title": title,
                    "error": f"下载内容不是有效 PDF (header={header!r})"}

        logger.info("PDF downloaded: %s (%d bytes)", out_path, total)
        return {
            "ok": True,
            "file_path": out_path,
            "file_size": total,
            "title": title,
            "error": None,
        }

    except Exception as e:
        logger.error("PDF download failed: %s", e)
        return {"ok": False, "file_path": "", "file_size": 0,
                "title": title, "error": str(e)}


# ── 搜索 ──────────────────────────────────────────────────────────────────────

def search_reports(keyword: str, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
    """搜索研究报告。"""
    if not keyword:
        return fetch_report_list(page=page, page_size=page_size, source="hot")

    cookies = load_cookies()
    if not cookies:
        return {"ok": False, "reports": [], "total": 0, "page": page,
                "page_size": page_size, "keyword": keyword, "error": "未配置 Cookies。"}

    # 缓存
    cache_key = f"search_{keyword}_{page}_{page_size}"
    if cache_key in _cache and time.time() - _cache_time.get(cache_key, 0) < CACHE_TTL:
        return _cache[cache_key]

    # 先用 keyword 参数请求 hot/browsing 接口
    # (hot 接口可能不支持 keyword, browsing 接口支持)
    data = _get(
        BASE_URL + API_BROWSING,
        params={"pageNum": page, "pageSize": page_size, "keyword": keyword},
        cookies=cookies,
    )
    if not data or data.get("code") != 0:
        # 降级到 hot 接口并做客户端过滤
        data = _get(BASE_URL + API_HOT_REPORTS,
                    params={"pageNum": page, "pageSize": page_size}, cookies=cookies)

    if not data:
        return {"ok": False, "reports": [], "total": 0, "page": page,
                "page_size": page_size, "keyword": keyword, "error": "搜索请求失败。"}

    reports, total = _parse_report_list(data)

    # 客户端关键字过滤
    kw_lower = keyword.lower()
    filtered = [r for r in reports if
                kw_lower in r.get("title", "").lower() or
                kw_lower in r.get("abstract", "").lower() or
                kw_lower in r.get("authors", "").lower()]
    if not filtered:
        filtered = reports  # 如果过滤后为空，返回全部

    result = {
        "ok": True,
        "reports": filtered,
        "total": len(filtered),
        "page": page,
        "page_size": page_size,
        "keyword": keyword,
        "error": None,
    }
    _cache[cache_key] = result
    _cache_time[cache_key] = time.time()
    return result


# ── 连接测试 ──────────────────────────────────────────────────────────────────

def test_connection() -> Dict[str, Any]:
    """测试 CICC Research 连接和 Cookies 有效性。"""
    cookies = load_cookies()
    if not cookies:
        return {
            "ok": False,
            "status": "no_cookies",
            "message": "未找到 Cookies 文件。请先登录 CICC Research 并导出 Cookies。",
            "details": {
                "cookies_file": COOKIES_FILE,
                "cookie_count": 0,
            },
        }

    # 1. 测试连通性 + Cookies 有效性 — 直接调用已验证的热门研报 API
    sess = _make_session(cookies)
    try:
        resp = sess.get(
            f"{BASE_URL}{API_HOT_REPORTS}",
            params={"pageNum": 1, "pageSize": 3},
            timeout=15,
        )
        status_code = resp.status_code
    except Exception as e:
        return {
            "ok": False,
            "status": "network_error",
            "message": f"网络请求失败: {e}",
            "details": {"error": str(e)},
        }

    if status_code in (401, 403):
        return {
            "ok": False,
            "status": "auth_failed",
            "message": f"认证失败 (HTTP {status_code})。Cookies 已过期，请重新登录。",
            "details": {"status_code": status_code},
        }

    # 2. 解析响应
    try:
        data = resp.json()
    except Exception:
        data = {}

    sample_reports = []
    if data.get("code") == 0:
        raw_data = data.get("data", {})
        content = raw_data.get("content", []) if isinstance(raw_data, dict) else raw_data
        if isinstance(content, list):
            sample_reports = content[:3]

    ok = data.get("code") == 0 and len(sample_reports) > 0

    return {
        "ok": ok,
        "status": "connected" if ok else "partial",
        "message": (
            f"连接成功！Cookies 有效，已获取 {len(sample_reports)} 条热门研报。"
            if ok else
            f"已连接到 CICC Research (HTTP {status_code})，但数据解析异常。"
            f" code={data.get('code')} msg={data.get('msg', '')}"
        ),
        "details": {
            "status_code": status_code,
            "api_endpoint": API_HOT_REPORTS,
            "sample_count": len(sample_reports),
            "cookie_count": len(cookies),
            "response_code": data.get("code"),
        },
    }


# ── 清除缓存 ──────────────────────────────────────────────────────────────────

def clear_cache():
    """清除所有缓存。"""
    _cache.clear()
    _cache_time.clear()
    global _discovered_list_api, _discovered_detail_api
    _discovered_list_api = None
    _discovered_detail_api = None


# ── CLI 测试 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("=== 测试 CICC Research 连接 ===")
        result = test_connection()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "list":
        print("=== 获取报告列表 ===")
        result = fetch_report_list(page=1, page_size=10)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "search":
        kw = sys.argv[2] if len(sys.argv) > 2 else "大宗商品"
        print(f"=== 搜索: {kw} ===")
        result = search_reports(kw)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "status":
        print("=== Cookies 状态 ===")
        print(json.dumps(get_cookies_status(), ensure_ascii=False, indent=2))
    else:
        print("用法:")
        print("  python cicc_research.py status   -- 查看 Cookies 状态")
        print("  python cicc_research.py test     -- 测试连接")
        print("  python cicc_research.py list     -- 获取报告列表")
        print("  python cicc_research.py search [关键词] -- 搜索报告")
