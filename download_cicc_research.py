#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中金研究报告批量下载器 (CICC Research Bulk Downloader)

每天运行一次，自动从 CICC Research 各分类板块下载最新研报 PDF 到本地。
已下载过的报告（按 reportId 去重）不会重复下载。

用法:
    python download_cicc_research.py              # 下载全部分类，每类最新 10 篇
    python download_cicc_research.py --count 5    # 每类最新 5 篇
    python download_cicc_research.py --dry-run    # 仅列出待下载，不实际下载
    python download_cicc_research.py --category 宏观经济 --category 大宗商品  # 仅下载指定分类

目录结构:
    memory/cicc_research/
    ├── 宏观经济/
    ├── 市场策略/
    ├── 行业研究/
    ├── 公司研究/
    ├── 大宗商品/
    ├── 晨会焦点/
    ├── ESG/
    ├── 全球外汇/
    ├── 全球研究/
    └── _download_history.json     ← 已下载记录（防止重复）
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# ── 将项目根目录加入 sys.path ────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cicc_research import (
    CATEGORY_MAP,
    _fetch_page_api,
    _get_x_time,
    _make_session,
    _parse_page_api_item,
    load_cookies,
    fetch_report_detail,
    BASE_URL,
    API_DETAIL,
)

# ── 配置 ──────────────────────────────────────────────────────────────────────
DOWNLOAD_ROOT = os.path.join(ROOT_DIR, "memory", "cicc_research")
HISTORY_FILE = os.path.join(DOWNLOAD_ROOT, "_download_history.json")
DEFAULT_COUNT = 10  # 每个分类默认下载最新几篇

import io

# 强制 stdout/stderr 使用 UTF-8 编码（解决 Windows GBK 环境下 emoji 输出问题）
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── 下载历史管理 ──────────────────────────────────────────────────────────────

def load_history() -> Dict:
    """加载已下载历史记录。
    格式: { "<report_id>": {"title": str, "category": str, "date": str, "downloaded_at": str, "file": str}, ... }
    """
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_history(history: Dict):
    """持久化下载历史。"""
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ── PDF 下载（带 signatureUrl 即时获取） ──────────────────────────────────────

def download_pdf(report_id: str, title: str, out_dir: str, cookies: dict) -> Dict:
    """下载单篇 PDF 到指定目录。

    Returns:
        {"ok": bool, "file_path": str, "file_size": int, "error": str|None}
    """
    os.makedirs(out_dir, exist_ok=True)
    sess = _make_session(cookies)

    # 1. 获取 signatureUrl
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
        if resp.status_code != 200:
            return {"ok": False, "file_path": "", "file_size": 0,
                    "error": f"获取详情失败 HTTP {resp.status_code}"}
        j = resp.json()
        if j.get("code") != 0:
            return {"ok": False, "file_path": "", "file_size": 0,
                    "error": f"API 返回错误 code={j.get('code')}: {j.get('desc', '')}"}
        detail = j.get("data", {})
    except Exception as e:
        return {"ok": False, "file_path": "", "file_size": 0,
                "error": f"请求详情 API 异常: {e}"}

    signature_url = detail.get("signatureUrl", "")
    if not signature_url:
        return {"ok": False, "file_path": "", "file_size": 0,
                "error": "signatureUrl 为空，无法下载 PDF"}

    # 用详情接口返回的标题（更准确）
    actual_title = detail.get("title", "") or title

    # 2. 下载 PDF
    safe_title = re.sub(r'[\\/:*?"<>|\r\n]', '_', actual_title)[:100].strip('. _')
    out_path = os.path.join(out_dir, f"{safe_title}.pdf")

    # 如果文件已存在，跳过
    if os.path.exists(out_path):
        sz = os.path.getsize(out_path)
        if sz > 1024:  # > 1KB 视为有效
            return {"ok": True, "file_path": out_path, "file_size": sz, "error": None}

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
            return {"ok": False, "file_path": "", "file_size": 0,
                    "error": f"PDF 下载失败 HTTP {dl.status_code}"}

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
            return {"ok": False, "file_path": "", "file_size": 0,
                    "error": f"下载内容非有效 PDF (header={header!r})"}

        return {"ok": True, "file_path": out_path, "file_size": total, "error": None}

    except Exception as e:
        return {"ok": False, "file_path": "", "file_size": 0, "error": str(e)}


# ── 主下载逻辑 ────────────────────────────────────────────────────────────────

def download_all_categories(
    count: int = DEFAULT_COUNT,
    dry_run: bool = False,
    categories: Optional[List[str]] = None,
) -> Dict:
    """遍历所有（或指定）分类，下载最新研报。

    Args:
        count: 每个分类获取的最新报告数量
        dry_run: 如果为 True，仅列出待下载列表，不实际下载
        categories: 仅下载这些分类（中文名），为 None 则下载全部

    Returns:
        {"total_new": int, "total_skipped": int, "total_failed": int, "details": [...]}
    """
    cookies = load_cookies()
    if not cookies:
        logger.error("❌ 未找到 Cookies，请先配置 cicc_research/cicc_cookies.json")
        return {"total_new": 0, "total_skipped": 0, "total_failed": 0,
                "details": [], "error": "未配置 Cookies"}

    history = load_history()
    downloaded_ids: Set[str] = set(history.keys())

    # 过滤分类
    target_categories = {}
    for cat_id, cat_name in CATEGORY_MAP.items():
        if categories is None or cat_name in categories:
            target_categories[cat_id] = cat_name

    if not target_categories:
        logger.error("未匹配到任何分类。可用分类: %s",
                      ", ".join(CATEGORY_MAP.values()))
        return {"total_new": 0, "total_skipped": 0, "total_failed": 0,
                "details": [], "error": "未匹配到分类"}

    total_new = 0
    total_skipped = 0
    total_failed = 0
    details = []

    for cat_id, cat_name in target_categories.items():
        cat_dir = os.path.join(DOWNLOAD_ROOT, cat_name)
        os.makedirs(cat_dir, exist_ok=True)

        logger.info("━━━ 📂 %s (ID=%d) ━━━", cat_name, cat_id)

        # 获取最新报告列表
        result = _fetch_page_api(
            cookies=cookies,
            category_id=cat_id,
            page=1,
            page_size=count,
        )

        if not result["ok"]:
            logger.warning("  ⚠️  获取 %s 列表失败: %s", cat_name, result.get("error"))
            details.append({
                "category": cat_name,
                "error": result.get("error"),
                "new": 0, "skipped": 0, "failed": 0,
            })
            continue

        items = result.get("content", [])
        if not items:
            logger.info("  (空) 该分类暂无报告")
            details.append({
                "category": cat_name, "error": None,
                "new": 0, "skipped": 0, "failed": 0,
            })
            continue

        cat_new = 0
        cat_skipped = 0
        cat_failed = 0

        for item in items:
            parsed = _parse_page_api_item(item)
            if not parsed:
                continue

            rid = parsed["id"]
            title = parsed["title"]
            date = parsed["date"]

            # 检查是否已下载
            if rid in downloaded_ids:
                logger.info("  ⏭️  [已有] %s (%s)", title[:50], date)
                cat_skipped += 1
                continue

            if dry_run:
                logger.info("  🔍 [待下载] %s (%s) ID=%s", title[:50], date, rid)
                cat_new += 1
                continue

            # 实际下载
            logger.info("  ⬇️  下载: %s (%s)", title[:50], date)
            dl_result = download_pdf(rid, title, cat_dir, cookies)

            if dl_result["ok"]:
                size_kb = dl_result["file_size"] / 1024
                logger.info("  ✅ 完成 %.1f KB → %s",
                            size_kb, os.path.basename(dl_result["file_path"]))
                cat_new += 1

                # 更新历史
                history[rid] = {
                    "title": title,
                    "category": cat_name,
                    "date": date,
                    "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "file": dl_result["file_path"],
                }
                downloaded_ids.add(rid)
                save_history(history)  # 每下载一篇就保存，防止中断丢失

                # 请求间隔，避免被限流
                time.sleep(1.5)
            else:
                logger.warning("  ❌ 失败: %s — %s", title[:40], dl_result["error"])
                cat_failed += 1
                time.sleep(0.5)

        total_new += cat_new
        total_skipped += cat_skipped
        total_failed += cat_failed

        details.append({
            "category": cat_name,
            "error": None,
            "new": cat_new,
            "skipped": cat_skipped,
            "failed": cat_failed,
        })

        logger.info("  → %s: 新下载 %d, 已有 %d, 失败 %d",
                     cat_name, cat_new, cat_skipped, cat_failed)

    return {
        "total_new": total_new,
        "total_skipped": total_skipped,
        "total_failed": total_failed,
        "details": details,
        "error": None,
    }


# ── CLI 入口 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="中金研究报告批量下载器 — 每天运行以获取最新研报 PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_cicc_research.py                          # 下载全部分类，每类最新 10 篇
  python download_cicc_research.py --count 5                # 每类最新 5 篇
  python download_cicc_research.py --dry-run                # 查看哪些报告待下载
  python download_cicc_research.py --category 宏观经济       # 仅下载宏观经济
  python download_cicc_research.py --category 大宗商品 --category ESG  # 下载多个分类
  python download_cicc_research.py --status                 # 查看下载统计

可用分类: """ + ", ".join(CATEGORY_MAP.values()),
    )
    parser.add_argument(
        "--count", "-n", type=int, default=DEFAULT_COUNT,
        help=f"每个分类获取最新报告数量 (默认: {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--dry-run", "-d", action="store_true",
        help="仅查看待下载列表，不实际下载",
    )
    parser.add_argument(
        "--category", "-c", action="append", dest="categories",
        help="指定下载分类（可多次使用，中文名称）",
    )
    parser.add_argument(
        "--status", "-s", action="store_true",
        help="查看已下载报告统计",
    )
    args = parser.parse_args()

    # ── 状态查看 ──
    if args.status:
        history = load_history()
        if not history:
            print("📭 尚未下载任何报告。")
            return

        # 按分类统计
        by_cat = {}
        for rid, info in history.items():
            cat = info.get("category", "未知")
            by_cat.setdefault(cat, []).append(info)

        print(f"\n📊 已下载报告统计 (共 {len(history)} 篇)")
        print("━" * 50)
        for cat, reports in sorted(by_cat.items()):
            dates = [r.get("date", "") for r in reports if r.get("date")]
            latest = max(dates) if dates else "N/A"
            print(f"  📂 {cat:12s}  {len(reports):3d} 篇  (最新: {latest})")
        print("━" * 50)

        oldest = min((r.get("downloaded_at", "") for r in history.values()), default="N/A")
        newest = max((r.get("downloaded_at", "") for r in history.values()), default="N/A")
        print(f"  首次下载: {oldest}")
        print(f"  最近下载: {newest}")
        return

    # ── 主下载流程 ──
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║       中金研究报告下载器 (CICC Research)         ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    mode = "🔍 预览模式 (dry-run)" if args.dry_run else "⬇️  下载模式"
    print(f"  模式: {mode}")
    print(f"  每类数量: {args.count}")
    if args.categories:
        print(f"  指定分类: {', '.join(args.categories)}")
    else:
        print(f"  分类: 全部 ({len(CATEGORY_MAP)} 个)")
    print(f"  保存目录: {DOWNLOAD_ROOT}")
    print()

    result = download_all_categories(
        count=args.count,
        dry_run=args.dry_run,
        categories=args.categories,
    )

    # ── 汇总 ──
    print()
    print("═" * 50)
    if args.dry_run:
        print(f"📋 待下载: {result['total_new']} 篇, 已有: {result['total_skipped']} 篇")
    else:
        print(f"✅ 新下载: {result['total_new']} 篇")
        print(f"⏭️  已有(跳过): {result['total_skipped']} 篇")
        if result['total_failed']:
            print(f"❌ 失败: {result['total_failed']} 篇")
    print("═" * 50)

    if result.get("error"):
        print(f"\n⚠️  {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
