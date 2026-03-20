"""Data collector — aggregates all raw data sources for the morning briefing.

Runs scrapers, news fetchers, book data extraction, and research lookups
concurrently via ThreadPoolExecutor, then returns a single dict that can
be handed to prompt_builder.
"""

import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Internal imports ─────────────────────────────────────────────────────────
from .external_feeds import (
    fetch_nitter_rss,
    fetch_telegram_channels,
    fetch_substack_feeds,
    fetch_polymarket_signals,
)
from .web_search import search_web
from .config import (
    WEB_SEARCH_QUERIES,
    BAILIAN_APP_ID,
    CICC_MORNING_FOCUS_DIR,
    CICC_COMMODITY_DIR,
    CICC_MACRO_DIR,
    CICC_STRATEGY_DIR,
    FEED_FETCH_TIMEOUT,
    DATA_COLLECT_TIMEOUT,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── PDF text extraction ──────────────────────────────────────────────────────

def _extract_pdf_text(pdf_path: str, max_pages: int = 8) -> str:
    """Extract text from a PDF file. Tries pdfplumber first, falls back to PyPDF2."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:max_pages]:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n\n".join(text_parts)
    except ImportError:
        pass
    except Exception as e:
        print(f"[SPEECH] pdfplumber error on {pdf_path}: {e}")

    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages[:max_pages]:
            t = page.extract_text()
            if t:
                text_parts.append(t)
        return "\n\n".join(text_parts)
    except Exception as e:
        print(f"[SPEECH] PyPDF2 error on {pdf_path}: {e}")
    return ""


def _get_latest_pdf(directory: str, max_count: int = 2) -> List[Dict[str, str]]:
    """Get latest PDF files from a directory, return list of {name, text}."""
    dir_path = PROJECT_ROOT / directory
    if not dir_path.exists():
        return []
    pdfs = sorted(dir_path.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    results = []
    for pdf in pdfs[:max_count]:
        text = _extract_pdf_text(str(pdf), max_pages=6)
        if text:
            results.append({"name": pdf.name, "text": text[:3000]})  # cap text length
    return results


# ── Data source fetchers ─────────────────────────────────────────────────────

def _fetch_finance_news() -> Dict[str, Any]:
    """Fetch Chinese financial news from existing news module."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from news.news_fetcher import fetch_category_news
        return {"ok": True, "data": fetch_category_news("finance")}
    except Exception as e:
        print(f"[SPEECH] Finance news error: {e}")
        return {"ok": False, "data": [], "error": str(e)}


def _fetch_nitter() -> Dict[str, Any]:
    try:
        data = fetch_nitter_rss()
        return {"ok": bool(data), "data": data}
    except Exception as e:
        print(f"[SPEECH] Nitter error: {e}")
        return {"ok": False, "data": [], "error": str(e)}


def _fetch_telegram() -> Dict[str, Any]:
    try:
        data = fetch_telegram_channels()
        return {"ok": bool(data), "data": data}
    except Exception as e:
        print(f"[SPEECH] Telegram error: {e}")
        return {"ok": False, "data": [], "error": str(e)}


def _fetch_substack() -> Dict[str, Any]:
    try:
        data = fetch_substack_feeds()
        return {"ok": bool(data), "data": data}
    except Exception as e:
        print(f"[SPEECH] Substack error: {e}")
        return {"ok": False, "data": [], "error": str(e)}


def _fetch_polymarket() -> Dict[str, Any]:
    try:
        data = fetch_polymarket_signals()
        return {"ok": bool(data), "data": data}
    except Exception as e:
        print(f"[SPEECH] Polymarket error: {e}")
        return {"ok": False, "data": [], "error": str(e)}


def _fetch_web_search() -> Dict[str, Any]:
    """Run targeted web searches for overnight market context."""
    try:
        all_results = []
        for query in WEB_SEARCH_QUERIES:
            results = search_web(query, max_results=3)
            all_results.extend(results)
        return {"ok": bool(all_results), "data": all_results}
    except Exception as e:
        print(f"[SPEECH] Web search error: {e}")
        return {"ok": False, "data": [], "error": str(e)}


def _fetch_cicc_morning_focus() -> Dict[str, Any]:
    """Read latest CICC morning focus PDFs."""
    try:
        pdfs = _get_latest_pdf(CICC_MORNING_FOCUS_DIR, max_count=2)
        return {"ok": bool(pdfs), "data": pdfs}
    except Exception as e:
        print(f"[SPEECH] CICC morning focus error: {e}")
        return {"ok": False, "data": [], "error": str(e)}


def _fetch_cicc_commodity() -> Dict[str, Any]:
    """Read latest CICC commodity research PDFs."""
    try:
        pdfs = _get_latest_pdf(CICC_COMMODITY_DIR, max_count=2)
        return {"ok": bool(pdfs), "data": pdfs}
    except Exception as e:
        print(f"[SPEECH] CICC commodity error: {e}")
        return {"ok": False, "data": [], "error": str(e)}


def _fetch_cicc_macro() -> Dict[str, Any]:
    """Read latest CICC macro research PDFs."""
    try:
        pdfs = _get_latest_pdf(CICC_MACRO_DIR, max_count=1)
        return {"ok": bool(pdfs), "data": pdfs}
    except Exception as e:
        return {"ok": False, "data": [], "error": str(e)}


def _fetch_cicc_strategy() -> Dict[str, Any]:
    """Read latest CICC strategy research PDFs."""
    try:
        pdfs = _get_latest_pdf(CICC_STRATEGY_DIR, max_count=1)
        return {"ok": bool(pdfs), "data": pdfs}
    except Exception as e:
        return {"ok": False, "data": [], "error": str(e)}


def _fetch_bailian_rag() -> Dict[str, Any]:
    """Query Bailian RAG for a morning market summary."""
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            return {"ok": False, "data": "", "error": "No DASHSCOPE_API_KEY"}

        from dashscope import Application
        prompt = (
            "请简要总结今日大宗商品和衍生品市场的最新动态、关键驱动因素、"
            "以及值得关注的风险事件。包括原油、黄金、铜、铁矿石等主要品种。"
            "同时请提供宏观经济方面的最新分析观点。"
        )
        response = Application.call(
            app_id=BAILIAN_APP_ID,
            prompt=prompt,
            api_key=api_key,
        )
        if response.status_code == 200:
            return {"ok": True, "data": response.output.text}
        else:
            return {"ok": False, "data": "", "error": f"Bailian: {response.code} {response.message}"}
    except Exception as e:
        print(f"[SPEECH] Bailian RAG error: {e}")
        return {"ok": False, "data": "", "error": str(e)}


# ── Book data extraction (called from Flask context) ─────────────────────────

def extract_book_data(app_data: Dict, contracts_fetcher=None, cross_gamma_fetcher=None) -> Dict[str, Any]:
    """Extract book positioning data from Flask app DATA dict.

    Parameters
    ----------
    app_data : dict
        The global DATA dict from app.py (sector_summaries, book_summaries, etc.)
    contracts_fetcher : callable, optional
        A function that returns (positions_list, error_string) — typically _fetch_qis_contracts
    cross_gamma_fetcher : callable, optional
        A function that returns the cross gamma aggregated result dict

    Returns
    -------
    dict with keys: sector_summaries, book_summaries, contracts, cross_gamma
    """
    result = {
        "sector_summaries": app_data.get("sector_summaries", {}),
        "book_summaries": app_data.get("book_summaries", {}),
        "sector_order": app_data.get("sector_order", []),
        "columns": app_data.get("columns", []),
        "contracts": [],
        "cross_gamma": None,
    }

    # Fetch contracts (strike/barrier levels)
    if contracts_fetcher:
        try:
            positions, err = contracts_fetcher()
            if positions and not err:
                result["contracts"] = positions
        except Exception as e:
            print(f"[SPEECH] Contracts fetch error: {e}")

    # Fetch cross gamma
    if cross_gamma_fetcher:
        try:
            result["cross_gamma"] = cross_gamma_fetcher()
        except Exception as e:
            print(f"[SPEECH] Cross gamma fetch error: {e}")

    return result


# ── Main collector ───────────────────────────────────────────────────────────

def collect_all_data(
    app_data: Optional[Dict] = None,
    contracts_fetcher=None,
    cross_gamma_fetcher=None,
) -> Dict[str, Any]:
    """Collect data from all sources concurrently.

    Returns a dict keyed by source name, each with {ok, data, error?}.
    """
    print("[SPEECH] Starting data collection...")
    start = datetime.now()

    # External data sources (run concurrently)
    tasks = {
        "finance_news": _fetch_finance_news,
        "nitter_x": _fetch_nitter,
        "telegram": _fetch_telegram,
        "substack": _fetch_substack,
        "polymarket": _fetch_polymarket,
        "web_search": _fetch_web_search,
        "cicc_morning_focus": _fetch_cicc_morning_focus,
        "cicc_commodity": _fetch_cicc_commodity,
        "cicc_macro": _fetch_cicc_macro,
        "cicc_strategy": _fetch_cicc_strategy,
        "bailian_rag": _fetch_bailian_rag,
    }

    collected: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(fn): name for name, fn in tasks.items()}
        try:
            for future in as_completed(futures, timeout=DATA_COLLECT_TIMEOUT):
                name = futures[future]
                try:
                    collected[name] = future.result(timeout=5)
                except Exception as e:
                    collected[name] = {"ok": False, "data": [], "error": str(e)}
                    print(f"[SPEECH] {name} failed: {e}")
        except TimeoutError:
            # Some futures didn't finish in time — mark them as failed, don't crash
            for fut, name in futures.items():
                if name not in collected:
                    collected[name] = {"ok": False, "data": [], "error": "Timed out"}
                    print(f"[SPEECH] {name} timed out (>{DATA_COLLECT_TIMEOUT}s), skipping")
                    fut.cancel()

    # Book data (synchronous, from Flask context)
    if app_data:
        collected["book"] = {
            "ok": True,
            "data": extract_book_data(app_data, contracts_fetcher, cross_gamma_fetcher),
        }

    elapsed = (datetime.now() - start).total_seconds()
    success_count = sum(1 for v in collected.values() if v.get("ok"))
    total_count = len(collected)
    print(f"[SPEECH] Data collection done in {elapsed:.1f}s — {success_count}/{total_count} sources OK")

    collected["_meta"] = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "success_count": success_count,
        "total_count": total_count,
    }
    return collected
