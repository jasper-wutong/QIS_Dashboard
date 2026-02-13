"""QIS SubBook Dashboard - Flask backend.

Serves a single-page dashboard that displays QIS SubBook data from
the latest EDSLib Excel file.  Single-ticker research is powered by
Alibaba Bailian RAG (阿里百炼知识库) for intelligent analysis.
"""

import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, render_template
from ticker_mapping import populate_names, resolve_sector, resolve_region, SECTOR_ORDER, SECTOR_ICONS

# 导入新闻聚合模块
from news.news_fetcher import fetch_all_news, fetch_category_news

# 导入百炼研究模块
from ali_bailian.bailian_research import research_ticker, chat_ticker, research_batch as bailian_research_batch

# -- Flask app -----------------------------------------------------------------
app = Flask(__name__)

# -- Helpers -------------------------------------------------------------------

def safe_val(v):
    """Convert a pandas value to a JSON-safe Python scalar."""
    if pd.isna(v):
        return None
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return None
    if isinstance(v, (int, float)):
        return v
    return str(v)


def to_records(frame: pd.DataFrame, columns: list) -> list:
    """Convert a DataFrame to a list-of-lists matching *columns*."""
    return [[safe_val(row.get(c)) for c in columns] for _, row in frame.iterrows()]


# -- Data loading & processing ------------------------------------------------

DATA_DIR = r"\\cicc.group\DFS\Pub\Workgrp\S_EQ Derivatives\3-部分共享\2-RB Formula\EDSLib_Realtime\EDSLib_Source\EDSLib_Realtime"
FILE_PREFIX = "EDSLib Realtime Result as of"


def load_data():
    """Read the most recent EDSLib Excel file and return (df_qis, date_str)."""
    files = sorted(
        (f for f in os.listdir(DATA_DIR) if f.startswith(FILE_PREFIX) and f.endswith(".xlsx")),
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"No '{FILE_PREFIX}*.xlsx' files found in {DATA_DIR}")
    filepath = os.path.join(DATA_DIR, files[0])
    print(f"Reading: {files[0]}")

    df = pd.read_excel(filepath, engine="openpyxl", header=2)
    df_qis = df[df["SubBook"].astype(str).str.contains("QIS", case=False, na=False)].reset_index(drop=True)
    date_str = files[0].replace(f"{FILE_PREFIX} ", "").replace(".xlsx", "")
    return df_qis, date_str


def aggregate_by_wind_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Merge rows sharing the same Wind Ticker (sum numerics, first for rest)."""
    if df.empty:
        return df
    sum_cols = [
        "Delta($)", "SOD Delta($)", "PDE Delta($)", "%Gamma($)", "PDE Gamma(%)",
        "Vega($)", "FX Vega($)", "Theta",
        "现货市值", "期货市值", "风险敞口", "T+1 风险敞口", "风险敞口(PDE)",
        "当日损益", "当日损益(PDE)", "当日损益(合约端)", "当日损益(对冲端)",
        "存续名义本金", "Delta Shares", "Open Shares", "Need To Trade",
        "Exposure pnl", "Gamma pnl", "Theta pnl", "Vega pnl",
        "Borrow pnl", "Residual",
    ]
    sum_cols = [c for c in sum_cols if c in df.columns]
    first_cols = [c for c in df.columns if c not in sum_cols and c != "Wind Ticker"]
    agg = {c: "sum" for c in sum_cols}
    agg.update({c: "first" for c in first_cols})
    return df.groupby("Wind Ticker", as_index=False).agg(agg)


def calc_summary(frame: pd.DataFrame) -> dict:
    """Compute aggregate summary metrics for a DataFrame slice."""
    def s(col):
        return float(frame[col].sum()) if col in frame.columns else 0.0
    return {
        "count": int(len(frame)),
        "delta": s("Delta($)"),
        "sod_delta": s("SOD Delta($)"),
        "vega": s("Vega($)"),
        "theta": s("Theta"),
        "gamma": s("%Gamma($)"),
        "exposure": s("风险敞口"),
        "pnl": s("当日损益"),
        "pnl_contract": s("当日损益(合约端)"),
        "pnl_hedge": s("当日损益(对冲端)"),
        "notional": s("存续名义本金"),
    }


def build_subbook_summaries(frame: pd.DataFrame, subbooks: list) -> dict:
    return {
        sb: calc_summary(frame[frame["SubBook"] == sb])
        for sb in subbooks
        if (frame["SubBook"] == sb).any()
    }


def process_data(df_qis: pd.DataFrame, date_str: str) -> dict:
    """Split, aggregate, summarise and return the full dashboard payload."""
    populate_names(df_qis)

    is_index = df_qis["Wind Ticker"].isna() & df_qis["涨跌幅"].notna()
    df_index = df_qis[is_index].reset_index(drop=True)
    df_other_raw = df_qis[~is_index].reset_index(drop=True)
    df_other = aggregate_by_wind_ticker(df_other_raw)

    # 添加板块分类和境内/境外分类
    df_other["_sector"] = df_other.apply(
        lambda row: resolve_sector(row.get("Wind Ticker"), row.get("标的物")),
        axis=1
    )
    df_other["_region"] = df_other.apply(
        lambda row: resolve_region(row.get("Wind Ticker"), row.get("标的物")),
        axis=1
    )

    columns = list(df_qis.columns)
    subbooks = sorted(df_qis["SubBook"].unique().tolist())

    for col in columns:
        if col not in df_index.columns:
            df_index[col] = None
        if col not in df_other.columns:
            df_other[col] = None

    # 按板块和境内/境外分组
    sector_data = {}
    sector_summaries = {}
    for sector in SECTOR_ORDER:
        df_sec = df_other[df_other["_sector"] == sector]
        if not df_sec.empty:
            # 分为境内和境外
            df_domestic = df_sec[df_sec["_region"] == "境内"]
            df_overseas = df_sec[df_sec["_region"] == "境外"]
            
            sector_data[sector] = {
                "domestic": to_records(df_domestic, columns) if not df_domestic.empty else [],
                "overseas": to_records(df_overseas, columns) if not df_overseas.empty else [],
            }
            
            # 汇总也用聚合后的数据，保证和表格显示一致
            sector_summaries[sector] = calc_summary(df_sec)

    return {
        "columns": columns,
        "subbooks": subbooks,
        "index_records": to_records(df_index, columns),
        "other_records": to_records(df_other, columns),
        "sector_data": sector_data,
        "sector_summaries": sector_summaries,
        "sector_order": [s for s in SECTOR_ORDER if s in sector_data],
        "sector_icons": SECTOR_ICONS,
        "index_summary": calc_summary(df_index),
        "other_summary": calc_summary(df_other_raw),
        "total_summary": calc_summary(df_qis),
        "index_sb_summary": build_subbook_summaries(df_index, subbooks),
        "other_sb_summary": build_subbook_summaries(df_other_raw, subbooks),
        "index_count": len(df_index),
        "other_raw_count": len(df_other_raw),
        "other_merged_count": len(df_other),
    }


# -- Research cache -----------------------------------------------------------
RESEARCH_CACHE: dict = {}   # {name: {ok, name, model, content, ...}}
CACHE_DATE: str = ""

# -- Load data on startup -----------------------------------------------------
print("Loading data...")
df_qis, DATE_STR = load_data()
DATA = process_data(df_qis, DATE_STR)
CACHE_DATE = DATE_STR
print(f"Data loaded: {DATE_STR}")
print(f"  指数: {DATA['index_count']} rows")
print(f"  其他标的物: {DATA['other_raw_count']} raw -> {DATA['other_merged_count']} merged")
print(f"  板块分类: {len(DATA['sector_order'])} 个板块")
for sec in DATA['sector_order']:
    sec_data = DATA['sector_data'][sec]
    domestic_cnt = len(sec_data.get('domestic', []))
    overseas_cnt = len(sec_data.get('overseas', []))
    print(f"    - {sec}: 境内 {domestic_cnt} / 境外 {overseas_cnt}")


# -- Routes --------------------------------------------------------------------

@app.route("/")
def dashboard():
    return render_template(
        "dashboard.html",
        date_str=DATE_STR,
        now_time=datetime.now().strftime("%H:%M"),
    )


@app.route("/api/data")
def api_data():
    return jsonify({
        "columns": DATA["columns"],
        "subbooks": DATA["subbooks"],
        "index_data": DATA["index_records"],
        "other_data": DATA["other_records"],
        "sector_data": DATA["sector_data"],
        "sector_summaries": DATA["sector_summaries"],
        "sector_order": DATA["sector_order"],
        "sector_icons": DATA["sector_icons"],
        "index_summary": DATA["index_summary"],
        "other_summary": DATA["other_summary"],
        "total_summary": DATA["total_summary"],
        "index_sb": DATA["index_sb_summary"],
        "other_sb": DATA["other_sb_summary"],
        "date_str": DATE_STR,
    })


@app.route("/api/refresh")
def refresh():
    global df_qis, DATE_STR, DATA, RESEARCH_CACHE, CACHE_DATE
    df_qis, DATE_STR = load_data()
    DATA = process_data(df_qis, DATE_STR)
    RESEARCH_CACHE.clear()
    CACHE_DATE = DATE_STR
    return jsonify({"status": "ok", "date_str": DATE_STR})


# -- News routes ---------------------------------------------------------------

@app.route("/news")
def news_page():
    """新闻展示页面"""
    return render_template(
        "news.html",
        date_str=DATE_STR,
        now_time=datetime.now().strftime("%H:%M"),
    )


@app.route("/api/news")
def api_news():
    """获取所有新闻源的新闻（分类返回）"""
    try:
        news_data = fetch_all_news(max_workers=10, timeout=30)
        return jsonify({
            "ok": True,
            "data": news_data
        })
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "data": None
        })


@app.route("/api/news/<category>")
def api_news_category(category):
    """获取单个分类的新闻"""
    try:
        if category not in ["finance", "hot", "tech"]:
            return jsonify({
                "ok": False,
                "error": f"未知分类: {category}，可选: finance, hot, tech",
                "data": None
            })
        
        news_data = fetch_category_news(category, max_workers=5, timeout=20)
        return jsonify({
            "ok": True,
            "data": news_data
        })
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "data": None
        })


# -- Single-ticker research via Bailian RAG -----------------------------------

RESEARCH_CLI = str(Path(__file__).resolve().parent / "research_cli.py")  # 保留，作为备用


@app.route("/api/research")
def api_research():
    """调用百炼RAG知识库进行标的研究分析。"""
    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "error": "missing 'name' parameter", "content": ""}), 400

    # Check cache
    if name in RESEARCH_CACHE:
        cached = dict(RESEARCH_CACHE[name])
        cached["cached"] = True
        return jsonify(cached)

    price = request.args.get("price", "NA").strip() or "NA"
    change = request.args.get("change", "NA").strip() or "NA"
    exposure = request.args.get("exposure", "NA").strip() or "NA"

    print(f"[RESEARCH] 调用百炼分析: {name} (price={price}, change={change}, exposure={exposure})")

    try:
        result = research_ticker(
            name=name,
            price=price,
            change=change,
            exposure=exposure,
        )
        
        # Cache successful results
        if result.get("ok") and result.get("content"):
            RESEARCH_CACHE[name] = result
            print(f"[RESEARCH] 分析完成: {name}")
        else:
            print(f"[RESEARCH] 分析失败: {name} - {result.get('error', 'unknown')}")
        
        return jsonify(result)

    except Exception as exc:
        print(f"[RESEARCH] 异常: {exc}")
        return jsonify({"ok": False, "name": name, "error": str(exc), "content": ""})


# -- Batch research via Bailian RAG -------------------------------------------

@app.route("/api/research/batch", methods=["POST"])
def api_research_batch():
    """批量研究多个标的（使用百炼RAG）。"""
    body = request.get_json(silent=True) or {}
    tickers = body.get("tickers", [])
    if not tickers:
        return jsonify({"ok": False, "error": "missing 'tickers' array", "results": []}), 400

    names = [t.get("name", "?") for t in tickers]
    print(f"[BATCH] 收到 {len(tickers)} 个标的: {', '.join(names)}")

    # Separate cached vs uncached
    cached_results = []
    uncached_tickers = []
    for t in tickers:
        name = t.get("name", "").strip()
        if not name:
            continue
        if name in RESEARCH_CACHE:
            entry = dict(RESEARCH_CACHE[name])
            entry["cached"] = True
            cached_results.append(entry)
        else:
            uncached_tickers.append(t)

    if cached_results:
        print(f"[BATCH]   {len(cached_results)} 已缓存, {len(uncached_tickers)} 待分析")

    # If all cached, return immediately
    if not uncached_tickers:
        print("[BATCH]   全部命中缓存")
        return jsonify({"ok": True, "results": cached_results, "all_cached": True})

    # 使用百炼批量研究
    uncached_names = [t.get("name", "?") for t in uncached_tickers]
    print(f"[BATCH]   调用百炼分析: {', '.join(uncached_names)}")
    
    try:
        new_results = bailian_research_batch(uncached_tickers)
        
        # Cache successful individual results
        ok_count = 0
        for r in new_results:
            if r.get("ok") and r.get("content"):
                RESEARCH_CACHE[r["name"]] = r
                ok_count += 1
        print(f"[BATCH]   完成: {ok_count}/{len(new_results)} 成功, model=bailian-rag")

        return jsonify({
            "ok": True,
            "model": "bailian-rag",
            "results": cached_results + new_results,
        })

    except Exception as exc:
        print(f"[BATCH]   错误: {exc}")
        return jsonify({"ok": False, "error": str(exc), "results": cached_results})


# -- Chat with research context via Bailian RAG -------------------------------

@app.route("/api/research/chat", methods=["POST"])
def api_research_chat():
    """与百炼进行多轮对话，讨论特定标的。"""
    body = request.get_json(silent=True) or {}
    name = body.get("name", "").strip()
    message = body.get("message", "").strip()
    history = body.get("history", [])
    
    if not name or not message:
        return jsonify({"ok": False, "error": "missing 'name' or 'message'", "content": ""}), 400
    
    price = body.get("price", "NA")
    change = body.get("change", "NA")
    exposure = body.get("exposure", "NA")
    
    print(f"[CHAT] 百炼对话: {name} - {message[:50]}...")
    
    try:
        result = chat_ticker(
            name=name,
            message=message,
            history=history,
            price=str(price),
            change=str(change),
            exposure=str(exposure),
        )
        
        if result.get("ok"):
            print(f"[CHAT] 对话完成: {name}")
        else:
            print(f"[CHAT] 对话失败: {name} - {result.get('error', 'unknown')}")
        
        return jsonify(result)

    except Exception as exc:
        print(f"[CHAT] 异常: {exc}")
        return jsonify({"ok": False, "name": name, "error": str(exc), "content": ""})


# -- Entrypoint ----------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5050)
