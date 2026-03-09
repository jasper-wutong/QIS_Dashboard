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
from ticker_mapping import populate_names, resolve_sector, resolve_region, resolve_instrument_type, SECTOR_ORDER, SECTOR_ICONS, resolve_name_to_wind_ticker

# 导入新闻聚合模块
from news.news_fetcher import fetch_all_news, fetch_category_news

# 导入百炼研究模块
from ali_bailian.bailian_research import research_ticker, chat_ticker, research_batch as bailian_research_batch

# 导入市场数据模块
from market_data import fetch_market_data, clear_cache as clear_market_cache, db_stats as market_db_stats, fetch_realtime_price

# 导入大宗商品研究模块
from commodity_research import build_research_panel, get_sector_for_ticker, get_term_structure_tickers

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
        "T+1 Need to Trade", "Expiring Delta",
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
        "exposure_pnl": s("Exposure pnl"),
        "delta_shares": s("Delta Shares"),
        "open_shares": s("Open Shares"),
        "need_to_trade": s("Need To Trade"),
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

    # ── QIS BOOK: 按合约类型拆分（境内期货 / 境外期货 / 境内ETF）
    df_other["_instrument_type"] = df_other.apply(
        lambda row: resolve_instrument_type(row.get("Wind Ticker")), axis=1
    )
    df_domestic_futures = df_other[df_other["_instrument_type"] == "境内期货"]
    df_overseas_futures = df_other[df_other["_instrument_type"] == "境外期货"]
    df_domestic_etf = df_other[df_other["_instrument_type"] == "境内ETF"]

    book_data = {
        "total": to_records(df_other, columns),
        "domestic_futures": to_records(df_domestic_futures, columns) if not df_domestic_futures.empty else [],
        "overseas_futures": to_records(df_overseas_futures, columns) if not df_overseas_futures.empty else [],
        "domestic_etf": to_records(df_domestic_etf, columns) if not df_domestic_etf.empty else [],
    }
    book_summaries = {
        "total": calc_summary(df_other),
        "domestic_futures": calc_summary(df_domestic_futures),
        "overseas_futures": calc_summary(df_overseas_futures),
        "domestic_etf": calc_summary(df_domestic_etf),
    }

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
        "book_data": book_data,
        "book_summaries": book_summaries,
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
    clear_market_cache()
    CACHE_DATE = DATE_STR
    return jsonify({"status": "ok", "date_str": DATE_STR})


# -- Analysis routes -----------------------------------------------------------

# Path setup for edslib wutong-tools (done once at module level)
import sys as _sys
from pathlib import Path as _Path
_WUTONG_TOOLS = str(_Path(r"D:\edslib\wutong-tools"))
_DATA_TOOLS    = str(_Path(r"D:\edslib\wutong-tools\data_tools"))
for _p in [_WUTONG_TOOLS, _DATA_TOOLS]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


@app.route("/analysis")
def analysis_page():
    return render_template(
        "analysis.html",
        date_str=DATE_STR,
        now_time=datetime.now().strftime("%H:%M"),
    )


@app.route("/api/analysis/maturity-timeline")
def api_maturity_timeline():
    """返回 QIS book (book_id=10) trade 合约按到期日聚合的名义本金时间轴数据."""
    import os as _os
    from collections import defaultdict

    # 绕过代理，访问内网 Hedge API
    _proxy_keys = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
    _saved = {k: _os.environ.get(k) for k in _proxy_keys}
    _saved_no = _os.environ.get("NO_PROXY", "")
    for k in _proxy_keys:
        _os.environ[k] = ""
    _os.environ["NO_PROXY"] = "*"
    _os.environ["no_proxy"] = "*"

    try:
        from generate_fullData import HedgeAPIClient
        date_str_today = datetime.now().strftime("%Y-%m-%d")
        client = HedgeAPIClient(timeout=30)
        raw = client.get_contracts(date_str_today, ["10"], verbose=False)
        positions = client._extract_positions(raw)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e), "timeline": []})
    finally:
        for k, v in _saved.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v
        _os.environ["NO_PROXY"] = _saved_no
        _os.environ["no_proxy"] = _saved_no

    # 仅保留 trade，且到期日 > 今天
    from datetime import date as _date
    today = _date.today()
    trades = [p for p in positions if p.get("positionType") == "trade"]

    by_date = defaultdict(lambda: {"total": 0.0, "underlyings": defaultdict(float)})
    for t in trades:
        exp_raw = (t.get("expiration") or "")[:10]
        if not exp_raw:
            continue
        try:
            exp_date = datetime.strptime(exp_raw, "%Y-%m-%d").date()
        except ValueError:
            continue
        if exp_date <= today:
            continue
        notional = abs(t.get("notional") or 0.0)
        if notional == 0:
            continue
        underlying = t.get("underlying") or "Unknown"
        by_date[exp_raw]["total"] += notional
        by_date[exp_raw]["underlyings"][underlying] += notional

    timeline = [
        {
            "date": d,
            "total": by_date[d]["total"],
            "underlyings": dict(
                sorted(by_date[d]["underlyings"].items(), key=lambda x: -x[1])
            ),
        }
        for d in sorted(by_date.keys())
    ]

    return jsonify({
        "ok": True,
        "date": date_str_today,
        "trade_count": len(trades),
        "timeline": timeline,
    })


# -- QIS BOOK routes -----------------------------------------------------------

@app.route("/qis-book")
def qis_book_page():
    """QIS BOOK 拆解展示页面"""
    return render_template(
        "qis_book.html",
        date_str=DATE_STR,
        now_time=datetime.now().strftime("%H:%M"),
    )


@app.route("/api/book-data")
def api_book_data():
    """返回 QIS BOOK 拆解数据（境内期货 / 境外期货 / 境内ETF）"""
    return jsonify({
        "columns": DATA["columns"],
        "book_data": DATA["book_data"],
        "book_summaries": DATA["book_summaries"],
        "date_str": DATE_STR,
    })


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


# -- Market data (charts) ------------------------------------------------------

@app.route("/api/market-data")
def api_market_data():
    """获取标的历史行情 + 技术指标 + 基本面数据。"""
    ticker = request.args.get("ticker", "").strip()
    name = request.args.get("name", "").strip()
    days = int(request.args.get("days", "180"))

    # 如果只传了 name 没传 ticker, 从数据中反查
    if not ticker and name:
        ticker = resolve_name_to_wind_ticker(
            name,
            df_data=DATA["other_records"],
            columns=DATA["columns"],
        )

    if not ticker:
        return jsonify({"ok": False, "error": "缺少 ticker 或 name 参数"}), 400

    print(f"[MARKET_DATA] 获取行情: {name or ticker} (ticker={ticker}, days={days})")

    try:
        result = fetch_market_data(wind_ticker=ticker, name=name, days=days)
        return jsonify(result)
    except Exception as exc:
        print(f"[MARKET_DATA] 异常: {exc}")
        return jsonify({"ok": False, "ticker": ticker, "name": name, "error": str(exc)})


@app.route("/api/market-db-stats")
def api_market_db_stats():
    """返回 SQLite 市场数据库的统计信息 (调试用)。"""
    return jsonify(market_db_stats())


@app.route("/api/realtime-price")
def api_realtime_price():
    """获取标的实时行情快照 (Wind wsq / Bloomberg ReferenceData)。"""
    ticker = request.args.get("ticker", "").strip()
    name = request.args.get("name", "").strip()

    if not ticker and name:
        ticker = resolve_name_to_wind_ticker(
            name,
            df_data=DATA["other_records"],
            columns=DATA["columns"],
        )

    if not ticker:
        return jsonify({"ok": False, "error": "缺少 ticker 或 name 参数"}), 400

    try:
        result = fetch_realtime_price(wind_ticker=ticker, name=name)
        return jsonify(result)
    except Exception as exc:
        print(f"[REALTIME] 异常: {exc}")
        return jsonify({"ok": False, "ticker": ticker, "name": name, "error": str(exc)})


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


# -- Commodity Research Routes -----------------------------------------------

@app.route("/api/research-factors")
def api_research_factors():
    """返回品种专属研究因子面板数据（量化因子 + 关键驱动 + 综合评分）。"""
    ticker = request.args.get("ticker", "").strip()
    name   = request.args.get("name", "").strip()
    sector = request.args.get("sector", "").strip() or None

    if not ticker and name:
        ticker = resolve_name_to_wind_ticker(
            name, df_data=DATA["other_records"], columns=DATA["columns"]
        )
    if not ticker:
        return jsonify({"ok": False, "error": "缺少 ticker 或 name 参数"}), 400

    print(f"[RESEARCH_FACTORS] {name or ticker} (ticker={ticker})")
    try:
        # 先获取市场数据（OHLCV + 基本面）
        mdata = fetch_market_data(wind_ticker=ticker, name=name, days=180)
        ohlcv = mdata.get("ohlcv", []) if mdata.get("ok") else []
        fundamentals = mdata.get("fundamentals", {}) if mdata.get("ok") else {}

        # 自动解析板块
        if not sector:
            sector = get_sector_for_ticker(ticker)

        panel = build_research_panel(
            wind_ticker=ticker,
            sector=sector,
            ohlcv=ohlcv,
            fundamentals=fundamentals,
        )
        panel["ok"] = True
        panel["ticker"] = ticker
        panel["name"] = name
        return jsonify(panel)
    except Exception as exc:
        print(f"[RESEARCH_FACTORS] 异常: {exc}")
        return jsonify({"ok": False, "ticker": ticker, "name": name, "error": str(exc)})


@app.route("/api/term-structure")
def api_term_structure():
    """获取期货合约期限结构（前N个月合约的当前价格）。"""
    ticker = request.args.get("ticker", "").strip()
    name   = request.args.get("name", "").strip()

    if not ticker and name:
        ticker = resolve_name_to_wind_ticker(
            name, df_data=DATA["other_records"], columns=DATA["columns"]
        )
    if not ticker:
        return jsonify({"ok": False, "error": "缺少 ticker 参数"}), 400

    ts_tickers = get_term_structure_tickers(ticker)
    if not ts_tickers:
        return jsonify({"ok": False, "error": f"暂无 {ticker} 的期限结构配置", "curve": []})

    print(f"[TERM_STRUCTURE] {ticker} → {ts_tickers}")
    curve_points = []
    errors = []
    for i, ts_ticker in enumerate(ts_tickers):
        try:
            rt = fetch_realtime_price(wind_ticker=ts_ticker, name="")
            if rt.get("ok") and rt.get("price") is not None:
                curve_points.append({
                    "contract": ts_ticker,
                    "month_n": i + 1,
                    "label": f"M{i+1}",
                    "price": rt["price"],
                    "change": rt.get("change"),
                    "change_pct": rt.get("change_pct"),
                })
            else:
                err_msg = rt.get("error", "无数据")
                errors.append(f"M{i+1}({ts_ticker}): {err_msg}")
                print(f"[TERM_STRUCTURE] {ts_ticker} 无数据: {err_msg}")
        except Exception as e:
            errors.append(f"M{i+1}({ts_ticker}): {e}")
            print(f"[TERM_STRUCTURE] {ts_ticker} 异常: {e}")
            continue

    # 判断数据源类型
    is_domestic = any(t.endswith(('.SHF', '.DCE', '.CZC', '.INE', '.CFE')) for t in ts_tickers)
    source_hint = "Wind (境内)" if is_domestic else "Bloomberg (境外)"

    ok = len(curve_points) >= 2  # 至少需要2个点才能画曲线
    return jsonify({
        "ok": ok,
        "ticker": ticker,
        "curve": curve_points,
        "count": len(curve_points),
        "total": len(ts_tickers),
        "source": source_hint,
        "errors": errors[:3] if errors else [],  # 最多返回3条错误信息
        "error": (
            f"仅获取到 {len(curve_points)}/{len(ts_tickers)} 个合约价格。"
            f"数据源: {source_hint}。"
            + (f" 首个错误: {errors[0]}" if errors else "")
        ) if not ok else None,
    })


# -- Entrypoint ----------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5050)
