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

import jinja2
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

# 导入中金研究模块
from cicc_research import (
    fetch_report_list as cicc_report_list,
    fetch_report_detail as cicc_report_detail,
    search_reports as cicc_search_reports,
    test_connection as cicc_test_connection,
    save_cookies as cicc_save_cookies,
    get_cookies_status as cicc_cookies_status,
    clear_cache as cicc_clear_cache,
    get_categories as cicc_get_categories,
)

# 导入早会发言模块
from speech import generate_morning_speech, get_cached_speech
from speech.generator import get_generation_status

# 导入PDE Scenario Ladder模块
from pde_ladder.loader import find_latest_qis_file, load_pde_ladder, SCENARIO_LABELS

# -- Flask app -----------------------------------------------------------------
app = Flask(__name__)
# 支持子模块自带的 templates 目录（如 cicc_research/templates/）
app.jinja_loader = jinja2.ChoiceLoader([
    app.jinja_loader,
    jinja2.FileSystemLoader(os.path.join(os.path.dirname(__file__), "cicc_research", "templates")),
])

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
import time as _time
from pathlib import Path as _Path
_WUTONG_TOOLS = str(_Path(r"D:\edslib\wutong-tools"))
_DATA_TOOLS    = str(_Path(r"D:\edslib\wutong-tools\data_tools"))
for _p in [_WUTONG_TOOLS, _DATA_TOOLS]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

# 5-min in-memory cache so parallel front-end calls share one Hedge API fetch
_CONTRACTS_CACHE: dict = {"data": None, "ts": 0.0}
_CONTRACTS_CACHE_TTL = 300  # seconds


def _fetch_qis_contracts():
    """Fetch all QIS book-10 contracts, cached 5 min to avoid repeated Hedge API hits."""
    import os as _os
    now = _time.time()
    if _CONTRACTS_CACHE["data"] is not None and (now - _CONTRACTS_CACHE["ts"]) < _CONTRACTS_CACHE_TTL:
        return _CONTRACTS_CACHE["data"], None

    _keys = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
    _saved = {k: _os.environ.get(k) for k in _keys}
    _saved_no = _os.environ.get("NO_PROXY", "")
    for k in _keys:
        _os.environ[k] = ""
    _os.environ["NO_PROXY"] = "*"
    _os.environ["no_proxy"] = "*"
    try:
        from generate_fullData import HedgeAPIClient
        client = HedgeAPIClient(timeout=30)
        raw = client.get_contracts(datetime.now().strftime("%Y-%m-%d"), ["10"], verbose=False)
        positions = client._extract_positions(raw)
        _CONTRACTS_CACHE["data"] = positions
        _CONTRACTS_CACHE["ts"] = now
        return positions, None
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, str(e)
    finally:
        for k, v in _saved.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v
        _os.environ["NO_PROXY"] = _saved_no
        _os.environ["no_proxy"] = _saved_no


@app.route("/analysis")
def analysis_page():
    return render_template(
        "analysis.html",
        date_str=DATE_STR,
        now_time=datetime.now().strftime("%H:%M"),
    )


@app.route("/api/analysis/maturity-timeline")
def api_maturity_timeline():
    """Per-expiry aggregation: notional, net delta, net gamma, net vega."""
    from collections import defaultdict
    from datetime import date as _date

    positions, err = _fetch_qis_contracts()
    if err:
        return jsonify({"ok": False, "error": err, "timeline": []})

    today = _date.today()
    trades = [p for p in positions if p.get("positionType") == "trade"]

    by_date = defaultdict(lambda: {
        "total": 0.0, "delta_net": 0.0, "gamma_net": 0.0, "vega_net": 0.0,
        "underlyings": defaultdict(float), "delta_by_u": defaultdict(float),
    })

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
        notional   = abs(t.get("notional") or 0.0)
        delta      = t.get("delta") or 0.0
        gamma      = t.get("gamma") or 0.0
        vega       = t.get("vega")  or 0.0
        underlying = t.get("underlying") or "Unknown"
        if notional == 0 and delta == 0:
            continue
        by_date[exp_raw]["total"]      += notional
        by_date[exp_raw]["delta_net"]  += delta
        by_date[exp_raw]["gamma_net"]  += gamma
        by_date[exp_raw]["vega_net"]   += vega
        by_date[exp_raw]["underlyings"][underlying]  += notional
        by_date[exp_raw]["delta_by_u"][underlying]   += delta

    timeline = [
        {
            "date":      d,
            "total":     by_date[d]["total"],
            "delta_net": by_date[d]["delta_net"],
            "gamma_net": by_date[d]["gamma_net"],
            "vega_net":  by_date[d]["vega_net"],
            "underlyings": dict(sorted(by_date[d]["underlyings"].items(), key=lambda x: -x[1])),
            "delta_by_u":  dict(sorted(by_date[d]["delta_by_u"].items(),  key=lambda x: -abs(x[1]))),
        }
        for d in sorted(by_date.keys())
    ]

    return jsonify({
        "ok": True,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "trade_count": len(trades),
        "timeline": timeline,
    })


@app.route("/api/analysis/contracts-detail")
def api_contracts_detail():
    """Per-trade detail for bubble / scatter visualizations."""
    from datetime import date as _date

    positions, err = _fetch_qis_contracts()
    if err:
        return jsonify({"ok": False, "error": err, "trades": []})

    today = _date.today()
    trades_out = []
    # collect unique underlyings with spot for selector
    underlying_meta: dict = {}

    for t in positions:
        if t.get("positionType") != "trade":
            continue
        exp_raw = (t.get("expiration") or "")[:10]
        if not exp_raw:
            continue
        try:
            exp_date = datetime.strptime(exp_raw, "%Y-%m-%d").date()
        except ValueError:
            continue
        if exp_date <= today:
            continue

        structure  = t.get("structure") or ""
        underlying = t.get("underlying") or "Unknown"
        init_price = t.get("initPrice") or 0.0
        spot       = t.get("spot") or 0.0
        delta      = t.get("delta") or 0.0
        notional   = abs(t.get("notional") or 0.0)

        # Absolute KO barrier level
        ko_raw = t.get("ko_prices")
        ko_abs = (ko_raw * init_price) if (isinstance(ko_raw, (int, float)) and ko_raw > 0 and init_price > 0) else None

        # Absolute KI barrier level
        ki_raw = t.get("ki_barrier")
        ki_abs = (ki_raw * init_price) if (isinstance(ki_raw, (int, float)) and ki_raw > 0 and init_price > 0) else None

        # Y level for bubble chart (strike or barrier)
        strike_abs = t.get("strikeAbs") or 0.0
        y_level = None
        y_type  = None
        if strike_abs > 0 and structure in ("vanilla", "asian"):
            y_level = strike_abs
            y_type  = "strike"
        elif ko_abs and ko_abs > 0 and structure in ("sharkfin", "snowball"):
            y_level = ko_abs
            y_type  = "ko_barrier"

        if y_level is None:
            continue  # skip if no meaningful price level

        trades_out.append({
            "id":          t.get("id"),
            "underlying":  underlying,
            "expiration":  exp_raw,
            "structure":   structure,
            "callPut":     t.get("callPut") or "",
            "y_level":     y_level,
            "y_type":      y_type,
            "spot":        spot,
            "initPrice":   init_price,
            "delta":       delta,
            "gamma":       t.get("gamma") or 0.0,
            "vega":        t.get("vega") or 0.0,
            "notional":    notional,
            "ko_abs":      ko_abs,
            "ki_abs":      ki_abs,
            "is_barrier":  bool(t.get("is_barrier", False)),
            "annual_coupon": t.get("annual_coupon") or 0.0,
        })

        if underlying not in underlying_meta and spot > 0:
            underlying_meta[underlying] = {"spot": spot, "count": 0, "total_delta": 0.0}
        if underlying in underlying_meta:
            underlying_meta[underlying]["count"] += 1
            underlying_meta[underlying]["total_delta"] += abs(delta)

    # Sort underlyings by total delta descending for the dropdown
    sorted_underlyings = sorted(underlying_meta.items(), key=lambda x: -x[1]["total_delta"])

    return jsonify({
        "ok":          True,
        "trades":      trades_out,
        "underlyings": [{"code": k, "spot": v["spot"], "count": v["count"],
                         "total_delta": v["total_delta"]} for k, v in sorted_underlyings],
    })


# -- Cross Gamma API (book-level, daily JSON) ----------------------------------

from cross_gamma import find_latest_file, find_all_files, load_cross_gamma_data, aggregate_book


@app.route("/api/analysis/cross-gamma")
def api_cross_gamma():
    """Book-level Cross Gamma matrix (pre-computed cash gamma in CNY).

    Iterates through available daily files (newest first) and skips files
    that contain no active trades (e.g. placeholder files with all-zero
    values) so the dashboard always shows the most recent *real* data.
    """
    candidates = find_all_files()
    if not candidates:
        return jsonify({"ok": False, "error": "No cross gamma JSON files found (network share unreachable, no local copy)"})

    # Try each candidate (newest first); skip files with no active trades
    last_error = None
    for date_str, fpath in candidates:
        raw_data, err = load_cross_gamma_data(fpath)
        if err:
            last_error = err
            print(f"[CROSS_GAMMA] skip {fpath.name}: {err}")
            continue

        result = aggregate_book(raw_data)
        stats = result["stats"]

        if stats["active_trades"] == 0:
            print(f"[CROSS_GAMMA] skip {fpath.name}: 0 active trades out of {stats['total_trades']}")
            continue

        print(f"[CROSS_GAMMA] date={date_str}  trades={stats['active_trades']}/{stats['total_trades']}  "
              f"underlyings={stats['n_underlyings']}  file={fpath.name}")

        return jsonify({
            "ok":              True,
            "date":            date_str,
            "currency":        "CNY",
            "tickers":         result["tickers"],
            "asset_classes":   result["asset_classes"],
            "matrix":          result["matrix"],
            "top_pairs":       result["top_pairs"],
            "trade_contribs":  result["trade_contribs"],
            "per_asset":       result["per_asset"],
            "total_trades":    stats["total_trades"],
            "active_trades":   stats["active_trades"],
            "n_underlyings":   stats["n_underlyings"],
            "total_gamma":     stats["total_gamma"],
            "total_abs_gamma": stats["total_abs_gamma"],
        })

    # All candidates exhausted
    return jsonify({"ok": False, "error": last_error or "All cross gamma files contain no active trades"})


# -- PDE Scenario Ladder -------------------------------------------------------

@app.route("/api/analysis/pde-ladder")
def api_pde_ladder():
    """PDE scenario ladder: book-level aggregated risks by spot bump.

    Reads the latest SPOT_LADDER_pde_QIS_*_raw_results.xlsx,
    sums each scenario column across all trades per sheet (risk type),
    and returns a matrix of scenarios × risks.
    """
    filepath, meta = find_latest_qis_file()
    if filepath is None:
        return jsonify({"ok": False, "error": meta})

    result, err = load_pde_ladder(filepath)
    if err:
        return jsonify({"ok": False, "error": err})

    print(f"[PDE_LADDER] date={result['date']}  trades={result['trade_count']}  "
          f"risks={result['risks']}  file={result['file_name']}")

    return jsonify({"ok": True, **result})


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


# -- QIS Index History ---------------------------------------------------------

_QIS_INDEX_CACHE = {}  # {name: result}


def _fetch_qis_index_from_hmc(index_name, start_date, end_date):
    """尝试从 HMC 获取 QIS 指数历史数据。"""
    _HMC_HELPER = str(Path(__file__).resolve().parent / "hmc_database" / "hmc_helper.py")
    _VENV_PY = str(Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe")

    if not os.path.isfile(_VENV_PY) or not os.path.isfile(_HMC_HELPER):
        return None

    import json as _json

    env = os.environ.copy()
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
              "grpc_proxy", "GRPC_PROXY"):
        env.pop(k, None)
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"

    start_ddb = start_date.replace("-", ".")
    end_ddb = end_date.replace("-", ".")

    # 尝试多种 SEC_ID 格式
    candidates = [
        index_name.upper() + ".WI",   # e.g. ARES2PRO.WI
        index_name.upper(),            # e.g. ARES2PRO
    ]

    # 尝试多张表 (table_name, date_col)
    tables = [
        ("dfs://HQUOT_CENTER_EOD", "CH_INDEX_DAY_QUOT", "TRADE_DT"),
        ("dfs://HQUOT_CENTER_EOD", "FUT_DAY_QUOT",       "TRAN_DATE"),
    ]

    for db, table, date_col in tables:
        for sec_id in candidates:
            sql = (
                "select {date_col}, OPEN_PRC, HIGH_PRC, LOW_PRC, CLOSE_PRC, TX_QTY "
                "from loadTable('{db}', '{table}') "
                "where {date_col} >= {start} and {date_col} <= {end} and SEC_ID = '{sec_id}' "
                "order by {date_col} asc"
            ).format(db=db, table=table, date_col=date_col, start=start_ddb, end=end_ddb, sec_id=sec_id)

            try:
                result = subprocess.run(
                    [_VENV_PY, _HMC_HELPER, "--mode", "query", "--sql", sql],
                    capture_output=True, text=True, timeout=60, env=env,
                )
            except Exception:
                continue

            stdout = result.stdout.strip()
            if not stdout:
                continue

            # 提取最后一行 JSON
            for line in reversed(stdout.split("\n")):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        data = _json.loads(line)
                    except ValueError:
                        continue
                    if data.get("ok") and data.get("data") and len(data["data"]) > 5:
                        ohlcv = []
                        for row in data["data"]:
                            date_raw = row.get(date_col)  # TRADE_DT or TRAN_DATE
                            date_str = str(date_raw)[:10] if date_raw else None
                            close_val = row.get("CLOSE_PRC")
                            if not date_str or close_val is None:
                                continue
                            ohlcv.append({
                                "date": date_str,
                                "open": row.get("OPEN_PRC"),
                                "high": row.get("HIGH_PRC"),
                                "low": row.get("LOW_PRC"),
                                "close": close_val,
                                "volume": row.get("TX_QTY"),
                            })
                        if ohlcv:
                            print(f"[QIS_INDEX] HMC 找到 {index_name} → {sec_id} ({table}): {len(ohlcv)} 条")
                            return {"ok": True, "source": "hmc", "ohlcv": ohlcv, "sec_id": sec_id}
                    break

    return None


def _fetch_qis_index_from_wind(index_name, start_date, end_date):
    """尝试从 Wind 获取 QIS 指数历史数据。"""
    _WIND_PY = r"C:\Users\wutong6\AppData\Local\Programs\Python\Python37\python.exe"
    _WIND_HELPER = str(Path(__file__).resolve().parent / "wind_helper.py")

    if not os.path.isfile(_WIND_PY) or not os.path.isfile(_WIND_HELPER):
        return None

    import json as _json

    # 尝试多种 Wind ticker 格式
    candidates = [
        index_name.upper() + ".WI",   # 万得自定义指数
        index_name.upper(),
    ]

    for ticker in candidates:
        try:
            result = subprocess.run(
                [_WIND_PY, _WIND_HELPER,
                 "--mode", "history",
                 "--ticker", ticker,
                 "--start", start_date,
                 "--end", end_date],
                capture_output=True, text=True, timeout=60,
            )
        except Exception:
            continue

        stdout = result.stdout.strip()
        if not stdout:
            continue

        for line in reversed(stdout.split("\n")):
            line = line.strip()
            if line.startswith("{"):
                try:
                    data = _json.loads(line)
                except ValueError:
                    continue
                if data.get("ok") and data.get("ohlcv") and len(data["ohlcv"]) > 5:
                    print(f"[QIS_INDEX] Wind 找到 {index_name} → {ticker}: {len(data['ohlcv'])} 条")
                    return {"ok": True, "source": "wind", "ohlcv": data["ohlcv"], "ticker": ticker}
                break

    return None


def _compute_index_analytics(ohlcv):
    """计算 QIS 指数分析指标。"""
    import numpy as np
    from datetime import datetime as _dt, timedelta as _td

    if not ohlcv or len(ohlcv) < 2:
        return {}

    df = pd.DataFrame(ohlcv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])

    if len(df) < 2:
        return {}

    close = df["close"].values
    dates = df["date"].values
    latest_close = close[-1]
    latest_date = pd.Timestamp(dates[-1])
    today = pd.Timestamp(_dt.today())

    # ── 涨跌幅计算 ──
    def pct_change_from_date(target_date):
        mask = df["date"] <= target_date
        if mask.any():
            ref = df.loc[mask, "close"].iloc[-1]
            return float((latest_close - ref) / ref) if ref != 0 else None
        return None

    def pct_change_last_n_days(n):
        target = latest_date - pd.Timedelta(days=n)
        return pct_change_from_date(target)

    analytics = {}

    # 近期涨跌幅
    chg_1d = float((close[-1] - close[-2]) / close[-2]) if len(close) >= 2 and close[-2] != 0 else None
    analytics["chg_1d"] = chg_1d
    analytics["chg_1w"] = pct_change_last_n_days(7)
    analytics["chg_1m"] = pct_change_last_n_days(30)
    analytics["chg_3m"] = pct_change_last_n_days(90)
    analytics["chg_6m"] = pct_change_last_n_days(180)
    analytics["chg_1y"] = pct_change_last_n_days(365)
    analytics["chg_3y"] = pct_change_last_n_days(365 * 3)
    analytics["chg_ytd"] = pct_change_from_date(pd.Timestamp(_dt(latest_date.year, 1, 1)))

    # 成立以来涨跌幅
    if close[0] != 0:
        analytics["chg_inception"] = float((latest_close - close[0]) / close[0])
    else:
        analytics["chg_inception"] = None

    # ── 最新价 / 最高 / 最低 ──
    analytics["latest_close"] = float(latest_close)
    analytics["latest_date"] = str(latest_date.date())
    analytics["first_date"] = str(pd.Timestamp(dates[0]).date())
    analytics["all_time_high"] = float(np.nanmax(close))
    analytics["all_time_low"] = float(np.nanmin(close))

    # 距离最高点回撤
    if analytics["all_time_high"] != 0:
        analytics["drawdown_from_ath"] = float((latest_close - analytics["all_time_high"]) / analytics["all_time_high"])
    else:
        analytics["drawdown_from_ath"] = None

    # ── 最大回撤 ──
    cummax = np.maximum.accumulate(close)
    drawdowns = (close - cummax) / cummax
    analytics["max_drawdown"] = float(np.nanmin(drawdowns))

    # 最大回撤的日期区间
    dd_end_idx = int(np.nanargmin(drawdowns))
    dd_start_idx = int(np.nanargmax(close[:dd_end_idx + 1])) if dd_end_idx > 0 else 0
    analytics["max_dd_start"] = str(pd.Timestamp(dates[dd_start_idx]).date())
    analytics["max_dd_end"] = str(pd.Timestamp(dates[dd_end_idx]).date())

    # ── 年化收益率 ──
    total_days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
    if total_days > 0 and close[0] > 0:
        total_return = latest_close / close[0]
        years = total_days / 365.25
        analytics["annualized_return"] = float(total_return ** (1 / years) - 1) if years > 0 else None
    else:
        analytics["annualized_return"] = None

    # ── 年化波动率 ──
    daily_returns = np.diff(close) / close[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]
    if len(daily_returns) > 10:
        analytics["annualized_vol"] = float(np.std(daily_returns) * np.sqrt(252))
        analytics["daily_vol"] = float(np.std(daily_returns))
    else:
        analytics["annualized_vol"] = None
        analytics["daily_vol"] = None

    # ── Sharpe Ratio (假设无风险利率 2.5%) ──
    rf = 0.025
    if analytics.get("annualized_return") is not None and analytics.get("annualized_vol") and analytics["annualized_vol"] > 0:
        analytics["sharpe_ratio"] = float((analytics["annualized_return"] - rf) / analytics["annualized_vol"])
    else:
        analytics["sharpe_ratio"] = None

    # ── Calmar Ratio (年化收益 / 最大回撤) ──
    if analytics.get("annualized_return") is not None and analytics["max_drawdown"] < 0:
        analytics["calmar_ratio"] = float(analytics["annualized_return"] / abs(analytics["max_drawdown"]))
    else:
        analytics["calmar_ratio"] = None

    # ── 近1年最大回撤 ──
    mask_1y = df["date"] >= (latest_date - pd.Timedelta(days=365))
    if mask_1y.sum() > 10:
        close_1y = df.loc[mask_1y, "close"].values
        cummax_1y = np.maximum.accumulate(close_1y)
        dd_1y = (close_1y - cummax_1y) / cummax_1y
        analytics["max_drawdown_1y"] = float(np.nanmin(dd_1y))
    else:
        analytics["max_drawdown_1y"] = None

    # ── 近1年波动率 ──
    if mask_1y.sum() > 20:
        close_1y = df.loc[mask_1y, "close"].values
        ret_1y = np.diff(close_1y) / close_1y[:-1]
        ret_1y = ret_1y[np.isfinite(ret_1y)]
        analytics["vol_1y"] = float(np.std(ret_1y) * np.sqrt(252))
    else:
        analytics["vol_1y"] = None

    # ── 偏度 / 峰度 ──
    if len(daily_returns) > 30:
        try:
            from scipy import stats as _stats
            analytics["skewness"] = float(_stats.skew(daily_returns))
            analytics["kurtosis"] = float(_stats.kurtosis(daily_returns))
        except ImportError:
            # scipy 未安装, 使用 numpy 手动计算
            mean_r = np.mean(daily_returns)
            std_r = np.std(daily_returns, ddof=0)
            if std_r > 0:
                n = len(daily_returns)
                analytics["skewness"] = float(np.mean(((daily_returns - mean_r) / std_r) ** 3))
                analytics["kurtosis"] = float(np.mean(((daily_returns - mean_r) / std_r) ** 4) - 3)
            else:
                analytics["skewness"] = None
                analytics["kurtosis"] = None
    else:
        analytics["skewness"] = None
        analytics["kurtosis"] = None

    # ── 连续上涨 / 下跌天数 ──
    if len(daily_returns) > 0:
        streak = 0
        if daily_returns[-1] > 0:
            for r in reversed(daily_returns):
                if r > 0:
                    streak += 1
                else:
                    break
            analytics["win_streak"] = streak
            analytics["lose_streak"] = 0
        elif daily_returns[-1] < 0:
            for r in reversed(daily_returns):
                if r < 0:
                    streak += 1
                else:
                    break
            analytics["win_streak"] = 0
            analytics["lose_streak"] = streak
        else:
            analytics["win_streak"] = 0
            analytics["lose_streak"] = 0
    else:
        analytics["win_streak"] = 0
        analytics["lose_streak"] = 0

    # ── 胜率 (日度) ──
    if len(daily_returns) > 0:
        analytics["win_rate"] = float(np.sum(daily_returns > 0) / len(daily_returns))
    else:
        analytics["win_rate"] = None

    # Clean up NaN/Inf
    for k, v in analytics.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            analytics[k] = None

    return analytics


@app.route("/api/qis-index-history")
def api_qis_index_history():
    """获取 QIS 指数历史数据 + 分析指标。优先 HMC, 回退 Wind。"""
    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "error": "缺少 name 参数"}), 400

    days = int(request.args.get("days", "3650"))  # 默认10年

    # 缓存
    cache_key = f"{name}_{days}"
    if cache_key in _QIS_INDEX_CACHE:
        cached = dict(_QIS_INDEX_CACHE[cache_key])
        cached["cached"] = True
        return jsonify(cached)

    from datetime import timedelta
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")

    print(f"[QIS_INDEX] 查询指数历史: {name} ({start_date} ~ {end_date})")

    # 1. 尝试 HMC
    result = _fetch_qis_index_from_hmc(name, start_date, end_date)

    # 2. 回退 Wind
    if not result:
        result = _fetch_qis_index_from_wind(name, start_date, end_date)

    if not result:
        return jsonify({"ok": False, "name": name, "error": f"HMC 和 Wind 均未找到 {name} 的历史数据"})

    ohlcv = result["ohlcv"]
    analytics = _compute_index_analytics(ohlcv)

    response = {
        "ok": True,
        "name": name,
        "source": result["source"],
        "ohlcv": ohlcv,
        "count": len(ohlcv),
        "analytics": analytics,
    }

    _QIS_INDEX_CACHE[cache_key] = response
    return jsonify(response)


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
    underlying = request.args.get("underlying", "").strip()  # 标的物 (可能含 BBG ticker)
    days = int(request.args.get("days", "1095"))  # 默认3年

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
        result = fetch_market_data(wind_ticker=ticker, name=name, days=days, underlying=underlying)
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
        mdata = fetch_market_data(wind_ticker=ticker, name=name, days=1095)
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


# -- Morning Meeting Speech routes ------------------------------------------------

def _get_cross_gamma_result():
    """Helper to fetch cross gamma data for speech module."""
    try:
        candidates = find_all_files()
        for date_str_cg, fpath in (candidates or []):
            raw_data, err = load_cross_gamma_data(fpath)
            if err:
                continue
            result = aggregate_book(raw_data)
            if result["stats"]["active_trades"] > 0:
                return {
                    "total_gamma": result["stats"]["total_gamma"],
                    "total_abs_gamma": result["stats"]["total_abs_gamma"],
                    "active_trades": result["stats"]["active_trades"],
                    "n_underlyings": result["stats"]["n_underlyings"],
                    "top_pairs": result["top_pairs"],
                    "per_asset": result["per_asset"],
                }
    except Exception as e:
        print(f"[SPEECH] Cross gamma helper error: {e}")
    return None


@app.route("/morning-meeting")
def morning_meeting_page():
    """早会发言页面"""
    return render_template(
        "morning_meeting.html",
        date_str=DATE_STR,
        now_time=datetime.now().strftime("%H:%M"),
    )


@app.route("/api/speech/latest")
def api_speech_latest():
    """返回今日已缓存的早会发言（如有）。"""
    cached = get_cached_speech()
    if cached:
        return jsonify(cached)
    return jsonify({"ok": False, "error": "今日尚未生成早会发言"})


@app.route("/api/speech/generate", methods=["POST"])
def api_speech_generate():
    """触发早会发言生成（后台线程执行，立即返回）。"""
    force = request.args.get("force", "false").lower() == "true"
    try:
        result = generate_morning_speech(
            app_data=DATA,
            contracts_fetcher=_fetch_qis_contracts,
            cross_gamma_fetcher=_get_cross_gamma_result,
            force=force,
        )
        # If cached result was returned directly, it has speech_part
        if result.get("speech_part"):
            return jsonify(result)
        # Otherwise it's an async trigger response
        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/speech/status")
def api_speech_status():
    """返回当前生成状态。"""
    return jsonify(get_generation_status())


# -- CICC Research routes (API only, page replaced by morning meeting) ----------

@app.route("/cicc-research")
def cicc_research_page():
    """中金研究院页面 — redirect to morning meeting"""
    from flask import redirect
    return redirect("/morning-meeting")


@app.route("/api/cicc-research/reports")
def api_cicc_reports():
    """获取中金研究报告列表"""
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("pageSize", 20))
    category = request.args.get("category", "")
    category_id = int(request.args.get("categoryId", 0) or 0)
    source = request.args.get("source", "hot")  # hot | latest | focus | recommend
    try:
        result = cicc_report_list(page=page, page_size=page_size, category=category,
                                  source=source, category_id=category_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "reports": []})


@app.route("/api/cicc-research/categories")
def api_cicc_categories():
    """返回可用报告分类列表"""
    try:
        return jsonify({"ok": True, "categories": cicc_get_categories()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "categories": []})


@app.route("/api/cicc-research/report/<report_id>")
def api_cicc_report_detail(report_id):
    """获取单篇研究报告详情"""
    try:
        result = cicc_report_detail(report_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "report": None})


@app.route("/api/cicc-research/search")
def api_cicc_search():
    """搜索中金研究报告"""
    keyword = request.args.get("keyword", "").strip()
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("pageSize", 20))
    if not keyword:
        return jsonify({"ok": False, "error": "请输入搜索关键词", "reports": []})
    try:
        result = cicc_search_reports(keyword=keyword, page=page, page_size=page_size)
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "reports": []})


@app.route("/api/cicc-research/test")
def api_cicc_test():
    """测试 CICC Research 连接"""
    try:
        result = cicc_test_connection()
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "status": "error", "message": str(e)})


@app.route("/api/cicc-research/cookies", methods=["POST"])
def api_cicc_save_cookies():
    """保存 CICC Research Cookies"""
    try:
        data = request.get_json()
        cookies_input = data.get("cookies")
        if not cookies_input:
            return jsonify({"ok": False, "error": "未提供 cookies 数据"})
        saved = cicc_save_cookies(cookies_input)
        return jsonify({"ok": True, "cookie_count": len(saved)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/cicc-research/cookies-status")
def api_cicc_cookies_status():
    """获取 Cookies 配置状态"""
    try:
        return jsonify(cicc_cookies_status())
    except Exception as e:
        return jsonify({"has_cookies": False, "error": str(e)})


@app.route("/api/cicc-research/clear-cache", methods=["POST"])
def api_cicc_clear_cache():
    """清除中金研究缓存"""
    cicc_clear_cache()
    return jsonify({"ok": True})


# -- Entrypoint ----------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5050)
