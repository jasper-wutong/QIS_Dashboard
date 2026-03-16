"""
市场数据服务模块 — 统一获取历史行情 + 技术指标计算。

数据持久化: SQLite 本地数据库 (market_data.db)
  - 首次请求: 从 Wind / Bloomberg 拉取 → 存入 SQLite
  - 后续请求: 优先从 SQLite 读取, 仅增量拉取缺失日期
  - /api/refresh 时仅清空内存缓存, 不清空 SQLite (历史数据不会变)

数据源路由:
  - 境内标的 (.SHF/.DCE/.CZC/.INE/.CFE) → Wind (万得 WindPy, 通过 subprocess)
  - 境外标的 (Index/Comdty/.HK/Eurex)   → Bloomberg (blpapi, 通过 subprocess)
  - 境内合约若数据不足 (新合约), 自动回退到主力连续合约 (如 HC.SHF)

技术指标 (纯 Python / pandas 计算):
  - MA 均线 (5/20/60)
  - MACD (12/26/9)
  - RSI (14)
  - Bollinger Bands (20, 2σ)

基本面数据:
  - 持仓量 (Open Interest)
  - 基差 (Basis = 现货 - 期货)

注意: 本模块仅使用真实数据, 无任何模拟/demo 降级。
"""

import os
import sys
import json
import math
import sqlite3
import traceback
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import numpy as np

# ── Bloomberg DLL 路径 ─────────────────────────────────────────────────────────
BBG_DLL_PATH = r"C:\blp\DAPI"
if os.path.isdir(BBG_DLL_PATH) and BBG_DLL_PATH not in os.environ.get("PATH", ""):
    os.environ["PATH"] = BBG_DLL_PATH + os.pathsep + os.environ.get("PATH", "")

# ── Wind 客户端路径 ────────────────────────────────────────────────────────────
_WIND_PATH = r"C:\Wind\Wind.NET.Client\WindNET\x64"
if _WIND_PATH not in sys.path:
    sys.path.insert(0, _WIND_PATH)


# ═══════════════════════════════════════════════════════════════════════════════
#  SQLite 数据库
# ═══════════════════════════════════════════════════════════════════════════════

DB_PATH = str(Path(__file__).resolve().parent / "market_data.db")


def _get_conn():
    # type: () -> sqlite3.Connection
    """获取 SQLite 连接。"""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _init_db():
    """创建表结构 (幂等)。"""
    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                ticker TEXT NOT NULL,
                date   TEXT NOT NULL,
                open   REAL,
                high   REAL,
                low    REAL,
                close  REAL,
                volume REAL,
                PRIMARY KEY (ticker, date)
            );
            CREATE TABLE IF NOT EXISTS open_interest (
                ticker TEXT NOT NULL,
                date   TEXT NOT NULL,
                value  REAL,
                PRIMARY KEY (ticker, date)
            );
            CREATE TABLE IF NOT EXISTS basis (
                ticker TEXT NOT NULL,
                date   TEXT NOT NULL,
                value  REAL,
                PRIMARY KEY (ticker, date)
            );
            CREATE TABLE IF NOT EXISTS fetch_meta (
                ticker      TEXT PRIMARY KEY,
                source      TEXT,
                first_date  TEXT,
                last_date   TEXT,
                updated_at  TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker ON ohlcv(ticker);
            CREATE INDEX IF NOT EXISTS idx_oi_ticker ON open_interest(ticker);
            CREATE INDEX IF NOT EXISTS idx_basis_ticker ON basis(ticker);
        """)
        conn.commit()
    finally:
        conn.close()


# 启动时初始化表
_init_db()


# ── SQLite 读写辅助 ───────────────────────────────────────────────────────────

def _db_get_date_range(ticker):
    # type: (str) -> Tuple[Optional[str], Optional[str]]
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT first_date, last_date FROM fetch_meta WHERE ticker = ?", (ticker,)
        ).fetchone()
        return (row[0], row[1]) if row else (None, None)
    finally:
        conn.close()


def _db_get_source(ticker):
    # type: (str) -> Optional[str]
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT source FROM fetch_meta WHERE ticker = ?", (ticker,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _db_save_ohlcv(ticker, rows):
    # type: (str, List[dict]) -> None
    if not rows:
        return
    conn = _get_conn()
    try:
        conn.executemany(
            """INSERT OR REPLACE INTO ohlcv (ticker, date, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [(ticker, r["date"], r.get("open"), r.get("high"),
              r.get("low"), r.get("close"), r.get("volume")) for r in rows],
        )
        conn.commit()
    finally:
        conn.close()


def _db_save_oi(ticker, rows):
    # type: (str, List[dict]) -> None
    if not rows:
        return
    conn = _get_conn()
    try:
        conn.executemany(
            """INSERT OR REPLACE INTO open_interest (ticker, date, value)
               VALUES (?, ?, ?)""",
            [(ticker, r["date"], r["value"]) for r in rows],
        )
        conn.commit()
    finally:
        conn.close()


def _db_save_basis(ticker, rows):
    # type: (str, List[dict]) -> None
    if not rows:
        return
    conn = _get_conn()
    try:
        conn.executemany(
            """INSERT OR REPLACE INTO basis (ticker, date, value)
               VALUES (?, ?, ?)""",
            [(ticker, r["date"], r["value"]) for r in rows],
        )
        conn.commit()
    finally:
        conn.close()


def _db_update_meta(ticker, source, first_date, last_date):
    # type: (str, str, str, str) -> None
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO fetch_meta (ticker, source, first_date, last_date, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (ticker, source, first_date, last_date, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
    finally:
        conn.close()


def _db_clear_ticker(ticker):
    # type: (str) -> None
    """清除指定 ticker 的所有缓存数据 (用于数据源升级)。"""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM ohlcv WHERE ticker = ?", (ticker,))
        conn.execute("DELETE FROM open_interest WHERE ticker = ?", (ticker,))
        conn.execute("DELETE FROM basis WHERE ticker = ?", (ticker,))
        conn.execute("DELETE FROM fetch_meta WHERE ticker = ?", (ticker,))
        conn.commit()
    finally:
        conn.close()


def _db_load_ohlcv(ticker, start_date, end_date):
    # type: (str, str, str) -> List[dict]
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT date, open, high, low, close, volume FROM ohlcv WHERE ticker=? AND date>=? AND date<=? ORDER BY date",
            (ticker, start_date, end_date),
        ).fetchall()
        return [{"date": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]} for r in rows]
    finally:
        conn.close()


def _db_load_oi(ticker, start_date, end_date):
    # type: (str, str, str) -> List[dict]
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT date, value FROM open_interest WHERE ticker=? AND date>=? AND date<=? ORDER BY date",
            (ticker, start_date, end_date),
        ).fetchall()
        return [{"date": r[0], "value": r[1]} for r in rows]
    finally:
        conn.close()


def _db_load_basis(ticker, start_date, end_date):
    # type: (str, str, str) -> List[dict]
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT date, value FROM basis WHERE ticker=? AND date>=? AND date<=? ORDER BY date",
            (ticker, start_date, end_date),
        ).fetchall()
        return [{"date": r[0], "value": r[1]} for r in rows]
    finally:
        conn.close()


def db_stats():
    # type: () -> Dict[str, Any]
    """返回数据库统计信息 (调试用)。"""
    conn = _get_conn()
    try:
        ticker_count = conn.execute("SELECT COUNT(DISTINCT ticker) FROM ohlcv").fetchone()[0]
        row_count = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        metas = conn.execute("SELECT ticker, source, first_date, last_date, updated_at FROM fetch_meta ORDER BY ticker").fetchall()
        return {
            "db_path": DB_PATH,
            "tickers": ticker_count,
            "ohlcv_rows": row_count,
            "details": [{"ticker": m[0], "source": m[1], "first": m[2], "last": m[3], "updated": m[4]} for m in metas],
        }
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  内存缓存 (避免同一 session 内反复查 SQLite)
# ═══════════════════════════════════════════════════════════════════════════════
_RESULT_CACHE = {}   # type: Dict[str, Dict[str, Any]]
_RESULT_CACHE_DATE = ""  # type: str


def _result_cache_key(ticker, days):
    # type: (str, int) -> str
    return "{}|{}".format(ticker, days)


def clear_cache():
    """清空内存缓存 (供 /api/refresh 调用)。不清空 SQLite 数据库。"""
    global _RESULT_CACHE, _RESULT_CACHE_DATE
    _RESULT_CACHE.clear()
    _RESULT_CACHE_DATE = ""


# ═══════════════════════════════════════════════════════════════════════════════
#  技术指标计算
# ═══════════════════════════════════════════════════════════════════════════════

def calc_ma(close: pd.Series, windows: List[int] = [5, 20, 60]) -> Dict[str, List]:
    """计算多条 MA 均线。返回 {ma5: [...], ma20: [...], ma60: [...]}"""
    result = {}
    for w in windows:
        ma = close.rolling(window=w, min_periods=1).mean()
        result[f"ma{w}"] = [None if pd.isna(v) else round(v, 4) for v in ma]
    return result


def calc_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Dict[str, List]:
    """计算 MACD。返回 {macd: [...], signal: [...], histogram: [...]}"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd": [None if pd.isna(v) else round(v, 4) for v in macd_line],
        "signal": [None if pd.isna(v) else round(v, 4) for v in signal_line],
        "histogram": [None if pd.isna(v) else round(v, 4) for v in histogram],
    }


def calc_rsi(close: pd.Series, period: int = 14) -> List:
    """计算 RSI (Relative Strength Index)。"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return [None if pd.isna(v) else round(v, 2) for v in rsi]


def calc_bollinger(
    close: pd.Series, period: int = 20, std_mult: float = 2.0
) -> Dict[str, List]:
    """计算布林带。返回 {upper: [...], middle: [...], lower: [...]}"""
    middle = close.rolling(window=period, min_periods=1).mean()
    std = close.rolling(window=period, min_periods=1).std()
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return {
        "upper": [None if pd.isna(v) else round(v, 4) for v in upper],
        "middle": [None if pd.isna(v) else round(v, 4) for v in middle],
        "lower": [None if pd.isna(v) else round(v, 4) for v in lower],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Wind 数据获取 (境内)
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_wind(wind_ticker, start_date, end_date):
    # type: (str, str, str) -> Dict[str, Any]
    """
    通过 Wind API 获取境内标的历史行情 + 持仓量。
    start_date / end_date: "YYYY-MM-DD"
    """
    try:
        from WindPy import w
    except (ImportError, OSError) as e:
        return {"ok": False, "error": "WindPy 无法加载 (需要 Python 3.7): {}".format(e)}

    if not w.isconnected():
        ret = w.start()
        if not w.isconnected():
            return {"ok": False, "error": "Wind 连接失败: {}".format(ret)}

    # 获取 OHLCV + 持仓量
    fields = "open,high,low,close,volume,oi"
    data = w.wsd(wind_ticker, fields, start_date, end_date, "")

    if data.ErrorCode != 0:
        return {"ok": False, "error": "Wind wsd 错误码: {}".format(data.ErrorCode)}

    if not data.Data or not data.Times:
        return {"ok": False, "error": "Wind 返回空数据"}

    dates = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in data.Times]
    n = len(dates)

    def _safe_list(arr):
        if arr is None:
            return [None] * n
        return [None if (v is None or (isinstance(v, float) and math.isnan(v))) else v for v in arr]

    open_prices = _safe_list(data.Data[0] if len(data.Data) > 0 else None)
    high_prices = _safe_list(data.Data[1] if len(data.Data) > 1 else None)
    low_prices = _safe_list(data.Data[2] if len(data.Data) > 2 else None)
    close_prices = _safe_list(data.Data[3] if len(data.Data) > 3 else None)
    volumes = _safe_list(data.Data[4] if len(data.Data) > 4 else None)
    oi_data = _safe_list(data.Data[5] if len(data.Data) > 5 else None)

    ohlcv = []
    for i in range(n):
        ohlcv.append({
            "date": dates[i],
            "open": open_prices[i],
            "high": high_prices[i],
            "low": low_prices[i],
            "close": close_prices[i],
            "volume": volumes[i],
        })

    open_interest = [{"date": dates[i], "value": oi_data[i]} for i in range(n) if oi_data[i] is not None]

    # 尝试获取基差 (现货-期货)
    basis_data = []
    try:
        basis_result = w.wsd(wind_ticker, "basis", start_date, end_date, "")
        if basis_result.ErrorCode == 0 and basis_result.Data and basis_result.Data[0]:
            basis_raw = _safe_list(basis_result.Data[0])
            basis_data = [
                {"date": dates[i], "value": basis_raw[i]}
                for i in range(min(n, len(basis_raw)))
                if basis_raw[i] is not None
            ]
    except Exception:
        pass  # 基差字段可能不可用

    return {
        "ok": True,
        "source": "wind",
        "ohlcv": ohlcv,
        "fundamentals": {
            "open_interest": open_interest,
            "basis": basis_data,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Wind 数据获取 — subprocess 方式 (支持不同 Python 版本)
# ═══════════════════════════════════════════════════════════════════════════════

import re as _re

# Wind 使用系统 Python 3.7 (含 WindPy)
_WIND_PYTHON = r"C:\Users\wutong6\AppData\Local\Programs\Python\Python37\python.exe"
_WIND_HELPER = str(Path(__file__).resolve().parent / "wind_helper.py")

# 境内期货交易所后缀
_DOMESTIC_SUFFIXES = (".SHF", ".DCE", ".CZC", ".INE", ".CFE")


def _is_domestic(ticker):
    # type: (str) -> bool
    """判断是否为境内期货 Wind Ticker。"""
    return any(ticker.upper().endswith(sfx) for sfx in _DOMESTIC_SUFFIXES)


def _to_continuous_ticker(wind_ticker):
    # type: (str) -> Optional[str]
    """
    将具体合约 Wind Ticker 转换为主力连续合约。

    HC2605.SHF  → HC.SHF
    CF605.CZC   → CF.CZC
    I2609.DCE   → I.DCE

    对于已经是连续合约格式的 (HC01.SHF 等), 返回 None。
    """
    m = _re.match(r'^([A-Za-z]+)(\d{3,6})\.(SHF|DCE|CZC|INE|CFE)$', wind_ticker, _re.IGNORECASE)
    if m:
        prefix = m.group(1).upper()
        date_part = m.group(2)
        exchange = m.group(3).upper()
        # 排除连续合约格式 (01-09 为连续合约编号)
        if len(date_part) == 2 and int(date_part) < 13:
            return None
        return "{}.{}".format(prefix, exchange)
    return None


def _extract_json_line(stdout):
    # type: (str) -> Optional[str]
    """从 stdout 中提取最后一行有效 JSON (跳过 Wind Welcome 横幅等)。"""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith('{'):
            return line
    return None


def _fetch_wind_subprocess(wind_ticker, start_date, end_date):
    # type: (str, str, str) -> Dict[str, Any]
    """
    通过 subprocess 调用 wind_helper.py (使用系统 Python 3.7 + WindPy) 获取历史数据。
    """
    import subprocess

    if not os.path.isfile(_WIND_PYTHON):
        return {"ok": False, "error": "未找到 Wind Python: {}".format(_WIND_PYTHON)}
    if not os.path.isfile(_WIND_HELPER):
        return {"ok": False, "error": "未找到 wind_helper.py: {}".format(_WIND_HELPER)}

    try:
        result = subprocess.run(
            [_WIND_PYTHON, _WIND_HELPER,
             "--mode", "history",
             "--ticker", wind_ticker,
             "--start", start_date,
             "--end", end_date],
            capture_output=True,
            text=True,
            timeout=60,
        )

        stdout = result.stdout.strip()
        if not stdout:
            stderr = result.stderr.strip()[:500] if result.stderr else "无输出"
            return {"ok": False, "error": "Wind helper 无输出: {}".format(stderr)}

        json_line = _extract_json_line(stdout)
        if not json_line:
            return {"ok": False, "error": "Wind helper 输出无 JSON: {}".format(stdout[:200])}

        try:
            data = json.loads(json_line)
        except ValueError:
            return {"ok": False, "error": "Wind helper 输出无法解析为 JSON: {}".format(json_line[:200])}

        return data

    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Wind 请求超时 (>60s)"}
    except Exception as exc:
        return {"ok": False, "error": "Wind 调用异常: {}".format(exc)}


def _fetch_wind_realtime_subprocess(wind_ticker):
    # type: (str) -> Dict[str, Any]
    """
    通过 subprocess 调用 wind_helper.py (snapshot 模式) 获取实时行情。
    """
    import subprocess

    if not os.path.isfile(_WIND_PYTHON):
        return {"ok": False, "error": "未找到 Wind Python: {}".format(_WIND_PYTHON)}
    if not os.path.isfile(_WIND_HELPER):
        return {"ok": False, "error": "未找到 wind_helper.py: {}".format(_WIND_HELPER)}

    try:
        result = subprocess.run(
            [_WIND_PYTHON, _WIND_HELPER,
             "--mode", "snapshot",
             "--ticker", wind_ticker],
            capture_output=True,
            text=True,
            timeout=15,
        )

        stdout = result.stdout.strip()
        if not stdout:
            stderr = result.stderr.strip()[:500] if result.stderr else "无输出"
            return {"ok": False, "error": "Wind snapshot 无输出: {}".format(stderr)}

        json_line = _extract_json_line(stdout)
        if not json_line:
            return {"ok": False, "error": "Wind snapshot 输出无 JSON: {}".format(stdout[:200])}

        try:
            data = json.loads(json_line)
        except ValueError:
            return {"ok": False, "error": "Wind snapshot 输出无法解析: {}".format(json_line[:200])}

        return data

    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Wind snapshot 超时 (>15s)"}
    except Exception as exc:
        return {"ok": False, "error": "Wind snapshot 异常: {}".format(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
#  Bloomberg 数据获取 (境外)
# ═══════════════════════════════════════════════════════════════════════════════

_VENV_PYTHON = str(Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe")
_BBG_HELPER = str(Path(__file__).resolve().parent / "bloomberg_helper.py")
_HMC_HELPER = str(Path(__file__).resolve().parent / "hmc_helper.py")
_HMC_EARLIEST = "2010-01-01"  # HMC 历史数据起点 (全量拉取)


def _fetch_bloomberg(bbg_ticker, start_date, end_date):
    # type: (str, str, str) -> Dict[str, Any]
    """
    通过 Bloomberg API 获取境外标的历史行情 + 持仓量。

    实现方式: 以 subprocess 调用 bloomberg_helper.py (在含 blpapi 的 venv Python 中运行),
    通过 stdout JSON 通信。这样 Flask 服务器 (Python 3.7) 不需要自己加载 blpapi。
    """
    import subprocess

    # 快速预检: Bloomberg 端口是否可达 (1 秒超时)
    import socket
    try:
        with socket.create_connection(("localhost", 8194), timeout=1):
            pass
    except (OSError, socket.timeout):
        return {"ok": False, "error": "Bloomberg Terminal 未运行 (localhost:8194 不可达)，请先登录 Bloomberg Terminal"}

    if not os.path.isfile(_VENV_PYTHON):
        return {"ok": False, "error": "未找到 venv Python: {}".format(_VENV_PYTHON)}
    if not os.path.isfile(_BBG_HELPER):
        return {"ok": False, "error": "未找到 bloomberg_helper.py: {}".format(_BBG_HELPER)}

    try:
        env = os.environ.copy()
        if os.path.isdir(BBG_DLL_PATH):
            env["PATH"] = BBG_DLL_PATH + os.pathsep + env.get("PATH", "")

        result = subprocess.run(
            [_VENV_PYTHON, _BBG_HELPER,
             "--ticker", bbg_ticker,
             "--start", start_date,
             "--end", end_date],
            capture_output=True,
            text=True,
            timeout=60,   # 最多等 60 秒
            env=env,
        )

        stdout = result.stdout.strip()
        if not stdout:
            stderr = result.stderr.strip()[:500] if result.stderr else "无输出"
            return {"ok": False, "error": "Bloomberg helper 无输出: {}".format(stderr)}

        try:
            data = json.loads(stdout)
        except ValueError:
            return {"ok": False, "error": "Bloomberg helper 输出无法解析为 JSON: {}".format(stdout[:200])}

        return data

    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Bloomberg 请求超时 (>60s)"}
    except Exception as exc:
        return {"ok": False, "error": "Bloomberg 调用异常: {}".format(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
#  HMC 数据获取 (优先来源) — 通过 subprocess 调用 hmc_helper.py
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_hmc_subprocess(wind_ticker, start_date, end_date, underlying=""):
    # type: (str, str, str, str) -> Dict[str, Any]
    """
    通过 subprocess 调用 hmc_helper.py 获取历史 OHLCV。

    路由规则:
      - 境内具体合约 (HC2605.SHF): SQL 查 FUT_DAY_QUOT, 含基差 + 持仓量
      - 境内连续合约 (HC.SHF): 返回失败 (让 Wind 处理)
      - 全球期货 (CL1 Comdty 格式): HMC 标准 --mode history API

    Args:
        underlying: 标的物列 (可能包含正确 Bloomberg ticker, 如 GXH6 Index)
    """
    import subprocess
    import re as _re_hmc

    if not os.path.isfile(_VENV_PYTHON):
        return {"ok": False, "error": "未找到 venv Python: {}".format(_VENV_PYTHON)}
    if not os.path.isfile(_HMC_HELPER):
        return {"ok": False, "error": "未找到 hmc_helper.py: {}".format(_HMC_HELPER)}

    env = os.environ.copy()
    # gRPC 不正确识别 NO_PROXY 通配符 (10.50.*), 必须彻底清除代理
    for _proxy_key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                       "grpc_proxy", "GRPC_PROXY"):
        env.pop(_proxy_key, None)
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"

    if _is_domestic(wind_ticker):
        # 支持具体合约 (HC2605.SHF) 和连续合约 (HC.SHF)
        # HMC FUT_DAY_QUOT 的 SEC_ID 包含交易所后缀, 如 "HC2605.SHF", "HC.SHF"
        m_specific = _re_hmc.match(r'^([A-Za-z]+)(\d{3,6})\.(SHF|DCE|CZC|INE|CFE|GFEX)$',
                                   wind_ticker, _re_hmc.IGNORECASE)
        m_continuous = _re_hmc.match(r'^([A-Za-z]+)\.(SHF|DCE|CZC|INE|CFE|GFEX)$',
                                     wind_ticker, _re_hmc.IGNORECASE)
        if not m_specific and not m_continuous:
            return {"ok": False, "error": "HMC 跳过非标格式: {}".format(wind_ticker)}

        # SEC_ID 直接使用完整 Wind Ticker (含交易所后缀)
        sec_id = wind_ticker.upper()
        start_ddb = start_date.replace("-", ".")    # "2010.01.01"
        end_ddb   = end_date.replace("-", ".")

        sql = (
            "select TRAN_DATE, OPEN_PRC, HIGH_PRC, LOW_PRC, CLOSE_PRC, "
            "TX_QTY, VOHP, BASIS "
            "from loadTable('dfs://HQUOT_CENTER_EOD', 'FUT_DAY_QUOT') "
            "where TRAN_DATE >= {start} and TRAN_DATE <= {end} and SEC_ID = '{sec_id}' "
            "order by TRAN_DATE asc"
        ).format(start=start_ddb, end=end_ddb, sec_id=sec_id)

        try:
            result = subprocess.run(
                [_VENV_PYTHON, _HMC_HELPER, "--mode", "query", "--sql", sql],
                capture_output=True, text=True, timeout=90, env=env,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "HMC SQL 超时 (>90s)"}
        except Exception as exc:
            return {"ok": False, "error": "HMC SQL 异常: {}".format(exc)}

        stdout = result.stdout.strip()
        if not stdout:
            return {"ok": False, "error": "HMC SQL 无输出: {}".format(
                result.stderr.strip()[:300] if result.stderr else "")}

        json_line = _extract_json_line(stdout)
        if not json_line:
            return {"ok": False, "error": "HMC SQL 无 JSON: {}".format(stdout[:300])}

        try:
            data = json.loads(json_line)
        except ValueError:
            return {"ok": False, "error": "HMC SQL JSON 解析失败: {}".format(json_line[:200])}

        if not data.get("ok"):
            return data

        rows = data.get("data", []) or []
        ohlcv = []
        oi_rows = []
        basis_rows_h = []
        for row in rows:
            date_raw = row.get("TRAN_DATE")
            date_str = str(date_raw)[:10] if date_raw else None
            if not date_str:
                continue
            close_val = row.get("CLOSE_PRC")
            if close_val is None:
                continue
            ohlcv.append({
                "date":   date_str,
                "open":   row.get("OPEN_PRC"),
                "high":   row.get("HIGH_PRC"),
                "low":    row.get("LOW_PRC"),
                "close":  close_val,
                "volume": row.get("TX_QTY"),
            })
            oi_val = row.get("VOHP")
            if oi_val is not None:
                oi_rows.append({"date": date_str, "value": oi_val})
            basis_val = row.get("BASIS")
            if basis_val is not None:
                basis_rows_h.append({"date": date_str, "value": basis_val})

        if not ohlcv:
            # 连接正常但此区间无数据 (合约未上市或已到期)
            return {"ok": True, "ohlcv": [], "source": "hmc", "fundamentals": {"open_interest": [], "basis": []}}

        return {
            "ok": True,
            "source": "hmc",
            "ohlcv": ohlcv,
            "fundamentals": {
                "open_interest": oi_rows,
                "basis": basis_rows_h,   # HMC SQL 直接含基差
            },
        }

    else:
        # 全球期货 → HMC 标准 API (使用 Bloomberg 代码)
        from ticker_mapping import resolve_bbg_ticker
        # 优先使用 underlying (标的物列) 作为 BBG ticker — 表中已有正确映射
        bbg = None
        if underlying and (" Index" in underlying or " Comdty" in underlying):
            bbg = underlying
        if not bbg:
            bbg = resolve_bbg_ticker(wind_ticker)
        if not bbg:
            return {"ok": False, "error": "无法映射为 Bloomberg/HMC ticker: {}".format(wind_ticker)}
        print("[MARKET_DATA] HMC 境外: {} → BBG={}".format(wind_ticker, bbg))

        try:
            result = subprocess.run(
                [_VENV_PYTHON, _HMC_HELPER,
                 "--mode", "history",
                 "--ticker", bbg,
                 "--start", start_date,
                 "--end", end_date],
                capture_output=True, text=True, timeout=90, env=env,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "HMC history 超时 (>90s)"}
        except Exception as exc:
            return {"ok": False, "error": "HMC history 异常: {}".format(exc)}

        stdout = result.stdout.strip()
        if not stdout:
            return {"ok": False, "error": "HMC helper 无输出: {}".format(
                result.stderr.strip()[:300] if result.stderr else "")}

        json_line = _extract_json_line(stdout)
        if not json_line:
            return {"ok": False, "error": "HMC helper 无 JSON: {}".format(stdout[:300])}

        try:
            data = json.loads(json_line)
        except ValueError:
            return {"ok": False, "error": "HMC JSON 解析失败: {}".format(json_line[:200])}

        if not data.get("ok"):
            return data

        rows = data.get("data", []) or []
        ohlcv = []
        oi_rows = []
        for row in rows:
            date_raw = row.get("date")
            date_str = str(date_raw)[:10] if date_raw else None
            if not date_str:
                continue
            close_val = row.get("close")
            if close_val is None:
                continue
            ohlcv.append({
                "date":   date_str,
                "open":   row.get("open"),
                "high":   row.get("high"),
                "low":    row.get("low"),
                "close":  close_val,
                "volume": row.get("volume"),
            })
            oi_val = row.get("oi")
            if oi_val is not None:
                oi_rows.append({"date": date_str, "value": oi_val})

        if not ohlcv:
            # 连接正常但此区间无数据 (合约未上市或已到期)
            return {"ok": True, "ohlcv": [], "source": "hmc", "fundamentals": {"open_interest": [], "basis": []}}

        return {
            "ok": True,
            "source": "hmc",
            "ohlcv": ohlcv,
            "fundamentals": {
                "open_interest": oi_rows,
                "basis": [],
            },
        }


def _fetch_hmc_realtime_subprocess(wind_ticker):
    # type: (str) -> Dict[str, Any]
    """
    通过 subprocess 调用 hmc_helper.py 获取最新行情:
    - 境内具体合约: SQL 查 FUT_DAY_QUOT 最新一行 (结算价/收盘价)
    - 境内连续合约: 返回失败 (让 Wind 处理)
    - 境外期货: HMC 标准 snapshot API (Bloomberg 代码)
    """
    import subprocess
    import re as _re_hmc_rt

    if not os.path.isfile(_VENV_PYTHON):
        return {"ok": False, "error": "未找到 venv Python"}
    if not os.path.isfile(_HMC_HELPER):
        return {"ok": False, "error": "未找到 hmc_helper.py"}

    env = os.environ.copy()
    # gRPC 不正确识别 NO_PROXY 通配符, 必须彻底清除代理
    for _proxy_key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                       "grpc_proxy", "GRPC_PROXY"):
        env.pop(_proxy_key, None)
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"

    if _is_domestic(wind_ticker):
        m = _re_hmc_rt.match(r'^([A-Za-z]+)(\d{3,6})\.(SHF|DCE|CZC|INE|CFE|GFEX)$',
                              wind_ticker, _re_hmc_rt.IGNORECASE)
        if not m:
            return {"ok": False, "error": "HMC RT 跳过连续合约: {}".format(wind_ticker)}

        sec_id = wind_ticker.upper()   # "HC2605.SHF" — 含交易所后缀
        sql = (
            "select top 1 TRAN_DATE, OPEN_PRC, HIGH_PRC, LOW_PRC, CLOSE_PRC, "
            "STTM_PRC, TX_QTY, VOHP "
            "from loadTable('dfs://HQUOT_CENTER_EOD', 'FUT_DAY_QUOT') "
            "where SEC_ID = '{sec_id}' "
            "order by TRAN_DATE desc"
        ).format(sec_id=sec_id)

        try:
            result = subprocess.run(
                [_VENV_PYTHON, _HMC_HELPER, "--mode", "query", "--sql", sql],
                capture_output=True, text=True, timeout=15, env=env,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "HMC RT 超时"}
        except Exception as exc:
            return {"ok": False, "error": "HMC RT 异常: {}".format(exc)}

        stdout = result.stdout.strip()
        if not stdout:
            return {"ok": False, "error": "HMC RT 无输出"}

        json_line = _extract_json_line(stdout)
        if not json_line:
            return {"ok": False, "error": "HMC RT 无 JSON"}

        try:
            data = json.loads(json_line)
        except ValueError:
            return {"ok": False, "error": "HMC RT JSON 解析失败"}

        if not data.get("ok"):
            return data

        rows = data.get("data", []) or []
        if not rows:
            return {"ok": False, "error": "HMC RT 返回空"}

        row = rows[0]
        # 优先结算价 STTM_PRC, 再用收盘价 CLOSE_PRC
        price = row.get("STTM_PRC") or row.get("CLOSE_PRC")
        if price is None:
            return {"ok": False, "error": "HMC RT 价格为空"}

        return {
            "ok":     True,
            "source": "hmc",
            "price":  price,
            "open":   row.get("OPEN_PRC"),
            "high":   row.get("HIGH_PRC"),
            "low":    row.get("LOW_PRC"),
            "volume": row.get("TX_QTY"),
            "time":   str(row.get("TRAN_DATE", ""))[:10],
        }

    else:
        # 全球期货 → HMC 标准 snapshot (Bloomberg 代码)
        from ticker_mapping import resolve_bbg_ticker
        bbg = resolve_bbg_ticker(wind_ticker)
        if not bbg:
            return {"ok": False, "error": "无法映射 HMC ticker: {}".format(wind_ticker)}

        try:
            result = subprocess.run(
                [_VENV_PYTHON, _HMC_HELPER,
                 "--mode", "snapshot",
                 "--ticker", bbg],
                capture_output=True, text=True, timeout=15, env=env,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "HMC snapshot 超时"}
        except Exception as exc:
            return {"ok": False, "error": "HMC snapshot 异常: {}".format(exc)}

        stdout = result.stdout.strip()
        if not stdout:
            return {"ok": False, "error": "HMC snapshot 无输出"}

        json_line = _extract_json_line(stdout)
        if not json_line:
            return {"ok": False, "error": "HMC snapshot 无 JSON: {}".format(stdout[:200])}

        try:
            data = json.loads(json_line)
        except ValueError:
            return {"ok": False, "error": "HMC snapshot JSON 解析失败"}

        if not data.get("ok"):
            return data

        items = data.get("data", []) or []
        if not items:
            return {"ok": False, "error": "HMC snapshot 返回空"}

        item = items[0]
        # 优先 settlement (结算价), 再用 last (收盘价)
        last = item.get("settlement") or item.get("last")
        if last is None:
            return {"ok": False, "error": "HMC snapshot last 为空"}

        return {
            "ok":     True,
            "source": "hmc",
            "price":  last,
            "open":   item.get("open"),
            "high":   item.get("high"),
            "low":    item.get("low"),
            "volume": item.get("volume"),
            "time":   str(item.get("timestamp", ""))[:19],
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  增量拉取 + SQLite 持久化
# ═══════════════════════════════════════════════════════════════════════════════

def _determine_fetch_range(ticker, desired_start, desired_end):
    # type: (str, str, str) -> Optional[Tuple[str, str]]
    """
    对比 SQLite 已有数据, 确定需要增量拉取的日期区间。
    返回 (fetch_start, fetch_end) or None (完全覆盖)。
    """
    db_first, db_last = _db_get_date_range(ticker)

    if db_first is None:
        return (desired_start, desired_end)

    last_dt = datetime.strptime(db_last, "%Y-%m-%d")
    end_dt = datetime.strptime(desired_end, "%Y-%m-%d")
    gap_days = (end_dt - last_dt).days

    if gap_days <= 1:
        if db_first <= desired_start:
            return None  # 完全覆盖
        else:
            return (desired_start, db_first)

    if db_first > desired_start:
        return (desired_start, desired_end)

    fetch_start = (last_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    return (fetch_start, desired_end)


def _fetch_and_store(ticker, start_date, end_date, region, underlying=""):
    # type: (str, str, str, str, str) -> Dict[str, Any]
    """
    拉取数据并存入 SQLite。

    路由规则 (优先级):
      1. HMC (境内/全球期货) — 长历史、实时更新
      2. 境内 → Wind (subprocess), 若 HMC 失败
         - 具体合约数据不足 (<60行) 时自动回退到主力连续合约
      3. 境外 → Bloomberg (subprocess), 若 HMC 失败

    Args:
        underlying: 标的物列 (可能含正确 BBG ticker)
    """
    # ── 1. 先尝试 HMC ──
    raw = _fetch_hmc_subprocess(ticker, start_date, end_date, underlying=underlying)
    _hmc_ok_with_data = raw.get("ok") and len(raw.get("ohlcv", [])) > 0
    _hmc_ok_empty = raw.get("ok") and len(raw.get("ohlcv", [])) == 0  # 连接正常但无数据

    if _hmc_ok_with_data:
        print("[MARKET_DATA] HMC 成功: {} ({} 行)".format(ticker, len(raw.get("ohlcv", []))))
    elif _hmc_ok_empty:
        # HMC 可达但此区间无数据: 记录“已搜索到 start_date”哨兵 — 无需 Wind 回退
        print("[MARKET_DATA] HMC {} 范围内无数据 ({} ~ {}), 记录哨兵".format(ticker, start_date, end_date))
        old_first, old_last = _db_get_date_range(ticker)
        if old_first and old_last:
            _db_update_meta(ticker, "hmc", min(start_date, old_first), old_last)
        return raw  # ok=True, ohlcv=[]; 下游会读取现有 DB 数据
    else:
        hmc_err = raw.get("error", "未知错误")
        print("[MARKET_DATA] HMC 失败 ({}), 回退: {}".format(hmc_err[:80], ticker))

        # ── 2. 境内: 回退 Wind ──
        if _is_domestic(ticker):
            raw = _fetch_wind_subprocess(ticker, start_date, end_date)

            # 若具体合约数据太少 (未上市充分), 尝试主力连续合约
            ohlcv_rows = len(raw.get("ohlcv", [])) if raw.get("ok") else 0
            if ohlcv_rows < 60:
                continuous = _to_continuous_ticker(ticker)
                if continuous and continuous != ticker:
                    print("[MARKET_DATA] {} 仅返回 {} 行, 尝试主力连续 {}".format(
                        ticker, ohlcv_rows, continuous))
                    raw2 = _fetch_wind_subprocess(continuous, start_date, end_date)
                    if raw2.get("ok") and len(raw2.get("ohlcv", [])) > ohlcv_rows:
                        raw = raw2
                        raw["_continuous_fallback"] = continuous
                        print("[MARKET_DATA] 使用主力连续 {} ({} 行)".format(
                            continuous, len(raw.get("ohlcv", []))))
        else:
            # ── 3. 境外: 回退 Bloomberg ──
            from ticker_mapping import resolve_bbg_ticker
            bbg_code = None
            if underlying and (" Index" in underlying or " Comdty" in underlying):
                bbg_code = underlying
            if not bbg_code:
                bbg_code = resolve_bbg_ticker(ticker)
            if not bbg_code:
                return {"ok": False, "error": "无法将 {} 映射为 Bloomberg ticker".format(ticker)}
            print("[MARKET_DATA] BBG 回退: {} → {}".format(ticker, bbg_code))
            raw = _fetch_bloomberg(bbg_code, start_date, end_date)

    if not raw.get("ok"):
        return raw

    ohlcv = raw.get("ohlcv", [])
    fund = raw.get("fundamentals", {})
    oi = fund.get("open_interest", [])
    basis_rows = fund.get("basis", [])

    _db_save_ohlcv(ticker, ohlcv)
    _db_save_oi(ticker, oi)
    _db_save_basis(ticker, basis_rows)

    if ohlcv:
        all_dates = [r["date"] for r in ohlcv]
        # new_first 取 start_date 与实际首行较小值:
        # 记录「已向前搜索到 start_date」，防止下次重复拉取合约上市前的空区间
        new_first = min(start_date, min(all_dates))
        new_last = max(all_dates)
        old_first, old_last = _db_get_date_range(ticker)
        if old_first:
            new_first = min(new_first, old_first)
            new_last = max(new_last, old_last)
        _db_update_meta(ticker, raw.get("source", "unknown"), new_first, new_last)

    return raw


# ═══════════════════════════════════════════════════════════════════════════════
#  统一入口 (唯一公开函数)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_market_data(wind_ticker, name="", days=180, underlying=""):
    # type: (str, str, int, str) -> Dict[str, Any]
    """
    统一获取历史行情 + 技术指标 + 基本面数据。

    流程:
      1. 检查内存缓存
      2. 计算目标日期范围
      3. 检查 SQLite, 确定增量拉取范围
      4. 如有缺失 → 从 Wind/Bloomberg 拉取 → 存入 SQLite
      5. 从 SQLite 全量读取目标范围
      6. 计算技术指标
      7. 返回完整 JSON

    仅使用真实数据, 无任何模拟降级。
    """
    from ticker_mapping import resolve_region

    # ── 1. 内存缓存 ──
    today_str = datetime.today().strftime("%Y-%m-%d")
    global _RESULT_CACHE_DATE
    if _RESULT_CACHE_DATE != today_str:
        _RESULT_CACHE.clear()
        _RESULT_CACHE_DATE = today_str

    cache_k = _result_cache_key(wind_ticker, days)
    if cache_k in _RESULT_CACHE:
        cached = dict(_RESULT_CACHE[cache_k])
        cached["cached"] = True
        return cached

    # ── 2. 目标日期范围 ──
    end_date = datetime.today().strftime("%Y-%m-%d")

    # 若数据库中没有该 ticker 的数据, HMC 首次全量拉取从 _HMC_EARLIEST 开始
    db_first_existing, db_last_existing = _db_get_date_range(wind_ticker)
    existing_source = _db_get_source(wind_ticker)

    hmc_available = os.path.isfile(_HMC_HELPER)

    # 确定起始日期:
    # 1. 首次拉取 (DB 无数据) → 从 _HMC_EARLIEST 开始
    # 2. 已有 Wind 数据但 HMC 可用 → 不清除 Wind 数据，而是让增量逻辑向前补充历史
    #    (start_date = _HMC_EARLIEST, _determine_fetch_range 会找出缺失的历史区间)
    # 3. 已有 HMC 数据 → 普通增量 (start_date 不影响 read_start, 仅用于增量范围计算)
    if db_first_existing is None:
        start_date = _HMC_EARLIEST if hmc_available else (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
        if hmc_available:
            print("[MARKET_DATA] 首次拉取 {}, 使用全量起点: {}".format(wind_ticker, start_date))
    elif existing_source == "hmc" and hmc_available:
        # HMC 数据已存在: 始终从 _HMC_EARLIEST 检查，防止滚动窗口 (today-10y) 引起的无限循环
        start_date = _HMC_EARLIEST
    elif existing_source == "wind" and hmc_available and _is_domestic(wind_ticker):
        # 向前补充: 让 _determine_fetch_range 计算 (Wind 数据之前) 的历史缺口
        start_date = _HMC_EARLIEST
        print("[MARKET_DATA] wind→hmc 升级: {} 从 {} 补充历史".format(wind_ticker, start_date))
    else:
        start_date = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")

    # ── 3. 数据源判断 ──
    region = resolve_region(wind_ticker)

    # ── 4. 增量拉取 ──
    fetch_range = _determine_fetch_range(wind_ticker, start_date, end_date)
    source = existing_source or ("hmc" if hmc_available else
                                 "wind" if _is_domestic(wind_ticker) else "bloomberg")

    if fetch_range is not None:
        fetch_start, fetch_end = fetch_range
        print("[MARKET_DATA] 增量拉取 {} : {} ~ {}".format(wind_ticker, fetch_start, fetch_end))

        raw = _fetch_and_store(wind_ticker, fetch_start, fetch_end, region, underlying=underlying)

        if not raw.get("ok"):
            # API 失败 — 检查数据库是否有旧数据可用
            existing = _db_load_ohlcv(wind_ticker, start_date, end_date)
            if existing:
                print("[MARKET_DATA] API 失败但数据库有旧数据 ({} 条): {}".format(len(existing), wind_ticker))
            else:
                return {
                    "ok": False,
                    "ticker": wind_ticker,
                    "name": name,
                    "error": raw.get("error", "数据拉取失败"),
                }

        source = raw.get("source", source)
    else:
        print("[MARKET_DATA] SQLite 已覆盖, 直接读取: {}".format(wind_ticker))

    # ── 5. 从 SQLite 读取 (按 days 限制展示区间; DB 中仍存有完整历史) ──
    read_start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    ohlcv = _db_load_ohlcv(wind_ticker, read_start, end_date)
    oi = _db_load_oi(wind_ticker, read_start, end_date)
    basis_rows = _db_load_basis(wind_ticker, read_start, end_date)

    if not ohlcv:
        return {"ok": False, "ticker": wind_ticker, "name": name, "error": "数据库中无 {} 的行情数据".format(wind_ticker)}

    # ── 6. 计算技术指标 ──
    df = pd.DataFrame(ohlcv)
    close = df["close"].astype(float)

    indicators = {}
    try:
        indicators.update(calc_ma(close, [5, 20, 60]))
    except Exception:
        pass
    try:
        indicators["macd"] = calc_macd(close)
    except Exception:
        pass
    try:
        indicators["rsi"] = calc_rsi(close)
    except Exception:
        pass
    try:
        indicators["bollinger"] = calc_bollinger(close)
    except Exception:
        pass

    # ── 7. 返回 ──
    result = {
        "ok": True,
        "ticker": wind_ticker,
        "name": name,
        "source": source,
        "ohlcv": ohlcv,
        "indicators": indicators,
        "fundamentals": {
            "open_interest": oi,
            "basis": basis_rows,
        },
    }

    _RESULT_CACHE[cache_k] = result
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  实时行情 (Real-time Snapshot)
# ═══════════════════════════════════════════════════════════════════════════════

_RT_CACHE = {}   # type: Dict[str, Dict[str, Any]]
_RT_CACHE_TS = {}  # type: Dict[str, float]   # ticker → timestamp
_RT_CACHE_TTL = 3  # 缓存 3 秒, 避免短时间内重复请求


def _fetch_realtime_wind(wind_ticker):
    # type: (str) -> Dict[str, Any]
    """通过 Wind wsq 获取境内标的实时行情快照。"""
    try:
        from WindPy import w
    except (ImportError, OSError) as e:
        return {"ok": False, "error": "WindPy 无法加载: {}".format(e)}

    if not w.isconnected():
        ret = w.start()
        if not w.isconnected():
            return {"ok": False, "error": "Wind 连接失败: {}".format(ret)}

    fields = "rt_last,rt_open,rt_high,rt_low,rt_vol,rt_chg,rt_pct_chg,rt_pre_close,rt_date,rt_time"
    data = w.wsq(wind_ticker, fields)

    if data.ErrorCode != 0:
        return {"ok": False, "error": "Wind wsq 错误码: {}".format(data.ErrorCode)}

    if not data.Data or not data.Fields:
        return {"ok": False, "error": "Wind wsq 返回空数据"}

    # wsq 返回结构: Data = [[val1], [val2], ...], Fields = [field1, field2, ...]
    result = {}
    for i, field in enumerate(data.Fields):
        val = data.Data[i][0] if data.Data[i] else None
        if val is not None and isinstance(val, float) and math.isnan(val):
            val = None
        key = field.upper().replace("RT_", "")
        result[key] = val

    # 处理时间
    rt_date = result.get("DATE")
    rt_time = result.get("TIME")
    time_str = ""
    if rt_date and hasattr(rt_date, "strftime"):
        time_str = rt_date.strftime("%Y-%m-%d")
    if rt_time and hasattr(rt_time, "strftime"):
        time_str += " " + rt_time.strftime("%H:%M:%S")
    elif rt_time:
        time_str += " " + str(rt_time)

    return {
        "ok": True,
        "source": "wind",
        "price": result.get("LAST"),
        "open": result.get("OPEN"),
        "high": result.get("HIGH"),
        "low": result.get("LOW"),
        "volume": result.get("VOL"),
        "change": result.get("CHG"),
        "change_pct": result.get("PCT_CHG"),
        "pre_close": result.get("PRE_CLOSE"),
        "time": time_str.strip(),
    }


def _fetch_realtime_bloomberg(bbg_ticker):
    # type: (str) -> Dict[str, Any]
    """通过 Bloomberg API 获取境外标的实时行情快照 (subprocess)。"""
    import subprocess
    import socket

    try:
        with socket.create_connection(("localhost", 8194), timeout=1):
            pass
    except (OSError, socket.timeout):
        return {"ok": False, "error": "Bloomberg Terminal 未运行"}

    if not os.path.isfile(_VENV_PYTHON):
        return {"ok": False, "error": "未找到 venv Python"}
    if not os.path.isfile(_BBG_HELPER):
        return {"ok": False, "error": "未找到 bloomberg_helper.py"}

    try:
        env = os.environ.copy()
        if os.path.isdir(BBG_DLL_PATH):
            env["PATH"] = BBG_DLL_PATH + os.pathsep + env.get("PATH", "")

        result = subprocess.run(
            [_VENV_PYTHON, _BBG_HELPER,
             "--mode", "snapshot",
             "--ticker", bbg_ticker],
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )

        stdout = result.stdout.strip()
        if not stdout:
            return {"ok": False, "error": "Bloomberg helper 无输出"}

        try:
            data = json.loads(stdout)
        except ValueError:
            return {"ok": False, "error": "Bloomberg helper 输出无法解析: {}".format(stdout[:200])}

        return data

    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Bloomberg snapshot 超时 (>15s)"}
    except Exception as exc:
        return {"ok": False, "error": "Bloomberg 调用异常: {}".format(exc)}


def fetch_realtime_price(wind_ticker, name=""):
    # type: (str, str) -> Dict[str, Any]
    """
    获取标的实时行情快照。优先级:
      - 境内 → Wind wsq 实时行情 → HMC EOD 兜底
      - 境外 → Bloomberg ReferenceData → HMC EOD 兜底
    HMC 只有 EOD 日线数据, 因此仅在 Wind / Bloomberg 都失败时作为兜底。
    """
    import time as _time

    # ── 短时缓存 ──
    now = _time.time()
    if wind_ticker in _RT_CACHE and (now - _RT_CACHE_TS.get(wind_ticker, 0)) < _RT_CACHE_TTL:
        cached = dict(_RT_CACHE[wind_ticker])
        cached["cached"] = True
        return cached

    result = None

    if _is_domestic(wind_ticker):
        # ── 境内: 优先 Wind 实时行情 ──
        result = _fetch_wind_realtime_subprocess(wind_ticker)
        if not (result.get("ok") and result.get("price") is not None):
            wind_err = result.get("error", "")
            print("[REALTIME] Wind 失败 ({}), 尝试 HMC 兜底: {}".format(wind_err[:60], wind_ticker))
            result = _fetch_hmc_realtime_subprocess(wind_ticker)
    else:
        # ── 境外: 优先 Bloomberg 实时行情 ──
        from ticker_mapping import resolve_bbg_ticker
        bbg_code = resolve_bbg_ticker(wind_ticker)
        if bbg_code:
            result = _fetch_realtime_bloomberg(bbg_code)
        else:
            result = {"ok": False, "error": "无法映射为 Bloomberg ticker: {}".format(wind_ticker)}

        if not (result.get("ok") and result.get("price") is not None):
            bbg_err = result.get("error", "")
            print("[REALTIME] Bloomberg 失败 ({}), 尝试 HMC 兜底: {}".format(bbg_err[:60], wind_ticker))
            result = _fetch_hmc_realtime_subprocess(wind_ticker)

    if result.get("ok"):
        result["ticker"] = wind_ticker
        result["name"] = name
        _RT_CACHE[wind_ticker] = result
        _RT_CACHE_TS[wind_ticker] = now

    return result
