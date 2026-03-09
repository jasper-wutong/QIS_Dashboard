"""
Wind 数据拉取辅助脚本 — 与主 Flask 进程隔离运行。

设计:
  - 由 market_data.py 通过 subprocess 调用 (使用含 WindPy 的 Python 3.7)
  - 接受命令行参数: --ticker WIND_TICKER --start YYYY-MM-DD --end YYYY-MM-DD
  - 结果以 JSON 打印到 stdout
  - 错误也以 {"ok": false, "error": "..."} JSON 格式输出

使用方法:
  python wind_helper.py --ticker "HC.SHF" --start 2025-09-01 --end 2026-03-06
  python wind_helper.py --mode snapshot --ticker "HC2605.SHF"
"""

import os
import sys
import math
import json
import argparse

# ── Wind 客户端路径 ────────────────────────────────────────────────────────────
_WIND_PATH = r"C:\Wind\Wind.NET.Client\WindNET\x64"
if _WIND_PATH not in sys.path:
    sys.path.insert(0, _WIND_PATH)


def _out(data):
    """向 stdout 输出 JSON, 然后退出。"""
    print(json.dumps(data, ensure_ascii=False))
    sys.stdout.flush()


def _safe_val(v):
    """将 NaN / Inf / None 转为 None。"""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _ensure_connected():
    """确保 Wind 已连接, 返回 (w, error_msg)。"""
    try:
        from WindPy import w
    except (ImportError, OSError) as e:
        return None, "WindPy 无法加载: {}".format(e)

    if not w.isconnected():
        # 抑制 Wind 启动时的 Welcome 横幅 (输出到 stdout)
        import io
        _old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            ret = w.start()
        finally:
            sys.stdout = _old_stdout
        if not w.isconnected():
            return None, "Wind 连接失败: {}".format(ret)

    return w, None


def fetch_history(wind_ticker, start_date, end_date):
    """获取历史 OHLCV + OI + Basis 并打印 JSON。"""
    w, err = _ensure_connected()
    if err:
        return _out({"ok": False, "error": err})

    # ── OHLCV + OI ──
    fields = "open,high,low,close,volume,oi"
    data = w.wsd(wind_ticker, fields, start_date, end_date, "")

    if data.ErrorCode != 0:
        return _out({"ok": False, "error": "Wind wsd 错误码: {} (ticker={})".format(data.ErrorCode, wind_ticker)})

    if not data.Data or not data.Times:
        return _out({"ok": False, "error": "Wind 返回空数据 (ticker={})".format(wind_ticker)})

    dates = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in data.Times]
    n = len(dates)

    def _safe_list(arr):
        if arr is None:
            return [None] * n
        return [_safe_val(v) for v in arr]

    open_prices  = _safe_list(data.Data[0] if len(data.Data) > 0 else None)
    high_prices  = _safe_list(data.Data[1] if len(data.Data) > 1 else None)
    low_prices   = _safe_list(data.Data[2] if len(data.Data) > 2 else None)
    close_prices = _safe_list(data.Data[3] if len(data.Data) > 3 else None)
    volumes      = _safe_list(data.Data[4] if len(data.Data) > 4 else None)
    oi_data      = _safe_list(data.Data[5] if len(data.Data) > 5 else None)

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

    oi_list = [{"date": dates[i], "value": oi_data[i]} for i in range(n) if oi_data[i] is not None]

    # ── Basis (现货-期货) ──
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

    _out({
        "ok": True,
        "source": "wind",
        "ohlcv": ohlcv,
        "rows": len(ohlcv),
        "fundamentals": {
            "open_interest": oi_list,
            "basis": basis_data,
        },
    })


def fetch_snapshot(wind_ticker):
    """获取实时行情快照 (wsq)。"""
    w, err = _ensure_connected()
    if err:
        return _out({"ok": False, "error": err})

    fields = "rt_last,rt_open,rt_high,rt_low,rt_vol,rt_chg,rt_pct_chg,rt_pre_close"
    data = w.wsq(wind_ticker, fields)

    if data.ErrorCode != 0:
        return _out({"ok": False, "error": "Wind wsq 错误码: {} (ticker={})".format(data.ErrorCode, wind_ticker)})

    if not data.Data or not data.Fields:
        return _out({"ok": False, "error": "Wind wsq 返回空数据 (ticker={})".format(wind_ticker)})

    result = {}
    for i, field in enumerate(data.Fields):
        val = data.Data[i][0] if data.Data[i] else None
        val = _safe_val(val)
        key = field.upper().replace("RT_", "")
        result[key] = val

    _out({
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
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wind 数据拉取辅助脚本")
    parser.add_argument("--mode", default="history", choices=["history", "snapshot"],
                        help="history=历史行情, snapshot=实时快照")
    parser.add_argument("--ticker", required=True, help="Wind Ticker, 如 HC.SHF")
    parser.add_argument("--start", default="", help="起始日期 YYYY-MM-DD (history模式)")
    parser.add_argument("--end", default="", help="结束日期 YYYY-MM-DD (history模式)")
    args = parser.parse_args()

    if args.mode == "snapshot":
        fetch_snapshot(args.ticker)
    else:
        if not args.start or not args.end:
            _out({"ok": False, "error": "--start 和 --end 参数在 history 模式下必填"})
        else:
            fetch_history(args.ticker, args.start, args.end)
