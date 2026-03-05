"""
Bloomberg 数据拉取辅助脚本 — 与主 Flask 进程隔离运行。

设计:
  - 由 market_data.py 通过 subprocess 调用 (使用 venv 中含 blpapi 的 Python)
  - 接受命令行参数: --ticker BBG_TICKER --start YYYY-MM-DD --end YYYY-MM-DD
  - 结果以 JSON 打印到 stdout
  - 错误也以 {"ok": false, "error": "..."} JSON 格式输出

使用方法:
  python bloomberg_helper.py --ticker "CL1 Comdty" --start 2025-09-01 --end 2026-03-05
"""

import os
import sys
import math
import json
import argparse

# ── Bloomberg DLL 路径 ──────────────────────────────────────────────────────
BBG_DLL_PATH = r"C:\blp\DAPI"
if os.path.isdir(BBG_DLL_PATH) and BBG_DLL_PATH not in os.environ.get("PATH", ""):
    os.environ["PATH"] = BBG_DLL_PATH + os.pathsep + os.environ.get("PATH", "")


def _out(data):
    """向 stdout 输出 JSON, 然后退出。"""
    print(json.dumps(data, ensure_ascii=False))
    sys.stdout.flush()


def fetch_snapshot(bbg_ticker):
    """获取实时行情快照 (ReferenceDataRequest) 并打印 JSON。"""
    import socket
    try:
        with socket.create_connection(("localhost", 8194), timeout=1):
            pass
    except (OSError, socket.timeout):
        return _out({"ok": False, "error": "Bloomberg Terminal 未运行"})

    try:
        import blpapi
    except (ImportError, OSError) as e:
        return _out({"ok": False, "error": "blpapi 无法加载: {}".format(e)})

    try:
        options = blpapi.SessionOptions()
        options.setServerHost("localhost")
        options.setServerPort(8194)
        session = blpapi.Session(options)
        if not session.start():
            return _out({"ok": False, "error": "无法连接 Bloomberg"})
        if not session.openService("//blp/refdata"):
            session.stop()
            return _out({"ok": False, "error": "无法打开 refdata 服务"})

        service = session.getService("//blp/refdata")
        request = service.createRequest("ReferenceDataRequest")
        request.getElement("securities").appendValue(bbg_ticker)
        for f in ["LAST_PRICE", "OPEN", "HIGH", "LOW", "VOLUME",
                   "CHG_NET_1D", "CHG_PCT_1D", "PREV_CLOSE_VALUE_DATE",
                   "PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PREV_CLOSING_PX"]:
            request.getElement("fields").appendValue(f)
        session.sendRequest(request)

        result = {}
        while True:
            event = session.nextEvent(5000)
            for msg in event:
                if msg.hasElement("securityData"):
                    sec_arr = msg.getElement("securityData")
                    for i in range(sec_arr.numValues()):
                        sec = sec_arr.getValue(i)
                        if sec.hasElement("fieldData"):
                            fd = sec.getElement("fieldData")

                            def _g(fn):
                                try:
                                    if fd.hasElement(fn):
                                        v = fd.getElementValue(fn)
                                        if isinstance(v, float) and math.isnan(v):
                                            return None
                                        return v
                                except Exception:
                                    pass
                                return None

                            result = {
                                "price": _g("LAST_PRICE") or _g("PX_LAST"),
                                "open": _g("OPEN") or _g("PX_OPEN"),
                                "high": _g("HIGH") or _g("PX_HIGH"),
                                "low": _g("LOW") or _g("PX_LOW"),
                                "volume": _g("VOLUME"),
                                "change": _g("CHG_NET_1D"),
                                "change_pct": _g("CHG_PCT_1D"),
                                "pre_close": _g("PREV_CLOSING_PX"),
                            }
            if event.eventType() == blpapi.Event.RESPONSE:
                break

        session.stop()

        if not result or result.get("price") is None:
            return _out({"ok": False, "error": "Bloomberg 未返回 {} 的实时数据".format(bbg_ticker)})

        result["ok"] = True
        result["source"] = "bloomberg"
        from datetime import datetime as _dt
        result["time"] = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
        return _out(result)

    except Exception as exc:
        return _out({"ok": False, "error": "Bloomberg snapshot 异常: {}".format(exc)})


def fetch(bbg_ticker, start_date, end_date):
    # type: (str, str, str) -> None
    """拉取数据并打印 JSON。"""
    import socket
    try:
        with socket.create_connection(("localhost", 8194), timeout=1):
            pass
    except (OSError, socket.timeout):
        return _out({"ok": False, "error": "Bloomberg Terminal 未运行 (localhost:8194 不可达)"})

    try:
        import blpapi
    except (ImportError, OSError) as e:
        return _out({"ok": False, "error": "blpapi 无法加载: {}".format(e)})

    try:
        options = blpapi.SessionOptions()
        options.setServerHost("localhost")
        options.setServerPort(8194)
        if hasattr(options, "setConnectTimeout"):
            options.setConnectTimeout(3000)
        session = blpapi.Session(options)
        try:
            started = session.start()
        except Exception as e:
            return _out({"ok": False, "error": "Bloomberg session.start() 异常: {}".format(e)})
        if not started:
            return _out({"ok": False, "error": "无法连接 Bloomberg，请确认 Terminal 已登录"})
        if not session.openService("//blp/refdata"):
            session.stop()
            return _out({"ok": False, "error": "无法打开 Bloomberg refdata 服务"})

        service = session.getService("//blp/refdata")
        start_str = start_date.replace("-", "")
        end_str = end_date.replace("-", "")

        request = service.createRequest("HistoricalDataRequest")
        request.getElement("securities").appendValue(bbg_ticker)
        for f in ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "VOLUME", "OPEN_INT"]:
            request.getElement("fields").appendValue(f)
        request.set("startDate", start_str)
        request.set("endDate", end_str)
        request.set("periodicitySelection", "DAILY")
        session.sendRequest(request)

        rows = []
        while True:
            event = session.nextEvent(5000)
            for msg in event:
                if msg.hasElement("securityData"):
                    sec_data = msg.getElement("securityData")
                    field_data_arr = sec_data.getElement("fieldData")
                    for i in range(field_data_arr.numValues()):
                        row = field_data_arr.getValue(i)
                        d = row.getElementAsDatetime("date")
                        d_str = d.date().strftime("%Y-%m-%d") if hasattr(d, "date") else str(d)

                        def _get(fn):
                            try:
                                if row.hasElement(fn):
                                    v = row.getElementValue(fn)
                                    return None if (isinstance(v, float) and math.isnan(v)) else v
                            except Exception:
                                pass
                            return None

                        rows.append({
                            "date": d_str,
                            "open": _get("PX_OPEN"),
                            "high": _get("PX_HIGH"),
                            "low": _get("PX_LOW"),
                            "close": _get("PX_LAST"),
                            "volume": _get("VOLUME"),
                            "oi": _get("OPEN_INT"),
                        })
            if event.eventType() == blpapi.Event.RESPONSE:
                break

        session.stop()

        if not rows:
            return _out({"ok": False, "error": "Bloomberg 未返回 {} 的历史数据".format(bbg_ticker)})

        ohlcv = [{"date": r["date"], "open": r["open"], "high": r["high"],
                  "low": r["low"], "close": r["close"], "volume": r["volume"]} for r in rows]
        oi = [{"date": r["date"], "value": r["oi"]} for r in rows if r.get("oi") is not None]

        return _out({
            "ok": True,
            "source": "bloomberg",
            "ohlcv": ohlcv,
            "fundamentals": {"open_interest": oi, "basis": []},
        })

    except Exception as exc:
        return _out({"ok": False, "error": "Bloomberg 异常: {}".format(exc)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bloomberg data helper")
    parser.add_argument("--ticker", required=True, help="Bloomberg ticker (e.g. 'CL1 Comdty')")
    parser.add_argument("--mode", default="historical", choices=["historical", "snapshot"],
                        help="Mode: 'historical' (default) or 'snapshot' (realtime)")
    parser.add_argument("--start", help="Start date YYYY-MM-DD (historical mode)")
    parser.add_argument("--end", help="End date YYYY-MM-DD (historical mode)")
    args = parser.parse_args()

    if args.mode == "snapshot":
        fetch_snapshot(args.ticker)
    else:
        if not args.start or not args.end:
            _out({"ok": False, "error": "historical 模式需要 --start 和 --end 参数"})
        else:
            fetch(args.ticker, args.start, args.end)
