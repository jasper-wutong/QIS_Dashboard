"""
fetch_wind.py — Agent-facing Wind (万得) data helper
======================================================
Wraps wind_helper.py subprocess calls using the Wind Python 3.7 environment.

Usage (CLI):
    python agent_workspace/fetch_wind.py --ticker AU.SHF --mode snapshot
    python agent_workspace/fetch_wind.py --ticker HC.SHF --mode historical --start 2025-01-01 --end 2025-03-18
    python agent_workspace/fetch_wind.py --mode metals    # bulk snapshot of domestic metals

Usage (import):
    from agent_workspace.fetch_wind import snapshot, historical, metals_snapshot
    data = snapshot("AU.SHF")
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

# ─── Paths ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
# Wind requires Python 3.7 (system install with WindPy)
_WIND_PYTHON = r"C:\Users\wutong6\AppData\Local\Programs\Python\Python37\python.exe"
_WIND_HELPER  = str(_ROOT / "wind_helper.py")


def _find_wind_python() -> str:
    """Return best available Wind Python path."""
    candidates = [
        _WIND_PYTHON,
        r"C:\Python37\python.exe",
        r"C:\Python37-32\python.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Fall back to current interpreter (may lack WindPy)
    return sys.executable


def _extract_json(stdout: str) -> Optional[Dict]:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                pass
    return None


# ─── Core calls ───────────────────────────────────────────────────────────────

def snapshot(wind_ticker: str) -> Dict[str, Any]:
    """Fetch real-time Wind snapshot (wsq) for a single ticker."""
    python = _find_wind_python()
    try:
        proc = subprocess.run(
            [python, _WIND_HELPER, "--mode", "snapshot", "--ticker", wind_ticker],
            capture_output=True, text=True, timeout=20,
        )
        data = _extract_json(proc.stdout)
        if data:
            data["ticker"] = wind_ticker
            return data
        err = proc.stderr.strip()[:300] if proc.stderr else proc.stdout.strip()[:300]
        return {"ok": False, "ticker": wind_ticker, "error": f"No JSON: {err}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "ticker": wind_ticker, "error": "Timeout (>20s)"}
    except Exception as exc:
        return {"ok": False, "ticker": wind_ticker, "error": str(exc)}


def historical(wind_ticker: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """Fetch OHLCV history for a Wind ticker."""
    python = _find_wind_python()
    try:
        proc = subprocess.run(
            [python, _WIND_HELPER, "--mode", "history",
             "--ticker", wind_ticker, "--start", start_date, "--end", end_date],
            capture_output=True, text=True, timeout=60,
        )
        data = _extract_json(proc.stdout)
        if data:
            data["ticker"] = wind_ticker
            return data
        err = proc.stderr.strip()[:300] if proc.stderr else proc.stdout.strip()[:300]
        return {"ok": False, "ticker": wind_ticker, "error": f"No JSON: {err}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "ticker": wind_ticker, "error": "Timeout (>60s)"}
    except Exception as exc:
        return {"ok": False, "ticker": wind_ticker, "error": str(exc)}


def multi_snapshot(tickers: List[str]) -> Dict[str, Any]:
    """Snapshot multiple Wind tickers; returns dict keyed by ticker."""
    return {t: snapshot(t) for t in tickers}


# ─── Preset ticker bundles ────────────────────────────────────────────────────

# Domestic futures main continuous contracts (Wind continuous format: SYMBOL.EXCHANGE)
PRECIOUS_METALS = {
    "gold_shfe":   "AU.SHF",    # 沪金主力连续
    "silver_shfe": "AG.SHF",    # 沪银主力连续
}

BASE_METALS = {
    "copper_shfe": "CU.SHF",
    "alu_shfe":    "AL.SHF",
    "zinc_shfe":   "ZN.SHF",
    "nickel_shfe": "NI.SHF",
    "tin_shfe":    "SN.SHF",
    "lead_shfe":   "PB.SHF",
    "iron_ore_dce":"I.DCE",
    "rebar_shfe":  "RB.SHF",
    "hrc_shfe":    "HC.SHF",
    "stainless":   "SS.SHF",
}

ENERGY = {
    "crude_ine":   "SC.INE",    # 上海原油
    "fuel_oil":    "FU.SHF",
    "bitumen":     "BU.SHF",
    "lpg":         "PG.DCE",
    "pvc":         "V.DCE",
    "pp":          "PP.DCE",
    "ldpe":        "L.DCE",
}

AGRI = {
    "soybean":     "A.DCE",
    "soy_meal":    "M.DCE",
    "soy_oil":     "Y.DCE",
    "palm_oil":    "P.DCE",
    "corn":        "C.DCE",
    "cotton":      "CF.CZC",
    "white_sugar": "SR.CZC",
}

RATES_FUTURES = {
    "t_bond_10y":  "T.CFE",     # 中国10Y国债期货
    "t_bond_5y":   "TF.CFE",    # 5Y国债期货
    "t_bond_2y":   "TS.CFE",    # 2Y国债期货
    "t_bond_30y":  "TL.CFE",    # 30Y国债期货
}

EQUITY_INDEX_FUTURES = {
    "csi300":  "IF.CFE",
    "csi500":  "IC.CFE",
    "sse50":   "IH.CFE",
    "csi1000": "IM.CFE",
}


def metals_snapshot() -> Dict[str, Any]:
    """Snapshot all domestic metals (precious + base)."""
    all_metals = {**PRECIOUS_METALS, **BASE_METALS}
    raw = multi_snapshot(list(all_metals.values()))

    out = {"ok": True, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "data": {}}
    lines = ["  [PRECIOUS]"]
    for label, ticker in PRECIOUS_METALS.items():
        r = raw.get(ticker, {})
        _fmt_row(lines, label, ticker, r, out)
    lines.append("  [BASE METALS]")
    for label, ticker in BASE_METALS.items():
        r = raw.get(ticker, {})
        _fmt_row(lines, label, ticker, r, out)
    out["summary"] = "\n".join(lines)
    return out


def gold_snapshot() -> Dict[str, Any]:
    """Fetch AU.SHF and AG.SHF snapshots with additional detail."""
    tickers = {"gold_shfe": "AU.SHF", "silver_shfe": "AG.SHF"}
    raw = multi_snapshot(list(tickers.values()))
    out = {"ok": True, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "data": {}}
    lines = ["  [Wind 贵金属实时行情]"]
    for label, ticker in tickers.items():
        r = raw.get(ticker, {})
        _fmt_row(lines, label, ticker, r, out)
    out["summary"] = "\n".join(lines)
    return out


def _fmt_row(lines: list, label: str, ticker: str, r: Dict, out: Dict) -> None:
    if r.get("ok"):
        price   = r.get("price")
        chg_pct = r.get("change_pct")
        out["data"][label] = {"ticker": ticker, "price": price, "change_pct": chg_pct,
                               "open": r.get("open"), "high": r.get("high"), "low": r.get("low")}
        pct_str = f"{chg_pct:+.2f}%" if chg_pct is not None else "N/A"
        lines.append(
            f"    {label:<20} {ticker:<14} {price:>10.2f}  {pct_str}" if price else
            f"    {label:<20} {ticker:<14} N/A"
        )
    else:
        out["data"][label] = {"ticker": ticker, "error": r.get("error", "?")}
        lines.append(f"    {label:<20} {ticker:<14} ERROR: {r.get('error','?')[:50]}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wind data fetcher for agent_workspace")
    parser.add_argument("--ticker",  help="Single Wind ticker, e.g. AU.SHF")
    parser.add_argument("--tickers", help="Comma-separated Wind tickers")
    parser.add_argument("--mode",    default="snapshot",
                        choices=["snapshot", "historical", "metals", "gold", "rates"])
    parser.add_argument("--start",   help="Start YYYY-MM-DD")
    parser.add_argument("--end",     help="End YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    end   = args.end   or datetime.now().strftime("%Y-%m-%d")
    start = args.start or (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    if args.mode == "metals":
        r = metals_snapshot()
        print(f"\n=== Wind Metals Snapshot [{r['timestamp']}] ===")
        print(r.get("summary", ""))
    elif args.mode == "gold":
        r = gold_snapshot()
        print(f"\n=== Wind Gold Snapshot [{r['timestamp']}] ===")
        print(r.get("summary", ""))
        print("\n[JSON]\n" + json.dumps(r["data"], indent=2, ensure_ascii=False))
    elif args.mode == "rates":
        raw = multi_snapshot(list(RATES_FUTURES.values()))
        print("\n  [Wind 国债期货]")
        for label, ticker in RATES_FUTURES.items():
            r = raw[ticker]
            if r.get("ok"):
                print(f"    {label:<16} {ticker:<12} {r.get('price','N/A'):>10}  {r.get('change_pct','N/A')}")
            else:
                print(f"    {label:<16} {ticker:<12} ERROR")
    elif args.mode == "snapshot":
        tickers = []
        if args.tickers:
            tickers = [t.strip() for t in args.tickers.split(",")]
        elif args.ticker:
            tickers = [args.ticker]
        else:
            print("Error: specify --ticker or --tickers"); sys.exit(1)
        for t in tickers:
            r = snapshot(t)
            print(json.dumps(r, indent=2, ensure_ascii=False))
    elif args.mode == "historical":
        if not args.ticker:
            print("Error: --ticker required for historical"); sys.exit(1)
        r = historical(args.ticker, start, end)
        if r.get("ok"):
            ohlcv = r.get("ohlcv", [])
            print(f"Fetched {len(ohlcv)} rows for {args.ticker}")
            if ohlcv:
                last = ohlcv[-1]
                print(f"Latest: {last}")
        else:
            print(json.dumps(r, indent=2, ensure_ascii=False))
