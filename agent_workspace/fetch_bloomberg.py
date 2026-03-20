"""
fetch_bloomberg.py — Agent-facing Bloomberg data helper
=========================================================
Wraps bloomberg/bloomberg_helper.py subprocess calls.
Works with the venv Python + blpapi.

Usage (CLI):
    python agent_workspace/fetch_bloomberg.py --tickers "GC1 Comdty,SI1 Comdty" --mode snapshot
    python agent_workspace/fetch_bloomberg.py --ticker "GC1 Comdty" --mode historical --start 2025-01-01 --end 2025-03-18

Usage (import):
    from agent_workspace.fetch_bloomberg import snapshot, historical, multi_snapshot
    data = snapshot("GC1 Comdty")
    report = multi_snapshot(["GC1 Comdty", "DXY Curncy", "GTII10 Govt"])
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
_VENV_PYTHON = str(_ROOT / ".venv" / "Scripts" / "python.exe")
_BBG_HELPER  = str(_ROOT / "bloomberg" / "bloomberg_helper.py")


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

def snapshot(bbg_ticker: str) -> Dict[str, Any]:
    """Fetch real-time snapshot for a single Bloomberg ticker."""
    python = _VENV_PYTHON if os.path.isfile(_VENV_PYTHON) else sys.executable
    try:
        proc = subprocess.run(
            [python, _BBG_HELPER, "--ticker", bbg_ticker, "--mode", "snapshot"],
            capture_output=True, text=True, timeout=15,
        )
        data = _extract_json(proc.stdout)
        if data:
            data["ticker"] = bbg_ticker
            return data
        err = proc.stderr.strip()[:300] if proc.stderr else proc.stdout.strip()[:300]
        return {"ok": False, "ticker": bbg_ticker, "error": f"No JSON in output: {err}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "ticker": bbg_ticker, "error": "Timeout (>15s)"}
    except Exception as exc:
        return {"ok": False, "ticker": bbg_ticker, "error": str(exc)}


def historical(bbg_ticker: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """Fetch OHLCV history for a Bloomberg ticker."""
    python = _VENV_PYTHON if os.path.isfile(_VENV_PYTHON) else sys.executable
    try:
        proc = subprocess.run(
            [python, _BBG_HELPER, "--ticker", bbg_ticker,
             "--mode", "historical", "--start", start_date, "--end", end_date],
            capture_output=True, text=True, timeout=60,
        )
        data = _extract_json(proc.stdout)
        if data:
            data["ticker"] = bbg_ticker
            return data
        err = proc.stderr.strip()[:300] if proc.stderr else proc.stdout.strip()[:300]
        return {"ok": False, "ticker": bbg_ticker, "error": f"No JSON: {err}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "ticker": bbg_ticker, "error": "Timeout (>60s)"}
    except Exception as exc:
        return {"ok": False, "ticker": bbg_ticker, "error": str(exc)}


def multi_snapshot(tickers: List[str]) -> Dict[str, Any]:
    """
    Fetch snapshots for multiple tickers in parallel (sequentially for stability).
    Returns dict keyed by ticker.
    """
    results = {}
    for t in tickers:
        results[t] = snapshot(t)
    return results


# ─── Preset bundles ───────────────────────────────────────────────────────────

GOLD_TICKERS = {
    "spot":        "XAU Curncy",        # Gold spot USD/oz
    "front_future":"GC1 Comdty",        # COMEX Gold continuous
    "au_apr26":    "AUAM26 Comdty",     # COMEX Apr-2026 future (our book)
    "real_rate":   "GTII10 Govt",       # US 10Y real yield (TIPS)
    "dxy":         "DXY Curncy",        # US Dollar Index
    "breakeven":   "USGGBE10 Index",    # 10Y inflation breakeven
    "gld_etf":     "GLDUS Equity",      # SPDR GLD ETF
    "cot_net":     "CFTGCNET Index",    # CFTC COT net position
    "silver":      "SI1 Comdty",        # Silver front future
    "platinum":    "PL1 Comdty",        # Platinum front future
}

RATES_TICKERS = {
    "us2y":   "USGG2YR Index",
    "us10y":  "USGG10YR Index",
    "us30y":  "USGG30YR Index",
    "cn10y":  "GCNY10YR Index",
    "de10y":  "GDBR10 Index",
    "vix":    "VIX Index",
    "move":   "MOVE Index",
}

ENERGY_TICKERS = {
    "wti":    "CL1 Comdty",
    "brent":  "CO1 Comdty",
    "natgas": "NG1 Comdty",
    "rbob":   "XB1 Comdty",
    "heat":   "HO1 Comdty",
}

METALS_TICKERS = {
    "copper_lme":  "LP1 Comdty",
    "alu_lme":     "LA1 Comdty",
    "zinc_lme":    "LX1 Comdty",
    "nickel_lme":  "LN1 Comdty",
    "iron_ore":    "TIOA Comdty",
}


def gold_snapshot() -> Dict[str, Any]:
    """Fetch all gold-related Bloomberg tickers and return a formatted summary."""
    raw = multi_snapshot(list(GOLD_TICKERS.values()))
    out = {"ok": True, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "data": {}}
    lines = []
    for label, ticker in GOLD_TICKERS.items():
        r = raw.get(ticker, {})
        if r.get("ok"):
            price      = r.get("price")
            chg_pct    = r.get("change_pct")
            out["data"][label] = {"ticker": ticker, "price": price, "change_pct": chg_pct}
            pct_str = f"{chg_pct:+.2f}%" if chg_pct is not None else "N/A"
            lines.append(f"  {label:<16} {ticker:<22} {price:>10.4f}  {pct_str}" if price else
                         f"  {label:<16} {ticker:<22} N/A")
        else:
            out["data"][label] = {"ticker": ticker, "error": r.get("error", "?")}
            lines.append(f"  {label:<16} {ticker:<22} ERROR: {r.get('error','?')[:50]}")
    out["summary"] = "\n".join(lines)
    return out


def market_snapshot() -> Dict[str, Any]:
    """Fetch a broad cross-asset snapshot: gold + rates + energy."""
    all_tickers = {**GOLD_TICKERS, **RATES_TICKERS, **ENERGY_TICKERS}
    raw = multi_snapshot(list(all_tickers.values()))
    out = {"ok": True, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "data": {}}
    sections = []
    for section_name, tdict in [("GOLD", GOLD_TICKERS), ("RATES", RATES_TICKERS), ("ENERGY", ENERGY_TICKERS)]:
        section_lines = [f"\n  [{section_name}]"]
        for label, ticker in tdict.items():
            r = raw.get(ticker, {})
            if r.get("ok"):
                price   = r.get("price")
                chg_pct = r.get("change_pct")
                out["data"][label] = {"ticker": ticker, "price": price, "change_pct": chg_pct}
                pct_str = f"{chg_pct:+.2f}%" if chg_pct is not None else "N/A"
                section_lines.append(
                    f"    {label:<16} {price:>10.4f}  {pct_str}" if price else
                    f"    {label:<16} N/A"
                )
            else:
                section_lines.append(f"    {label:<16} ERROR")
        sections.append("\n".join(section_lines))
    out["summary"] = "\n".join(sections)
    return out


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bloomberg data fetcher for agent_workspace")
    parser.add_argument("--ticker",  help="Single BBG ticker")
    parser.add_argument("--tickers", help="Comma-separated BBG tickers for multi-snapshot")
    parser.add_argument("--mode",    default="snapshot", choices=["snapshot", "historical", "gold", "market"])
    parser.add_argument("--start",   help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    end = args.end or datetime.now().strftime("%Y-%m-%d")
    start = args.start or (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    if args.mode == "gold":
        result = gold_snapshot()
        print(f"\n=== Gold Market Snapshot [{result['timestamp']}] ===")
        print(result.get("summary", ""))
        print("\n[JSON]\n" + json.dumps(result["data"], indent=2, ensure_ascii=False))
    elif args.mode == "market":
        result = market_snapshot()
        print(f"\n=== Cross-Asset Snapshot [{result['timestamp']}] ===")
        print(result.get("summary", ""))
    elif args.mode == "snapshot":
        tickers = []
        if args.tickers:
            tickers = [t.strip() for t in args.tickers.split(",")]
        elif args.ticker:
            tickers = [args.ticker]
        else:
            print("Error: specify --ticker or --tickers for snapshot mode"); sys.exit(1)
        for t in tickers:
            r = snapshot(t)
            print(json.dumps(r, indent=2, ensure_ascii=False))
    elif args.mode == "historical":
        if not args.ticker:
            print("Error: --ticker required for historical mode"); sys.exit(1)
        r = historical(args.ticker, start, end)
        print(json.dumps(r, indent=2, ensure_ascii=False))
