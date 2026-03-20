"""
analyze.py — One-command asset analysis for QIS小助手
======================================================
Aggregates Bloomberg real-time data, Wind domestic prices, and Cross Gamma
into a structured report ready for analysis or speech generation.

Usage:
    python agent_workspace/analyze.py --asset gold
    python agent_workspace/analyze.py --asset copper
    python agent_workspace/analyze.py --asset oil
    python agent_workspace/analyze.py --asset metals     # all domestic metals
    python agent_workspace/analyze.py --asset book       # full book cross gamma summary
    python agent_workspace/analyze.py --asset market     # broad cross-asset snapshot

Output is always printed to stdout (human-readable + JSON block at the end).
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

# ─── Imports from agent_workspace ─────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fetch_bloomberg  import (
    snapshot       as bbg_snapshot,
    historical     as bbg_historical,
    multi_snapshot as bbg_multi,
    gold_snapshot  as bbg_gold,
    market_snapshot as bbg_market,
    ENERGY_TICKERS, METALS_TICKERS, RATES_TICKERS,
)
from fetch_wind import (
    snapshot       as wind_snapshot,
    gold_snapshot  as wind_gold,
    metals_snapshot as wind_metals,
    RATES_FUTURES, EQUITY_INDEX_FUTURES,
)
from fetch_cross_gamma import (
    asset_exposure as cg_asset,
    top_pairs      as cg_top,
    book_summary   as cg_book,
)


# ─── Helper ───────────────────────────────────────────────────────────────────

def _sep(title: str = "") -> str:
    line = "─" * 72
    if title:
        pad = max(0, (72 - len(title) - 2) // 2)
        return f"\n{'─'*pad} {title} {'─'*pad}\n"
    return f"\n{line}\n"


def _fmt_price(val, decimals: int = 2) -> str:
    if val is None:
        return "N/A"
    return f"{val:,.{decimals}f}"


def _fmt_pct(val) -> str:
    if val is None:
        return "N/A"
    return f"{val:+.2f}%"


# ═══════════════════════════════════════════════════════════════════════════════
#  Asset analysis routines
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_gold(verbose: bool = True) -> Dict[str, Any]:
    """Full gold analysis: BBG real-time + Wind SHFE + Cross Gamma."""
    print(_sep("GOLD ANALYSIS"))
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ── Bloomberg: price + macro drivers ──────────────────────────────────────
    print("  [Bloomberg Real-time]")
    bbg = bbg_gold()
    if bbg.get("ok"):
        d = bbg["data"]
        rows = [
            ("Gold Spot (XAU)",    "spot",        "$/oz"),
            ("Gold Front (GC1)",   "front_future","$/oz"),
            ("Gold Apr26 (AUAM26)","au_apr26",    "$/oz"),
            ("Silver (SI1)",       "silver",      "$/oz"),
            ("Platinum (PL1)",     "platinum",    "$/oz"),
            ("Real Rate 10Y",      "real_rate",   "%"),
            ("DXY",                "dxy",         ""),
            ("10Y Breakeven",      "breakeven",   "%"),
            ("GLD ETF Holdinigs",  "gld_etf",     ""),
            ("COT Net Long",       "cot_net",     ""),
        ]
        for label, key, unit in rows:
            entry = d.get(key, {})
            if "error" not in entry:
                price   = entry.get("price")
                chg_pct = entry.get("change_pct")
                print(f"    {label:<28} {_fmt_price(price):>12} {unit:<5}  {_fmt_pct(chg_pct)}")
    else:
        print("    (Bloomberg unavailable — check Terminal connection)")

    # ── Wind: SHFE gold ───────────────────────────────────────────────────────
    print("\n  [Wind SHFE 实时]")
    w_gold = wind_gold()
    if w_gold.get("ok"):
        for label, entry in w_gold["data"].items():
            if "error" not in entry:
                price   = entry.get("price")
                chg_pct = entry.get("change_pct")
                high    = entry.get("high")
                low     = entry.get("low")
                print(f"    {label:<20} {_fmt_price(price):>12} CNY/g  "
                      f"{_fmt_pct(chg_pct)}  H:{_fmt_price(high)}  L:{_fmt_price(low)}")
    else:
        print("    (Wind unavailable — check Wind Terminal)")

    # ── Cross Gamma ───────────────────────────────────────────────────────────
    print()
    cg = cg_asset("AU")
    if cg.get("ok"):
        print(cg["summary"])
    else:
        print(f"  [Cross Gamma] ERROR: {cg.get('error')}")

    # ── Rates context ─────────────────────────────────────────────────────────
    print(_sep("Rates Context"))
    rates_raw = bbg_multi(list(RATES_TICKERS.values()))
    for label, ticker in RATES_TICKERS.items():
        r = rates_raw.get(ticker, {})
        if r.get("ok"):
            print(f"    {label:<10} {_fmt_price(r.get('price'), 3):>10}  {_fmt_pct(r.get('change_pct'))}")

    # ── Assemble structured output ─────────────────────────────────────────────
    result = {
        "ok": True,
        "asset": "gold",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bloomberg": bbg.get("data", {}),
        "wind": w_gold.get("data", {}),
        "cross_gamma": {
            "date": cg.get("date"),
            "row_gamma_cny": cg.get("row_gamma_cny"),
            "own_gamma_cny": cg.get("own_gamma_cny"),
            "top_pairs": cg.get("pairs", [])[:10],
        },
    }
    return result


def analyze_copper(verbose: bool = True) -> Dict[str, Any]:
    """Copper analysis: LME (BBG) + SHFE (Wind) + Cross Gamma."""
    print(_sep("COPPER ANALYSIS"))
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # LME copper
    print("  [Bloomberg LME Copper]")
    cu_tickers = {
        "Copper LME 3M": "LP1 Comdty",
        "Copper COMEX":  "HG1 Comdty",
        "LME Inventory": "LMCADY Index",
        "DXY":           "DXY Curncy",
    }
    for label, ticker in cu_tickers.items():
        r = bbg_snapshot(ticker)
        if r.get("ok"):
            print(f"    {label:<22} {_fmt_price(r.get('price')):>12}  {_fmt_pct(r.get('change_pct'))}")

    # Wind SHFE copper
    print("\n  [Wind SHFE 沪铜]")
    r_wind = wind_snapshot("CU.SHF")
    if r_wind.get("ok"):
        print(f"    CU.SHF  {_fmt_price(r_wind.get('price')):>12} CNY/t  {_fmt_pct(r_wind.get('change_pct'))}")

    # Cross Gamma
    print()
    cg = cg_asset("CU")
    if cg.get("ok"):
        print(cg["summary"])

    return {"ok": True, "asset": "copper",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def analyze_oil(verbose: bool = True) -> Dict[str, Any]:
    """Crude oil analysis: WTI + Brent (BBG) + SHFE SC (Wind) + Cross Gamma."""
    print(_sep("CRUDE OIL ANALYSIS"))
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("  [Bloomberg]")
    oil_tickers = {
        "WTI (CL1)":        "CL1 Comdty",
        "Brent (CO1)":      "CO1 Comdty",
        "Nat Gas NG1":      "NG1 Comdty",
        "RBOB Gasoline":    "XB1 Comdty",
        "Heating Oil":      "HO1 Comdty",
        "EIA Crude Stocks": "DOECRUOP Index",
    }
    for label, ticker in oil_tickers.items():
        r = bbg_snapshot(ticker)
        if r.get("ok"):
            print(f"    {label:<24} {_fmt_price(r.get('price')):>12}  {_fmt_pct(r.get('change_pct'))}")

    print("\n  [Wind INE 上海原油]")
    r_sc = wind_snapshot("SC.INE")
    if r_sc.get("ok"):
        print(f"    SC.INE  {_fmt_price(r_sc.get('price')):>12} CNY/bbl  {_fmt_pct(r_sc.get('change_pct'))}")

    print()
    cg = cg_asset("SC")
    if cg.get("ok"):
        print(cg["summary"])

    return {"ok": True, "asset": "oil",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def analyze_metals(verbose: bool = True) -> Dict[str, Any]:
    """All domestic metals snapshot."""
    print(_sep("DOMESTIC METALS SNAPSHOT"))
    r = wind_metals()
    if r.get("ok"):
        print(r["summary"])
    # Also LME
    print(_sep("LME Metals (Bloomberg)"))
    for label, ticker in METALS_TICKERS.items():
        r2 = bbg_snapshot(ticker)
        if r2.get("ok"):
            print(f"  {label:<18} {_fmt_price(r2.get('price')):>12}  {_fmt_pct(r2.get('change_pct'))}")
    return {"ok": True, "asset": "metals"}


def analyze_book(verbose: bool = True) -> Dict[str, Any]:
    """Full book cross gamma summary."""
    print(_sep("QIS BOOK CROSS GAMMA"))
    r = cg_book()
    if r.get("ok"):
        print(r["summary"])
        print()
        tp = cg_top(30)
        print(tp["summary"])
    else:
        print(f"ERROR: {r.get('error')}")
    return r


def analyze_market(verbose: bool = True) -> Dict[str, Any]:
    """Broad cross-asset market snapshot."""
    print(_sep("CROSS-ASSET MARKET SNAPSHOT"))
    r = bbg_market()
    if r.get("ok"):
        print(r["summary"])
    # Domestic rates
    print(_sep("Domestic Rates (Wind)"))
    for label, ticker in RATES_FUTURES.items():
        r2 = wind_snapshot(ticker)
        if r2.get("ok"):
            print(f"  {label:<16} {_fmt_price(r2.get('price')):>10}  {_fmt_pct(r2.get('change_pct'))}")
    # Equity index futures
    print(_sep("A-Share Equity Futures (Wind)"))
    for label, ticker in EQUITY_INDEX_FUTURES.items():
        r2 = wind_snapshot(ticker)
        if r2.get("ok"):
            print(f"  {label:<10} {_fmt_price(r2.get('price')):>10}  {_fmt_pct(r2.get('change_pct'))}")
    return {"ok": True, "asset": "market"}


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

ASSET_DISPATCH = {
    "gold":    analyze_gold,
    "au":      analyze_gold,
    "copper":  analyze_copper,
    "cu":      analyze_copper,
    "oil":     analyze_oil,
    "crude":   analyze_oil,
    "sc":      analyze_oil,
    "metals":  analyze_metals,
    "book":    analyze_book,
    "market":  analyze_market,
    "macro":   analyze_market,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QIS Agent One-Click Asset Analyzer")
    parser.add_argument(
        "--asset", "-a",
        default="gold",
        choices=list(ASSET_DISPATCH.keys()),
        help="Asset to analyze (default: gold)"
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output at the end")
    args = parser.parse_args()

    fn = ASSET_DISPATCH.get(args.asset.lower())
    if fn is None:
        print(f"Unknown asset: {args.asset}. Available: {', '.join(ASSET_DISPATCH)}")
        sys.exit(1)

    result = fn()

    if args.json:
        print(_sep("JSON Output"))
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
