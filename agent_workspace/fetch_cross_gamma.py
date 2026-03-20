"""
fetch_cross_gamma.py — Agent-facing Cross Gamma analysis helper
================================================================
Loads the latest (or a specific) Cross Gamma JSON and runs book aggregation.
Returns structured data useful for analysing hedging needs and correlated exposure.

Usage (CLI):
    python agent_workspace/fetch_cross_gamma.py                          # latest file
    python agent_workspace/fetch_cross_gamma.py --date 2026-03-17        # specific date
    python agent_workspace/fetch_cross_gamma.py --asset AU               # filter by asset
    python agent_workspace/fetch_cross_gamma.py --top 20                 # top 20 pairs

Usage (import):
    from agent_workspace.fetch_cross_gamma import load_latest, asset_exposure, top_pairs
    result = load_latest()
    au_exposure = asset_exposure("AU")
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# ─── Add parent to sys.path so cross_gamma package is importable ─────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cross_gamma.loader    import find_latest_file, find_all_files, load_cross_gamma_data
from cross_gamma.aggregator import aggregate_book

# EDS → 1% cash gamma conversion factor (same as aggregator)
_EDS_SCALE = 0.01

# Maps English shorthand to canonical Chinese display names used by ticker_map
_ASSET_ALIASES: Dict[str, str] = {
    "AU":    "黄金",
    "GOLD":  "黄金",
    "AG":    "白银",
    "SILVER":"白银",
    "CU":    "铜",
    "COPPER":"铜",
    "AL":    "铝",
    "NI":    "镍",
    "ZN":    "锌",
    "SC":    "原油SC",
    "OIL":   "原油WTI",
    "WTI":   "原油WTI",
    "BRENT": "布油Brent",
    "RB":    "螺纹钢",
    "HC":    "热卷",
    "I":     "铁矿",
    "IF":    "沪深300",
    "IC":    "中证500",
    "IM":    "中证1000",
    "IH":    "上证50",
    "T":     "国债10Y T",
    "TF":    "国债5Y TF",
    "TS":    "国债2Y TS",
    "TY":    "美债10Y TY",
    "FV":    "美债5Y FV",
    "TU":    "美债2Y TU",
}


def _resolve_asset(asset_code: str) -> List[str]:
    """Return all canonical names that match an asset code (English or Chinese)."""
    code = asset_code.strip().upper()
    if code in _ASSET_ALIASES:
        return [_ASSET_ALIASES[code]]
    # Try Chinese directly (e.g. "黄金")
    return [asset_code.strip()]


# ─── Core helpers ─────────────────────────────────────────────────────────────

def load_latest(date: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and aggregate cross gamma data.

    Parameters
    ----------
    date : YYYY-MM-DD string, or None for the most recent file.

    Returns
    -------
    dict with keys: ok, date, file_path, aggregated (full aggregate_book result),
    tickers, top_pairs, per_asset
    """
    if date:
        # Find the specific date
        files = find_all_files()
        match = [(d, p) for d, p in files if d == date]
        if not match:
            return {"ok": False, "error": f"No cross gamma file found for date={date}"}
        file_path, date_str = match[0][1], match[0][0]
    else:
        result = find_latest_file()
        file_path, date_or_err = result
        if file_path is None:
            return {"ok": False, "error": date_or_err or "No cross gamma files found"}
        date_str = date_or_err  # on success the second element is the date string

    try:
        raw_data, load_err = load_cross_gamma_data(file_path)
        if raw_data is None:
            return {"ok": False, "error": f"Failed to load {file_path}: {load_err}"}
    except Exception as exc:
        return {"ok": False, "error": f"Failed to load {file_path}: {exc}"}

    try:
        agg = aggregate_book(raw_data)
    except Exception as exc:
        return {"ok": False, "error": f"aggregate_book failed: {exc}"}

    return {
        "ok": True,
        "date": date_str,
        "file_path": str(file_path),
        "aggregated": agg,
        "tickers": agg.get("tickers", []),
        "top_pairs": agg.get("top_pairs", []),
        "per_asset": agg.get("per_asset", []),
        "stats": agg.get("stats", {}),
    }


def asset_exposure(asset_code: str, date: Optional[str] = None) -> Dict[str, Any]:
    """
    Return all cross gamma pairs involving a specific asset.

    Accepts English codes (AU, CU, IF, TF...) or Chinese names (黄金, 铜...).
    Values in CNY, 1% move cross gamma (already scaled by EDS_SCALE inside aggregator).
    """
    base = load_latest(date)
    if not base["ok"]:
        return base

    canonical_names = _resolve_asset(asset_code)
    all_top_pairs   = base.get("top_pairs", [])   # keys: a, b, value
    per_asset_list  = base.get("per_asset", [])   # keys: ticker, asset_class, row_gamma, own_gamma

    matched_pairs = []
    for p in all_top_pairs:
        a_name = p.get("a", "")
        b_name = p.get("b", "")
        if a_name in canonical_names or b_name in canonical_names:
            # Normalise so the target asset is always 'a'
            if b_name in canonical_names and a_name not in canonical_names:
                matched_pairs.append({"a": b_name, "b": a_name, "value": p["value"]})
            else:
                matched_pairs.append(p)
    matched_pairs.sort(key=lambda x: abs(x.get("value", 0)), reverse=True)

    # Self gamma from per_asset row_gamma (sum of all pairs involving this asset)
    self_entry = None
    for a in per_asset_list:
        if a.get("ticker", "") in canonical_names:
            self_entry = a
            break
    row_gamma  = self_entry.get("row_gamma",  None) if self_entry else None
    own_gamma  = self_entry.get("own_gamma",  None) if self_entry else None
    asset_class = self_entry.get("asset_class", "?") if self_entry else "?"

    display_name = canonical_names[0]
    lines = [
        f"Cross Gamma Exposure — {display_name} ({asset_code}) | {asset_class} | {base['date']}",
        f"  Row gamma (all cross pairs): {row_gamma:>+16,.0f} CNY/1%" if row_gamma is not None else "  Row gamma: N/A",
        f"  Own gamma (diagonal):        {own_gamma:>+16,.0f} CNY/1%" if own_gamma is not None else "  Own gamma: N/A",
        f"",
        f"  Top cross-gamma pairs (EDS scaled, CNY/1%):",
    ]
    for p in matched_pairs[:15]:
        a_n = p.get("a", "?")
        b_n = p.get("b", "?")
        val = p.get("value", 0)
        lines.append(f"    {a_n:<20} × {b_n:<20}  {val:>+16,.0f}")

    return {
        "ok":              True,
        "date":            base["date"],
        "asset":           display_name,
        "asset_class":     asset_class,
        "row_gamma_cny":   row_gamma,
        "own_gamma_cny":   own_gamma,
        "pairs":           matched_pairs[:20],
        "summary":         "\n".join(lines),
    }


def top_pairs(n: int = 30, date: Optional[str] = None) -> Dict[str, Any]:
    """Return the top N cross-gamma pairs (by absolute value) across the whole book."""
    base = load_latest(date)
    if not base["ok"]:
        return base

    pairs = base.get("top_pairs", [])[:n]
    lines = [f"Top {len(pairs)} Cross Gamma Pairs (date: {base['date']})  [CNY/1%]"]
    for i, p in enumerate(pairs, 1):
        a   = p.get("a", "?")
        b   = p.get("b", "?")
        val = p.get("value", 0)
        lines.append(f"  {i:>2}. {a:<20} × {b:<20}  {val:>+16,.0f}")

    return {
        "ok":    True,
        "date":  base["date"],
        "pairs": pairs,
        "summary": "\n".join(lines),
    }


def book_summary(date: Optional[str] = None) -> Dict[str, Any]:
    """High-level book summary: stats + per-asset row gammas."""
    base = load_latest(date)
    if not base["ok"]:
        return base

    stats     = base.get("stats", {})
    per_asset = base.get("per_asset", [])

    lines = [
        f"QIS Book Cross Gamma Summary — {base['date']}",
        f"  Underlyings  : {stats.get('n_underlyings', 'N/A')}",
        f"  Active trades: {stats.get('active_trades', 'N/A')} / {stats.get('total_trades', 'N/A')}",
        f"  Total gamma  : {stats.get('total_gamma', 0):>+16,.0f} CNY/1%",
        f"  Total|gamma| : {stats.get('total_abs_gamma', 0):>16,.0f} CNY/1%",
        "",
        "  Per-Asset Row Gamma (top 20 by |row_gamma|)  [CNY/1%]:",
    ]
    sorted_assets = sorted(per_asset, key=lambda x: abs(x.get("row_gamma", 0)), reverse=True)
    for a in sorted_assets[:20]:
        ticker  = a.get("ticker", "?")
        ac      = a.get("asset_class", "")
        row_g   = a.get("row_gamma", 0)
        own_g   = a.get("own_gamma", 0)
        lines.append(f"    {ticker:<20} {ac:<12}  row={row_g:>+16,.0f}  own={own_g:>+14,.0f}")

    return {
        "ok":        True,
        "date":      base["date"],
        "stats":     stats,
        "per_asset": sorted_assets,
        "summary":   "\n".join(lines),
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross Gamma analysis helper")
    parser.add_argument("--date",  help="Date YYYY-MM-DD (default: latest)")
    parser.add_argument("--asset", help="Filter by asset code, e.g. AU, CU, IF")
    parser.add_argument("--top",   type=int, default=30, help="Show top N pairs")
    parser.add_argument("--mode",  default="summary",
                        choices=["summary", "top_pairs", "asset", "list_files"])
    args = parser.parse_args()

    if args.mode == "list_files":
        files = find_all_files()
        print(f"Found {len(files)} cross gamma files:")
        for d, p in files[:20]:
            print(f"  {d}  {p}")
        sys.exit(0)

    if args.mode == "asset" or args.asset:
        if not args.asset:
            print("Error: --asset required in asset mode"); sys.exit(1)
        r = asset_exposure(args.asset, args.date)
        print(r.get("summary", ""))
        if not r["ok"]:
            print("ERROR:", r.get("error"))
    elif args.mode == "top_pairs":
        r = top_pairs(args.top, args.date)
        print(r.get("summary", ""))
    else:  # summary
        r = book_summary(args.date)
        print(r.get("summary", ""))
        print()
        # Also show top pairs
        tp = top_pairs(args.top, args.date)
        print(tp.get("summary", ""))
