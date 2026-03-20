"""Book analyzer — extracts strike concentration, gamma exposure, key risks.

Takes raw book data (positions, sector summaries, cross gamma) and distills
it into a structured analysis suitable for the prompt builder.
"""

from collections import defaultdict
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional


def _safe(v, default=0.0):
    """Safely coerce to float."""
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def analyze_strike_concentration(positions: List[Dict]) -> Dict[str, Any]:
    """Identify where strikes / barriers are clustered by underlying.

    Returns per-underlying summary of key price levels.
    """
    if not positions:
        return {}

    by_underlying: Dict[str, list] = defaultdict(list)
    today = date.today()

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

        underlying = t.get("underlying") or "Unknown"
        spot = _safe(t.get("spot"))
        strike_abs = _safe(t.get("strikeAbs"))
        init_price = _safe(t.get("initPrice"))
        notional = abs(_safe(t.get("notional")))
        delta = _safe(t.get("delta"))
        gamma = _safe(t.get("gamma"))
        vega = _safe(t.get("vega"))
        structure = t.get("structure") or ""
        call_put = t.get("callPut") or ""

        # Compute barrier levels
        ko_raw = t.get("ko_prices")
        ko_abs = (ko_raw * init_price) if (isinstance(ko_raw, (int, float)) and ko_raw > 0 and init_price > 0) else None
        ki_raw = t.get("ki_barrier")
        ki_abs = (ki_raw * init_price) if (isinstance(ki_raw, (int, float)) and ki_raw > 0 and init_price > 0) else None

        by_underlying[underlying].append({
            "expiration": exp_raw,
            "structure": structure,
            "callPut": call_put,
            "spot": spot,
            "strike": strike_abs,
            "ko": ko_abs,
            "ki": ki_abs,
            "notional": notional,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
        })

    # Summarize per underlying
    result = {}
    for und, trades in by_underlying.items():
        if not trades:
            continue
        spot = max(t["spot"] for t in trades if t["spot"] > 0) or 0
        total_notional = sum(t["notional"] for t in trades)
        net_delta = sum(t["delta"] for t in trades)
        net_gamma = sum(t["gamma"] for t in trades)
        net_vega = sum(t["vega"] for t in trades)

        # Find strike clusters
        strikes = [t["strike"] for t in trades if t["strike"] and t["strike"] > 0]
        ko_levels = [t["ko"] for t in trades if t["ko"] and t["ko"] > 0]
        ki_levels = [t["ki"] for t in trades if t["ki"] and t["ki"] > 0]

        # Bucket strikes into ranges (±2% bands)
        strike_buckets: Dict[str, float] = defaultdict(float)
        if spot > 0:
            for s in strikes:
                pct = (s / spot - 1) * 100
                bucket = f"{round(pct / 2) * 2:+.0f}%"
                strike_buckets[bucket] += 1

        # Near-term expiries (next 2 weeks)
        cutoff = (today + timedelta(days=14)).isoformat()
        near_expiry_trades = [t for t in trades if t["expiration"] <= cutoff]
        near_expiry_notional = sum(t["notional"] for t in near_expiry_trades)

        # Structures breakdown
        structures: Dict[str, int] = defaultdict(int)
        for t in trades:
            structures[t["structure"] or "unknown"] += 1

        result[und] = {
            "trade_count": len(trades),
            "spot": spot,
            "total_notional": total_notional,
            "net_delta": net_delta,
            "net_gamma": net_gamma,
            "net_vega": net_vega,
            "strike_concentration": dict(sorted(strike_buckets.items())),
            "ko_levels": sorted(set(round(k, 2) for k in ko_levels))[:5],
            "ki_levels": sorted(set(round(k, 2) for k in ki_levels))[:5],
            "near_expiry_notional": near_expiry_notional,
            "near_expiry_count": len(near_expiry_trades),
            "structures": dict(structures),
        }

    return result


def analyze_sector_exposure(sector_summaries: Dict) -> List[Dict[str, Any]]:
    """Summarize net exposure by sector for the speech.

    Returns list of sectors sorted by absolute exposure.
    """
    sectors = []
    for sector, summary in sector_summaries.items():
        if not summary:
            continue
        exposure = _safe(summary.get("exposure") or summary.get("net_exposure"))
        delta = _safe(summary.get("delta") or summary.get("net_delta"))
        gamma = _safe(summary.get("gamma") or summary.get("net_gamma"))
        vega = _safe(summary.get("vega") or summary.get("net_vega"))
        theta = _safe(summary.get("theta") or summary.get("net_theta"))
        pnl = _safe(summary.get("pnl") or summary.get("daily_pnl"))
        count = int(_safe(summary.get("count") or summary.get("position_count")))

        sectors.append({
            "sector": sector,
            "exposure": exposure,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "pnl": pnl,
            "count": count,
        })
    sectors.sort(key=lambda x: abs(x["exposure"]), reverse=True)
    return sectors


def analyze_cross_gamma(cross_gamma_data: Optional[Dict]) -> Dict[str, Any]:
    """Extract key risk signals from cross gamma matrix."""
    if not cross_gamma_data:
        return {"available": False}

    return {
        "available": True,
        "total_gamma": cross_gamma_data.get("total_gamma", 0),
        "total_abs_gamma": cross_gamma_data.get("total_abs_gamma", 0),
        "active_trades": cross_gamma_data.get("active_trades", 0),
        "n_underlyings": cross_gamma_data.get("n_underlyings", 0),
        "top_pairs": cross_gamma_data.get("top_pairs", [])[:10],
        "per_asset": cross_gamma_data.get("per_asset", [])[:10],
    }


def analyze_book_for_recommendations(book_data: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive book analysis for morning speech.

    Returns structured analysis with:
    - strike concentration per underlying
    - sector exposure summary
    - cross gamma risk summary
    - book bias (long/short gamma, vega etc.)
    - upcoming expiries
    """
    result = {
        "strike_concentration": {},
        "sector_exposure": [],
        "cross_gamma": {"available": False},
        "book_bias": {},
        "book_summaries": {},
    }

    # Strike & barrier analysis from raw contracts
    contracts = book_data.get("contracts", [])
    if contracts:
        result["strike_concentration"] = analyze_strike_concentration(contracts)

    # Sector exposure
    sector_summaries = book_data.get("sector_summaries", {})
    if sector_summaries:
        result["sector_exposure"] = analyze_sector_exposure(sector_summaries)

    # Cross gamma
    cg = book_data.get("cross_gamma")
    if cg:
        result["cross_gamma"] = analyze_cross_gamma(cg)

    # Book summaries (total, domestic futures, overseas futures, ETF)
    result["book_summaries"] = book_data.get("book_summaries", {})

    # Overall book bias
    total = result["book_summaries"].get("total", {})
    if total:
        net_delta = _safe(total.get("delta") or total.get("net_delta"))
        net_gamma = _safe(total.get("gamma") or total.get("net_gamma"))
        net_vega = _safe(total.get("vega") or total.get("net_vega"))
        net_theta = _safe(total.get("theta") or total.get("net_theta"))

        bias = []
        if net_gamma > 0:
            bias.append("long gamma")
        elif net_gamma < 0:
            bias.append("short gamma")
        if net_vega > 0:
            bias.append("long vega")
        elif net_vega < 0:
            bias.append("short vega")
        if net_delta > 0:
            bias.append("net long delta")
        elif net_delta < 0:
            bias.append("net short delta")

        result["book_bias"] = {
            "net_delta": net_delta,
            "net_gamma": net_gamma,
            "net_vega": net_vega,
            "net_theta": net_theta,
            "description": ", ".join(bias) if bias else "neutral",
        }

    return result
