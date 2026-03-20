"""Aggregate per-trade cross gamma into book-level analytics.

Input  : raw data dict  {trade_id: {"cross_gamma[A,B]": value, ...}, ...}
Output : structured result with book-level matrix, top pairs, trade contributions, etc.
"""

from __future__ import annotations

import re
from collections import defaultdict

from .ticker_map import normalize, normalize_pair, get_asset_class, sort_key

# Regex to parse keys like  cross_gamma[TICKER_A,TICKER_B]
_CG_KEY_RE = re.compile(r"^cross_gamma\[(.+?),(.+?)\]$")

# EDS reports cross gamma in units of ∂²V/(∂r_A·∂r_B) where r = ΔS/S (return).
# These values are 100× larger than the 1% cash cross gamma that the dashboard
# uses (%Gamma($)).  Multiplying by 0.01 converts to 1% cash gamma in CNY.
_EDS_SCALE = 0.01

# Threshold: trades with total |cross gamma| < this in CNY are considered inactive
_NOISE_THRESHOLD = 1.0  # 1 CNY


def aggregate_book(raw_data: dict) -> dict:
    """Aggregate all trades into a book-level cross gamma result.

    Parameters
    ----------
    raw_data : dict
        {trade_id: {"cross_gamma[A,B]": value, ...}, ...}

    Returns
    -------
    dict with keys:
        tickers        : list[str]       – sorted canonical names
        asset_classes  : list[str]       – per ticker
        matrix         : list[list[float]]  – N×N symmetric cash gamma (CNY)
        top_pairs      : list[dict]      – top 30 pairs by |value|
        trade_contribs : list[dict]      – per-trade contribution summary
        per_asset      : list[dict]      – per-ticker total exposure
        stats          : dict            – summary statistics
    """

    # ------------------------------------------------------------------
    # Pass 1 : collect all normalised pairs and per-trade contributions
    # ------------------------------------------------------------------
    # pair_key (canonical_a, canonical_b) → total value across all trades
    book_pairs: dict[tuple[str, str], float] = defaultdict(float)
    # per-trade summary
    trade_summaries: list[dict] = []
    total_trades = 0
    active_trades = 0
    skipped_trades = 0

    for trade_id, entries in raw_data.items():
        total_trades += 1
        if not entries:
            skipped_trades += 1
            continue

        # ── Per-trade pair accumulation ───────────────────────────
        # 1. Average: near/far month contracts on the same underlying
        #    (e.g. DMH6+DMM6 → DAX) produce duplicate entries with
        #    identical values.  We average them.
        # 2. Scale: multiply by _EDS_SCALE to convert EDS raw units
        #    to 1% cash cross gamma in CNY.
        _pair_sum:   dict[tuple[str, str], float] = defaultdict(float)
        _pair_count: dict[tuple[str, str], int]   = defaultdict(int)

        for key, value in entries.items():
            m = _CG_KEY_RE.match(key)
            if not m:
                continue
            raw_a, raw_b = m.group(1).strip(), m.group(2).strip()
            canon_a, canon_b = normalize_pair(raw_a, raw_b)
            _pair_sum[(canon_a, canon_b)] += value
            _pair_count[(canon_a, canon_b)] += 1

        # De-duplicate: take the average for each canonical pair, then
        # apply the EDS → 1%-cash-gamma unit conversion.
        trade_pairs: dict[tuple[str, str], float] = {
            k: _pair_sum[k] / _pair_count[k] * _EDS_SCALE for k in _pair_sum
        }

        # Compute trade-level absolute gamma from the scaled pairs
        trade_abs = sum(abs(v) for v in trade_pairs.values())

        if trade_abs < _NOISE_THRESHOLD:
            skipped_trades += 1
            continue

        active_trades += 1

        # Find this trade's top pair
        top_pair = max(trade_pairs.items(), key=lambda x: abs(x[1]), default=None)
        top_pair_label = ""
        top_pair_value = 0.0
        if top_pair:
            top_pair_label = f"{top_pair[0][0]} × {top_pair[0][1]}"
            top_pair_value = top_pair[1]

        trade_summaries.append({
            "trade_id":       trade_id,
            "total_abs":      trade_abs,
            "top_pair":       top_pair_label,
            "top_pair_value": top_pair_value,
            "n_pairs":        len(trade_pairs),
        })

        # Accumulate into book-level
        for pair_key, val in trade_pairs.items():
            book_pairs[pair_key] += val

    # Sort trade contributions by absolute value descending
    trade_summaries.sort(key=lambda x: -x["total_abs"])

    # ------------------------------------------------------------------
    # Pass 2 : build matrix and analytics
    # ------------------------------------------------------------------
    # Collect all canonical tickers from the book-level pairs
    tickers_set: set[str] = set()
    for (a, b) in book_pairs:
        tickers_set.add(a)
        tickers_set.add(b)

    tickers = sorted(tickers_set, key=sort_key)
    n = len(tickers)
    idx_map = {t: i for i, t in enumerate(tickers)}
    asset_classes = [get_asset_class(t) for t in tickers]

    # Build N×N matrix
    matrix = [[0.0] * n for _ in range(n)]
    for (a, b), val in book_pairs.items():
        i, j = idx_map[a], idx_map[b]
        if i == j:
            # Diagonal: cross gamma between near/far month of same underlying
            matrix[i][j] += val
        else:
            matrix[i][j] += val
            matrix[j][i] += val  # symmetric

    # Top pairs (off-diagonal only, sorted by absolute value)
    off_diag_pairs = []
    for (a, b), val in book_pairs.items():
        if a == b:
            continue
        off_diag_pairs.append({"a": a, "b": b, "value": val})
    off_diag_pairs.sort(key=lambda x: -abs(x["value"]))

    # Per-asset total exposure (sum of row, excluding diagonal)
    per_asset = []
    for i, t in enumerate(tickers):
        row_sum = sum(matrix[i][j] for j in range(n) if j != i)
        diag_val = matrix[i][i]
        per_asset.append({
            "ticker":      t,
            "asset_class": asset_classes[i],
            "row_gamma":   row_sum,      # total cross gamma exposure
            "own_gamma":   diag_val,     # diagonal (same-underlying near/far)
        })
    per_asset.sort(key=lambda x: -abs(x["row_gamma"]))

    # Summary statistics
    total_book_gamma = sum(v for v in book_pairs.values())
    total_abs_gamma = sum(abs(v) for v in book_pairs.values())

    # Compute percentage contribution per trade
    for ts in trade_summaries:
        ts["pct_of_book"] = (ts["total_abs"] / total_abs_gamma * 100) if total_abs_gamma > 0 else 0.0

    return {
        "tickers":         tickers,
        "asset_classes":   asset_classes,
        "matrix":          matrix,
        "top_pairs":       off_diag_pairs[:30],
        "trade_contribs":  trade_summaries[:50],
        "per_asset":       per_asset,
        "stats": {
            "total_trades":    total_trades,
            "active_trades":   active_trades,
            "skipped_trades":  skipped_trades,
            "n_underlyings":   n,
            "total_gamma":     total_book_gamma,
            "total_abs_gamma": total_abs_gamma,
        },
    }
