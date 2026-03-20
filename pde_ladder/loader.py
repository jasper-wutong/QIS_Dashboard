"""Load daily QIS PDE Scenario Ladder Excel files.

Primary source : network share  \\\\cicc.group\\...\\pde scenario ladder
Fallback       : local pde_ladder/ directory  (for dev / offline)

File naming convention:
    SPOT_LADDER_pde_QIS_YYYY-MM-DD_raw_results.xlsx   ← QIS book (we want this)
    SPOT_LADDER_pde_wud_YYYY-MM-DD_raw_results.xlsx   ← wutong book (ignore)

Each xlsx contains sheets: tv, delta, gamma, vega, rho, rhoQ, theta
Each sheet has columns:
    trade_id | underlying | book_id | scenario_1 .. scenario_15

Scenario mapping (spot bump percentages):
    scenario_1 = +20%     scenario_9  = -2%
    scenario_2 = +15%     scenario_10 = -4%
    scenario_3 = +10%     scenario_11 = -6%
    scenario_4 = +8%      scenario_12 = -8%
    scenario_5 = +6%      scenario_13 = -10%
    scenario_6 = +4%      scenario_14 = -15%
    scenario_7 = +2%      scenario_15 = -20%
    scenario_8 = reference (0%)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
#  Network / local paths
# ---------------------------------------------------------------------------

PDE_NETWORK_DIR = (
    r"\\cicc.group\DFS\Pub\Workgrp\S_EQ Derivatives"
    r"\1-EDS组内共享\个人共享\wangxj\pde scenario ladder"
)

PDE_LOCAL_DIR = Path(__file__).resolve().parent

# Only match QIS files, not wud / other books
_QIS_RE = re.compile(
    r"SPOT_LADDER_pde_QIS_(\d{4}-\d{2}-\d{2})_raw_results\.xlsx$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
#  Scenario labels
# ---------------------------------------------------------------------------

SCENARIO_COLS = [f"scenario_{i}" for i in range(1, 16)]

SCENARIO_LABELS = [
    "+20%", "+15%", "+10%", "+8%", "+6%", "+4%", "+2%",
    "reference",
    "-2%", "-4%", "-6%", "-8%", "-10%", "-15%", "-20%",
]

# Map scenario_N → display label
SCENARIO_MAP = dict(zip(SCENARIO_COLS, SCENARIO_LABELS))

# ---------------------------------------------------------------------------
#  Directory scanning
# ---------------------------------------------------------------------------


def _scan_dir(directory: str | Path) -> list[tuple[str, Path]]:
    """Return (date_str, full_path) pairs for QIS scenario ladder files."""
    results: list[tuple[str, Path]] = []
    try:
        p = Path(directory)
        if not p.exists():
            return results
        for f in p.iterdir():
            m = _QIS_RE.search(f.name)
            if m:
                results.append((m.group(1), f))
    except (OSError, PermissionError):
        pass
    return results


def find_all_files() -> list[tuple[str, Path]]:
    """Return all QIS scenario ladder files sorted by date descending.

    Searches network share first, supplements with local copies.
    De-duplicates by date (network preferred).
    """
    candidates = _scan_dir(PDE_NETWORK_DIR)
    seen_dates = {d for d, _ in candidates}

    for d, f in _scan_dir(PDE_LOCAL_DIR):
        if d not in seen_dates:
            candidates.append((d, f))
            seen_dates.add(d)

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates


def find_latest_qis_file() -> tuple[Path | None, str | None]:
    """Find the most recent QIS PDE scenario ladder file.

    Returns
    -------
    (file_path, date_str) on success
    (None, error_message) on failure
    """
    candidates = find_all_files()
    if not candidates:
        return None, (
            "No SPOT_LADDER_pde_QIS_*_raw_results.xlsx found "
            "(network share unreachable, no local copy)"
        )
    date_str, fpath = candidates[0]
    return fpath, date_str


# ---------------------------------------------------------------------------
#  Load & aggregate
# ---------------------------------------------------------------------------


def load_pde_ladder(filepath: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read PDE scenario ladder xlsx and aggregate per-scenario sums.

    Parameters
    ----------
    filepath : Path to the xlsx file.

    Returns
    -------
    (result_dict, None) on success.
    (None, error_message) on failure.

    result_dict keys
    ----------------
    date        : str            – valuation date from filename
    file_name   : str            – filename only
    trade_count : int            – number of trade rows
    risks       : list[str]      – sheet names  e.g. ["tv","delta",...]
    scenarios   : list[str]      – display labels e.g. ["+20%",...,"-20%"]
    matrix      : list[list[float]]  – rows=scenarios (15), cols=risks
    per_underlying : dict        – {underlying: {risk: [scenario_sums...]}}
    """
    try:
        all_sheets = pd.read_excel(
            filepath, sheet_name=None, engine="openpyxl", header=0,
        )
    except (OSError, PermissionError) as e:
        return None, f"Cannot read {filepath.name}: {e}"
    except Exception as e:
        return None, f"Error parsing {filepath.name}: {e}"

    # Extract date from filename
    m = _QIS_RE.search(filepath.name)
    date_str = m.group(1) if m else "unknown"

    risks: list[str] = []
    # matrix[scenario_idx][risk_idx] = aggregated sum
    matrix: list[list[float]] = [[] for _ in range(len(SCENARIO_COLS))]
    trade_count = 0

    # per-underlying breakdown: {underlying: {risk: [15 scenario sums]}}
    per_underlying: dict[str, dict[str, list[float]]] = {}

    for sheet_name, df in all_sheets.items():
        # Validate expected columns exist
        present_scenarios = [c for c in SCENARIO_COLS if c in df.columns]
        if not present_scenarios:
            continue

        risks.append(str(sheet_name))
        risk_idx = len(risks) - 1

        if trade_count == 0:
            trade_count = len(df)

        # Aggregate: sum each scenario column across all trades (NaN → 0)
        for s_idx, scol in enumerate(SCENARIO_COLS):
            if scol in df.columns:
                total = float(df[scol].fillna(0).sum())
            else:
                total = 0.0
            matrix[s_idx].append(total)

        # Per-underlying aggregation
        if "underlying" in df.columns:
            grouped = df.groupby("underlying")[present_scenarios].sum()
            for und, row in grouped.iterrows():
                und_str = str(und)
                if und_str not in per_underlying:
                    per_underlying[und_str] = {}
                sums = [float(row.get(sc, 0) if sc in row.index else 0) for sc in SCENARIO_COLS]
                per_underlying[und_str][str(sheet_name)] = sums

    if not risks:
        return None, f"No valid sheets with scenario columns in {filepath.name}"

    return {
        "date": date_str,
        "file_name": filepath.name,
        "trade_count": trade_count,
        "risks": risks,
        "scenarios": list(SCENARIO_LABELS),
        "matrix": matrix,
        "per_underlying": per_underlying,
    }, None
