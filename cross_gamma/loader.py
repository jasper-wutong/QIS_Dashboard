"""Load daily QIS Cross Gamma JSON files.

Primary source : network share under EDSLib_Source/QIS Cross Gamma
Fallback       : local cross_gamma/ directory (for dev / offline)

File naming convention: cross_gamma_YYYY-MM-DD.json
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# Network share holding daily cross gamma JSONs
CG_NETWORK_DIR = (
    r"\\cicc.group\DFS\Pub\Workgrp\S_EQ Derivatives"
    r"\3-部分共享\2-RB Formula\EDSLib_Realtime\EDSLib_Source\QIS Cross Gamma"
)

# Local fallback (developer copy or manual backup)
CG_LOCAL_DIR = Path(__file__).resolve().parent

# Regex for filenames like "cross_gamma_2026-03-17.json"
_DATE_RE = re.compile(r"cross_gamma[_-](\d{4}-\d{2}-\d{2})\.json$", re.IGNORECASE)


def _scan_dir(directory: str | Path) -> list[tuple[str, Path]]:
    """Return a list of (date_str, full_path) pairs from a directory."""
    results: list[tuple[str, Path]] = []
    try:
        p = Path(directory)
        if not p.exists():
            return results
        for f in p.iterdir():
            m = _DATE_RE.search(f.name)
            if m:
                results.append((m.group(1), f))
    except (OSError, PermissionError):
        pass
    return results


def find_all_files() -> list[tuple[str, Path]]:
    """Return all cross gamma JSON files sorted by date descending.

    Searches the network share first; merges with local directory.
    De-duplicates by date (network preferred over local).

    Returns
    -------
    list of (date_str, file_path), newest first.  Empty list if none found.
    """
    # Network is primary
    candidates = _scan_dir(CG_NETWORK_DIR)
    seen_dates = {d for d, _ in candidates}

    # Supplement with local copies for dates not on the network share
    for d, f in _scan_dir(CG_LOCAL_DIR):
        if d not in seen_dates:
            candidates.append((d, f))
            seen_dates.add(d)

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates


def find_latest_file() -> tuple[Path | None, str | None]:
    """Find the most recent cross gamma JSON file.

    Searches the network share first, falls back to the local directory.

    Returns
    -------
    (file_path, date_str) or (None, error_message)
    """
    candidates = find_all_files()

    if not candidates:
        return None, "No cross gamma JSON files found (network share unreachable, no local copy)"

    date_str, fpath = candidates[0]
    return fpath, date_str


def load_cross_gamma_data(filepath: Path) -> tuple[dict | None, str | None]:
    """Parse a cross gamma JSON file.

    Returns
    -------
    (data_dict, None) on success – data_dict is {trade_id: {pair_key: value, ...}, ...}
    (None, error_message) on failure
    """
    try:
        raw = filepath.read_text(encoding="utf-8")
    except (OSError, PermissionError) as e:
        return None, f"Cannot read {filepath.name}: {e}"

    # Strip possible JS-style comment header (/** ... */)
    raw = re.sub(r"/\*\*.*?\*/", "", raw, flags=re.DOTALL).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"JSON parse error in {filepath.name}: {e}"

    if not isinstance(data, dict):
        return None, f"Unexpected JSON structure in {filepath.name} (expected object)"

    return data, None
