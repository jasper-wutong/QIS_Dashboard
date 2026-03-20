"""
memory_manager.py — Agent persistent memory & methodology evolution
====================================================================
Stores daily analysis records, tracks prediction accuracy vs outcomes,
and evolves the agent's analysis methodology over time.

Directory layout:
    agent_workspace/memory/
    ├── methodology.md           # Evolving analysis methodology (live doc)
    ├── methodology_changelog.md # Version history of methodology changes
    ├── daily/                   # Per-day analysis records
    │   ├── 2026-03-20.json
    │   ├── 2026-03-21.json
    │   └── ...
    └── reviews/                 # Periodic accuracy reviews
        ├── review_2026-03-W12.md
        └── ...

Usage (CLI):
    python agent_workspace/memory_manager.py save --date 2026-03-20 --asset gold --json '{...}'
    python agent_workspace/memory_manager.py recall --date 2026-03-20
    python agent_workspace/memory_manager.py recall --last 7
    python agent_workspace/memory_manager.py review --asset gold --last 14
    python agent_workspace/memory_manager.py methodology

Usage (import):
    from agent_workspace.memory_manager import save_analysis, recall, methodology
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─── Paths ────────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent
_MEMORY_DIR = _HERE / "memory"
_DAILY_DIR  = _MEMORY_DIR / "daily"
_REVIEW_DIR = _MEMORY_DIR / "reviews"
_METHODOLOGY    = _MEMORY_DIR / "methodology.md"
_METHODOLOGY_CL = _MEMORY_DIR / "methodology_changelog.md"


def _ensure_dirs():
    _DAILY_DIR.mkdir(parents=True, exist_ok=True)
    _REVIEW_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. SAVE — persist an analysis record
# ═══════════════════════════════════════════════════════════════════════════════

def save_analysis(
    asset: str,
    analysis: Dict[str, Any],
    judgment: str,
    key_levels: Optional[Dict[str, Any]] = None,
    date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Save one analysis entry for a given asset and date.

    Parameters
    ----------
    asset      : e.g. "gold", "copper", "oil", "book", "macro"
    analysis   : structured data dict (prices, cross gamma, etc.)
    judgment   : the agent's written judgment/conviction (text)
    key_levels : optional dict of support/resistance/targets
    date       : YYYY-MM-DD (default: today)

    Returns
    -------
    {"ok": True, "file": path, "entries": N}
    """
    _ensure_dirs()
    date = date or datetime.now().strftime("%Y-%m-%d")
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    fpath = _DAILY_DIR / f"{date}.json"

    # Load existing entries for the day (multiple assets/analyses per day)
    existing = []
    if fpath.exists():
        try:
            existing = json.loads(fpath.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                existing = [existing]  # upgrade single entry
        except Exception:
            existing = []

    entry = {
        "timestamp":  ts,
        "asset":      asset.lower(),
        "judgment":   judgment,
        "key_levels": key_levels or {},
        "data":       analysis,
    }
    existing.append(entry)

    fpath.write_text(json.dumps(existing, indent=2, ensure_ascii=False, default=str),
                     encoding="utf-8")
    return {"ok": True, "file": str(fpath), "entries": len(existing)}


# ═══════════════════════════════════════════════════════════════════════════════
#  2. RECALL — load past analyses
# ═══════════════════════════════════════════════════════════════════════════════

def recall(
    date: Optional[str] = None,
    asset: Optional[str] = None,
    last_n_days: int = 1,
) -> Dict[str, Any]:
    """
    Recall saved analyses.

    If date is given, return that day's entries.
    Otherwise return the last N days of entries.
    Optionally filter by asset.
    """
    _ensure_dirs()
    files = sorted(_DAILY_DIR.glob("*.json"), reverse=True)

    if date:
        target = _DAILY_DIR / f"{date}.json"
        if not target.exists():
            return {"ok": False, "error": f"No memory for {date}"}
        files = [target]
    else:
        files = files[:last_n_days]

    results = []
    for f in files:
        try:
            entries = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(entries, dict):
                entries = [entries]
            for e in entries:
                if asset and e.get("asset", "").lower() != asset.lower():
                    continue
                results.append(e)
        except Exception:
            pass

    return {
        "ok": True,
        "count": len(results),
        "entries": results,
    }


def recall_judgments(asset: str, last_n_days: int = 30) -> List[Dict]:
    """Return a list of past judgments for an asset, for accuracy review."""
    r = recall(asset=asset, last_n_days=last_n_days)
    return [
        {
            "date":       e.get("timestamp", "?")[:10],
            "asset":      e.get("asset"),
            "judgment":   e.get("judgment"),
            "key_levels": e.get("key_levels"),
        }
        for e in r.get("entries", [])
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  3. REVIEW — compare past judgments against actual outcomes
# ═══════════════════════════════════════════════════════════════════════════════

def save_review(
    review_text: str,
    period: str = "",
    asset: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Save a methodology review (written by the agent after comparing predictions
    against actual market outcomes).

    Parameters
    ----------
    review_text : Markdown text of the review
    period      : e.g. "2026-03-W12" or "2026-03"
    asset       : optional, if review is asset-specific
    """
    _ensure_dirs()
    period = period or datetime.now().strftime("%Y-%m-%d")
    suffix = f"_{asset}" if asset else ""
    fname  = f"review_{period}{suffix}.md"
    fpath  = _REVIEW_DIR / fname
    fpath.write_text(review_text, encoding="utf-8")
    return {"ok": True, "file": str(fpath)}


def list_reviews() -> List[str]:
    """List all saved review files."""
    _ensure_dirs()
    return sorted([f.name for f in _REVIEW_DIR.glob("*.md")], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  4. METHODOLOGY — read/update evolving methodology doc
# ═══════════════════════════════════════════════════════════════════════════════

def read_methodology() -> str:
    """Read the current analysis methodology."""
    if _METHODOLOGY.exists():
        return _METHODOLOGY.read_text(encoding="utf-8")
    return "(methodology.md not found — please initialize it)"


def update_methodology(new_content: str, change_note: str = "") -> Dict[str, Any]:
    """
    Overwrite the methodology file and append a changelog entry.

    Parameters
    ----------
    new_content : full new markdown content for methodology.md
    change_note : brief description of what changed and why
    """
    _ensure_dirs()

    # Append changelog
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cl_entry = f"\n## {ts}\n{change_note}\n"
    if _METHODOLOGY_CL.exists():
        old = _METHODOLOGY_CL.read_text(encoding="utf-8")
    else:
        old = "# Methodology Changelog\n\nEvery update to methodology.md is logged here.\n"
    _METHODOLOGY_CL.write_text(old + cl_entry, encoding="utf-8")

    # Write new methodology
    _METHODOLOGY.write_text(new_content, encoding="utf-8")

    return {"ok": True, "methodology": str(_METHODOLOGY), "changelog": str(_METHODOLOGY_CL)}


# ═══════════════════════════════════════════════════════════════════════════════
#  5. STATUS — quick overview of memory state
# ═══════════════════════════════════════════════════════════════════════════════

def status() -> Dict[str, Any]:
    """Return a summary of what's in the memory store."""
    _ensure_dirs()
    daily_files  = sorted(_DAILY_DIR.glob("*.json"), reverse=True)
    review_files = sorted(_REVIEW_DIR.glob("*.md"), reverse=True)

    daily_dates = [f.stem for f in daily_files[:10]]

    # Count total entries and unique assets
    total_entries = 0
    all_assets = set()
    for f in daily_files:
        try:
            entries = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(entries, dict):
                entries = [entries]
            total_entries += len(entries)
            for e in entries:
                all_assets.add(e.get("asset", "?"))
        except Exception:
            pass

    return {
        "ok":              True,
        "daily_files":     len(daily_files),
        "total_entries":   total_entries,
        "assets_tracked":  sorted(all_assets),
        "recent_dates":    daily_dates,
        "review_files":    len(review_files),
        "recent_reviews":  [f.name for f in review_files[:5]],
        "methodology_exists": _METHODOLOGY.exists(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent Memory Manager")
    sub = parser.add_subparsers(dest="cmd")

    # save
    p_save = sub.add_parser("save", help="Save an analysis entry")
    p_save.add_argument("--date",  help="YYYY-MM-DD (default: today)")
    p_save.add_argument("--asset", required=True, help="Asset name")
    p_save.add_argument("--judgment", default="", help="Agent's judgment text")
    p_save.add_argument("--json",  default="{}", help="JSON data string")

    # recall
    p_recall = sub.add_parser("recall", help="Recall past analyses")
    p_recall.add_argument("--date",  help="Specific date")
    p_recall.add_argument("--asset", help="Filter by asset")
    p_recall.add_argument("--last",  type=int, default=7, help="Last N days")

    # review
    p_review = sub.add_parser("review", help="Show past reviews")

    # methodology
    p_meth = sub.add_parser("methodology", help="Print current methodology")

    # status
    p_stat = sub.add_parser("status", help="Memory status overview")

    args = parser.parse_args()

    if args.cmd == "save":
        data = json.loads(args.json)
        r = save_analysis(args.asset, data, args.judgment, date=args.date)
        print(json.dumps(r, indent=2))

    elif args.cmd == "recall":
        r = recall(date=args.date, asset=args.asset, last_n_days=args.last)
        for e in r.get("entries", []):
            print(f"\n[{e.get('timestamp','')}] {e.get('asset','?').upper()}")
            j = e.get("judgment", "")
            if j:
                print(f"  Judgment: {j[:200]}")
            kl = e.get("key_levels", {})
            if kl:
                print(f"  Key Levels: {json.dumps(kl, ensure_ascii=False)}")
        print(f"\nTotal: {r.get('count', 0)} entries")

    elif args.cmd == "review":
        reviews = list_reviews()
        if reviews:
            for rv in reviews:
                print(f"  {rv}")
        else:
            print("  (no reviews yet)")

    elif args.cmd == "methodology":
        print(read_methodology())

    elif args.cmd == "status":
        s = status()
        print(f"Memory Status:")
        print(f"  Daily files    : {s['daily_files']}")
        print(f"  Total entries  : {s['total_entries']}")
        print(f"  Assets tracked : {', '.join(s['assets_tracked']) or 'none'}")
        print(f"  Recent dates   : {', '.join(s['recent_dates'][:5]) or 'none'}")
        print(f"  Reviews        : {s['review_files']}")
        print(f"  Methodology    : {'✓' if s['methodology_exists'] else '✗'}")

    else:
        parser.print_help()
