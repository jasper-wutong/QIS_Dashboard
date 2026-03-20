"""Generator — orchestrates data collection, prompt building, and LLM call.

The main entry point for the speech module. Called from Flask routes.
Uses subprocess to invoke speech_cli.py (avoiding async conflicts with Flask).
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .book_analyzer import analyze_book_for_recommendations
from .data_collector import collect_all_data
from .prompt_builder import build_speech_prompt, parse_speech_response
from .config import SPEECH_CACHE_DIR, COPILOT_SDK_TIMEOUT

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / SPEECH_CACHE_DIR

# ── In-memory state ──────────────────────────────────────────────────────────
_generation_lock = threading.Lock()
_generation_status = {
    "in_progress": False,
    "started_at": None,
    "phase": "",
    "error": None,
}


def get_generation_status() -> Dict[str, Any]:
    """Return current generation status."""
    return dict(_generation_status)


def _update_status(phase: str, in_progress: bool = True, error: str = None):
    _generation_status["phase"] = phase
    _generation_status["in_progress"] = in_progress
    _generation_status["error"] = error
    if in_progress and _generation_status["started_at"] is None:
        _generation_status["started_at"] = datetime.now().isoformat()


# ── Cache ────────────────────────────────────────────────────────────────────

def _cache_path(date_str: str = None) -> Path:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"speech_{date_str}.json"


def get_cached_speech(date_str: str = None) -> Optional[Dict[str, Any]]:
    """Return today's cached speech if it exists."""
    path = _cache_path(date_str)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _save_cache(result: Dict[str, Any], date_str: str = None):
    path = _cache_path(date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# ── Copilot SDK call via subprocess ──────────────────────────────────────────

def _call_copilot_sdk(prompt: str) -> Dict[str, Any]:
    """Call speech_cli.py via subprocess (same pattern as research_cli.py)."""
    # Write prompt to temp file (may be very long)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", encoding="utf-8", delete=False
    ) as f:
        f.write(prompt)
        prompt_file = f.name

    try:
        cli_script = str(PROJECT_ROOT / "speech" / "speech_cli.py")
        cmd = [
            sys.executable, cli_script,
            "--prompt-file", prompt_file,
            "--timeout", str(COPILOT_SDK_TIMEOUT),
        ]
        print(f"[SPEECH] Running: {' '.join(cmd[:3])} --prompt-file <file> --timeout {COPILOT_SDK_TIMEOUT}")

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=COPILOT_SDK_TIMEOUT + 30,  # buffer
            cwd=str(PROJECT_ROOT),
            encoding="utf-8",
        )

        # Log stderr (SDK debug output)
        if proc.stderr:
            for line in (proc.stderr or "").strip().split("\n"):
                print(f"[SPEECH_CLI] {line}")

        # Parse stdout as JSON
        stdout = (proc.stdout or "").strip()
        if not stdout:
            return {"ok": False, "error": "speech_cli returned empty output", "content": ""}

        # Find the JSON in stdout (skip any non-JSON lines)
        for line in reversed(stdout.split("\n")):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

        return {"ok": False, "error": f"Could not parse speech_cli output", "content": stdout[:500]}

    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Copilot SDK timed out", "content": ""}
    except Exception as e:
        return {"ok": False, "error": str(e), "content": ""}
    finally:
        try:
            os.unlink(prompt_file)
        except OSError:
            pass


# ── Main generation function ─────────────────────────────────────────────────

def _do_generate(app_data: Dict, contracts_fetcher, cross_gamma_fetcher):
    """Background worker — runs the full generation pipeline."""
    try:
        _update_status("collecting data")

        # Phase 1: Collect all data
        print("[SPEECH] Phase 1: Collecting data from all sources...")
        collected = collect_all_data(
            app_data=app_data,
            contracts_fetcher=contracts_fetcher,
            cross_gamma_fetcher=cross_gamma_fetcher,
        )

        # Phase 2: Analyze book
        _update_status("analyzing book")
        print("[SPEECH] Phase 2: Analyzing book positioning...")
        book_data = collected.get("book", {}).get("data", {})
        book_analysis = analyze_book_for_recommendations(book_data)

        # Phase 3: Build prompt
        _update_status("building prompt")
        print("[SPEECH] Phase 3: Building prompt...")
        prompt = build_speech_prompt(collected, book_analysis)
        print(f"[SPEECH] Prompt length: {len(prompt)} chars")

        # Phase 4: Call Copilot SDK (single call, all data in prompt)
        _update_status("generating with AI (claude-sonnet-4.6)")
        print("[SPEECH] Phase 4: Calling Copilot SDK...")
        sdk_result = _call_copilot_sdk(prompt)

        if not sdk_result.get("ok"):
            error = sdk_result.get("error", "Unknown error")
            _update_status("failed", in_progress=False, error=error)
            print(f"[SPEECH] SDK call failed: {error}")
            return

        # Phase 5: Parse response
        _update_status("parsing response")
        content = sdk_result.get("content") or ""
        parsed = parse_speech_response(content)

        # Build sources summary
        meta = collected.get("_meta", {})
        sources_used = []
        for key in ["finance_news", "nitter_x", "telegram", "substack", "polymarket",
                     "web_search", "cicc_morning_focus", "cicc_commodity", "cicc_macro",
                     "cicc_strategy", "bailian_rag", "book"]:
            info = collected.get(key, {})
            sources_used.append({
                "name": key,
                "ok": info.get("ok", False),
                "count": len(info.get("data", [])) if isinstance(info.get("data"), list) else (1 if info.get("data") else 0),
            })

        result = {
            "ok": True,
            "speech_part": parsed["speech_part"],
            "recommendation_part": parsed["recommendation_part"],
            "model": sdk_result.get("model", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "sources_used": sources_used,
            "generation_time": meta.get("elapsed_seconds", 0),
        }

        # Cache
        _save_cache(result)
        _update_status("done", in_progress=False)
        print("[SPEECH] Generation complete!")

    except Exception as e:
        traceback.print_exc()
        _update_status("error", in_progress=False, error=str(e))
        print(f"[SPEECH] Generation error: {e}")
    finally:
        _generation_lock.release()
        _generation_status["started_at"] = None


def generate_morning_speech(
    app_data: Dict = None,
    contracts_fetcher=None,
    cross_gamma_fetcher=None,
    force: bool = False,
) -> Dict[str, Any]:
    """Trigger morning meeting briefing generation in a background thread.

    Returns immediately with status. The frontend polls /api/speech/status
    and /api/speech/latest to get the result when done.

    Parameters
    ----------
    app_data : dict
        The global DATA dict from app.py.
    contracts_fetcher : callable
        Function returning (positions, error) — typically _fetch_qis_contracts.
    cross_gamma_fetcher : callable
        Function returning cross gamma aggregated dict.
    force : bool
        If True, regenerate even if cached version exists for today.

    Returns
    -------
    dict with keys: ok, message (immediate response — generation runs in background)
    """
    # Check cache first
    if not force:
        cached = get_cached_speech()
        if cached and cached.get("ok"):
            print("[SPEECH] Returning cached speech for today")
            return cached

    # Prevent concurrent generation
    if not _generation_lock.acquire(blocking=False):
        return {
            "ok": False,
            "error": "Generation already in progress",
            "phase": _generation_status.get("phase", "unknown"),
        }

    # Launch background thread — POST returns immediately
    _update_status("starting")
    t = threading.Thread(
        target=_do_generate,
        args=(app_data, contracts_fetcher, cross_gamma_fetcher),
        daemon=True,
    )
    t.start()
    print("[SPEECH] Background generation thread started")

    return {
        "ok": True,
        "message": "Generation started in background",
        "phase": "starting",
    }
