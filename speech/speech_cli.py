#!/usr/bin/env python3
"""Standalone CLI for generating morning briefing via Copilot SDK.

Called via subprocess from the Flask app (same pattern as research_cli.py)
to avoid async event loop conflicts.

Usage:
  python -m speech.speech_cli --prompt-file /tmp/speech_prompt.txt
  python -m speech.speech_cli --prompt "Your prompt here"

Outputs JSON to stdout:
  {"ok": true, "content": "...", "model": "gpt-5.2"}
"""

import argparse
import asyncio
import importlib
import json
import os
import shutil
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# ── SDK bootstrap (same logic as research_cli.py) ────────────────────────────

def _install_dateutil_fallback() -> None:
    try:
        importlib.import_module("dateutil.parser")
        return
    except ModuleNotFoundError:
        pass
    parser_module = types.ModuleType("dateutil.parser")
    def _parse(value: str) -> datetime:
        if not isinstance(value, str):
            raise TypeError("date string expected")
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    parser_module.parse = _parse
    dateutil_module = types.ModuleType("dateutil")
    dateutil_module.parser = parser_module
    sys.modules["dateutil"] = dateutil_module
    sys.modules["dateutil.parser"] = parser_module


def _bootstrap_sdk() -> None:
    _install_dateutil_fallback()
    try:
        importlib.import_module("copilot")
        return
    except ModuleNotFoundError:
        pass
    local_sdk = Path(__file__).resolve().parent.parent / "copilot-sdk" / "python"
    local_pkg = local_sdk / "copilot" / "__init__.py"
    if local_pkg.exists():
        sys.path.insert(0, str(local_sdk))
        try:
            importlib.import_module("copilot")
            return
        except ModuleNotFoundError:
            pass
    raise ModuleNotFoundError("copilot SDK not found")


_bootstrap_sdk()
from copilot import CopilotClient  # noqa: E402

PREFERRED_MODELS = ["claude-sonnet-4.6", "claude-sonnet-4", "gpt-5.2", "gpt-5", "gpt-5.1"]


# ── SDK helpers ──────────────────────────────────────────────────────────────
# NOTE: No tools registered for the model — all data is pre-collected and
# included in the prompt. We want exactly ONE LLM call, no tool use loops.

def _get_cli_path() -> str:
    cli_path = os.getenv("COPILOT_CLI_PATH")
    if cli_path:
        return cli_path
    common_paths = [
        Path.home() / "AppData" / "Local" / "GitHub" / "copilot-cli" / "copilot.exe",
        Path.home() / "AppData" / "Roaming" / "GitHub" / "copilot-cli" / "copilot.exe",
    ]
    for p in common_paths:
        if p.exists():
            return str(p)
    cli_path = shutil.which("copilot")
    if cli_path and cli_path.lower().endswith(".exe"):
        return cli_path
    raise RuntimeError("copilot CLI not found")


def _build_provider_from_env() -> Optional[Dict[str, Any]]:
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return {
            "type": "openai",
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "api_key": openai_key,
        }
    return None


async def _resolve_model(client: CopilotClient) -> str:
    requested = os.getenv("COPILOT_MODEL", "gpt-5.2")
    try:
        models = await client.list_models()
    except Exception:
        return requested
    available = [m.get("id") for m in models if m.get("id")]
    if not available:
        return requested
    if requested in available:
        return requested
    for model in PREFERRED_MODELS:
        if model in available:
            return model
    return available[0] if available else requested


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ── Main generation ──────────────────────────────────────────────────────────

async def generate(prompt: str, timeout: int = 360) -> Dict[str, Any]:
    """Send the mega-prompt to Copilot SDK and return the response."""
    _log("[SPEECH_CLI] Starting Copilot client...")
    client = CopilotClient({
        "cli_path": _get_cli_path(),
        "cwd": str(Path(__file__).resolve().parent.parent),
        "log_level": os.getenv("COPILOT_LOG_LEVEL", "info"),
    })
    await client.start()
    _log("[SPEECH_CLI] Client started")

    try:
        model = await _resolve_model(client)
        _log(f"[SPEECH_CLI] Using model: {model}")

        config: Dict[str, Any] = {
            "model": model,
            "streaming": False,
            # No tools — single LLM call with all data pre-collected in prompt
        }
        provider = _build_provider_from_env()
        if provider:
            config["provider"] = provider

        session = await client.create_session(config)
        _log("[SPEECH_CLI] Session created, sending prompt...")

        try:
            response = await session.send_and_wait({"prompt": prompt}, timeout=timeout)
            content = ""
            if response and getattr(response, "data", None):
                content = getattr(response.data, "content", "") or ""

            _log(f"[SPEECH_CLI] Response received, length={len(content)} chars")
            return {
                "ok": True,
                "content": (content or "").strip(),
                "model": model,
            }
        finally:
            await session.destroy()

    except asyncio.CancelledError as exc:
        return {"ok": False, "error": f"Cancelled/timeout: {exc}", "content": ""}
    except Exception as exc:
        _log(f"[SPEECH_CLI] ERROR: {exc}")
        return {"ok": False, "error": str(exc), "content": ""}
    finally:
        await client.stop()
        _log("[SPEECH_CLI] Client stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Morning speech generator via Copilot SDK")
    parser.add_argument("--prompt", default=None, help="The prompt text directly")
    parser.add_argument("--prompt-file", default=None, help="Path to a file containing the prompt")
    parser.add_argument("--timeout", type=int, default=360, help="Timeout in seconds")
    args = parser.parse_args()

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = sys.stdin.read()

    if not prompt.strip():
        print(json.dumps({"ok": False, "error": "No prompt provided"}))
        sys.exit(1)

    result = asyncio.run(generate(prompt, timeout=args.timeout))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
