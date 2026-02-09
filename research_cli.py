#!/usr/bin/env python3
"""
Standalone CLI script that calls Copilot SDK to get research for a single ticker.

Usage:
  python3 research_cli.py "黄金期货"
  python3 research_cli.py "白银期货" --price 7800 --change 0.012 --exposure 4000000

Outputs JSON to stdout. Designed to be called via subprocess from the Flask app,
bypassing the SDK startup issues that occur inside Flask's event loop.
"""

import argparse
import asyncio
import html as _html
import importlib
import json
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


# ── SDK bootstrap (same as copilot_sdk_demo.py) ──────────────────────────────

def _install_dateutil_fallback() -> None:
    try:
        importlib.import_module("dateutil.parser")
        return
    except ModuleNotFoundError:
        pass

    import types
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


def _bootstrap_sdk_import() -> None:
    _install_dateutil_fallback()

    try:
        importlib.import_module("copilot")
        return
    except ModuleNotFoundError:
        pass

    local_sdk_python = Path(__file__).resolve().parent / "copilot-sdk" / "python"
    local_pkg = local_sdk_python / "copilot" / "__init__.py"
    if local_pkg.exists():
        sys.path.insert(0, str(local_sdk_python))
        try:
            importlib.import_module("copilot")
            return
        except ModuleNotFoundError:
            pass

    raise ModuleNotFoundError("copilot SDK not found")


_bootstrap_sdk_import()
from copilot import CopilotClient, define_tool  # noqa: E402

PREFERRED_MODELS = ["gpt-5.2", "gpt-5", "gpt-5.1", "gemini-3-pro-preview"]


# ── Web search tool ───────────────────────────────────────────────────────────

def _strip_html_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value)


def _normalize_ddg_url(raw_url: str) -> str:
    url = _html.unescape(raw_url).strip()
    if url.startswith("//"):
        url = f"https:{url}"
    if "duckduckgo.com/l/?" in url:
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        if "uddg" in params and params["uddg"]:
            url = urllib.parse.unquote(params["uddg"][0])
    return url


def _search_public_web(query: str, max_results: int = 5) -> list:
    q = str(query or "").strip()
    if not q:
        return []
    try:
        url = f"https://duckduckgo.com/html/?q={urllib.parse.quote(q)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return []

    results = []
    seen: set = set()
    pattern = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        flags=re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(text):
        u = _normalize_ddg_url(m.group(1))
        if not u.startswith("http") or u in seen:
            continue
        seen.add(u)
        title = _strip_html_tags(_html.unescape(m.group(2))).strip()
        if not title:
            title = u
        results.append({"title": title, "url": u})
        if len(results) >= max_results:
            break
    return results


class WebSearchParams(BaseModel):
    query: str = Field(description="Search query for market news, macro events, inventory data etc.")
    max_results: int = Field(default=5, ge=1, le=8)


@define_tool(description="Search the internet for latest market information, news, inventory data, macro events, and any financial information.")
async def search_web(params: WebSearchParams) -> str:
    results = _search_public_web(params.query, params.max_results)
    return json.dumps({"results": results}, ensure_ascii=False)


def _get_cli_path() -> str:
    cli_path = os.getenv("COPILOT_CLI_PATH") or shutil.which("copilot")
    if not cli_path:
        raise RuntimeError("copilot CLI not found")
    return cli_path


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


def _fmt_exposure(exposure_str: str) -> str:
    """Format exposure value for display in prompt."""
    if not exposure_str or exposure_str == "NA":
        return "未知"
    try:
        val = float(exposure_str)
        wan = val / 10000
        if abs(wan) >= 10000:
            return f"{wan / 10000:.1f}亿元"
        return f"{wan:.0f}万元"
    except (ValueError, TypeError):
        return exposure_str


def _build_batch_prompt(tickers: list) -> str:
    """Build a single prompt that covers multiple tickers at once."""
    today = datetime.now().strftime("%Y-%m-%d")

    ticker_lines = []
    for i, t in enumerate(tickers, 1):
        name = t["name"]
        price = t.get("price", "NA")
        change = t.get("change", "NA")
        exposure = t.get("exposure", "NA")
        parts = [name]
        if price and price != "NA":
            parts.append(f"现价: {price}")
        if change and change != "NA":
            parts.append(f"涨跌幅: {change}")
        exposure_display = _fmt_exposure(exposure)
        if exposure and exposure != "NA":
            try:
                exp_val = float(exposure)
                direction = "多头" if exp_val > 0 else "空头" if exp_val < 0 else "零"
                parts.append(f"敞口: {exposure_display}（{direction}）")
            except (ValueError, TypeError):
                parts.append(f"敞口: {exposure_display}")
        ticker_lines.append(f"{i}. {' | '.join(parts)}")

    ticker_list_str = "\n".join(ticker_lines)
    names_str = "、".join(t["name"] for t in tickers)

    return f"""你是一名资深金融大宗商品与衍生品研究分析师，服务于一家大型投行的QIS（量化投资策略）交易台。
当前日期: {today}

请对以下 {len(tickers)} 个标的物进行批量研究分析：

{ticker_list_str}

请你做以下事情:

1. **联网搜索**：请先用 search_web 搜索 "{names_str} 最新行情 新闻" 等关键词，获取最新市场信息。可以搜索多次。

2. **逐个分析**：对每个标的物，请按以下格式输出分析：

===【标的名称】===

**市场概况**（50-80字）：近期价格走势要点。

**核心驱动因素**（80-120字）：宏观面、供需面、政策面的关键驱动。

**敞口调整建议**：基于分析，结合当前敞口，明确给出敞口是否合理、建议调整方向和目标金额。格式：「当前敞口XXX，建议XXX，敞口建议调整至XXX万元」

重要要求：
- 每个标的物的分析必须以 ===【标的名称】=== 开头（名称与上面列表中的完全一致）
- 必须覆盖所有 {len(tickers)} 个标的物，不要遗漏
- 用中文回答，自然段落格式"""


def _parse_batch_response(content: str, tickers: list) -> list:
    """Parse a batch response into individual ticker results."""
    results = []

    for i, t in enumerate(tickers):
        name = t["name"]
        marker = f"===【{name}】==="
        start = content.find(marker)
        if start == -1:
            results.append({"name": name, "content": "", "ok": False, "error": "未在批量结果中找到该标的分析"})
            continue

        # Find the next marker or end of content
        after_marker = start + len(marker)
        next_start = len(content)
        for other_t in tickers:
            if other_t["name"] == name:
                continue
            other_marker = f"===【{other_t['name']}】==="
            pos = content.find(other_marker, after_marker)
            if pos != -1 and pos < next_start:
                next_start = pos

        section = content[after_marker:next_start].strip()
        results.append({"name": name, "content": section, "ok": True})

    return results


def _build_prompt(name: str, price: str, change: str, exposure: str) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    position_info = ""
    if price and price != "NA":
        position_info += f"\n- 当前价格: {price}"
    if change and change != "NA":
        position_info += f"\n- 涨跌幅: {change}"

    exposure_display = _fmt_exposure(exposure)
    if exposure and exposure != "NA":
        try:
            exp_val = float(exposure)
            direction = "多头" if exp_val > 0 else "空头" if exp_val < 0 else "零"
            position_info += f"\n- 当前风险敞口: {exposure_display}（{direction}方向）"
        except (ValueError, TypeError):
            position_info += f"\n- 当前风险敞口: {exposure_display}"

    return f"""你是一名资深金融大宗商品与衍生品研究分析师，服务于一家大型投行的QIS（量化投资策略）交易台。
当前日期: {today}

请对【{name}】进行全面深度研究分析。你可以使用 search_web 工具联网搜索最新信息。

我当前的持仓信息:{position_info}

请你做以下事情:

1. **联网搜索**：请先用 search_web 搜索 "{name} 最新行情 新闻 库存 供需"，以及 "{name} 宏观 政策 央行" 等关键词，获取最新市场信息。

2. **市场概况**（100-150字）：近期价格走势、成交量与持仓量变化、期限结构（contango/backwardation）。

3. **核心驱动因素**（150-200字）：
   - 宏观面：美联储/央行政策、美元指数、利率、通胀数据
   - 供需面：产量/产能、库存变化（交易所库存、社会库存）、进出口、季节性
   - 政策面：关税、环保限产、战略收储等
   - 地缘与事件：地缘冲突、异常天气、产业链上下游联动

4. **技术面要点**（50-80字）：关键支撑/阻力位、均线系统、MACD/RSI等指标信号。

5. **风险提示**（50-80字）：主要下行/上行风险因素。

6. **敞口调整建议**（这段非常重要，必须单独成段）：
   基于以上分析，结合我目前 {exposure_display} 的敞口，明确给出：
   - 当前敞口是否合理
   - 建议如何调整（增大/减小/平掉/反向）
   - 建议目标敞口金额（万元）
   格式示例：「目前敞口{exposure_display}，建议XXX，敞口建议调整至XXX万元」

请用中文回答，自然段落格式，不需要JSON。每个部分用加粗标题分隔。"""


async def run_research(
    name: str,
    price: str = "NA",
    change: str = "NA",
    exposure: str = "NA",
) -> Dict[str, Any]:
    client = CopilotClient(
        {
            "cli_path": _get_cli_path(),
            "cwd": str(Path.cwd()),
            "log_level": os.getenv("COPILOT_LOG_LEVEL", "info"),
        }
    )
    await client.start()
    try:
        model = await _resolve_model(client)
        config: Dict[str, Any] = {
            "model": model,
            "streaming": False,
            "tools": [search_web],
        }

        provider = _build_provider_from_env()
        if provider:
            config["provider"] = provider

        session = await client.create_session(config)
        try:
            prompt = _build_prompt(name, price, change, exposure)
            response = await session.send_and_wait({"prompt": prompt}, timeout=180)
            content = ""
            if response and getattr(response, "data", None):
                content = getattr(response.data, "content", "") or ""

            return {
                "ok": True,
                "name": name,
                "model": model,
                "content": content.strip(),
            }
        finally:
            await session.destroy()
    except Exception as exc:
        return {
            "ok": False,
            "name": name,
            "error": str(exc),
            "content": "",
        }
    finally:
        await client.stop()


def _log(msg: str) -> None:
    """Log to stderr (captured by app.py for console display)."""
    print(msg, file=sys.stderr, flush=True)


async def run_batch_research(tickers: list) -> Dict[str, Any]:
    """Run research for multiple tickers in a single Copilot SDK call."""
    names = [t["name"] for t in tickers]
    _log(f"[CLI/BATCH] Starting batch research for {len(tickers)} tickers: {', '.join(names)}")
    client = CopilotClient(
        {
            "cli_path": _get_cli_path(),
            "cwd": str(Path.cwd()),
            "log_level": os.getenv("COPILOT_LOG_LEVEL", "info"),
        }
    )
    await client.start()
    _log("[CLI/BATCH] Copilot client started")
    try:
        model = await _resolve_model(client)
        _log(f"[CLI/BATCH] Using model: {model}")
        config: Dict[str, Any] = {
            "model": model,
            "streaming": False,
            "tools": [search_web],
        }

        provider = _build_provider_from_env()
        if provider:
            config["provider"] = provider

        session = await client.create_session(config)
        _log("[CLI/BATCH] Session created, sending prompt...")
        try:
            prompt = _build_batch_prompt(tickers)
            response = await session.send_and_wait({"prompt": prompt}, timeout=300)
            content = ""
            if response and getattr(response, "data", None):
                content = getattr(response.data, "content", "") or ""

            _log(f"[CLI/BATCH] Response received, length={len(content)} chars")
            results = _parse_batch_response(content, tickers)
            ok_count = sum(1 for r in results if r.get("ok"))
            _log(f"[CLI/BATCH] Parsed: {ok_count}/{len(results)} tickers found in response")
            for r in results:
                r["model"] = model

            return {"ok": True, "model": model, "results": results}
        finally:
            await session.destroy()
    except Exception as exc:
        _log(f"[CLI/BATCH] ERROR: {exc}")
        return {
            "ok": False,
            "error": str(exc),
            "results": [{"name": t["name"], "content": "", "ok": False, "error": str(exc)} for t in tickers],
        }
    finally:
        await client.stop()
        _log("[CLI/BATCH] Client stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Copilot SDK research CLI")
    parser.add_argument("name", nargs="?", default=None, help="Ticker/asset name, e.g. 黄金期货")
    parser.add_argument("--price", default="NA", help="Current price")
    parser.add_argument("--change", default="NA", help="Price change ratio")
    parser.add_argument("--exposure", default="NA", help="Current risk exposure in yuan")
    parser.add_argument("--batch-json", default=None, help="JSON array of tickers for batch mode")
    args = parser.parse_args()

    if args.batch_json:
        tickers = json.loads(args.batch_json)
        result = asyncio.run(run_batch_research(tickers))
    elif args.name:
        result = asyncio.run(run_research(args.name, args.price, args.change, args.exposure))
    else:
        result = {"ok": False, "error": "Either 'name' or '--batch-json' is required"}

    # Output JSON to stdout for subprocess consumption
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
