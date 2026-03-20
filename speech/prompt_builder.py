"""Prompt builder — assembles the mega-prompt for the Copilot SDK model.

Takes collected data (from data_collector) and book analysis (from
book_analyzer) and constructs a structured prompt that instructs the model
to produce a Goldman-trader-style morning briefing in two parts.
"""

import json
from datetime import datetime
from typing import Any, Dict, List


def _fmt_wan(val: float) -> str:
    """Format a CNY value in 万 or 亿."""
    if val is None:
        return "N/A"
    wan = val / 10000
    if abs(wan) >= 10000:
        return f"{wan / 10000:.2f}亿"
    return f"{wan:,.0f}万"


def _truncate(text: str, max_len: int = 500) -> str:
    return text[:max_len] + "…" if len(text) > max_len else text


# ── Section builders ─────────────────────────────────────────────────────────

def _build_news_section(collected: Dict) -> str:
    """Format financial news headlines."""
    news = collected.get("finance_news", {}).get("data", [])
    if not news:
        return "(无中文财经新闻数据)"

    # Flatten if nested by source
    items = []
    if isinstance(news, dict):
        for source_name, source_items in news.items():
            if isinstance(source_items, list):
                for item in source_items[:5]:
                    title = item.get("title", "")
                    source = item.get("source", source_name)
                    items.append(f"- [{source}] {title}")
    elif isinstance(news, list):
        for item in news[:30]:
            title = item.get("title", "")
            source = item.get("source", "")
            items.append(f"- [{source}] {title}")

    return "\n".join(items[:30]) if items else "(无数据)"


def _build_x_twitter_section(collected: Dict) -> str:
    """Format X/Twitter feed data."""
    tweets = collected.get("nitter_x", {}).get("data", [])
    if not tweets:
        return "(无X/Twitter数据 — Nitter不可达)"
    lines = []
    for t in tweets[:25]:
        source = t.get("source", "X")
        content = t.get("content", t.get("title", ""))[:200]
        lines.append(f"- [{source}] {content}")
    return "\n".join(lines)


def _build_telegram_section(collected: Dict) -> str:
    """Format Telegram channel data."""
    messages = collected.get("telegram", {}).get("data", [])
    if not messages:
        return "(无Telegram数据)"
    lines = []
    for m in messages[:15]:
        source = m.get("source", "Telegram")
        content = m.get("content", "")[:200]
        lines.append(f"- [{source}] {content}")
    return "\n".join(lines)


def _build_substack_section(collected: Dict) -> str:
    """Format Substack newsletter data."""
    posts = collected.get("substack", {}).get("data", [])
    if not posts:
        return "(无Substack数据)"
    lines = []
    for p in posts[:10]:
        source = p.get("source", "Substack")
        title = p.get("title", "")[:100]
        content = p.get("content", "")[:300]
        lines.append(f"- [{source}] **{title}**\n  {content}")
    return "\n".join(lines)


def _build_polymarket_section(collected: Dict) -> str:
    """Format Polymarket prediction market data."""
    markets = collected.get("polymarket", {}).get("data", [])
    if not markets:
        return "(无Polymarket数据)"
    lines = []
    for m in markets[:15]:
        q = m.get("question", "")[:100]
        prob = m.get("yes_probability", "N/A")
        lines.append(f"- {q} → YES: {prob}")
    return "\n".join(lines)


def _build_web_search_section(collected: Dict) -> str:
    """Format web search results."""
    results = collected.get("web_search", {}).get("data", [])
    if not results:
        return "(无网络搜索结果)"
    lines = []
    for r in results[:15]:
        title = r.get("title", "")[:100]
        snippet = r.get("snippet", "")[:200]
        lines.append(f"- {title}: {snippet}")
    return "\n".join(lines)


def _build_cicc_research_section(collected: Dict) -> str:
    """Format CICC research PDF summaries."""
    sections = []

    # Morning focus
    focus = collected.get("cicc_morning_focus", {}).get("data", [])
    if focus:
        sections.append("### 中金晨会焦点")
        for pdf in focus[:2]:
            sections.append(f"**{pdf['name']}**:")
            sections.append(_truncate(pdf["text"], 2000))

    # Commodity
    commodity = collected.get("cicc_commodity", {}).get("data", [])
    if commodity:
        sections.append("### 中金大宗商品研究")
        for pdf in commodity[:2]:
            sections.append(f"**{pdf['name']}**:")
            sections.append(_truncate(pdf["text"], 1500))

    # Macro
    macro = collected.get("cicc_macro", {}).get("data", [])
    if macro:
        sections.append("### 中金宏观经济研究")
        for pdf in macro[:1]:
            sections.append(f"**{pdf['name']}**:")
            sections.append(_truncate(pdf["text"], 1500))

    # Strategy
    strategy = collected.get("cicc_strategy", {}).get("data", [])
    if strategy:
        sections.append("### 中金市场策略研究")
        for pdf in strategy[:1]:
            sections.append(f"**{pdf['name']}**:")
            sections.append(_truncate(pdf["text"], 1500))

    return "\n\n".join(sections) if sections else "(无CICC研究报告数据)"


def _build_bailian_section(collected: Dict) -> str:
    """Format Bailian RAG response."""
    text = collected.get("bailian_rag", {}).get("data", "")
    if not text:
        return "(无百炼RAG数据)"
    return _truncate(text, 2000)


def _build_book_section(book_analysis: Dict) -> str:
    """Format book positioning data for the prompt."""
    parts = []

    # Overall bias
    bias = book_analysis.get("book_bias", {})
    if bias:
        parts.append(f"**整体持仓偏向**: {bias.get('description', 'N/A')}")
        parts.append(f"  Net Delta: {_fmt_wan(bias.get('net_delta', 0))}")
        parts.append(f"  Net Gamma: {_fmt_wan(bias.get('net_gamma', 0))}")
        parts.append(f"  Net Vega: {_fmt_wan(bias.get('net_vega', 0))}")
        parts.append(f"  Net Theta: {_fmt_wan(bias.get('net_theta', 0))}")

    # Sector exposure
    sectors = book_analysis.get("sector_exposure", [])
    if sectors:
        parts.append("\n**板块敞口 (按绝对值排序)**:")
        for s in sectors[:10]:
            parts.append(
                f"  - {s['sector']}: 敞口 {_fmt_wan(s['exposure'])}, "
                f"Delta {_fmt_wan(s['delta'])}, Gamma {_fmt_wan(s['gamma'])}, "
                f"Vega {_fmt_wan(s['vega'])}, PnL {_fmt_wan(s['pnl'])}"
            )

    # Strike concentration
    strikes = book_analysis.get("strike_concentration", {})
    if strikes:
        parts.append("\n**各标的物 Strike/Barrier 集中情况**:")
        for und, info in list(strikes.items())[:8]:
            spot = info.get("spot", 0)
            parts.append(
                f"  {und}: Spot={spot:.2f}, "
                f"trades={info.get('trade_count', 0)}, "
                f"notional={_fmt_wan(info.get('total_notional', 0))}, "
                f"net_delta={_fmt_wan(info.get('net_delta', 0))}, "
                f"net_gamma={_fmt_wan(info.get('net_gamma', 0))}"
            )
            conc = info.get("strike_concentration", {})
            if conc:
                conc_str = ", ".join(f"{k}: {int(v)} trades" for k, v in list(conc.items())[:6])
                parts.append(f"    Strike 集中: {conc_str}")
            ko = info.get("ko_levels", [])
            if ko:
                parts.append(f"    KO barriers: {ko}")
            ki = info.get("ki_levels", [])
            if ki:
                parts.append(f"    KI barriers: {ki}")
            if info.get("near_expiry_count", 0) > 0:
                parts.append(
                    f"    近2周到期: {info['near_expiry_count']} trades, "
                    f"notional {_fmt_wan(info['near_expiry_notional'])}"
                )
            structs = info.get("structures", {})
            if structs:
                parts.append(f"    结构: {dict(structs)}")

    # Cross gamma
    cg = book_analysis.get("cross_gamma", {})
    if cg.get("available"):
        parts.append(f"\n**Cross Gamma 风险**:")
        parts.append(f"  Total Cash Gamma: {_fmt_wan(cg.get('total_gamma', 0))} CNY")
        parts.append(f"  Total |Gamma|: {_fmt_wan(cg.get('total_abs_gamma', 0))} CNY")
        parts.append(f"  Active trades: {cg.get('active_trades', 0)}, Underlyings: {cg.get('n_underlyings', 0)}")
        top_pairs = cg.get("top_pairs", [])
        if top_pairs:
            parts.append("  Top gamma pairs:")
            for p in top_pairs[:8]:
                a = p.get("asset_a") or p.get("ticker_a", "?")
                b = p.get("asset_b") or p.get("ticker_b", "?")
                val = p.get("value") or p.get("gamma", 0)
                parts.append(f"    {a} × {b}: {_fmt_wan(val)} CNY")
        per_asset = cg.get("per_asset", [])
        if per_asset:
            parts.append("  Per-asset gamma exposure:")
            for a in per_asset[:8]:
                name = a.get("ticker") or a.get("name", "?")
                total = a.get("total") or a.get("gamma", 0)
                parts.append(f"    {name}: {_fmt_wan(total)} CNY")

    return "\n".join(parts) if parts else "(无Book数据)"


# ── Main prompt assembly ─────────────────────────────────────────────────────

SECTION_DELIMITER = "\n\n===SECTION_BREAK===\n\n"


def build_speech_prompt(collected: Dict, book_analysis: Dict) -> str:
    """Build the mega-prompt for Copilot SDK to generate morning briefing.

    The prompt instructs the model to produce TWO parts separated by
    ===SECTION_BREAK=== :
    1. 早会发言 — Morning meeting speech
    2. 期权结构推荐 — Options structure recommendations
    """
    today = datetime.now().strftime("%Y年%m月%d日 %A")

    news_section = _build_news_section(collected)
    x_section = _build_x_twitter_section(collected)
    telegram_section = _build_telegram_section(collected)
    substack_section = _build_substack_section(collected)
    polymarket_section = _build_polymarket_section(collected)
    web_section = _build_web_search_section(collected)
    cicc_section = _build_cicc_research_section(collected)
    bailian_section = _build_bailian_section(collected)
    book_section = _build_book_section(book_analysis)

    prompt = f"""你是一名顶级投行（Goldman Sachs / Morgan Stanley 级别）QIS衍生品交易台的 Head Trader。
你每天早上在晨会上向整个衍生品团队和销售团队做发言。你的风格是：
- **直接、果断、有观点** — 不是读新闻，而是给出 tradeable insights
- **数据驱动** — 引用具体价格、vol水平、gamma暴露、预测市场概率
- **分析链条清晰** — 从macro event → market impact → our book exposure → actionable recommendation
- **用中文为主，关键金融术语用英文**（gamma, delta, vega, vol surface, risk reversal, skew, term structure, butterfly, condor, collar, straddle, strangle, spread, barrier, knock-out, knock-in, snowball, autocall, sharkfin, accumulator, decumulator, digital等）
- **像交易员说话**，不像分析师写报告 — 简洁有力，有edge，有conviction

日期: {today}

═══════════════════════════════════════════════════════════════
数据源 A: 中国财经新闻 (同花顺/财新/东方财富/金十/华尔街见闻等)
═══════════════════════════════════════════════════════════════
{news_section}

═══════════════════════════════════════════════════════════════
数据源 B: X/Twitter 实时信息 (来自关键金融账号)
═══════════════════════════════════════════════════════════════
{x_section}

═══════════════════════════════════════════════════════════════
数据源 C: Telegram 频道
═══════════════════════════════════════════════════════════════
{telegram_section}

═══════════════════════════════════════════════════════════════
数据源 D: Substack 深度分析
═══════════════════════════════════════════════════════════════
{substack_section}

═══════════════════════════════════════════════════════════════
数据源 E: Polymarket 预测市场
═══════════════════════════════════════════════════════════════
{polymarket_section}

═══════════════════════════════════════════════════════════════
数据源 F: 网络搜索（隔夜市场动态）
═══════════════════════════════════════════════════════════════
{web_section}

═══════════════════════════════════════════════════════════════
数据源 G: 中金研究报告（晨会焦点 + 宏观 + 策略 + 大宗商品研究）
═══════════════════════════════════════════════════════════════
{cicc_section}

═══════════════════════════════════════════════════════════════
数据源 H: 百炼RAG知识库分析
═══════════════════════════════════════════════════════════════
{bailian_section}

═══════════════════════════════════════════════════════════════
数据源 I: 我们的 BOOK 持仓数据（Greeks、Strikes、Barriers、Cross Gamma）
═══════════════════════════════════════════════════════════════
{book_section}

═══════════════════════════════════════════════════════════════
你的任务
═══════════════════════════════════════════════════════════════

请生成两个部分，用 ===SECTION_BREAK=== 分隔：

## 第一部分: 早会发言 (Morning Briefing)

像 Goldman trader 一样做晨会发言。要求：

1. **开场总结** (2-3句): 隔夜市场最重要的3件事，不要废话。
2. **宏观 & 利率** (1段): Fed/PBOC/ECB动态、利率曲线变化、美元走向。引用Polymarket预测概率（如有）。
3. **大宗商品** (1段): 原油、黄金、铜、铁矿等关键品种的overnight moves和驱动因素。结合中金研究观点。
4. **权益市场** (1段): 中/美/欧股市、波动率、VIX变化。
5. **地缘政治 & 事件** (1段): 任何可能 move the needle 的地缘/政策事件。优先引用X/Twitter上的前沿信息。
6. **我们的book关注点** (1段): 基于上面的book数据，哪些标的物的gamma/delta暴露需要特别关注？Near-term到期的头寸？Cross gamma集中度风险？
7. **今日key levels & triggers** (bullet points): 关键价格水平、今日可能触发 KO/KI barrier 的标的。

风格要求: 
- 必须有 **original insights**，不是简单信息罗列
- 必须有 **analysis chain**: event → impact → book implication
- 引用信息时标注来源（如 "据@DeItaone, ..." 或 "中金晨会焦点指出..."）
- 总长度 800-1200字

===SECTION_BREAK===

## 第二部分: 期权结构推荐 (Options Structure Recommendations)

基于以上所有数据（市场环境 + 研究观点 + 我们的book + risk），给销售团队推荐 3-5 个期权结构。每个推荐包含：

1. **结构名称**: 如 "CSI300 3M Bull Call Spread" 或 "黄金 6M Collar (sell call, buy put)"
2. **市场观点**: 为什么现在推这个结构？什么 market view 驱动？
3. **结构描述**: 具体的 strike levels（参考当前 spot 和 vol surface），tenor，payoff profile
4. **与book的关系**: 这个结构如何与我们当前 book positioning 互补？是增加 gamma exposure 还是 hedge 现有 risk？
5. **风险提示**: downside scenario，max loss，需要关注的 trigger levels
6. **适合客户类型**: 哪类客户会感兴趣（对冲型 / 增强收益型 / 方向型）

期权结构可以包括但不限于：
- Vanilla options (call/put)
- Spreads (bull/bear call/put spread)
- Straddle / Strangle
- Risk reversal
- Collar
- Butterfly / Condor
- Barrier options (knock-in / knock-out)
- Snowball / Autocall
- Sharkfin
- Accumulator / Decumulator
- Asian options
- Digital / Binary

确保推荐是 **具体的、可执行的**，不是泛泛的分析。给出具体 underlying、tenor、approximate strike levels。

总长度 600-1000字"""

    return prompt


def parse_speech_response(content: str) -> Dict[str, str]:
    """Parse the model response into two sections.

    Returns
    -------
    dict with keys: speech_part, recommendation_part
    """
    if not content:
        return {"speech_part": "", "recommendation_part": ""}
    parts = content.split("===SECTION_BREAK===")
    speech_part = parts[0].strip() if len(parts) > 0 else content.strip()
    recommendation_part = parts[1].strip() if len(parts) > 1 else ""

    # Clean up section headers if present
    for prefix in ["## 第一部分:", "## 第一部分：", "# 第一部分", "## Part 1"]:
        if speech_part.startswith(prefix):
            speech_part = speech_part[len(prefix):].strip()
    for prefix in ["## 第二部分:", "## 第二部分：", "# 第二部分", "## Part 2"]:
        if recommendation_part.startswith(prefix):
            recommendation_part = recommendation_part[len(prefix):].strip()

    return {
        "speech_part": speech_part,
        "recommendation_part": recommendation_part,
    }
