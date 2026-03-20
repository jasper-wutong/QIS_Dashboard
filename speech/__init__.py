"""Speech module — Morning meeting briefing generator.

Aggregates data from multiple sources (news, X/Nitter, Telegram, Substack,
Polymarket, CICC research, Bailian RAG, book risk, cross gamma) and feeds
everything to Copilot SDK to generate a Goldman-trader-style morning briefing.

Exports:
    generate_morning_speech  — trigger a fresh generation
    get_cached_speech        — return today's cached speech (if any)
"""

from .generator import generate_morning_speech, get_cached_speech

__all__ = ["generate_morning_speech", "get_cached_speech"]
