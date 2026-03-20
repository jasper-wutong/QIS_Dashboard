"""External data feed scrapers — Nitter RSS, Telegram, Substack, Polymarket."""

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup

from .config import (
    NITTER_INSTANCES,
    NITTER_ACCOUNTS,
    TELEGRAM_CHANNELS,
    SUBSTACK_FEEDS,
    POLYMARKET_API_URL,
    POLYMARKET_LIMIT,
    FEED_FETCH_TIMEOUT,
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


# ── Nitter / X RSS ───────────────────────────────────────────────────────────

def _find_working_nitter() -> str | None:
    """Probe Nitter instances; return first that responds."""
    for base in NITTER_INSTANCES:
        try:
            r = requests.get(f"{base}/", timeout=5, headers=_HEADERS)
            if r.status_code < 500:
                return base
        except Exception:
            continue
    return None


def _parse_rss_items(xml_text: str, source: str) -> List[Dict[str, str]]:
    """Parse RSS feed XML into a flat list of items."""
    items: List[Dict[str, str]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items

    # RSS 2.0
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        desc_raw = (item.findtext("description") or "").strip()
        # Strip HTML tags from description
        desc = re.sub(r"<[^>]+>", "", desc_raw).strip()
        if title or desc:
            items.append({
                "title": title or desc[:80],
                "content": desc[:500] if desc else title,
                "url": link,
                "time": pub,
                "source": source,
            })
    # Atom
    for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
        title = (entry.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
        link_el = entry.find("{http://www.w3.org/2005/Atom}link")
        link = link_el.get("href", "") if link_el is not None else ""
        pub = (entry.findtext("{http://www.w3.org/2005/Atom}published")
               or entry.findtext("{http://www.w3.org/2005/Atom}updated") or "").strip()
        content_el = entry.find("{http://www.w3.org/2005/Atom}content")
        desc_raw = content_el.text if content_el is not None and content_el.text else ""
        desc = re.sub(r"<[^>]+>", "", desc_raw).strip()
        if title or desc:
            items.append({
                "title": title or desc[:80],
                "content": desc[:500] if desc else title,
                "url": link,
                "time": pub,
                "source": source,
            })
    return items


def fetch_nitter_rss() -> List[Dict[str, str]]:
    """Fetch latest tweets from configured X accounts via Nitter RSS."""
    base = _find_working_nitter()
    if not base:
        print("[SPEECH] No reachable Nitter instance found")
        return []

    all_tweets: List[Dict[str, str]] = []
    for account in NITTER_ACCOUNTS:
        try:
            url = f"{base}/{account}/rss"
            r = requests.get(url, timeout=FEED_FETCH_TIMEOUT, headers=_HEADERS)
            if r.status_code == 200:
                items = _parse_rss_items(r.text, f"X/@{account}")
                all_tweets.extend(items[:10])  # cap per account
            else:
                print(f"[SPEECH] Nitter {account}: HTTP {r.status_code}")
        except Exception as e:
            print(f"[SPEECH] Nitter {account} error: {e}")
    return all_tweets


# ── Telegram ─────────────────────────────────────────────────────────────────

def fetch_telegram_channels() -> List[Dict[str, str]]:
    """Scrape public Telegram channel previews (no API key needed)."""
    all_messages: List[Dict[str, str]] = []
    for channel in TELEGRAM_CHANNELS:
        try:
            url = f"https://t.me/s/{channel}"
            r = requests.get(url, timeout=FEED_FETCH_TIMEOUT, headers=_HEADERS)
            if r.status_code != 200:
                print(f"[SPEECH] Telegram {channel}: HTTP {r.status_code}")
                continue

            soup = BeautifulSoup(r.text, "html.parser")
            messages = soup.select(".tgme_widget_message_wrap")
            for msg in messages[-15:]:  # last 15 messages
                text_el = msg.select_one(".tgme_widget_message_text")
                time_el = msg.select_one("time")
                if not text_el:
                    continue
                text = text_el.get_text(strip=True)[:500]
                ts = time_el.get("datetime", "") if time_el else ""
                link_el = msg.select_one(".tgme_widget_message_date")
                link = link_el.get("href", "") if link_el else f"https://t.me/{channel}"
                all_messages.append({
                    "title": text[:80],
                    "content": text,
                    "url": link,
                    "time": ts,
                    "source": f"Telegram/@{channel}",
                })
        except Exception as e:
            print(f"[SPEECH] Telegram {channel} error: {e}")
    return all_messages


# ── Substack RSS ─────────────────────────────────────────────────────────────

def fetch_substack_feeds() -> List[Dict[str, str]]:
    """Fetch latest posts from configured Substack newsletters via RSS."""
    all_posts: List[Dict[str, str]] = []
    for feed_url in SUBSTACK_FEEDS:
        try:
            r = requests.get(feed_url, timeout=FEED_FETCH_TIMEOUT, headers=_HEADERS)
            if r.status_code != 200:
                print(f"[SPEECH] Substack {feed_url}: HTTP {r.status_code}")
                continue
            # Extract newsletter name from URL
            name = feed_url.split("//")[1].split(".")[0] if "//" in feed_url else feed_url
            items = _parse_rss_items(r.text, f"Substack/{name}")
            # Only keep posts from last 3 days
            all_posts.extend(items[:5])
        except Exception as e:
            print(f"[SPEECH] Substack error: {e}")
    return all_posts


# ── Polymarket Gamma API ─────────────────────────────────────────────────────

def fetch_polymarket_signals() -> List[Dict[str, Any]]:
    """Fetch top prediction markets from Polymarket Gamma API."""
    markets: List[Dict[str, Any]] = []
    try:
        # Fetch active markets sorted by volume
        url = f"{POLYMARKET_API_URL}/markets"
        params = {
            "limit": POLYMARKET_LIMIT,
            "active": True,
            "closed": False,
            "order": "volume",
            "ascending": False,
        }
        r = requests.get(url, params=params, timeout=FEED_FETCH_TIMEOUT, headers=_HEADERS)
        if r.status_code != 200:
            print(f"[SPEECH] Polymarket API: HTTP {r.status_code}")
            return []

        data = r.json()
        for m in data if isinstance(data, list) else data.get("data", data.get("markets", [])):
            question = m.get("question") or m.get("title") or ""
            outcome_prices = m.get("outcomePrices") or m.get("outcome_prices") or ""
            # Parse outcome probabilities
            try:
                if isinstance(outcome_prices, str):
                    probs = json.loads(outcome_prices)
                elif isinstance(outcome_prices, list):
                    probs = outcome_prices
                else:
                    probs = []
                yes_prob = float(probs[0]) if probs else None
            except (json.JSONDecodeError, IndexError, TypeError, ValueError):
                yes_prob = None

            volume = m.get("volume") or m.get("volumeNum") or 0
            markets.append({
                "question": question,
                "yes_probability": f"{yes_prob:.0%}" if yes_prob is not None else "N/A",
                "volume": volume,
                "url": f"https://polymarket.com/event/{m.get('slug', m.get('conditionId', ''))}",
                "source": "Polymarket",
            })
    except Exception as e:
        print(f"[SPEECH] Polymarket error: {e}")
    return markets
