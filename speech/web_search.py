"""Web search utility — DuckDuckGo (default) or Brave Search API."""

import html as _html
import json
import os
import re
import urllib.parse
import urllib.request
from typing import List, Dict

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


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


def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search DuckDuckGo HTML and return list of {title, url, snippet}."""
    q = str(query or "").strip()
    if not q:
        return []
    try:
        url = f"https://duckduckgo.com/html/?q={urllib.parse.quote(q)}"
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
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
    snippet_pattern = re.compile(
        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
        flags=re.IGNORECASE | re.DOTALL,
    )
    snippets = [_strip_html(_html.unescape(m.group(1))).strip() for m in snippet_pattern.finditer(text)]

    for i, m in enumerate(pattern.finditer(text)):
        u = _normalize_ddg_url(m.group(1))
        if not u.startswith("http") or u in seen:
            continue
        seen.add(u)
        title = _strip_html(_html.unescape(m.group(2))).strip() or u
        snippet = snippets[i] if i < len(snippets) else ""
        results.append({"title": title, "url": u, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


def search_brave(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search via Brave Search API (requires BRAVE_API_KEY env var)."""
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return []
    try:
        url = "https://api.search.brave.com/res/v1/web/search"
        req = urllib.request.Request(
            f"{url}?q={urllib.parse.quote(query)}&count={max_results}",
            headers={
                "User-Agent": _UA,
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        results = []
        for item in data.get("web", {}).get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
            })
        return results[:max_results]
    except Exception as e:
        print(f"[SPEECH] Brave Search error: {e}")
        return []


def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search the web using best available engine (Brave > DuckDuckGo)."""
    if os.getenv("BRAVE_API_KEY"):
        results = search_brave(query, max_results)
        if results:
            return results
    return search_duckduckgo(query, max_results)
