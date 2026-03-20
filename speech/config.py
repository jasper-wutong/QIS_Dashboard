"""Speech module configuration — data source URLs, accounts, defaults."""

# ── Nitter RSS (X/Twitter) ────────────────────────────────────────────────────
# Public Nitter instances (try in order; first reachable wins)
NITTER_INSTANCES = [
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.net",
]

# Finance / macro / derivatives accounts to follow on X
NITTER_ACCOUNTS = [
    "zaborhedge",        # Zaborovskiy — macro trader
    "DeItaone",          # Walter Bloomberg — breaking news
    "JavierBlas",        # Javier Blas — commodities
    "markets",           # Bloomberg Markets
    "Fxhedgers",         # FX / macro
    "GoldTelegraph_",    # Gold / precious metals
    "MacroAlf",          # Alfonso Peccatiello — macro
    "SoberLook",         # Sober Look — credit / rates
    "Kgreifeld",         # Katie Greifeld — derivatives / vol
    "VolatilityBounce",  # Volatility trader
]

# ── Telegram Channels (public, scraped via t.me/s/) ──────────────────────────
TELEGRAM_CHANNELS = [
    "WatcherGuru",            # Crypto + macro breaking news
    "MarketHedge",            # Options / hedge fund flows
    "CommodityWeather",       # Weather + commodities
    "financialjuice",         # FinancialJuice — macro headlines
]

# ── Substack RSS Feeds ────────────────────────────────────────────────────────
SUBSTACK_FEEDS = [
    "https://thebeartrapsreport.substack.com/feed",   # Larry McDonald — macro
    "https://markoithoughts.substack.com/feed",        # Marko Kolanovic successor
    "https://goldfix.substack.com/feed",               # Goldfix — precious metals
    "https://rfreedman.substack.com/feed",             # Robin Freedman — commodity options
]

# ── Polymarket Gamma API ─────────────────────────────────────────────────────
POLYMARKET_API_URL = "https://gamma-api.polymarket.com"
POLYMARKET_LIMIT = 20  # top N markets by volume

# ── Web Search ────────────────────────────────────────────────────────────────
# Targeted queries to run for morning briefing context
WEB_SEARCH_QUERIES = [
    "overnight market moves futures commodities",
    "Fed ECB central bank latest news today",
    "options market volatility VIX today",
    "China macro policy PBOC latest",
    "geopolitical risk oil gold sanctions",
]

# ── Bailian RAG ──────────────────────────────────────────────────────────────
BAILIAN_APP_ID = "5be4e5cbe00f478390842a0254bd8abb"

# ── CICC Research PDF paths (relative to project root) ───────────────────────
CICC_MORNING_FOCUS_DIR = "memory/cicc_research/晨会焦点"
CICC_COMMODITY_DIR = "memory/cicc_research/大宗商品"
CICC_MACRO_DIR = "memory/cicc_research/宏观经济"
CICC_STRATEGY_DIR = "memory/cicc_research/市场策略"

# ── Output cache ─────────────────────────────────────────────────────────────
SPEECH_CACHE_DIR = "memory/speech"

# ── Copilot SDK ──────────────────────────────────────────────────────────────
PREFERRED_MODELS = ["claude-sonnet-4.6", "claude-sonnet-4", "gpt-5.2", "gpt-5", "gpt-5.1"]

# ── Timeouts ─────────────────────────────────────────────────────────────────
FEED_FETCH_TIMEOUT = 15       # seconds per individual feed
DATA_COLLECT_TIMEOUT = 60     # seconds for all data collection (graceful timeout)
COPILOT_SDK_TIMEOUT = 360     # seconds for LLM generation
