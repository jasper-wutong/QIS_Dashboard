"""Ticker normalisation for QIS Cross Gamma.

Maps Bloomberg + Wind tickers to canonical Chinese display names.
Merges near/far month contracts of the same underlying.
Falls back to raw ticker for anything unknown (new instruments appear regularly).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Raw ticker → canonical display name
# ---------------------------------------------------------------------------
# Near / far month contracts of the same underlying share one canonical name.
# BBG and Wind tickers for the same underlying also share one canonical name.
TICKER_TO_CANONICAL: dict[str, str] = {
    # ── Equity Index ──────────────────────────────────────────────────────
    "000688.SH":       "科创50",
    "SZ399006 Index":  "创业板指",
    "IFBH26 Index":    "沪深300",
    "IFBJ26 Index":    "沪深300",
    "IFDH26 Index":    "沪深300",
    "IF2603.CFE":      "沪深300",
    "IC2603.CFE":      "中证500",
    "IM2603.CFE":      "中证1000",
    "DMH6 Index":      "DAX",
    "DMM6 Index":      "DAX",
    "ESM6 Index":      "标普S&P",
    "FFDH26 Index":    "FTSE",
    "FFDJ26 Index":    "FTSE",
    "GXH6 Index":      "EuroStoxx",
    "GXM6 Index":      "EuroStoxx",
    "NQH6 Index":      "纳指NQ",
    "NQM6 Index":      "纳指NQ",
    "NKM6 Index":      "日经225",
    "NOM6 Index":      "OMXS30",

    # ── Precious Metals ───────────────────────────────────────────────────
    "AUAM26 Comdty":   "黄金",
    "AU2606.SHF":      "黄金",
    "GCM6 Comdty":     "黄金",
    "AG2606.SHF":      "白银",

    # ── Base Metals ───────────────────────────────────────────────────────
    "ACK26 Comdty":    "铝",
    "AEK26 Comdty":    "铝",
    "CUK26 Comdty":    "铜",
    "CU2605.SHF":      "铜",
    "HGK6 Comdty":     "铜",
    "NI2605.SHF":      "镍",
    "ZN2605.SHF":      "锌",

    # ── Energy ────────────────────────────────────────────────────────────
    "CLK6 Comdty":     "原油WTI",
    "COM6 Comdty":     "布油Brent",
    "SCPM26 Comdty":   "原油SC",
    "SC2606.INE":      "原油SC",

    # ── Ferrous / Industrial ──────────────────────────────────────────────
    "I2605.DCE":       "铁矿",
    "J2605.DCE":       "焦炭",
    "HC2605.SHF":      "热卷",
    "RB2605.SHF":      "螺纹钢",
    "SP2605.SHF":      "纸浆",

    # ── Agriculture ───────────────────────────────────────────────────────
    "M2605.DCE":       "豆粕",
    "Y2605.DCE":       "豆油",
    "P2605.DCE":       "棕榈油",
    "OI605.CZC":       "菜油",
    "RM605.CZC":       "菜粕",
    "CF605.CZC":       "棉花",
    "SR605.CZC":       "白糖",

    # ── Chemical ──────────────────────────────────────────────────────────
    "L2605.DCE":       "塑料",
    "PP2605.DCE":      "聚丙烯",
    "EG2605.DCE":      "乙二醇",
    "TA605.CZC":       "PTA",
    "MA605.CZC":       "甲醇",
    "SA605.CZC":       "纯碱",
    "FG605.CZC":       "玻璃",

    # ── Bonds ─────────────────────────────────────────────────────────────
    "TFCM26 Comdty":   "国债5Y TF",
    "TFTM26 Comdty":   "国债5Y TF",
    "TF2606.CFE":      "国债5Y TF",
    "T2606.CFE":       "国债10Y T",
    "TL2606.CFE":      "国债30Y TL",
    "TS2606.CFE":      "国债2Y TS",
    "TYM6 Comdty":     "美债10Y TY",
    "FVM6 Comdty":     "美债5Y FV",
    "RXM6 Comdty":     "德债Bund",
    "JBM6 Comdty":     "日债JGB",

    # ── ETF ───────────────────────────────────────────────────────────────
    "511380.SH":       "可转债ETF",
    "512890.SH":       "红利ETF",

    # ── FX ────────────────────────────────────────────────────────────────
    "CNYHKD":          "CNYHKD",

    # ── Other ─────────────────────────────────────────────────────────────
    "ARES2PRO":        "ARES2PRO",
}

# ---------------------------------------------------------------------------
# Asset class lookup
# ---------------------------------------------------------------------------
_INDEX = {
    "科创50", "创业板指", "沪深300", "中证500", "中证1000",
    "DAX", "标普S&P", "FTSE", "EuroStoxx", "纳指NQ", "日经225", "OMXS30",
}
_COMMODITY = {
    "黄金", "白银", "铝", "铜", "镍", "锌",
    "原油WTI", "布油Brent", "原油SC",
    "铁矿", "焦炭", "热卷", "螺纹钢", "纸浆",
    "豆粕", "豆油", "棕榈油", "菜油", "菜粕", "棉花", "白糖",
    "塑料", "聚丙烯", "乙二醇", "PTA", "甲醇", "纯碱", "玻璃",
}
_BOND = {
    "国债5Y TF", "国债10Y T", "国债30Y TL", "国债2Y TS",
    "美债10Y TY", "美债5Y FV", "德债Bund", "日债JGB",
}
_ETF = {"可转债ETF", "红利ETF"}
_FX = {"CNYHKD"}


def get_asset_class(canonical: str) -> str:
    """Return the asset class for a canonical ticker name."""
    if canonical in _INDEX:
        return "Index"
    if canonical in _BOND:
        return "Bond"
    if canonical in _COMMODITY:
        return "Commodity"
    if canonical in _ETF:
        return "ETF"
    if canonical in _FX:
        return "FX"
    return "Other"


# Sort priority: Index → Bond → Commodity → ETF → FX → Other
_CLASS_ORDER = {"Index": 0, "Bond": 1, "Commodity": 2, "ETF": 3, "FX": 4, "Other": 5}


def sort_key(canonical: str) -> tuple:
    """Sort key for displaying tickers: Index → Bond → Commodity → ETF → FX → Other."""
    cls = get_asset_class(canonical)
    return (_CLASS_ORDER.get(cls, 5), canonical)


def normalize(ticker: str) -> str:
    """Return the canonical display name for a raw ticker.

    Falls back to the raw ticker if not found in the map.
    """
    return TICKER_TO_CANONICAL.get(ticker, ticker)


def normalize_pair(ticker_a: str, ticker_b: str) -> tuple[str, str]:
    """Normalise a pair of raw tickers, returned in sorted canonical order."""
    a = normalize(ticker_a)
    b = normalize(ticker_b)
    return (min(a, b), max(a, b)) if a <= b else (b, a)
