"""
期货/ETF/指数 Ticker → 中文名称 映射表及解析逻辑。
只需维护合约前缀（如 AG、ES、GC），程序自动识别月份/年份变化。
"""
import re
from typing import Optional, List
import pandas as pd

# ── 板块分类 ─────────────────────────────────────────────────────────────────
SECTOR_ORDER = [
    "贵金属", "有色金属", "焦煤钢矿", "非金属建材", "能源", "化工",
    "油脂油料", "软商品", "农副产品", "谷物", "航运", "股", "债", "其他"
]

SECTOR_ICONS = {
    "贵金属": "🥇", "有色金属": "🔩", "焦煤钢矿": "⛏️", "非金属建材": "🧱",
    "能源": "⛽", "化工": "🧪", "油脂油料": "🫒", "软商品": "🍬",
    "农副产品": "🐷", "谷物": "🌾", "航运": "🚢", "股": "📈", "债": "💰", "其他": "📦"
}

# 合约前缀 → 板块
SECTOR_MAP = {
    # 贵金属
    "AG": "贵金属", "AU": "贵金属", "GC": "贵金属", "SI": "贵金属",
    # 有色金属
    "CU": "有色金属", "NI": "有色金属", "ZN": "有色金属", "AL": "有色金属",
    "PB": "有色金属", "SN": "有色金属", "SS": "有色金属", "BC": "有色金属", "HG": "有色金属",
    # 焦煤钢矿
    "HC": "焦煤钢矿", "RB": "焦煤钢矿", "I": "焦煤钢矿", "J": "焦煤钢矿", "JM": "焦煤钢矿",
    "SF": "焦煤钢矿", "SM": "焦煤钢矿",
    # 非金属建材
    "FG": "非金属建材", "SA": "非金属建材",
    # 能源
    "SC": "能源", "LU": "能源", "NR": "能源", "BU": "能源", "RU": "能源", "FU": "能源",
    "CO": "能源", "CL": "能源", "PG": "能源", "ZC": "能源",
    # 化工
    "EG": "化工", "L": "化工", "PP": "化工", "EB": "化工", "V": "化工",
    "MA": "化工", "TA": "化工", "PF": "化工", "UR": "化工", "SP": "化工",
    # 油脂油料
    "M": "油脂油料", "P": "油脂油料", "Y": "油脂油料", "OI": "油脂油料",
    "RM": "油脂油料", "PK": "油脂油料",
    # 软商品
    "CF": "软商品", "SR": "软商品", "AP": "软商品", "CJ": "软商品", "CY": "软商品",
    # 农副产品
    "JD": "农副产品", "LH": "农副产品",
    # 谷物
    "C": "谷物", "A": "谷物", "B": "谷物", "CS": "谷物", "RR": "谷物",
    # 股指期货 - 境内
    "IC": "股", "IF": "股", "IM": "股", "IH": "股",
    # 债券期货 - 境内
    "T": "债", "TF": "债", "TL": "债", "TS": "债",
    # 股指期货 - 港股
    "HSIF": "股", "HHIF": "股", "HTIF": "股",
    # 金融期货 - Eurex
    "FDAX": "股", "FGBL": "债",
    # 股指期货 - 境外
    "ES": "股", "NQ": "股", "DM": "股", "NK": "股", "NO": "股",
    # 债券期货 - 境外
    "FV": "债", "TY": "债", "TU": "债", "US": "债", "JB": "债",
}

# ── 期货合约前缀 → 中文名称 ──────────────────────────────────────────────────
FUTURES_NAME_MAP = {
    # ── 上期所 (.SHF) ──
    "AG": "白银期货", "AU": "黄金期货", "CU": "铜期货",
    "HC": "热轧卷板", "NI": "镍期货", "RB": "螺纹钢",
    "SP": "纸浆期货", "ZN": "锌期货", "AL": "铝期货",
    "PB": "铅期货", "SN": "锡期货", "SS": "不锈钢",
    "BU": "沥青期货", "RU": "橡胶期货", "FU": "燃料油",
    # ── 上期能源 (.INE) ──
    "SC": "原油期货", "LU": "低硫燃料油", "NR": "20号胶", "BC": "国际铜",
    # ── 大商所 (.DCE) ──
    "I": "铁矿石", "M": "豆粕期货", "C": "玉米期货",
    "EG": "乙二醇", "J": "焦炭期货", "JM": "焦煤期货",
    "L": "聚乙烯(塑料)", "P": "棕榈油", "PP": "聚丙烯",
    "Y": "豆油期货", "A": "豆一期货", "B": "豆二期货",
    "CS": "淀粉期货", "EB": "苯乙烯", "JD": "鸡蛋期货",
    "LH": "生猪期货", "PG": "液化石油气", "RR": "粳米期货", "V": "PVC期货",
    # ── 郑商所 (.CZC) ──
    "CF": "棉花期货", "FG": "玻璃期货", "MA": "甲醇期货",
    "OI": "菜油期货", "RM": "菜粕期货", "SA": "纯碱期货",
    "SR": "白糖期货", "TA": "PTA期货", "AP": "苹果期货",
    "CJ": "红枣期货", "CY": "棉纱期货", "PF": "短纤期货",
    "PK": "花生期货", "SF": "硅铁期货", "SM": "锰硅期货",
    "UR": "尿素期货", "ZC": "动力煤",
    # ── 中金所 (.CFE) ──
    "IC": "中证500股指", "IF": "沪深300股指",
    "IM": "中证1000股指", "IH": "上证50股指",
    "T": "10Y国债期货", "TF": "5Y国债期货",
    "TL": "30Y国债期货", "TS": "2Y国债期货",
    # ── 港股期货 (.HK) ──
    "HSIF": "恒生指数期货", "HHIF": "恒生国企指数期货", "HTIF": "恒生科技指数期货",
    # ── Eurex ──
    "FDAX": "DAX指数期货", "FGBL": "德国国债期货",
    # ── Bloomberg 境外指数 (Index) ──
    "ES": "标普500期货", "NQ": "纳指100期货",
    "DM": "道指迷你期货", "NK": "日经225期货(SGX)",
    "NO": "日经225迷你(OSE)",
    # ── Bloomberg 境外商品/债券 (Comdty) ──
    "CO": "布伦特原油", "GC": "COMEX黄金",
    "HG": "COMEX铜", "SI": "COMEX白银", "CL": "WTI原油",
    "FV": "美国5Y国债", "TY": "美国10Y国债",
    "TU": "美国2Y国债", "US": "美国30Y国债",
    "JB": "日本国债(JGB)",
}

# ── ETF / 指数代码 → 中文名称 ────────────────────────────────────────────────
ETF_NAME_MAP = {
    "159915.SZ": "创业板ETF", "588000.SH": "科创50ETF",
    "511380.SH": "可转债ETF", "512890.SH": "红利低波ETF",
    "000688.SH": "科创50指数", "SZ399006": "创业板指",
}

# ── 特殊指数代码（无月份码）────────────────────────────────────────────────────
SPECIAL_INDEX_MAP = {
    "SPX": "标普500指数", "NDX": "纳指100指数", "RTY": "罗素2000指数",
}

# Bloomberg 月份代码: F=1月 G=2月 H=3月 J=4月 K=5月 M=6月
#                     N=7月 Q=8月 U=9月 V=10月 X=11月 Z=12月
_BBG_MONTH_CODES = set("FGHJKMNQUVXZ")


# ── 解析函数 ─────────────────────────────────────────────────────────────────
def resolve_ticker_name(ticker_str):
    """从 Wind Ticker 或标的物代码解析出中文名称。

    规则:
      境内: AG2606.SHF → AG → 白银期货
            CF605.CZC  → CF → 棉花期货
            HSIF2602.HK → HSIF → 恒生指数期货
            FDAX2603    → FDAX → DAX指数期货
      境外: ESH6 Index  → ES → 标普500期货
            GCJ6 Comdty → GC → COMEX黄金
    """
    if pd.isna(ticker_str):
        return None
    ticker = str(ticker_str).strip()
    if not ticker:
        return None

    # 1) ETF / 指数精确匹配
    if ticker in ETF_NAME_MAP:
        return ETF_NAME_MAP[ticker]

    # 2) 境内期货: XX2606.SHF / XX605.CZC / XX2602.CFE / XX2604.INE
    m = re.match(r'^([A-Za-z]+)\d+\.(SHF|DCE|CZC|INE|CFE)$', ticker)
    if m:
        return FUTURES_NAME_MAP.get(m.group(1).upper())

    # 3) 港股期货: HHIF2602.HK
    m = re.match(r'^([A-Za-z]+)\d+\.HK$', ticker)
    if m:
        return FUTURES_NAME_MAP.get(m.group(1).upper())

    # 4) Eurex: FDAX2603, FGBL2603 (纯字母+数字, 无交易所后缀)
    m = re.match(r'^([A-Za-z]{2,})\d{4}$', ticker)
    if m:
        return FUTURES_NAME_MAP.get(m.group(1).upper())

    # 5) Bloomberg: ESH6 Index / GCJ6 Comdty (前缀 + 月份码 + 年份 + 类型)
    m = re.match(r'^([A-Za-z]+?)([FGHJKMNQUVXZ])(\d{1,2})\s+(Index|Comdty)$', ticker)
    if m and m.group(2) in _BBG_MONTH_CODES:
        return FUTURES_NAME_MAP.get(m.group(1).upper())

    # 6) 简单指数代码: SPX Index
    if ticker.endswith(" Index"):
        code = ticker.replace(" Index", "").strip().upper()
        return SPECIAL_INDEX_MAP.get(code)

    return None


def populate_names(df):
    """为 名称 列为空或 '-' 的行, 根据 Wind Ticker / 标的物 自动填充中文名称。"""
    for i, row in df.iterrows():
        current = row.get("名称")
        if pd.isna(current) or str(current).strip() in ("", "-"):
            name = resolve_ticker_name(row.get("Wind Ticker"))
            if name is None:
                name = resolve_ticker_name(row.get("标的物"))
            if name:
                df.at[i, "名称"] = name


def _extract_prefix(ticker_str):
    """从 Ticker 字符串提取合约前缀用于板块分类。"""
    if pd.isna(ticker_str):
        return None
    ticker = str(ticker_str).strip()
    if not ticker:
        return None

    # 1) ETF / 指数精确匹配 - 归类为金融期货
    if ticker in ETF_NAME_MAP:
        return None  # ETF 不分类

    # 2) 境内期货: XX2606.SHF / XX605.CZC / XX2602.CFE / XX2604.INE
    m = re.match(r'^([A-Za-z]+)\d+\.(SHF|DCE|CZC|INE|CFE)$', ticker)
    if m:
        return m.group(1).upper()

    # 3) 港股期货: HHIF2602.HK
    m = re.match(r'^([A-Za-z]+)\d+\.HK$', ticker)
    if m:
        return m.group(1).upper()

    # 4) Eurex: FDAX2603, FGBL2603 (纯字母+数字, 无交易所后缀)
    m = re.match(r'^([A-Za-z]{2,})\d{4}$', ticker)
    if m:
        return m.group(1).upper()

    # 5) Bloomberg: ESH6 Index / GCJ6 Comdty (前缀 + 月份码 + 年份 + 类型)
    m = re.match(r'^([A-Za-z]+?)([FGHJKMNQUVXZ])(\d{1,2})\s+(Index|Comdty)$', ticker)
    if m and m.group(2) in _BBG_MONTH_CODES:
        return m.group(1).upper()

    return None


def resolve_sector(ticker_str, underlying_str=None):
    """根据 Wind Ticker 或 标的物 解析板块分类。

    Args:
        ticker_str: Wind Ticker
        underlying_str: 标的物（备用）

    Returns:
        str: 板块名称（如 "贵金属"），未识别返回 "其他"
    """
    prefix = _extract_prefix(ticker_str)
    if prefix and prefix in SECTOR_MAP:
        return SECTOR_MAP[prefix]

    # 尝试使用标的物
    if underlying_str:
        prefix = _extract_prefix(underlying_str)
        if prefix and prefix in SECTOR_MAP:
            return SECTOR_MAP[prefix]

    return "其他"


# ── 境内期货交易所后缀 ─────────────────────────────────────────────────────────
DOMESTIC_FUTURES_SUFFIXES = ('.SHF', '.DCE', '.CZC', '.INE', '.CFE')


def resolve_instrument_type(ticker_str):
    """分类 ticker 为 '境内期货' / '境外期货' / '境内ETF'。

    分类规则（对齐 QIS_BOARD）：
      - 境内期货: .SHF / .DCE / .CZC / .INE / .CFE
      - 境内ETF:  属于 ETF_NAME_MAP（.SH / .SZ 的 ETF 和指数）
      - 境外期货: 其余（.HK / Bloomberg Index|Comdty / Eurex）
    """
    if pd.isna(ticker_str):
        return "境内期货"  # fallback
    ticker = str(ticker_str).strip()
    if not ticker:
        return "境内期货"

    # 境内期货: domestic futures exchanges
    if any(ticker.upper().endswith(sfx) for sfx in DOMESTIC_FUTURES_SUFFIXES):
        return "境内期货"

    # 境内 ETF / 指数
    if ticker in ETF_NAME_MAP:
        return "境内ETF"

    # 其余均为境外期货
    return "境外期货"


def resolve_region(ticker_str, underlying_str=None):
    """根据 Wind Ticker 或 标的物 解析境内/境外。

    Args:
        ticker_str: Wind Ticker
        underlying_str: 标的物（备用）

    Returns:
        str: "境内" 或 "境外"
    """
    def _check_region(ticker):
        if pd.isna(ticker):
            return None
        ticker = str(ticker).strip()
        if not ticker:
            return None

        # 境内期货: .SHF, .DCE, .CZC, .INE, .CFE
        if re.search(r'\.(SHF|DCE|CZC|INE|CFE)$', ticker, re.IGNORECASE):
            return "境内"

        # 境外: .HK
        if ticker.endswith('.HK'):
            return "境外"

        # 境外: Bloomberg 格式 (XXX Index / XXX Comdty)
        if ' Index' in ticker or ' Comdty' in ticker:
            return "境外"

        # 境外: Eurex 格式 (FDAX2603, FGBL2603)
        if re.match(r'^(FDAX|FGBL)\d{4}$', ticker):
            return "境外"

        # 境内 ETF: .SH, .SZ
        if re.search(r'\.(SH|SZ)$', ticker, re.IGNORECASE):
            return "境内"

        return None

    region = _check_region(ticker_str)
    if region:
        return region

    if underlying_str:
        region = _check_region(underlying_str)
        if region:
            return region

    return "境内"  # 默认境内


# ── 反向映射: 中文名称 → 合约前缀 ────────────────────────────────────────────
_NAME_TO_PREFIX: dict = {v: k for k, v in FUTURES_NAME_MAP.items()}
_NAME_TO_ETF: dict = {v: k for k, v in ETF_NAME_MAP.items()}
_NAME_TO_SPECIAL: dict = {v: k for k, v in SPECIAL_INDEX_MAP.items()}


def resolve_bbg_ticker(wind_ticker: str) -> Optional[str]:
    """
    将 Wind Ticker 映射为 Bloomberg 可识别格式。

    示例:
      ESH6 Index → ESH6 Index  (已是 Bloomberg 格式, 直接返回)
      GCJ6 Comdty → GCJ6 Comdty
      HSIF2602.HK → HSIF2602.HK  (需要特殊处理)
      FDAX2603 → FDAX2603 Index (Eurex)

    对于已经是 Bloomberg 格式的 ticker (含 Index / Comdty), 直接返回。
    """
    if pd.isna(wind_ticker):
        return None
    ticker = str(wind_ticker).strip()
    if not ticker:
        return None

    # 已经是 Bloomberg 格式
    if " Index" in ticker or " Comdty" in ticker:
        return ticker

    # 港股期货: HSIF2602.HK → 不变 (Bloomberg 实际用不同格式, 但留作扩展)
    if ticker.endswith(".HK"):
        return None  # 暂不支持 HK futures via Bloomberg

    # Eurex: FDAX2603 → 尝试直接返回
    m = re.match(r'^(FDAX|FGBL)\d{4}$', ticker)
    if m:
        return ticker

    return ticker  # 对境外 ticker, 原样传递给 Bloomberg (最大兼容)


def resolve_name_to_wind_ticker(
    name: str,
    df_data: Optional[list] = None,
    columns: Optional[list] = None,
) -> Optional[str]:
    """
    从中文名称反查 Wind Ticker。

    优先从 df_data (dashboard 的 other_records) 中精确匹配 '名称' 列,
    如无匹配则尝试通过 FUTURES_NAME_MAP 反向查找前缀。

    Args:
        name: 中文名称, e.g. "黄金期货", "铜期货"
        df_data: dashboard 的 other_records (list of lists)
        columns: 对应的列名列表

    Returns:
        Wind Ticker 字符串, 或 None
    """
    if not name:
        return None

    # 1) 从实际数据中精确匹配
    if df_data and columns:
        name_idx = None
        ticker_idx = None
        underlying_idx = None
        for i, c in enumerate(columns):
            if c == "名称":
                name_idx = i
            elif c == "Wind Ticker":
                ticker_idx = i
            elif c == "标的物":
                underlying_idx = i

        if name_idx is not None and ticker_idx is not None:
            for row in df_data:
                if row[name_idx] == name and row[ticker_idx]:
                    return str(row[ticker_idx])

        # 也尝试匹配标的物列
        if underlying_idx is not None and ticker_idx is not None:
            for row in df_data:
                if row[underlying_idx] == name and row[ticker_idx]:
                    return str(row[ticker_idx])

    # 2) ETF 反向查找
    if name in _NAME_TO_ETF:
        return _NAME_TO_ETF[name]

    # 3) 特殊指数
    if name in _NAME_TO_SPECIAL:
        return _NAME_TO_SPECIAL[name]

    return None
