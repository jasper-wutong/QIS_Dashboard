"""
期货/ETF/指数 Ticker → 中文名称 映射表及解析逻辑。
只需维护合约前缀（如 AG、ES、GC），程序自动识别月份/年份变化。
"""
import re
import pandas as pd

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
