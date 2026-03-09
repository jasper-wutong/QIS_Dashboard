"""
大宗商品研究中心 — 品种专属研究因子配置与量化评分。

设计参考: Goldman Sachs CIRA、JPMorgan Commodity Research、摩根士丹利 Global Commodities
每个品种/板块独立配置研究逻辑，包含:
 - 关键基本面驱动因子 (定性描述 + 信号方向)
 - 量化因子列表 (可从 OHLCV/OI 计算)
 - 期限结构合约映射 (Bloomberg / Wind Tickers)
 - 季节性规律月份
 - 研究方法论备注
"""

import math
import statistics
from typing import Dict, Any, List, Optional
from datetime import datetime

# ══════════════════════════════════════════════════════════════════
#  板块专属研究配置
# ══════════════════════════════════════════════════════════════════

SECTOR_RESEARCH_CONFIG: Dict[str, Dict] = {

    "能源": {
        "icon": "⛽",
        "en_name": "Energy",
        "sub_sectors": ["原油", "天然气", "燃料油", "液化气"],
        "key_drivers": [
            {"factor": "EIA原油库存",     "bbg_ticker": "DOECRUOP Index", "direction": "bearish_if_build",
             "description": "周度美国商业原油库存变化。库存增加→看空；库存减少→看多"},
            {"factor": "OPEC+产量执行率", "bbg_ticker": None,              "direction": "bullish_if_cut",
             "description": "OPEC+成员国减产执行率。执行率越高→供应越紧→看多"},
            {"factor": "库欣库存",        "bbg_ticker": "DOECUS Index",   "direction": "bearish_if_build",
             "description": "WTI交割地库欣库存变化，直接影响WTI-Brent价差"},
            {"factor": "Baker Hughes钻井数", "bbg_ticker": "BAKERACT Index", "direction": "bearish_if_rise",
             "description": "美国活跃钻井数。钻井增加→页岩油产量预期上升→看空"},
            {"factor": "裂解价差 Crack Spread", "bbg_ticker": None,         "direction": "bullish_if_wide",
             "description": "炼厂利润(3-2-1 Crack)反映下游需求强度。价差扩大→炼厂需求旺→看多原油"},
            {"factor": "汽油需求",        "bbg_ticker": "DOEGMOTP Index", "direction": "bullish_if_rise",
             "description": "美国四周平均汽油需求量，反映消费端季节性需求"},
        ],
        "quant_factors": ["momentum_1m", "momentum_3m", "basis_signal", "oi_trend", "vol_regime", "seasonal_score"],
        "term_structure": {
            "CL":   [f"CL{i} Comdty" for i in range(1, 9)],   # WTI  M1-M8
            "CO":   [f"CO{i} Comdty" for i in range(1, 9)],   # Brent M1-M8
            "SC":   [f"SC0{i}.INE"   for i in range(1, 7)],   # 上海原油 M1-M6
            "LU":   [f"LU0{i}.INE"   for i in range(1, 7)],   # 低硫燃料
            "FU":   [f"FU0{i}.SHF"   for i in range(1, 7)],
            "PG":   [f"PG0{i}.DCE"   for i in range(1, 7)],
        },
        "seasonal_pattern": {"strong": [9, 10, 11, 12, 1, 2], "weak": [3, 4, 5, 6]},
        "research_notes": (
            "核心分析框架：全球供需平衡表（IEA/OPEC/EIA三方预测对比）+ 库存库欣周期。"
            "OPEC+政策是最大单变量风险。美国页岩油产量弹性在 $65-75 WTI 区间激活。"
            "布伦特-WTI价差（通常$2-5）扩大时关注地缘运输风险。"
            "关键季节性：夏季驾驶旺季(6-8月)汽油需求↑；冬季供暖(12-2月)馏分油需求↑。"
        ),
        "correlation_assets": [
            {"name": "DXY美元指数", "direction": "negative", "bbg": "DXY Curncy"},
            {"name": "全球PMI",     "direction": "positive",  "bbg": "MXAPJ Index"},
        ],
    },

    "贵金属": {
        "icon": "🥇",
        "en_name": "Precious Metals",
        "sub_sectors": ["黄金", "白银"],
        "key_drivers": [
            {"factor": "美国实际利率",    "bbg_ticker": "GTII10 Govt",    "direction": "bearish_if_rise",
             "description": "10Y TIPS实际收益率。实际利率上升→持有黄金机会成本增加→看空(核心驱动)"},
            {"factor": "美元指数DXY",     "bbg_ticker": "DXY Curncy",     "direction": "bearish_if_rise",
             "description": "美元走强→以美元计价的黄金对非美买家更贵→看空。相关系数约-0.7"},
            {"factor": "10Y盈亏平衡通胀", "bbg_ticker": "USGGBE10 Index", "direction": "bullish_if_rise",
             "description": "市场隐含通胀预期。通胀预期上升→黄金作为通胀对冲资产→看多"},
            {"factor": "SPDR GLD持仓",    "bbg_ticker": "GLDUS Equity",   "direction": "bullish_if_rise",
             "description": "全球最大黄金ETF持仓量。机构资金流向领先指标，变化方向=短期情绪"},
            {"factor": "央行购金",        "bbg_ticker": None,              "direction": "bullish_if_buy",
             "description": "全球央行净购金量(WGC数据)。新兴市场央行持续增持是2022年后结构需求"},
            {"factor": "COT投机净多",     "bbg_ticker": "CFTGCNET Index", "direction": "sentiment_extreme_bearish",
             "description": "CFTC黄金投机净头寸。极高净多→拥挤→反转风险；极低→底部支撑"},
        ],
        "quant_factors": ["momentum_1m", "momentum_3m", "momentum_6m", "oi_trend", "vol_regime", "seasonal_score"],
        "term_structure": {
            "GC": [f"GC{i} Comdty" for i in range(1, 7)],
            "SI": [f"SI{i} Comdty" for i in range(1, 7)],
            "AU": [f"AU0{i}.SHF" for i in range(1, 7)],
            "AG": [f"AG0{i}.SHF" for i in range(1, 7)],
        },
        "seasonal_pattern": {"strong": [1, 2, 8, 9, 10, 11], "weak": [3, 4, 6, 7]},
        "research_notes": (
            "Goldman Sachs黄金框架: 黄金价格 ≈ -25×实际利率(%) + 通胀风险溢价 + 地缘风险溢价。"
            "实际利率解释黄金约60-70%的价格变动。2022年后央行购金(新兴市场去美元化)作为结构性新增需求。"
            "白银具有工业属性(太阳能/电子)，金银比>80时白银相对低估，<65时相对高估。"
        ),
        "correlation_assets": [
            {"name": "实际利率TIPS",  "direction": "negative", "bbg": "GTII10 Govt"},
            {"name": "DXY美元",       "direction": "negative", "bbg": "DXY Curncy"},
            {"name": "VIX隐波",       "direction": "positive",  "bbg": "VIX Index"},
        ],
    },

    "有色金属": {
        "icon": "🔩",
        "en_name": "Base Metals",
        "sub_sectors": ["铜", "铝", "锌", "镍", "铅", "锡"],
        "key_drivers": [
            {"factor": "中国制造业PMI",   "bbg_ticker": "CPMINDX Index",  "direction": "bullish_if_rise",
             "description": "中国制造业PMI>50扩张。铜需求约50%来自中国，PMI是核心领先指标"},
            {"factor": "LME交易所库存",   "bbg_ticker": "LMCADY Index",   "direction": "bearish_if_rise",
             "description": "LME铜库存变化。库存连续下降是库存紧张信号（Backwardation驱动）"},
            {"factor": "上期所SHFE库存",  "bbg_ticker": None,              "direction": "bearish_if_rise",
             "description": "上海期货交易所铜库存，每周五更新，反映国内实货松紧"},
            {"factor": "铜冶炼加工费TC/RC", "bbg_ticker": None,            "direction": "bearish_if_low",
             "description": "加工费反映矿端供应松紧。TC低(<$50)→矿端紧张→冶炼产量受限→看多铜"},
            {"factor": "中国房地产新开工", "bbg_ticker": None,              "direction": "bullish_if_rise",
             "description": "建筑用铜约占中国铜需求25%，新开工数据领先铜需求约3-6个月"},
            {"factor": "新能源用铜",      "bbg_ticker": None,              "direction": "structural_bullish",
             "description": "每辆EV含铜约60-80kg(传统车8kg)，风电光伏电缆用铜。长期结构性需求"},
        ],
        "quant_factors": ["momentum_1m", "momentum_3m", "basis_signal", "oi_trend", "vol_regime"],
        "term_structure": {
            "HG": [f"HG{i} Comdty" for i in range(1, 7)],   # COMEX铜
            "CU": [f"CU0{i}.SHF" for i in range(1, 7)],     # SHFE铜
            "AL": [f"AL0{i}.SHF" for i in range(1, 7)],
            "ZN": [f"ZN0{i}.SHF" for i in range(1, 7)],
            "NI": [f"NI0{i}.SHF" for i in range(1, 7)],
        },
        "seasonal_pattern": {"strong": [3, 4, 5, 9, 10], "weak": [1, 2, 7, 8]},
        "research_notes": (
            "铜是宏观经济的'铜博士'(Dr. Copper)，与全球GDP强相关(r≈0.7)。"
            "铜价核心驱动：中国需求(50%+)、矿山供应干扰率、TC/RC加工费、LME/SHFE库存差。"
            "铜市场定价模型: 库存/需求比(S/U)分位数 → 现货溢价/贴水 → 期货曲线形态。"
            "镍受新能源电池(硫酸镍)和不锈钢双重影响，分析需区分高/低冰镍路线。"
        ),
        "correlation_assets": [
            {"name": "中国PMI", "direction": "positive", "bbg": "CPMINDX Index"},
            {"name": "美元DXY", "direction": "negative", "bbg": "DXY Curncy"},
        ],
    },

    "焦煤钢矿": {
        "icon": "⛏️",
        "en_name": "Ferrous Metals",
        "sub_sectors": ["螺纹钢", "铁矿石", "焦炭", "焦煤", "热卷"],
        "key_drivers": [
            {"factor": "中国钢铁产量",    "bbg_ticker": None,              "direction": "bullish_if_rise",
             "description": "中国月度粗钢产量(占全球~55%)。环比增减直接影响铁矿石/焦煤需求"},
            {"factor": "钢厂高炉开工率",  "bbg_ticker": None,              "direction": "bullish_if_rise",
             "description": "Mysteel追踪高炉开工率。开工率上升→铁矿石/焦炭消耗增加→看多"},
            {"factor": "钢厂利润(吨钢)",  "bbg_ticker": None,              "direction": "correlated",
             "description": "钢厂吨钢盈亏=螺纹价格-原料成本。亏损→减产→压制成材价格，但支撑利润"},
            {"factor": "铁矿石港口库存",  "bbg_ticker": None,              "direction": "bearish_if_rise",
             "description": "中国铁矿石港口库存总量(Mysteel)。库存高→买矿意愿弱→看空铁矿"},
            {"factor": "地产竣工面积",    "bbg_ticker": None,              "direction": "bullish_if_rise",
             "description": "建筑用钢约占中国钢材消费60%，竣工领先钢需1-2季度"},
            {"factor": "限产政策信号",    "bbg_ticker": None,              "direction": "bullish_if_restrict",
             "description": "粗钢控产令、限产减排政策→供应端收缩→成材价格压缩利润，矿价反而受压"},
        ],
        "quant_factors": ["momentum_1m", "momentum_3m", "basis_signal", "oi_trend", "vol_regime", "seasonal_score"],
        "term_structure": {
            "RB": [f"RB0{i}.SHF" for i in range(1, 7)],
            "I":  [f"I0{i}.DCE"  for i in range(1, 7)],
            "J":  [f"J0{i}.DCE"  for i in range(1, 7)],
            "JM": [f"JM0{i}.DCE" for i in range(1, 7)],
            "HC": [f"HC0{i}.SHF" for i in range(1, 7)],
        },
        "seasonal_pattern": {"strong": [3, 4, 5, 9, 10], "weak": [1, 2, 6, 7, 12]},
        "research_notes": (
            "黑色系高度依赖中国基建+地产政策。产业链利润分配是核心分析框架："
            "矿山利润(铁矿溢价) → 焦化利润 → 钢厂吨钢利润 → 下游终端利润。"
            "铁矿石需重点跟踪四大矿山(淡水河谷/力拓/必和必拓/FMG)发货量。"
            "双焦(焦炭/焦煤)受政策性供给扰动大，煤矿安全检查→焦煤短缺信号。"
        ),
        "correlation_assets": [
            {"name": "钢铁行业PMI", "direction": "positive", "bbg": None},
            {"name": "中国房地产指数", "direction": "positive", "bbg": None},
        ],
    },

    "化工": {
        "icon": "🧪",
        "en_name": "Chemicals",
        "sub_sectors": ["甲醇", "PTA", "乙二醇", "苯乙烯", "聚乙烯", "聚丙烯", "PVC", "尿素"],
        "key_drivers": [
            {"factor": "原油/石脑油价格",  "bbg_ticker": "CO1 Comdty",     "direction": "cost_driver",
             "description": "化工品以原油为原料，油价是成本端核心驱动，油价↑→化工成本端压力"},
            {"factor": "化工品加工利润",   "bbg_ticker": None,              "direction": "bullish_if_wide",
             "description": "成品价格-原料成本=加工利润。利润压缩→装置负荷下降→供应收缩→反弹"},
            {"factor": "装置开工率",       "bbg_ticker": None,              "direction": "bearish_if_rise",
             "description": "下游纺织/包装/农膜等行业开工率，反映终端需求季节性强弱"},
            {"factor": "仓单/库存",        "bbg_ticker": None,              "direction": "bearish_if_high",
             "description": "郑商所/大商所注册仓单是近期供给代理变量，仓单增加→近月承压"},
            {"factor": "国内/海外供需差",  "bbg_ticker": None,              "direction": "varies",
             "description": "中国化工品出口增加→国内供应偏紧→看多；进口增加→供应宽松→看空"},
        ],
        "quant_factors": ["momentum_1m", "momentum_3m", "basis_signal", "oi_trend", "vol_regime"],
        "term_structure": {
            "MA": [f"MA0{i}.CZC" for i in range(1, 7)],
            "TA": [f"TA0{i}.CZC" for i in range(1, 7)],
            "EG": [f"EG0{i}.DCE" for i in range(1, 7)],
            "PP": [f"PP0{i}.DCE" for i in range(1, 7)],
            "L":  [f"L0{i}.DCE"  for i in range(1, 7)],
        },
        "seasonal_pattern": {"strong": [3, 4, 5, 8, 9], "weak": [1, 2, 6, 7]},
        "research_notes": (
            "化工品研究核心：成本-利润-库存三角框架。"
            "油头路线(乙烯/PX)跟踪石脑油裂解价差；煤头路线(甲醇/尿素)跟踪煤炭价格。"
            "PTA研究需对应聚酯产业链: Paraxylene → PTA → PET/聚酯瓶/涤纶。"
            "关注装置检修计划(5月/10月集中)对供应的阶段性扰动。"
        ),
        "correlation_assets": [
            {"name": "布伦特原油", "direction": "positive", "bbg": "CO1 Comdty"},
            {"name": "中国制造PMI", "direction": "positive", "bbg": "CPMINDX Index"},
        ],
    },

    "油脂油料": {
        "icon": "🫒",
        "en_name": "Oilseeds & Vegetable Oils",
        "sub_sectors": ["豆粕", "豆油", "棕榈油", "菜粕", "菜油"],
        "key_drivers": [
            {"factor": "USDA WASDE报告",   "bbg_ticker": None,              "direction": "fundamental",
             "description": "月度全球大豆/菜籽供需平衡表。期末库存/库存消费比是定价锚。每月12日前后发布"},
            {"factor": "南美大豆产量",      "bbg_ticker": None,              "direction": "bearish_if_high",
             "description": "巴西/阿根廷大豆(全球供应~55%)开始收获(2-5月)，产量高→看空豆粕"},
            {"factor": "拉尼娜/厄尔尼诺",   "bbg_ticker": "CENSO Index",     "direction": "varies_by_crop",
             "description": "La Niña→南美干旱→大豆减产看多；El Niño→东南亚棕榈油减产→棕榈看多"},
            {"factor": "马来西亚棕榈油库存", "bbg_ticker": None,              "direction": "bearish_if_high",
             "description": "马来西亚棕榈油局(MPOB)月报库存数据，影响全球植物油定价"},
            {"factor": "美国大豆压榨量",    "bbg_ticker": "NOPA Index",      "direction": "bullish_if_rise",
             "description": "NOPA月度压榨量。压榨旺→豆粕需求旺→紧缩压榨利润Crush Spread"},
            {"factor": "中国生猪利润",      "bbg_ticker": None,              "direction": "bullish_if_profit",
             "description": "生猪养殖利润>0→扩栏→豆粕需求增加。生猪-饲料价差是豆粕需求领先指标"},
        ],
        "quant_factors": ["momentum_1m", "momentum_3m", "basis_signal", "oi_trend", "seasonal_score"],
        "term_structure": {
            "M":  [f"M0{i}.DCE"  for i in range(1, 7)],    # 豆粕
            "Y":  [f"Y0{i}.DCE"  for i in range(1, 7)],    # 豆油
            "P":  [f"P0{i}.DCE"  for i in range(1, 7)],    # 棕榈油
            "OI": [f"OI0{i}.CZC" for i in range(1, 7)],   # 菜油
            "RM": [f"RM0{i}.CZC" for i in range(1, 7)],   # 菜粕
        },
        "seasonal_pattern": {"strong": [5, 6, 7, 8], "weak": [10, 11, 12, 1]},  # 美豆生长季看多
        "research_notes": (
            "油脂油料核心分析：全球供需平衡表(USDA WASDE定价锚) + 天气风险溢价(南美/美国播种/生长)。"
            "豆粕定价: 芝加哥CBOT大豆→压榨利润(Crush Spread) → 豆粕+豆油。"
            "棕榈油有强季节性: 1-6月产量低(旱季)，7-10月产量高。MPOB月报发布日必看。"
            "中国豆粕含蛋白率需求: 生猪/禽类养殖规模是根本需求，注意非洲猪瘟等疫情冲击。"
        ),
        "correlation_assets": [
            {"name": "CBOT大豆", "direction": "positive", "bbg": "S 1 Comdty"},
            {"name": "马来棕榈", "direction": "positive", "bbg": None},
        ],
    },

    "农副产品": {
        "icon": "🐷",
        "en_name": "Livestock & Agri",
        "sub_sectors": ["生猪", "鸡蛋"],
        "key_drivers": [
            {"factor": "能繁母猪存栏",     "bbg_ticker": None, "direction": "bearish_if_high",
             "description": "能繁母猪存栏量领先生猪供应约10个月。历史分位高→未来猪价承压"},
            {"factor": "猪粮比",           "bbg_ticker": None, "direction": "bullish_if_low",
             "description": "猪粮比<6警戒线→养殖亏损→去化产能→未来猪价支撑。比值低是买入信号"},
            {"factor": "供宰栏生猪体重",   "bbg_ticker": None, "direction": "bearish_if_heavy",
             "description": "出栏体重超重→养殖户惜售→后续供应集中释放→价格压力"},
            {"factor": "非洲猪瘟风险",     "bbg_ticker": None, "direction": "bullish_if_outbreak",
             "description": "ASF疫情爆发→供应超预期收缩→短期价格大幅冲高。季节性冬季高发"},
            {"factor": "蛋鸡存栏量",       "bbg_ticker": None, "direction": "bearish_if_high",
             "description": "Mysteel蛋鸡在产存栏量，领先鸡蛋供应约1-2个月"},
        ],
        "quant_factors": ["momentum_1m", "momentum_3m", "basis_signal", "oi_trend", "seasonal_score"],
        "term_structure": {
            "LH": [f"LH0{i}.DCE" for i in range(1, 7)],
            "JD": [f"JD0{i}.DCE" for i in range(1, 7)],
        },
        "seasonal_pattern": {"strong": [12, 1, 2, 3], "weak": [5, 6, 7]},
        "research_notes": (
            "生猪期货核心：猪周期(约3-4年)= 产能扩张→供过于求→亏损去化→供给收缩→价格上涨。"
            "能繁母猪存栏量(10个月领先指标)是最重要的前瞻数据。"
            "鸡蛋具有强季节性: 中秋节/春节前采购高峰→价格季节性上涨；夏季需求淡季→回落。"
        ),
        "correlation_assets": [
            {"name": "玉米(饲料成本)", "direction": "negative", "bbg": "C 1 Comdty"},
            {"name": "豆粕(饲料成本)", "direction": "negative", "bbg": None},
        ],
    },

    "谷物": {
        "icon": "🌾",
        "en_name": "Grains",
        "sub_sectors": ["玉米", "小麦", "大豆", "淀粉", "粳米"],
        "key_drivers": [
            {"factor": "USDA供需平衡表",   "bbg_ticker": None,              "direction": "fundamental",
             "description": "月度全球库存消费比(S/U)。S/U每下降1ppt对应玉米价格约+5-8% (历史经验)"},
            {"factor": "美国播种意向",      "bbg_ticker": None,              "direction": "bearish_if_increase",
             "description": "3月末USDA播种意向报告。玉米/大豆种植面积决定当年供应预期"},
            {"factor": "作物生长评级",      "bbg_ticker": None,              "direction": "bearish_if_good",
             "description": "周度USDA作物生长报告(Good+Excellent%)。评级下降→产量风险→看多"},
            {"factor": "美国出口检验",      "bbg_ticker": None,              "direction": "bullish_if_strong",
             "description": "周度出口检验量/装船量。中国/墨西哥/日本为主要买家，强出口→看多"},
            {"factor": "乙醇政策/需求",     "bbg_ticker": None,              "direction": "bullish_if_mandate",
             "description": "美国玉米40%用于乙醇生产，RFS政策调整影响玉米消耗基准"},
            {"factor": "黑海出口扰动",      "bbg_ticker": None,              "direction": "bullish_if_disrupt",
             "description": "乌克兰/俄罗斯是全球小麦/玉米主要出口国，地缘冲突是主要尾部风险"},
        ],
        "quant_factors": ["momentum_1m", "momentum_3m", "basis_signal", "oi_trend", "seasonal_score"],
        "term_structure": {
            "C":  [f"C0{i}.DCE"  for i in range(1, 7)],    # 玉米
            "CS": [f"CS0{i}.DCE" for i in range(1, 7)],   # 淀粉
        },
        "seasonal_pattern": {"strong": [5, 6, 7], "weak": [10, 11, 12]},  # 美玉米生长季
        "research_notes": (
            "谷物定价锚：USDA WASDE期末库存/消费比(S/U)分位数定价。"
            "玉米S/U每降低1%→历史对应玉米价格约+3-7%。S/U在10-12%区间是历史均衡。"
            "国内玉米供需独立分析：政策托底(收购价)+临储去库+进口依存度+深加工需求。"
            "小麦有强替代效应：小麦大幅便宜于玉米时→饲料端替代→玉米需求受压。"
        ),
        "correlation_assets": [
            {"name": "CBOT玉米", "direction": "positive", "bbg": "C 1 Comdty"},
            {"name": "美元DXY",  "direction": "negative", "bbg": "DXY Curncy"},
        ],
    },

    "软商品": {
        "icon": "🍬",
        "en_name": "Soft Commodities",
        "sub_sectors": ["棉花", "白糖", "苹果", "棉纱"],
        "key_drivers": [
            {"factor": "全球糖类S/U比",     "bbg_ticker": None,              "direction": "bearish_if_high",
             "description": "ISO/LMC全球食糖库存消费比。S/U<20%是历史看多触发点"},
            {"factor": "巴西甘蔗产量",      "bbg_ticker": None,              "direction": "bearish_if_high",
             "description": "巴西(全球~45%出口份额)甘蔗压榨量和制糖/制乙醇比例分配"},
            {"factor": "印度产量+出口政策", "bbg_ticker": None,              "direction": "bearish_if_export",
             "description": "印度出口配额政策变化是白糖最大政策尾部风险"},
            {"factor": "棉花消费/库存",     "bbg_ticker": None,              "direction": "bearish_if_high",
             "description": "中国纺织消费为全球棉花最大需求端，中国国储政策是价格重要支撑"},
        ],
        "quant_factors": ["momentum_1m", "momentum_3m", "basis_signal", "oi_trend", "seasonal_score"],
        "term_structure": {
            "CF": [f"CF0{i}.CZC" for i in range(1, 7)],
            "SR": [f"SR0{i}.CZC" for i in range(1, 7)],
            "AP": [f"AP0{i}.CZC" for i in range(1, 7)],
        },
        "seasonal_pattern": {"strong": [9, 10, 11], "weak": [3, 4, 5]},
        "research_notes": (
            "白糖核心：全球S/U比 + 巴西制糖/制乙醇比例 + 印度出口政策。"
            "棉花：中国国储轮换(抛储/收储)是国内定价关键政策事件，需关注轮储时间窗口。"
        ),
        "correlation_assets": [
            {"name": "原油(乙醇替代)", "direction": "positive", "bbg": "CO1 Comdty"},
            {"name": "巴西雷亚尔",     "direction": "negative", "bbg": "USDBRL Curncy"},
        ],
    },

    "非金属建材": {
        "icon": "🧱",
        "en_name": "Non-metallic Building Materials",
        "sub_sectors": ["玻璃", "纯碱"],
        "key_drivers": [
            {"factor": "房地产竣工面积",    "bbg_ticker": None, "direction": "bullish_if_rise",
             "description": "建筑用平板玻璃与竣工面积强相关，竣工节奏领先玻璃需求约2-3月"},
            {"factor": "玻璃库存",          "bbg_ticker": None, "direction": "bearish_if_high",
             "description": "沙河/华沙等重点贸易商玻璃库存(Mysteel)。高库存压制盘面"},
            {"factor": "光伏玻璃需求",      "bbg_ticker": None, "direction": "structural_bullish",
             "description": "光伏装机量增速推动浮法玻璃结构需求，关注光伏玻璃扩产进度"},
            {"factor": "纯碱下游需求",      "bbg_ticker": None, "direction": "demand_driver",
             "description": "纯碱50%用于玻璃，另30%用于光伏玻璃。光伏景气度影响纯碱需求"},
        ],
        "quant_factors": ["momentum_1m", "momentum_3m", "basis_signal", "oi_trend"],
        "term_structure": {
            "FG": [f"FG0{i}.CZC" for i in range(1, 7)],
            "SA": [f"SA0{i}.CZC" for i in range(1, 7)],
        },
        "seasonal_pattern": {"strong": [3, 4, 5, 9, 10], "weak": [1, 2, 6, 7]},
        "research_notes": (
            "玻璃/纯碱高度依赖房地产竣工周期+光伏装机景气度。"
            "关注生产端冷修/点火产线数量作为供应侧领先指标。"
        ),
        "correlation_assets": [
            {"name": "地产指数", "direction": "positive", "bbg": None},
            {"name": "光伏装机", "direction": "positive", "bbg": None},
        ],
    },

    "航运": {
        "icon": "🚢",
        "en_name": "Shipping",
        "sub_sectors": ["集运指数"],
        "key_drivers": [
            {"factor": "上海出口集运指数SCFI", "bbg_ticker": None,           "direction": "price_reference",
             "description": "SCFI综合指数是集运期货定价基准，期货以SCFI为标的"},
            {"factor": "集装箱运力供应",       "bbg_ticker": None,           "direction": "bearish_if_oversupply",
             "description": "新船交付计划决定运力增速，2024-2025年大量新船交付是长期压制"},
            {"factor": "中国出口增速",          "bbg_ticker": None,           "direction": "bullish_if_strong",
             "description": "中国出口PMI/出口金额同比增速反映货运需求端"},
            {"factor": "红海/苏伊士事件",       "bbg_ticker": None,           "direction": "bullish_if_divert",
             "description": "航线绕行增加航程距离→等效运力下降→推升运费。2024胡塞武装袭击案例"},
        ],
        "quant_factors": ["momentum_1m", "momentum_3m", "basis_signal", "oi_trend"],
        "term_structure": {
            "EC": [f"EC0{i}.INE" for i in range(1, 7)],
        },
        "seasonal_pattern": {"strong": [9, 10, 11], "weak": [1, 2, 3]},
        "research_notes": (
            "集运期货(EC)是2023年上市新品种，以SCFI(上海出口集装箱运价指数)为标的。"
            "运费受供需双重驱动：需求端=中国出口；供应端=全球运力。红海事件类地缘扰动是尾部风险。"
        ),
        "correlation_assets": [
            {"name": "全球贸易量", "direction": "positive", "bbg": None},
            {"name": "波罗的海干散货BDI", "direction": "positive", "bbg": "BDIY Index"},
        ],
    },

    "股": {
        "icon": "📈",
        "en_name": "Equity Index Futures",
        "sub_sectors": ["沪深300", "上证50", "中证500", "中证1000", "标普500", "纳指100", "日经225"],
        "key_drivers": [
            {"factor": "宏观经济预期",      "bbg_ticker": None,              "direction": "bullish_if_growth",
             "description": "GDP增长预期、PMI数据改善→企业盈利预期上调→看多"},
            {"factor": "货币政策",          "bbg_ticker": "FDTRMID Index",  "direction": "bullish_if_cut",
             "description": "降息预期→折现率下降→估值提升→看多。Fed Watch工具跟踪降息概率"},
            {"factor": "估值水平(PE/PB)",   "bbg_ticker": None,              "direction": "mean_revert",
             "description": "A股沪深300/纳斯达克当前PE与历史分位数对比。极端低估是均值回归机会"},
            {"factor": "北向资金流动",      "bbg_ticker": None,              "direction": "bullish_if_inflow",
             "description": "陆股通净买入是境外资金的重要晴雨表，大额流入往往伴随行情回暖"},
            {"factor": "VIX隐含波动率",     "bbg_ticker": "VIX Index",      "direction": "bearish_if_spike",
             "description": "VIX>30是市场恐慌信号，历史上是买入良机而非追空时机"},
        ],
        "quant_factors": ["momentum_1m", "momentum_3m", "momentum_6m", "vol_regime"],
        "term_structure": {
            "IF": [f"IF0{i}.CFE" for i in range(1, 5)],
            "IC": [f"IC0{i}.CFE" for i in range(1, 5)],
            "IM": [f"IM0{i}.CFE" for i in range(1, 5)],
            "IH": [f"IH0{i}.CFE" for i in range(1, 5)],
            "ES": [f"ES{i} Index" for i in range(1, 5)],
            "NQ": [f"NQ{i} Index" for i in range(1, 5)],
        },
        "seasonal_pattern": {"strong": [10, 11, 12, 1], "weak": [5, 6, 9]},
        "research_notes": (
            "股指期货研究框架：宏观/盈利/估值/资金流四维度。"
            "国内股指关注：政策底→情绪底→市场底三阶段。沪深300≈大市值/蓝筹；中证1000≈小盘/成长。"
            "海外股指：美联储政策路径(利率+缩表)是核心宏观变量。AI/科技主题对纳指影响显著。"
        ),
        "correlation_assets": [
            {"name": "VIX隐波",       "direction": "negative", "bbg": "VIX Index"},
            {"name": "美元指数",      "direction": "negative", "bbg": "DXY Curncy"},
            {"name": "10Y美债收益率", "direction": "negative", "bbg": "USGG10YR Index"},
        ],
    },

    "债": {
        "icon": "💰",
        "en_name": "Bond Futures",
        "sub_sectors": ["国债期货10Y", "国债期货5Y", "国债期货2Y", "国债期货30Y"],
        "key_drivers": [
            {"factor": "中央银行政策",      "bbg_ticker": "CNRR7D Index",   "direction": "bullish_if_cut",
             "description": "央行MLF/LPR/OMO利率方向。降息→债券价格↑，加息→债券价格↓(核心驱动)"},
            {"factor": "通货膨胀CPI/PPI",   "bbg_ticker": "CNCPIYOY Index", "direction": "bearish_if_high",
             "description": "通胀压力强→央行收紧货币→债券看空；通缩预期→宽松→债券看多"},
            {"factor": "经济增长预期",      "bbg_ticker": "CPMINDX Index",  "direction": "bearish_if_strong",
             "description": "经济强→央行收紧→债券承压；经济弱→货币宽松预期→债券受益"},
            {"factor": "债券供给压力",      "bbg_ticker": None,              "direction": "bearish_if_supply",
             "description": "特殊债/国债发行计划增加→供给压力→收益率上行→债券价格下跌"},
            {"factor": "资金面(R001/R007)",  "bbg_ticker": "CNRR7D Index",  "direction": "bullish_if_loose",
             "description": "银行间市场资金利率反映货币宽松度，资金面宽松→买债情绪好"},
        ],
        "quant_factors": ["momentum_1m", "momentum_3m", "vol_regime"],
        "term_structure": {
            "T":  [f"T0{i}.CFE"  for i in range(1, 5)],
            "TF": [f"TF0{i}.CFE" for i in range(1, 5)],
            "TL": [f"TL0{i}.CFE" for i in range(1, 5)],
            "TS": [f"TS0{i}.CFE" for i in range(1, 5)],
        },
        "seasonal_pattern": {"strong": [1, 2, 11, 12], "weak": [3, 4, 5]},
        "research_notes": (
            "国债期货定价：IRR(隐含回购利率)套利 + 久期/凸性配置策略。"
            "关键信号：期限利差(10Y-2Y国债)形态；海外债券与国内联动性（汇率约束）。"
            "配置性资金（保险/理财）是长端需求主力，监管政策变化影响需求端。"
        ),
        "correlation_assets": [
            {"name": "央行MLF利率",   "direction": "negative", "bbg": None},
            {"name": "美国10Y国债",   "direction": "positive",  "bbg": "USGG10YR Index"},
        ],
    },
}


# ══════════════════════════════════════════════════════════════════
#  Ticker → Sector 映射 (复用 ticker_mapping 逻辑)
# ══════════════════════════════════════════════════════════════════

# 合约前缀 → 板块 (与 ticker_mapping.py 保持一致)
PREFIX_TO_SECTOR = {
    "AG": "贵金属", "AU": "贵金属", "GC": "贵金属", "SI": "贵金属",
    "CU": "有色金属", "NI": "有色金属", "ZN": "有色金属", "AL": "有色金属",
    "PB": "有色金属", "SN": "有色金属", "SS": "有色金属", "BC": "有色金属", "HG": "有色金属",
    "HC": "焦煤钢矿", "RB": "焦煤钢矿", "I":  "焦煤钢矿", "J":  "焦煤钢矿",
    "JM": "焦煤钢矿", "SF": "焦煤钢矿", "SM": "焦煤钢矿",
    "FG": "非金属建材", "SA": "非金属建材",
    "SC": "能源", "LU": "能源", "NR": "能源", "BU": "能源", "RU": "能源",
    "FU": "能源", "CO": "能源", "CL": "能源", "PG": "能源", "ZC": "能源",
    "EG": "化工", "L":  "化工", "PP": "化工", "EB": "化工", "V":  "化工",
    "MA": "化工", "TA": "化工", "PF": "化工", "UR": "化工", "SP": "化工",
    "M":  "油脂油料", "P":  "油脂油料", "Y":  "油脂油料", "OI": "油脂油料",
    "RM": "油脂油料", "PK": "油脂油料",
    "CF": "软商品", "SR": "软商品", "AP": "软商品", "CJ": "软商品", "CY": "软商品",
    "JD": "农副产品", "LH": "农副产品",
    "C":  "谷物", "A":  "谷物", "B":  "谷物", "CS": "谷物", "RR": "谷物",
    "IC": "股", "IF": "股", "IM": "股", "IH": "股",
    "HSIF": "股", "HHIF": "股", "HTIF": "股",
    "FDAX": "股", "ES": "股", "NQ": "股", "DM": "股", "NK": "股", "NO": "股",
    "T":  "债", "TF": "债", "TL": "债", "TS": "债",
    "FGBL": "债", "FV": "债", "TY": "债", "TU": "债", "US": "债", "JB": "债",
    "EC": "航运",
}


# Bloomberg 月份代码: F=1月 G=2月 H=3月 J=4月 K=5月 M=6月 N=7月 Q=8月 U=9月 V=10月 X=11月 Z=12月
_BBG_MONTH_CODES = set('FGHJKMNQUVXZ')


def _extract_prefix(ticker_str: str) -> str:
    """
    从任意格式的期货 ticker 中提取合约前缀。

    支持格式:
      Bloomberg generic:   "CL1 Comdty" "GC2 Comdty" "ES1 Index"  → CL, GC, ES
      Bloomberg specific:  "CLZ5 Comdty" "HGK6 Comdty" "GCJ26 Comdty" → CL, HG, GC
      Wind domestic:       "CU2501.SHF" "M2505.DCE" "RB2510.SHF" → CU, M, RB
      Wind generic:        "CU01.SHF" "M01.DCE" → CU, M

    策略: 先尝试 generic 解析，若前缀已知则直接返回；否则再尝试
    specific-month 解析来处理 HGK6 这类含月份代码的 ticker。
    """
    import re
    t = ticker_str.strip().upper()

    # 1) Wind domestic: PREFIX + digits + .EXCHANGE
    wm = re.match(r'^([A-Z]{1,4})\d+\.', t)
    if wm:
        return wm.group(1)

    # 2) Bloomberg generic: PREFIX + DIGIT(S) (+ space or end)
    #    e.g. CL1 Comdty, HG1 Comdty, GC2 Comdty, ES1 Index
    bbg_gen = re.match(r'^([A-Z]{1,5})\d', t)
    if bbg_gen:
        prefix = bbg_gen.group(1)
        if prefix in PREFIX_TO_SECTOR:
            return prefix  # Known prefix → use it directly

    # 3) Bloomberg specific-month: PREFIX + MONTH_CODE + YEAR_DIGIT(S)
    #    e.g. HGK6 Comdty (HG + K=May + 2026)
    #    CLZ5 Comdty (CL + Z=Dec + 2025)
    #    Only try this when generic prefix was NOT recognized
    bbg_spec = re.match(r'^([A-Z]{1,4})([A-Z])(\d{1,2})(?:\s|$)', t)
    if bbg_spec and bbg_spec.group(2) in _BBG_MONTH_CODES:
        prefix = bbg_spec.group(1)
        if prefix in PREFIX_TO_SECTOR:
            return prefix

    # 4) Fall back to generic result (even if not in PREFIX_TO_SECTOR)
    if bbg_gen:
        return bbg_gen.group(1)

    # 5) Eurex style: FDAX2603, FGBL2603
    eurex = re.match(r'^(FDAX|FGBL)\d', t)
    if eurex:
        return eurex.group(1)

    return ""


def get_sector_for_ticker(wind_ticker: str) -> str:
    """从 Wind/Bloomberg Ticker 解析板块分类。"""
    if not wind_ticker:
        return "其他"
    prefix = _extract_prefix(wind_ticker)
    return PREFIX_TO_SECTOR.get(prefix, "其他")


def get_contract_prefix(wind_ticker: str) -> str:
    """提取合约前缀，如 'CL1 Comdty' → 'CL', 'HGK6 Comdty' → 'HG', 'M2501.DCE' → 'M'"""
    return _extract_prefix(wind_ticker)


def get_research_config(wind_ticker: str, sector: Optional[str] = None) -> Dict[str, Any]:
    """获取品种专属研究配置，找不到时返回通用配置。"""
    if not sector:
        sector = get_sector_for_ticker(wind_ticker)
    config = SECTOR_RESEARCH_CONFIG.get(sector)
    if not config:
        return {
            "sector": "其他", "icon": "📦", "en_name": "Other",
            "key_drivers": [], "quant_factors": ["momentum_1m", "momentum_3m", "oi_trend"],
            "term_structure": {}, "seasonal_pattern": {"strong": [], "weak": []},
            "research_notes": "暂无专属研究框架。",
            "correlation_assets": [],
        }
    return {**config, "sector": sector}


def get_term_structure_tickers(wind_ticker: str, n_months: int = 6) -> List[str]:
    """根据合约前缀返回期限结构合约列表(M1-Mn)。"""
    prefix = get_contract_prefix(wind_ticker)
    config = SECTOR_RESEARCH_CONFIG.get(get_sector_for_ticker(wind_ticker), {})
    ts_map = config.get("term_structure", {})
    tickers = ts_map.get(prefix, [])
    return tickers[:n_months]


# ══════════════════════════════════════════════════════════════════
#  量化因子计算
#  输入: ohlcv (list of dicts), fundamentals (dict from market_data)
#  输出: scored factor dict
# ══════════════════════════════════════════════════════════════════

def _safe_float(v):
    try:
        f = float(v)
        return f if (not math.isnan(f) and not math.isinf(f)) else None
    except (TypeError, ValueError):
        return None


def calc_quant_factors(ohlcv: List[Dict], fundamentals: Dict) -> Dict[str, Any]:
    """
    计算各量化因子的数值和评分 (score: -1=看空, 0=中性, +1=看多)。

    Returns:
        {
          "momentum_1m":  {"value": 0.032, "score": 1, "label": "3.20%", "desc": "..."},
          "momentum_3m":  {...},
          "momentum_6m":  {...},
          "basis_signal": {...},
          "oi_trend":     {...},
          "vol_regime":   {...},
          "roll_yield":   {...},
          "seasonal_score": {...},
        }
    """
    factors = {}
    closes = [d["close"] for d in ohlcv if d.get("close") is not None]

    # ── Momentum ─────────────────────────────────────────────────
    def momentum_factor(n_days: int, key: str, label: str):
        if len(closes) >= n_days + 1:
            ret = (closes[-1] / closes[-n_days - 1]) - 1.0
            score = 1 if ret > 0.01 else (-1 if ret < -0.01 else 0)
            # Strong signals
            if ret > 0.05: score = 2
            if ret < -0.05: score = -2
            factors[key] = {
                "value": round(ret * 100, 2), "score": score,
                "label": f"{'+' if ret >= 0 else ''}{ret*100:.2f}%",
                "desc": f"{label}价格回报率",
                "unit": "%"
            }
        else:
            factors[key] = {"value": None, "score": 0, "label": "N/A", "desc": label, "unit": "%"}

    momentum_factor(20,  "momentum_1m", "近1个月")
    momentum_factor(60,  "momentum_3m", "近3个月")
    momentum_factor(120, "momentum_6m", "近6个月")

    # ── Basis Signal ──────────────────────────────────────────────
    basis_data = fundamentals.get("basis", [])
    if basis_data:
        values = [b["value"] for b in basis_data if b.get("value") is not None]
        if values:
            latest_basis = values[-1]
            price_ref = closes[-1] if closes else 1.0
            basis_pct = (latest_basis / price_ref * 100) if price_ref else 0
            # Backwardation(正基差) = 现货贵 = 库存紧张 = 看多信号
            score = 1 if latest_basis > 0 else (-1 if latest_basis < 0 else 0)
            struct = "Backwardation(近高远低)" if latest_basis > 0 else "Contango(近低远高)"
            factors["basis_signal"] = {
                "value": round(latest_basis, 4), "score": score,
                "label": f"{'+' if latest_basis >= 0 else ''}{latest_basis:.2f}",
                "desc": f"基差={struct}",
                "unit": ""
            }
    if "basis_signal" not in factors:
        factors["basis_signal"] = {"value": None, "score": 0, "label": "N/A", "desc": "基差数据不可用", "unit": ""}

    # ── OI Trend ─────────────────────────────────────────────────
    oi_data = fundamentals.get("open_interest", [])
    if oi_data and len(oi_data) >= 5:
        oi_vals = [d["value"] for d in oi_data[-10:] if d.get("value") is not None]
        if len(oi_vals) >= 5:
            oi_change = (oi_vals[-1] - oi_vals[-5]) / oi_vals[-5] * 100 if oi_vals[-5] else 0
            score = 1 if oi_change > 2 else (-1 if oi_change < -2 else 0)
            factors["oi_trend"] = {
                "value": round(oi_change, 2), "score": score,
                "label": f"{'+' if oi_change >= 0 else ''}{oi_change:.2f}%",
                "desc": f"近5日持仓量变化{'↑增仓' if oi_change > 0 else '↓减仓' if oi_change < 0 else '持平'}",
                "unit": "%"
            }
    if "oi_trend" not in factors:
        factors["oi_trend"] = {"value": None, "score": 0, "label": "N/A", "desc": "持仓量数据不可用", "unit": ""}

    # ── Volatility Regime ─────────────────────────────────────────
    if len(closes) >= 60:
        def realized_vol(n):
            returns = [(closes[i] / closes[i-1] - 1) for i in range(1, n+1) if closes[i-1]]
            if not returns: return None
            mean = sum(returns) / len(returns)
            var = sum((r - mean) ** 2 for r in returns) / len(returns)
            return math.sqrt(var * 252) * 100  # annualized %

        rv20 = realized_vol(20)
        rv60 = realized_vol(60)
        if rv20 and rv60:
            vol_ratio = rv20 / rv60
            score = -1 if vol_ratio > 1.3 else (1 if vol_ratio < 0.7 else 0)
            regime = "高波动扩张" if vol_ratio > 1.3 else ("低波动压缩" if vol_ratio < 0.7 else "正常波动")
            factors["vol_regime"] = {
                "value": round(rv20, 2), "score": score,
                "label": f"{rv20:.1f}% (ann.)",
                "desc": f"20D年化波动率 {regime} (60D={rv60:.1f}%)",
                "unit": "%"
            }
    if "vol_regime" not in factors:
        factors["vol_regime"] = {"value": None, "score": 0, "label": "N/A", "desc": "波动率数据不足", "unit": ""}

    # ── Roll Yield Proxy ──────────────────────────────────────────
    # 用基差估算 Roll Yield: 若大贴水则roll yield为负(多头亏损)
    if factors.get("basis_signal", {}).get("value") is not None and closes:
        basis_val = factors["basis_signal"]["value"]
        price = closes[-1] if closes[-1] else 1
        # Annual roll yield estimate = -12 × (basis / price)  (monthly approximation)
        annual_roll = -(basis_val / price) * 12 * 100 if price else 0
        factors["roll_yield"] = {
            "value": round(annual_roll, 2), "score": 1 if annual_roll > 0 else (-1 if annual_roll < -3 else 0),
            "label": f"{'+' if annual_roll >= 0 else ''}{annual_roll:.2f}%",
            "desc": f"估算年化展期收益率 (基于基差推算)",
            "unit": "%"
        }

    # ── Seasonal Score ────────────────────────────────────────────
    current_month = datetime.now().month
    factors["seasonal_score"] = {
        "value": current_month, "score": 0,
        "label": f"{current_month}月",
        "desc": "季节性评分需结合品种配置",  # Will be updated by caller with sector config
        "unit": ""
    }

    return factors


def enrich_seasonal_score(factors: Dict, sector_config: Dict) -> None:
    """根据板块配置更新季节性因子评分。"""
    current_month = datetime.now().month
    strong_months = sector_config.get("seasonal_pattern", {}).get("strong", [])
    weak_months   = sector_config.get("seasonal_pattern", {}).get("weak", [])
    if current_month in strong_months:
        status = "当前处于季节性强势月份"
        score = 1
    elif current_month in weak_months:
        status = "当前处于季节性弱势月份"
        score = -1
    else:
        status = "季节性中性月份"
        score = 0
    strong_str = "/".join(str(m) + "月" for m in strong_months) if strong_months else "无"
    weak_str   = "/".join(str(m) + "月" for m in weak_months)   if weak_months   else "无"
    factors["seasonal_score"] = {
        "value": current_month, "score": score,
        "label": f"{current_month}月",
        "desc": f"{status}｜强势月:{strong_str}",
        "unit": ""
    }


def calc_composite_score(factors: Dict, quant_factor_keys: List[str]) -> Dict[str, Any]:
    """
    计算综合信号评分。
    Returns: {"total": 0.4, "signal": "中性偏多", "color": "yellow", "bars": 3}
    """
    scores = []
    for key in quant_factor_keys:
        if key in factors and factors[key].get("value") is not None:
            s = factors[key].get("score", 0)
            # Clamp to [-2, 2]
            scores.append(max(-2, min(2, s)))

    if not scores:
        return {"total": 0, "signal": "数据不足", "color": "gray", "bars": 2}

    avg = sum(scores) / len(scores)

    if avg >= 1.2:
        return {"total": avg, "signal": "强烈看多", "color": "green",      "bars": 5}
    elif avg >= 0.4:
        return {"total": avg, "signal": "温和看多", "color": "lightgreen",  "bars": 4}
    elif avg >= -0.4:
        return {"total": avg, "signal": "中性",     "color": "yellow",     "bars": 3}
    elif avg >= -1.2:
        return {"total": avg, "signal": "温和看空", "color": "orange",     "bars": 2}
    else:
        return {"total": avg, "signal": "强烈看空", "color": "red",        "bars": 1}


def build_research_panel(wind_ticker: str, sector: Optional[str] = None,
                         ohlcv: Optional[List] = None,
                         fundamentals: Optional[Dict] = None) -> Dict[str, Any]:
    """
    构建完整研究面板数据，供 /api/research-factors 接口调用。

    Returns完整的研究面板数据结构，交给前端渲染。
    """
    ohlcv = ohlcv or []
    fundamentals = fundamentals or {}

    sector_config = get_research_config(wind_ticker, sector)
    prefix = get_contract_prefix(wind_ticker)

    # Quantitative factors
    factors = calc_quant_factors(ohlcv, fundamentals)
    enrich_seasonal_score(factors, sector_config)

    # Composite score
    composite = calc_composite_score(factors, sector_config.get("quant_factors", []))

    # Term structure tickers
    ts_tickers = get_term_structure_tickers(wind_ticker)

    # ── Data quality assessment ──
    has_ohlcv = len(ohlcv) > 10
    has_basis = bool(fundamentals.get("basis"))
    has_oi = bool(fundamentals.get("open_interest"))
    real_factor_count = sum(
        1 for k in sector_config.get("quant_factors", [])
        if factors.get(k, {}).get("value") is not None
    )
    total_factor_count = len(sector_config.get("quant_factors", []))

    if has_ohlcv and real_factor_count >= total_factor_count * 0.7:
        data_quality = "good"
        data_quality_label = "数据充足 — 因子基于真实行情计算"
    elif has_ohlcv:
        data_quality = "partial"
        data_quality_label = "部分数据可用 — 部分因子可能为N/A（基差/持仓量缺失）"
    else:
        data_quality = "poor"
        data_quality_label = "行情数据不可用 — 所有因子为N/A（请检查数据源连接）"

    return {
        "sector": sector_config.get("sector", "其他"),
        "icon": sector_config.get("icon", "📦"),
        "en_name": sector_config.get("en_name", ""),
        "prefix": prefix,
        "key_drivers": sector_config.get("key_drivers", []),
        "quant_factors": {k: factors.get(k, {}) for k in sector_config.get("quant_factors", [])},
        "composite": composite,
        "term_structure_tickers": ts_tickers,
        "research_notes": sector_config.get("research_notes", ""),
        "correlation_assets": sector_config.get("correlation_assets", []),
        "seasonal_pattern": sector_config.get("seasonal_pattern", {}),
        "current_month": datetime.now().month,
        "data_quality": data_quality,
        "data_quality_label": data_quality_label,
        "ohlcv_count": len(ohlcv),
    }
