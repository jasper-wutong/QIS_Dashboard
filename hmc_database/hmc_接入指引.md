# HMC 历史行情中心 — 接入指引

> 最后更新: 2026-03-11 | 账号: `g_eq_eds_flow` | 环境: `PROD_OFFLINE` / `TRAIN`

---

## 1. 概述

历史行情中心 (HMC) 是中金公司内部的 DDB（DolphinDB）数据平台，提供：
- **日频行情** (EOD)：国内期货、全球期货、股票、债券、基金、指数、期权
- **分钟行情** (1/3/5/15/30/60min)：国内期货、期权（截至 2025-07）
- **Tick 行情** (LV1/LV2)：国内/海外期货、股票、债券、基金（截至 2025-07）
- **静态信息**：合约主信息、基础信息、交易日历等
- 通过 gRPC SDK 接入，支持 **SQL 脚本执行** 和 **标准化查询** 两种方式

---

## 2. 安装 SDK

```bash
# 使用公司 pip 源（需要绕过外网代理）
pip install hmc_sdk==1.1.5.rc0 \
  -i https://repo.cicc.com.cn/artifactory/api/pypi/public-pypi-virtual/simple \
  --trusted-host repo.cicc.com.cn
```

如果 pandas 版本冲突（Python 3.14 只有 pandas 3.x），可分步安装：
```bash
pip install hmc_sdk==1.1.5.rc0 --no-deps \
  -i https://repo.cicc.com.cn/artifactory/api/pypi/public-pypi-virtual/simple \
  --trusted-host repo.cicc.com.cn

pip install "grpcio>=1.62.3" "protobuf>=4.0.0" "deprecated>=1.2.1" \
  -i https://repo.cicc.com.cn/artifactory/api/pypi/public-pypi-virtual/simple \
  --trusted-host repo.cicc.com.cn
```

---

## 3. 认证配置

**凭据** (三选一，优先级从高到低)：

| 方式 | 说明 |
|------|------|
| 环境变量 | `HMC_APP_ID` + `HMC_TOKEN` |
| 项目文件 | `hmc_token.txt`（第1行=app_id, 第2行=token） |
| 默认路径 | `D:\token.txt`（SDK 默认，同上格式） |

当前账号：
```
用户名 (app_id): g_eq_eds_flow
密码   (token):  bf'4J#Fo<;"J
```

**网络配置**：HMC 走内网直连，需绕过代理：

```powershell
# PowerShell 方式
$env:NO_PROXY = "hmc-dev.cicc.com,hmc-prod.cicc.com,10.50.*,localhost,127.0.0.1"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
```

> `hmc_helper.py` 已内置自动绕过代理逻辑，无需手动设置。

---

## 4. 环境与集群

| 参数 | 值 | 说明 |
|------|----|------|
| environment | **PROD_OFFLINE** | 测试网段访问生产环境（推荐） |
| cluster | **TRAIN** | 训练集群（推荐） |

其他环境（备用）：
- `PROD` — 生产环境（需生产网段）
- `DEV` — 测试环境
- `REALTIME` — 实时集群

---

## 5. 使用 hmc_helper.py (CLI)

### 5.1 连接测试

```bash
python hmc_helper.py --mode test
# {"ok": true, "message": "HMC 连接测试成功", "environment": "PROD_OFFLINE", "cluster": "TRAIN"}
```

### 5.2 SQL 查询 (最常用)

```bash
# 查国内期货日行情 — HC (热轧卷板) 
python hmc_helper.py --mode query \
  --sql "select SEC_ID, TRAN_DATE, OPEN_PRC, HIGH_PRC, LOW_PRC, CLOSE_PRC, STTM_PRC, TX_QTY, VOHP, BASIS \
         from loadTable('dfs://HQUOT_CENTER_EOD', 'FUT_DAY_QUOT') \
         where TRAN_DATE = 2026.03.10 and SEC_ID like 'HC%'"

# 查全球期货 — CL (原油)
python hmc_helper.py --mode query \
  --sql "select BLG_CODE, TRADE_DT, OPEN_PRC, HIGH_PRC, LOW_PRC, CLOSE_PRC, PX_LAST, TX_QTY, VOHP \
         from loadTable('dfs://HQUOT_CENTER_EOD', 'GLB_FUT_DAY_QUOT') \
         where TRADE_DT >= 2026.03.01 and BLG_CODE like 'CL%' \
         order by TRADE_DT desc"

# 查分钟日内行情 (数据截至 2025-07-29)
python hmc_helper.py --mode query \
  --sql "select top 10 CONTRACTID, TRDDATE, BARTIME, OPENPRICE, HIGHPRICE, LOWPRICE, CLOSEPRICE, VOLUME, OPENINTS \
         from loadTable('dfs://HQUOT_CENTER_DT', 'CH_FUTURE_MINUTE_LV1') \
         where TRDDATE = 2025.07.29 and CONTRACTID = 'IC2508'"
```

### 5.3 查看元数据

```bash
# 列出所有数据库
python hmc_helper.py --mode metadata --action databases

# 列出某库所有表
python hmc_helper.py --mode metadata --action tables --db "dfs://HQUOT_CENTER_EOD"

# 列出某表所有列
python hmc_helper.py --mode metadata --action columns --db "dfs://HQUOT_CENTER_EOD" --table "FUT_DAY_QUOT"
```

### 5.4 标准化查询 (期货)

```bash
# 期货日行情
python hmc_helper.py --ticker "HC2505" --start 2026-01-01 --end 2026-03-10

# 合约列表
python hmc_helper.py --mode futures_list --exchange SHFE
```

---

## 6. Python 代码示例

### 6.1 SQL 脚本执行

```python
from hmc_sdk.client.hmc_rpc_client import HmcRpcClient
from hmc_sdk.config.cluster import Cluster
from hmc_sdk.config.environment import Environment
from hmc_sdk.config.hmc_config import HmcConfig
from hmc_sdk.client.database.script_client import ScriptClient

# 初始化客户端
config = HmcConfig(
    environment=Environment.PROD_OFFLINE,
    cluster=Cluster.TRAIN,
    app_id="g_eq_eds_flow",
    token="bf'4J#Fo<;\"J",
    query_timeout_sec=60,
)
client = HmcRpcClient(config)
client.initialize()

# 建立脚本客户端
script_client = ScriptClient(client)

# 执行查询
sql = """
select SEC_ID, TRAN_DATE, OPEN_PRC, HIGH_PRC, LOW_PRC, CLOSE_PRC, TX_QTY, VOHP
from loadTable('dfs://HQUOT_CENTER_EOD', 'FUT_DAY_QUOT')
where TRAN_DATE = 2026.03.10 and SEC_ID like 'HC%'
"""
reply = script_client.run_script(sql, timeout_sec=60)
df = script_client.parse_result(reply)
print(df)

# 关闭
client.close()
```

### 6.2 标准化期货查询

```python
from hmc_sdk.client.standard.futures_client import FuturesClient
from hmc_sdk.model.data.data_pb2 import DataForm, Area

futures_client = FuturesClient(client)  # client 同上已初始化

data_list = futures_client.get_futures_time_range(
    code_list=["ESA Index", "NQA Index", "CL1 Comdty"],
    data_form=DataForm.EOD,
    start_time="2025.09.01",
    end_time="2026.03.11",
)
```

---

## 7. 数据库与表一览

### 7.1 五大数据库

| 数据库 | 说明 |
|--------|------|
| `dfs://HQUOT_CENTER_EOD` | **日频行情** (最常用) |
| `dfs://HQUOT_CENTER` | **Tick/快照行情** (LV1/LV2) |
| `dfs://HQUOT_CENTER_DT` | **分钟 K 线** (1/3/5/15/30/60min) |
| `dfs://ODSDP` | **ODS 数据** (DATAYES 源、交易所 L2 等) |
| `dfs://ODSDP_DT` | ODS 数据 (DT 衍生) |

### 7.2 EOD 日频表 (`dfs://HQUOT_CENTER_EOD`)

| 表名 | 日期列 | 代码列 | 说明 |
|------|--------|--------|------|
| **FUT_DAY_QUOT** | `TRAN_DATE` | `SEC_ID` | 国内期货日行情 ✅ |
| **GLB_FUT_DAY_QUOT** | `TRADE_DT` | `BLG_CODE` | 全球期货日行情 (彭博代码) ✅ |
| **GLB_FUT_MSTR** | — | — | 全球期货主信息 |
| **FUT_BSC_INFO** | — | — | 期货基础信息 |
| **FUT_CONTI_MAP** | — | — | 连续合约映射 |
| **OPT_DAY_QUOT** | — | — | 期权日行情 |
| **OPT_GEN_INFO** | — | — | 期权基础信息 |
| **STK_DAY_QUOT** | — | — | 股票日行情 |
| **GLB_STK_DAY_QUOT** | — | — | 全球股票日行情 |
| **CH_INDEX_DAY_QUOT** | — | — | 国内指数日行情 |
| **GLB_INDEX_DAY_QUOT** | — | — | 全球指数日行情 |
| **BOND_QUOT_FULL_PRC** | — | — | 债券全价行情 |
| **BOND_QUOT_NET_PRC** | — | — | 债券净价行情 |
| **IBOR_RATE_QUOT** | — | — | IBOR 利率 |
| **CNBD_BOND_YIELD_CURV** | — | — | 中债收益率曲线 |
| **CNBD_VALUE** | — | — | 中债估值 |
| **FUND_DAY_QUOT_EXT** | — | — | 基金日行情 |
| **GLB_TRAN_CALD** | — | — | 全球交易日历 |

### 7.3 国内期货日行情列 (FUT_DAY_QUOT)

| 列名 | 类型 | 说明 |
|------|------|------|
| `SEC_ID` | SYMBOL | 证券ID (如 `HC.SHF`, `A.DCE`) |
| `TRAN_DATE` | DATE | 交易日期 |
| `BEF_STTM_PRC` | DOUBLE | 前结算价 |
| `OPEN_PRC` | DOUBLE | 开盘价 |
| `HIGH_PRC` | DOUBLE | 最高价 |
| `LOW_PRC` | DOUBLE | 最低价 |
| `CLOSE_PRC` | DOUBLE | 收盘价 |
| `STTM_PRC` | DOUBLE | 结算价 |
| `TX_QTY` | DOUBLE | 成交量 |
| `TX_AMT` | DOUBLE | 成交金额 |
| `VOHP` | DOUBLE | 持仓量 |
| `CHNG` | DOUBLE | 涨跌 |
| `BASIS` | DOUBLE | 基差 |
| `CONTI_INDEN_CD` | SYMBOL | 连续合约代码 |

### 7.4 全球期货日行情列 (GLB_FUT_DAY_QUOT)

| 列名 | 类型 | 说明 |
|------|------|------|
| `BLG_CODE` | SYMBOL | 彭博代码 (如 `CL1 Comdty`) |
| `TRADE_DT` | DATE | 交易日期 |
| `OPEN_PRC` | DOUBLE | 开盘价 |
| `HIGH_PRC` | DOUBLE | 最高价 |
| `LOW_PRC` | DOUBLE | 最低价 |
| `CLOSE_PRC` | DOUBLE | 收盘价 |
| `PX_LAST` | DOUBLE | 收盘价/结算价 |
| `STTM_PRC` | DOUBLE | 结算价 |
| `TX_QTY` | DOUBLE | 成交量 |
| `VOHP` | DOUBLE | 持仓量 |
| `AVGD_VOL_D30` | DOUBLE | 30天日均成交量 |

### 7.5 分钟 K 线表 (`dfs://HQUOT_CENTER_DT`)

| 表名 | 频率 | 说明 |
|------|------|------|
| `CH_FUTURE_MINUTE_LV1` | 1 分钟 | 国内期货 |
| `CH_FUTURE_MINUTE03_LV1` | 3 分钟 | 国内期货 |
| `CH_FUTURE_MINUTE05_LV1` | 5 分钟 | 国内期货 |
| `CH_FUTURE_MINUTE15_LV1` | 15 分钟 | 国内期货 |
| `CH_FUTURE_MINUTE30_LV1` | 30 分钟 | 国内期货 |
| `CH_FUTURE_MINUTE60_LV1` | 60 分钟 | 国内期货 |
| `CH_OPTION_MINUTE_LV1` ~ `60` | 1~60 分钟 | 国内期权 |

**分钟 K 线列结构：**

| 列名 | 类型 | 说明 |
|------|------|------|
| `CONTRACTID` | STRING | 合约代码 (如 `IC2508`) |
| `TRDDATE` | DATE | 交易日期 |
| `BARTIME` | MINUTE | 分钟时间 (如 `09:31m`) |
| `OPENPRICE` | DOUBLE | 开盘价 |
| `HIGHPRICE` | DOUBLE | 最高价 |
| `LOWPRICE` | DOUBLE | 最低价 |
| `CLOSEPRICE` | DOUBLE | 收盘价 |
| `VOLUME` | DOUBLE | 成交量 |
| `VALUE` | DOUBLE | 成交金额 |
| `VWAP` | DOUBLE | 分钟均价 |
| `OPENINTS` | DOUBLE | 持仓量 |

> ⚠️ **分钟数据可用性**：分钟聚合管道停止于 **2025-07-29**，Tick 原始数据正常，但分钟 K 线未继续合成入库。如需最新分钟数据，请联系 HMC 管理员 (魏永长 / 安志翔) 确认数据订阅状态，或自行从 Tick 数据聚合。

### 7.6 Tick/快照表 (`dfs://HQUOT_CENTER`)

| 表名 | 说明 |
|------|------|
| `CH_FUTURE_QUOTATION_LV1` | 国内期货 LV1 行情 (含买卖盘) |
| `CH_FUTURE_SNAPSHOT_LV2` | 国内期货 LV2 快照 |
| `OS_FUTURE_QUOTATION_LV1` | 海外期货 LV1 行情 |
| `CH_STK_QUOTATION_LV1` / `SNAPSHOT_LV2` | A 股 LV1/L2 |
| `CH_BOND_QUOTATION_LV1` / `MINUTE_LV2` | 债券 |
| `CH_INDEX_QUOTATION_LV1` / `MINUTE_LV2` | 指数 |
| `CH_OPTION_SNAPSHOT_LV2` | 期权 L2 |
| `HK_SEC_ORDER_BOOK` | 港股委托簿 |

> ✅ Tick 和 Snapshot 数据**实时更新**（LV1 至 2026-03-11，LV2 至 2026-03-10），EOD 和 Tick 均权限完整。

---

## 8. 数据可用性总结

| 数据类型 | 可用 | 最新日期 | 备注 |
|----------|------|----------|------|
| 国内期货 EOD | ✅ | **2026-03-10** | 实时更新 |
| 全球期货 EOD | ✅ | **2026-03-07** | 实时更新 |
| 股票/指数/债券 EOD | ✅ | 最近 | 实时更新 |
| 国内期货 LV1 Tick | ✅ | **2026-03-11** | 实时入库，原始逐笔行情 |
| 国内期货 LV2 Snapshot | ✅ | **2026-03-10** | 实时入库，含10档买卖盘 |
| 海外期货 LV1 Tick | ✅ | **2026-03-11** | 实时入库 |
| 国内期货分钟线 (1/3/5/15/30/60min) | ⚠️ | **2025-07-29** | 分钟聚合管道停止，Tick 正常但分钟未合成 |
| DATAYES 分钟线 (ODSDP) | ⚠️ | **2025-07-29** | 同上 |

---

## 9. 常见查询示例

### 查热轧卷板 (HC) 最近 N 天
```sql
select SEC_ID, TRAN_DATE, OPEN_PRC, HIGH_PRC, LOW_PRC, CLOSE_PRC, STTM_PRC, TX_QTY, VOHP, BASIS
from loadTable('dfs://HQUOT_CENTER_EOD', 'FUT_DAY_QUOT')
where TRAN_DATE >= 2026.03.01 and SEC_ID like 'HC%'
order by TRAN_DATE desc
```

### 查原油 (CL) 全球期货
```sql
select BLG_CODE, TRADE_DT, PX_LAST, TX_QTY, VOHP
from loadTable('dfs://HQUOT_CENTER_EOD', 'GLB_FUT_DAY_QUOT')
where TRADE_DT >= 2026.01.01 and BLG_CODE like 'CL%'
order by TRADE_DT desc
```

### 查连续合约映射
```sql
select * from loadTable('dfs://HQUOT_CENTER_EOD', 'FUT_CONTI_MAP')
where SEC_ID = 'HC.SHF'
```

### 查交易日历
```sql
select top 30 * from loadTable('dfs://HQUOT_CENTER_EOD', 'GLB_TRAN_CALD')
```

### 查期权日行情
```sql
select top 10 * from loadTable('dfs://HQUOT_CENTER_EOD', 'OPT_DAY_QUOT')
where TRAN_DATE = 2026.03.10
```

### 查 Tick 行情 (LV1，逐笔，实时更新)
```sql
-- 查 IC2506 今日 Tick (日期格式: 整数 20260311)
select top 200 strCode, iTrdDate, iTime, fLast, fBidPrice, fAskPrice, iVolume, iOpenInterest
from loadTable('dfs://HQUOT_CENTER', 'CH_FUTURE_QUOTATION_LV1')
where iTrdDate = 20260311 and strCode = 'IC2506'
order by iTime asc
```

### 查 LV2 快照 (含10档买卖，实时更新)
```sql
-- 查 HC2506 今日 LV2 快照 (日期格式: DATE)
select top 200 InstruID, TrdDay, TrdTS, LastPrice, Volume, OpenInt,
       BidPrice[0] as Bid1, AskPrice[0] as Ask1
from loadTable('dfs://HQUOT_CENTER', 'CH_FUTURE_SNAPSHOT_LV2')
where TrdDay = 2026.03.10 and InstruID = 'HC2506'
order by TrdTS asc
```

---

## 11. 非价格数据一览

除行情之外，HMC 还提供大量参考、基本面和衍生数据，全部在 `dfs://HQUOT_CENTER_EOD`。

### 11.1 股票参考与估值

#### STK_GEN_INFO — 股票基本信息 (74列)
| 列名 | 说明 |
|------|------|
| `SEC_ID` / `SYMBOL` | 证券ID / 代码 |
| `CNAME` / `CSNAME` | 中文全称 / 简称 |
| `ENAME` / `ESNAME` | 英文全称 / 简称 |
| `ISIN` | ISIN代码 |
| `LIST_DATE` / `DLIST_DATE` | 上市 / 退市日 |
| `TRAN_MKT` / `LIST_BOAR_CD` | 交易所 / 板块 |
| `IS_SHSC` / `IS_HKSC` | 沪深港通标识 |
| `IS_H` / `IS_AH` | H股 / AH股 |
| `STK_TYPE` / `DUR_STS` | 股票类型 / 当前状态 |
| `IS_MARGIN` / `IS_BACKDOOR_LIST` | 融资融券 / 借壳标识 |
| `IS_CNS_500` / `IS_SHSZ_300` | 是否中证500 / 沪深300成分 |
| `SW_LV1/2/3_INDT_CLSF_CD/NAME` | 申万一/二/三级行业 |
| `WIND_LV1/2/3/4_INDT_CLSF_CD/NAME` | Wind一/二/三/四级行业 |
| `CSRC_LV1/2/3/4_INDT_CLSF_CD/NAME` | 证监会一/二/三/四级行业 |
| `HS_LVL1/2/3_INDT_CLSF_CD/NAME` | 恒生一/二/三级行业 |
| `UNITPERLOT` / `MIN_PRC_CHG_UNIT` | 每手股数 / 最小价格变动 |

#### STK_DAY_QUOT_VAL — 股票日估值 (50列)
| 列名 | 说明 |
|------|------|
| `VAL_MV` / `DQ_MV` | 总市值 / 流通市值 |
| `VAL_PE` / `VAL_PE_TTM` | 市盈率 / 市盈率TTM |
| `VAL_PB_NEW` | 市净率 |
| `VAL_PCF_OCF` / `VAL_PCF_OCFTTM` | 市现率(经营) / TTM |
| `VAL_PS` / `VAL_PS_TTM` | 市销率 / TTM |
| `PRICE_DIV_DPS` | 股价/每股股息 |
| `DQ_TURN` / `DQ_FREETURNOVER` | 换手率 / 自由换手率 |
| `TOT_SHR_TODAY` / `FLOAT_A_SHR_TODAY` | 总股本 / 流通A股 |
| `FREE_SHARES_TODAY` | 自由流通股本 |
| `NET_PROFIT_PARENT_COMP_TTM/LYR` | 归母净利润 TTM/LYR |
| `NET_ASSETS_TODAY` | 净资产 |
| `OPER_REV_TTM/LYR` | 营业收入 TTM/LYR |
| `PQ_HIGH_52W` / `PQ_LOW_52W` | 52周高/低价 |
| `AVGD_MVAL_D40` | 40日均流通市值 |

### 11.2 期货合约参考

#### FUT_BSC_INFO — 期货基础信息 (33列)
| 列名 | 说明 |
|------|------|
| `SEC_ID` / `SYMBOL` | 合约ID / 代码 |
| `STD_INDEN_CD` | 标准合约代码 |
| `TRAN_MKT` | 交易所 |
| `LIST_DATE` / `FINAL_TRAN_DATE` | 上市 / 最后交易日 |
| `FINAL_DLV_DAY` / `STTM_DAY` | 最后交割日 / 结算日 |
| `DLV_MONTH` | 交割月份 |
| `LSTG_STANDARD_PRC` | 挂牌基准价 |
| `IS_TREA_FUT` / `IS_MERC_FUT` / `IS_INDEX_FUT` | 国债/商品/股指期货标识 |
| `CURR_CD` | 货币 |

#### FUT_STD_INDEN_ATTR — 期货标准合约属性 (45列)
| 列名 | 说明 |
|------|------|
| `STD_INDEN_CD` / `STD_INDEN_NAME` | 标准合约代码 / 名称 |
| `TRAN_MEAS_UNIT` / `TRAN_UNIT_PERU` | 交易计量单位 / 每手 |
| `INDEN_MULT` | 合约乘数 |
| `MIN_CHG_PRC` | 最小报价单位 |
| `MINER_TRAN_MARG` / `MIN_TRAN_MARG_STD` | 最低保证金率 |
| `FIRST_TRAN_MARG` | 首次交易保证金 |
| `TRAN_COMM_FEE` / `DLV_COMM_FEE` | 交易/交割手续费 |
| `INDEN_MONTH_COMNT` | 合约月份说明 |
| `DLV_MODE_COMNT` / `DLV_ADDR_COMNT` | 交割方式/地点说明 |
| `POS_LIMIT_COMNT` | 持仓限额说明 |
| `FUTUTRES_TYPE_CD` / `VAR_SUBCA_CD/NAME` | 期货品类一/二级分类 |

#### FUT_CONTI_MAP — 连续合约映射 (13列)
每个滚动合约（如主力/近月连续）在每个时间段映射到的实际合约，含 `BEGIN_DT` / `END_DATE` 有效区间。

### 11.3 期权合约参考

#### OPT_GEN_INFO — 期权基本信息 (31列)
| 列名 | 说明 |
|------|------|
| `SEC_ID` / `OPT_CTRCT_ID` | 期权ID / 标准合约ID |
| `CALLPUT` | 看涨/看跌 |
| `STRIKEPRICE` | 行权价格 |
| `MONTH` / `MATURITYDATE` | 到期月 / 到期日 |
| `FTDATE` / `LASTTRADINGDATE` | 首个交易日 / 最后交易日 |
| `LDDATE` / `EXERCISINGEND` | 最后交割日 / 行权截止日 |
| `LPRICE` | 挂牌基准价 |
| `COUNIT` / `ADJ_SIGN` | 合约单位 / 调整标志 |

#### OPT_STD_ATTR — 期权标准化属性 (36列)
| 列名 | 说明 |
|------|------|
| `SUBJ_SEC_ID` | 标的证券ID |
| `OPT_TYPE` / `EUROAMERICANBERMUDA` | 期权类型 / 行权方式 |
| `SETTLEMENTMETHOD` | 结算方式 |
| `STRIKERATIO` / `COVALUE` | 合约比例 / 合约面值 |
| `TRADEFEE` / `POSLIMIT` | 交易费率 / 持仓限额 |
| `THOURS` | 交易时间 |
| `QUOTEUNIT` | 报价货币单位 |

### 11.4 全球证券主信息

#### GLB_FUT_MSTR — 全球期货主信息 (43列)
Bloomberg代码、WIND代码、FIGI、合约类型、交割日期、合约乘数、保证金率、最小报价单位、Bloomberg行业L1/L2/L3分类

#### GLB_STK_MSTR — 全球股票主信息 (48列)
Bloomberg代码、WIND代码、FIGI、SEDOL、ISIN、公司中英文名、上市/退市日、GICS行业、市场状态、IPO日期、ADR比例、Bloomberg行业L1/L2/L3

#### GLB_INDEX_MSTR — 全球指数主信息 (30列)
Bloomberg/WIND/FIGI代码、指数名称、交易所、国家、发布商、基期、基值

### 11.5 债券参考与估值

#### BOND_GEN_INFO — 债券基本信息 (122列)
| 字段类型 | 字段 |
|----------|------|
| 发行信息 | `ISSUER` / `ISSUER_TYPE`, `PAR_VALUE`, `COUPON_RATE`, `ISSUE_PRICE`, `PLAN_AMOUNT` / `ACT_AMOUNT` |
| 期限 | `TERM_YEAR` / `TERM_DAY`, 各种起止日期 |
| 票息结构 | `PAYMENT_TYPE`, `PAY_FREQUENCY`, `DAY_COUNT`, `COUPON_TXT`, `IS_CALLABLE`, `IS_CHOOSERIGHT` |
| 分类 | Wind一/二级, CCDC一/二级, 行业 L1/L2 |
| 评级 | `BOND_RATING`, `B_TENDRST_REFERYIELD` |
| 特殊属性 | `IS_CVTB`(可转债), `IS_ABS`, `IS_PERPETUAL`, `IS_CORP_BOND`, `IS_NET_PRC` |
| 担保 | `GUARANTOR` / `GUARANT_TYPE`, `IS_INRIGHT` |
| 其他 | `OUTSTANDING_BALANCE`, `LATEST_COUPONRATE`, `ISIN`, `IS_TAXFREE` |

#### CNBD_VALUE — 中债估值 (39列)
全价/净价/应计利息、**到期收益率(YTM)**、**修正久期**、**凸性**、**BPV**、**利差久期/凸性**、加权平均结算全价系列（含久期/凸性/BPV）

#### CNBD_BOND_YIELD_CURV — 中债收益率曲线 (18列)
按曲线ID（如国债即期）和标准期限给出 `YIELD`、基准利率、曲线值

### 11.6 利率与黄金

#### IBOR_RATE_QUOT — IBOR 利率行情 (11列)
`RATE_CD`（利率代码如SHIBOR_ON/1W/3M）+ `RATE(%)` + `CHNG_PCT`

#### GOLD_SPOT_DAY_QUOT — 黄金现货日行情 (26列)
上交所黄金现货：OHLCV + 持仓量 + 结算量/结算价 + 延迟补偿费支付方式

### 11.7 国债期货专用数据

#### TREA_FUT_DAY_QUOT_EXT — 国债期货延伸行情 (25列)
| 列名 | 说明 |
|------|------|
| `DLV_INT` / `RANGE_INT` | 交割利息 / 区间利息 |
| `DLV_COST` | 交割成本 |
| `IRR` | 隐含回购利率 (IRR) |
| `BASE_DIFF` | 基差 |
| `CF` | 转换因子 |
| `ACC_INT` | 累积应计利息 |
| `LATEST_COUPONRATE` | 最新票面利率 |
| `REMA_TERM_DAY` | 剩余天数 |

#### TREA_FUT_CHPST_ABLE_DLV_STK — 最便宜可交割券 (CTD, 15列)
每天每个国债期货合约的全市场 CTD、银行间 CTD、沪市 CTD、深市 CTD（各含证券代码 + IRR）

### 11.8 复权因子与基金

#### GLB_ADJ_FACTOR — 全球复权因子 (19列)
`EX_DT`（除权日）+ `ADJ_FACTOR`（复权因子）+ `BACK_ADJ_FACTOR`（前复权）+ `SUB_ADJ_FACTOR`

#### FUND_GEN_INFO — 基金基本信息 (95列)
管理人/托管人、投资类型/风格/范围、费率（管理费/托管费/销售服务费）、发行/上市/到期日、ETF/SHSC/REITs标识、Wind三级分类

### 11.9 Bloomberg原始数据 (dfs://HQUOT_CENTER)

| 表名 | 说明 |
|------|------|
| `BBG_QUOFCF` | Bloomberg 商品期货行情 |
| `BBG_QUOFIF` | Bloomberg 股指期货行情 |
| `BBG_QUOFTF` | Bloomberg 国债期货行情 |
| `BBG_QUOFXF` | Bloomberg 外汇期货行情 |

### 11.10 综合汇总

| 数据品类 | 表数 | 代表字段 |
|----------|------|----------|
| 股票参考 | 2 | 行业分类(SW/Wind/CSRC，三级)，ISIN，股本，涨跌停状态 |
| 股票估值 | 1 | PE/PB/PS/PCF (TTM/LYR)，市值，换手率，营收/利润 |
| 期货参考 | 3 | 合约规格，保证金率，手续费，连续合约映射 |
| 期权参考 | 2 | 行权价/到期日，行权方式，合约单位，持仓限额 |
| 全球证券主表 | 3 | Bloomberg/WIND/FIGI/ISIN/SEDOL，行业分类，市场状态 |
| 债券参考 | 1 | 发行人，票息结构，评级，评级，可转债/ABS/永续标识 |
| 债券估值 | 2 | YTM，修正久期，凸性，BPV，中债净价/全价 |
| 收益率曲线 | 1 | 国债/金融债等多条曲线，多期限 yield |
| 利率 | 1 | SHIBOR/LIBOR 等利率代码 |
| 黄金现货 | 1 | 上交所黄金现货 OHLCV + 持仓 |
| 国债期货延伸 | 2 | IRR，基差，转换因子，CTD（全市/银行间/沪深） |
| 复权因子 | 1 | 全球前复权、后复权因子 |
| 基金参考 | 1 | 管理人，费率，投资风格，ETF/REITs标识 |
| Bloomberg L2 | 4 | 期货 BBG 行情（CF/IF/TF/FX）|

1. **代理问题**：HMC 走 `10.50.*` 内网, 必须绕过 HTTP 代理，否则 gRPC 连接会被 302 重定向拦截
2. **查询要带过滤条件**：全表扫描会超时, 务必用 `where` 限定日期范围
3. **pandas 兼容性**：`hmc_sdk` 声明依赖 `pandas<=2.3.2`, 实际 pandas 3.x 也能正常工作
4. **MINUTE 类型解析**：DT 库分钟表的 `BARTIME` 列为 MINUTE 类型 (如 `09:31m`), `parse_result()` 无法直接转 DataFrame；Tick 表的 `iTime` 是 INT，LV2 快照的 `TrdTS` 是 TIMESTAMP，两者 `parse_result()` 正常
5. **分钟线 vs Tick**：分钟聚合管道自 2025-07-29 停止，Tick 原始数据持续入库至今（可自行从 Tick 自建分钟线）
6. **线程关闭警告**：`Exception in thread Thread-1 (_run)` 是 SDK 的 atexit handler 线程退出日志, 可忽略
7. **联系人**：数据权限 → 魏永长; SDK 版本 → 安志翔; 邮箱格式: `姓名拼音@cicc.com.cn`