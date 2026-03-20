# agent_workspace

**Python data-fetching toolkit for QIS小助手**

Provides real-time and historical market data (Bloomberg + Wind) plus Cross Gamma analysis, directly from the command line or via import.

---

## Files

| File | Purpose |
|---|---|
| `analyze.py` | **Main entry point** — one-command analysis per asset |
| `fetch_bloomberg.py` | Bloomberg real-time snapshots + historical OHLCV |
| `fetch_wind.py` | Wind (万得) domestic futures snapshots + historical OHLCV |
| `fetch_cross_gamma.py` | Load + aggregate QIS Cross Gamma book data |
| `memory_manager.py` | **Agent memory** — save/recall analyses, methodology evolution |
| `memory/methodology.md` | Living analysis methodology document |
| `memory/daily/*.json` | Per-day analysis records |
| `memory/reviews/*.md` | Periodic accuracy reviews |

---

## Quick Start

### Gold analysis (Bloomberg + Wind + Cross Gamma)
```powershell
cd D:\QIS_DASHBOARD
.venv\Scripts\python agent_workspace\analyze.py --asset gold
```

### Browse available assets
```
gold / au       → Gold: COMEX GC1, SHFE AU.SHF, Cross Gamma
copper / cu     → Copper: LME LP1, SHFE CU.SHF, Cross Gamma
oil / crude     → Crude: WTI CL1, Brent CO1, INE SC.INE
metals          → All domestic metals snapshot (Wind)
book            → Full book cross gamma summary
market / macro  → Cross-asset snapshot (BBG + Wind)
```

### Single Bloomberg snapshot
```powershell
.venv\Scripts\python agent_workspace\fetch_bloomberg.py --ticker "GC1 Comdty" --mode snapshot
.venv\Scripts\python agent_workspace\fetch_bloomberg.py --mode gold
.venv\Scripts\python agent_workspace\fetch_bloomberg.py --mode market
```

### Single Wind snapshot
```powershell
# Uses Python 3.7 with WindPy automatically
python agent_workspace\fetch_wind.py --ticker AU.SHF --mode snapshot
python agent_workspace\fetch_wind.py --mode metals
python agent_workspace\fetch_wind.py --mode gold
```

### Cross Gamma
```powershell
.venv\Scripts\python agent_workspace\fetch_cross_gamma.py               # book summary
.venv\Scripts\python agent_workspace\fetch_cross_gamma.py --asset AU    # gold exposure
.venv\Scripts\python agent_workspace\fetch_cross_gamma.py --asset CU    # copper exposure
.venv\Scripts\python agent_workspace\fetch_cross_gamma.py --mode top_pairs --top 20
.venv\Scripts\python agent_workspace\fetch_cross_gamma.py --mode list_files
```

---

## Bloomberg Ticker Reference

| Label | Ticker | Description |
|---|---|---|
| Gold spot | `XAU Curncy` | Gold spot USD/oz |
| Gold front | `GC1 Comdty` | COMEX Gold continuous |
| Gold Apr26 | `AUAM26 Comdty` | Our book's main contract |
| Silver | `SI1 Comdty` | COMEX Silver |
| Real rate | `GTII10 Govt` | US 10Y TIPS yield |
| DXY | `DXY Curncy` | US Dollar Index |
| Breakeven | `USGGBE10 Index` | 10Y inflation breakeven |
| GLD ETF | `GLDUS Equity` | SPDR GLD holdings |
| COT net | `CFTGCNET Index` | CFTC gold speculative net |
| WTI | `CL1 Comdty` | WTI crude |
| Brent | `CO1 Comdty` | Brent crude |
| Copper LME | `LP1 Comdty` | LME copper 3M |
| VIX | `VIX Index` | Equity volatility |
| MOVE | `MOVE Index` | Rates volatility |

## Wind Ticker Reference

| Label | Ticker | Description |
|---|---|---|
| 沪金 | `AU.SHF` | 上海期货交易所 黄金主力连续 |
| 沪银 | `AG.SHF` | 上海期货交易所 白银主力连续 |
| 沪铜 | `CU.SHF` | 上海期货交易所 铜主力连续 |
| 螺纹 | `RB.SHF` | 上海期货交易所 螺纹钢主力连续 |
| 热卷 | `HC.SHF` | 上海期货交易所 热轧卷板主力连续 |
| 铁矿 | `I.DCE` | 大商所 铁矿石主力连续 |
| 上海原油 | `SC.INE` | 上海国际能源交易中心 原油主力连续 |
| 10Y国债期货 | `T.CFE` | 中金所 10年期国债期货主力连续 |
| 5Y国债期货 | `TF.CFE` | 中金所 5年期国债期货主力连续 |
| 沪深300期货 | `IF.CFE` | 中金所 股指期货主力连续 |

---

## Architecture

All calls reuse the existing subprocess isolation pattern from `market_data.py`:
- **Bloomberg data** → `.venv\Scripts\python.exe bloomberg\bloomberg_helper.py`  
- **Wind data** → `C:\...\Python37\python.exe wind_helper.py`  
- **Cross Gamma** → direct `import cross_gamma` (no subprocess needed)

This ensures Bloomberg's `blpapi` runs in the correct venv, and Wind's `WindPy` runs in Python 3.7 — just like the main application.

---

## Requirements

- Bloomberg Terminal must be running and logged in (localhost:8194)
- Wind Terminal must be running (for WFT/wsq calls)
- Cross Gamma JSON files on the network share OR local `cross_gamma/` folder

---

## Memory System

The agent maintains persistent memory to track analysis accuracy and evolve methodology.

### Save an analysis
```python
from agent_workspace.memory_manager import save_analysis
save_analysis(
    asset="gold",
    analysis={"price": 4698, "dxy": 103.2, "real_rate": 1.85},
    judgment="★★★★ CONVICTION: 看涨黄金，目标4900",
    key_levels={"support": 4550, "resistance": 4750, "target": 4900},
)
```

### Recall past analyses
```powershell
.venv\Scripts\python agent_workspace\memory_manager.py recall --asset gold --last 7
.venv\Scripts\python agent_workspace\memory_manager.py status
```

### Read/update methodology
```powershell
.venv\Scripts\python agent_workspace\memory_manager.py methodology
```

### Review cycle
1. Accumulate 5+ days of analysis records
2. Recall past judgments and compare vs actual prices
3. Identify systematic errors
4. Update `memory/methodology.md` with lessons learned
5. Save review to `memory/reviews/`
