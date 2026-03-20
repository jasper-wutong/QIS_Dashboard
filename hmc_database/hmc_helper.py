"""
HMC (华泰证券) 数据拉取辅助脚本 — 与主 Flask 进程隔离运行。

设计:
  - 由 market_data.py 通过 subprocess 调用
  - 接受命令行参数: --ticker CODE --start YYYY-MM-DD --end YYYY-MM-DD --mode [history|snapshot|futures]
  - 结果以 JSON 打印到 stdout
  - 错误也以 {"ok": false, "error": "..."} JSON 格式输出

认证配置 (优先级从高到低):
  1. 环境变量 HMC_APP_ID, HMC_TOKEN
  2. 本地文件 hmc_token.txt (第1行=app_id, 第2行=token)
  3. D:/token.txt (HMC SDK 默认路径)

使用方法:
  python hmc_helper.py --mode test
  python hmc_helper.py --ticker "T2506" --start 2025-01-01 --end 2026-03-11
  python hmc_helper.py --mode snapshot --ticker "T2506"
  python hmc_helper.py --mode futures_list --exchange SHFE
  python hmc_helper.py --mode query --sql "select top 10 * from loadTable('dfs://HQUOT_CENTER_EOD', 'GLB_STK_MSTR')"
  python hmc_helper.py --mode metadata --action databases
"""

import os
import sys
import math
import json
import argparse
import logging
import datetime

# ── 绕过代理访问 HMC 内网 ──
_HMC_NO_PROXY = "hmc-dev.cicc.com,hmc-prod.cicc.com,10.50.*,localhost,127.0.0.1"
for _k in ("NO_PROXY", "no_proxy"):
    existing = os.environ.get(_k, "")
    if "hmc" not in existing.lower():
        os.environ[_k] = ",".join(filter(None, [existing, _HMC_NO_PROXY]))
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    _v = os.environ.get(_k, "")
    if _v:
        # gRPC 不支持 NO_PROXY 通配符, 必须清除代理让 HMC 走直连
        os.environ.pop(_k, None)
os.environ.pop("grpc_proxy", None)
os.environ.pop("GRPC_PROXY", None)


def _out(data):
    """向 stdout 输出 JSON, 然后退出。"""
    print(json.dumps(data, ensure_ascii=False, default=_json_default))
    sys.stdout.flush()


def _json_default(obj):
    """JSON 序列化 fallback: 处理 date, datetime, Timestamp 等。"""
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    if hasattr(obj, 'strftime'):
        return obj.strftime("%Y-%m-%d")
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    return str(obj)


def _safe_val(v):
    """将 NaN / Inf / None 转为 None。"""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _load_credentials():
    """加载 HMC 认证凭据, 返回 (app_id, token) 或 raise。"""
    # 1. 环境变量
    app_id = os.environ.get("HMC_APP_ID", "").strip()
    token  = os.environ.get("HMC_TOKEN", "").strip()
    if app_id and token:
        return app_id, token

    # 2. 本地配置文件
    local_token_file = os.path.join(os.path.dirname(__file__), "hmc_token.txt")
    for path in [local_token_file, r"D:\token.txt", r"D:/token.txt"]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            if len(lines) >= 2:
                return lines[0], lines[1]

    raise RuntimeError(
        "未找到 HMC 认证信息。请设置环境变量 HMC_APP_ID / HMC_TOKEN，"
        "或创建 hmc_token.txt (第1行=app_id, 第2行=token)"
    )


def _build_client(environment_name="PROD_OFFLINE", cluster_name="TRAIN"):
    """构建并初始化 HmcRpcClient，返回 (client, error_msg)。"""
    try:
        from hmc_sdk.client.hmc_rpc_client import HmcRpcClient
        from hmc_sdk.config.cluster import Cluster
        from hmc_sdk.config.environment import Environment
        from hmc_sdk.config.hmc_config import HmcConfig
    except ImportError as e:
        return None, "hmc_sdk 未安装或导入失败: {}".format(e)

    try:
        app_id, token = _load_credentials()
    except RuntimeError as e:
        return None, str(e)

    env_map = {
        "LOCAL":        Environment.LOCAL,
        "DEV":          Environment.DEV,
        "PROD":         Environment.PROD,
        "PROD_OFFLINE": Environment.PROD_OFFLINE,
    }
    cluster_map = {
        "TRAIN":    Cluster.TRAIN,
        "REALTIME": Cluster.REALTIME,
        "TEST":     Cluster.TEST,
    }

    env     = env_map.get(environment_name.upper(), Environment.PROD)
    cluster = cluster_map.get(cluster_name.upper(), Cluster.REALTIME)

    try:
        config = HmcConfig(
            environment=env,
            cluster=cluster,
            app_id=app_id,
            token=token,
            query_timeout_sec=30,
        )
        client = HmcRpcClient(config)
        client.initialize()
        return client, None
    except Exception as e:
        return None, "HMC 客户端初始化失败: {}".format(e)


def _safe_close(client):
    """安全关闭 client, 忽略异常。"""
    try:
        if client:
            client.close()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 数据获取函数
# ─────────────────────────────────────────────────────────────────────────────

def run_script(sql, environment="PROD_OFFLINE", cluster="TRAIN"):
    """执行 DDB SQL 脚本并输出结果 JSON。"""
    client, err = _build_client(environment, cluster)
    if err:
        return _out({"ok": False, "error": err})

    try:
        from hmc_sdk.client.database.script_client import ScriptClient

        script_client = ScriptClient(client)
        reply = script_client.run_script(sql, timeout_sec=60)

        # 尝试解析结果为 DataFrame
        parsed = False
        try:
            df = script_client.parse_result(reply)
            if df is not None and hasattr(df, 'to_dict'):
                records = df.to_dict(orient='records')
                # 清理 NaN
                for rec in records:
                    for k, v in rec.items():
                        rec[k] = _safe_val(v)
                parsed = True
        except Exception:
            pass

        _safe_close(client)

        if parsed:
            return _out({"ok": True, "data": records, "count": len(records)})

        # 原始返回
        result_str = getattr(reply, 'result', '')
        value_bytes = getattr(reply, 'value', b'')
        return _out({"ok": True, "result": result_str,
                     "value_size": len(value_bytes)})

    except Exception as e:
        _safe_close(client)
        return _out({"ok": False, "error": "脚本执行失败: {}".format(e)})


def fetch_metadata(action="databases", db_name="", table_name="",
                   environment="PROD_OFFLINE", cluster="TRAIN"):
    """获取 DDB 元数据信息 (库/表/列)。"""
    client, err = _build_client(environment, cluster)
    if err:
        return _out({"ok": False, "error": err})

    try:
        from hmc_sdk.client.database.metadata_client import MetadataClient

        meta_client = MetadataClient(client)

        if action == "databases":
            result = meta_client.get_all_databases()
        elif action == "tables":
            if not db_name:
                client.close()
                return _out({"ok": False, "error": "--db 参数必填"})
            result = meta_client.get_all_tables(db_name)
        elif action == "columns":
            if not db_name or not table_name:
                client.close()
                return _out({"ok": False, "error": "--db 和 --table 参数必填"})
            result = meta_client.get_all_columns(db_name, table_name)
        else:
            client.close()
            return _out({"ok": False, "error": "未知 action: {}".format(action)})

        _safe_close(client)
        return _out({"ok": True, "result": str(result)})

    except Exception as e:
        _safe_close(client)
        return _out({"ok": False, "error": "获取元数据失败: {}".format(e)})


def fetch_futures_history(ticker, start_date, end_date,
                          environment="PROD_OFFLINE", cluster="TRAIN"):
    """获取期货历史 EOD 数据并输出 JSON。"""
    client, err = _build_client(environment, cluster)
    if err:
        return _out({"ok": False, "error": err})

    try:
        from hmc_sdk.client.standard.futures_client import FuturesClient
        from hmc_sdk.model.data.data_pb2 import DataForm, Area

        futures_client = FuturesClient(client)

        # 判断是否国内期货
        area = Area.CN_ML if any(
            ticker.upper().endswith(x)
            for x in [".SHF", ".DCE", ".ZCE", ".CFFEX", ".INE", ".GFEX"]
        ) else Area.ALL   # HMC SDK 无 Area.OVERSEAS, 用 ALL 兜底 (但实际HMC无境外期货数据)

        data_list = futures_client.get_futures_time_range(
            code_list=[ticker],
            data_form=DataForm.EOD,
            start_time=start_date.replace("-", "."),
            end_time=end_date.replace("-", "."),
            area=area,
        )

        rows = []
        for item in (data_list or []):
            rows.append({
                "date":   getattr(item, "date",   None),
                "open":   _safe_val(getattr(item, "open",   None)),
                "high":   _safe_val(getattr(item, "high",   None)),
                "low":    _safe_val(getattr(item, "low",    None)),
                "close":  _safe_val(getattr(item, "close",  None)),
                "volume": _safe_val(getattr(item, "volume", None)),
                "oi":     _safe_val(getattr(item, "open_interest", None)),
            })

        _safe_close(client)
        return _out({"ok": True, "ticker": ticker, "data": rows, "count": len(rows)})

    except Exception as e:
        _safe_close(client)
        return _out({"ok": False, "error": "获取期货历史数据失败: {}".format(e)})


def fetch_futures_snapshot(ticker, environment="PROD_OFFLINE", cluster="TRAIN"):
    """获取期货最新EOD快照数据并输出 JSON。

    使用 FuturesClient.get_futures_latest(DataForm.EOD) 获取最新日线数据。
    注意: HMC 标准 API 仅支持境内期货; 境外代码会直接返回失败。
    """
    client, err = _build_client(environment, cluster)
    if err:
        return _out({"ok": False, "error": err})

    try:
        from hmc_sdk.client.standard.futures_client import FuturesClient
        from hmc_sdk.model.data.data_pb2 import DataForm, Area

        is_domestic = any(
            ticker.upper().endswith(x)
            for x in [".SHF", ".DCE", ".ZCE", ".CFFEX", ".INE", ".GFEX"]
        )
        if not is_domestic:
            _safe_close(client)
            return _out({"ok": False, "error": "HMC 标准 API 不支持境外期货: {}".format(ticker)})

        futures_client = FuturesClient(client)

        data_list = futures_client.get_futures_latest(
            code_list=[ticker],
            data_form=DataForm.EOD,
            area=Area.CN_ML,
        )

        results = []
        for item in (data_list or []):
            ts_data = getattr(item, "time_series_data", None) or []
            for d in ts_data:
                import datetime as _dt
                ts_ms = getattr(d, "time", None)
                ts_str = ""
                if ts_ms:
                    try:
                        ts_str = _dt.datetime.fromtimestamp(ts_ms / 1000.0).strftime("%Y-%m-%d")
                    except Exception:
                        ts_str = str(ts_ms)
                _data_range = getattr(item, "data_range", None)
                _data_code = getattr(_data_range, "data_code", ticker)
                # protobuf repeated fields are not plain list; convert to str
                _data_code_str = str(_data_code)
                if _data_code_str.startswith("["):
                    # e.g. "['AG2606.SHF']" → take first element
                    try:
                        _data_code_str = list(_data_code)[0]
                    except (IndexError, TypeError):
                        _data_code_str = ticker
                results.append({
                    "code":       _data_code_str,
                    "last":       _safe_val(getattr(d, "close",            None)),
                    "settlement": _safe_val(getattr(d, "settlement_price", None)),
                    "open":       _safe_val(getattr(d, "open",             None)),
                    "high":       _safe_val(getattr(d, "high",             None)),
                    "low":        _safe_val(getattr(d, "low",              None)),
                    "volume":     _safe_val(getattr(d, "volume",           None)),
                    "oi":         _safe_val(getattr(d, "open_interest",    None)),
                    "timestamp":  ts_str,
                })

        _safe_close(client)
        return _out({"ok": True, "data": results})

    except Exception as e:
        _safe_close(client)
        return _out({"ok": False, "error": "获取期货快照失败: {}".format(e)})


def fetch_futures_list(exchange=None, environment="PROD_OFFLINE", cluster="TRAIN"):
    """获取可交易期货合约列表并输出 JSON。"""
    client, err = _build_client(environment, cluster)
    if err:
        return _out({"ok": False, "error": err})

    try:
        from hmc_sdk.client.standard.futures_client import FuturesClient
        from hmc_sdk.model.data.data_pb2 import Exchange as HmcExchange, TradeState

        futures_client = FuturesClient(client)

        exchange_map = {
            "SHFE":  HmcExchange.SHFE,
            "DCE":   HmcExchange.DCE,
            "ZCE":   HmcExchange.ZCE,
            "CFFEX": HmcExchange.CFFEX,
            "INE":   HmcExchange.INE,
            "GFEX":  HmcExchange.GFEX,
        }

        kwargs = {"trade_state": TradeState.TRADE}
        if exchange and exchange.upper() in exchange_map:
            kwargs["exchange"] = exchange_map[exchange.upper()]

        data_list = futures_client.get_future_trading_ticker(**kwargs)

        codes = []
        for item in (data_list or []):
            dr = getattr(item, "data_range", None)
            if dr:
                codes.extend(list(dr.full_data_code))

        _safe_close(client)
        return _out({"ok": True, "codes": codes, "count": len(codes)})

    except Exception as e:
        _safe_close(client)
        return _out({"ok": False, "error": "获取合约列表失败: {}".format(e)})


def test_connection(environment="PROD_OFFLINE", cluster="TRAIN"):
    """测试 HMC 连接是否正常。"""
    client, err = _build_client(environment, cluster)
    if err:
        return _out({"ok": False, "error": err})
    _safe_close(client)
    return _out({"ok": True, "message": "HMC 连接测试成功",
                 "environment": environment, "cluster": cluster})


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HMC SDK 数据拉取工具")
    parser.add_argument("--mode",        default="history",
                        choices=["history", "snapshot", "futures_list",
                                 "query", "metadata", "test"],
                        help="运行模式")
    parser.add_argument("--ticker",      default="",    help="合约代码")
    parser.add_argument("--start",       default="",    help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end",         default="",    help="结束日期 YYYY-MM-DD")
    parser.add_argument("--exchange",    default="",    help="交易所 (SHFE/DCE/ZCE/CFFEX/INE/GFEX)")
    parser.add_argument("--sql",         default="",    help="DDB SQL 语句 (mode=query)")
    parser.add_argument("--action",      default="databases",
                        choices=["databases", "tables", "columns"],
                        help="元数据操作 (mode=metadata)")
    parser.add_argument("--db",          default="",    help="DDB 库名")
    parser.add_argument("--table",       default="",    help="DDB 表名")
    parser.add_argument("--environment", default="PROD_OFFLINE",
                        choices=["LOCAL", "DEV", "PROD", "PROD_OFFLINE"],
                        help="HMC 环境 (默认 PROD_OFFLINE)")
    parser.add_argument("--cluster",     default="TRAIN",
                        choices=["TRAIN", "REALTIME", "TEST"],
                        help="HMC 集群 (默认 TRAIN)")
    args = parser.parse_args()

    if args.mode == "history":
        if not args.ticker:
            return _out({"ok": False, "error": "--ticker 参数必填"})
        if not args.start or not args.end:
            return _out({"ok": False, "error": "--start 和 --end 参数必填"})
        fetch_futures_history(args.ticker, args.start, args.end,
                              args.environment, args.cluster)

    elif args.mode == "snapshot":
        if not args.ticker:
            return _out({"ok": False, "error": "--ticker 参数必填"})
        fetch_futures_snapshot(args.ticker, args.environment, args.cluster)

    elif args.mode == "futures_list":
        fetch_futures_list(args.exchange or None,
                           args.environment, args.cluster)

    elif args.mode == "query":
        if not args.sql:
            return _out({"ok": False, "error": "--sql 参数必填"})
        run_script(args.sql, args.environment, args.cluster)

    elif args.mode == "metadata":
        fetch_metadata(args.action, args.db, args.table,
                       args.environment, args.cluster)

    elif args.mode == "test":
        test_connection(args.environment, args.cluster)

    else:
        return _out({"ok": False, "error": "未知 mode: {}".format(args.mode)})


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
