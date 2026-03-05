"""
百炼RAG研究分析模块

基于阿里百炼RAG知识库和DashScope API，对期货/商品标的进行深度研究分析。
当百炼API失败时（如DataInspectionFailed），自动fallback到Copilot SDK。
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from dashscope import Application

# 从项目根目录加载 .env 文件
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# 百炼应用ID（RAG知识库应用）
BAILIAN_APP_ID = os.getenv("BAILIAN_APP_ID", "5be4e5cbe00f478390842a0254bd8abb")

# Copilot SDK CLI脚本路径
RESEARCH_CLI_PATH = Path(__file__).parent.parent / 'research_cli.py'


def _copilot_fallback_research(
    name: str,
    price: str = "NA",
    change: str = "NA",
    exposure: str = "NA",
) -> Dict[str, Any]:
    """
    使用Copilot SDK作为fallback进行研究分析
    通过subprocess调用research_cli.py来避免事件循环冲突
    """
    print(f"[COPILOT FALLBACK] 调用Copilot SDK分析: {name}")
    
    python_exe = sys.executable
    cmd = [
        python_exe,
        str(RESEARCH_CLI_PATH),
        name,
        "--price", price,
        "--change", change,
        "--exposure", exposure,
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5分钟超时
            cwd=str(RESEARCH_CLI_PATH.parent),
        )
        
        if result.returncode != 0:
            print(f"[COPILOT FALLBACK] CLI返回错误: {result.stderr}")
            return {
                "ok": False,
                "name": name,
                "model": "copilot-sdk",
                "content": "",
                "error": f"Copilot CLI错误: {result.stderr[:200] if result.stderr else 'unknown'}"
            }
        
        # 解析JSON输出
        try:
            data = json.loads(result.stdout)
            data["model"] = data.get("model", "copilot-sdk")
            print(f"[COPILOT FALLBACK] 分析完成: {name}, model={data.get('model')}")
            return data
        except json.JSONDecodeError:
            return {
                "ok": False,
                "name": name,
                "model": "copilot-sdk",
                "content": "",
                "error": f"无法解析Copilot响应: {result.stdout[:200]}"
            }
            
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "name": name,
            "model": "copilot-sdk",
            "content": "",
            "error": "Copilot SDK调用超时"
        }
    except Exception as e:
        return {
            "ok": False,
            "name": name,
            "model": "copilot-sdk",
            "content": "",
            "error": f"Copilot SDK异常: {str(e)}"
        }


def _copilot_fallback_chat(
    name: str,
    message: str,
    history: list,
    price: str = "NA",
    change: str = "NA",
    exposure: str = "NA",
) -> Dict[str, Any]:
    """
    使用Copilot SDK作为fallback进行多轮对话
    """
    print(f"[COPILOT FALLBACK/CHAT] 调用Copilot SDK对话: {name}")
    
    python_exe = sys.executable
    cmd = [
        python_exe,
        str(RESEARCH_CLI_PATH),
        name,
        "--chat", message,
        "--history", json.dumps(history, ensure_ascii=False),
        "--price", price,
        "--change", change,
        "--exposure", exposure,
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,  # 3分钟超时
            cwd=str(RESEARCH_CLI_PATH.parent),
        )
        
        if result.returncode != 0:
            print(f"[COPILOT FALLBACK/CHAT] CLI返回错误: {result.stderr}")
            return {
                "ok": False,
                "name": name,
                "model": "copilot-sdk",
                "content": "",
                "error": f"Copilot CLI错误: {result.stderr[:200] if result.stderr else 'unknown'}"
            }
        
        try:
            data = json.loads(result.stdout)
            data["model"] = data.get("model", "copilot-sdk")
            print(f"[COPILOT FALLBACK/CHAT] 对话完成: {name}")
            return data
        except json.JSONDecodeError:
            return {
                "ok": False,
                "name": name,
                "model": "copilot-sdk",
                "content": "",
                "error": f"无法解析Copilot响应"
            }
            
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "name": name,
            "model": "copilot-sdk",
            "content": "",
            "error": "Copilot SDK对话超时"
        }
    except Exception as e:
        return {
            "ok": False,
            "name": name,
            "model": "copilot-sdk",
            "content": "",
            "error": f"Copilot SDK异常: {str(e)}"
        }


def _fmt_exposure(exposure_str: str) -> str:
    """格式化敞口数值显示"""
    if not exposure_str or exposure_str == "NA":
        return "未知"
    try:
        val = float(exposure_str)
        wan = val / 10000
        if abs(wan) >= 10000:
            return f"{wan / 10000:.1f}亿元"
        return f"{wan:.0f}万元"
    except (ValueError, TypeError):
        return exposure_str


def _extract_ticker_base(name: str) -> str:
    """从标的名称中提取品种基础名称
    
    例如: "铜期货" -> "铜", "CU2602.SHF" -> "CU", "黄金期货" -> "黄金"
    """
    import re
    # 常见期货后缀
    name = name.replace("期货", "").replace("主力", "").strip()
    
    # 如果是代码格式 (如 CU2602.SHF)，提取前缀
    match = re.match(r'^([A-Z]+)', name.upper())
    if match:
        return match.group(1)
    
    return name


def _build_research_prompt(
    name: str,
    price: str = "NA",
    change: str = "NA",
    exposure: str = "NA",
) -> str:
    """构建研究分析的prompt"""
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 格式化持仓信息
    position_info = ""
    if price and price != "NA":
        position_info += f"\n- 当前价格: {price}"
    if change and change != "NA":
        position_info += f"\n- 涨跌幅: {change}"

    exposure_display = _fmt_exposure(exposure)
    exposure_direction = ""
    if exposure and exposure != "NA":
        try:
            exp_val = float(exposure)
            direction = "多头" if exp_val > 0 else "空头" if exp_val < 0 else "零"
            position_info += f"\n- 当前风险敞口: {exposure_display}（{direction}方向）"
            exposure_direction = direction
        except (ValueError, TypeError):
            position_info += f"\n- 当前风险敞口: {exposure_display}"
    
    # 提取品种名称用于知识库检索
    ticker_base = _extract_ticker_base(name)

    return f"""你是一名资深金融大宗商品与衍生品研究分析师，服务于一家大型投行的QIS（量化投资策略）交易台。
当前日期: {today}

请基于知识库中的研报信息，对【{name}】进行全面深度研究分析。

我当前的持仓信息:{position_info}

请按以下结构进行分析（每部分务必详细）:

## 1. 市场概况（150-200字）
基于研报数据，分析近期价格走势、成交量与持仓量变化、期限结构（contango/backwardation）等。

## 2. 核心驱动因素（200-250字）
- **宏观面**：美联储/央行政策、美元指数、利率、通胀数据对该品种的影响
- **供需面**：产量/产能、库存变化（交易所库存、社会库存）、进出口、季节性因素
- **政策面**：关税、环保限产、战略收储等政策影响
- **地缘与事件**：地缘冲突、异常天气、产业链上下游联动等

## 3. 技术面要点（80-100字）
关键支撑/阻力位、均线系统、MACD/RSI等指标信号。

## 4. 风险提示（80-100字）
主要下行/上行风险因素，需要关注的风险事件。

## 5. 敞口调整建议【重要】
这一部分非常重要，必须单独成段，给出明确的操作建议：

基于以上分析，结合目前 {exposure_display} 的{exposure_direction}敞口：
- 当前敞口是否合理？
- 建议如何调整？（增大/减小/平掉/反向）
- 建议目标敞口金额是多少万元？
- 具体理由是什么？

**请用以下格式明确给出建议：**
> 当前敞口: {exposure_display}（{exposure_direction}）
> 建议操作: （增大/减小/维持/平仓）
> 目标敞口: XXX万元
> 调整理由: （简要说明）

请用中文回答，语言专业但易懂。如果知识库中没有相关研报，请明确说明并给出你基于市场常识的分析。"""


def _build_chat_prompt(
    name: str,
    message: str,
    history: list,
    price: str = "NA",
    change: str = "NA",
    exposure: str = "NA",
) -> str:
    """构建多轮对话的prompt"""
    exposure_display = _fmt_exposure(exposure)
    
    # 构建上下文
    context = f"""你是一名资深金融大宗商品与衍生品研究分析师。
当前讨论的标的: {name}
当前价格: {price}
涨跌幅: {change}
当前风险敞口: {exposure_display}

请基于知识库中的研报信息和上下文，简洁专业地回答用户问题。"""
    
    # 构建对话历史
    history_text = ""
    if history:
        for h in history:
            role = "用户" if h.get("role") == "user" else "助手"
            history_text += f"\n【{role}】: {h.get('content', '')}\n"
    
    return f"""{context}

--- 对话历史 ---
{history_text}

【用户】: {message}

请回答："""


def research_ticker(
    name: str,
    price: str = "NA",
    change: str = "NA",
    exposure: str = "NA",
    app_id: Optional[str] = None,
    max_retries: int = 3,
    use_fallback: bool = True,
) -> Dict[str, Any]:
    """
    对单个标的进行研究分析
    
    Args:
        name: 标的名称（如 "铜期货"、"CU2602.SHF"）
        price: 当前价格
        change: 涨跌幅
        exposure: 风险敞口
        app_id: 百炼应用ID（可选，默认使用环境变量配置）
        max_retries: 最大重试次数
        use_fallback: 当百炼失败时是否使用Copilot SDK作为fallback（默认True）
    
    Returns:
        {
            "ok": bool,
            "name": str,
            "model": str,
            "content": str,
            "error": str (如果失败)
        }
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        # API key未配置时，直接尝试fallback
        if use_fallback:
            print(f"[BAILIAN] {name} 未配置API Key，尝试Copilot SDK fallback...")
            return _copilot_fallback_research(name, price, change, exposure)
        return {
            "ok": False,
            "name": name,
            "model": "bailian-rag",
            "content": "",
            "error": "未配置 DASHSCOPE_API_KEY，请检查 .env 文件"
        }
    
    app_id = app_id or BAILIAN_APP_ID
    prompt = _build_research_prompt(name, price, change, exposure)
    
    last_error = None
    for attempt in range(max_retries):
        try:
            response = Application.call(
                app_id=app_id,
                prompt=prompt,
                api_key=api_key,
            )
            
            if response.status_code == 200:
                content = response.output.text if response.output else ""
                return {
                    "ok": True,
                    "name": name,
                    "model": "bailian-rag",
                    "content": content.strip(),
                }
            
            # 如果是InternalError，重试
            if response.code == "InternalError" and attempt < max_retries - 1:
                print(f"[BAILIAN] {name} 第{attempt+1}次调用失败(InternalError)，{2 ** attempt}秒后重试...")
                time.sleep(2 ** attempt)  # 指数退避：1s, 2s, 4s
                last_error = f"code={response.code}, message={response.message}"
                continue
            
            # 其他错误（如DataInspectionFailed），尝试fallback
            error_msg = f"百炼API调用失败: code={response.code}, message={response.message}"
            print(f"[BAILIAN] {name} {error_msg}")
            
            if use_fallback:
                print(f"[BAILIAN] {name} 尝试Copilot SDK fallback...")
                return _copilot_fallback_research(name, price, change, exposure)
            
            return {
                "ok": False,
                "name": name,
                "model": "bailian-rag",
                "content": "",
                "error": error_msg
            }
            
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                print(f"[BAILIAN] {name} 第{attempt+1}次调用异常({e})，{2 ** attempt}秒后重试...")
                time.sleep(2 ** attempt)
                continue
            
            # 异常后尝试fallback
            print(f"[BAILIAN] {name} 百炼API异常: {str(e)}")
            if use_fallback:
                print(f"[BAILIAN] {name} 尝试Copilot SDK fallback...")
                return _copilot_fallback_research(name, price, change, exposure)
            
            return {
                "ok": False,
                "name": name,
                "model": "bailian-rag",
                "content": "",
                "error": f"百炼API异常: {str(e)}"
            }
    
    # 重试耗尽后尝试fallback
    print(f"[BAILIAN] {name} 重试{max_retries}次后仍失败: {last_error}")
    if use_fallback:
        print(f"[BAILIAN] {name} 尝试Copilot SDK fallback...")
        return _copilot_fallback_research(name, price, change, exposure)
    
    return {
        "ok": False,
        "name": name,
        "model": "bailian-rag",
        "content": "",
        "error": f"百炼API重试{max_retries}次后仍失败: {last_error}"
    }


def chat_ticker(
    name: str,
    message: str,
    history: list,
    price: str = "NA",
    change: str = "NA",
    exposure: str = "NA",
    app_id: Optional[str] = None,
    max_retries: int = 3,
    use_fallback: bool = True,
) -> Dict[str, Any]:
    """
    与百炼进行多轮对话
    
    Args:
        name: 标的名称
        message: 用户消息
        history: 对话历史 [{"role": "user/assistant", "content": "..."}]
        price: 当前价格
        change: 涨跌幅
        exposure: 风险敞口
        app_id: 百炼应用ID
        max_retries: 最大重试次数
        use_fallback: 当百炼失败时是否使用Copilot SDK作为fallback（默认True）
    
    Returns:
        {
            "ok": bool,
            "name": str,
            "model": str,
            "content": str,
            "error": str (如果失败)
        }
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        if use_fallback:
            print(f"[BAILIAN/CHAT] {name} 未配置API Key，尝试Copilot SDK fallback...")
            return _copilot_fallback_chat(name, message, history, price, change, exposure)
        return {
            "ok": False,
            "name": name,
            "model": "bailian-rag",
            "content": "",
            "error": "未配置 DASHSCOPE_API_KEY"
        }
    
    app_id = app_id or BAILIAN_APP_ID
    prompt = _build_chat_prompt(name, message, history, price, change, exposure)
    
    last_error = None
    for attempt in range(max_retries):
        try:
            response = Application.call(
                app_id=app_id,
                prompt=prompt,
                api_key=api_key,
            )
            
            if response.status_code == 200:
                content = response.output.text if response.output else ""
                return {
                    "ok": True,
                    "name": name,
                    "model": "bailian-rag",
                    "content": content.strip(),
                }
            
            # 如果是InternalError，重试
            if response.code == "InternalError" and attempt < max_retries - 1:
                print(f"[BAILIAN/CHAT] {name} 第{attempt+1}次调用失败，重试中...")
                time.sleep(2 ** attempt)
                last_error = f"code={response.code}, message={response.message}"
                continue
            
            # 其他错误（如DataInspectionFailed），尝试fallback
            error_msg = f"百炼API调用失败: code={response.code}, message={response.message}"
            print(f"[BAILIAN/CHAT] {name} {error_msg}")
            
            if use_fallback:
                print(f"[BAILIAN/CHAT] {name} 尝试Copilot SDK fallback...")
                return _copilot_fallback_chat(name, message, history, price, change, exposure)
            
            return {
                "ok": False,
                "name": name,
                "model": "bailian-rag",
                "content": "",
                "error": error_msg
            }
            
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                print(f"[BAILIAN/CHAT] {name} 第{attempt+1}次调用异常，重试中...")
                time.sleep(2 ** attempt)
                continue
            
            print(f"[BAILIAN/CHAT] {name} 百炼API异常: {str(e)}")
            if use_fallback:
                print(f"[BAILIAN/CHAT] {name} 尝试Copilot SDK fallback...")
                return _copilot_fallback_chat(name, message, history, price, change, exposure)
            
            return {
                "ok": False,
                "name": name,
                "model": "bailian-rag",
                "content": "",
                "error": f"百炼API异常: {str(e)}"
            }
    
    print(f"[BAILIAN/CHAT] {name} 重试{max_retries}次后仍失败: {last_error}")
    if use_fallback:
        print(f"[BAILIAN/CHAT] {name} 尝试Copilot SDK fallback...")
        return _copilot_fallback_chat(name, message, history, price, change, exposure)
    
    return {
        "ok": False,
        "name": name,
        "model": "bailian-rag",
        "content": "",
        "error": f"百炼API重试{max_retries}次后仍失败: {last_error}"
    }


def research_batch(
    tickers: list,
    app_id: Optional[str] = None,
) -> list:
    """
    批量研究多个标的（逐个调用）
    
    Args:
        tickers: [{"name": "铜期货", "price": "...", "change": "...", "exposure": "..."}]
        app_id: 百炼应用ID
    
    Returns:
        [{"ok": bool, "name": str, "model": str, "content": str, ...}]
    """
    results = []
    for t in tickers:
        result = research_ticker(
            name=t.get("name", ""),
            price=t.get("price", "NA"),
            change=t.get("change", "NA"),
            exposure=t.get("exposure", "NA"),
            app_id=app_id,
        )
        results.append(result)
    return results


# ── 测试代码 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 测试单个标的研究
    result = research_ticker(
        name="铜期货",
        price="76500",
        change="0.012",
        exposure="5000000",
    )
    
    print("=" * 60)
    print(f"标的: {result.get('name')}")
    print(f"状态: {'成功' if result.get('ok') else '失败'}")
    print(f"模型: {result.get('model')}")
    print("=" * 60)
    
    if result.get("ok"):
        print(result.get("content"))
    else:
        print(f"错误: {result.get('error')}")
