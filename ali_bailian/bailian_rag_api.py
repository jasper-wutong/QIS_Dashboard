import os
from pathlib import Path
from dotenv import load_dotenv
from dashscope import Application

# 从项目根目录加载 .env 文件
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

def call_bailian_app(app_id, prompt):
    """
    调用百炼应用（RAG 知识库）
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    
    if not api_key:
        print("错误: 未找到 DASHSCOPE_API_KEY，请检查 .env 文件。")
        return

    print(f"正在向应用 {app_id} 发送请求...")
    
    try:
        # 调用百炼应用 API
        response = Application.call(
            app_id=app_id,
            prompt=prompt,
            api_key=api_key
        )

        if response.status_code != 200:
            print(f"请求失败: code={response.code}, message={response.message}")
        else:
            # 打印模型生成的回答
            print(f"\n回答:\n{response.output.text}")

    except Exception as e:
        print(f"发生异常: {e}")

if __name__ == "__main__":
    # ⚠️ 请将下面的 ID 替换为你在百炼控制台创建的应用 ID
    YOUR_APP_ID = "5be4e5cbe00f478390842a0254bd8abb"
    
    user_question = "你如何看待黄金的走势？"
    call_bailian_app(YOUR_APP_ID, user_question)