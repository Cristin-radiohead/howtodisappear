import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# 你之前搭建的 Coze 智能体（信息现在配置在 One-API 渠道里）
COZE_BOT_ID = "7693506287732211738"   # 和 One-API 渠道里的模型名一致

client = OpenAI(
    api_key=os.getenv("ONEAPI_KEY"),
    base_url="http://172.22.28.104:3000/v1"
)

def chat_stream(user_input, user_id="cli_user_001"):
    """通过 One-API 调用指定的 Coze 智能体"""
    response = client.chat.completions.create(
        model="bot-7693506287732211738",# 直接用 Bot ID，一眼看出调的是哪个智能体
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ],
        stream=False
    )
    print(response.choices[0].message.content)

def main():
    print(f"Coze CLI 已启动（Bot ID: {COZE_BOT_ID}），输入 exit 或 quit 退出。\n")
    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            break
        if not user_input:
            continue
        print("AI: ", end="", flush=True)
        chat_stream(user_input)
        print()

if __name__ == "__main__":
    main()