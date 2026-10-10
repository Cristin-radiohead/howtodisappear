
import os
import json
import requests
from dotenv import load_dotenv
from pathlib import Path

# 用绝对路径加载 .env，避免 PyCharm 工作目录不对导致读不到
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

COZE_API_TOKEN = os.getenv("COZE_API_TOKEN")
COZE_BOT_ID = os.getenv("COZE_BOT_ID")

if not COZE_API_TOKEN or not COZE_BOT_ID:
    print("env路径:", env_path)
    print("文件存在吗:", env_path.exists())
    print("TOKEN读到:", repr(COZE_API_TOKEN))
    print("BOT_ID读到:", repr(COZE_BOT_ID))
    raise ValueError("请检查 .env 文件，确保 COZE_API_TOKEN 和 COZE_BOT_ID 已正确填写")

API_URL = "https://api.coze.cn/v3/chat"  # 国内站；国际站改为 api.coze.com
HEADERS = {
    "Authorization": f"Bearer {COZE_API_TOKEN}",
    "Content-Type": "application/json"
}

import time

def chat_stream(user_input, user_id="cli_user_001"):
    """非流式 + 轮询，获取智能体的完整回复"""
    # 第一步：发起对话
    payload = {
        "bot_id": COZE_BOT_ID,
        "user_id": user_id,
        "stream": False,
        "auto_save_history": True,
        "additional_messages": [
            {
                "role": "user",
                "content": user_input,
                "content_type": "text"
            }
        ]
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code != 200:
        print(f"\n发起对话失败，状态码：{response.status_code}")
        print(response.text)
        return

    result = response.json()
    if result.get("code") != 0:
        print(f"\n发起对话出错：{result.get('msg')}")
        return

    chat_id = result["data"]["id"]
    conversation_id = result["data"]["conversation_id"]
    print(f"[已发起] chat_id={chat_id}")

    # 第二步：轮询对话状态
    retrieve_url = "https://api.coze.cn/v3/chat/retrieve"
    params = {"conversation_id": conversation_id, "chat_id": chat_id}

    max_retries = 30          # 最多等 30 次
    interval = 1              # 每次间隔 1 秒

    for i in range(max_retries):
        time.sleep(interval)
        r = requests.get(retrieve_url, headers=HEADERS, params=params)
        if r.status_code != 200:
            print(f"[轮询] 请求失败：{r.status_code}")
            continue

        data = r.json()
        status = data.get("data", {}).get("status")

        if status == "completed":
            print(f"[完成] 耗时约 {i + 1} 秒")
            break
        elif status == "failed":
            print(f"[失败] {data.get('data', {}).get('last_error', {}).get('msg')}")
            return
        else:
            print(f"[轮询 {i + 1}] 状态：{status}")
    else:
        print("[超时] 轮询次数用完，对话仍未完成")
        return

    # 第三步：获取消息列表，提取 AI 回复
    list_url = "https://api.coze.cn/v3/chat/message/list"
    list_params = {"conversation_id": conversation_id, "chat_id": chat_id}
    r = requests.get(list_url, headers=HEADERS, params=list_params)

    if r.status_code != 200:
        print(f"\n获取消息失败，状态码：{r.status_code}")
        print(r.text)
        return

    msg_data = r.json()
    messages = msg_data.get("data", [])

    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("type") == "answer":
            print(f"\nAI: {msg.get('content', '')}")
            return

    print("\n[未找到 assistant 回复]")
    print(json.dumps(messages, ensure_ascii=False, indent=2))

    # 逐行解析 SSE 流（带完整调试）
    for line in response.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8")
        print(f"[原始行] {line_str}")  # 调试：打印每一行

        if line_str.startswith("data:"):
            data_str = line_str[5:].strip()

            # 大小写不敏感地跳过 done 标记
            if data_str.lower() in ("[done]", "done"):
                print("[流结束]")
                break

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                print(f"[非JSON] {data_str}")
                continue

            if not isinstance(data, dict):
                print(f"[非字典] {data}")
                continue

            event = data.get("event")
            if event == "conversation.message.delta":
                content = data.get("data", {}).get("content", "")
                print(content, end="", flush=True)
            else:
                print(f"[事件] {event}")

    print()

def main():
    print("Coze CLI 已启动，输入 exit 或 quit 退出。\n")
    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            break
        if not user_input:
            continue
        print("AI: ", end="", flush=True)
        chat_stream(user_input)

if __name__ == "__main__":
    main()