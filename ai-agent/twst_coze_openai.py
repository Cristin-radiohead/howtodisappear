import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

client = OpenAI(
    api_key=os.getenv("ONEAPI_KEY"),
    base_url="http://172.28.22.104:3000/v1"   # 改这里：指向 One-API
)

response = client.chat.completions.create(
    model="232323",                      # 改这里：随便填，One-API 会路由
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    stream=False
)

print("Coze（经 One-API）回复：")
print(response.choices[0].message.content)