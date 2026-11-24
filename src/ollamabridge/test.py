import asyncio

from ollamabridge.client import chat
from ollamabridge.converter import get_content


async def main():
    messages = [
        {"role": "user", "content": "你好，请介绍一下自己。"}
    ]

    upstream_response = await chat(messages)
    ollama_response = get_content(upstream_response)

    print(ollama_response)


asyncio.run(main())