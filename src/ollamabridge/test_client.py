import asyncio
from ollamabridge.client import chat


async def main():
    result = await chat([
        {"role": "user", "content": "你好，请简单介绍一下自己。"}
    ])
    print(result)


asyncio.run(main())