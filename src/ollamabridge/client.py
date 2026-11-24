from ollamabridge.config import settings
import httpx


async def chat(messages):
    url = f"{settings.deepseek_base_url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {settings.deepseek_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": settings.deepseek_model,
        "messages": messages,
        "stream": False,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()