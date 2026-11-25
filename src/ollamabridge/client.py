from typing import Any
from ollamabridge.config import settings
from ollamabridge.converter import convert_messages_to_openai
import httpx
import json


_OPTION_MAPPING = {
    "temperature": "temperature",
    "top_p": "top_p",
    "num_predict": "max_tokens",
    "stop": "stop",
    "seed": "seed",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
}


def _build_request(
    messages: list[dict[str, Any]],
    stream: bool,
    options: dict[str, Any] | None = None,
    response_format: str | dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
):
    url = f"{settings.base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": settings.model,
        "messages": convert_messages_to_openai(messages),
        "stream": stream,
    }

    if options:
        for src_key, dst_key in _OPTION_MAPPING.items():
            if src_key in options and options[src_key] is not None:
                payload[dst_key] = options[src_key]

    if response_format == "json":
        payload["response_format"] = {"type": "json_object"}
    elif isinstance(response_format, dict):
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "ollama_format", "schema": response_format},
        }

    if tools:
        payload["tools"] = tools

    return url, headers, payload


async def chat(
    messages: list[dict[str, Any]],
    options: dict[str, Any] | None = None,
    response_format: str | dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
):
    url, headers, payload = _build_request(
        messages,
        stream=False,
        options=options,
        response_format=response_format,
        tools=tools,
    )

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        response = await client.post(
            url,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()


async def chat_stream(
    messages: list[dict[str, Any]],
    options: dict[str, Any] | None = None,
    response_format: str | dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
):
    url, headers, payload = _build_request(
        messages,
        stream=True,
        options=options,
        response_format=response_format,
        tools=tools,
    )

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=None)) as client:
        async with client.stream(
            "POST",
            url,
            headers=headers,
            json = payload,
        ) as response:
            response.raise_for_status()
        
            async for line in response.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue

                data = line.removeprefix("data:").strip()

                if data == "[DONE]":
                    break

                yield json.loads(data)