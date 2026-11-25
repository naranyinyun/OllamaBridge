import json
from datetime import datetime, timezone
from typing import Any
import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from ollamabridge.client import chat, chat_stream
from ollamabridge.config import settings
from ollamabridge.converter import get_content, get_content_stream


app = FastAPI()


class ChatMessage(BaseModel):
    role: str
    content: str | None = ""
    images: list[str] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = True
    options: dict[str, Any] | None = None
    format: str | dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None


class ShowRequest(BaseModel):
    name: str | None = None
    model: str | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@app.get("/")
async def ollamabridge_healthcheck():
    return PlainTextResponse("Ollama is running")


@app.get("/api/version")
async def ollama_version():
    return {"version": "0.5.0"}


@app.get("/api/tags")
async def ollama_tags():
    return {
        "models": [
            {
                "name": settings.model,
                "model": settings.model,
                "modified_at": _utc_now_iso(),
                "size": 0,
                "digest": "",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": settings.provider,
                    "families": [settings.provider],
                    "parameter_size": "unknown",
                    "quantization_level": "unknown",
                },
            }
        ]
    }


@app.post("/api/show")
async def ollama_show(request: ShowRequest):
    model_name = request.model or request.name or settings.model
    return {
        "modelfile": f"FROM {model_name}",
        "parameters": "",
        "template": "",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": settings.provider,
            "families": [settings.provider],
            "parameter_size": "unknown",
            "quantization_level": "unknown",
        },
        "capabilities": ["completion", "vision", "tools"],
    }


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    messages = [message.model_dump(exclude_none=True) for message in request.messages]

    if request.stream:
        async def generate():
            sent_done = False
            try:
                async for chunk in chat_stream(
                    messages,
                    options=request.options,
                    response_format=request.format,
                    tools=request.tools,
                ):
                    converted = get_content_stream(chunk, request_model=request.model)
                    if converted is not None:
                        if converted.get("done"):
                            sent_done = True
                        yield json.dumps(converted, ensure_ascii=False) + "\n"

                if not sent_done:
                    yield json.dumps(
                        {
                            "model": request.model,
                            "created_at": _utc_now_iso(),
                            "message": {"role": "assistant", "content": ""},
                            "done": True,
                            "done_reason": "stop",
                        },
                        ensure_ascii=False,
                    ) + "\n"
            except httpx.HTTPStatusError as exc:
                yield json.dumps(
                    {"error": exc.response.text or str(exc)},
                    ensure_ascii=False,
                ) + "\n"
            except httpx.RequestError as exc:
                yield json.dumps({"error": str(exc)}, ensure_ascii=False) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    try:
        response = await chat(
            messages,
            options=request.options,
            response_format=request.format,
            tools=request.tools,
        )
    except httpx.HTTPStatusError as exc:
        return JSONResponse(
            status_code=exc.response.status_code,
            content={"error": exc.response.text or str(exc)},
        )
    except httpx.RequestError as exc:
        return JSONResponse(
            status_code=502,
            content={"error": str(exc)},
        )

    return get_content(response, request_model=request.model)

