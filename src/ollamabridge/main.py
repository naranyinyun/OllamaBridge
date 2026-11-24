import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ollamabridge.client import chat, chat_stream
from ollamabridge.converter import get_content, get_content_stream


app = FastAPI()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False



@app.get("/")
async def ollamabridge_healthcheck():
    return {
        "message": "OllamaBridge is running properly."
    }

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    messages = [message.model_dump() for message in request.messages]

    if request.stream:
        async def generate():
            async for chunk in chat_stream(messages):
                converted = get_content_stream(chunk)
                if converted is not None:
                    yield json.dumps(converted, ensure_ascii=False) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    response = await chat(messages)
    result = get_content(response)
    return result
