from fastapi import FastAPI
from pydantic import BaseModel
from ollamabridge.client import chat
from ollamabridge.converter import get_content


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
    response = await chat(messages)
    result = get_content(response)
    return result
