from pydantic import BaseModel, ConfigDict, Field

class APIModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)


class Message(APIModel):
    role: str
    content: str


class Choice(APIModel):
    message: Message
    finish_reason: str | None = None


class ChatCompletion(APIModel):
    model: str | None = None
    choices: list[Choice] = Field(min_length=1)

def get_content(data):
    response = ChatCompletion.model_validate(data)
    message = response.choices[0].message

    return {
        "model": response.model,
        "message": {
            "role": message.role,
            "content": message.content,
        },
        "done": True,
    }

def get_content_stream(data):
    choices = data.get("choices") or []
    if not choices:
        return None

    choice = choices[0]
    delta = choice.get("delta") or {}
    finish_reason = choice.get("finish_reason")

    return {
        "model": data.get("model"),
        "message": {
            "role": "assistant",
            "content": delta.get("content") or "",
        },
        "done": finish_reason is not None,
    }