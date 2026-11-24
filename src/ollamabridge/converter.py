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