from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)


class Message(APIModel):
    role: str
    content: str | None = ""
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class Choice(APIModel):
    message: Message
    finish_reason: str | None = None


class ChatCompletion(APIModel):
    model: str | None = None
    choices: list[Choice] = Field(min_length=1)
    usage: dict[str, Any] | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _detect_image_mime(b64_data: str) -> str:
    if b64_data.startswith("iVBORw"):
        return "image/png"
    if b64_data.startswith("R0lGOD"):
        return "image/gif"
    if b64_data.startswith("UklGR"):
        return "image/webp"
    return "image/jpeg"


def convert_messages_to_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Ollama chat messages (including images and tool calls) to OpenAI format."""
    converted: list[dict[str, Any]] = []
    for msg in messages:
        item: dict[str, Any] = {"role": msg["role"]}
        images = msg.get("images") or []
        content = msg.get("content")

        if images:
            parts: list[dict[str, Any]] = []
            if content:
                parts.append({"type": "text", "text": content})
            for img in images:
                if img.startswith(("data:", "http://", "https://")):
                    url = img
                else:
                    mime = _detect_image_mime(img)
                    url = f"data:{mime};base64,{img}"
                parts.append({"type": "image_url", "image_url": {"url": url}})
            item["content"] = parts
        else:
            item["content"] = content if content is not None else ""

        if msg.get("tool_calls"):
            item["tool_calls"] = msg["tool_calls"]
        if msg.get("tool_call_id"):
            item["tool_call_id"] = msg["tool_call_id"]
        converted.append(item)
    return converted


def _build_ollama_message(
    role: str,
    content: str | None,
    thinking: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "role": role,
        "content": content or "",
    }
    if thinking:
        msg["thinking"] = thinking
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def get_content(data: dict[str, Any], request_model: str | None = None) -> dict[str, Any]:
    response = ChatCompletion.model_validate(data)
    choice = response.choices[0]
    message = choice.message

    result: dict[str, Any] = {
        "model": request_model or response.model,
        "created_at": _utc_now_iso(),
        "message": _build_ollama_message(
            role=message.role,
            content=message.content,
            thinking=message.reasoning_content,
            tool_calls=message.tool_calls,
        ),
        "done": True,
        "done_reason": choice.finish_reason or "stop",
    }

    if response.usage:
        if "prompt_tokens" in response.usage:
            result["prompt_eval_count"] = response.usage["prompt_tokens"]
        if "completion_tokens" in response.usage:
            result["eval_count"] = response.usage["completion_tokens"]

    return result


def get_content_stream(
    data: dict[str, Any], request_model: str | None = None
) -> dict[str, Any] | None:
    choices = data.get("choices") or []
    if not choices:
        return None

    choice = choices[0]
    delta = choice.get("delta") or {}
    finish_reason = choice.get("finish_reason")
    done = finish_reason is not None

    result: dict[str, Any] = {
        "model": request_model or data.get("model"),
        "created_at": _utc_now_iso(),
        "message": _build_ollama_message(
            role=delta.get("role") or "assistant",
            content=delta.get("content"),
            thinking=delta.get("reasoning_content"),
            tool_calls=delta.get("tool_calls"),
        ),
        "done": done,
    }

    if done:
        result["done_reason"] = finish_reason

    usage = data.get("usage")
    if done and isinstance(usage, dict):
        if "prompt_tokens" in usage:
            result["prompt_eval_count"] = usage["prompt_tokens"]
        if "completion_tokens" in usage:
            result["eval_count"] = usage["completion_tokens"]

    return result