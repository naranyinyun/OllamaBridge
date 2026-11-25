from ollamabridge.converter import convert_messages_to_openai, get_content, get_content_stream

data = {
    "model": "deepseek-chat",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "你好，我是 DeepSeek。",
                "reasoning_content": "先礼貌打招呼",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 8},
}

result = get_content(data)

print(result)

assert result["model"] == "deepseek-chat"
assert result["message"]["role"] == "assistant"
assert result["message"]["content"] == "你好，我是 DeepSeek。"
assert result["message"]["thinking"] == "先礼貌打招呼"
assert result["done"] is True
assert result["done_reason"] == "stop"
assert result["prompt_eval_count"] == 12
assert result["eval_count"] == 8
assert result["created_at"].endswith("Z")

# Test image modality conversion (PNG base64 + URL)
converted_msgs = convert_messages_to_openai([
    {
        "role": "user",
        "content": "请描述这张图片",
        "images": ["iVBORw0KGgoAAAANSUhEUg==", "data:image/jpeg;base64,/9j/4AAQ"],
    }
])
assert len(converted_msgs) == 1
parts = converted_msgs[0]["content"]
assert parts[0] == {"type": "text", "text": "请描述这张图片"}
assert parts[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="}}
assert parts[2] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"}}

# Test stream chunk conversion with reasoning_content
stream_chunk = get_content_stream(
    {
        "model": "deepseek-reasoner",
        "choices": [
            {"delta": {"content": "答案", "reasoning_content": "思考中"}, "finish_reason": None}
        ],
    },
    request_model="llama3.2",
)
assert stream_chunk["model"] == "llama3.2"
assert stream_chunk["message"]["content"] == "答案"
assert stream_chunk["message"]["thinking"] == "思考中"
assert stream_chunk["done"] is False

print("All tests passed!")