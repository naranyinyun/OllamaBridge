from ollamabridge.converter import get_content

data = {
    "model": "deepseek-chat",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "你好，我是 DeepSeek。"
            },
            "finish_reason": "stop"
        }
    ]
}

result = get_content(data)

print(result)

assert result["model"] == "deepseek-chat"
assert result["message"]["role"] == "assistant"
assert result["message"]["content"] == "你好，我是 DeepSeek。"
assert result["done"] is True

print("All tests passed!")