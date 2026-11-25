# OllamaBridge

OllamaBridge 可以模拟 Ollama 客户端的行为，以便通过 Ollama 使用云端 OpenAI 兼容的大模型接口。

目前可以模拟文本、流式文本和图片能力的行为。

## 使用

将此仓库克隆到本地后，使用 `uv`：

```bash
uv sync
```

以便安装依赖。

```bash
uv run uvicorn ollamabridge.main:app --host 127.0.0.1 --port 11434 
```

即可在 11434 端口上运行服务。