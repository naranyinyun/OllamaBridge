import os
import tempfile
from ollamabridge.config import Settings, settings

# 1. Verify current .env loads cleanly (backward-compatible with DEEPSEEK_* defaults)
assert settings.provider
assert settings.base_url
assert settings.model
assert settings.api_key

# 2. Verify switching provider via PROVIDER=<name> + <NAME>_*
with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
    f.write(
        "PROVIDER=openai\n"
        "OPENAI_API_KEY=sk-test-openai\n"
        "OPENAI_MODEL=gpt-4o-mini\n"
    )
    env_path = f.name

try:
    openai_settings = Settings(_env_file=env_path)
    assert openai_settings.provider == "openai"
    assert openai_settings.api_key == "sk-test-openai"
    assert openai_settings.base_url == "https://api.openai.com/v1"
    assert openai_settings.model == "gpt-4o-mini"
finally:
    os.unlink(env_path)

# 3. Verify generic UPSTREAM_* configuration for arbitrary OpenAI-compatible services
with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
    f.write(
        "PROVIDER=qwen\n"
        "UPSTREAM_API_KEY=sk-test-qwen\n"
        "UPSTREAM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/\n"
        "UPSTREAM_MODEL=qwen-plus\n"
    )
    env_path = f.name

try:
    qwen_settings = Settings(_env_file=env_path)
    assert qwen_settings.api_key == "sk-test-qwen"
    assert qwen_settings.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert qwen_settings.model == "qwen-plus"
finally:
    os.unlink(env_path)

print("Config loaded and provider switching tests passed!")