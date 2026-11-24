from ollamabridge.config import settings

assert settings.deepseek_base_url
assert settings.deepseek_model
assert settings.deepseek_api_key

print("Config loaded successfully!")