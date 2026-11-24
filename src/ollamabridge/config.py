from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

settings = Settings()
