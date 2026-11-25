from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "openai": "https://api.openai.com/v1",
}


class Settings(BaseSettings):
    provider: str = "deepseek"

    upstream_api_key: str | None = None
    upstream_base_url: str | None = None
    upstream_model: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    def _get_extra(self, key: str) -> str | None:
        if not self.model_extra:
            return None
        for k, v in self.model_extra.items():
            if k.lower() == key.lower() and isinstance(v, str) and v.strip():
                return v.strip()
        return None

    @property
    def api_key(self) -> str:
        prefix = self.provider.strip().lower()
        return (self.upstream_api_key or self._get_extra(f"{prefix}_api_key") or "").strip()

    @property
    def base_url(self) -> str:
        prefix = self.provider.strip().lower()
        url = (
            self.upstream_base_url
            or self._get_extra(f"{prefix}_base_url")
            or DEFAULT_BASE_URLS.get(prefix, "")
        )
        return url.strip().rstrip("/")

    @property
    def model(self) -> str:
        prefix = self.provider.strip().lower()
        return (self.upstream_model or self._get_extra(f"{prefix}_model") or "").strip()

    @model_validator(mode="after")
    def validate_upstream_config(self) -> "Settings":
        prefix = self.provider.strip().upper()
        missing = []
        if not self.api_key:
            missing.append(f"UPSTREAM_API_KEY or {prefix}_API_KEY")
        if not self.base_url:
            missing.append(f"UPSTREAM_BASE_URL or {prefix}_BASE_URL")
        if not self.model:
            missing.append(f"UPSTREAM_MODEL or {prefix}_MODEL")
        if missing:
            raise ValueError(
                f"Missing upstream configuration for provider '{self.provider}': "
                + ", ".join(missing)
            )
        return self


settings = Settings()
