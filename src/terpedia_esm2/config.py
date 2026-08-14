from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ESM2_", case_sensitive=False)

    model_id: str = "facebook/esm2_t6_8M_UR50D"
    model_revision: str | None = "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"
    device: str = "auto"
    max_sequence_length: int = Field(default=1022, ge=1)
    max_batch_sequences: int = Field(default=32, ge=1)
    max_batch_tokens: int = Field(default=8192, ge=1)
    batch_wait_ms: float = Field(default=8.0, ge=0, le=1000)
    queue_capacity: int = Field(default=512, ge=1)
    normalize_embeddings: bool = False
    trust_remote_code: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
