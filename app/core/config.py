from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Explicit alias so OPENAI_API_KEY is always recognized (avoids any env name mapping confusion).
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 800
    chunk_overlap: int = 100


def _project_root() -> Path:
    # app/core/config.py -> core/ -> app/ -> project root
    return Path(__file__).resolve().parents[2]


# Load .env using an absolute path so it works no matter the current working directory.
load_dotenv(dotenv_path=str(_project_root() / ".env"), override=False)
settings = Settings()

# Ensure libraries that read directly from the environment can find the key
if settings.openai_api_key and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key


