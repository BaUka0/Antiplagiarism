from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Antiplagiarism API"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    
    # Falls back to local Postgres when DATABASE_URL is not set.
    DATABASE_URL: str | None = None
    
    # App config
    MODEL_ID: str = "ibm-granite/granite-embedding-311m-multilingual-r2"
    UPLOAD_DIR: str = str(Path.home() / ".codex" / "memories" / "antiplagiarism" / "uploads")
    TESSERACT_CMD: str | None = None
    
    MATCH_THRESHOLD: float = 0.90
    SUSPECT_OVERLAP: float = 0.20
    SKIP_INTRO_CHUNKS: int = 2
    TOP_K_MATCHES: int = 10
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    @field_validator("UPLOAD_DIR", "TESSERACT_CMD", mode="before")
    @classmethod
    def expand_path_values(cls, value):
        if value is None or not isinstance(value, str):
            return value
        return str(Path(value).expanduser())

settings = Settings()
