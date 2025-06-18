import os
from typing import List, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "MDM RL System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    DATABASE_URL: str = "postgresql://mdm_user:mdm_password@localhost:5432/mdm_db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    REDIS_URL: str = "redis://localhost:6379"
    REDIS_POOL_SIZE: int = 10

    API_V1_STR: str = "/api/v1"
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    ML_MODELS_PATH: str = "/app/models"
    MODEL_CACHE_TTL: int = 3600  # 1 hour

    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    CHUNK_SIZE: int = 10000
    MAX_WORKERS: int = 4

    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()