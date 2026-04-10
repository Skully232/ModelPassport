from pydantic_settings import BaseSettings
from functools import lru_cache
from pydantic import ValidationError
import os

class Settings(BaseSettings):
    # App
    APP_NAME: str = "ModelPassport"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    # Google AI
    GEMINI_API_KEY: str
    GOOGLE_CLOUD_PROJECT: str = ""

    # Certificate
    CERT_PREFIX: str = "MP-2026"
    VERIFY_BASE_URL: str = "https://modelpassport.ai/verify"

    # Security
    SECRET_KEY: str = "dev-secret-change-in-production"
    API_KEY_HEADER: str = "X-API-Key"

    # Storage
    UPLOAD_MAX_SIZE_MB: int = 50
    DELETE_UPLOADS_AFTER_AUDIT: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as e:
        # Intercept the validation error to provide a clear message if GEMINI_API_KEY is missing
        if any("GEMINI_API_KEY" in str(err.get("loc", "")) for err in e.errors()):
            raise ValueError("GEMINI_API_KEY is required. Please set it in your .env file") from None
        raise

settings = get_settings()
