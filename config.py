from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # API Keys
    groq_api_key: Optional[str] = None
    huggingfacehub_api_token: Optional[str] = None
    
    # Application Settings
    app_name: str = "ChatBot Backend"
    app_version: str = "1.0.0"
    debug: bool = True
    
    # CORS Settings
    cors_origins: list = ["*"]
    
    # File Upload Settings (in-memory processing)
    max_file_size_mb: int = 10
    allowed_extensions: set = {
        "txt", "pdf", "doc", "docx", 
        "png", "jpg", "jpeg", "gif", "bmp"
    }
    
    # Vector Store Settings (in-memory)
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Model Settings
    default_model: str = "groq_llama3_70b"
    temperature: float = 0.3
    max_tokens: int = 4096
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

settings = Settings()

# No directories needed - everything in memory
