"""
Configuration settings for workflow-content service
"""

import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # Service configuration
    SERVICE_NAME: str = "workflow-content"
    PORT: int = int(os.getenv("PORT", "8002"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3002", 
        "http://localhost:3003",
        "http://localhost:8000",
        "http://workflow-gateway:8000",
        "http://workflow-orchestrator:8006",
        "*"  # Allow all origins for development
    ]
    
    # Database settings
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:postgres@localhost:5432/workflow_content"
    )
    
    # Redis settings for caching
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/1")
    
    # Directories
    LYRICS_DIR: str = os.getenv("LYRICS_DIR", "/tmp/lyrics")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "/tmp/output")
    
    # NLP settings
    SPACY_MODEL: str = "en_core_web_sm"
    MAX_TEXT_LENGTH: int = 10000
    
    # Analysis settings
    MAX_THEMES: int = 10
    DEFAULT_LANGUAGE: str = "en"
    
    # Service discovery
    WORKFLOW_ORCHESTRATOR_URL: str = os.getenv(
        "WORKFLOW_ORCHESTRATOR_URL", 
        "http://workflow-orchestrator:8006"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 