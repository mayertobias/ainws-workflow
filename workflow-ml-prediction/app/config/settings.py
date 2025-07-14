"""
Configuration settings for workflow-ml-prediction service
"""

import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # Service configuration
    SERVICE_NAME: str = "workflow-ml-prediction"
    PORT: int = int(os.getenv("PORT", "8004"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://workflow-gateway:8000",
        "http://workflow-orchestrator:8006"
    ]
    
    # Redis settings for prediction cache
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/3")
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1 hour
    
    # MinIO/S3 settings for model storage
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "ml-models")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"
    
    # Model storage and registry
    MODELS_DIR: str = os.getenv("MODELS_DIR", "/app/models")
    MODEL_REGISTRY_PATH: str = os.getenv("MODEL_REGISTRY_PATH", "/app/models/model_registry.json")
    
    # Model serving settings
    MAX_CACHED_MODELS: int = int(os.getenv("MAX_CACHED_MODELS", "10"))
    MODEL_LOAD_TIMEOUT_SECONDS: int = int(os.getenv("MODEL_LOAD_TIMEOUT_SECONDS", "30"))
    PREDICTION_TIMEOUT_SECONDS: int = int(os.getenv("PREDICTION_TIMEOUT_SECONDS", "5"))
    
    # Model registry auto-update settings
    MODEL_REGISTRY_UPDATE_INTERVAL_HOURS: int = int(os.getenv("MODEL_REGISTRY_UPDATE_INTERVAL_HOURS", "24"))
    MODEL_REGISTRY_STARTUP_DELAY_MINUTES: int = int(os.getenv("MODEL_REGISTRY_STARTUP_DELAY_MINUTES", "60"))
    
    # Batch prediction settings
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "1000"))
    BATCH_PROCESSING_TIMEOUT_MINUTES: int = int(os.getenv("BATCH_PROCESSING_TIMEOUT_MINUTES", "30"))
    
    # Feature validation
    MIN_AUDIO_FEATURES: int = 4
    REQUIRED_FEATURES: List[str] = [
        "tempo", "energy", "danceability", "valence"
    ]
    
    # Service discovery
    WORKFLOW_AUDIO_URL: str = os.getenv(
        "WORKFLOW_AUDIO_URL", 
        "http://workflow-audio:8001"
    )
    WORKFLOW_CONTENT_URL: str = os.getenv(
        "WORKFLOW_CONTENT_URL", 
        "http://workflow-content:8002"
    )
    WORKFLOW_ML_TRAINING_URL: str = os.getenv(
        "WORKFLOW_ML_TRAINING_URL", 
        "http://workflow-ml-training:8003"
    )
    
    # Performance settings
    ASYNC_WORKERS: int = int(os.getenv("ASYNC_WORKERS", "4"))
    MAX_CONCURRENT_PREDICTIONS: int = int(os.getenv("MAX_CONCURRENT_PREDICTIONS", "100"))
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 