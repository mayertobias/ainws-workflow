"""
Configuration settings for the workflow-audio microservice
"""

import os
import re
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import field_validator
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings"""
    
    # Service configuration
    service_name: str = "workflow-audio-analysis"
    service_version: str = "1.0.0"
    debug: bool = False
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8001
    
    # Database configuration (NEW - Critical for data persistence)
    database_url: str = "postgresql://postgres:postgres@postgres-audio:5432/workflow_audio"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_pool_timeout: int = 30
    
    # Redis configuration (NEW - For caching and idempotency)
    redis_url: str = "redis://redis-audio:6379/0"
    redis_cache_ttl: int = 3600  # 1 hour
    redis_idempotency_ttl: int = 7200  # 2 hours
    
    # File paths
    base_dir: Path = Path(__file__).parent.parent.parent
    uploads_dir: Path = base_dir / "uploads"
    output_dir: Path = base_dir / "output"
    temp_dir: Path = base_dir / "temp"
    
    # Audio analysis settings
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    supported_formats: list = [".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg"]
    
    @field_validator('max_file_size', mode='before')
    @classmethod
    def parse_file_size(cls, v):
        """Parse file size from string like '100MB' to bytes"""
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            # Parse strings like "100MB", "50GB", "1KB"
            match = re.match(r'^(\d+(?:\.\d+)?)\s*([KMGT]?B?)$', v.upper())
            if match:
                size, unit = match.groups()
                size = float(size)
                
                multipliers = {
                    'B': 1,
                    'KB': 1024,
                    'MB': 1024**2,
                    'GB': 1024**3,
                    'TB': 1024**4,
                    '': 1  # No unit means bytes
                }
                
                return int(size * multipliers.get(unit, 1))
        
        # Fallback to default if parsing fails
        return 100 * 1024 * 1024
    
    # Essentia settings
    essentia_sample_rate: int = 44100
    essentia_frame_size: int = 2048
    essentia_hop_size: int = 1024
    
    # Processing settings
    max_processing_time: int = 300  # 5 minutes
    enable_comprehensive_analysis: bool = True
    
    # Data persistence settings (NEW)
    enable_result_persistence: bool = True
    enable_feature_caching: bool = True
    enable_idempotency: bool = True
    analysis_result_ttl_days: int = 30  # Keep analysis results for 30 days
    
    # Performance settings (NEW)
    max_concurrent_analyses: int = 10
    analysis_queue_size: int = 100
    enable_background_processing: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # CORS settings
    cors_origins: list = ["*"]
    cors_methods: list = ["*"]
    cors_headers: list = ["*"]
    
    class Config:
        env_file = ".env"
        env_prefix = "AUDIO_"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

def ensure_directories():
    """Ensure required directories exist"""
    settings = get_settings()
    for directory in [settings.uploads_dir, settings.output_dir, settings.temp_dir]:
        directory.mkdir(parents=True, exist_ok=True) 