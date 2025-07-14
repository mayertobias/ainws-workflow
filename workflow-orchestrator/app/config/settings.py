"""
Configuration settings for workflow-orchestrator service
"""

import os
from typing import List, Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Configuration settings for workflow-orchestrator service."""
    
    # Service Configuration
    SERVICE_NAME: str = "workflow-orchestrator"
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    PORT: int = Field(default=8006, description="Service port")
    
    # Database Configuration
    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:5436/workflow_orchestrator",
        description="PostgreSQL database URL"
    )
    
    # Redis Configuration for task queue and state management
    REDIS_URL: str = Field(
        default="redis://localhost:6383/0",
        description="Redis URL for task queue"
    )
    
    # Celery Configuration
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6383/1",
        description="Celery broker URL"
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6383/2",
        description="Celery result backend URL"
    )
    
    # Service Discovery - Microservice URLs
    WORKFLOW_AUDIO_URL: str = Field(
        default="http://localhost:8001",
        description="workflow-audio service URL"
    )
    WORKFLOW_CONTENT_URL: str = Field(
        default="http://localhost:8002",
        description="workflow-content service URL"
    )
    WORKFLOW_ML_TRAINING_URL: str = Field(
        default="http://localhost:8003",
        description="workflow-ml-training service URL"
    )
    WORKFLOW_ML_PREDICTION_URL: str = Field(
        default="http://localhost:8004",
        description="workflow-ml-prediction service URL"
    )
    WORKFLOW_INTELLIGENCE_URL: str = Field(
        default="http://localhost:8005",
        description="workflow-intelligence service URL"
    )
    WORKFLOW_STORAGE_URL: str = Field(
        default="http://localhost:8007",
        description="workflow-storage service URL"
    )
    
    # Workflow Configuration
    MAX_PARALLEL_WORKFLOWS: int = Field(
        default=50,
        description="Maximum number of parallel workflows"
    )
    WORKFLOW_TIMEOUT_SECONDS: int = Field(
        default=3600,  # 1 hour
        description="Default workflow timeout in seconds"
    )
    MAX_RETRY_ATTEMPTS: int = Field(
        default=3,
        description="Maximum retry attempts for failed tasks"
    )
    RETRY_DELAY_SECONDS: int = Field(
        default=5,
        description="Delay between retry attempts"
    )
    
    # Service Timeouts
    SERVICE_TIMEOUT_SECONDS: int = Field(
        default=300,  # 5 minutes
        description="Timeout for service calls"
    )
    AUDIO_ANALYSIS_TIMEOUT: int = Field(
        default=180,  # 3 minutes
        description="Timeout for audio analysis"
    )
    ML_TRAINING_TIMEOUT: int = Field(
        default=7200,  # 2 hours
        description="Timeout for ML training"
    )
    AI_INSIGHTS_TIMEOUT: int = Field(
        default=120,  # 2 minutes
        description="Timeout for AI insights"
    )
    
    # Circuit Breaker Configuration
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(
        default=5,
        description="Number of failures before opening circuit"
    )
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = Field(
        default=60,
        description="Recovery timeout for circuit breaker"
    )
    
    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE: int = Field(
        default=1000,
        description="Maximum requests per minute"
    )
    MAX_CONCURRENT_WORKFLOWS: int = Field(
        default=100,
        description="Maximum concurrent workflows"
    )
    
    # Health Check Configuration
    HEALTH_CHECK_INTERVAL: int = Field(
        default=30,
        description="Health check interval in seconds"
    )
    SERVICE_DEPENDENCY_TIMEOUT: int = Field(
        default=10,
        description="Timeout for dependency health checks"
    )
    
    # Workflow Templates
    DEFAULT_WORKFLOW_TEMPLATES: List[str] = Field(
        default=[
            "comprehensive_analysis",
            "hit_prediction",
            "audio_only_analysis",
            "ai_insights_only",
            "production_analysis"
        ],
        description="Available workflow templates"
    )
    
    # Notification Configuration
    ENABLE_NOTIFICATIONS: bool = Field(
        default=True,
        description="Enable workflow completion notifications"
    )
    NOTIFICATION_WEBHOOK_URL: Optional[str] = Field(
        default=None,
        description="Webhook URL for notifications"
    )
    
    # Monitoring Configuration
    ENABLE_METRICS: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    METRICS_RETENTION_DAYS: int = Field(
        default=30,
        description="Days to retain metrics data"
    )
    
    # Security Configuration
    API_KEY: Optional[str] = Field(
        default=None,
        description="API key for service authentication"
    )
    REQUIRE_AUTH: bool = Field(
        default=False,
        description="Require authentication for API access"
    )
    
    # Development Configuration
    MOCK_SERVICES: bool = Field(
        default=False,
        description="Use mock services for development"
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    def get_service_urls(self) -> Dict[str, str]:
        """Get all microservice URLs."""
        return {
            "audio": self.WORKFLOW_AUDIO_URL,
            "content": self.WORKFLOW_CONTENT_URL,
            "ml_training": self.WORKFLOW_ML_TRAINING_URL,
            "ml_prediction": self.WORKFLOW_ML_PREDICTION_URL,
            "intelligence": self.WORKFLOW_INTELLIGENCE_URL,
            "storage": self.WORKFLOW_STORAGE_URL
        }
    
    def get_service_timeouts(self) -> Dict[str, int]:
        """Get service-specific timeouts."""
        return {
            "audio": self.AUDIO_ANALYSIS_TIMEOUT,
            "content": self.SERVICE_TIMEOUT_SECONDS,
            "ml_training": self.ML_TRAINING_TIMEOUT,
            "ml_prediction": self.SERVICE_TIMEOUT_SECONDS,
            "intelligence": self.AI_INSIGHTS_TIMEOUT,
            "storage": self.SERVICE_TIMEOUT_SECONDS
        }


# Global settings instance
settings = Settings() 