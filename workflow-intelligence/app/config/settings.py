"""
Configuration settings for workflow-intelligence service
"""

import os
from typing import List, Dict, Any, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # Service configuration
    SERVICE_NAME: str = "workflow-intelligence"
    PORT: int = int(os.getenv("PORT", "8005"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://workflow-gateway:8000",
        "http://workflow-orchestrator:8006"
    ]
    
    # Database settings for insights history
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:postgres@localhost:5435/workflow_intelligence"
    )
    
    # Redis settings for AI response caching
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6382/1")
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "7200"))  # 2 hours
    
    # AI Provider Configuration
    AI_PROVIDER: str = os.getenv("AI_PROVIDER", "auto")  # 'openai', 'gemini', 'ollama', 'auto'
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
    
    # Gemini Configuration (Google AI)
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
    GEMINI_MAX_TOKENS: int = int(os.getenv("GEMINI_MAX_TOKENS", "2000"))
    
    # Anthropic Configuration
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
    ANTHROPIC_TEMPERATURE: float = float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7"))
    ANTHROPIC_MAX_TOKENS: int = int(os.getenv("ANTHROPIC_MAX_TOKENS", "2000"))
    
    # Ollama Configuration (for local models)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama2")
    OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
    
    # HuggingFace Configuration
    HF_MODEL_NAME: str = os.getenv("HF_MODEL_NAME", "microsoft/DialoGPT-medium")
    HF_TASK: str = os.getenv("HF_TASK", "text-generation")
    
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
    WORKFLOW_ML_PREDICTION_URL: str = os.getenv(
        "WORKFLOW_ML_PREDICTION_URL", 
        "http://workflow-ml-prediction:8004"
    )
    
    # Insights configuration
    MAX_PROMPT_LENGTH: int = int(os.getenv("MAX_PROMPT_LENGTH", "8000"))
    ENABLE_PROMPT_CACHING: bool = os.getenv("ENABLE_PROMPT_CACHING", "true").lower() == "true"
    PROMPT_CACHE_TTL: int = int(os.getenv("PROMPT_CACHE_TTL", "3600"))  # 1 hour
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE: int = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))
    MAX_REQUESTS_PER_HOUR: int = int(os.getenv("MAX_REQUESTS_PER_HOUR", "1000"))
    
    # Cost control
    MAX_TOKENS_PER_REQUEST: int = int(os.getenv("MAX_TOKENS_PER_REQUEST", "4000"))
    ENABLE_COST_TRACKING: bool = os.getenv("ENABLE_COST_TRACKING", "true").lower() == "true"
    
    # Analysis types
    SUPPORTED_ANALYSIS_TYPES: List[str] = [
        "musical_meaning",
        "hit_comparison", 
        "novelty_assessment",
        "genre_comparison",
        "production_feedback",
        "strategic_insights",
        "comprehensive_analysis"
    ]
    
    # Agent configuration
    DEFAULT_AGENT_TYPE: str = os.getenv("DEFAULT_AGENT_TYPE", "standard")  # 'standard', 'comprehensive'
    ENABLE_AGENT_SELECTION: bool = os.getenv("ENABLE_AGENT_SELECTION", "true").lower() == "true"
    
    # Performance settings
    ASYNC_WORKERS: int = int(os.getenv("ASYNC_WORKERS", "4"))
    MAX_CONCURRENT_INSIGHTS: int = int(os.getenv("MAX_CONCURRENT_INSIGHTS", "10"))
    INSIGHT_TIMEOUT_SECONDS: int = int(os.getenv("INSIGHT_TIMEOUT_SECONDS", "120"))
    
    def get_llm_config(self, provider: str) -> Dict[str, Any]:
        """Get LLM configuration for a specific provider"""
        if provider == "openai":
            return {
                "api_key": self.OPENAI_API_KEY,
                "model_name": self.OPENAI_MODEL,
                "temperature": self.OPENAI_TEMPERATURE,
                "max_tokens": self.OPENAI_MAX_TOKENS
            }
        elif provider == "gemini":
            return {
                "api_key": self.GEMINI_API_KEY,
                "model_name": self.GEMINI_MODEL,
                "temperature": self.GEMINI_TEMPERATURE,
                "max_output_tokens": self.GEMINI_MAX_TOKENS
            }
        elif provider == "anthropic":
            return {
                "api_key": self.ANTHROPIC_API_KEY,
                "model_name": self.ANTHROPIC_MODEL,
                "temperature": self.ANTHROPIC_TEMPERATURE,
                "max_tokens": self.ANTHROPIC_MAX_TOKENS
            }
        elif provider == "ollama":
            return {
                "base_url": self.OLLAMA_BASE_URL,
                "model": self.OLLAMA_MODEL,
                "temperature": self.OLLAMA_TEMPERATURE
            }
        elif provider == "huggingface":
            return {
                "model_name": self.HF_MODEL_NAME,
                "task": self.HF_TASK
            }
        else:
            return {}
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 