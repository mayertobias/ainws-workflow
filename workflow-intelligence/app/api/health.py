"""
Health check API endpoints for workflow-intelligence service
"""

import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends

from ..models.responses import HealthResponse
from ..services.llm_providers import LLMProviderFactory
from ..services.agents import AgentManager
from ..services.intelligence_service import IntelligenceService
from ..config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Service instances for health checks
agent_manager = AgentManager()

async def get_intelligence_service():
    """Dependency to get intelligence service instance."""
    return IntelligenceService()

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="workflow-intelligence",
        timestamp=datetime.utcnow(),
        version="1.0.0"
    )

@router.get("/detailed")
async def detailed_health_check():
    """Detailed health check with dependency status."""
    try:
        health_data = {
            "service": settings.SERVICE_NAME,
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "dependencies": {},
            "providers": {},
            "agents": {},
            "configuration": {}
        }
        
        # Check Redis connection
        health_data["dependencies"]["redis"] = await _check_redis_health()
        
        # Check database connection (if needed)
        health_data["dependencies"]["database"] = await _check_database_health()
        
        # Check LLM providers
        health_data["providers"] = await _check_provider_health()
        
        # Check agents
        health_data["agents"] = await _check_agent_health()
        
        # Check configuration
        health_data["configuration"] = _check_configuration()
        
        # Determine overall status
        overall_status = _determine_overall_status(health_data)
        health_data["status"] = overall_status
        
        return health_data
        
    except Exception as e:
        logger.error(f"Error in detailed health check: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/ready")
async def readiness_check():
    """Readiness probe for Kubernetes/container orchestration."""
    try:
        # Check if service is ready to handle requests
        ready_checks = {
            "providers_available": False,
            "agents_ready": False,
            "dependencies_ok": False
        }
        
        # Check if at least one provider is available
        provider_info = LLMProviderFactory.get_provider_info()
        ready_checks["providers_available"] = any(
            info.get("available", False) for info in provider_info.values()
        )
        
        # Check if agents can be created
        try:
            agent = await agent_manager.get_agent("standard", "auto")
            ready_checks["agents_ready"] = agent is not None
        except Exception:
            ready_checks["agents_ready"] = False
        
        # Check basic dependencies
        redis_health = await _check_redis_health()
        ready_checks["dependencies_ok"] = redis_health.get("status") != "error"
        
        # Service is ready if basic requirements are met
        is_ready = (
            ready_checks["providers_available"] and 
            ready_checks["dependencies_ok"]
        )
        
        if is_ready:
            return {
                "status": "ready",
                "timestamp": datetime.utcnow().isoformat(),
                "checks": ready_checks
            }
        else:
            raise HTTPException(
                status_code=503, 
                detail={
                    "status": "not_ready",
                    "timestamp": datetime.utcnow().isoformat(),
                    "checks": ready_checks
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in readiness check: {e}")
        raise HTTPException(status_code=500, detail=f"Readiness check failed: {str(e)}")

@router.get("/live")
async def liveness_check():
    """Liveness probe for Kubernetes/container orchestration."""
    try:
        # Basic liveness check - service is running
        return {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat(),
            "service": settings.SERVICE_NAME,
            "uptime_info": "Service is running"
        }
    except Exception as e:
        logger.error(f"Error in liveness check: {e}")
        raise HTTPException(status_code=500, detail=f"Liveness check failed: {str(e)}")

@router.get("/providers")
async def provider_health():
    """Check health of all LLM providers."""
    try:
        return await _check_provider_health()
    except Exception as e:
        logger.error(f"Error checking provider health: {e}")
        raise HTTPException(status_code=500, detail=f"Provider health check failed: {str(e)}")

@router.get("/agents")
async def agent_health():
    """Check health of all agents."""
    try:
        return await _check_agent_health()
    except Exception as e:
        logger.error(f"Error checking agent health: {e}")
        raise HTTPException(status_code=500, detail=f"Agent health check failed: {str(e)}")

@router.post("/test")
async def test_generation(
    intelligence_service: IntelligenceService = Depends(get_intelligence_service)
):
    """Test AI generation capability end-to-end."""
    try:
        from ..models.intelligence import (
            InsightGenerationRequest, AnalysisType, AgentType, AIProvider,
            AudioFeaturesInput, SongMetadata
        )
        
        # Create a test request
        test_request = InsightGenerationRequest(
            audio_features=AudioFeaturesInput(
                tempo=120.0,
                energy=0.75,
                valence=0.6
            ),
            song_metadata=SongMetadata(
                title="Test Song",
                artist="Test Artist",
                genre="pop"
            ),
            analysis_types=[AnalysisType.MUSICAL_MEANING],
            agent_type=AgentType.STANDARD,
            ai_provider=AIProvider.AUTO,
            use_cache=False,
            max_tokens=500
        )
        
        # Generate test insights
        start_time = datetime.utcnow()
        response = await intelligence_service.generate_insights(test_request)
        end_time = datetime.utcnow()
        
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        return {
            "status": "success" if response.status.value != "failed" else "error",
            "test_type": "end_to_end_generation",
            "processing_time_ms": processing_time,
            "response_status": response.status.value,
            "cached": response.cached,
            "insights_generated": len(response.insights.analysis_types_completed) if response.insights else 0,
            "error_message": response.error_message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in test generation: {e}")
        return {
            "status": "error",
            "test_type": "end_to_end_generation",
            "error_message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Helper functions for health checks

async def _check_redis_health() -> Dict[str, Any]:
    """Check Redis connection health."""
    try:
        import redis
        redis_client = redis.from_url(settings.REDIS_URL)
        
        # Test connection
        redis_client.ping()
        
        # Get basic info
        info = redis_client.info()
        
        return {
            "status": "healthy",
            "connected": True,
            "redis_version": info.get("redis_version", "unknown"),
            "used_memory": info.get("used_memory_human", "unknown"),
            "connected_clients": info.get("connected_clients", 0)
        }
        
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        return {
            "status": "error",
            "connected": False,
            "error": str(e)
        }

async def _check_database_health() -> Dict[str, Any]:
    """Check database connection health."""
    try:
        # For now, we'll assume database is healthy if configured
        # In a real implementation, you'd test the actual connection
        
        if settings.DATABASE_URL:
            return {
                "status": "configured",
                "url": settings.DATABASE_URL.replace(
                    settings.DATABASE_URL.split('@')[0].split('//')[-1], 
                    "***"
                ) if '@' in settings.DATABASE_URL else "configured"
            }
        else:
            return {
                "status": "not_configured",
                "message": "Database URL not provided"
            }
            
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

async def _check_provider_health() -> Dict[str, Any]:
    """Check LLM provider health."""
    try:
        provider_info = LLMProviderFactory.get_provider_info()
        
        health_summary = {
            "total_providers": len(provider_info),
            "available_providers": 0,
            "providers": provider_info
        }
        
        # Count available providers
        for provider_name, info in provider_info.items():
            if info.get("available", False):
                health_summary["available_providers"] += 1
        
        # Try auto-detection
        try:
            auto_provider = LLMProviderFactory.auto_detect_provider()
            health_summary["auto_detected"] = auto_provider.__class__.__name__ if auto_provider else None
        except Exception as e:
            health_summary["auto_detected_error"] = str(e)
        
        return health_summary
        
    except Exception as e:
        logger.error(f"Provider health check failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

async def _check_agent_health() -> Dict[str, Any]:
    """Check agent health."""
    try:
        agent_health = await agent_manager.health_check()
        
        # Add test generation
        test_result = await agent_manager.test_agent_generation()
        agent_health["test_generation"] = test_result
        
        return agent_health
        
    except Exception as e:
        logger.error(f"Agent health check failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

def _check_configuration() -> Dict[str, Any]:
    """Check service configuration."""
    try:
        config_status = {
            "ai_provider": settings.AI_PROVIDER,
            "cache_enabled": settings.REDIS_URL is not None,
            "cache_ttl": settings.CACHE_TTL_SECONDS,
            "max_tokens": settings.MAX_TOKENS_PER_REQUEST,
            "rate_limiting": {
                "requests_per_minute": settings.MAX_REQUESTS_PER_MINUTE,
                "requests_per_hour": settings.MAX_REQUESTS_PER_HOUR
            },
            "analysis_types": len(settings.SUPPORTED_ANALYSIS_TYPES),
            "debug_mode": settings.DEBUG
        }
        
        # Check for API keys (without revealing them)
        config_status["api_keys"] = {
            "openai": bool(settings.OPENAI_API_KEY),
            "gemini": bool(settings.GEMINI_API_KEY),
            "anthropic": bool(settings.ANTHROPIC_API_KEY)
        }
        
        return config_status
        
    except Exception as e:
        logger.error(f"Configuration check failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

def _determine_overall_status(health_data: Dict[str, Any]) -> str:
    """Determine overall service status from health data."""
    try:
        # Check if any critical components are failing
        redis_status = health_data.get("dependencies", {}).get("redis", {}).get("status")
        providers_available = health_data.get("providers", {}).get("available_providers", 0)
        
        # Critical failures
        if redis_status == "error":
            return "degraded"  # Can still work without cache
        
        if providers_available == 0:
            return "unhealthy"  # Cannot work without providers
        
        # Check agents
        agent_status = health_data.get("agents", {})
        if agent_status.get("status") == "error":
            return "degraded"
        
        return "healthy"
        
    except Exception as e:
        logger.error(f"Error determining overall status: {e}")
        return "unknown" 