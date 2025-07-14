"""
Health check API endpoints for workflow-orchestrator service
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
import httpx

from ..models.responses import HealthResponse
from ..models.workflow import ServiceHealthStatus, OrchestrationHealthResponse
from ..config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        service=settings.SERVICE_NAME,
        timestamp=datetime.utcnow(),
        version="1.0.0"
    )

@router.get("/detailed")
async def detailed_health_check():
    """Detailed health check with all dependencies."""
    try:
        health_data = {
            "service": settings.SERVICE_NAME,
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "dependencies": {},
            "services": [],
            "queue_status": {},
            "configuration": {}
        }
        
        # Check Redis connection
        health_data["dependencies"]["redis"] = await _check_redis_health()
        
        # Check database connection (if configured)
        health_data["dependencies"]["database"] = await _check_database_health()
        
        # Check microservice dependencies
        health_data["services"] = await _check_microservices_health()
        
        # Check task queue status
        health_data["queue_status"] = await _check_queue_status()
        
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
        ready_checks = {
            "redis_available": False,
            "services_reachable": False,
            "workers_ready": False
        }
        
        # Check Redis connectivity
        redis_health = await _check_redis_health()
        ready_checks["redis_available"] = redis_health.get("status") != "error"
        
        # Check if critical services are reachable
        critical_services = ["audio", "content", "ml_prediction", "intelligence"]
        service_health = await _check_microservices_health()
        healthy_services = [s for s in service_health if s["status"] == "healthy" and s["service_name"] in critical_services]
        ready_checks["services_reachable"] = len(healthy_services) >= 2  # At least 2 critical services
        
        # Check worker readiness (simple check)
        ready_checks["workers_ready"] = True  # Workers are always ready in this implementation
        
        # Service is ready if basic requirements are met
        is_ready = (
            ready_checks["redis_available"] and 
            ready_checks["services_reachable"]
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
        return {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat(),
            "service": settings.SERVICE_NAME,
            "uptime_info": "Service is running"
        }
    except Exception as e:
        logger.error(f"Error in liveness check: {e}")
        raise HTTPException(status_code=500, detail=f"Liveness check failed: {str(e)}")

@router.get("/services")
async def services_health():
    """Check health of all microservice dependencies."""
    try:
        return {
            "services": await _check_microservices_health(),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error checking services health: {e}")
        raise HTTPException(status_code=500, detail=f"Services health check failed: {str(e)}")

# Helper functions

async def _check_redis_health() -> Dict[str, Any]:
    """Check Redis connection health."""
    try:
        import redis.asyncio as redis
        redis_client = redis.from_url(settings.REDIS_URL)
        
        # Test connection
        await redis_client.ping()
        
        # Get basic info
        info = await redis_client.info()
        await redis_client.close()
        
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
        if settings.DATABASE_URL and "postgresql" in settings.DATABASE_URL:
            # For now, just check if URL is configured
            # In production, you'd test actual connection
            return {
                "status": "configured",
                "type": "postgresql",
                "url_configured": True
            }
        else:
            return {
                "status": "not_configured",
                "message": "Database URL not provided or not PostgreSQL"
            }
            
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

async def _check_microservices_health() -> list:
    """Check health of all microservice dependencies."""
    services = []
    service_urls = settings.get_service_urls()
    
    async def check_service(service_name: str, url: str) -> ServiceHealthStatus:
        """Check individual service health."""
        start_time = datetime.utcnow()
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{url}/health")
                end_time = datetime.utcnow()
                response_time = (end_time - start_time).total_seconds() * 1000
                
                if response.status_code == 200:
                    return ServiceHealthStatus(
                        service_name=service_name,
                        url=url,
                        status="healthy",
                        response_time_ms=response_time,
                        last_checked=end_time
                    )
                else:
                    return ServiceHealthStatus(
                        service_name=service_name,
                        url=url,
                        status="unhealthy",
                        response_time_ms=response_time,
                        last_checked=end_time,
                        error_message=f"HTTP {response.status_code}"
                    )
                    
        except Exception as e:
            end_time = datetime.utcnow()
            return ServiceHealthStatus(
                service_name=service_name,
                url=url,
                status="error",
                last_checked=end_time,
                error_message=str(e)
            )
    
    # Check all services concurrently
    tasks = [check_service(name, url) for name, url in service_urls.items()]
    service_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in service_results:
        if isinstance(result, ServiceHealthStatus):
            services.append(result.dict())
        else:
            # Handle exception
            services.append({
                "service_name": "unknown",
                "status": "error",
                "error_message": str(result),
                "last_checked": datetime.utcnow()
            })
    
    return services

async def _check_queue_status() -> Dict[str, Any]:
    """Check task queue status."""
    try:
        # In a real implementation, you'd check queue length from Redis
        # For now, return basic status
        return {
            "queue_length": 0,  # Would get from Redis
            "workers_active": 5,  # From orchestrator
            "queue_healthy": True
        }
    except Exception as e:
        logger.warning(f"Queue status check failed: {e}")
        return {
            "queue_healthy": False,
            "error": str(e)
        }

def _check_configuration() -> Dict[str, Any]:
    """Check service configuration."""
    try:
        config_status = {
            "service_name": settings.SERVICE_NAME,
            "port": settings.PORT,
            "debug_mode": settings.DEBUG,
            "workflow_timeout": settings.WORKFLOW_TIMEOUT_SECONDS,
            "max_parallel_workflows": settings.MAX_PARALLEL_WORKFLOWS,
            "retry_attempts": settings.MAX_RETRY_ATTEMPTS,
            "rate_limiting": {
                "max_requests_per_minute": settings.MAX_REQUESTS_PER_MINUTE,
                "max_concurrent_workflows": settings.MAX_CONCURRENT_WORKFLOWS
            },
            "timeouts": settings.get_service_timeouts(),
            "templates_available": len(settings.DEFAULT_WORKFLOW_TEMPLATES)
        }
        
        # Check service URLs configuration
        config_status["service_urls"] = {
            name: bool(url) for name, url in settings.get_service_urls().items()
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
        # Check Redis status
        redis_status = health_data.get("dependencies", {}).get("redis", {}).get("status")
        if redis_status == "error":
            return "degraded"
        
        # Check critical services
        services = health_data.get("services", [])
        critical_services = ["audio", "content", "ml_prediction", "intelligence"]
        
        healthy_critical = 0
        for service in services:
            if service.get("service_name") in critical_services and service.get("status") == "healthy":
                healthy_critical += 1
        
        if healthy_critical < 2:
            return "degraded"
        
        # Check queue status
        queue_status = health_data.get("queue_status", {})
        if not queue_status.get("queue_healthy", True):
            return "degraded"
        
        return "healthy"
        
    except Exception as e:
        logger.error(f"Error determining overall status: {e}")
        return "unknown" 