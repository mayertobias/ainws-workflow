"""
Health check endpoints for workflow-ml-prediction service
"""

from fastapi import APIRouter, Depends
from ..models.responses import HealthResponse
from ..services.predictor import MLPredictorService

router = APIRouter()

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy")

@router.get("/ready", response_model=HealthResponse)
async def readiness_check(
    predictor: MLPredictorService = Depends(lambda: MLPredictorService())
):
    """Readiness check endpoint"""
    try:
        # Quick check of dependencies
        dependencies_ok = True
        
        # Check if we can access storage (basic check)
        if predictor.minio_client:
            try:
                predictor.minio_client.list_buckets()
            except Exception:
                dependencies_ok = False
        
        # Check Redis connection
        if predictor.redis_client:
            try:
                predictor.redis_client.ping()
            except Exception:
                dependencies_ok = False
        
        if dependencies_ok:
            return HealthResponse(status="ready")
        else:
            return HealthResponse(status="not_ready")
            
    except Exception:
        return HealthResponse(status="not_ready")

@router.get("/live", response_model=HealthResponse)
async def liveness_check():
    """Liveness check endpoint"""
    return HealthResponse(status="alive") 