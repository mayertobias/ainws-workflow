"""
Health check endpoints for workflow-content service
"""

from fastapi import APIRouter
from ..models.responses import HealthResponse

router = APIRouter()

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy")

@router.get("/ready", response_model=HealthResponse)
async def readiness_check():
    """Readiness check endpoint"""
    return HealthResponse(status="ready")

@router.get("/live", response_model=HealthResponse)  
async def liveness_check():
    """Liveness check endpoint"""
    return HealthResponse(status="alive") 