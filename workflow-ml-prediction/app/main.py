"""
workflow-ml-prediction: ML Prediction Microservice

This service handles machine learning model predictions for the workflow system.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Dict, Any, Optional
import asyncio
import uvloop

from .api.prediction import router as prediction_router
from .api.health import router as health_router
from .config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set event loop policy for better performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Create FastAPI application
app = FastAPI(
    title="Workflow ML Prediction Service",
    description="Machine learning model prediction microservice for hit song science",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, prefix="/health", tags=["health"])
app.include_router(prediction_router, prefix="/predict", tags=["prediction"])

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "workflow-ml-prediction",
        "version": "1.0.0",
        "status": "running",
        "description": "ML prediction microservice for hit song science"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    ) 