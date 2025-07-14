"""
Workflow ML Train Service

Clean, focused ML training pipeline service with:
- Dynamic strategy selection (no config file changes)
- Service discovery and feature agreement
- MLflow integration for experiment tracking
- Airflow integration for pipeline visualization
- Real-time progress monitoring

This service replaces the complex workflow-ml-training with a clean, 
pipeline-focused approach.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any
import uvicorn

from .api.pipeline import router as pipeline_router
from .api.monitoring import router as monitoring_router
from .api.features import router as features_router
from .pipeline.orchestrator import PipelineOrchestrator
from .monitoring.progress_tracker import ProgressTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
orchestrator = None
progress_tracker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global orchestrator, progress_tracker
    
    # Startup
    logger.info("🚀 Starting Workflow ML Train Service")
    
    try:
        # Initialize core components
        progress_tracker = ProgressTracker()
        orchestrator = PipelineOrchestrator()
        
        # Store in app state
        app.state.orchestrator = orchestrator
        app.state.progress_tracker = progress_tracker
        
        logger.info("✅ Workflow ML Train Service started successfully")
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start service: {e}")
        raise
    finally:
        # Shutdown
        logger.info("🛑 Shutting down Workflow ML Train Service")
        if orchestrator:
            await orchestrator.cleanup()

# Create FastAPI application
app = FastAPI(
    title="Workflow ML Train Service",
    description="""
    Clean ML training pipeline service with dynamic strategy selection.
    
    ## Key Features
    
    * **Dynamic Strategy Selection**: Change training strategies at runtime via API/CLI
    * **Service Discovery**: Auto-discover available features from audio/content services  
    * **Feature Agreement**: Interactive feature selection and validation
    * **MLflow Integration**: Comprehensive experiment tracking and model registry
    * **Airflow Integration**: Pipeline visualization and monitoring
    * **Real-time Progress**: Live pipeline status and progress tracking
    
    ## Available Strategies
    
    * **audio_only**: Train using only audio features
    * **multimodal**: Train using both audio and content features
    * **custom**: User-defined feature selection
    
    ## Quick Start
    
    ```bash
    # Start audio-only training
    curl -X POST "/pipeline/train" -d '{"strategy": "audio_only", "experiment_name": "test"}'
    
    # Start multimodal training
    curl -X POST "/pipeline/train" -d '{"strategy": "multimodal", "experiment_name": "test"}'
    ```
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(pipeline_router, prefix="/pipeline", tags=["Pipeline"])
app.include_router(monitoring_router, prefix="/monitoring", tags=["Monitoring"])
app.include_router(features_router, prefix="/features", tags=["Features"])

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "workflow-ml-train",
        "version": "1.0.0",
        "components": {
            "orchestrator": "healthy" if orchestrator else "unavailable",
            "progress_tracker": "healthy" if progress_tracker else "unavailable"
        }
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "workflow-ml-train",
        "version": "1.0.0",
        "description": "Clean ML training pipeline service",
        "status": "running",
        "features": [
            "dynamic_strategy_selection",
            "service_discovery", 
            "feature_agreement",
            "mlflow_integration",
            "airflow_integration",
            "real_time_monitoring"
        ],
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "pipeline": "/pipeline",
            "monitoring": "/monitoring", 
            "features": "/features"
        },
        "strategies": ["audio_only", "multimodal", "custom"],
        "quick_start": {
            "audio_only": "POST /pipeline/train {'strategy': 'audio_only'}",
            "multimodal": "POST /pipeline/train {'strategy': 'multimodal'}",
            "monitor": "GET /monitoring/status/{pipeline_id}"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8005,
        reload=True,
        log_level="info"
    ) 