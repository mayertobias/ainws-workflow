"""
Main FastAPI application for workflow-orchestrator service
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import structlog
import os

from .api import health, workflows, ab_testing
from .config.settings import settings
from .services.orchestrator import WorkflowOrchestrator

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Global orchestrator instance
orchestrator_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global orchestrator_instance
    
    # Startup
    logger.info(f"Starting {settings.SERVICE_NAME} on port {settings.PORT}")
    
    try:
        # Initialize orchestrator
        orchestrator_instance = WorkflowOrchestrator()
        await orchestrator_instance.start_workers(num_workers=5)
        
        # Store in app state for access in endpoints
        app.state.orchestrator = orchestrator_instance
        
        logger.info(f"{settings.SERVICE_NAME} started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start {settings.SERVICE_NAME}: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info(f"Shutting down {settings.SERVICE_NAME}")
        
        if orchestrator_instance:
            try:
                await orchestrator_instance.close()
                logger.info("Orchestrator closed successfully")
            except Exception as e:
                logger.error(f"Error closing orchestrator: {e}")

# Create FastAPI app
app = FastAPI(
    title="Workflow Orchestrator Service",
    description="Workflow orchestration and A/B testing service for HSS platform",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(workflows.router, prefix="/workflows", tags=["Workflows"])
app.include_router(ab_testing.router, prefix="/ab-testing", tags=["A/B Testing"])

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "workflow-orchestrator",
        "version": "1.0.0",
        "status": "running",
        "features": ["workflow_orchestration", "ab_testing", "statistical_analysis"]
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Dependency for getting orchestrator
async def get_orchestrator() -> WorkflowOrchestrator:
    """Get the orchestrator instance."""
    if hasattr(app.state, 'orchestrator'):
        return app.state.orchestrator
    else:
        raise HTTPException(
            status_code=503,
            detail="Orchestrator service not available"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG
    ) 