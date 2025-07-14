"""
FastAPI application for workflow-intelligence microservice

This service handles AI-powered insights generation with support for multiple LLM providers
and agentic AI multi-agent analysis.
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .config.settings import settings
from .api import intelligence, health, workflow_integration, agentic_ai
from .services.llm_providers import LLMProviderFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info(f"Starting {settings.SERVICE_NAME} on port {settings.PORT}")
    
    # Test LLM provider availability
    try:
        providers = LLMProviderFactory.list_providers()
        logger.info(f"Available LLM providers: {providers}")
        
        # Test auto-detection
        provider = LLMProviderFactory.auto_detect_provider()
        if provider:
            logger.info(f"Auto-detected LLM provider: {provider.__class__.__name__}")
        else:
            logger.warning("No LLM providers available - AI insights will be limited")
    except Exception as e:
        logger.warning(f"LLM provider initialization warning: {e}")
    
    # Initialize agentic AI system
    try:
        logger.info("Initializing agentic AI system with specialized agents")
        # The orchestrator is initialized in the agentic_ai module
        logger.info("Agentic AI system ready for multi-agent analysis")
    except Exception as e:
        logger.warning(f"Agentic AI initialization warning: {e}")
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {settings.SERVICE_NAME}")

# Create FastAPI application
app = FastAPI(
    title="Workflow Intelligence Service",
    description="AI-powered insights generation microservice with multi-provider LLM support and agentic AI multi-agent analysis",
    version="2.0.0",
    debug=settings.DEBUG,
    lifespan=lifespan
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
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(intelligence.router, prefix="/insights", tags=["Intelligence"])
app.include_router(workflow_integration.router, prefix="/workflow", tags=["Workflow Integration"])
app.include_router(agentic_ai.router, prefix="/agentic", tags=["Agentic AI"])

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": settings.SERVICE_NAME,
        "version": "2.0.0",
        "status": "healthy",
        "port": settings.PORT,
        "debug": settings.DEBUG,
        "features": [
            "LLM Provider Support",
            "Workflow Integration", 
            "Agentic AI Multi-Agent Analysis"
        ],
        "providers": LLMProviderFactory.list_providers(),
        "agentic_capabilities": [
            "Music Analysis Agent",
            "Commercial Analysis Agent",
            "Quality Assurance Agent",
            "Multi-Agent Coordination",
            "Cross-Validation"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG
    ) 