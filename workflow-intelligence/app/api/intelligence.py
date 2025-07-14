"""
Intelligence API endpoints for workflow-intelligence service
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse

from ..models.intelligence import (
    InsightGenerationRequest, InsightGenerationResponse, BatchInsightRequest, BatchInsightResponse,
    ProviderListResponse, ProviderInfo, InsightMetrics, CacheStats,
    PromptTemplateRequest, PromptTemplate, AnalysisType, AgentType, AIProvider
)
from ..models.responses import SuccessResponse, ErrorResponse
from ..services.intelligence_service import IntelligenceService
from ..services.llm_providers import LLMProviderFactory
from ..services.agents import AgentManager
from ..services.prompt_manager import PromptManager
from ..config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Service instances
intelligence_service = IntelligenceService()
agent_manager = AgentManager()
prompt_manager = PromptManager()

async def get_intelligence_service() -> IntelligenceService:
    """Dependency to get intelligence service instance."""
    return intelligence_service

@router.post("/generate", response_model=InsightGenerationResponse)
async def generate_insights(
    request: InsightGenerationRequest,
    background_tasks: BackgroundTasks,
    service: IntelligenceService = Depends(get_intelligence_service)
):
    """
    Generate AI insights for a song.
    
    This endpoint analyzes provided song data and generates insights using AI.
    Supports multiple analysis types, agent types, and AI providers.
    """
    try:
        logger.info(f"Generating insights with {len(request.analysis_types)} analysis types")
        
        # Validate request
        if not request.analysis_types:
            raise HTTPException(
                status_code=400, 
                detail="At least one analysis type must be specified"
            )
        
        # Generate insights
        response = await service.generate_insights(request)
        
        # Log metrics in background
        background_tasks.add_task(_log_insight_generation, request, response)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate insights: {str(e)}"
        )

@router.post("/batch", response_model=BatchInsightResponse)
async def generate_batch_insights(
    request: BatchInsightRequest,
    background_tasks: BackgroundTasks,
    service: IntelligenceService = Depends(get_intelligence_service)
):
    """
    Generate insights for multiple songs in batch.
    
    Processes up to 50 insight requests concurrently for improved efficiency.
    """
    try:
        if len(request.requests) > 50:
            raise HTTPException(
                status_code=400,
                detail="Batch size cannot exceed 50 requests"
            )
        
        logger.info(f"Processing batch of {len(request.requests)} insight requests")
        
        import asyncio
        import uuid
        
        batch_id = request.batch_id or str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # Process requests concurrently
        tasks = [
            service.generate_insights(req) for req in request.requests
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        failed_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error response
                error_response = InsightGenerationResponse(
                    status="failed",
                    insight_id=f"batch_{batch_id}_{i}",
                    error_message=str(result),
                    cached=False
                )
                successful_results.append(error_response)
                failed_count += 1
            else:
                successful_results.append(result)
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        batch_response = BatchInsightResponse(
            batch_id=batch_id,
            total_requests=len(request.requests),
            completed_requests=len(successful_results) - failed_count,
            failed_requests=failed_count,
            results=successful_results,
            processing_time_ms=processing_time
        )
        
        # Log batch metrics
        background_tasks.add_task(_log_batch_processing, request, batch_response)
        
        return batch_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)}"
        )

@router.get("/providers", response_model=ProviderListResponse)
async def list_providers():
    """List all available AI providers and their status."""
    try:
        provider_info = LLMProviderFactory.get_provider_info()
        
        providers = []
        for provider_name, info in provider_info.items():
            # Convert config_keys list to a configuration dictionary
            config_keys = info.get("config_keys", [])
            configuration = {key: "configured" if info.get("available", False) else "not_configured" 
                           for key in config_keys}
            
            provider = ProviderInfo(
                provider_name=provider_name,
                available=info.get("available", False),
                configuration=configuration,
                capabilities=["text_generation", "analysis"],
                cost_per_token=_get_estimated_cost(provider_name)
            )
            providers.append(provider)
        
        # Determine default and auto-detected provider
        default_provider = settings.AI_PROVIDER if settings.AI_PROVIDER != "auto" else None
        
        auto_detected = None
        try:
            auto_provider = LLMProviderFactory.auto_detect_provider()
            auto_detected = auto_provider.__class__.__name__ if auto_provider else None
        except Exception:
            pass
        
        return ProviderListResponse(
            providers=providers,
            default_provider=default_provider,
            auto_detected=auto_detected
        )
        
    except Exception as e:
        logger.error(f"Error listing providers: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list providers: {str(e)}"
        )

@router.get("/templates")
async def list_templates():
    """List all available prompt templates."""
    try:
        templates = prompt_manager.list_templates()
        return {
            "templates": templates,
            "total_templates": len(templates),
            "analysis_types": [at.value for at in AnalysisType]
        }
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list templates: {str(e)}"
        )

@router.post("/templates")
async def create_template(template_request: PromptTemplateRequest):
    """Create or update a prompt template."""
    try:
        import uuid
        template = PromptTemplate(
            template_id=str(uuid.uuid4()),
            name=template_request.name,
            description=template_request.description,
            template_text=template_request.template_text,
            variables=template_request.variables,
            analysis_type=template_request.analysis_type
        )
        
        success = prompt_manager.add_custom_template(template)
        
        if success:
            return SuccessResponse(
                message="Template created successfully",
                data={"template_id": template.template_id}
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Failed to create template"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating template: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create template: {str(e)}"
        )

@router.get("/templates/{template_id}")
async def get_template(template_id: str):
    """Get a specific prompt template by ID."""
    try:
        template = prompt_manager.get_template(template_id)
        if not template:
            raise HTTPException(
                status_code=404,
                detail="Template not found"
            )
        return template
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting template: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get template: {str(e)}"
        )

@router.get("/metrics", response_model=InsightMetrics)
async def get_metrics(
    service: IntelligenceService = Depends(get_intelligence_service)
):
    """Get service metrics and usage statistics."""
    try:
        metrics = await service.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics: {str(e)}"
        )

@router.get("/cache/stats")
async def get_cache_stats(
    service: IntelligenceService = Depends(get_intelligence_service)
):
    """Get cache statistics and information."""
    try:
        stats = await service.get_cache_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache stats: {str(e)}"
        )

@router.delete("/cache/clear")
async def clear_cache(
    service: IntelligenceService = Depends(get_intelligence_service)
):
    """Clear all cached insights."""
    try:
        result = await service.clear_cache()
        return SuccessResponse(
            message="Cache cleared successfully",
            data=result
        )
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )

@router.get("/agents")
async def list_agents():
    """List available agent types and their information."""
    try:
        from ..services.agents import IntelligenceAgentFactory
        
        factory = IntelligenceAgentFactory()
        agent_types = factory.list_agent_types()
        
        agents_info = []
        for agent_type in agent_types:
            info = factory.get_agent_info(agent_type)
            agents_info.append(info)
        
        return {
            "agents": agents_info,
            "default_agent": settings.DEFAULT_AGENT_TYPE,
            "total_agents": len(agents_info)
        }
        
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list agents: {str(e)}"
        )

@router.post("/test/{agent_type}")
async def test_agent(
    agent_type: str,
    provider: Optional[str] = Query(default="auto", description="AI provider to test")
):
    """Test a specific agent type with a provider."""
    try:
        if agent_type not in [at.value for at in AgentType]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent type. Available: {[at.value for at in AgentType]}"
            )
        
        result = await agent_manager.test_agent_generation(
            agent_type=agent_type,
            provider_preference=provider
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing agent: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent test failed: {str(e)}"
        )

@router.get("/analysis-types")
async def list_analysis_types():
    """List all available analysis types."""
    try:
        analysis_types = []
        for analysis_type in AnalysisType:
            analysis_types.append({
                "type": analysis_type.value,
                "name": analysis_type.value.replace("_", " ").title(),
                "description": _get_analysis_description(analysis_type)
            })
        
        return {
            "analysis_types": analysis_types,
            "total_types": len(analysis_types),
            "supported_combinations": [
                ["musical_meaning", "hit_comparison"],
                ["novelty_assessment", "strategic_insights"],
                ["production_feedback", "strategic_insights"]
            ]
        }
        
    except Exception as e:
        logger.error(f"Error listing analysis types: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list analysis types: {str(e)}"
        )

@router.get("/config")
async def get_service_config():
    """Get current service configuration (non-sensitive)."""
    try:
        config = {
            "service_name": settings.SERVICE_NAME,
            "version": "1.0.0",
            "ai_provider": settings.AI_PROVIDER,
            "default_agent": settings.DEFAULT_AGENT_TYPE,
            "supported_analysis_types": settings.SUPPORTED_ANALYSIS_TYPES,
            "cache_ttl_seconds": settings.CACHE_TTL_SECONDS,
            "max_tokens_per_request": settings.MAX_TOKENS_PER_REQUEST,
            "rate_limiting": {
                "max_requests_per_minute": settings.MAX_REQUESTS_PER_MINUTE,
                "max_requests_per_hour": settings.MAX_REQUESTS_PER_HOUR
            },
            "features": {
                "caching_enabled": bool(settings.REDIS_URL),
                "cost_tracking": settings.ENABLE_COST_TRACKING,
                "prompt_caching": settings.ENABLE_PROMPT_CACHING,
                "agent_selection": settings.ENABLE_AGENT_SELECTION
            }
        }
        
        return config
        
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get config: {str(e)}"
        )

# Helper functions

async def _log_insight_generation(request: InsightGenerationRequest, response: InsightGenerationResponse):
    """Log insight generation for analytics."""
    try:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "analysis_types": [at.value for at in request.analysis_types],
            "agent_type": request.agent_type.value,
            "ai_provider": request.ai_provider.value,
            "status": response.status.value,
            "cached": response.cached,
            "processing_time_ms": response.insights.processing_time_ms if response.insights else None,
            "insights_generated": len(response.insights.analysis_types_completed) if response.insights else 0
        }
        
        logger.info(f"Insight generation completed: {log_data}")
        
    except Exception as e:
        logger.error(f"Error logging insight generation: {e}")

async def _log_batch_processing(request: BatchInsightRequest, response: BatchInsightResponse):
    """Log batch processing for analytics."""
    try:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "batch_id": response.batch_id,
            "total_requests": response.total_requests,
            "completed_requests": response.completed_requests,
            "failed_requests": response.failed_requests,
            "processing_time_ms": response.processing_time_ms,
            "success_rate": response.completed_requests / response.total_requests if response.total_requests > 0 else 0
        }
        
        logger.info(f"Batch processing completed: {log_data}")
        
    except Exception as e:
        logger.error(f"Error logging batch processing: {e}")

def _get_estimated_cost(provider_name: str) -> Optional[float]:
    """Get estimated cost per token for a provider."""
    # Rough estimates (per 1K tokens)
    cost_estimates = {
        "OpenAIProvider": 0.03,  # GPT-4 rough estimate
        "GeminiProvider": 0.001,  # Gemini Flash is very cheap
        "AnthropicProvider": 0.015,  # Claude pricing
        "OllamaProvider": 0.0,  # Local, free
        "HuggingFaceProvider": 0.0  # Local, free
    }
    
    return cost_estimates.get(provider_name)

def _get_analysis_description(analysis_type: AnalysisType) -> str:
    """Get description for an analysis type."""
    descriptions = {
        AnalysisType.MUSICAL_MEANING: "Analyzes the emotional core and artistic meaning of the song",
        AnalysisType.HIT_COMPARISON: "Compares against successful hits to assess commercial potential",
        AnalysisType.NOVELTY_ASSESSMENT: "Evaluates innovation and uniqueness compared to current trends",
        AnalysisType.GENRE_COMPARISON: "Compares against genre conventions and benchmarks",
        AnalysisType.PRODUCTION_FEEDBACK: "Provides technical production quality assessment",
        AnalysisType.STRATEGIC_INSIGHTS: "Offers strategic business insights for release and promotion",
        AnalysisType.COMPREHENSIVE_ANALYSIS: "Combines all analysis types for complete assessment"
    }
    
    return descriptions.get(analysis_type, "Analysis description not available") 