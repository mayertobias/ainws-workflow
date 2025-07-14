"""
Agentic AI API for Intelligence Service

This module provides API endpoints for the agentic AI system, integrating
multiple specialized agents for comprehensive analysis.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from ..agents.agent_orchestrator import AgentOrchestrator
from ..models.agentic_models import (
    AgenticAnalysisRequest, AgenticAnalysisResponse, 
    AgentRole, CoordinationStrategy
)
from ..models.workflow_integration import ComprehensiveAnalysisRequest

logger = logging.getLogger(__name__)

# Initialize router and orchestrator
router = APIRouter()
orchestrator = AgentOrchestrator()

@router.post("/analyze/agentic", response_model=AgenticAnalysisResponse)
async def perform_agentic_analysis(request: AgenticAnalysisRequest) -> AgenticAnalysisResponse:
    """
    Perform comprehensive analysis using multiple specialized AI agents.
    
    This endpoint coordinates multiple specialized agents to provide
    in-depth analysis with agent-specific insights and cross-validation.
    
    Args:
        request: Agentic analysis request with coordination strategy and requirements
        
    Returns:
        AgenticAnalysisResponse: Comprehensive multi-agent analysis results
    """
    try:
        logger.info(f"Starting agentic analysis: {request.request_id}")
        
        # Validate request
        if not request.required_agent_roles:
            raise HTTPException(
                status_code=400, 
                detail="At least one agent role must be specified"
            )
        
        # Check agent availability
        available_roles = list(orchestrator.agents.keys())
        unavailable_roles = [role for role in request.required_agent_roles if role not in available_roles]
        
        if unavailable_roles:
            logger.warning(f"Unavailable agent roles: {unavailable_roles}")
            # Continue with available agents
        
        # Perform agentic analysis
        response = await orchestrator.perform_agentic_analysis(request)
        
        logger.info(f"Completed agentic analysis: {request.request_id}")
        return response
        
    except Exception as e:
        logger.error(f"Agentic analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/comprehensive-agentic")
async def comprehensive_agentic_analysis(request: ComprehensiveAnalysisRequest) -> JSONResponse:
    """
    Perform comprehensive analysis using agentic AI system.
    
    This endpoint converts a standard comprehensive analysis request into
    an agentic analysis request and coordinates multiple agents.
    
    Args:
        request: Standard comprehensive analysis request
        
    Returns:
        JSONResponse: Enhanced analysis with multi-agent insights
    """
    try:
        logger.info("Converting comprehensive analysis to agentic analysis")
        
        # Convert to agentic request
        agentic_request = AgenticAnalysisRequest(
            coordination_strategy=CoordinationStrategy.COLLABORATIVE,
            required_agent_roles=[
                AgentRole.MUSIC_ANALYSIS,
                AgentRole.COMMERCIAL_ANALYSIS,
                AgentRole.QUALITY_ASSURANCE
            ],
            analysis_depth="comprehensive",
            focus_areas=["musical_analysis", "commercial_viability", "market_positioning"],
            enable_learning=True,
            enable_memory=True,
            max_execution_time=300,
            quality_threshold=0.7,
            input_data={
                "audio_analysis": request.audio_analysis.dict() if request.audio_analysis else {},
                "content_analysis": request.content_analysis.dict() if request.content_analysis else {},
                "hit_prediction": request.hit_prediction.dict() if request.hit_prediction else {},
                "song_metadata": request.song_metadata.dict() if request.song_metadata else {}
            }
        )
        
        # Perform agentic analysis
        agentic_response = await orchestrator.perform_agentic_analysis(agentic_request)
        
        # Convert back to enhanced comprehensive response
        enhanced_response = _convert_to_enhanced_response(agentic_response)
        
        return JSONResponse(content=enhanced_response)
        
    except Exception as e:
        logger.error(f"Comprehensive agentic analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/status")
async def get_agents_status() -> JSONResponse:
    """
    Get status of all agents in the system.
    
    Returns:
        JSONResponse: Status information for all agents
    """
    try:
        agent_status = {}
        
        for role, agent in orchestrator.agents.items():
            agent_status[role.value] = {
                "agent_id": agent.profile.agent_id,
                "name": agent.profile.name,
                "status": agent.profile.status.value,
                "experience_level": agent.profile.experience_level,
                "confidence_score": agent.profile.confidence_score,
                "last_active": agent.profile.last_active.isoformat(),
                "performance_metrics": agent.get_performance_metrics()
            }
        
        return JSONResponse(content={
            "agents": agent_status,
            "orchestrator_metrics": orchestrator.get_orchestrator_metrics(),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to get agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/{agent_role}/profile")
async def get_agent_profile(agent_role: str) -> JSONResponse:
    """
    Get detailed profile for a specific agent.
    
    Args:
        agent_role: Role of the agent to get profile for
        
    Returns:
        JSONResponse: Detailed agent profile
    """
    try:
        # Find agent by role
        role_enum = None
        for role in AgentRole:
            if role.value == agent_role:
                role_enum = role
                break
        
        if not role_enum or role_enum not in orchestrator.agents:
            raise HTTPException(status_code=404, detail=f"Agent with role {agent_role} not found")
        
        agent = orchestrator.agents[role_enum]
        
        profile_data = {
            "profile": {
                "agent_id": agent.profile.agent_id,
                "name": agent.profile.name,
                "role": agent.profile.role.value,
                "description": agent.profile.description,
                "expertise_areas": agent.get_expertise_areas(),
                "preferred_tools": [tool.value for tool in agent.get_preferred_tools()],
                "experience_level": agent.profile.experience_level,
                "confidence_score": agent.profile.confidence_score,
                "status": agent.profile.status.value,
                "created_at": agent.profile.created_at.isoformat(),
                "last_active": agent.profile.last_active.isoformat()
            },
            "performance_metrics": agent.get_performance_metrics(),
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "confidence_level": cap.confidence_level,
                    "complexity_level": cap.complexity_level,
                    "required_tools": [tool.value for tool in cap.required_tools]
                }
                for cap in agent.profile.capabilities
            ] if hasattr(agent.profile, 'capabilities') else []
        }
        
        return JSONResponse(content=profile_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/orchestrator/coordinate")
async def coordinate_agents(coordination_request: Dict[str, Any]) -> JSONResponse:
    """
    Manually coordinate agents for specific tasks.
    
    Args:
        coordination_request: Custom coordination request
        
    Returns:
        JSONResponse: Coordination results
    """
    try:
        strategy = coordination_request.get("strategy", "collaborative")
        agent_roles = coordination_request.get("agent_roles", [])
        input_data = coordination_request.get("input_data", {})
        
        # Convert strategy string to enum
        strategy_enum = CoordinationStrategy.COLLABORATIVE
        for strat in CoordinationStrategy:
            if strat.value == strategy:
                strategy_enum = strat
                break
        
        # Convert role strings to enums
        role_enums = []
        for role_str in agent_roles:
            for role in AgentRole:
                if role.value == role_str:
                    role_enums.append(role)
                    break
        
        # Create agentic request
        agentic_request = AgenticAnalysisRequest(
            coordination_strategy=strategy_enum,
            required_agent_roles=role_enums,
            analysis_depth="targeted",
            input_data=input_data,
            max_execution_time=180
        )
        
        # Perform coordination
        response = await orchestrator.perform_agentic_analysis(agentic_request)
        
        return JSONResponse(content={
            "coordination_results": {
                "strategy_used": response.coordination_strategy_used.value,
                "agents_involved": response.agents_involved,
                "coordination_quality": response.coordination_quality,
                "coordination_efficiency": response.coordination_efficiency,
                "total_processing_time_ms": response.total_processing_time_ms
            },
            "agent_contributions": len(response.agent_contributions),
            "overall_confidence": response.overall_confidence,
            "quality_score": response.quality_score
        })
        
    except Exception as e:
        logger.error(f"Agent coordination failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/orchestrator/metrics")
async def get_orchestrator_metrics() -> JSONResponse:
    """
    Get orchestrator performance metrics.
    
    Returns:
        JSONResponse: Orchestrator performance data
    """
    try:
        metrics = orchestrator.get_orchestrator_metrics()
        return JSONResponse(content=metrics)
        
    except Exception as e:
        logger.error(f"Failed to get orchestrator metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _convert_to_enhanced_response(agentic_response: AgenticAnalysisResponse) -> Dict[str, Any]:
    """
    Convert agentic response to enhanced comprehensive response format.
    
    Args:
        agentic_response: Agentic analysis response
        
    Returns:
        Enhanced response dictionary
    """
    # Extract insights by agent type
    music_insights = []
    commercial_insights = []
    technical_insights = []
    
    for contribution in agentic_response.agent_contributions:
        if contribution.agent_role == AgentRole.MUSIC_ANALYSIS:
            music_insights.extend(contribution.insights)
        elif contribution.agent_role == AgentRole.COMMERCIAL_ANALYSIS:
            commercial_insights.extend(contribution.insights)
        elif contribution.agent_role == AgentRole.TECHNICAL_ANALYSIS:
            technical_insights.extend(contribution.insights)
    
    # Build enhanced response
    enhanced_response = {
        "executive_summary": agentic_response.executive_summary,
        "overall_score": agentic_response.quality_score,
        "confidence_level": agentic_response.overall_confidence,
        
        # Traditional insights structure maintained for compatibility
        "insights": [
            {
                "category": "Multi-Agent Analysis",
                "title": "Comprehensive Assessment", 
                "description": agentic_response.executive_summary,
                "confidence": agentic_response.overall_confidence,
                "supporting_evidence": agentic_response.evidence_base[:3]
            }
        ],
        
        # Enhanced with agent-specific insights
        "agent_insights": {
            "musical_analysis": music_insights,
            "commercial_analysis": commercial_insights,
            "technical_analysis": technical_insights
        },
        
        # Agent collaboration metadata
        "collaboration_metadata": {
            "coordination_strategy": agentic_response.coordination_strategy_used.value,
            "agents_involved": len(agentic_response.agents_involved),
            "coordination_quality": agentic_response.coordination_quality,
            "coordination_efficiency": agentic_response.coordination_efficiency,
            "cross_validation_results": agentic_response.cross_validation_results
        },
        
        # Enhanced recommendations
        "strategic_recommendations": agentic_response.consensus_findings,
        "tactical_recommendations": [
            rec for contribution in agentic_response.agent_contributions 
            for rec in contribution.recommendations[:2]
        ],
        
        # Risk and opportunity analysis
        "risk_factors": agentic_response.conflicting_views,
        "opportunities": [
            insight for contribution in agentic_response.agent_contributions
            for insight in contribution.insights
            if "opportunity" in insight.lower() or "potential" in insight.lower()
        ][:3],
        
        # Technical metadata
        "analysis_metadata": {
            "processing_time_ms": agentic_response.total_processing_time_ms,
            "innovation_score": agentic_response.innovation_score,
            "tools_used": agentic_response.tools_usage_summary,
            "reasoning_transparency": agentic_response.reasoning_transparency,
            "analysis_timestamp": agentic_response.analysis_completed_at.isoformat()
        }
    }
    
    return enhanced_response
