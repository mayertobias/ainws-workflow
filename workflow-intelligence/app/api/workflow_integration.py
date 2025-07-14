"""
Workflow Integration API for Intelligence Service

This module provides the critical endpoint that integrates with the rest
of the workflow microservices to receive comprehensive analysis data and generate
AI insights.
"""

import logging
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse

from ..models.workflow_integration import (
    ComprehensiveAnalysisRequest, ComprehensiveAnalysisResponse,
    AudioAnalysisResult, ContentAnalysisResult, HitPredictionResult, SongMetadata,
    AnalysisInsight
)
from ..models.responses import SuccessResponse, ErrorResponse
from ..services.intelligence_service import IntelligenceService
from ..services.enhanced_analysis_service import EnhancedAnalysisService
from ..services.professional_report_generator import ProfessionalReportGenerator
from ..services.data_transformation_service import DataTransformationService
from ..config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/debug/llm-status")
async def debug_llm_status():
    """Debug endpoint to check LLM service status."""
    from ..services.agent_llm_service import AgentLLMService
    
    # Create a test agent LLM service
    agent_llm = AgentLLMService()
    
    return {
        "llm_service_stats": agent_llm.get_service_stats(),
        "provider_available": agent_llm.llm_provider is not None,
        "provider_type": agent_llm.llm_provider.__class__.__name__ if agent_llm.llm_provider else None
    }

# ENHANCED: Use AgentOrchestrator with preserved superior insights
from ..agents.agent_orchestrator import AgentOrchestrator
from ..services.professional_report_generator import ProfessionalReportGenerator

# Service instances - AgentOrchestrator now contains all superior insights
agent_orchestrator = AgentOrchestrator()
report_generator = ProfessionalReportGenerator()
intelligence_service = IntelligenceService()
enhanced_analysis_service = EnhancedAnalysisService()
data_transformer = DataTransformationService()

async def get_agent_orchestrator() -> AgentOrchestrator:
    """Dependency to get enhanced agent orchestrator instance."""
    return agent_orchestrator

async def get_intelligence_service() -> IntelligenceService:
    """Dependency to get intelligence service instance."""
    return intelligence_service

@router.post("/analyze/comprehensive", response_model=ComprehensiveAnalysisResponse)
async def analyze_song_comprehensive(
    request: ComprehensiveAnalysisRequest,
    background_tasks: BackgroundTasks,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
):
    """
    **CRITICAL WORKFLOW INTEGRATION ENDPOINT**
    
    This is the main endpoint that receives complete analysis from:
    - Audio Analysis Service (musical features)
    - Content Analysis Service (lyrics insights)
    - ML Prediction Service (hit potential)
    
    And generates comprehensive AI insights.
    
    This endpoint will be enhanced with agentic AI in Phase 2.
    """
    try:
        start_time = datetime.utcnow()
        logger.info(f"Starting comprehensive analysis for: {request.song_metadata.title} by {request.song_metadata.artist}")
        
        # Validate input data completeness
        validation_result = await _validate_analysis_data(request)
        if not validation_result["is_valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid analysis data: {validation_result['errors']}"
            )
        
        # Log warnings if any
        if validation_result["warnings"]:
            for warning in validation_result["warnings"]:
                logger.warning(warning)
        
        # ENHANCED: Generate comprehensive analysis using AgentOrchestrator with preserved superior insights
        # Build request data with proper fallbacks for missing content
        content_analysis = request.content_analysis.dict() if request.content_analysis else {}
        
        # If no lyrics provided, create instrumental analysis context
        if not content_analysis.get("raw_lyrics"):
            content_analysis.update({
                "analysis_type": "instrumental",
                "lyrical_content": "none",
                "content_focus": "purely instrumental track",
                "instrumental_score": 1.0,
                "narrative_structure": "none",
                "message_clarity": "instrumental expression"
            })
            
        request_data = {
            "song_metadata": request.song_metadata.dict() if request.song_metadata else {},
            "audio_analysis": request.audio_analysis.dict() if request.audio_analysis else {},
            "content_analysis": content_analysis,
            "hit_prediction": request.hit_prediction.dict() if request.hit_prediction else {},
            "request_id": str(uuid.uuid4())
        }
        
        analysis_result = await orchestrator.analyze_comprehensive_with_insights(request_data)
        
        # DETAILED LOGGING: Show exactly what AgentOrchestrator returns
        logger.info("="*80)
        logger.info(f"🤖 RAW AGENT ORCHESTRATOR RESPONSE for: {request.song_metadata.title}")
        logger.info("="*80)
        logger.info(f"🔍 Response Type: {type(analysis_result)}")
        logger.info(f"🔍 Response Keys: {list(analysis_result.keys()) if isinstance(analysis_result, dict) else 'Not a dict'}")
        
        if isinstance(analysis_result, dict):
            for key, value in analysis_result.items():
                logger.info(f"🔍 [{key}]: {type(value)} - {len(value) if isinstance(value, (list, dict, str)) else value}")
                
                # Log detailed content for key fields
                if key == "executive_summary" and isinstance(value, str):
                    logger.info(f"    📋 Executive Summary: {value[:300]}{'...' if len(value) > 300 else ''}")
                elif key == "findings" and isinstance(value, list):
                    logger.info(f"    💡 Findings ({len(value)}):")
                    for i, finding in enumerate(value[:5], 1):  # Show first 5 findings
                        logger.info(f"      {i}. {str(finding)[:200]}{'...' if len(str(finding)) > 200 else ''}")
                elif key == "insights" and isinstance(value, list):
                    logger.info(f"    🧠 Insights ({len(value)}):")
                    for i, insight in enumerate(value[:5], 1):  # Show first 5 insights
                        logger.info(f"      {i}. {str(insight)[:200]}{'...' if len(str(insight)) > 200 else ''}")
                elif key == "recommendations" and isinstance(value, list):
                    logger.info(f"    🚀 Recommendations ({len(value)}):")
                    for i, rec in enumerate(value[:5], 1):  # Show first 5 recommendations
                        logger.info(f"      {i}. {str(rec)[:200]}{'...' if len(str(rec)) > 200 else ''}")
                elif key == "evidence" and isinstance(value, list):
                    logger.info(f"    📊 Evidence ({len(value)}):")
                    for i, evidence in enumerate(value[:3], 1):  # Show first 3 evidence items
                        logger.info(f"      {i}. {str(evidence)[:150]}{'...' if len(str(evidence)) > 150 else ''}")
                elif key in ["methodology", "tools_used", "enhanced_features", "confidence_level", "overall_score"]:
                    logger.info(f"    ⚙️ {key}: {value}")
        
        logger.info("="*80)
        logger.info("🎉 END OF RAW AGENT ORCHESTRATOR RESPONSE")
        logger.info("="*80)
        
        # Convert to response model with data transformation
        response = _convert_to_response_model(analysis_result, request)
        
        # LAYER 2: Skip data transformation for ComprehensiveAnalysisResponse
        # The AgentOrchestrator already provides rich, structured data that's better
        # than any transformation. The frontend's defensive programming handles it.
        logger.info("Skipping data transformation - AgentOrchestrator provides rich structured data")
        
        # Calculate processing time
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds() * 1000
        response.processing_time_ms = processing_time
        
        # DETAILED LOGGING: Show exactly what insights are being generated
        logger.info("="*80)
        logger.info(f"🎵 COMPREHENSIVE RESPONSE ANALYSIS for: {request.song_metadata.title} by {request.song_metadata.artist}")
        logger.info("="*80)
        
        # Log executive summary
        logger.info(f"📋 Executive Summary: {response.executive_summary}")
        logger.info(f"⭐ Overall Score: {response.overall_score:.3f}")
        logger.info(f"🎯 Confidence Level: {response.confidence_level:.3f}")
        
        # Log detailed insights
        logger.info(f"💡 Generated {len(response.insights)} Total Insights:")
        for i, insight in enumerate(response.insights, 1):
            logger.info(f"  {i}. [{insight.category.upper()}] {insight.title}")
            logger.info(f"     Description: {insight.description[:200]}{'...' if len(insight.description) > 200 else ''}")
            logger.info(f"     Confidence: {insight.confidence:.3f}")
            logger.info(f"     Evidence: {len(insight.evidence)} items")
            logger.info(f"     Recommendations: {len(insight.recommendations)} items")
        
        # Log strategic recommendations
        logger.info(f"🚀 Strategic Recommendations ({len(response.strategic_recommendations)}):")
        for i, rec in enumerate(response.strategic_recommendations, 1):
            logger.info(f"  {i}. {rec[:150]}{'...' if len(rec) > 150 else ''}")
        
        # Log tactical recommendations
        logger.info(f"⚡ Tactical Recommendations ({len(response.tactical_recommendations)}):")
        for i, rec in enumerate(response.tactical_recommendations, 1):
            logger.info(f"  {i}. {rec[:150]}{'...' if len(rec) > 150 else ''}")
        
        # Log risk factors
        if response.risk_factors:
            logger.info(f"⚠️ Risk Factors ({len(response.risk_factors)}):")
            for i, risk in enumerate(response.risk_factors, 1):
                logger.info(f"  {i}. {risk}")
        
        # Log opportunities
        if response.opportunities:
            logger.info(f"🎯 Opportunities ({len(response.opportunities)}):")
            for i, opp in enumerate(response.opportunities, 1):
                logger.info(f"  {i}. {opp}")
        
        # Log market insights
        if response.market_positioning:
            logger.info(f"📊 Market Positioning: {response.market_positioning}")
        
        if response.target_demographics:
            logger.info(f"👥 Target Demographics: {', '.join(response.target_demographics)}")
        
        if response.competitive_analysis:
            logger.info(f"🏆 Competitive Analysis: {response.competitive_analysis[:200]}{'...' if len(response.competitive_analysis) > 200 else ''}")
        
        # Log technical insights
        if response.production_feedback:
            logger.info(f"🎛️ Production Feedback ({len(response.production_feedback)}):")
            for i, feedback in enumerate(response.production_feedback, 1):
                logger.info(f"  {i}. {feedback}")
        
        if response.technical_strengths:
            logger.info(f"💪 Technical Strengths ({len(response.technical_strengths)}):")
            for i, strength in enumerate(response.technical_strengths, 1):
                logger.info(f"  {i}. {strength}")
        
        if response.technical_improvements:
            logger.info(f"🔧 Technical Improvements ({len(response.technical_improvements)}):")
            for i, improvement in enumerate(response.technical_improvements, 1):
                logger.info(f"  {i}. {improvement}")
        
        logger.info(f"⏱️ Processing Time: {processing_time:.0f}ms")
        logger.info("="*80)
        logger.info("🎉 END OF COMPREHENSIVE RESPONSE ANALYSIS")
        logger.info("="*80)
        
        # Log analysis completion in background
        background_tasks.add_task(_log_comprehensive_analysis, request, response)
        
        logger.info(f"Comprehensive analysis completed for: {request.song_metadata.title} in {processing_time:.0f}ms")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Comprehensive analysis failed: {str(e)}"
        )

def _extract_opportunities_from_findings(findings: List[str]) -> List[str]:
    """Extract opportunities from LLM findings."""
    opportunities = []
    for finding in findings:
        if "opportunit" in finding.lower():
            opportunities.append(finding)  # No truncation - return full content
    return opportunities if opportunities else ["Review detailed findings for opportunities"]

def _extract_market_positioning_from_findings(findings: List[str]) -> str:
    """Extract market positioning from LLM findings."""
    for finding in findings:
        if "market" in finding.lower() or "positioning" in finding.lower():
            return finding  # No truncation - return full content
    return "Review detailed findings for market positioning insights"

def _extract_target_demographics_from_findings(findings: List[str]) -> List[str]:
    """Extract target demographics from LLM findings."""
    demographics = []
    for finding in findings:
        if "demographic" in finding.lower() or "audience" in finding.lower():
            demographics.append(finding)  # No truncation - return full content
    return demographics if demographics else ["Review detailed findings for demographic insights"]

def _extract_competitive_analysis_from_findings(findings: List[str]) -> str:
    """Extract competitive analysis from LLM findings."""
    for finding in findings:
        if "competit" in finding.lower() or "landscape" in finding.lower() or "market" in finding.lower():
            return finding  # No truncation - return full content
    return "Market analysis shows competitive dynamics within the genre"

def _extract_production_feedback_from_findings(findings: List[str]) -> List[str]:
    """Extract production feedback from LLM findings."""
    feedback = []
    for finding in findings:
        if "production" in finding.lower() or "sound" in finding.lower() or "acousticness" in finding.lower() or "spectral" in finding.lower():
            feedback.append(finding)  # No truncation - return full content
    return feedback if feedback else ["Review detailed findings for production insights"]

def _extract_technical_strengths_from_findings(findings: List[str]) -> List[str]:
    """Extract technical strengths from LLM findings."""
    strengths = []
    for finding in findings:
        if "strength" in finding.lower() or "strong" in finding.lower():
            strengths.append(finding)  # No truncation - return full content
    return strengths if strengths else ["Review detailed findings for technical strengths"]

def _extract_technical_improvements_from_findings(findings: List[str]) -> List[str]:
    """Extract technical improvements from LLM findings."""
    improvements = []
    for finding in findings:
        if "improve" in finding.lower() or "enhance" in finding.lower() or "consider" in finding.lower() or "explore" in finding.lower():
            improvements.append(finding)  # No truncation - return full content
    return improvements if improvements else ["Consider optimizing based on detailed analysis findings"]

def _convert_to_response_model(analysis_result: Dict[str, Any], original_request: ComprehensiveAnalysisRequest) -> ComprehensiveAnalysisResponse:
    """Convert enhanced analysis result to response model format."""
    try:
        # DEBUG: Log the structure we're working with
        logger.info(f"🔍 DEBUG: Converting analysis_result with keys: {list(analysis_result.keys()) if analysis_result else 'None'}")
        
        # FIXED: Extract insights from the actual AgentOrchestrator response structure
        insights = []
        
        # Extract findings as detailed insights
        findings = analysis_result.get("findings", [])
        for i, finding in enumerate(findings):
            insights.append(AnalysisInsight(
                category="detailed_analysis",
                title=f"Finding {i+1}",
                description=finding if isinstance(finding, str) else str(finding),
                confidence=analysis_result.get("confidence_level", 0.8),
                evidence=[finding] if isinstance(finding, str) else [str(finding)],
                recommendations=[]
            ))
        
        # Extract insights from the AgentOrchestrator response
        ai_insights = analysis_result.get("insights", [])
        for i, insight in enumerate(ai_insights):
            insights.append(AnalysisInsight(
                category="ai_insights",
                title=f"AI Insight {i+1}",
                description=insight if isinstance(insight, str) else str(insight),
                confidence=analysis_result.get("confidence_level", 0.8),
                evidence=[insight] if isinstance(insight, str) else [str(insight)],
                recommendations=[]
            ))
        
        # Extract recommendations as actionable insights
        recommendations = analysis_result.get("recommendations", [])
        for i, recommendation in enumerate(recommendations):
            insights.append(AnalysisInsight(
                category="recommendations",
                title=f"Recommendation {i+1}",
                description=recommendation if isinstance(recommendation, str) else str(recommendation),
                confidence=analysis_result.get("confidence_level", 0.8),
                evidence=[],
                recommendations=[recommendation] if isinstance(recommendation, str) else [str(recommendation)]
            ))
        
        # Extract evidence as supporting insights
        evidence = analysis_result.get("evidence", [])
        if evidence:
            insights.append(AnalysisInsight(
                category="evidence",
                title="Supporting Evidence",
                description=f"Analysis supported by {len(evidence)} evidence points",
                confidence=analysis_result.get("confidence_level", 0.8),
                evidence=evidence if isinstance(evidence, list) else [str(evidence)],
                recommendations=[]
            ))
        
        # Add methodology insight
        methodology = analysis_result.get("methodology", "")
        if methodology:
            insights.append(AnalysisInsight(
                category="methodology",
                title="Analysis Methodology",
                description=methodology if isinstance(methodology, str) else str(methodology),
                confidence=analysis_result.get("confidence_level", 0.8),
                evidence=[],
                recommendations=[]
            ))
        
        # Add tools used insight
        tools_used = analysis_result.get("tools_used", [])
        if tools_used:
            insights.append(AnalysisInsight(
                category="tools_analysis",
                title="Analysis Tools & Techniques",
                description=f"Analysis employed {len(tools_used)} specialized tools and techniques",
                confidence=analysis_result.get("confidence_level", 0.8),
                evidence=tools_used if isinstance(tools_used, list) else [str(tools_used)],
                recommendations=[]
            ))
        
        # Add enhanced features insight
        enhanced_features = analysis_result.get("enhanced_features", [])
        if enhanced_features:
            insights.append(AnalysisInsight(
                category="enhanced_features",
                title="Enhanced Analysis Features",
                description=f"Analysis utilized {len(enhanced_features)} enhanced features for superior insights",
                confidence=analysis_result.get("confidence_level", 0.8),
                evidence=enhanced_features if isinstance(enhanced_features, list) else [str(enhanced_features)],
                recommendations=[]
            ))
        
        # Hit prediction insights
        hit_probability = original_request.hit_prediction.hit_probability if original_request.hit_prediction else 0.0
        if hit_probability > 0:
            insights.append(AnalysisInsight(
                category="hit_prediction",
                title="Hit Potential Assessment",
                description=f"AI prediction indicates {hit_probability*100:.1f}% hit potential based on comprehensive feature analysis",
                confidence=original_request.hit_prediction.confidence_score if original_request.hit_prediction else 0.8,
                evidence=[f"Prediction score: {hit_probability:.3f}", "Based on multimodal ML analysis"],
                recommendations=["Focus on identified strengths", "Address potential weaknesses"]
            ))
        
        # Ensure we always have at least one insight
        if not insights:
            insights.append(AnalysisInsight(
                category="general_analysis",
                title="General Analysis",
                description="Basic song analysis completed using enhanced AI workflow",
                confidence=0.7,
                evidence=["Song metadata processed", "Basic features extracted"],
                recommendations=["Consider additional feature analysis", "Explore genre-specific optimization"]
            ))
        
        # DEBUG: Log insights before creating response
        logger.info(f"🔍 DEBUG: Created {len(insights)} insights for response")
        
        # Create response with safe defaults
        request_id = original_request.request_id if hasattr(original_request, 'request_id') else str(uuid.uuid4())
        executive_summary = analysis_result.get("executive_summary", "Comprehensive analysis completed using enhanced parallel agentic workflow.")
        
        # Extract individual scores from AgentOrchestrator overall_assessment
        overall_assessment = analysis_result.get("overall_assessment", {})
        overall_score = overall_assessment.get("overall_potential", 0.7)
        confidence_level = analysis_result.get("confidence_level", 0.8)
        
        # Extract strategic and tactical recommendations
        strategic_recommendations = []
        tactical_recommendations = []
        
        # Split recommendations into strategic and tactical
        all_recommendations = analysis_result.get("recommendations", [])
        for i, rec in enumerate(all_recommendations):
            if i % 2 == 0:
                strategic_recommendations.append(rec if isinstance(rec, str) else str(rec))
            else:
                tactical_recommendations.append(rec if isinstance(rec, str) else str(rec))
        
        # Add default recommendations if empty
        if not strategic_recommendations:
            strategic_recommendations = ["Optimize musical elements for target audience"]
        if not tactical_recommendations:
            tactical_recommendations = ["Focus on identified high-impact features"]
        
        # DEBUG: Log what we're about to use
        logger.info(f"🔍 DEBUG: Creating response with request_id={request_id}, insights_count={len(insights)}, confidence_level={confidence_level}")
        
        response = ComprehensiveAnalysisResponse(
            request_id=request_id,
            insights=insights,
            executive_summary=executive_summary,
            overall_score=overall_score,
            confidence_level=confidence_level,
            overall_assessment=overall_assessment,  # Add individual scores from AgentOrchestrator
            processing_time_ms=analysis_result.get("processing_time_ms", 0.0),
            strategic_recommendations=strategic_recommendations,
            tactical_recommendations=tactical_recommendations,
            risk_factors=analysis_result.get("risk_factors", []),
            opportunities=analysis_result.get("opportunities", _extract_opportunities_from_findings(analysis_result.get("findings", []))),
            market_positioning=analysis_result.get("market_positioning", _extract_market_positioning_from_findings(analysis_result.get("findings", []))),
            target_demographics=analysis_result.get("target_demographics", _extract_target_demographics_from_findings(analysis_result.get("findings", []))),
            competitive_analysis=analysis_result.get("competitive_analysis", _extract_competitive_analysis_from_findings(analysis_result.get("findings", []))),
            production_feedback=analysis_result.get("production_feedback", _extract_production_feedback_from_findings(analysis_result.get("findings", []))),
            technical_strengths=analysis_result.get("technical_strengths", _extract_technical_strengths_from_findings(analysis_result.get("findings", []))),
            technical_improvements=analysis_result.get("technical_improvements", _extract_technical_improvements_from_findings(analysis_result.get("findings", [])))
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error converting analysis result to response model: {e}")
        # Return minimal response
        return ComprehensiveAnalysisResponse(
            analysis_id=str(uuid.uuid4()),
            request_id=str(uuid.uuid4()),
            song_metadata=original_request.song_metadata,
            insights=[],
            executive_summary=f"Analysis conversion failed: {str(e)}",
            overall_score=0.0,
            confidence_score=0.0,
            processing_time_ms=0.0,
            timestamp=datetime.utcnow(),
            enhanced_features={"error": str(e)}
        )

@router.post("/analyze/from-orchestrator", response_model=ComprehensiveAnalysisResponse)
async def analyze_from_orchestrator(
    audio_analysis: AudioAnalysisResult,
    content_analysis: ContentAnalysisResult,
    hit_prediction: HitPredictionResult,
    song_metadata: SongMetadata,
    analysis_config: Optional[Dict[str, Any]] = None,
    service: IntelligenceService = Depends(get_intelligence_service)
):
    """
    **ORCHESTRATOR INTEGRATION ENDPOINT**
    
    This endpoint is called by the workflow-orchestrator service after it has
    collected all the analysis results from the different microservices.
    
    This is the primary integration point for the complete workflow.
    """
    try:
        # Construct comprehensive analysis request
        request = ComprehensiveAnalysisRequest(
            audio_analysis=audio_analysis,
            content_analysis=content_analysis,
            hit_prediction=hit_prediction,
            song_metadata=song_metadata,
            analysis_depth=analysis_config.get("depth", "comprehensive") if analysis_config else "comprehensive",
            focus_areas=analysis_config.get("focus_areas", []) if analysis_config else [],
            target_audience=analysis_config.get("target_audience") if analysis_config else None,
            business_context=analysis_config.get("business_context") if analysis_config else None,
            request_id=str(uuid.uuid4())
        )
        
        logger.info(f"Orchestrator triggered analysis for: {song_metadata.title}")
        
        # Route to comprehensive analysis
        return await analyze_song_comprehensive(request, BackgroundTasks(), service)
        
    except Exception as e:
        logger.error(f"Error in orchestrator analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Orchestrator analysis failed: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint for workflow integration"""
    try:
        return {
            "status": "healthy",
            "service": "workflow-intelligence-integration",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service unhealthy")

@router.get("/status")
async def get_status(
    service: IntelligenceService = Depends(get_intelligence_service)
):
    """
    Get current status of the intelligence service and its integrations.
    """
    try:
        # Check service availability
        service_status = {
            "service_name": settings.SERVICE_NAME,
            "port": settings.PORT,
            "debug_mode": settings.DEBUG,
            "ai_provider": settings.AI_PROVIDER,
            "supported_analysis_types": settings.SUPPORTED_ANALYSIS_TYPES,
            "max_requests_per_minute": settings.MAX_REQUESTS_PER_MINUTE,
            "cache_enabled": settings.ENABLE_PROMPT_CACHING,
        }
        
        # Check LLM provider availability
        from ..services.llm_providers import LLMProviderFactory
        providers_info = LLMProviderFactory.get_provider_info()
        
        return {
            "status": "operational",
            "service_config": service_status,
            "llm_providers": providers_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get service status: {str(e)}"
        )

@router.post("/analyze/generate-report")
async def generate_professional_report(
    request: ComprehensiveAnalysisRequest,
    background_tasks: BackgroundTasks,
    export_formats: List[str] = Query(default=["html", "pdf"], description="Export formats: html, pdf, json"),
    service: IntelligenceService = Depends(get_intelligence_service)
):
    """
    **ENHANCED REPORT GENERATION ENDPOINT**
    
    Generate professional analysis reports with export functionality.
    
    This endpoint:
    1. Performs enhanced comprehensive analysis
    2. Generates professional reports with visual charts
    3. Exports in multiple formats (HTML, PDF, JSON)
    4. Returns download links and report metadata
    
    Export formats:
    - html: Professional HTML report with embedded charts
    - pdf: Publication-ready PDF with professional formatting  
    - json: Raw analysis data for programmatic access
    """
    try:
        start_time = datetime.utcnow()
        logger.info(f"Starting report generation for: {request.song_metadata.title}")
        
        # Check if AI insights are pre-computed (from frontend UI)
        if request.ai_insights:
            logger.info("Using pre-computed AI insights from frontend UI")
            enhanced_results = request.ai_insights
        else:
            # Use AgentOrchestrator for comprehensive analysis with superior insights
            logger.info("Generating new AI insights using AgentOrchestrator")
            orchestrator = await get_agent_orchestrator()
            
            # Prepare request data for orchestrator
            request_data = {
                "audio_analysis": request.audio_analysis.model_dump() if request.audio_analysis else {},
                "content_analysis": request.content_analysis.model_dump() if request.content_analysis else {},
                "hit_prediction": request.hit_prediction.model_dump() if request.hit_prediction else {},
                "song_metadata": request.song_metadata.model_dump(),
                "request_id": request.request_id or str(uuid.uuid4())
            }
            
            # Perform comprehensive analysis with AgentOrchestrator's superior insights
            enhanced_results = await orchestrator.analyze_comprehensive_with_insights(request_data)
        
        # Generate professional reports
        logger.info("Generating professional reports with visual charts")
        generated_files = report_generator.generate_comprehensive_report(
            analysis_results=enhanced_results,
            song_metadata={
                'title': request.song_metadata.title,
                'artist': request.song_metadata.artist,
                'genre': request.song_metadata.genre or 'Unknown'
            },
            export_formats=export_formats
        )
        
        # Calculate processing time
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        # Log report generation in background
        background_tasks.add_task(
            _log_report_generation, 
            request, 
            generated_files, 
            processing_time
        )
        
        logger.info(f"Report generation completed in {processing_time:.0f}ms")
        
        return {
            "status": "success",
            "analysis_summary": {
                "overall_score": enhanced_results.get("overall_assessment", {}).get("overall_score", 0.75),
                "confidence_level": enhanced_results.get("confidence_level", 0.8),
                "insights_count": len(enhanced_results.get("insights", []))
            },
            "generated_reports": generated_files,
            "export_formats": export_formats,
            "processing_time_ms": processing_time,
            "timestamp": datetime.utcnow().isoformat(),
            "analysis_type": "enhanced_comprehensive_with_reports"
        }
        
    except Exception as e:
        logger.error(f"Error generating professional report: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(e)}"
        )

@router.get("/analyze/report-status/{request_id}")
async def get_report_status(request_id: str):
    """
    Get the status of a report generation request.
    
    This endpoint can be used to check the progress of long-running
    report generation tasks.
    """
    try:
        # In a production system, this would check a job queue or database
        # For now, return a simple status response
        return {
            "request_id": request_id,
            "status": "completed",  # In production: pending, processing, completed, failed
            "progress_percentage": 100,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Report generation system operational"
        }
        
    except Exception as e:
        logger.error(f"Error checking report status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get report status: {str(e)}"
        )

async def _log_report_generation(
    request: ComprehensiveAnalysisRequest,
    generated_files: Dict[str, str],
    processing_time: float
):
    """Log report generation completion for monitoring."""
    try:
        log_data = {
            "song_title": request.song_metadata.title,
            "artist": request.song_metadata.artist,
            "generated_formats": list(generated_files.keys()),
            "file_count": len(generated_files),
            "processing_time_ms": processing_time,
            "timestamp": datetime.utcnow().isoformat(),
            "service_version": "enhanced_v2.0"
        }
        
        logger.info(f"Report generation completed: {log_data}")
        
    except Exception as e:
        logger.error(f"Error logging report generation: {e}")

# Helper functions

async def _validate_analysis_data(request: ComprehensiveAnalysisRequest) -> Dict[str, Any]:
    """Validate that analysis data is complete and consistent."""
    errors = []
    warnings = []
    
    # Check audio analysis completeness
    audio = request.audio_analysis
    if not audio.tempo and not audio.energy and not audio.danceability:
        errors.append("Audio analysis appears incomplete - missing basic features")
    
    # Check content analysis completeness
    content = request.content_analysis
    if not content.raw_lyrics and not content.sentiment_score:
        warnings.append("Content analysis appears incomplete - limited lyrical data")
    
    # Check hit prediction completeness
    prediction = request.hit_prediction
    if prediction.hit_probability is None or prediction.hit_probability == 0:
        warnings.append("No hit prediction data provided - using fallback values")
    
    # Check metadata completeness
    metadata = request.song_metadata
    if not metadata.title or not metadata.artist:
        errors.append("Song metadata incomplete - title and artist required")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

async def _generate_comprehensive_analysis(
    request: ComprehensiveAnalysisRequest,
    service: IntelligenceService
) -> ComprehensiveAnalysisResponse:
    """
    Generate comprehensive analysis using the enhanced analysis service.
    
    This implementation uses professional-grade analysis chains with:
    - Musical meaning analysis
    - Hit pattern comparison with statistical benchmarking
    - Novelty assessment and innovation scoring
    - Production quality analysis
    - Comprehensive synthesis and recommendations
    """
    try:
        # Convert request to enhanced analysis format
        enhanced_request = _convert_to_enhanced_request(request)
        
        # Generate comprehensive analysis using enhanced service
        logger.info("Starting enhanced comprehensive analysis")
        enhanced_results = await enhanced_analysis_service.analyze_song(enhanced_request)
        
        # Convert enhanced results to response format
        response = await _convert_enhanced_results_to_response(request, enhanced_results)
        
        logger.info("Enhanced comprehensive analysis completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error in enhanced comprehensive analysis: {e}")
        # Fallback to basic analysis if enhanced fails
        logger.warning("Falling back to basic analysis")
        return await _generate_basic_analysis_fallback(request)
        
    except Exception as fallback_error:
        logger.error(f"Fallback analysis also failed: {fallback_error}")
        raise

def _convert_to_enhanced_request(request: ComprehensiveAnalysisRequest):
    """Convert workflow integration request to enhanced analysis service format."""
    from ..models.intelligence import InsightGenerationRequest, AudioFeatures, LyricsAnalysis, SongMetadata as EnhancedSongMetadata
    
    # Convert audio features
    audio_features = None
    if request.audio_analysis:
        audio_features = AudioFeatures(
            tempo=request.audio_analysis.tempo,
            energy=request.audio_analysis.energy,
            danceability=request.audio_analysis.danceability,
            valence=request.audio_analysis.valence,
            acousticness=request.audio_analysis.acousticness,
            loudness=request.audio_analysis.loudness,
            instrumentalness=request.audio_analysis.instrumentalness,
            liveness=request.audio_analysis.liveness,
            speechiness=request.audio_analysis.speechiness
        )
    
    # Convert lyrics analysis
    lyrics_analysis = None
    if request.content_analysis:
        lyrics_analysis = LyricsAnalysis(
            raw_text=request.content_analysis.raw_lyrics,
            sentiment_polarity=request.content_analysis.sentiment_score,
            emotion_scores=request.content_analysis.emotion_scores or {},
            themes=request.content_analysis.themes or [],
            complexity_score=request.content_analysis.complexity_score
        )
    
    # Convert song metadata
    song_metadata = None
    if request.song_metadata:
        song_metadata = EnhancedSongMetadata(
            title=request.song_metadata.title,
            artist=request.song_metadata.artist,
            genre=request.song_metadata.genre,
            duration=request.song_metadata.duration_seconds,
            release_date=request.song_metadata.release_date.isoformat() if request.song_metadata.release_date else None
        )
    
    return InsightGenerationRequest(
        audio_features=audio_features,
        lyrics_analysis=lyrics_analysis,
        song_metadata=song_metadata,
        hit_prediction_score=request.hit_prediction.hit_probability,
        analysis_depth=request.analysis_depth,
        focus_areas=request.focus_areas or []
    )

async def _convert_enhanced_results_to_response(
    request: ComprehensiveAnalysisRequest,
    enhanced_results: Dict[str, Any]
) -> ComprehensiveAnalysisResponse:
    """Convert enhanced analysis results to workflow integration response format."""
    
    # Extract analysis components
    musical_meaning = enhanced_results.get('musical_meaning', {})
    hit_comparison = enhanced_results.get('hit_comparison', {})
    novelty_assessment = enhanced_results.get('novelty_assessment', {})
    production_analysis = enhanced_results.get('production_analysis', {})
    comprehensive_analysis = enhanced_results.get('comprehensive_analysis', {})
    
    # Generate insights from enhanced results
    insights = _extract_insights_from_enhanced_results(enhanced_results)
    
    # Extract recommendations
    recommendations = _extract_recommendations_from_enhanced_results(enhanced_results)
    
    # Calculate overall score from enhanced results
    overall_score = _calculate_enhanced_overall_score(enhanced_results, request)
    
    # Generate executive summary from comprehensive analysis
    executive_summary = _generate_enhanced_executive_summary(comprehensive_analysis, request)
    
    return ComprehensiveAnalysisResponse(
        request_id=request.request_id or str(uuid.uuid4()),
        analysis_type="enhanced_comprehensive",
        executive_summary=executive_summary,
        overall_score=overall_score,
        confidence_level=enhanced_results.get('analysis_metadata', {}).get('statistical_confidence', 0.85),
        insights=insights,
        strategic_recommendations=recommendations.get('strategic', []),
        tactical_recommendations=recommendations.get('tactical', []),
        risk_factors=_extract_risk_factors(enhanced_results),
        opportunities=_extract_opportunities(enhanced_results),
        market_positioning=_extract_market_positioning(enhanced_results),
        target_demographics=_extract_target_demographics(enhanced_results),
        production_feedback=_extract_production_feedback(enhanced_results),
        technical_strengths=_extract_technical_strengths(enhanced_results),
        technical_improvements=_extract_technical_improvements(enhanced_results),
        processing_time_ms=0  # Will be set by caller
    )

def _extract_insights_from_enhanced_results(enhanced_results: Dict[str, Any]) -> List[AnalysisInsight]:
    """Extract insights from enhanced analysis results."""
    insights = []
    
    # Musical meaning insights
    musical_meaning = enhanced_results.get('musical_meaning', {})
    if musical_meaning.get('emotional_landscape'):
        emotional = musical_meaning['emotional_landscape']
        insights.append(AnalysisInsight(
            category="Musical Character",
            title="Emotional Landscape",
            description=emotional.get('primary_emotion', 'Complex emotional character'),
            confidence=0.9,
            evidence=[emotional.get('emotional_arc', 'Emotional analysis completed')],
            recommendations=[]
        ))
    
    # Hit comparison insights  
    hit_comparison = enhanced_results.get('hit_comparison', {})
    if hit_comparison.get('statistical_analysis'):
        stat_analysis = hit_comparison['statistical_analysis']
        insights.append(AnalysisInsight(
            category="Commercial Potential",
            title="Hit Probability Analysis",
            description=f"Statistical analysis indicates {stat_analysis.get('overall_hit_probability', 'moderate')} commercial potential",
            confidence=0.85,
            evidence=stat_analysis.get('key_success_factors', []),
            recommendations=hit_comparison.get('optimization_strategy', {}).get('immediate_actions', [])
        ))
    
    # Innovation insights
    novelty = enhanced_results.get('novelty_assessment', {})
    if novelty.get('innovation_scores'):
        innovation = novelty['innovation_scores']
        overall_innovation = innovation.get('overall_innovation', 0.5)
        insights.append(AnalysisInsight(
            category="Innovation Assessment",
            title="Creative Differentiation",
            description=f"Innovation score of {overall_innovation:.1%} indicates {'high' if overall_innovation > 0.7 else 'moderate' if overall_innovation > 0.4 else 'low'} creative differentiation",
            confidence=0.8,
            evidence=novelty.get('differentiation_analysis', {}).get('unique_selling_points', []),
            recommendations=novelty.get('strategic_recommendations', {}).get('leverage_innovation', [])
        ))
    
    return insights

def _extract_recommendations_from_enhanced_results(enhanced_results: Dict[str, Any]) -> Dict[str, List[str]]:
    """Extract categorized recommendations from enhanced results."""
    recommendations = {'strategic': [], 'tactical': []}
    
    # Extract from comprehensive analysis
    comprehensive = enhanced_results.get('comprehensive_analysis', {})
    action_plan = comprehensive.get('action_plan', {})
    
    recommendations['strategic'].extend(action_plan.get('production_priorities', [])[:3])
    recommendations['tactical'].extend(action_plan.get('arrangement_improvements', [])[:3])
    
    # Extract from hit comparison
    hit_comparison = enhanced_results.get('hit_comparison', {})
    optimization = hit_comparison.get('optimization_strategy', {})
    recommendations['strategic'].extend(optimization.get('immediate_actions', [])[:2])
    
    return recommendations

def _calculate_enhanced_overall_score(enhanced_results: Dict[str, Any], request: ComprehensiveAnalysisRequest) -> float:
    """Calculate overall score from enhanced analysis results."""
    score_components = []
    
    # Hit probability component (40%)
    hit_comparison = enhanced_results.get('hit_comparison', {})
    hit_prob_str = hit_comparison.get('statistical_analysis', {}).get('overall_hit_probability', '50%')
    try:
        hit_prob = float(hit_prob_str.replace('%', '')) / 100
        score_components.append(hit_prob * 0.4)
    except:
        score_components.append(0.5 * 0.4)
    
    # Innovation component (20%)
    novelty = enhanced_results.get('novelty_assessment', {})
    innovation_score = novelty.get('innovation_scores', {}).get('overall_innovation', 0.5)
    score_components.append(float(innovation_score) * 0.2)
    
    # Production quality component (25%)
    production = enhanced_results.get('production_analysis', {})
    tech_assessment = production.get('technical_assessment', {})
    grade = tech_assessment.get('overall_technical_grade', 'C')
    grade_score = {'A': 0.95, 'B': 0.85, 'C': 0.75, 'D': 0.65, 'F': 0.5}.get(grade, 0.75)
    score_components.append(grade_score * 0.25)
    
    # Musical coherence component (15%)
    musical_meaning = enhanced_results.get('musical_meaning', {})
    if musical_meaning:
        score_components.append(0.8 * 0.15)  # Assume good musical coherence if analysis completed
    else:
        score_components.append(0.5 * 0.15)
    
    return sum(score_components)

def _generate_enhanced_executive_summary(comprehensive_analysis: Dict[str, Any], request: ComprehensiveAnalysisRequest) -> str:
    """Generate executive summary from enhanced comprehensive analysis."""
    executive_summary = comprehensive_analysis.get('executive_summary', {})
    
    if executive_summary:
        commercial_potential = executive_summary.get('commercial_potential', 'Medium')
        confidence = executive_summary.get('confidence_level', 'N/A')
        investment_rec = executive_summary.get('investment_recommendation', '')
        
        summary = f"Professional analysis of '{request.song_metadata.title}' by {request.song_metadata.artist} "
        summary += f"indicates {commercial_potential.lower()} commercial potential with {confidence} confidence. "
        summary += investment_rec
        
        return summary
    else:
        return f"Enhanced analysis completed for '{request.song_metadata.title}' by {request.song_metadata.artist}."

def _extract_risk_factors(enhanced_results: Dict[str, Any]) -> List[str]:
    """Extract risk factors from enhanced analysis."""
    risks = []
    
    # From hit comparison
    hit_comparison = enhanced_results.get('hit_comparison', {})
    stat_analysis = hit_comparison.get('statistical_analysis', {})
    risks.extend(stat_analysis.get('risk_factors', []))
    
    # From production analysis
    production = enhanced_results.get('production_analysis', {})
    improvement_plan = production.get('improvement_plan', {})
    risks.extend([f"Production: {fix}" for fix in improvement_plan.get('critical_fixes', [])[:2]])
    
    return risks

def _extract_opportunities(enhanced_results: Dict[str, Any]) -> List[str]:
    """Extract opportunities from enhanced analysis."""
    opportunities = []
    
    # From novelty assessment
    novelty = enhanced_results.get('novelty_assessment', {})
    differentiation = novelty.get('differentiation_analysis', {})
    opportunities.extend(differentiation.get('unique_selling_points', [])[:2])
    
    # From market strategy
    comprehensive = enhanced_results.get('comprehensive_analysis', {})
    market_strategy = comprehensive.get('market_strategy', {})
    opportunities.extend(market_strategy.get('promotional_angles', [])[:2])
    
    return opportunities

def _extract_market_positioning(enhanced_results: Dict[str, Any]) -> str:
    """Extract market positioning from enhanced analysis."""
    comprehensive = enhanced_results.get('comprehensive_analysis', {})
    market_strategy = comprehensive.get('market_strategy', {})
    return market_strategy.get('positioning_strategy', 'Standard market positioning recommended')

def _extract_target_demographics(enhanced_results: Dict[str, Any]) -> List[str]:
    """Extract target demographics from enhanced analysis."""
    comprehensive = enhanced_results.get('comprehensive_analysis', {})
    market_strategy = comprehensive.get('market_strategy', {})
    primary_demo = market_strategy.get('primary_demographic', '')
    
    if primary_demo:
        return [primary_demo]
    return ['Mainstream music audiences']

def _extract_production_feedback(enhanced_results: Dict[str, Any]) -> List[str]:
    """Extract production feedback from enhanced analysis."""
    production = enhanced_results.get('production_analysis', {})
    improvement_plan = production.get('improvement_plan', {})
    feedback = improvement_plan.get('enhancement_suggestions', [])
    return feedback[:3]  # Top 3 suggestions

def _extract_technical_strengths(enhanced_results: Dict[str, Any]) -> List[str]:
    """Extract technical strengths from enhanced analysis."""
    production = enhanced_results.get('production_analysis', {})
    mix_evaluation = production.get('mix_evaluation', {})
    
    strengths = []
    if mix_evaluation.get('clarity_score', '0/10').startswith(('8', '9', '10')):
        strengths.append("Excellent mix clarity")
    if mix_evaluation.get('balance_score', '0/10').startswith(('8', '9', '10')):
        strengths.append("Well-balanced frequency spectrum")
    
    return strengths

def _extract_technical_improvements(enhanced_results: Dict[str, Any]) -> List[str]:
    """Extract technical improvements from enhanced analysis."""
    production = enhanced_results.get('production_analysis', {})
    improvement_plan = production.get('improvement_plan', {})
    return improvement_plan.get('critical_fixes', [])

async def _generate_basic_analysis_fallback(request: ComprehensiveAnalysisRequest) -> ComprehensiveAnalysisResponse:
    """Generate basic analysis as fallback when enhanced analysis fails."""
    logger.info("Generating basic analysis fallback")
    
    # Generate basic insights
    insights = await _generate_insights(request)
    
    # Calculate basic overall score
    overall_score = _calculate_overall_score(request)
    
    # Generate basic executive summary
    executive_summary = _generate_executive_summary(request, insights, overall_score)
    
    # Generate basic recommendations
    strategic_recs, tactical_recs = _generate_recommendations(request, insights)
    
    return ComprehensiveAnalysisResponse(
        request_id=request.request_id or str(uuid.uuid4()),
        analysis_type="basic_fallback",
        executive_summary=executive_summary,
        overall_score=overall_score,
        confidence_level=_calculate_confidence_level(request),
        insights=insights,
        strategic_recommendations=strategic_recs,
        tactical_recommendations=tactical_recs,
        risk_factors=["Enhanced analysis unavailable"],
        opportunities=["Consider re-running analysis when service is available"],
        market_positioning="Standard positioning recommended",
        target_demographics=["General audience"],
        production_feedback=["Basic analysis - limited feedback available"],
        technical_strengths=["Analysis limited due to service unavailability"],
        technical_improvements=["Enhanced analysis recommended for detailed feedback"],
        processing_time_ms=0
    )

async def _generate_insights(request: ComprehensiveAnalysisRequest) -> List[AnalysisInsight]:
    """Generate detailed insights from the analysis data."""
    insights = []
    
    # Musical Analysis Insights
    audio = request.audio_analysis
    if audio.tempo:
        tempo_insight = AnalysisInsight(
            category="Musical Analysis",
            title="Tempo Analysis",
            description=f"The song has a tempo of {audio.tempo:.1f} BPM, which is {'within' if 80 <= audio.tempo <= 140 else 'outside'} the typical range for commercial hits.",
            confidence=0.9,
            evidence=[f"Tempo: {audio.tempo:.1f} BPM"],
            recommendations=["Consider tempo adjustments if targeting mainstream radio"] if audio.tempo > 140 or audio.tempo < 80 else []
        )
        insights.append(tempo_insight)
    
    if audio.energy is not None:
        energy_insight = AnalysisInsight(
            category="Musical Analysis",
            title="Energy Level",
            description=f"The song has {'high' if audio.energy > 0.7 else 'moderate' if audio.energy > 0.4 else 'low'} energy ({audio.energy:.2f}), which affects its commercial appeal.",
            confidence=0.85,
            evidence=[f"Energy level: {audio.energy:.2f}"],
            recommendations=["High energy tracks work well for playlists and radio"] if audio.energy > 0.7 else []
        )
        insights.append(energy_insight)
    
    # Content Analysis Insights
    content = request.content_analysis
    if content.sentiment_score is not None:
        sentiment_insight = AnalysisInsight(
            category="Lyrical Analysis",
            title="Emotional Sentiment",
            description=f"The lyrics convey {'positive' if content.sentiment_score > 0.1 else 'negative' if content.sentiment_score < -0.1 else 'neutral'} sentiment ({content.sentiment_score:.2f}).",
            confidence=0.8,
            evidence=[f"Sentiment score: {content.sentiment_score:.2f}"],
            recommendations=["Positive sentiment generally correlates with commercial success"] if content.sentiment_score > 0.1 else []
        )
        insights.append(sentiment_insight)
    
    # Hit Prediction Insights
    prediction = request.hit_prediction
    if prediction.hit_probability is not None:
        hit_insight = AnalysisInsight(
            category="Commercial Potential",
            title="Hit Probability Assessment",
            description=f"The song has a {prediction.hit_probability:.1%} probability of becoming a hit based on ML analysis.",
            confidence=prediction.confidence_score or 0.75,
            evidence=[f"Hit probability: {prediction.hit_probability:.1%}"],
            recommendations=["Strong commercial potential - consider prioritizing for release"] if prediction.hit_probability > 0.7 else []
        )
        insights.append(hit_insight)
    
    return insights

def _calculate_overall_score(request: ComprehensiveAnalysisRequest) -> float:
    """Calculate an overall quality/potential score."""
    scores = []
    
    # Hit prediction weight (40%)
    if request.hit_prediction.hit_probability is not None:
        scores.append(request.hit_prediction.hit_probability * 0.4)
    
    # Audio quality weight (30%)
    audio = request.audio_analysis
    if audio.energy is not None and audio.danceability is not None:
        audio_score = (audio.energy + audio.danceability) / 2
        scores.append(audio_score * 0.3)
    
    # Content quality weight (30%)
    content = request.content_analysis
    if content.sentiment_score is not None:
        # Normalize sentiment score to 0-1 range
        sentiment_normalized = (content.sentiment_score + 1) / 2
        scores.append(sentiment_normalized * 0.3)
    
    return sum(scores) if scores else 0.5

def _calculate_confidence_level(request: ComprehensiveAnalysisRequest) -> float:
    """Calculate confidence level in the analysis."""
    confidence_factors = []
    
    # Data completeness
    data_completeness = 0
    if request.audio_analysis.tempo is not None:
        data_completeness += 0.33
    if request.content_analysis.raw_lyrics:
        data_completeness += 0.33
    if request.hit_prediction.hit_probability is not None:
        data_completeness += 0.34
    
    confidence_factors.append(data_completeness)
    
    # ML model confidence
    if request.hit_prediction.confidence_score is not None:
        confidence_factors.append(request.hit_prediction.confidence_score)
    
    return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5

def _generate_executive_summary(request: ComprehensiveAnalysisRequest, insights: List[AnalysisInsight], overall_score: float) -> str:
    """Generate an executive summary of the analysis."""
    song_title = request.song_metadata.title
    artist = request.song_metadata.artist
    
    hit_prob = request.hit_prediction.hit_probability or 0
    
    summary = f"Analysis of '{song_title}' by {artist} reveals "
    
    if overall_score > 0.8:
        summary += "exceptional commercial potential with strong musical and lyrical elements. "
    elif overall_score > 0.6:
        summary += "solid commercial potential with notable strengths. "
    else:
        summary += "moderate commercial potential with areas for improvement. "
    
    summary += f"The track shows a {hit_prob:.1%} probability of commercial success based on ML analysis."
    
    return summary

def _generate_recommendations(request: ComprehensiveAnalysisRequest, insights: List[AnalysisInsight]) -> tuple:
    """Generate strategic and tactical recommendations."""
    strategic_recs = []
    tactical_recs = []
    
    hit_prob = request.hit_prediction.hit_probability or 0
    
    # Strategic recommendations
    if hit_prob > 0.7:
        strategic_recs.append("Prioritize this track for major label release")
        strategic_recs.append("Allocate significant marketing budget")
    elif hit_prob > 0.5:
        strategic_recs.append("Consider for independent release with targeted marketing")
    else:
        strategic_recs.append("Evaluate for niche market or artistic release")
    
    # Tactical recommendations
    audio = request.audio_analysis
    if audio.energy and audio.energy > 0.7:
        tactical_recs.append("Target high-energy playlists and workout music")
    
    if audio.danceability and audio.danceability > 0.7:
        tactical_recs.append("Submit to dance and electronic music playlists")
    
    content = request.content_analysis
    if content.sentiment_score and content.sentiment_score > 0.5:
        tactical_recs.append("Leverage positive messaging in marketing campaigns")
    
    return strategic_recs, tactical_recs

def _assess_risks_and_opportunities(request: ComprehensiveAnalysisRequest) -> tuple:
    """Assess risks and opportunities."""
    risks = []
    opportunities = []
    
    # Risk assessment
    hit_prob = request.hit_prediction.hit_probability or 0
    if hit_prob < 0.3:
        risks.append("Low commercial viability based on current market trends")
    
    audio = request.audio_analysis
    if audio.tempo and (audio.tempo > 160 or audio.tempo < 70):
        risks.append("Tempo may limit mainstream radio play")
    
    # Opportunity assessment
    if hit_prob > 0.6:
        opportunities.append("Strong potential for playlist placement")
    
    if audio.energy and audio.energy > 0.8:
        opportunities.append("Suitable for sync licensing in high-energy media")
    
    return risks, opportunities

def _generate_market_positioning(request: ComprehensiveAnalysisRequest) -> str:
    """Generate market positioning recommendation."""
    genre = request.song_metadata.genre or "Unknown"
    audio = request.audio_analysis
    
    if audio.danceability and audio.danceability > 0.7:
        return f"Position as a danceable {genre} track for club and festival markets"
    elif audio.energy and audio.energy > 0.7:
        return f"Position as a high-energy {genre} track for rock/pop markets"
    else:
        return f"Position as an alternative {genre} track for indie markets"

def _identify_target_demographics(request: ComprehensiveAnalysisRequest) -> List[str]:
    """Identify target demographics."""
    demographics = []
    
    audio = request.audio_analysis
    content = request.content_analysis
    
    # Age demographics based on energy and style
    if audio.energy and audio.energy > 0.7:
        demographics.append("18-34 year olds")
    else:
        demographics.append("25-45 year olds")
    
    # Content-based demographics
    if content.sentiment_score and content.sentiment_score > 0.5:
        demographics.append("Mainstream pop audiences")
    
    # Genre-based demographics
    genre = request.song_metadata.genre
    if genre:
        demographics.append(f"{genre} music enthusiasts")
    
    return demographics

def _generate_production_feedback(request: ComprehensiveAnalysisRequest) -> List[str]:
    """Generate production quality feedback."""
    feedback = []
    
    audio = request.audio_analysis
    
    if audio.audio_quality_score and audio.audio_quality_score < 0.7:
        feedback.append("Audio quality could be improved for commercial release")
    
    if audio.loudness and audio.loudness < -16:
        feedback.append("Track may benefit from additional mastering for competitive loudness")
    
    if not feedback:
        feedback.append("Production quality appears suitable for commercial release")
    
    return feedback

def _identify_technical_strengths(request: ComprehensiveAnalysisRequest) -> List[str]:
    """Identify technical strengths."""
    strengths = []
    
    audio = request.audio_analysis
    
    if audio.energy and audio.energy > 0.7:
        strengths.append("Strong energy levels maintain listener engagement")
    
    if audio.danceability and audio.danceability > 0.7:
        strengths.append("Excellent danceability for club and playlist appeal")
    
    if audio.audio_quality_score and audio.audio_quality_score > 0.8:
        strengths.append("High audio quality suitable for all platforms")
    
    return strengths

def _suggest_technical_improvements(request: ComprehensiveAnalysisRequest) -> List[str]:
    """Suggest technical improvements."""
    improvements = []
    
    audio = request.audio_analysis
    
    if audio.energy and audio.energy < 0.4:
        improvements.append("Consider increasing energy levels for better commercial appeal")
    
    if audio.loudness and audio.loudness < -20:
        improvements.append("Consider additional mastering to achieve competitive loudness")
    
    if audio.danceability and audio.danceability < 0.3:
        improvements.append("Consider rhythmic adjustments to improve danceability")
    
    return improvements

async def _log_comprehensive_analysis(
    request: ComprehensiveAnalysisRequest, 
    response: ComprehensiveAnalysisResponse
):
    """Log comprehensive analysis completion for monitoring and analytics."""
    try:
        log_data = {
            "song_title": request.song_metadata.title,
            "artist": request.song_metadata.artist,
            "analysis_depth": request.analysis_depth,
            "overall_score": response.overall_score,
            "hit_probability": request.hit_prediction.hit_probability,
            "processing_time_ms": response.processing_time_ms,
            "confidence_level": response.confidence_level,
            "insights_count": len(response.insights),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Analysis completed: {log_data}")
        
        # TODO: Send to monitoring/analytics service
        
    except Exception as e:
        logger.error(f"Error logging analysis: {e}") 