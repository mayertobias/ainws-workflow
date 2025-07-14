"""
Agent Orchestrator for Agentic AI System

Coordinates multiple specialized agents to perform comprehensive analysis through
multi-agent collaboration, task distribution, and result synthesis.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent
from .music_analysis_agent import MusicAnalysisAgent
from .commercial_analysis_agent import CommercialAnalysisAgent
from .novelty_assessment_agent import NoveltyAssessmentAgent
from .enhanced_quality_assurance_agent import EnhancedQualityAssuranceAgent
from ..models.agentic_models import (
    AgentProfile, AgentRole, ToolType, AgentTask, TaskType,
    AgentContribution, AgenticAnalysisRequest, AgenticAnalysisResponse,
    CoordinationStrategy, AgentStatus
)

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """
    Orchestrates multiple specialized agents for comprehensive analysis.
    
    Responsibilities:
    - Agent lifecycle management
    - Task distribution and coordination
    - Inter-agent communication
    - Result synthesis and quality assurance
    - Performance monitoring
    """
    
    def __init__(self):
        """Initialize the agent orchestrator with preserved superior insights."""
        self.agents: Dict[AgentRole, BaseAgent] = {}
        self.active_tasks: Dict[str, AgentTask] = {}
        self.coordination_strategies: Dict[str, CoordinationStrategy] = {}
        
        # PRESERVED: Superior insight services from original architecture
        from ..services.historical_benchmarking_service import HistoricalBenchmarkingService
        from ..services.intelligent_caching_service import IntelligentCachingService
        from ..services.historical_data_integrator import HistoricalDataIntegrator
        
        self.historical_benchmarking = HistoricalBenchmarkingService()
        self.intelligent_caching = IntelligentCachingService()
        self.historical_data = HistoricalDataIntegrator()
        
        # Performance tracking (enhanced with preserved metrics)
        self.total_analyses = 0
        self.successful_analyses = 0
        self.average_processing_time = 0.0
        
        # Initialize agents
        self._initialize_agents()
        
        logger.info("Agent Orchestrator initialized with specialized agents and preserved superior insights")
    
    async def coordinate_analysis(self, request_data: Dict[str, Any]) -> AgenticAnalysisResponse:
        """
        Coordinate analysis with a simplified interface for testing.
        
        Args:
            request_data: Dictionary containing analysis data
            
        Returns:
            AgenticAnalysisResponse: Analysis results
        """
        # Create a simplified agentic analysis request
        agentic_request = AgenticAnalysisRequest(
            request_id=f"coord_{datetime.utcnow().timestamp()}",
            input_data=request_data,
            required_agent_roles=[AgentRole.MUSIC_ANALYSIS, AgentRole.COMMERCIAL_ANALYSIS],
            coordination_strategy=CoordinationStrategy.COLLABORATIVE,
            focus_areas=["comprehensive_analysis"],
            constraints=[],
            max_execution_time=300
        )
        
        return await self.perform_agentic_analysis(agentic_request)
    
    async def analyze_comprehensive_with_insights(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced comprehensive analysis that preserves all superior insights.
        
        This method integrates:
        - Historical benchmarking and genre analysis
        - ML prediction integration with sophisticated feature transformation
        - Intelligent caching and performance metrics
        - Multi-agent parallel execution
        """
        start_time = datetime.utcnow()
        analysis_id = f"comprehensive_{start_time.timestamp()}"
        
        try:
            logger.info(f"Starting enhanced comprehensive analysis {analysis_id}")
            
            # STEP 1: Check intelligent cache first
            cache_key = self.intelligent_caching.generate_cache_key(request_data)
            cached_result = await self.intelligent_caching.get_cached_analysis(cache_key)
            
            if cached_result:
                logger.info(f"Returning cached comprehensive analysis {analysis_id}")
                self.intelligent_caching.update_performance_metrics(cached_result)
                return cached_result
            
            # STEP 2: Enrich request with superior insights
            enriched_request = await self._enrich_request_with_insights(request_data)
            
            # STEP 3: Use provided hit prediction data (from ML Prediction Service)
            hit_prediction = enriched_request.get("hit_prediction", {})
            
            # Check for both field name formats: hit_probability (from frontend) and prediction (internal)
            hit_prob = hit_prediction.get("hit_probability") or hit_prediction.get("prediction", 0)
            confidence = hit_prediction.get("confidence_score") or hit_prediction.get("confidence", 0)
            
            if not hit_prediction or hit_prob == 0:
                logger.warning("No hit prediction data provided - using fallback values")
                hit_prediction = {
                    "prediction": 0.5,
                    "hit_probability": 0.5,
                    "confidence": 0.5,
                    "confidence_score": 0.5,
                    "model_used": "fallback",
                    "note": "No ML prediction data provided"
                }
                enriched_request["hit_prediction"] = hit_prediction
            else:
                # Normalize field names for internal use
                hit_prediction["prediction"] = hit_prob
                hit_prediction["confidence"] = confidence
                logger.info(f"Using hit prediction: {hit_prob:.3f} with confidence: {confidence:.3f}")
                enriched_request["hit_prediction"] = hit_prediction
            
            # STEP 4: Get historical benchmarking analysis
            genre = enriched_request.get("song_metadata", {}).get("genre", "pop")
            benchmark_analysis = self.historical_benchmarking.analyze_against_benchmarks(
                enriched_request.get("audio_analysis", {}),
                genre
            )
            enriched_request["benchmark_analysis"] = benchmark_analysis
            
            # STEP 5: Execute specialized agents in parallel (core agent logic preserved)
            logger.info("Executing specialized agents in parallel with enhanced context...")
            
            # Create proper AgentTask objects for each agent
            music_task = AgentTask(
                agent_id="music_analysis_agent",
                task_type=TaskType.ANALYSIS,
                description="Perform comprehensive musical analysis including audio features, harmonic analysis, and genre classification",
                input_data={"focus": "musical_analysis", "benchmarks": benchmark_analysis}
            )
            
            commercial_task = AgentTask(
                agent_id="commercial_analysis_agent", 
                task_type=TaskType.ANALYSIS,
                description="Perform commercial viability analysis including hit prediction and market assessment",
                input_data={"focus": "commercial_analysis", "hit_prediction": hit_prediction}
            )
            
            agent_tasks = [
                self._execute_agent_safely(AgentRole.MUSIC_ANALYSIS, enriched_request, music_task),
                self._execute_agent_safely(AgentRole.COMMERCIAL_ANALYSIS, enriched_request, commercial_task)
            ]
            
            # Add NoveltyAssessment if available
            if AgentRole.NOVELTY_ASSESSMENT in self.agents:
                novelty_task = AgentTask(
                    agent_id="novelty_assessment_agent",
                    task_type=TaskType.ANALYSIS,
                    description="Assess innovation and novelty factors for commercial differentiation",
                    input_data={"focus": "novelty_assessment"}
                )
                agent_tasks.append(
                    self._execute_agent_safely(AgentRole.NOVELTY_ASSESSMENT, enriched_request, novelty_task)
                )
            
            # Log what data agents will receive
            logger.info(f"Sending data to agents:")
            logger.info(f"  - Audio features available: {len(enriched_request.get('audio_analysis', {}))}")
            logger.info(f"  - Content features available: {len(enriched_request.get('content_analysis', {}))}")
            logger.info(f"  - Hit prediction: {hit_prediction.get('prediction', 'N/A')}")
            logger.info(f"  - Genre: {enriched_request.get('song_metadata', {}).get('genre', 'N/A')}")
            
            # Execute all agent tasks in parallel
            results = await asyncio.gather(*agent_tasks, return_exceptions=True)
            
            # DETAILED LOGGING: Show what each agent returned
            logger.info("="*60)
            logger.info("🤖 AGENT EXECUTION RESULTS")
            logger.info("="*60)
            
            for i, result in enumerate(results):
                agent_name = ["MusicAnalysisAgent", "CommercialAnalysisAgent", "NoveltyAssessmentAgent"][i] if i < 3 else f"Agent{i}"
                if isinstance(result, Exception):
                    logger.error(f"🚨 {agent_name} FAILED: {result}")
                else:
                    logger.info(f"✅ {agent_name} SUCCESS:")
                    if isinstance(result, dict):
                        logger.info(f"   📊 Keys: {list(result.keys())}")
                        if "findings" in result:
                            logger.info(f"   🔍 Findings: {result['findings'][:3] if isinstance(result['findings'], list) else result['findings']}")
                        if "insights" in result:
                            logger.info(f"   💡 Insights: {result['insights'][:3] if isinstance(result['insights'], list) else result['insights']}")
                        if "recommendations" in result:
                            logger.info(f"   🚀 Recommendations: {result['recommendations'][:3] if isinstance(result['recommendations'], list) else result['recommendations']}")
                        if "confidence_level" in result:
                            logger.info(f"   🎯 Confidence: {result['confidence_level']}")
                    else:
                        logger.info(f"   📋 Result: {str(result)[:200]}...")
            
            logger.info("="*60)
            logger.info("🎉 END OF AGENT EXECUTION RESULTS")
            logger.info("="*60)
            
            # STEP 6: Process agent results with error handling
            music_analysis = results[0] if not isinstance(results[0], Exception) else {"error": str(results[0]), "agent": "MusicAnalysisAgent"}
            commercial_analysis = results[1] if not isinstance(results[1], Exception) else {"error": str(results[1]), "agent": "CommercialAnalysisAgent"}
            novelty_analysis = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else {"analysis": "novelty_not_available"}
            
            # Convert AgentContribution objects to dictionaries for QualityAssurance agent
            if hasattr(music_analysis, 'model_dump'):
                music_analysis_dict = music_analysis.model_dump()
            elif hasattr(music_analysis, 'dict'):
                music_analysis_dict = music_analysis.dict()
            else:
                music_analysis_dict = music_analysis
            
            if hasattr(commercial_analysis, 'model_dump'):
                commercial_analysis_dict = commercial_analysis.model_dump()
            elif hasattr(commercial_analysis, 'dict'):
                commercial_analysis_dict = commercial_analysis.dict()
            else:
                commercial_analysis_dict = commercial_analysis
            
            if hasattr(novelty_analysis, 'model_dump'):
                novelty_analysis_dict = novelty_analysis.model_dump()
            elif hasattr(novelty_analysis, 'dict'):
                novelty_analysis_dict = novelty_analysis.dict()
            else:
                novelty_analysis_dict = novelty_analysis
            
            # DETAILED LOGGING: Show what's being passed to QualityAssurance
            logger.info("="*60)
            logger.info("🔍 DATA PASSED TO QUALITY ASSURANCE AGENT")
            logger.info("="*60)
            logger.info(f"🎵 Music Analysis Keys: {list(music_analysis_dict.keys()) if isinstance(music_analysis_dict, dict) else 'Not a dict'}")
            if isinstance(music_analysis_dict, dict):
                logger.info(f"   📊 Music Findings: {music_analysis_dict.get('findings', 'None')}")
                logger.info(f"   📊 Music Insights: {music_analysis_dict.get('insights', 'None')}")
            
            logger.info(f"💼 Commercial Analysis Keys: {list(commercial_analysis_dict.keys()) if isinstance(commercial_analysis_dict, dict) else 'Not a dict'}")
            if isinstance(commercial_analysis_dict, dict):
                logger.info(f"   📊 Commercial Findings: {commercial_analysis_dict.get('findings', 'None')}")
                logger.info(f"   📊 Commercial Insights: {commercial_analysis_dict.get('insights', 'None')}")
            
            logger.info(f"🚀 Novelty Analysis Keys: {list(novelty_analysis_dict.keys()) if isinstance(novelty_analysis_dict, dict) else 'Not a dict'}")
            if isinstance(novelty_analysis_dict, dict):
                logger.info(f"   📊 Novelty Findings: {novelty_analysis_dict.get('findings', 'None')}")
                logger.info(f"   📊 Novelty Insights: {novelty_analysis_dict.get('insights', 'None')}")
            
            logger.info("="*60)
            logger.info("🎉 END OF QA INPUT DATA")
            logger.info("="*60)
            
            # STEP 7: Synthesize with QualityAssurance agent if available
            synthesis_request = {
                "original_request": enriched_request,
                "music_analysis": music_analysis_dict,
                "commercial_analysis": commercial_analysis_dict,
                "novelty_analysis": novelty_analysis_dict,
                "benchmark_analysis": benchmark_analysis,
                "hit_prediction": hit_prediction,
                "analysis_id": analysis_id
            }
            
            if AgentRole.QUALITY_ASSURANCE in self.agents:
                qa_result = await self.agents[AgentRole.QUALITY_ASSURANCE].analyze_task(
                    synthesis_request,
                    {"focus": "comprehensive_synthesis"}
                )
                # Extract the rich content from QualityAssurance agent result
                qa_dict = qa_result.dict() if hasattr(qa_result, 'dict') else qa_result
                
                # Create final report in the format expected by _convert_to_response_model
                final_report = {
                    "executive_summary": f"Comprehensive analysis completed using enhanced parallel agentic workflow with {len(qa_dict.get('findings', []))} findings and {len(qa_dict.get('insights', []))} insights.",
                    "findings": qa_dict.get("findings", []),
                    "insights": qa_dict.get("insights", []),  
                    "recommendations": qa_dict.get("recommendations", []),
                    "evidence": qa_dict.get("evidence", []),
                    "methodology": qa_dict.get("methodology", "Enhanced parallel agentic analysis"),
                    "tools_used": qa_dict.get("tools_used", []),
                    "confidence_level": qa_dict.get("confidence_level", 0.85),
                    "overall_assessment": {
                        "artistic_score": music_analysis_dict.get("confidence_level", 0.7),
                        "commercial_score": commercial_analysis_dict.get("confidence_level", 0.7),
                        "innovation_score": novelty_analysis_dict.get("confidence_level", 0.7),
                        "hit_prediction_score": hit_prediction.get("prediction", 0.0)
                    }
                }
            else:
                # Fallback synthesis
                final_report = self._create_fallback_synthesis(synthesis_request)
            
            # STEP 8: Add enhanced metadata and performance tracking
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds() * 1000
            
            # Ensure final_report is a dictionary before updating
            if not isinstance(final_report, dict):
                final_report = final_report.dict() if hasattr(final_report, 'dict') else {}
            
            final_report.update({
                "analysis_id": analysis_id,
                "processing_time_ms": processing_time,
                "timestamp": end_time.isoformat(),
                "architecture": "enhanced_parallel_agentic_workflow",
                "agents_used": [
                    "MusicAnalysisAgent", 
                    "CommercialAnalysisAgent", 
                    "NoveltyAssessmentAgent" if AgentRole.NOVELTY_ASSESSMENT in self.agents else "NoveltyAssessment_NotAvailable",
                    "QualityAssuranceAgent"
                ],
                "superior_insights": {
                    "historical_benchmarking": benchmark_analysis,
                    "ml_prediction": hit_prediction,
                    "cache_performance": self.intelligent_caching.metrics
                },
                "enhanced_features": [
                    "genre_statistical_analysis",
                    "ml_feature_transformation", 
                    "intelligent_caching",
                    "performance_metrics"
                ]
            })
            
            # STEP 9: Cache result and update metrics
            await self.intelligent_caching.cache_analysis_result(cache_key, final_report)
            self.intelligent_caching.update_performance_metrics(final_report)
            
            logger.info(f"Enhanced comprehensive analysis {analysis_id} completed in {processing_time:.0f}ms")
            return final_report
            
        except Exception as e:
            logger.error(f"Error in enhanced comprehensive analysis {analysis_id}: {e}")
            error_result = {
                "error": str(e),
                "analysis_id": analysis_id,
                "status": "failed",
                "timestamp": datetime.utcnow().isoformat(),
                "architecture": "enhanced_parallel_agentic_workflow"
            }
            self.intelligent_caching.update_performance_metrics(error_result)
            return error_result
    
    async def _enrich_request_with_insights(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich request with additional context from superior insight services."""
        enriched = request_data.copy()
        
        try:
            # Add historical data context if available
            if hasattr(self, 'historical_data'):
                historical_context = await self.historical_data.get_relevant_context(request_data)
                enriched["historical_context"] = historical_context
            
            # Add current market timestamp
            enriched["market_timestamp"] = datetime.utcnow().isoformat()
            
            # Add production standards context
            if "audio_analysis" in enriched:
                production_assessment = self.historical_benchmarking.get_production_assessment(
                    enriched["audio_analysis"]
                )
                enriched["production_assessment"] = production_assessment
            
            return enriched
            
        except Exception as e:
            logger.warning(f"Failed to enrich request context: {e}")
            return request_data
    
    async def _execute_agent_safely(self, agent_role: AgentRole, request: Dict[str, Any], task: 'AgentTask') -> 'AgentContribution':
        """Execute an agent safely with proper error handling."""
        try:
            if agent_role not in self.agents:
                logger.error(f"Agent {agent_role} not available")
                return self._create_error_contribution(agent_role, "Agent not available")
            
            agent = self.agents[agent_role]
            result = await agent.analyze_task(request, task)
            return result
            
        except Exception as e:
            logger.error(f"{agent_role} Agent failed: {e}")
            return self._create_error_contribution(agent_role, str(e))
    
    def _create_error_contribution(self, agent_role: AgentRole, error_message: str) -> 'AgentContribution':
        """Create an error contribution when an agent fails."""
        return {
            "agent_name": str(agent_role),
            "agent_role": str(agent_role),
            "confidence_score": 0.0,
            "key_insights": [f"Agent analysis failed: {error_message}"],
            "recommendations": ["Review agent configuration and input data"],
            "processing_time_ms": 0.0,
            "status": "failed",
            "error": error_message
        }
    
    def _create_fallback_synthesis(self, synthesis_request: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback synthesis when QualityAssurance agent is not available."""
        try:
            music_analysis = synthesis_request.get("music_analysis", {})
            commercial_analysis = synthesis_request.get("commercial_analysis", {})
            novelty_analysis = synthesis_request.get("novelty_analysis", {})
            benchmark_analysis = synthesis_request.get("benchmark_analysis", {})
            hit_prediction = synthesis_request.get("hit_prediction", {})
            
            # Create basic synthesis
            fallback_synthesis = {
                "executive_summary": "Comprehensive analysis completed using enhanced parallel agentic workflow with preserved superior insights.",
                "overall_assessment": {
                    "artistic_score": music_analysis.get("confidence_score", 0.7),
                    "commercial_score": commercial_analysis.get("confidence_score", 0.7),
                    "innovation_score": novelty_analysis.get("confidence_score", 0.7),
                    "hit_prediction_score": hit_prediction.get("prediction", 0.0)
                },
                "key_insights": {
                    "musical_insights": music_analysis.get("key_insights", ["Musical analysis completed"]),
                    "commercial_insights": commercial_analysis.get("key_insights", ["Commercial analysis completed"]),
                    "benchmark_insights": benchmark_analysis.get("recommendations", ["Genre benchmarking completed"])
                },
                "synthesis_method": "fallback_synthesis",
                "quality_note": "Full synthesis requires QualityAssurance agent - individual analyses available"
            }
            
            return fallback_synthesis
            
        except Exception as e:
            logger.error(f"Error in fallback synthesis: {e}")
            return {
                "error": str(e),
                "synthesis_method": "fallback_synthesis_failed",
                "individual_analyses_available": True
            }
    
    def _initialize_agents(self):
        """Initialize all specialized agents."""
        # Music Analysis Agent
        music_profile = MusicAnalysisAgent.create_default_profile()
        self.agents[AgentRole.MUSIC_ANALYSIS] = MusicAnalysisAgent(music_profile)
        
        # Commercial Analysis Agent  
        commercial_profile = CommercialAnalysisAgent.create_default_profile()
        self.agents[AgentRole.COMMERCIAL_ANALYSIS] = CommercialAnalysisAgent(commercial_profile)
        
        # Novelty Assessment Agent (NEW - Step 2)
        novelty_profile = NoveltyAssessmentAgent.create_default_profile()
        self.agents[AgentRole.NOVELTY_ASSESSMENT] = NoveltyAssessmentAgent(novelty_profile)
        
        # Enhanced Quality Assurance Agent with LLM-powered reasoning
        self.agents[AgentRole.QUALITY_ASSURANCE] = EnhancedQualityAssuranceAgent()
        
        logger.info(f"Initialized {len(self.agents)} specialized agents")
    
    async def perform_agentic_analysis(self, request: AgenticAnalysisRequest) -> AgenticAnalysisResponse:
        """
        Perform comprehensive agentic analysis using multiple specialized agents.
        
        Args:
            request: Agentic analysis request
            
        Returns:
            AgenticAnalysisResponse: Comprehensive analysis results
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting agentic analysis: {request.request_id}")
            
            # Determine coordination strategy
            coordination_strategy = request.coordination_strategy
            self.coordination_strategies[request.request_id] = coordination_strategy
            
            # Create tasks for required agents
            tasks = self._create_agent_tasks(request)
            
            # Execute analysis based on coordination strategy
            if coordination_strategy == CoordinationStrategy.COLLABORATIVE:
                agent_contributions = await self._execute_collaborative_analysis(tasks, request)
            elif coordination_strategy == CoordinationStrategy.PARALLEL:
                agent_contributions = await self._execute_parallel_analysis(tasks, request)
            elif coordination_strategy == CoordinationStrategy.SEQUENTIAL:
                agent_contributions = await self._execute_sequential_analysis(tasks, request)
            else:
                # Default to collaborative
                agent_contributions = await self._execute_collaborative_analysis(tasks, request)
            
            # Synthesize results
            response = await self._synthesize_results(request, agent_contributions, start_time)
            
            # Update performance metrics
            self.total_analyses += 1
            self.successful_analyses += 1
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.average_processing_time = (self.average_processing_time * (self.total_analyses - 1) + processing_time) / self.total_analyses
            
            logger.info(f"Completed agentic analysis: {request.request_id} in {processing_time:.0f}ms")
            
            return response
            
        except Exception as e:
            logger.error(f"Agentic analysis failed: {e}")
            self.total_analyses += 1
            
            # Return error response
            return AgenticAnalysisResponse(
                request_id=request.request_id,
                coordination_strategy_used=request.coordination_strategy,
                agents_involved=[],
                coordination_quality=0.0,
                agent_contributions=[],
                executive_summary=f"Analysis failed: {str(e)}",
                consensus_findings=[],
                conflicting_views=[f"Analysis failed due to error: {str(e)}"],
                evidence_base=[],
                reasoning_transparency=[],
                cross_validation_results={},
                overall_confidence=0.0,
                quality_score=0.0,
                innovation_score=0.0,
                total_processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                coordination_efficiency=0.0,
                tools_usage_summary={},
                learning_events_recorded=0,
                memory_entries_created=0,
                knowledge_improvements=[],
                analysis_started_at=start_time,
                analysis_completed_at=datetime.utcnow()
            )
    
    def _create_agent_tasks(self, request: AgenticAnalysisRequest) -> Dict[AgentRole, AgentTask]:
        """Create tasks for each required agent."""
        tasks = {}
        
        for role in request.required_agent_roles:
            if role in self.agents:
                task = AgentTask(
                    agent_id=self.agents[role].profile.agent_id,
                    task_type=TaskType.ANALYSIS,
                    description=f"Perform {role.value} analysis on provided data",
                    input_data=request.input_data,
                    requirements=request.focus_areas,
                    constraints=request.constraints,
                    priority=8,
                    max_execution_time=request.max_execution_time
                )
                tasks[role] = task
                self.active_tasks[task.task_id] = task
        
        return tasks
    
    async def _execute_collaborative_analysis(self, tasks: Dict[AgentRole, AgentTask], 
                                            request: AgenticAnalysisRequest) -> List[AgentContribution]:
        """Execute analysis with collaborative coordination."""
        contributions = []
        
        # Execute all agents in parallel but allow for communication
        agent_futures = []
        
        for role, task in tasks.items():
            if role in self.agents:
                future = self.agents[role].execute_task(task, request.input_data)
                agent_futures.append((role, future))
        
        # Wait for all agents to complete
        for role, future in agent_futures:
            try:
                contribution = await future
                contributions.append(contribution)
                logger.info(f"Agent {role.value} completed collaborative analysis")
            except Exception as e:
                logger.error(f"Agent {role.value} failed in collaborative analysis: {e}")
        
        return contributions
    
    async def _execute_parallel_analysis(self, tasks: Dict[AgentRole, AgentTask],
                                       request: AgenticAnalysisRequest) -> List[AgentContribution]:
        """Execute analysis with parallel coordination."""
        contributions = []
        
        # Execute all agents simultaneously
        agent_futures = []
        
        for role, task in tasks.items():
            if role in self.agents:
                future = asyncio.create_task(self.agents[role].execute_task(task, request.input_data))
                agent_futures.append((role, future))
        
        # Gather results
        results = await asyncio.gather(*[future for _, future in agent_futures], return_exceptions=True)
        
        for i, ((role, _), result) in enumerate(zip(agent_futures, results)):
            if isinstance(result, Exception):
                logger.error(f"Agent {role.value} failed in parallel analysis: {result}")
            else:
                contributions.append(result)
                logger.info(f"Agent {role.value} completed parallel analysis")
        
        return contributions
    
    async def _execute_sequential_analysis(self, tasks: Dict[AgentRole, AgentTask],
                                         request: AgenticAnalysisRequest) -> List[AgentContribution]:
        """Execute analysis with sequential coordination."""
        contributions = []
        
        # Define execution order (music -> commercial -> QA)
        execution_order = [
            AgentRole.MUSIC_ANALYSIS,
            AgentRole.COMMERCIAL_ANALYSIS,
            AgentRole.QUALITY_ASSURANCE
        ]
        
        # Execute agents in sequence
        for role in execution_order:
            if role in tasks and role in self.agents:
                try:
                    task = tasks[role]
                    contribution = await self.agents[role].execute_task(task, request.input_data)
                    contributions.append(contribution)
                    logger.info(f"Agent {role.value} completed sequential analysis")
                    
                    # Pass results to next agent (simplified)
                    if len(contributions) > 1:
                        # Update input data with previous results
                        request.input_data["previous_analysis"] = contribution.dict()
                        
                except Exception as e:
                    logger.error(f"Agent {role.value} failed in sequential analysis: {e}")
        
        return contributions
    
    async def _synthesize_results(self, request: AgenticAnalysisRequest,
                                agent_contributions: List[AgentContribution],
                                start_time: datetime) -> AgenticAnalysisResponse:
        """Synthesize agent contributions into comprehensive response."""
        
        # Calculate coordination metrics
        coordination_quality = self._calculate_coordination_quality(agent_contributions)
        coordination_efficiency = self._calculate_coordination_efficiency(agent_contributions, start_time)
        
        # Synthesize findings and insights
        all_findings = []
        all_insights = []
        all_recommendations = []
        all_evidence = []
        
        for contribution in agent_contributions:
            all_findings.extend(contribution.findings)
            all_insights.extend(contribution.insights)
            all_recommendations.extend(contribution.recommendations)
            all_evidence.extend(contribution.evidence)
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(agent_contributions)
        
        # Identify consensus and conflicts
        consensus_findings = self._identify_consensus_findings(agent_contributions)
        conflicting_views = self._identify_conflicting_views(agent_contributions)
        
        # Cross-validation
        cross_validation_results = self._perform_cross_validation(agent_contributions)
        
        # Calculate overall metrics
        overall_confidence = sum(c.confidence_level for c in agent_contributions) / len(agent_contributions)
        quality_score = coordination_quality * 0.4 + overall_confidence * 0.6
        innovation_score = self._calculate_innovation_score(agent_contributions)
        
        # Tools usage summary
        tools_usage_summary = self._summarize_tools_usage(agent_contributions)
        
        return AgenticAnalysisResponse(
            request_id=request.request_id,
            coordination_strategy_used=request.coordination_strategy,
            agents_involved=[c.agent_id for c in agent_contributions],
            coordination_quality=coordination_quality,
            agent_contributions=agent_contributions,
            executive_summary=executive_summary,
            consensus_findings=consensus_findings,
            conflicting_views=conflicting_views,
            evidence_base=all_evidence,
            reasoning_transparency=self._build_transparency_report(agent_contributions),
            cross_validation_results=cross_validation_results,
            overall_confidence=overall_confidence,
            quality_score=quality_score,
            innovation_score=innovation_score,
            total_processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            coordination_efficiency=coordination_efficiency,
            tools_usage_summary=tools_usage_summary,
            learning_events_recorded=0,  # Simplified for now
            memory_entries_created=0,    # Simplified for now
            knowledge_improvements=[],   # Simplified for now
            analysis_started_at=start_time,
            analysis_completed_at=datetime.utcnow()
        )
    
    def _calculate_coordination_quality(self, contributions: List[AgentContribution]) -> float:
        """Calculate quality of agent coordination."""
        if not contributions:
            return 0.0
        
        # Factors: completion rate, confidence levels, processing time consistency
        completion_rate = len(contributions) / len(self.agents)
        avg_confidence = sum(c.confidence_level for c in contributions) / len(contributions)
        
        # Processing time consistency (lower variance = better coordination)
        processing_times = [c.processing_time_ms for c in contributions]
        avg_time = sum(processing_times) / len(processing_times)
        time_variance = sum((t - avg_time) ** 2 for t in processing_times) / len(processing_times)
        time_consistency = max(0, 1 - (time_variance / (avg_time ** 2)))
        
        return (completion_rate * 0.4 + avg_confidence * 0.4 + time_consistency * 0.2)
    
    def _calculate_coordination_efficiency(self, contributions: List[AgentContribution],
                                         start_time: datetime) -> float:
        """Calculate coordination efficiency."""
        if not contributions:
            return 0.0
        
        total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        agent_times = [c.processing_time_ms for c in contributions]
        
        # Efficiency = (sum of individual times) / (total elapsed time)
        # Higher values indicate better parallelization
        efficiency = sum(agent_times) / max(total_time, 1)
        
        return min(efficiency, 1.0)  # Cap at 1.0
    
    def _generate_executive_summary(self, contributions: List[AgentContribution]) -> str:
        """Generate executive summary from agent contributions."""
        if not contributions:
            return "No analysis results available."
        
        # Extract key insights from each agent
        music_insights = []
        commercial_insights = []
        
        for contribution in contributions:
            if contribution.agent_role == AgentRole.MUSIC_ANALYSIS:
                music_insights = contribution.insights[:2]  # Top 2 insights
            elif contribution.agent_role == AgentRole.COMMERCIAL_ANALYSIS:
                commercial_insights = contribution.insights[:2]  # Top 2 insights
        
        # Combine into executive summary
        summary_parts = ["Comprehensive analysis reveals:"]
        
        if music_insights:
            summary_parts.append(f"Musical Analysis: {' '.join(music_insights)}")
        
        if commercial_insights:
            summary_parts.append(f"Commercial Analysis: {' '.join(commercial_insights)}")
        
        # Add overall confidence
        avg_confidence = sum(c.confidence_level for c in contributions) / len(contributions)
        summary_parts.append(f"Analysis confidence: {avg_confidence:.1%}")
        
        return " ".join(summary_parts)
    
    def _identify_consensus_findings(self, contributions: List[AgentContribution]) -> List[str]:
        """Identify findings that multiple agents agree on."""
        # Simplified consensus detection based on keyword matching
        consensus = []
        
        all_findings = []
        for contribution in contributions:
            all_findings.extend(contribution.findings)
        
        # Look for common themes/keywords
        keywords = ["high", "strong", "commercial", "energy", "potential", "quality"]
        
        for keyword in keywords:
            matching_findings = [f for f in all_findings if keyword.lower() in f.lower()]
            if len(matching_findings) >= 2:  # At least 2 agents mention it
                consensus.append(f"Multiple agents identified {keyword} characteristics")
        
        return consensus[:3]  # Return top 3 consensus findings
    
    def _identify_conflicting_views(self, contributions: List[AgentContribution]) -> List[str]:
        """Identify areas where agents disagree."""
        # Simplified conflict detection
        conflicts = []
        
        # Check confidence levels
        confidences = [c.confidence_level for c in contributions]
        if max(confidences) - min(confidences) > 0.3:
            conflicts.append("Agents show varying confidence levels in their analysis")
        
        # Check recommendations
        all_recommendations = []
        for contribution in contributions:
            all_recommendations.extend(contribution.recommendations)
        
        # Look for contradictory terms
        contradictory_pairs = [("independent", "major"), ("niche", "mainstream"), ("low", "high")]
        
        for term1, term2 in contradictory_pairs:
            has_term1 = any(term1.lower() in rec.lower() for rec in all_recommendations)
            has_term2 = any(term2.lower() in rec.lower() for rec in all_recommendations)
            
            if has_term1 and has_term2:
                conflicts.append(f"Conflicting recommendations regarding {term1} vs {term2} approach")
        
        return conflicts
    
    def _perform_cross_validation(self, contributions: List[AgentContribution]) -> Dict[str, float]:
        """Perform cross-validation between agents."""
        validation_results = {}
        
        for contribution in contributions:
            agent_role = contribution.agent_role.value
            
            # Validate against confidence and evidence
            evidence_quality = len(contribution.evidence) / 10.0  # Normalize to 0-1
            confidence_validation = contribution.confidence_level
            
            validation_score = (evidence_quality + confidence_validation) / 2
            validation_results[agent_role] = min(validation_score, 1.0)
        
        return validation_results
    
    def _calculate_innovation_score(self, contributions: List[AgentContribution]) -> float:
        """Calculate innovation score based on unique insights."""
        if not contributions:
            return 0.0
        
        # Count unique insights and novel recommendations
        all_insights = []
        for contribution in contributions:
            all_insights.extend(contribution.insights)
        
        # Simple innovation score based on diversity and uniqueness
        unique_insights = len(set(all_insights))
        total_insights = len(all_insights)
        
        diversity_score = unique_insights / max(total_insights, 1)
        
        # Boost for creative recommendations
        creative_keywords = ["innovative", "unique", "creative", "novel", "experimental"]
        creativity_boost = 0.0
        
        for contribution in contributions:
            for rec in contribution.recommendations:
                if any(keyword in rec.lower() for keyword in creative_keywords):
                    creativity_boost += 0.1
        
        return min(diversity_score + creativity_boost, 1.0)
    
    def _summarize_tools_usage(self, contributions: List[AgentContribution]) -> Dict[str, int]:
        """Summarize tools usage across all agents."""
        tools_summary = {}
        
        for contribution in contributions:
            for tool in contribution.tools_used:
                tools_summary[tool] = tools_summary.get(tool, 0) + 1
        
        return tools_summary
    
    def _build_transparency_report(self, contributions: List[AgentContribution]) -> List[str]:
        """Build transparency report showing reasoning process."""
        transparency = []
        
        transparency.append("Multi-agent analysis process:")
        
        for i, contribution in enumerate(contributions, 1):
            agent_name = contribution.agent_role.value.replace("_", " ").title()
            transparency.append(f"{i}. {agent_name}: {contribution.methodology}")
            
            # Add key reasoning steps  
            if contribution.reasoning_chain:
                transparency.append(f"   Key steps: {'; '.join(contribution.reasoning_chain[:2])}")
        
        return transparency
    
    def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics."""
        success_rate = self.successful_analyses / max(self.total_analyses, 1)
        
        return {
            "total_analyses": self.total_analyses,
            "successful_analyses": self.successful_analyses,
            "success_rate": success_rate,
            "average_processing_time_ms": self.average_processing_time,
            "active_agents": len(self.agents),
            "active_tasks": len(self.active_tasks),
            "agent_performance": {
                role.value: agent.get_performance_metrics() 
                for role, agent in self.agents.items()
            }
        }

