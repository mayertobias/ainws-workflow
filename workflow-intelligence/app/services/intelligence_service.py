"""
Intelligence Service for workflow-intelligence microservice

Handles AI insights generation with multi-provider support, caching, and prompt management.
"""

import asyncio
import time
import uuid
import hashlib
import json
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging

# Import feature translator as proper package
from hss_feature_translator import FeatureTranslator

import redis
from ..config.settings import settings
from ..models.intelligence import (
    InsightGenerationRequest, InsightGenerationResponse, 
    AIInsightResult, AnalysisType, InsightStatus,
    MusicalMeaningInsight, HitComparisonInsight, NoveltyAssessmentInsight,
    ProductionFeedback, StrategicInsights, InsightMetrics
)
from .llm_providers import LLMProviderFactory, BaseLLMProvider
from .prompt_manager import PromptManager
from .agents import IntelligenceAgentFactory
from .historical_data_integrator import HistoricalDataIntegrator
from .structured_output_parser import StructuredOutputParser

logger = logging.getLogger(__name__)

class IntelligenceService:
    """
    Handles AI-powered insights generation for Hit Song Science.
    
    This service provides:
    - Multi-provider AI insight generation
    - Prompt template management
    - Response caching
    - Cost tracking and rate limiting
    - Historical data integration for sophisticated analysis
    """
    
    def __init__(self):
        """Initialize the Intelligence service."""
        # Initialize Redis for caching
        try:
            self.redis_client = redis.from_url(settings.REDIS_URL)
            self.redis_client.ping()
            logger.info("Redis connection established for AI caching")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. AI responses will not be cached.")
            self.redis_client = None
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager()
        
        # Initialize agent factory
        self.agent_factory = IntelligenceAgentFactory()
        
        # Initialize historical data integrator
        self.historical_data = HistoricalDataIntegrator()
        
        # Initialize structured output parser
        self.structured_parser = StructuredOutputParser()
        
        # Initialize feature translator
        try:
            self.feature_translator = FeatureTranslator()
            logger.info("FeatureTranslator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FeatureTranslator: {e}")
            self.feature_translator = None
        
        # Service metrics
        self.metrics = {
            'total_insights_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0.0,
            'provider_usage': {},
            'analysis_type_counts': {},
            'errors': 0
        }
        
        logger.info("Intelligence Service initialized successfully with historical data integration")
    
    async def generate_insights(self, request: InsightGenerationRequest) -> InsightGenerationResponse:
        """
        Generate AI insights based on the request.
        
        Args:
            request: Insight generation request
            
        Returns:
            Generated insights response
        """
        start_time = time.time()
        insight_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Generating insights for request {insight_id}")
            
            # ENHANCED: Get actual hit potential from ML prediction service
            hit_potential_data = await self._get_hit_potential(request)
            
            # ENHANCED: Generate historical data benchmarks
            historical_analysis = {}
            if request.audio_features:
                historical_analysis = self.historical_data.analyze_song_against_benchmarks(
                    request.audio_features.dict(),
                    request.song_metadata.genre if request.song_metadata and request.song_metadata.genre else "pop"
                )
            
            # Check cache first
            cached_result = None
            if request.use_cache and self.redis_client:
                cache_key = self._generate_cache_key(request)
                cached_result = await self._get_cached_insights(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for insights {insight_id}")
                    # Convert cached dict to proper response format
                    cached_insights = AIInsightResult(**cached_result) if isinstance(cached_result, dict) else cached_result
                    cached_insights.hit_potential_score = hit_potential_data.get('prediction', 0.0) or 0.0
                    cached_insights.hit_confidence = hit_potential_data.get('confidence', 0.0)
                    cached_insights.model_used = hit_potential_data.get('model_used', 'cached')
                    
                    return InsightGenerationResponse(
                        status=InsightStatus.COMPLETED,
                        insight_id=insight_id,
                        insights=cached_insights,
                        cached=True,
                        hit_potential_score=hit_potential_data.get('prediction', 0.0) or 0.0,
                        hit_confidence=hit_potential_data.get('confidence', 0.0),
                        model_used=hit_potential_data.get('model_used', 'cached')
                    )
            
            self.metrics['cache_misses'] += 1
            
            # Get AI agent
            agent = await self._get_agent(request.agent_type, request.ai_provider)
            if not agent:
                raise RuntimeError("No AI agent available")
            
            # Generate insights for each analysis type
            insights_result = AIInsightResult()
            insights_result.analysis_types_completed = []
            insights_result.agent_used = request.agent_type.value
            insights_result.provider_used = agent.provider_name if hasattr(agent, 'provider_name') else 'unknown'
            
            # ADDED: Include hit potential in insights
            insights_result.hit_potential_score = hit_potential_data.get('prediction', 0.0) or 0.0
            insights_result.hit_confidence = hit_potential_data.get('confidence', 0.0)
            insights_result.model_used = hit_potential_data.get('model_used', 'unknown')
            
            raw_outputs = {} if request.include_raw_output else None
            
            # Process each analysis type
            for analysis_type in request.analysis_types:
                try:
                    insight = await self._generate_single_analysis(
                        agent, analysis_type, request, raw_outputs
                    )
                    
                    if insight:
                        setattr(insights_result, analysis_type.value, insight)
                        insights_result.analysis_types_completed.append(analysis_type)
                        
                        # Track analysis type usage
                        type_key = analysis_type.value
                        self.metrics['analysis_type_counts'][type_key] = \
                            self.metrics['analysis_type_counts'].get(type_key, 0) + 1
                        
                except Exception as e:
                    logger.error(f"Error generating {analysis_type} analysis: {e}")
                    continue
            
            # Generate executive summary if multiple analyses
            if len(insights_result.analysis_types_completed) > 1:
                insights_result.executive_summary = await self._generate_executive_summary(
                    request, insights_result, historical_analysis
                )
            
            # Generate key recommendations
            insights_result.key_recommendations = await self._generate_recommendations(
                agent, insights_result, request
            )
            
            # Set metadata
            processing_time = (time.time() - start_time) * 1000
            insights_result.processing_time_ms = processing_time
            insights_result.confidence_score = self._calculate_confidence_score(insights_result)
            
            if raw_outputs:
                insights_result.raw_outputs = raw_outputs
            
            # Cache result
            if request.use_cache and self.redis_client:
                cache_key = self._generate_cache_key(request)
                await self._cache_insights(cache_key, insights_result.dict())
            
            # Update metrics
            self.metrics['total_insights_generated'] += 1
            self.metrics['total_processing_time'] += processing_time
            
            # Track provider usage
            provider_name = insights_result.provider_used
            self.metrics['provider_usage'][provider_name] = \
                self.metrics['provider_usage'].get(provider_name, 0) + 1
            
            return InsightGenerationResponse(
                status=InsightStatus.COMPLETED,
                insight_id=insight_id,
                insights=insights_result,
                cached=False,
                hit_potential_score=hit_potential_data.get('prediction', 0.0),
                hit_confidence=hit_potential_data.get('confidence', 0.0),
                model_used=hit_potential_data.get('model_used', 'unknown')
            )
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error generating insights: {e}", exc_info=True)
            
            return InsightGenerationResponse(
                status=InsightStatus.FAILED,
                insight_id=insight_id,
                error_message=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
                cached=False
            )
    
    async def _get_agent(self, agent_type, provider_preference):
        """Get an AI agent instance."""
        try:
            return await self.agent_factory.create_agent(
                agent_type=agent_type.value,
                provider_preference=provider_preference.value
            )
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            return None
    
    async def _get_hit_potential(self, request: InsightGenerationRequest) -> Dict[str, Any]:
        """Get hit potential from ML prediction service."""
        try:
            # Extract audio features from request data
            audio_features = request.audio_features.dict() if request.audio_features else {}
            content_features = request.lyrics_analysis.dict() if request.lyrics_analysis else {}
            
            # ENHANCED: Log feature count for debugging
            logger.info(f"Intelligence service received {len(audio_features)} audio features: {list(audio_features.keys())[:5]}...")
            if content_features:
                logger.info(f"Intelligence service received {len(content_features)} content features: {list(content_features.keys())}")
            
            # If we have features, call ML prediction service
            if audio_features or content_features:
                import httpx
                
                # Use FeatureTranslator for proper feature transformation
                transformed_features = {}
                
                if self.feature_translator:
                    try:
                        # Transform audio features using FeatureTranslator
                        if audio_features:
                            # audio_features is already a flat dict from request.audio_features.dict()
                            # We need to directly map to the expected audio_* feature names
                            audio_mapped = {}
                            for key, value in audio_features.items():
                                # Map to audio_* format expected by ML models
                                if not key.startswith('audio_'):
                                    audio_mapped[f'audio_{key}'] = value
                                else:
                                    audio_mapped[key] = value
                            
                            transformed_features.update(audio_mapped)
                            logger.info(f"Mapped {len(audio_mapped)} audio features to ML format")
                        
                        # Transform content features using FeatureTranslator
                        if content_features:
                            content_translated = self.feature_translator.content_producer_to_consumer(content_features)
                            transformed_features.update(content_translated)
                            logger.info(f"FeatureTranslator processed {len(content_translated)} content features")
                            
                    except Exception as e:
                        logger.warning(f"FeatureTranslator failed: {e}. Using fallback approach.")
                        # Fallback to simple feature mapping if FeatureTranslator fails
                        for feature, value in audio_features.items():
                            if value is not None:
                                try:
                                    feature_name = f"audio_{feature}" if not feature.startswith("audio_") else feature
                                    transformed_features[feature_name] = float(value)
                                except (ValueError, TypeError):
                                    continue
                else:
                    logger.warning("FeatureTranslator not available. Using fallback approach.")
                    # Fallback to simple feature mapping
                    for feature, value in audio_features.items():
                        if value is not None:
                            try:
                                feature_name = f"audio_{feature}" if not feature.startswith("audio_") else feature
                                transformed_features[feature_name] = float(value)
                            except (ValueError, TypeError):
                                continue
                
                # Use the correct ML prediction service URL with proper formatting
                ml_prediction_url = (settings.WORKFLOW_ML_PREDICTION_URL or "http://workflow-ml-prediction:8004").rstrip('/')
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    try:
                        logger.info(f"Calling ML prediction service with {len(transformed_features)} features")
                        response = await client.post(
                            f"{ml_prediction_url}/predict/smart/single",
                            json={
                                "song_features": transformed_features,
                                "explain_prediction": True
                            }
                        )
                        
                        if response.status_code == 200:
                            prediction_data = response.json()
                            logger.info(f"ML prediction successful: {prediction_data.get('prediction', 'N/A')}")
                            
                            # CRITICAL FIX: Ensure prediction is never None
                            prediction_value = prediction_data.get('prediction', 0.0)
                            if prediction_value is None:
                                prediction_value = 0.0
                            
                            confidence_value = prediction_data.get('confidence', 0.0)
                            if confidence_value is None:
                                confidence_value = 0.0
                            
                            return {
                                'prediction': prediction_value,
                                'confidence': confidence_value,
                                'model_used': prediction_data.get('model_used', 'unknown')
                            }
                        else:
                            logger.warning(f"ML prediction service returned {response.status_code}: {response.text}")
                            
                    except Exception as e:
                        logger.error(f"Error calling ML prediction service: {e}")
                        
            # Return default values if no features or service call failed
            return {
                'prediction': 0.0,
                'confidence': 0.0,
                'model_used': 'unavailable'
            }
            
        except Exception as e:
            logger.error(f"Error getting hit potential: {e}")
            return {
                'prediction': 0.0,
                'confidence': 0.0,
                'model_used': 'error'
            }
    
    async def _generate_single_analysis(
        self, 
        agent,
        analysis_type: AnalysisType,
        request: InsightGenerationRequest,
        enhanced_context: Dict[str, Any]
    ) -> Optional[Any]:
        """Generate analysis for a single type using the provided agent with structured output parsing."""
        try:
            # Get base prompt for this analysis type
            base_prompt = await self.prompt_manager.get_formatted_prompt(
                analysis_type, request, enhanced_context
            )
            
            if not base_prompt:
                logger.warning(f"No prompt found for analysis type: {analysis_type}")
                return None
            
            # LAYER 1: Create enhanced prompt with strict JSON format instructions
            enhanced_prompt = self.structured_parser.create_enhanced_prompt(
                base_prompt, analysis_type
            )
            
            # Generate insight using the agent
            response = await agent.generate(enhanced_prompt)
            
            # LAYER 1: Use structured output parser for validation
            parsed_result = self.structured_parser.parse_structured_output(
                response, analysis_type
            )
            
            if parsed_result:
                # Convert Pydantic model to dict for JSON serialization
                return parsed_result.model_dump()
            else:
                logger.error(f"Structured parsing failed for {analysis_type}")
                # Last resort: return error structure
                return {
                    "error": "Structured parsing failed",
                    "analysis_type": analysis_type.value,
                    "raw_response": response[:200] + "..." if len(response) > 200 else response
                }
                
        except Exception as e:
            logger.error(f"Error generating single analysis for {analysis_type}: {e}")
            return {
                "error": str(e),
                "analysis_type": analysis_type.value,
                "status": "failed"
            }
    
    async def _parse_analysis_response(self, analysis_type: AnalysisType, response: str) -> Optional[Any]:
        """Parse AI response into structured insight object."""
        try:
            # Try to extract JSON if present
            cleaned_response = self._clean_ai_response(response)
            
            if analysis_type == AnalysisType.MUSICAL_MEANING:
                return self._parse_musical_meaning(cleaned_response)
            elif analysis_type == AnalysisType.HIT_COMPARISON:
                return self._parse_hit_comparison(cleaned_response)
            elif analysis_type == AnalysisType.NOVELTY_ASSESSMENT:
                return self._parse_novelty_assessment(cleaned_response)
            elif analysis_type == AnalysisType.PRODUCTION_FEEDBACK:
                return self._parse_production_feedback(cleaned_response)
            elif analysis_type == AnalysisType.STRATEGIC_INSIGHTS:
                return self._parse_strategic_insights(cleaned_response)
            else:
                logger.warning(f"Unknown analysis type: {analysis_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error parsing {analysis_type} response: {e}")
            return None
    
    def _clean_ai_response(self, response: str) -> str:
        """Clean and normalize AI response."""
        import re
        import json
        
        # Remove code blocks
        response = re.sub(r'```(?:json)?\n(.*?)\n```', r'\1', response, flags=re.DOTALL)
        
        # Try multiple JSON extraction patterns
        patterns = [
            r'\{.*\}',  # Any object
            r'\[.*\]',  # Any array
        ]
        
        for pattern in patterns:
            json_match = re.search(pattern, response, re.DOTALL)
            if json_match:
                try:
                    potential_json = json_match.group()
                    # Validate it's actually JSON
                    parsed = json.loads(potential_json)
                    return json.dumps(parsed, indent=2)
                except json.JSONDecodeError:
                    continue
        
        return response.strip()
    
    def _parse_musical_meaning(self, response: str) -> MusicalMeaningInsight:
        """Parse musical meaning analysis response."""
        try:
            data = json.loads(response)
            return MusicalMeaningInsight(**data)
        except json.JSONDecodeError:
            # Fallback to text parsing
            return MusicalMeaningInsight(
                emotional_core=self._extract_section(response, "emotional core"),
                musical_narrative=self._extract_section(response, "narrative"),
                cultural_context=self._extract_section(response, "cultural"),
                listener_impact=self._extract_section(response, "impact"),
                key_strengths=self._extract_list(response, "strengths"),
                improvement_areas=self._extract_list(response, "improvement")
            )
    
    def _parse_hit_comparison(self, response: str) -> HitComparisonInsight:
        """Parse hit comparison analysis response."""
        try:
            data = json.loads(response)
            return HitComparisonInsight(**data)
        except json.JSONDecodeError:
            return HitComparisonInsight(
                market_positioning=self._extract_section(response, "positioning"),
                target_audience=self._extract_section(response, "audience"),
                commercial_strengths=self._extract_list(response, "strengths"),
                commercial_weaknesses=self._extract_list(response, "weaknesses")
            )
    
    def _parse_novelty_assessment(self, response: str) -> NoveltyAssessmentInsight:
        """Parse novelty assessment response."""
        try:
            data = json.loads(response)
            return NoveltyAssessmentInsight(**data)
        except json.JSONDecodeError:
            return NoveltyAssessmentInsight(
                unique_elements=self._extract_list(response, "unique"),
                trend_alignment=self._extract_section(response, "trend"),
                risk_assessment=self._extract_section(response, "risk"),
                market_readiness=self._extract_section(response, "readiness")
            )
    
    def _parse_production_feedback(self, response: str) -> ProductionFeedback:
        """Parse production feedback response."""
        try:
            data = json.loads(response)
            return ProductionFeedback(**data)
        except json.JSONDecodeError:
            return ProductionFeedback(
                overall_quality=self._extract_section(response, "quality"),
                technical_strengths=self._extract_list(response, "strengths"),
                technical_issues=self._extract_list(response, "issues"),
                sonic_character=self._extract_section(response, "character")
            )
    
    def _parse_strategic_insights(self, response: str) -> StrategicInsights:
        """Parse strategic insights response."""
        try:
            data = json.loads(response)
            return StrategicInsights(**data)
        except json.JSONDecodeError:
            return StrategicInsights(
                market_opportunity=self._extract_section(response, "opportunity"),
                competitive_advantage=self._extract_section(response, "advantage"),
                release_strategy=self._extract_section(response, "strategy"),
                promotional_angles=self._extract_list(response, "promotional")
            )
    
    def _extract_section(self, text: str, keyword: str) -> Optional[str]:
        """Extract a section from text based on keyword."""
        import re
        patterns = [
            rf"{keyword}:?\s*([^\n]+)",
            rf"{keyword.title()}:?\s*([^\n]+)",
            rf"{keyword.upper()}:?\s*([^\n]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_list(self, text: str, keyword: str) -> List[str]:
        """Extract a list from text based on keyword."""
        import re
        
        # Look for bullet points or numbered lists
        section = self._extract_section(text, keyword)
        if not section:
            return []
        
        # Split by common list indicators
        items = re.split(r'[•\-\*\d+\.\)]\s*', section)
        return [item.strip() for item in items if item.strip()]
    
    async def _generate_executive_summary(
        self, 
        request: InsightGenerationRequest,
        result: AIInsightResult,
        enhanced_context: Dict[str, Any]
    ) -> str:
        """Generate executive summary of all insights."""
        try:
            # Get agent for summary generation
            agent = await self._get_agent(request.agent_type, request.ai_provider)
            if not agent:
                return "Executive summary unavailable - no AI agent available"
            
            # ENHANCED: Ensure hit_potential_score is never None
            hit_potential_score = result.hit_potential_score if result.hit_potential_score is not None else 0.0
            hit_confidence = result.hit_confidence if result.hit_confidence is not None else 0.0
            
            # Prepare summary context
            summary_context = {
                "hit_potential_score": hit_potential_score,
                "hit_confidence": hit_confidence,
                "song_title": request.song_metadata.title if request.song_metadata and request.song_metadata.title else "Unknown",
                "artist": request.song_metadata.artist if request.song_metadata and request.song_metadata.artist else "Unknown",
                "genre": request.song_metadata.genre if request.song_metadata and request.song_metadata.genre else "Unknown",
                "insights_generated": len(request.analysis_types),
                "commercial_viability": self._assess_commercial_viability(hit_potential_score),
                "key_findings": self._extract_key_findings(result),
                "critical_actions": self._extract_critical_actions(result)
            }
            
            executive_prompt = f"""
You are a senior A&R executive providing a comprehensive executive summary of this song analysis.

SONG DETAILS:
- Title: {summary_context['song_title']}
- Artist: {summary_context['artist']}
- Genre: {summary_context['genre']}
- Hit Potential Score: {summary_context['hit_potential_score']:.1%}
- Confidence: {summary_context['hit_confidence']:.1%}

ANALYSIS OVERVIEW:
- {summary_context['insights_generated']} analysis types completed
- Commercial Viability: {summary_context['commercial_viability']}

KEY FINDINGS:
{chr(10).join(f"• {finding}" for finding in summary_context['key_findings'])}

CRITICAL ACTIONS:
{chr(10).join(f"• {action}" for action in summary_context['critical_actions'])}

Provide a concise, actionable executive summary (150-200 words) that:
1. Summarizes the commercial potential assessment
2. Highlights the most critical insights
3. Provides clear next steps for the artist/label
4. Includes specific data points and recommendations

Focus on actionable intelligence that can drive business decisions.
"""
            
            summary = await agent.generate(executive_prompt)
            return summary
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return f"Executive summary: This song shows {result.hit_potential_score:.1%} hit potential with {result.hit_confidence:.1%} confidence. Analysis completed successfully."

    def _assess_commercial_viability(self, hit_score: Optional[float]) -> str:
        """Assess commercial viability based on hit score."""
        # CRITICAL FIX: Handle None values that can come from ML prediction failures
        if hit_score is None:
            return "UNKNOWN - Unable to assess commercial viability (prediction data unavailable)"
        
        if hit_score >= 0.7:
            return "HIGH - Strong commercial potential"
        elif hit_score >= 0.4:
            return "MODERATE - Good commercial appeal"
        elif hit_score >= 0.15:
            return "NICHE - Limited commercial appeal"
        else:
            return "EXPERIMENTAL - Artistic/experimental appeal"

    def _extract_key_findings(self, result: AIInsightResult) -> List[str]:
        """Extract key findings from analysis results."""
        findings = []
        
        if result.hit_comparison:
            findings.append(f"Hit alignment score: {getattr(result.hit_comparison, 'hit_alignment_score', 'N/A')}")
        
        if result.novelty_assessment:
            findings.append(f"Innovation score: {getattr(result.novelty_assessment, 'innovation_score', 'N/A')}")
        
        if result.production_feedback:
            findings.append(f"Production quality: {getattr(result.production_feedback, 'overall_quality', 'N/A')}")
        
        if not findings:
            # ENHANCED: Ensure hit_potential_score is never None
            hit_score = result.hit_potential_score if result.hit_potential_score is not None else 0.0
            findings.append(f"Hit potential score: {hit_score:.1%}")
        
        return findings

    def _extract_critical_actions(self, result: AIInsightResult) -> List[str]:
        """Extract critical actions from analysis results."""
        actions = []
        
        if result.strategic_insights:
            market_op = getattr(result.strategic_insights, 'market_opportunity', '')
            if market_op:
                actions.append(f"Market opportunity: {market_op[:100]}...")
        
        if result.production_feedback:
            tech_issues = getattr(result.production_feedback, 'technical_issues', [])
            if tech_issues and len(tech_issues) > 0:
                actions.append(f"Production priority: {tech_issues[0] if isinstance(tech_issues, list) else tech_issues}")
        
        if not actions:
            actions.append("Review detailed analysis for specific recommendations")
        
        return actions
    
    async def _generate_recommendations(
        self, 
        agent, 
        insights: AIInsightResult, 
        request: InsightGenerationRequest
    ) -> List[str]:
        """Generate key recommendations."""
        try:
            prompt = await self.prompt_manager.get_recommendations_prompt(insights, request)
            if prompt:
                response = await agent.generate(prompt, max_tokens=300)
                return self._extract_list(response, "recommendations")
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        return []
    
    def _calculate_confidence_score(self, insights: AIInsightResult) -> float:
        """Calculate overall confidence score for insights."""
        completed_count = len(insights.analysis_types_completed)
        if completed_count == 0:
            return 0.0
        
        # Base confidence on number of completed analyses
        base_score = min(completed_count / len(AnalysisType), 1.0)
        
        # Adjust based on quality indicators
        if insights.executive_summary:
            base_score += 0.1
        if insights.key_recommendations:
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    def _generate_cache_key(self, request: InsightGenerationRequest) -> str:
        """Generate cache key for request."""
        # Create a hash of the request data
        cache_data = {
            'audio_features': request.audio_features.dict() if request.audio_features else None,
            'lyrics_analysis': request.lyrics_analysis.dict() if request.lyrics_analysis else None,
            'hit_analysis': request.hit_analysis.dict() if request.hit_analysis else None,
            'song_metadata': request.song_metadata.dict() if request.song_metadata else None,
            'analysis_types': [t.value for t in request.analysis_types],
            'agent_type': request.agent_type.value,
            'ai_provider': request.ai_provider.value
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    async def _get_cached_insights(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached insights result."""
        try:
            if self.redis_client:
                cached_data = self.redis_client.get(f"insights:{cache_key}")
                if cached_data:
                    return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Error getting cached insights: {e}")
        return None
    
    async def _cache_insights(self, cache_key: str, insights: Dict[str, Any]) -> None:
        """Cache insights result."""
        try:
            if self.redis_client:
                self.redis_client.setex(
                    f"insights:{cache_key}",
                    settings.CACHE_TTL_SECONDS,
                    json.dumps(insights, default=str)
                )
        except Exception as e:
            logger.warning(f"Error caching insights: {e}")
    
    async def get_metrics(self) -> InsightMetrics:
        """Get service metrics."""
        cache_hit_rate = 0.0
        if self.metrics['total_insights_generated'] > 0:
            cache_hit_rate = self.metrics['cache_hits'] / self.metrics['total_insights_generated']
        
        avg_processing_time = 0.0
        if self.metrics['total_insights_generated'] > 0:
            avg_processing_time = self.metrics['total_processing_time'] / self.metrics['total_insights_generated']
        
        error_rate = 0.0
        if self.metrics['total_insights_generated'] > 0:
            error_rate = self.metrics['errors'] / self.metrics['total_insights_generated']
        
        return InsightMetrics(
            total_insights_generated=self.metrics['total_insights_generated'],
            cache_hit_rate=cache_hit_rate,
            average_processing_time_ms=avg_processing_time,
            provider_usage=self.metrics['provider_usage'],
            analysis_type_distribution=self.metrics['analysis_type_counts'],
            error_rate=error_rate
        )
    
    async def clear_cache(self) -> Dict[str, Any]:
        """Clear all cached insights."""
        try:
            if self.redis_client:
                keys = self.redis_client.keys("insights:*")
                if keys:
                    deleted = self.redis_client.delete(*keys)
                    return {"status": "success", "deleted_keys": deleted}
            return {"status": "success", "deleted_keys": 0}
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            if not self.redis_client:
                return {"status": "error", "error": "Redis not available"}
            
            # Get cache info
            info = self.redis_client.info()
            keys = self.redis_client.keys("insights:*")
            
            return {
                "status": "success",
                "total_cached_items": len(keys),
                "cache_memory_usage": info.get('used_memory_human', 'unknown'),
                "redis_version": info.get('redis_version', 'unknown'),
                "connected_clients": info.get('connected_clients', 0)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _generate_multiple_insights(
        self, 
        request: InsightGenerationRequest,
        historical_analysis: Dict[str, Any],
        hit_potential_data: Dict[str, Any]
    ) -> AIInsightResult:
        """Generate insights for multiple analysis types."""
        try:
            # Get AI agent
            agent = await self._get_agent(request.agent_type, request.provider_preference)
            if not agent:
                raise RuntimeError("No AI agent available")
            
            # Generate insights for each analysis type
            insights = {}
            
            # Prepare enhanced context data
            enhanced_context = {
                "hit_potential": hit_potential_data,
                "historical_analysis": historical_analysis,
                "genre_benchmarks": self.historical_data.genre_benchmarks,
                "production_standards": self.historical_data.historical_data["production_standards"],
                "market_intelligence": self.historical_data.historical_data["market_intelligence"]
            }
            
            for analysis_type in request.analysis_types:
                try:
                    insight = await self._generate_single_analysis(
                        agent, analysis_type, request, enhanced_context
                    )
                    if insight:
                        insights[analysis_type.value] = insight
                except Exception as e:
                    logger.error(f"Error generating {analysis_type} analysis: {e}")
                    insights[analysis_type.value] = {"error": str(e)}
            
            # Map insights to result structure
            result = AIInsightResult(
                hit_potential_score=hit_potential_data.get('prediction', 0.0),
                hit_confidence=hit_potential_data.get('confidence', 0.0),
                model_used=hit_potential_data.get('model_used', 'unknown')
            )
            
            # Map specific insights
            if AnalysisType.MUSICAL_MEANING in request.analysis_types:
                musical_data = insights.get("musical_meaning", {})
                if isinstance(musical_data, dict) and "error" not in musical_data:
                    result.musical_meaning = MusicalMeaningInsight(**musical_data)
            
            if AnalysisType.HIT_COMPARISON in request.analysis_types:
                hit_data = insights.get("hit_comparison", {})
                if isinstance(hit_data, dict) and "error" not in hit_data:
                    result.hit_comparison = HitComparisonInsight(**hit_data)
            
            if AnalysisType.NOVELTY_ASSESSMENT in request.analysis_types:
                novelty_data = insights.get("novelty_assessment", {})
                if isinstance(novelty_data, dict) and "error" not in novelty_data:
                    result.novelty_assessment = NoveltyAssessmentInsight(**novelty_data)
            
            if AnalysisType.PRODUCTION_FEEDBACK in request.analysis_types:
                production_data = insights.get("production_feedback", {})
                if isinstance(production_data, dict) and "error" not in production_data:
                    result.production_feedback = ProductionFeedback(**production_data)
            
            if AnalysisType.STRATEGIC_INSIGHTS in request.analysis_types:
                strategic_data = insights.get("strategic_insights", {})
                if isinstance(strategic_data, dict) and "error" not in strategic_data:
                    result.strategic_insights = StrategicInsights(**strategic_data)
            
            # Generate executive summary
            result.executive_summary = await self._generate_executive_summary(
                request, result, enhanced_context
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in multiple insights generation: {e}")
            raise 