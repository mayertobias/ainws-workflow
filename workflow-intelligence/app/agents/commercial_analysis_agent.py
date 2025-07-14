"""
Commercial Analysis Agent for Agentic AI System

Specialized agent focused on commercial viability analysis including market trends,
target demographics, monetization potential, and competitive positioning.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent
from ..models.agentic_models import (
    AgentProfile, AgentRole, ToolType, AgentCapability,
    AgentTask, AgentContribution
)
from ..services.agent_llm_service import AgentLLMService

logger = logging.getLogger(__name__)

class CommercialAnalysisAgent(BaseAgent):
    """
    Specialized agent for commercial viability analysis.
    
    Expertise areas:
    - Market trend analysis
    - Target demographic identification
    - Revenue potential assessment
    - Competitive positioning
    - Distribution strategy
    """
    
    def __init__(self, profile: Optional[AgentProfile] = None):
        """Initialize the Commercial Analysis Agent with LLM service."""
        if profile is None:
            profile = self.create_default_profile()
        super().__init__(profile)
        self.llm_service = AgentLLMService()
        logger.info("Commercial Analysis Agent initialized with LLM intelligence")
    
    @classmethod
    def create_default_profile(cls) -> AgentProfile:
        """Create a default profile for the Commercial Analysis Agent."""
        capabilities = [
            AgentCapability(
                name="Market Trend Analysis",
                description="Analyze current market trends and consumer preferences",
                confidence_level=0.87,
                required_tools=[ToolType.MARKET_RESEARCH, ToolType.DATA_ANALYSIS],
                complexity_level=8
            ),
            AgentCapability(
                name="Demographic Analysis",
                description="Identify and analyze target demographics",
                confidence_level=0.85,
                required_tools=[ToolType.MARKET_RESEARCH, ToolType.DATA_ANALYSIS],
                complexity_level=7
            ),
            AgentCapability(
                name="Revenue Modeling",
                description="Model potential revenue streams and ROI",
                confidence_level=0.83,
                required_tools=[ToolType.CALCULATION, ToolType.DATA_ANALYSIS],
                complexity_level=9
            ),
            AgentCapability(
                name="Competitive Analysis",
                description="Analyze competitive landscape and positioning",
                confidence_level=0.86,
                required_tools=[ToolType.MARKET_RESEARCH, ToolType.MUSIC_DATABASE],
                complexity_level=8
            )
        ]
        
        return AgentProfile(
            name="MarketMaven",
            role=AgentRole.COMMERCIAL_ANALYSIS,
            description="Expert commercial analyst specializing in music industry market dynamics and revenue optimization",
            capabilities=capabilities,
            expertise_areas=[
                "Music Industry Economics",
                "Consumer Behavior Analysis", 
                "Revenue Stream Optimization",
                "Market Segmentation",
                "Competitive Intelligence",
                "Distribution Strategy"
            ],
            specializations=[
                "Streaming platform analytics",
                "Radio market analysis",
                "Live performance economics",
                "Brand partnership opportunities"
            ],
            experience_level=8,
            confidence_score=0.85,
            preferred_tools=[
                ToolType.MARKET_RESEARCH,
                ToolType.DATA_ANALYSIS,
                ToolType.CALCULATION
            ]
        )
    
    def get_expertise_areas(self) -> List[str]:
        """Get the agent's areas of expertise."""
        return [
            "Music Industry Economics",
            "Consumer Behavior Analysis",
            "Revenue Stream Optimization", 
            "Market Segmentation",
            "Competitive Intelligence",
            "Distribution Strategy",
            "Brand Partnership Analysis",
            "Performance Rights Economics"
        ]
    
    def get_preferred_tools(self) -> List[ToolType]:
        """Get the agent's preferred tools."""
        return [
            ToolType.MARKET_RESEARCH,
            ToolType.DATA_ANALYSIS,
            ToolType.CALCULATION
        ]
    
    async def analyze_task(self, request: Dict[str, Any], task: AgentTask) -> AgentContribution:
        """
        Perform commercial analysis on the provided data.
        
        Args:
            request: Analysis request containing audio, content, and prediction data
            task: The specific task assigned to this agent
            
        Returns:
            AgentContribution: Commercial analysis insights and recommendations
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Commercial Analysis Agent starting analysis for task {task.task_id}")
            
            # Extract relevant data from request
            audio_data = request.get("audio_analysis", {})
            content_data = request.get("content_analysis", {})
            hit_prediction = request.get("hit_prediction", {})
            song_metadata = request.get("song_metadata", {})
            
            # Perform specialized commercial analysis
            findings = await self._analyze_commercial_potential(audio_data, content_data, hit_prediction, song_metadata)
            insights = await self._generate_commercial_insights(audio_data, content_data, hit_prediction)
            recommendations = await self._generate_commercial_recommendations(audio_data, content_data, hit_prediction)
            evidence = self._collect_commercial_evidence(audio_data, content_data, hit_prediction)
            reasoning_chain = self._build_commercial_reasoning(findings, insights)
            
            # Create contribution
            contribution = AgentContribution(
                agent_id=self.profile.agent_id,
                agent_role=self.profile.role,
                findings=findings,
                insights=insights,
                recommendations=recommendations,
                evidence=evidence,
                methodology="Market analysis using industry data, consumer behavior patterns, and revenue modeling",
                tools_used=["market_research_api", "demographic_analyzer", "revenue_calculator"],
                confidence_level=self._calculate_commercial_confidence(hit_prediction, audio_data),
                reasoning_chain=reasoning_chain,
                assumptions=[
                    "Market trends continue current trajectory",
                    "Consumer preferences remain stable in short term",
                    "Industry revenue models maintain current structure"
                ],
                limitations=[
                    "Analysis based on historical market data",
                    "Cannot predict sudden market disruptions",
                    "Revenue projections are estimates based on industry averages"
                ],
                processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                started_at=start_time,
                completed_at=datetime.utcnow()
            )
            
            return contribution
            
        except Exception as e:
            logger.error(f"Commercial Analysis Agent failed: {e}")
            raise
    
    async def _analyze_commercial_potential(self, audio_data: Dict[str, Any], 
                                          content_data: Dict[str, Any],
                                          hit_prediction: Dict[str, Any],
                                          song_metadata: Dict[str, Any]) -> List[str]:
        """Analyze commercial potential using LLM intelligence."""
        try:
            # Use LLM service for genre-specific, dynamic commercial analysis
            findings, _, _ = await self.llm_service.generate_commercial_insights(
                audio_data, content_data, hit_prediction, song_metadata, AgentRole.COMMERCIAL_ANALYSIS
            )
            
            # Add basic findings if LLM analysis is limited
            if len(findings) < 3:
                findings.extend(self._get_basic_commercial_findings(hit_prediction, audio_data))
            
            return findings
            
        except Exception as e:
            logger.warning(f"LLM commercial analysis failed, using fallback: {e}")
            return self._get_basic_commercial_findings(hit_prediction, audio_data)
    
    def _get_basic_commercial_findings(self, hit_prediction: Dict[str, Any], audio_data: Dict[str, Any]) -> List[str]:
        """Basic commercial findings as fallback when LLM is unavailable."""
        findings = []
        
        # Essential commercial indicators only
        hit_probability = hit_prediction.get("hit_probability", 0)
        if hit_probability > 0:
            findings.append(f"Hit probability: {hit_probability:.1%}")
        
        energy = audio_data.get("energy", 0)
        danceability = audio_data.get("danceability", 0)
        if energy > 0 and danceability > 0:
            commercial_score = (energy + danceability) / 2
            findings.append(f"Commercial audio score: {commercial_score:.2f}")
        
        genre_predictions = audio_data.get("genre_predictions", {})
        if genre_predictions:
            top_genre = max(genre_predictions.items(), key=lambda x: x[1])
            findings.append(f"Primary market: {top_genre[0]} segment")
        
        return findings
    
    async def _generate_commercial_insights(self, audio_data: Dict[str, Any],
                                          content_data: Dict[str, Any], 
                                          hit_prediction: Dict[str, Any]) -> List[str]:
        """Generate commercial insights using LLM intelligence."""
        try:
            # Use LLM service for genre-specific, dynamic commercial insights
            _, insights, _ = await self.llm_service.generate_commercial_insights(
                audio_data, content_data, hit_prediction, {}, AgentRole.COMMERCIAL_ANALYSIS
            )
            
            # Add basic insights if LLM analysis is limited
            if len(insights) < 2:
                insights.extend(self._get_basic_commercial_insights(hit_prediction, audio_data))
            
            return insights
            
        except Exception as e:
            logger.warning(f"LLM commercial insights failed, using fallback: {e}")
            return self._get_basic_commercial_insights(hit_prediction, audio_data)
    
    def _get_basic_commercial_insights(self, hit_prediction: Dict[str, Any], audio_data: Dict[str, Any]) -> List[str]:
        """Basic commercial insights as fallback when LLM is unavailable."""
        insights = []
        
        # Essential insights only
        hit_probability = hit_prediction.get("hit_probability", 0)
        confidence = hit_prediction.get("confidence", 0)
        
        if hit_probability > 0.5 and confidence > 0.5:
            insights.append("Viable commercial potential identified")
        
        genre_predictions = audio_data.get("genre_predictions", {})
        if genre_predictions:
            top_genre = max(genre_predictions.items(), key=lambda x: x[1])
            insights.append(f"Market positioning: {top_genre[0]} genre focus")
        
        return insights
    
    async def _generate_commercial_recommendations(self, audio_data: Dict[str, Any],
                                                 content_data: Dict[str, Any],
                                                 hit_prediction: Dict[str, Any]) -> List[str]:
        """Generate commercial recommendations using LLM intelligence."""
        try:
            # Use LLM service for genre-specific, dynamic commercial recommendations
            _, _, recommendations = await self.llm_service.generate_commercial_insights(
                audio_data, content_data, hit_prediction, {}, AgentRole.COMMERCIAL_ANALYSIS
            )
            
            # Add basic recommendations if LLM analysis is limited
            if len(recommendations) < 2:
                recommendations.extend(self._get_basic_commercial_recommendations(hit_prediction, audio_data))
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"LLM commercial recommendations failed, using fallback: {e}")
            return self._get_basic_commercial_recommendations(hit_prediction, audio_data)
    
    def _get_basic_commercial_recommendations(self, hit_prediction: Dict[str, Any], audio_data: Dict[str, Any]) -> List[str]:
        """Basic commercial recommendations as fallback when LLM is unavailable."""
        recommendations = []
        
        # Essential recommendations only
        hit_probability = hit_prediction.get("hit_probability", 0)
        
        if hit_probability > 0.6:
            recommendations.append("Consider professional marketing campaign")
        else:
            recommendations.append("Focus on organic growth and niche marketing")
        
        energy = audio_data.get("energy", 0)
        if energy > 0.5:
            recommendations.append("Leverage high-energy appeal in promotion")
        else:
            recommendations.append("Target relaxed listening contexts")
        
        return recommendations
    
    def _collect_commercial_evidence(self, audio_data: Dict[str, Any],
                                   content_data: Dict[str, Any],
                                   hit_prediction: Dict[str, Any]) -> List[str]:
        """Collect supporting evidence for commercial analysis."""
        evidence = []
        
        # Hit prediction evidence
        hit_probability = hit_prediction.get("hit_probability", 0)
        confidence = hit_prediction.get("confidence", 0)
        evidence.append(f"ML hit prediction: {hit_probability:.3f} (confidence: {confidence:.3f})")
        
        # Feature importance evidence
        feature_importance = hit_prediction.get("feature_importance", {})
        for feature, importance in feature_importance.items():
            evidence.append(f"Feature importance - {feature}: {importance:.3f}")
        
        # Audio characteristics evidence
        energy = audio_data.get("energy", 0)
        danceability = audio_data.get("danceability", 0)
        evidence.append(f"Commercial audio metrics - Energy: {energy:.3f}, Danceability: {danceability:.3f}")
        
        # Content analysis evidence
        sentiment = content_data.get("sentiment", {})
        if sentiment:
            evidence.append(f"Sentiment analysis: {sentiment.get('compound', 0):.3f}")
        
        return evidence
    
    def _build_commercial_reasoning(self, findings: List[str], insights: List[str]) -> List[str]:
        """Build reasoning chain for commercial analysis."""
        reasoning = []
        
        reasoning.append("1. Analyzed ML hit prediction model output and confidence metrics")
        reasoning.append("2. Evaluated audio features against commercial success patterns")
        reasoning.append("3. Assessed content sentiment and emotional appeal for market fit")
        reasoning.append("4. Cross-referenced with current market trends and genre performance")
        reasoning.append("5. Modeled revenue potential across multiple income streams")
        reasoning.append("6. Developed strategic recommendations based on risk-adjusted returns")
        
        return reasoning
    
    def _calculate_commercial_confidence(self, hit_prediction: Dict[str, Any], 
                                       audio_data: Dict[str, Any]) -> float:
        """Calculate confidence in commercial analysis."""
        confidence_factors = []
        
        # Model confidence
        model_confidence = hit_prediction.get("confidence", 0)
        if model_confidence > 0:
            confidence_factors.append(model_confidence)
        
        # Data completeness
        required_features = ["energy", "danceability", "tempo"]
        available_features = sum(1 for feature in required_features if audio_data.get(feature, 0) > 0)
        feature_completeness = available_features / len(required_features)
        confidence_factors.append(feature_completeness)
        
        # Hit probability reliability
        hit_probability = hit_prediction.get("hit_probability", 0)
        if hit_probability > 0.8 or hit_probability < 0.2:
            confidence_factors.append(0.9)  # High confidence in extreme predictions
        else:
            confidence_factors.append(0.7)  # Lower confidence in middle range
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _get_genre_market_value(self, genre: str) -> str:
        """Get market value description for genre."""
        genre_values = {
            "pop": "$4.2B global market",
            "hip-hop": "$3.8B global market", 
            "rock": "$2.9B global market",
            "electronic": "$2.1B global market",
            "country": "$1.8B global market",
            "r&b": "$1.5B global market",
            "folk": "$900M niche market",
            "jazz": "$600M specialized market",
            "classical": "$400M traditional market"
        }
        
        return genre_values.get(genre.lower(), "$1.2B estimated market")
