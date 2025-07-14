"""
Novelty Assessment Agent for Innovation and Trend Analysis

This agent specializes in evaluating musical innovation, uniqueness, and market timing
for new songs relative to current trends and historical patterns.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from .base_agent import BaseAgent
from ..models.agentic_models import AgentProfile, AgentRole, AgentContribution, ToolType, AgentTask

logger = logging.getLogger(__name__)

class NoveltyAssessmentAgent(BaseAgent):
    """
    Specialized agent for analyzing musical innovation and market novelty.
    
    This agent provides expert analysis on:
    - Musical innovation and creative risk-taking
    - Market timing and trend alignment  
    - Competitive differentiation factors
    - Cultural and artistic novelty assessment
    """
    
    def __init__(self, profile: AgentProfile):
        super().__init__(profile)
        self.agent_type = "NoveltyAssessment"
        logger.info(f"NoveltyAssessmentAgent initialized: {profile.name}")
        
    @classmethod
    def create_default_profile(cls) -> AgentProfile:
        """Create default profile for Novelty Assessment Agent."""
        from ..models.agentic_models import AgentCapability, ToolType
        
        return AgentProfile(
            name="TrendScope",
            role=AgentRole.NOVELTY_ASSESSMENT,
            description="Innovation expert and trend analyst specializing in musical novelty assessment",
            capabilities=[
                AgentCapability(
                    name="musical_innovation_analysis",
                    description="Analyzes musical elements for innovation and creativity",
                    confidence_level=0.9,
                    required_tools=[ToolType.MUSIC_DATABASE, ToolType.DATA_ANALYSIS],
                    complexity_level=8
                ),
                AgentCapability(
                    name="trend_pattern_recognition",
                    description="Identifies and analyzes market and musical trends",
                    confidence_level=0.85,
                    required_tools=[ToolType.MARKET_RESEARCH, ToolType.DATA_ANALYSIS],
                    complexity_level=7
                ),
                AgentCapability(
                    name="cultural_timing_assessment",
                    description="Evaluates cultural relevance and timing of musical content",
                    confidence_level=0.8,
                    required_tools=[ToolType.MARKET_RESEARCH],
                    complexity_level=7
                ),
                AgentCapability(
                    name="competitive_differentiation",
                    description="Assesses competitive positioning and uniqueness in the market",
                    confidence_level=0.85,
                    required_tools=[ToolType.MARKET_RESEARCH, ToolType.DATA_ANALYSIS],
                    complexity_level=8
                ),
                AgentCapability(
                    name="market_readiness_evaluation",
                    description="Evaluates readiness for market based on current trends",
                    confidence_level=0.8,
                    required_tools=[ToolType.MARKET_RESEARCH, ToolType.DATA_ANALYSIS],
                    complexity_level=7
                )
            ],
            expertise_areas=[
                "Musical Innovation", 
                "Market Trends",
                "Cultural Analysis",
                "Competitive Intelligence",
                "Timing Strategy"
            ],
            experience_level=9,
            confidence_score=0.88
        )
    
    async def analyze_task(self, request: Dict[str, Any], task: AgentTask) -> AgentContribution:
        """
        Analyze novelty and innovation aspects of a song.
        
        Args:
            request: Analysis request containing song data and context
            task: Specific task parameters for novelty assessment
            
        Returns:
            AgentContribution with novelty analysis results
        """
        start_time = datetime.utcnow()
        
        try:
            song_title = request.get('song_metadata', {}).get('title', 'Unknown')
            logger.info(f"NoveltyAssessmentAgent analyzing innovation for: {song_title}")
            
            # Prepare novelty analysis context
            analysis_context = self._prepare_novelty_context(request)
            
            # Generate novelty assessment using specialized reasoning
            novelty_assessment = await self._perform_novelty_analysis(analysis_context)
            
            # Structure and validate results
            structured_result = self._structure_novelty_response(novelty_assessment, request)
            
            logger.info(f"NoveltyAssessmentAgent completed analysis for: {song_title}")
            
            # Create AgentContribution object
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds() * 1000
            
            return AgentContribution(
                agent_id=self.profile.agent_id,
                agent_role=self.profile.role,
                findings=structured_result.get('unique_elements', []),
                insights=[
                    f"Innovation level: {structured_result.get('innovation_category', 'unknown')}",
                    f"Innovation score: {structured_result.get('innovation_score', 0.0):.2f}",
                    f"Cultural innovation: {structured_result.get('cultural_innovation', 'unknown')}"
                ],
                recommendations=structured_result.get('differentiation_factors', []),
                evidence=structured_result.get('precedent_analysis', []) if isinstance(structured_result.get('precedent_analysis'), list) else [structured_result.get('precedent_analysis', '')],
                methodology="novelty_assessment_analysis",
                tools_used=["music_database", "trend_analysis", "innovation_metrics"],
                confidence_level=structured_result.get('confidence_score', 0.82),
                reasoning_chain=[
                    "Analyzed musical innovation indicators",
                    "Assessed market timing and trend alignment",
                    "Evaluated cultural and artistic novelty",
                    "Calculated overall innovation score"
                ],
                processing_time_ms=processing_time,
                started_at=start_time,
                completed_at=end_time
            )
            
        except Exception as e:
            logger.error(f"Error in NoveltyAssessmentAgent analysis: {e}")
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds() * 1000
            
            return AgentContribution(
                agent_id=self.profile.agent_id,
                agent_role=self.profile.role,
                findings=[f"Analysis failed: {str(e)}"],
                insights=[],
                recommendations=[],
                evidence=[],
                methodology="error_handling",
                tools_used=[],
                confidence_level=0.0,
                reasoning_chain=[f"Error occurred during novelty assessment: {str(e)}"],
                processing_time_ms=processing_time,
                started_at=start_time,
                completed_at=end_time
            )
    
    def _prepare_novelty_context(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare comprehensive context for novelty assessment."""
        
        audio_analysis = request.get("audio_analysis", {})
        content_analysis = request.get("content_analysis", {})
        song_metadata = request.get("song_metadata", {})
        hit_prediction = request.get("hit_prediction", {})
        benchmark_analysis = request.get("benchmark_analysis", {})
        
        # Extract innovation indicators from audio features
        innovation_indicators = self._extract_innovation_indicators(audio_analysis)
        
        # Analyze genre and trend context
        genre_context = self._analyze_genre_context(song_metadata, benchmark_analysis)
        
        # Assess market timing factors
        market_timing = self._assess_market_timing(request)
        
        context = {
            "song_metadata": song_metadata,
            "innovation_indicators": innovation_indicators,
            "genre_context": genre_context,
            "market_timing": market_timing,
            "hit_prediction_score": hit_prediction.get("prediction", 0.0),
            "model_confidence": hit_prediction.get("confidence", 0.0),
            "benchmark_alignment": benchmark_analysis.get("overall_alignment", {}),
            "content_uniqueness": self._assess_content_uniqueness(content_analysis),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        return context
    
    def _extract_innovation_indicators(self, audio_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract indicators of musical innovation from audio features."""
        
        indicators = {
            "tempo_innovation": False,
            "energy_profile": "conventional",
            "harmonic_complexity": "standard",
            "production_uniqueness": "typical",
            "genre_fusion": False
        }
        
        try:
            # Analyze tempo innovation (outside typical ranges)
            tempo = audio_analysis.get("tempo", 120)
            if tempo < 60 or tempo > 180:
                indicators["tempo_innovation"] = True
                indicators["tempo_risk"] = "high" if tempo < 60 or tempo > 200 else "moderate"
            
            # Analyze energy profile patterns
            energy = audio_analysis.get("energy", 0.5)
            valence = audio_analysis.get("valence", 0.5)
            danceability = audio_analysis.get("danceability", 0.5)
            
            if energy > 0.9 or energy < 0.1:
                indicators["energy_profile"] = "extreme"
            elif (energy > 0.8 and valence < 0.3) or (energy < 0.3 and valence > 0.8):
                indicators["energy_profile"] = "contrasting"
            
            # Assess harmonic/production complexity
            acousticness = audio_analysis.get("acousticness", 0.5)
            instrumentalness = audio_analysis.get("instrumentalness", 0.5)
            
            if instrumentalness > 0.7:
                indicators["harmonic_complexity"] = "high_instrumental"
            elif acousticness < 0.1 and energy > 0.8:
                indicators["production_uniqueness"] = "highly_produced"
            elif acousticness > 0.9:
                indicators["production_uniqueness"] = "minimalist"
            
            # Detect potential genre fusion
            if audio_analysis.get("genre_primary"):
                indicators["genre_fusion"] = self._detect_genre_fusion(audio_analysis)
            
        except Exception as e:
            logger.warning(f"Error extracting innovation indicators: {e}")
        
        return indicators
    
    def _analyze_genre_context(self, song_metadata: Dict[str, Any], benchmark_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze genre context and trend positioning."""
        
        genre = song_metadata.get("genre", "unknown").lower()
        
        context = {
            "primary_genre": genre,
            "genre_maturity": "established",
            "trend_status": "stable",
            "innovation_opportunity": "moderate",
            "market_saturation": "medium"
        }
        
        try:
            # Assess genre maturity and trend status
            genre_trends = {
                "pop": {"maturity": "mature", "trend": "evolving", "saturation": "high"},
                "rock": {"maturity": "mature", "trend": "stable", "saturation": "high"}, 
                "electronic": {"maturity": "evolving", "trend": "growing", "saturation": "medium"},
                "hip_hop": {"maturity": "mature", "trend": "dominant", "saturation": "high"},
                "indie": {"maturity": "emerging", "trend": "growing", "saturation": "low"},
                "folk": {"maturity": "classic", "trend": "revival", "saturation": "low"}
            }
            
            if genre in genre_trends:
                trend_data = genre_trends[genre]
                context.update(trend_data)
            
            # Analyze benchmark deviation as innovation indicator
            if benchmark_analysis.get("overall_alignment", {}).get("score", 0.5) < 0.4:
                context["innovation_opportunity"] = "high"
                context["differentiation_potential"] = "strong"
            elif benchmark_analysis.get("overall_alignment", {}).get("score", 0.5) > 0.8:
                context["innovation_opportunity"] = "low"
                context["differentiation_potential"] = "challenging"
            
        except Exception as e:
            logger.warning(f"Error analyzing genre context: {e}")
        
        return context
    
    def _assess_market_timing(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Assess market timing and cultural readiness for innovation."""
        
        timing = {
            "cultural_moment": "neutral",
            "market_readiness": "moderate",
            "innovation_risk": "medium",
            "timing_score": 0.5
        }
        
        try:
            # Simple market timing assessment based on available data
            hit_prediction = request.get("hit_prediction", {})
            prediction_score = hit_prediction.get("prediction", 0.0)
            confidence = hit_prediction.get("confidence", 0.0)
            
            # High prediction with high confidence suggests good timing
            if prediction_score > 0.7 and confidence > 0.8:
                timing["market_readiness"] = "high"
                timing["timing_score"] = 0.8
            elif prediction_score < 0.4:
                timing["innovation_risk"] = "high"
                timing["timing_score"] = 0.3
            
            # Assess based on current timestamp (placeholder for real trend data)
            current_year = datetime.utcnow().year
            timing["analysis_year"] = current_year
            timing["cultural_moment"] = "digital_transformation_era"
            
        except Exception as e:
            logger.warning(f"Error assessing market timing: {e}")
        
        return timing
    
    def _assess_content_uniqueness(self, content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess lyrical and thematic uniqueness."""
        
        uniqueness = {
            "lyrical_innovation": "standard",
            "thematic_novelty": "conventional", 
            "narrative_structure": "typical",
            "cultural_relevance": "moderate"
        }
        
        try:
            # Analyze lexical diversity as innovation indicator
            lexical_diversity = content_analysis.get("lexical_diversity", 0.5)
            if lexical_diversity > 0.8:
                uniqueness["lyrical_innovation"] = "high"
            elif lexical_diversity < 0.3:
                uniqueness["lyrical_innovation"] = "low"
            
            # Assess sentiment complexity
            polarity = content_analysis.get("sentiment_polarity", 0.0)
            subjectivity = content_analysis.get("sentiment_subjectivity", 0.5)
            
            if abs(polarity) < 0.2 and subjectivity > 0.7:
                uniqueness["thematic_novelty"] = "complex_emotional"
            elif abs(polarity) > 0.8:
                uniqueness["thematic_novelty"] = "polarized"
            
            # Simple word count analysis
            word_count = content_analysis.get("word_count", 100)
            if word_count > 200:
                uniqueness["narrative_structure"] = "extensive"
            elif word_count < 50:
                uniqueness["narrative_structure"] = "minimalist"
            
        except Exception as e:
            logger.warning(f"Error assessing content uniqueness: {e}")
        
        return uniqueness
    
    def _detect_genre_fusion(self, audio_analysis: Dict[str, Any]) -> bool:
        """Detect potential genre fusion based on audio characteristics."""
        
        try:
            # Simple fusion detection based on contrasting characteristics
            acousticness = audio_analysis.get("acousticness", 0.5)
            energy = audio_analysis.get("energy", 0.5)
            danceability = audio_analysis.get("danceability", 0.5)
            
            # Electronic-acoustic fusion
            if 0.3 < acousticness < 0.7 and energy > 0.7:
                return True
            
            # High energy with low danceability (rock-electronic fusion)
            if energy > 0.8 and danceability < 0.4:
                return True
            
            # Other fusion indicators could be added here
            
        except Exception as e:
            logger.warning(f"Error detecting genre fusion: {e}")
        
        return False
    
    async def _perform_novelty_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the core novelty analysis using agent reasoning."""
        
        try:
            # Build novelty analysis prompt
            novelty_prompt = self._build_novelty_prompt(context)
            
            # For now, we'll generate a structured analysis based on the context
            # In production, this would call the LLM service
            analysis = self._generate_novelty_insights(context)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in novelty analysis: {e}")
            return self._create_fallback_analysis(context)
    
    def _build_novelty_prompt(self, context: Dict[str, Any]) -> str:
        """Build specialized prompt for novelty assessment."""
        
        prompt = f"""You are TrendScope, an innovation expert and trend analyst in the music industry. 
Analyze the novelty and innovation aspects of this song:

SONG CONTEXT:
- Title: {context.get('song_metadata', {}).get('title', 'Unknown')}
- Genre: {context.get('genre_context', {}).get('primary_genre', 'unknown')}
- Hit Prediction: {context.get('hit_prediction_score', 0.0):.2f}

INNOVATION INDICATORS:
{context.get('innovation_indicators', {})}

GENRE CONTEXT:
{context.get('genre_context', {})}

MARKET TIMING:
{context.get('market_timing', {})}

CONTENT UNIQUENESS:
{context.get('content_uniqueness', {})}

Provide detailed novelty assessment focusing on:
1. Musical innovation level and specific unique elements
2. Market timing and trend alignment
3. Cultural innovation and artistic risk-taking
4. Commercial viability vs. innovation balance
5. Competitive differentiation factors

Return structured JSON with innovation scores and detailed analysis."""
        
        return prompt
    
    def _generate_novelty_insights(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate novelty insights based on context analysis."""
        
        innovation_indicators = context.get("innovation_indicators", {})
        genre_context = context.get("genre_context", {})
        market_timing = context.get("market_timing", {})
        content_uniqueness = context.get("content_uniqueness", {})
        
        # Calculate innovation score
        innovation_score = self._calculate_innovation_score(
            innovation_indicators, genre_context, content_uniqueness
        )
        
        # Determine innovation category
        if innovation_score >= 0.8:
            innovation_category = "breakthrough"
        elif innovation_score >= 0.6:
            innovation_category = "moderate"
        else:
            innovation_category = "incremental"
        
        # Generate unique elements list
        unique_elements = self._identify_unique_elements(innovation_indicators, content_uniqueness)
        
        # Generate risk assessment
        risk_assessment = self._assess_innovation_risk(innovation_score, market_timing, genre_context)
        
        analysis = {
            "innovation_score": round(innovation_score, 3),
            "unique_elements": unique_elements,
            "innovation_category": innovation_category,
            "trend_alignment": {
                "emerging_trends": genre_context.get("trend_status", "stable"),
                "market_timing": market_timing.get("timing_score", 0.5)
            },
            "cultural_innovation": content_uniqueness.get("thematic_novelty", "conventional"),
            "production_innovation": innovation_indicators.get("production_uniqueness", "typical"),
            "risk_assessment": risk_assessment,
            "differentiation_factors": self._identify_differentiation_factors(context),
            "precedent_analysis": f"Innovation level comparable to {innovation_category} musical developments",
            "adoption_prediction": self._predict_adoption_pattern(innovation_score, market_timing),
            "confidence_score": 0.82,
            "agent": "NoveltyAssessmentAgent",
            "analysis_timestamp": context.get("analysis_timestamp")
        }
        
        return analysis
    
    def _calculate_innovation_score(self, indicators: Dict, genre: Dict, content: Dict) -> float:
        """Calculate overall innovation score from multiple factors."""
        
        score = 0.5  # Base score
        
        # Audio innovation factors
        if indicators.get("tempo_innovation"):
            score += 0.15
        
        if indicators.get("energy_profile") in ["extreme", "contrasting"]:
            score += 0.1
        
        if indicators.get("genre_fusion"):
            score += 0.2
        
        if indicators.get("production_uniqueness") in ["highly_produced", "minimalist"]:
            score += 0.1
        
        # Content innovation factors  
        if content.get("lyrical_innovation") == "high":
            score += 0.15
        
        if content.get("thematic_novelty") in ["complex_emotional", "polarized"]:
            score += 0.1
        
        # Genre context adjustment
        if genre.get("innovation_opportunity") == "high":
            score += 0.1
        elif genre.get("innovation_opportunity") == "low":
            score -= 0.1
        
        return min(1.0, max(0.0, score))
    
    def _identify_unique_elements(self, indicators: Dict, content: Dict) -> list:
        """Identify specific unique elements in the song."""
        
        elements = []
        
        if indicators.get("tempo_innovation"):
            elements.append("Unconventional tempo choice creates distinctive rhythmic character")
        
        if indicators.get("genre_fusion"):
            elements.append("Cross-genre fusion creates unique sonic palette")
        
        if indicators.get("energy_profile") == "contrasting":
            elements.append("Contrasting energy-valence dynamic creates emotional complexity")
        
        if content.get("lyrical_innovation") == "high":
            elements.append("High lexical diversity demonstrates sophisticated wordcraft")
        
        if content.get("narrative_structure") in ["extensive", "minimalist"]:
            structure = content["narrative_structure"]
            elements.append(f"{structure.title()} narrative approach differs from genre conventions")
        
        if not elements:
            elements.append("Subtle innovations within established genre conventions")
        
        return elements
    
    def _assess_innovation_risk(self, innovation_score: float, timing: Dict, genre: Dict) -> Dict[str, Any]:
        """Assess the commercial risk of the innovation level."""
        
        if innovation_score > 0.8:
            commercial_risk = "high"
            artistic_reward = "potentially transformative"
        elif innovation_score > 0.6:
            commercial_risk = "moderate"
            artistic_reward = "significant artistic impact"
        else:
            commercial_risk = "low"
            artistic_reward = "incremental artistic development"
        
        market_readiness = timing.get("market_readiness", "moderate")
        
        return {
            "commercial_risk": commercial_risk,
            "artistic_reward": artistic_reward,
            "market_readiness": market_readiness
        }
    
    def _identify_differentiation_factors(self, context: Dict[str, Any]) -> list:
        """Identify key competitive differentiation factors."""
        
        factors = []
        
        innovation_indicators = context.get("innovation_indicators", {})
        content_uniqueness = context.get("content_uniqueness", {})
        
        if innovation_indicators.get("production_uniqueness") != "typical":
            factors.append(f"Distinctive production style: {innovation_indicators['production_uniqueness']}")
        
        if content_uniqueness.get("thematic_novelty") != "conventional":
            factors.append(f"Unique thematic approach: {content_uniqueness['thematic_novelty']}")
        
        if innovation_indicators.get("genre_fusion"):
            factors.append("Cross-genre innovation creates market differentiation")
        
        if not factors:
            factors.append("Competitive positioning through execution excellence")
        
        return factors
    
    def _predict_adoption_pattern(self, innovation_score: float, timing: Dict) -> str:
        """Predict how the innovation might be adopted by the market."""
        
        timing_score = timing.get("timing_score", 0.5)
        
        if innovation_score > 0.8 and timing_score > 0.7:
            return "Rapid adoption likely due to strong innovation with good market timing"
        elif innovation_score > 0.8:
            return "Gradual adoption expected - innovation ahead of market readiness"
        elif innovation_score > 0.6 and timing_score > 0.6:
            return "Steady adoption within target segments"
        else:
            return "Traditional adoption pattern within established market"
    
    def _create_fallback_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback analysis when detailed analysis fails."""
        
        return {
            "innovation_score": 0.5,
            "unique_elements": ["Analysis incomplete - basic innovation assessment"],
            "innovation_category": "incremental",
            "trend_alignment": {"status": "analysis_incomplete"},
            "cultural_innovation": "assessment_pending",
            "risk_assessment": {"commercial_risk": "unknown", "artistic_reward": "pending_analysis"},
            "differentiation_factors": ["Detailed analysis unavailable"],
            "confidence_score": 0.3,
            "agent": "NoveltyAssessmentAgent",
            "status": "fallback_analysis"
        }
    
    def _structure_novelty_response(self, analysis: Dict[str, Any], request: Dict[str, Any]) -> Dict[str, Any]:
        """Structure the novelty analysis response with metadata."""
        
        # Add metadata and context
        analysis.update({
            "song_title": request.get("song_metadata", {}).get("title", "Unknown"),
            "analysis_type": "novelty_assessment",
            "agent": "NoveltyAssessmentAgent",
            "key_insights": [
                f"Innovation level: {analysis.get('innovation_category', 'unknown')}",
                f"Innovation score: {analysis.get('innovation_score', 0.0):.2f}",
                f"Market timing: {analysis.get('trend_alignment', {}).get('market_timing', 'unknown')}"
            ]
        })
        
        return analysis
    
    def get_expertise_areas(self) -> List[str]:
        """Get the agent's areas of expertise."""
        return [
            "Musical Innovation",
            "Market Trends", 
            "Cultural Analysis",
            "Competitive Intelligence",
            "Timing Strategy"
        ]
    
    def get_preferred_tools(self) -> List[ToolType]:
        """Get the agent's preferred tools."""
        return [
            ToolType.MUSIC_DATABASE,
            ToolType.DATA_ANALYSIS,
            ToolType.MARKET_RESEARCH
        ]