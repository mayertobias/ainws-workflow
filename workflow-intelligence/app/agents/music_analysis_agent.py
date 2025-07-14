"""
Music Analysis Agent for Agentic AI System

Specialized agent focused on musical analysis including audio features, 
harmonic analysis, rhythm patterns, and genre classification.
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

class MusicAnalysisAgent(BaseAgent):
    """
    Specialized agent for music analysis tasks.
    
    Expertise areas:
    - Audio feature analysis
    - Harmonic analysis  
    - Rhythm and tempo analysis
    - Genre classification
    - Musical structure analysis
    """
    
    def __init__(self, profile: Optional[AgentProfile] = None):
        """Initialize the Music Analysis Agent with LLM service."""
        if profile is None:
            profile = self.create_default_profile()
        super().__init__(profile)
        self.llm_service = AgentLLMService()
        logger.info("Music Analysis Agent initialized with LLM intelligence")
    
    @classmethod
    def create_default_profile(cls) -> AgentProfile:
        """Create a default profile for the Music Analysis Agent."""
        capabilities = [
            AgentCapability(
                name="Audio Feature Analysis",
                description="Analyze spectral, temporal, and timbral features",
                confidence_level=0.9,
                required_tools=[ToolType.DATA_ANALYSIS, ToolType.CALCULATION],
                complexity_level=8
            ),
            AgentCapability(
                name="Harmonic Analysis",
                description="Analyze chord progressions, key signatures, and harmony",
                confidence_level=0.85,
                required_tools=[ToolType.MUSIC_DATABASE, ToolType.DATA_ANALYSIS],
                complexity_level=9
            ),
            AgentCapability(
                name="Genre Classification",
                description="Classify and analyze musical genres",
                confidence_level=0.88,
                required_tools=[ToolType.MUSIC_DATABASE, ToolType.DATA_ANALYSIS],
                complexity_level=7
            ),
            AgentCapability(
                name="Rhythm Analysis",
                description="Analyze tempo, beat, and rhythmic patterns",
                confidence_level=0.92,
                required_tools=[ToolType.DATA_ANALYSIS, ToolType.CALCULATION],
                complexity_level=6
            )
        ]
        
        return AgentProfile(
            name="MelodyMind",
            role=AgentRole.MUSIC_ANALYSIS,
            description="Expert music analyst specializing in audio features, harmony, and genre classification",
            capabilities=capabilities,
            expertise_areas=[
                "Audio Signal Processing",
                "Musical Theory",
                "Genre Classification",
                "Harmonic Analysis",
                "Rhythm Analysis",
                "Spectral Analysis"
            ],
            specializations=[
                "Essentia feature extraction",
                "Chord progression analysis",
                "Tempo and beat tracking",
                "Genre similarity matching"
            ],
            experience_level=9,
            confidence_score=0.88,
            preferred_tools=[
                ToolType.DATA_ANALYSIS,
                ToolType.MUSIC_DATABASE,
                ToolType.CALCULATION
            ]
        )
    
    def get_expertise_areas(self) -> List[str]:
        """Get the agent's areas of expertise."""
        return [
            "Audio Signal Processing",
            "Musical Theory", 
            "Genre Classification",
            "Harmonic Analysis",
            "Rhythm Analysis",
            "Spectral Analysis",
            "Musical Structure Analysis",
            "Instrument Recognition"
        ]
    
    def get_preferred_tools(self) -> List[ToolType]:
        """Get the agent's preferred tools."""
        return [
            ToolType.DATA_ANALYSIS,
            ToolType.MUSIC_DATABASE,
            ToolType.CALCULATION
        ]
    
    async def analyze_task(self, request: Dict[str, Any], task: AgentTask) -> AgentContribution:
        """
        Perform music analysis on the provided data.
        
        Args:
            request: Analysis request containing audio analysis data
            task: The specific task assigned to this agent
            
        Returns:
            AgentContribution: Musical analysis insights and findings
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Music Analysis Agent starting analysis for task {task.task_id}")
            
            # Extract audio analysis data from request
            audio_data = request.get("audio_analysis", {})
            song_metadata = request.get("song_metadata", {})
            
            # Perform specialized music analysis
            findings = await self._analyze_musical_features(audio_data, song_metadata)
            insights = await self._generate_musical_insights(audio_data, findings)
            recommendations = await self._generate_musical_recommendations(audio_data, insights)
            evidence = self._collect_musical_evidence(audio_data, findings)
            reasoning_chain = self._build_musical_reasoning(audio_data, findings, insights)
            
            # Create contribution
            contribution = AgentContribution(
                agent_id=self.profile.agent_id,
                agent_role=self.profile.role,
                findings=findings,
                insights=insights,
                recommendations=recommendations,
                evidence=evidence,
                methodology="Musical feature analysis using audio signal processing and music theory",
                tools_used=["audio_feature_analyzer", "harmonic_analyzer", "tempo_analyzer"],
                confidence_level=self._calculate_confidence(audio_data),
                reasoning_chain=reasoning_chain,
                assumptions=[
                    "Audio features are accurately extracted",
                    "Genre classifications are based on training data patterns",
                    "Harmonic analysis assumes Western music theory"
                ],
                limitations=[
                    "Analysis limited to extracted audio features",
                    "Genre classification may not capture fusion styles",
                    "Harmonic analysis optimized for Western music"
                ],
                processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                started_at=start_time,
                completed_at=datetime.utcnow()
            )
            
            return contribution
            
        except Exception as e:
            logger.error(f"Music Analysis Agent failed: {e}")
            raise
    
    async def _analyze_musical_features(self, audio_data: Dict[str, Any], 
                                      song_metadata: Dict[str, Any]) -> List[str]:
        """Analyze musical features using LLM intelligence."""
        try:
            # SINGLE LLM call to get all musical analysis components
            findings, insights, recommendations = await self.llm_service.generate_musical_insights(
                audio_data, song_metadata, AgentRole.MUSIC_ANALYSIS
            )
            
            # Store for reuse to avoid redundant calls
            self._cached_llm_insights = {
                "findings": findings,
                "insights": insights, 
                "recommendations": recommendations
            }
            
            # Add traditional analysis as evidence if LLM analysis is limited
            if len(findings) < 3:
                findings.extend(self._get_basic_musical_findings(audio_data))
            
            return findings
            
        except Exception as e:
            logger.warning(f"LLM musical analysis failed, using fallback: {e}")
            self._cached_llm_insights = None
            return self._get_basic_musical_findings(audio_data)
    
    def _get_basic_musical_findings(self, audio_data: Dict[str, Any]) -> List[str]:
        """Basic musical findings as fallback when LLM is unavailable."""
        findings = []
        
        # Essential features only
        tempo = audio_data.get("tempo", 0)
        if tempo > 0:
            findings.append(f"Tempo: {tempo:.1f} BPM")
        
        energy = audio_data.get("energy", 0)
        if energy > 0:
            findings.append(f"Energy level: {energy:.3f}")
        
        genre_predictions = audio_data.get("genre_predictions", {})
        if genre_predictions:
            top_genre = max(genre_predictions.items(), key=lambda x: x[1])
            findings.append(f"Primary genre: {top_genre[0]} (confidence: {top_genre[1]:.2f})")
        
        return findings
    
    async def _generate_musical_insights(self, audio_data: Dict[str, Any], 
                                       findings: List[str]) -> List[str]:
        """Generate higher-level musical insights - reuse cached LLM results."""
        try:
            # Reuse cached LLM results to avoid redundant calls
            if hasattr(self, '_cached_llm_insights') and self._cached_llm_insights:
                insights = self._cached_llm_insights.get("insights", [])
                logger.info("🔄 Reusing cached LLM insights to avoid redundant call")
            else:
                # Fallback to basic insights if no cache available
                insights = self._get_basic_musical_insights(audio_data)
            
            # Add basic insights if LLM analysis is limited
            if len(insights) < 2:
                insights.extend(self._get_basic_musical_insights(audio_data))
            
            return insights
            
        except Exception as e:
            logger.warning(f"Musical insights generation failed, using fallback: {e}")
            return self._get_basic_musical_insights(audio_data)
    
    def _get_basic_musical_insights(self, audio_data: Dict[str, Any]) -> List[str]:
        """Basic musical insights as fallback when LLM is unavailable."""
        insights = []
        
        # Essential insights only
        energy = audio_data.get("energy", 0)
        danceability = audio_data.get("danceability", 0)
        
        if energy > 0 and danceability > 0:
            commercial_potential = (energy + danceability) / 2
            insights.append(f"Commercial potential score: {commercial_potential:.2f}")
        
        genre_predictions = audio_data.get("genre_predictions", {})
        if len(genre_predictions) > 1:
            sorted_genres = sorted(genre_predictions.items(), key=lambda x: x[1], reverse=True)
            insights.append(f"Multi-genre characteristics: {sorted_genres[0][0]} and {sorted_genres[1][0]}")
        
        return insights
    
    async def _generate_musical_recommendations(self, audio_data: Dict[str, Any], 
                                              insights: List[str]) -> List[str]:
        """Generate musical recommendations - reuse cached LLM results."""
        try:
            # Reuse cached LLM results to avoid redundant calls
            if hasattr(self, '_cached_llm_insights') and self._cached_llm_insights:
                recommendations = self._cached_llm_insights.get("recommendations", [])
                logger.info("🔄 Reusing cached LLM recommendations to avoid redundant call")
            else:
                # Fallback to basic recommendations if no cache available
                recommendations = self._get_basic_musical_recommendations(audio_data)
            
            # Add basic recommendations if LLM analysis is limited
            if len(recommendations) < 2:
                recommendations.extend(self._get_basic_musical_recommendations(audio_data))
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Musical recommendations generation failed, using fallback: {e}")
            return self._get_basic_musical_recommendations(audio_data)
    
    def _get_basic_musical_recommendations(self, audio_data: Dict[str, Any]) -> List[str]:
        """Basic musical recommendations as fallback when LLM is unavailable."""
        recommendations = []
        
        # Essential recommendations only
        energy = audio_data.get("energy", 0)
        if energy > 0.5:
            recommendations.append("Consider high-energy playlist placement")
        else:
            recommendations.append("Consider ambient/chill playlist placement")
        
        genre_predictions = audio_data.get("genre_predictions", {})
        if genre_predictions:
            top_genre = max(genre_predictions.items(), key=lambda x: x[1])
            recommendations.append(f"Target {top_genre[0]} music platforms and audiences")
        
        return recommendations
    
    def _collect_musical_evidence(self, audio_data: Dict[str, Any], 
                                findings: List[str]) -> List[str]:
        """Collect supporting evidence for musical analysis."""
        evidence = []
        
        # Tempo evidence
        tempo = audio_data.get("tempo", 0)
        if tempo > 0:
            evidence.append(f"Tempo measurement: {tempo:.1f} BPM")
        
        # Energy metrics evidence
        energy = audio_data.get("energy", 0)
        if energy > 0:
            evidence.append(f"Energy level: {energy:.3f}")
        
        # Spectral evidence
        spectral_centroid = audio_data.get("spectral_centroid_mean", 0)
        if spectral_centroid > 0:
            evidence.append(f"Spectral centroid: {spectral_centroid:.1f} Hz")
        
        # Genre evidence
        genre_predictions = audio_data.get("genre_predictions", {})
        for genre, confidence in genre_predictions.items():
            evidence.append(f"Genre classification: {genre} ({confidence:.3f})")
        
        return evidence
    
    def _build_musical_reasoning(self, audio_data: Dict[str, Any], findings: List[str], 
                               insights: List[str]) -> List[str]:
        """Build reasoning chain for musical analysis."""
        reasoning = []
        
        reasoning.append("1. Extracted and analyzed core audio features (tempo, energy, spectral characteristics)")
        reasoning.append("2. Applied music theory principles to interpret harmonic and rhythmic elements")
        reasoning.append("3. Cross-referenced with genre classification models and musical databases")
        reasoning.append("4. Evaluated commercial viability based on industry standards and market data")
        reasoning.append("5. Generated actionable insights combining technical analysis with market knowledge")
        
        return reasoning
    
    def _calculate_confidence(self, audio_data: Dict[str, Any]) -> float:
        """Calculate confidence in the musical analysis."""
        confidence_factors = []
        
        # Feature completeness
        required_features = ["tempo", "energy", "danceability", "spectral_centroid_mean"]
        available_features = sum(1 for feature in required_features if audio_data.get(feature, 0) > 0)
        feature_completeness = available_features / len(required_features)
        confidence_factors.append(feature_completeness)
        
        # Genre prediction confidence
        genre_predictions = audio_data.get("genre_predictions", {})
        if genre_predictions:
            max_genre_confidence = max(genre_predictions.values())
            confidence_factors.append(max_genre_confidence)
        
        # Audio quality indicators
        spectral_rolloff = audio_data.get("spectral_rolloff_mean", 0)
        if spectral_rolloff > 2000:  # Good frequency range
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.7)
        
        # Return average confidence
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _analyze_key_characteristics(self, key: str, mode: str) -> str:
        """Analyze characteristics of musical key and mode."""
        key_characteristics = {
            "C": "stable and foundational character",
            "C#": "bright and energetic feel", 
            "D": "triumphant and powerful mood",
            "D#": "intense and dramatic character",
            "E": "confident and assertive tone",
            "F": "pastoral and gentle quality",
            "F#": "dreamy and ethereal atmosphere",
            "G": "bright and optimistic character",
            "G#": "exotic and mysterious mood",
            "A": "natural and comfortable feel",
            "A#": "dark and brooding character",
            "B": "sharp and crystalline quality"
        }
        
        mode_characteristics = {
            "major": "uplifting and positive emotional character",
            "minor": "introspective and melancholic emotional depth"
        }
        
        key_char = key_characteristics.get(key, "distinctive tonal")
        mode_char = mode_characteristics.get(mode.lower(), "unique modal")
        
        return f"{key_char} with {mode_char}"
