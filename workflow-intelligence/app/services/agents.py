"""
AI Agents for workflow-intelligence

This module implements different AI agent types for music analysis with multi-provider support.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import asyncio

from ..models.intelligence import AgentType, AIProvider
from ..config.settings import settings
from .llm_providers import LLMProviderFactory, BaseLLMProvider

logger = logging.getLogger(__name__)

class BaseIntelligenceAgent(ABC):
    """Base class for AI intelligence agents."""
    
    def __init__(self, provider: BaseLLMProvider, agent_config: Dict[str, Any] = None):
        """Initialize the agent with an LLM provider."""
        self.provider = provider
        self.config = agent_config or {}
        self.provider_name = provider.__class__.__name__ if provider else "unknown"
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.provider_name}")
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate insights using the agent's approach."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "agent_type": self.__class__.__name__,
            "provider": self.provider_name,
            "provider_available": self.provider.is_available() if self.provider else False,
            "config": self.config
        }

class StandardAgent(BaseIntelligenceAgent):
    """
    Standard AI agent for general music analysis.
    
    Provides balanced analysis with good performance and reasonable cost.
    """
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate insights using standard approach."""
        if not self.provider or not self.provider.is_available():
            raise RuntimeError("Provider not available for StandardAgent")
        
        try:
            # Use standard parameters for cost-effectiveness
            generation_kwargs = {
                "max_tokens": kwargs.get("max_tokens", 1500),
                "temperature": kwargs.get("temperature", 0.7),
                **kwargs
            }
            
            self.logger.debug(f"Generating with parameters: {generation_kwargs}")
            
            response = await self.provider.generate(prompt, **generation_kwargs)
            
            if not response or not response.strip():
                raise ValueError("Empty response from provider")
            
            self.logger.info(f"Generated {len(response)} characters of insight")
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error in StandardAgent generation: {e}")
            raise

class ComprehensiveAgent(BaseIntelligenceAgent):
    """
    Comprehensive AI agent for deep music analysis.
    
    Provides detailed, multi-perspective analysis with higher cost but better quality.
    """
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate insights using comprehensive approach."""
        if not self.provider or not self.provider.is_available():
            raise RuntimeError("Provider not available for ComprehensiveAgent")
        
        try:
            # Use enhanced parameters for higher quality
            generation_kwargs = {
                "max_tokens": kwargs.get("max_tokens", 3000),
                "temperature": kwargs.get("temperature", 0.6),  # Slightly lower for more focused output
                **kwargs
            }
            
            # For comprehensive analysis, we might use multiple passes
            if self.config.get("multi_pass", False):
                return await self._multi_pass_generation(prompt, **generation_kwargs)
            else:
                return await self._single_pass_generation(prompt, **generation_kwargs)
            
        except Exception as e:
            self.logger.error(f"Error in ComprehensiveAgent generation: {e}")
            raise
    
    async def _single_pass_generation(self, prompt: str, **kwargs) -> str:
        """Single-pass generation for comprehensive analysis."""
        # Enhance the prompt for more comprehensive analysis
        enhanced_prompt = f"""
{prompt}

Please provide a comprehensive and detailed analysis. Consider multiple perspectives and provide specific, actionable insights. 
Include relevant examples and comparisons where appropriate. Focus on depth and nuance in your analysis.
"""
        
        response = await self.provider.generate(enhanced_prompt, **kwargs)
        
        if not response or not response.strip():
            raise ValueError("Empty response from provider")
        
        self.logger.info(f"Generated comprehensive analysis: {len(response)} characters")
        return response.strip()
    
    async def _multi_pass_generation(self, prompt: str, **kwargs) -> str:
        """Multi-pass generation for even more comprehensive analysis."""
        self.logger.info("Using multi-pass generation for comprehensive analysis")
        
        # First pass: initial analysis
        initial_response = await self.provider.generate(prompt, **kwargs)
        
        # Second pass: refine and expand
        refinement_prompt = f"""
Based on this initial analysis:

{initial_response}

Please expand and refine this analysis with additional insights, considering:
1. Alternative perspectives that may have been missed
2. Deeper implications of the findings
3. More specific actionable recommendations
4. Additional context that could be valuable

Original prompt context:
{prompt[:1000]}...
"""
        
        refined_response = await self.provider.generate(
            refinement_prompt, 
            max_tokens=kwargs.get("max_tokens", 2000)
        )
        
        # Combine responses intelligently
        combined_response = f"{initial_response}\n\n--- Enhanced Analysis ---\n\n{refined_response}"
        
        self.logger.info(f"Generated multi-pass analysis: {len(combined_response)} characters")
        return combined_response

class MusicAnalysisAgent(BaseIntelligenceAgent):
    """
    Specialized Music Analysis Agent with domain expertise.
    
    Incorporates music theory, industry knowledge, and commercial analysis capabilities
    similar to the original ai_agent_cmp.py implementation.
    """
    
    def __init__(self, provider: BaseLLMProvider, agent_config: Dict[str, Any] = None):
        """Initialize the Music Analysis Agent."""
        super().__init__(provider, agent_config)
        
        # Music theory and industry knowledge
        self.genre_patterns = {
            "pop": {"tempo_range": (90, 140), "valence_target": 0.6, "energy_target": 0.7},
            "rock": {"tempo_range": (100, 160), "valence_target": 0.5, "energy_target": 0.8},
            "hip-hop": {"tempo_range": (70, 140), "valence_target": 0.5, "energy_target": 0.7},
            "electronic": {"tempo_range": (120, 140), "valence_target": 0.6, "energy_target": 0.8},
            "country": {"tempo_range": (80, 130), "valence_target": 0.5, "energy_target": 0.6}
        }
        
        # Commercial viability factors
        self.commercial_factors = {
            "danceability": {"weight": 0.25, "optimal": 0.6},
            "energy": {"weight": 0.20, "optimal": 0.7},
            "valence": {"weight": 0.20, "optimal": 0.6},
            "tempo": {"weight": 0.15, "optimal": 120},
            "popularity_indicators": {"weight": 0.20}
        }
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate music-specific insights using domain expertise."""
        if not self.provider or not self.provider.is_available():
            raise RuntimeError("Provider not available for MusicAnalysisAgent")
        
        # Get analysis type and data from kwargs
        analysis_type = kwargs.get("analysis_type", "general")
        song_data = kwargs.get("song_data", {})
        
        # Enhance prompt with music domain expertise
        enhanced_prompt = self._enhance_prompt_with_domain_knowledge(
            prompt, analysis_type, song_data
        )
        
        try:
            generation_kwargs = {
                "max_tokens": kwargs.get("max_tokens", 2500),
                "temperature": kwargs.get("temperature", 0.65),
            }
            
            response = await self.provider.generate(enhanced_prompt, **generation_kwargs)
            
            if not response or not response.strip():
                raise ValueError("Empty response from provider")
            
            self.logger.info(f"Generated music analysis: {len(response)} characters")
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error in MusicAnalysisAgent generation: {e}")
            raise
    
    def _enhance_prompt_with_domain_knowledge(self, prompt: str, analysis_type: str, song_data: Dict[str, Any]) -> str:
        """Enhance prompts with music theory and industry knowledge."""
        
        # Extract key features for context
        audio_features = song_data.get("audio_analysis", {})
        content_features = song_data.get("content_analysis", {})
        hit_potential = song_data.get("hit_prediction", {})
        
        # Build domain context
        domain_context = self._build_music_domain_context(audio_features, content_features)
        
        if analysis_type == "musical_meaning":
            return self._create_musical_meaning_prompt(prompt, domain_context, audio_features)
        elif analysis_type == "hit_comparison":
            return self._create_hit_comparison_prompt(prompt, domain_context, audio_features, hit_potential)
        elif analysis_type == "novelty_assessment":
            return self._create_novelty_assessment_prompt(prompt, domain_context, audio_features, content_features)
        elif analysis_type == "strategic_insights":
            return self._create_strategic_insights_prompt(prompt, domain_context, audio_features, content_features, hit_potential)
        else:
            return f"{prompt}\n\n{domain_context}"
    
    def _build_music_domain_context(self, audio_features: Dict[str, Any], content_features: Dict[str, Any]) -> str:
        """Build comprehensive music domain context."""
        context_parts = [
            "MUSIC ANALYSIS DOMAIN EXPERTISE:",
            "You are an expert music analyst, producer, and A&R specialist with deep knowledge of:",
            "• Music theory (harmony, rhythm, melody, structure)",
            "• Commercial music production and mixing standards", 
            "• Genre-specific patterns and market trends",
            "• Hit song analysis and commercial viability assessment",
            "• Lyrical analysis and narrative structure evaluation",
            "",
            "ANALYSIS FRAMEWORK:",
            "• Use quantitative data to support qualitative insights",
            "• Reference industry standards and genre norms",
            "• Provide actionable recommendations for improvement",
            "• Consider both artistic merit and commercial potential",
            "• Focus on specific, measurable characteristics",
            "",
        ]
        
        # Add feature-specific context
        if audio_features:
            tempo = audio_features.get("audio_tempo", 0)
            energy = audio_features.get("audio_energy", 0)
            valence = audio_features.get("audio_valence", 0)
            
            context_parts.extend([
                f"CURRENT SONG CHARACTERISTICS:",
                f"• Tempo: {tempo} BPM ({'moderate' if 90 <= tempo <= 140 else 'fast' if tempo > 140 else 'slow'} pace)",
                f"• Energy: {energy:.2f} ({'high' if energy > 0.7 else 'moderate' if energy > 0.4 else 'low'} energy)",
                f"• Valence: {valence:.2f} ({'positive' if valence > 0.6 else 'neutral' if valence > 0.4 else 'negative'} mood)",
                ""
            ])
        
        return "\n".join(context_parts)
    
    def _create_musical_meaning_prompt(self, base_prompt: str, context: str, audio_features: Dict[str, Any]) -> str:
        """Create specialized musical meaning analysis prompt."""
        return f"""
{context}

MUSICAL MEANING ANALYSIS TASK:
Analyze the musical characteristics to extract deep meaning about the song's emotional and artistic impact.

{base_prompt}

Focus on these specific aspects:
1. TEMPO CHARACTERISTICS: Analyze {audio_features.get('audio_tempo', 'unknown')} BPM in context of genre and emotional impact
2. ENERGY PROFILE: How does {audio_features.get('audio_energy', 'unknown')} energy level affect listener experience?
3. EMOTIONAL LANDSCAPE: What mood does {audio_features.get('audio_valence', 'unknown')} valence create?
4. HARMONIC CONTENT: Analyze key, mode, and harmonic complexity
5. RHYTHMIC QUALITIES: Danceability and groove characteristics
6. SONIC CHARACTER: Production style, mix qualities, and timbral choices

Provide insights that would be valuable to:
• Music producers planning arrangement decisions
• A&R executives evaluating commercial potential
• Artists seeking to understand their creative direction

Format your response with clear sections and specific, actionable insights.
"""

    def _create_hit_comparison_prompt(self, base_prompt: str, context: str, audio_features: Dict[str, Any], hit_data: Dict[str, Any]) -> str:
        """Create hit pattern comparison prompt with industry data."""
        hit_score = hit_data.get("prediction", 0) if hit_data else 0
        
        return f"""
{context}

HIT PATTERN COMPARISON ANALYSIS:
Compare this song's characteristics against successful hit patterns and industry benchmarks.

Current Hit Potential: {hit_score:.1%}

{base_prompt}

ANALYSIS FRAMEWORK:
1. GENRE BENCHMARKING: Compare features to successful songs in the same genre
2. COMMERCIAL FACTORS: Assess danceability, energy, and catchiness metrics
3. STATISTICAL ANALYSIS: Identify how features align with or deviate from hit patterns
4. MARKET POSITIONING: Where does this song fit in the current music landscape?
5. IMPROVEMENT OPPORTUNITIES: Specific recommendations to increase hit potential

COMMERCIAL VIABILITY FACTORS:
• Danceability: {audio_features.get('audio_danceability', 'unknown')} (target: 0.6+)
• Energy: {audio_features.get('audio_energy', 'unknown')} (target: 0.7+)
• Valence: {audio_features.get('audio_valence', 'unknown')} (target: genre-dependent)
• Tempo: {audio_features.get('audio_tempo', 'unknown')} BPM (optimal: 120-130 BPM)

Provide specific recommendations that an A&R executive or producer could implement to increase commercial viability.
"""

    def _create_novelty_assessment_prompt(self, base_prompt: str, context: str, audio_features: Dict[str, Any], content_features: Dict[str, Any]) -> str:
        """Create novelty and originality assessment prompt."""
        return f"""
{context}

NOVELTY & ORIGINALITY ASSESSMENT:
Evaluate the song's uniqueness and innovation potential in the current music landscape.

{base_prompt}

ASSESSMENT CRITERIA:
1. MUSICAL ORIGINALITY (0-1 scale):
   • Harmonic progression uniqueness
   • Rhythmic pattern innovation
   • Melodic structure creativity
   • Production technique innovation

2. LYRICAL UNIQUENESS (0-1 scale):
   • Thematic originality
   • Narrative structure innovation
   • Vocabulary and language use
   • Metaphorical creativity

3. GENRE-BENDING POTENTIAL:
   • Cross-genre elements identified
   • Innovation within genre constraints
   • Market differentiation factors

4. COMMERCIAL IMPLICATIONS:
   • How uniqueness affects marketability
   • Target audience identification
   • Competitive advantages

Provide specific scores (0.0-1.0) for each category and explain the reasoning behind each score.
Include recommendations for enhancing originality while maintaining commercial appeal.
"""

    def _create_strategic_insights_prompt(self, base_prompt: str, context: str, audio_features: Dict[str, Any], content_features: Dict[str, Any], hit_data: Dict[str, Any]) -> str:
        """Create strategic business insights prompt."""
        return f"""
{context}

STRATEGIC BUSINESS INSIGHTS:
Provide actionable intelligence for music industry decision-making.

{base_prompt}

STRATEGIC ANALYSIS AREAS:
1. MARKET OPPORTUNITY:
   • Target demographic identification
   • Playlist placement potential
   • Radio format compatibility
   • Streaming platform optimization

2. RELEASE STRATEGY:
   • Optimal release timing
   • Marketing angle recommendations
   • Promotional partnership opportunities
   • Social media strategy alignment

3. MONETIZATION POTENTIAL:
   • Revenue stream optimization
   • Sync licensing opportunities
   • Live performance viability
   • Merchandise and branding potential

4. RISK ASSESSMENT:
   • Market saturation concerns
   • Genre trend alignment
   • Competitive landscape analysis
   • Investment recommendation

DECISION FRAMEWORK:
Should this song be:
[ ] Released immediately with major promotion
[ ] Developed further before release
[ ] Released as part of a strategic album rollout
[ ] Considered for specific licensing opportunities
[ ] Reworked to improve commercial viability

Provide specific, actionable recommendations with timeline and resource requirements.
"""

class IntelligenceAgentFactory:
    """Factory for creating intelligence agents."""
    
    _agent_classes = {
        AgentType.STANDARD: StandardAgent,
        AgentType.COMPREHENSIVE: ComprehensiveAgent,
        AgentType.MUSIC_ANALYSIS: MusicAnalysisAgent
    }
    
    def __init__(self):
        """Initialize the agent factory."""
        self.logger = logging.getLogger(__name__)
    
    async def create_agent(
        self, 
        agent_type: str, 
        provider_preference: str = "auto",
        agent_config: Dict[str, Any] = None
    ) -> Optional[BaseIntelligenceAgent]:
        """
        Create an intelligence agent.
        
        Args:
            agent_type: Type of agent to create
            provider_preference: Preferred LLM provider
            agent_config: Additional agent configuration
            
        Returns:
            Intelligence agent instance or None if creation fails
        """
        try:
            # Validate agent type
            if agent_type not in [at.value for at in self._agent_classes.keys()]:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            agent_enum = AgentType(agent_type)
            
            # Get LLM provider
            provider = await self._get_provider(provider_preference)
            if not provider:
                self.logger.error("No LLM provider available for agent creation")
                return None
            
            # Get agent configuration
            config = self._get_agent_config(agent_enum, agent_config)
            
            # Create agent
            agent_class = self._agent_classes[agent_enum]
            agent = agent_class(provider=provider, agent_config=config)
            
            self.logger.info(f"Created {agent_type} agent with {provider.__class__.__name__} provider")
            return agent
            
        except Exception as e:
            self.logger.error(f"Error creating agent: {e}")
            return None
    
    async def _get_provider(self, provider_preference: str) -> Optional[BaseLLMProvider]:
        """Get LLM provider based on preference."""
        try:
            if provider_preference == "auto":
                # Auto-detect best available provider
                provider = LLMProviderFactory.auto_detect_provider()
                if provider:
                    self.logger.info(f"Auto-detected provider: {provider.__class__.__name__}")
                return provider
            else:
                # Use specific provider
                provider_config = settings.get_llm_config(provider_preference)
                provider = LLMProviderFactory.create_provider(provider_preference, **provider_config)
                
                if provider and provider.is_available():
                    self.logger.info(f"Using requested provider: {provider_preference}")
                    return provider
                else:
                    self.logger.warning(f"Requested provider {provider_preference} not available, falling back to auto-detect")
                    return LLMProviderFactory.auto_detect_provider()
                    
        except Exception as e:
            self.logger.error(f"Error getting provider: {e}")
            return None
    
    def _get_agent_config(self, agent_type: AgentType, user_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get configuration for agent type."""
        base_configs = {
            AgentType.STANDARD: {
                "multi_pass": False,
                "max_retries": 2,
                "timeout_seconds": 60,
                "cost_optimization": True
            },
            AgentType.COMPREHENSIVE: {
                "multi_pass": settings.DEBUG,  # Enable multi-pass in debug mode
                "max_retries": 3,
                "timeout_seconds": 120,
                "cost_optimization": False
            }
        }
        
        config = base_configs.get(agent_type, {})
        
        # Merge with user config
        if user_config:
            config.update(user_config)
        
        return config
    
    def list_agent_types(self) -> List[str]:
        """List available agent types."""
        return [agent_type.value for agent_type in self._agent_classes.keys()]
    
    def get_agent_info(self, agent_type: str) -> Dict[str, Any]:
        """Get information about an agent type."""
        try:
            agent_enum = AgentType(agent_type)
            agent_class = self._agent_classes[agent_enum]
            
            return {
                "agent_type": agent_type,
                "class_name": agent_class.__name__,
                "description": agent_class.__doc__,
                "default_config": self._get_agent_config(agent_enum)
            }
        except Exception as e:
            return {"error": str(e)}

class AgentManager:
    """
    Manages multiple agents and provides agent selection and load balancing.
    """
    
    def __init__(self):
        """Initialize the agent manager."""
        self.active_agents: Dict[str, BaseIntelligenceAgent] = {}
        self.agent_factory = IntelligenceAgentFactory()
        self.logger = logging.getLogger(__name__)
    
    async def get_agent(
        self, 
        agent_type: str = "standard", 
        provider_preference: str = "auto"
    ) -> Optional[BaseIntelligenceAgent]:
        """
        Get or create an agent of the specified type.
        
        Args:
            agent_type: Type of agent needed
            provider_preference: Preferred LLM provider
            
        Returns:
            Agent instance or None if unavailable
        """
        agent_key = f"{agent_type}_{provider_preference}"
        
        # Check if we have an active agent
        if agent_key in self.active_agents:
            agent = self.active_agents[agent_key]
            if agent.provider and agent.provider.is_available():
                return agent
            else:
                # Remove inactive agent
                del self.active_agents[agent_key]
        
        # Create new agent
        agent = await self.agent_factory.create_agent(
            agent_type=agent_type,
            provider_preference=provider_preference
        )
        
        if agent:
            self.active_agents[agent_key] = agent
            self.logger.info(f"Created and cached new agent: {agent_key}")
        
        return agent
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all active agents."""
        health_status = {
            "active_agents": len(self.active_agents),
            "agents": {}
        }
        
        for agent_key, agent in self.active_agents.items():
            try:
                agent_info = agent.get_info()
                health_status["agents"][agent_key] = {
                    "status": "healthy" if agent_info["provider_available"] else "unhealthy",
                    "info": agent_info
                }
            except Exception as e:
                health_status["agents"][agent_key] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return health_status
    
    def clear_inactive_agents(self):
        """Remove agents with inactive providers."""
        inactive_keys = []
        
        for agent_key, agent in self.active_agents.items():
            if not agent.provider or not agent.provider.is_available():
                inactive_keys.append(agent_key)
        
        for key in inactive_keys:
            del self.active_agents[key]
            self.logger.info(f"Removed inactive agent: {key}")
        
        return len(inactive_keys)
    
    async def test_agent_generation(
        self, 
        agent_type: str = "standard", 
        provider_preference: str = "auto"
    ) -> Dict[str, Any]:
        """Test agent generation capability."""
        try:
            agent = await self.get_agent(agent_type, provider_preference)
            if not agent:
                return {"status": "error", "message": "Could not create agent"}
            
            test_prompt = "Briefly describe what makes a song emotionally compelling to listeners."
            
            start_time = asyncio.get_event_loop().time()
            response = await agent.generate(test_prompt, max_tokens=200)
            end_time = asyncio.get_event_loop().time()
            
            return {
                "status": "success",
                "agent_type": agent_type,
                "provider": agent.provider_name,
                "response_length": len(response),
                "response_time_ms": (end_time - start_time) * 1000,
                "test_response": response[:200] + "..." if len(response) > 200 else response
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "message": str(e),
                "agent_type": agent_type,
                "provider_preference": provider_preference
            } 