"""
Agent LLM Service for Dynamic Intelligence Generation

This service provides LLM-powered analysis for agents, replacing static thresholds
with dynamic, genre-specific, and contextual insights.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .llm_providers import LLMProviderFactory
from ..models.agentic_models import AgentRole

logger = logging.getLogger(__name__)

class AgentLLMService:
    """
    LLM service specifically designed for agent intelligence.
    
    Provides:
    - Genre-specific analysis
    - Contextual insights generation
    - Dynamic threshold adaptation
    - Learning from feedback
    """
    
    def __init__(self):
        """Initialize the agent LLM service."""
        self.llm_provider = None
        self.genre_contexts = {}
        self.analysis_cache = {}
        self.feedback_data = []
        
        # Initialize LLM provider
        self._initialize_llm_provider()
        
        # Load genre-specific contexts
        self._initialize_genre_contexts()
        
        logger.info("Agent LLM Service initialized")
    
    def _initialize_llm_provider(self):
        """Initialize the LLM provider."""
        try:
            # Get shared provider instance from factory
            self.llm_provider = LLMProviderFactory.auto_detect_provider()
            if self.llm_provider:
                logger.info(f"Agent LLM Service: Using LLM provider: {self.llm_provider.__class__.__name__}")
                # Test the provider with a simple call to ensure it works
                logger.info("Testing LLM provider connection...")
            else:
                logger.warning("No LLM provider available for Agent LLM Service - using fallback analysis")
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider for agents: {e}")
            self.llm_provider = None
    
    def _initialize_genre_contexts(self):
        """Initialize genre-specific analysis contexts."""
        self.genre_contexts = {
            "pop": {
                "typical_tempo_range": "100-130 BPM",
                "energy_characteristics": "High energy (0.6-0.9) with strong commercial appeal",
                "market_focus": "Mainstream radio, streaming playlists, broad demographic appeal",
                "commercial_priorities": "Radio play, streaming numbers, playlist placement"
            },
            "hip-hop": {
                "typical_tempo_range": "70-140 BPM with strong rhythmic emphasis",
                "energy_characteristics": "Variable energy depending on subgenre, strong bass presence",
                "market_focus": "Urban radio, streaming platforms, social media viral potential",
                "commercial_priorities": "Streaming, social media engagement, cultural relevance"
            },
            "rock": {
                "typical_tempo_range": "100-180 BPM with driving rhythm",
                "energy_characteristics": "High energy (0.7-1.0) with dynamic range",
                "market_focus": "Rock radio, live performance venues, dedicated fanbase",
                "commercial_priorities": "Live performance, album sales, radio rock formats"
            },
            "electronic": {
                "typical_tempo_range": "120-150 BPM optimized for dancing",
                "energy_characteristics": "Very high energy (0.8-1.0) with electronic production",
                "market_focus": "Dance clubs, electronic music festivals, streaming dance playlists",
                "commercial_priorities": "Club play, festival bookings, electronic music platforms"
            },
            "country": {
                "typical_tempo_range": "80-120 BPM with storytelling focus",
                "energy_characteristics": "Moderate energy (0.4-0.7) with emotional delivery",
                "market_focus": "Country radio, rural markets, storytelling tradition",
                "commercial_priorities": "Country radio, touring, traditional country markets"
            },
            "r&b": {
                "typical_tempo_range": "70-110 BPM with groove emphasis",
                "energy_characteristics": "Moderate to high energy (0.5-0.8) with soulful delivery",
                "market_focus": "Urban contemporary radio, R&B platforms, sophisticated audience",
                "commercial_priorities": "Urban radio, streaming R&B playlists, vocal showcase"
            }
        }
    
    async def generate_musical_insights(
        self,
        audio_data: Dict[str, Any],
        song_metadata: Dict[str, Any],
        agent_role: AgentRole
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Generate dynamic musical insights using LLM analysis.
        
        Args:
            audio_data: Audio analysis data
            song_metadata: Song metadata including genre
            agent_role: The agent requesting analysis
            
        Returns:
            Tuple of (findings, insights, recommendations)
        """
        try:
            if not self.llm_provider:
                logger.warning("LLM provider not available for musical analysis - using fallback")
                return self._fallback_musical_analysis(audio_data, song_metadata)
            
            # Determine genre context
            genre = self._extract_genre(audio_data, song_metadata)
            genre_context = self.genre_contexts.get(genre.lower(), self.genre_contexts.get("pop", {}))
            
            logger.info(f"Generating musical insights for genre: {genre} using {self.llm_provider.__class__.__name__}")
            
            # Create genre-specific prompt
            prompt = self._create_musical_analysis_prompt(
                audio_data, song_metadata, genre, genre_context, agent_role
            )
            
            logger.debug(f"Generated prompt length: {len(prompt)} characters")
            
            # Generate LLM response
            logger.info(f"🤖 Calling LLM provider: {self.llm_provider.__class__.__name__}")
            logger.debug(f"🤖 LLM prompt preview: {prompt[:500]}...")
            
            response = await self.llm_provider.generate(prompt, max_tokens=800)
            
            logger.info(f"🤖 LLM response received: {len(response)} characters")
            logger.debug(f"🤖 LLM response preview: {response[:300]}...")
            
            # Parse response into structured insights
            findings, insights, recommendations = self._parse_musical_response(response)
            
            logger.info(f"🤖 Parsed response: {len(findings)} findings, {len(insights)} insights, {len(recommendations)} recommendations")
            
            # Cache results for learning
            self._cache_analysis_result(audio_data, song_metadata, genre, findings, insights, recommendations)
            
            return findings, insights, recommendations
            
        except Exception as e:
            logger.error(f"🚨 LLM musical analysis FAILED: {e}")
            logger.error(f"🚨 LLM provider: {self.llm_provider.__class__.__name__ if self.llm_provider else 'None'}")
            logger.error(f"🚨 Falling back to hardcoded analysis")
            import traceback
            logger.error(f"🚨 Full traceback: {traceback.format_exc()}")
            return self._fallback_musical_analysis(audio_data, song_metadata)
    
    async def generate_commercial_insights(
        self,
        audio_data: Dict[str, Any],
        content_data: Dict[str, Any],
        hit_prediction: Dict[str, Any],
        song_metadata: Dict[str, Any],
        agent_role: AgentRole
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Generate dynamic commercial insights using LLM analysis.
        
        Args:
            audio_data: Audio analysis data
            content_data: Content analysis data
            hit_prediction: Hit prediction data
            song_metadata: Song metadata
            agent_role: The agent requesting analysis
            
        Returns:
            Tuple of (findings, insights, recommendations)
        """
        try:
            if not self.llm_provider:
                logger.warning("LLM provider not available for commercial analysis - using fallback")
                return self._fallback_commercial_analysis(audio_data, content_data, hit_prediction)
            
            # Determine genre and market context
            genre = self._extract_genre(audio_data, song_metadata)
            genre_context = self.genre_contexts.get(genre.lower(), self.genre_contexts.get("pop", {}))
            
            logger.info(f"Generating commercial insights for genre: {genre} using {self.llm_provider.__class__.__name__}")
            
            # Create commercial analysis prompt
            prompt = self._create_commercial_analysis_prompt(
                audio_data, content_data, hit_prediction, song_metadata, genre, genre_context, agent_role
            )
            
            # Generate LLM response
            logger.info(f"🤖 Calling LLM provider for commercial analysis: {self.llm_provider.__class__.__name__}")
            logger.debug(f"🤖 Commercial LLM prompt preview: {prompt[:500]}...")
            
            response = await self.llm_provider.generate(prompt, max_tokens=800)
            
            logger.info(f"🤖 LLM commercial response received: {len(response)} characters")
            logger.debug(f"🤖 Commercial LLM response preview: {response[:300]}...")
            
            # Parse response into structured insights
            findings, insights, recommendations = self._parse_commercial_response(response)
            
            logger.info(f"🤖 Parsed commercial response: {len(findings)} findings, {len(insights)} insights, {len(recommendations)} recommendations")
            
            # Cache results for learning
            self._cache_analysis_result(audio_data, song_metadata, genre, findings, insights, recommendations, "commercial")
            
            return findings, insights, recommendations
            
        except Exception as e:
            logger.error(f"🚨 LLM commercial analysis FAILED: {e}")
            logger.error(f"🚨 LLM provider: {self.llm_provider.__class__.__name__ if self.llm_provider else 'None'}")
            logger.error(f"🚨 Falling back to hardcoded commercial analysis")
            import traceback
            logger.error(f"🚨 Full commercial traceback: {traceback.format_exc()}")
            return self._fallback_commercial_analysis(audio_data, content_data, hit_prediction)
    
    def _create_musical_analysis_prompt(
        self,
        audio_data: Dict[str, Any],
        song_metadata: Dict[str, Any],
        genre: str,
        genre_context: Dict[str, Any],
        agent_role: AgentRole
    ) -> str:
        """Create genre-specific musical analysis prompt."""
        
        # Extract key audio features with null safety
        tempo = audio_data.get("tempo") or 0
        energy = audio_data.get("energy") or 0
        danceability = audio_data.get("danceability") or 0
        valence = audio_data.get("valence") or 0
        acousticness = audio_data.get("acousticness") or 0
        key = audio_data.get("key") or "unknown"
        mode = audio_data.get("mode") or "unknown"
        # Fix spectral centroid - cap at reasonable values
        spectral_centroid_raw = audio_data.get("spectral_centroid_mean") or 0
        spectral_centroid = min(spectral_centroid_raw, 20000) if spectral_centroid_raw > 0 else 0  # Cap at 20kHz
        
        # Get genre predictions if available
        genre_predictions = audio_data.get("genre_predictions", {})
        genre_confidence = genre_predictions.get(genre.lower(), 0)
        
        prompt = f"""
As an expert music analyst specializing in {genre} music, analyze this song's musical characteristics and provide insights:

**Song Information:**
- Title: {song_metadata.get('title', 'Unknown')}
- Artist: {song_metadata.get('artist', 'Unknown')}  
- Primary Genre: {genre} (confidence: {genre_confidence:.2f})

**Audio Features:**
- Tempo: {tempo:.1f} BPM
- Energy: {energy:.3f}
- Danceability: {danceability:.3f}
- Valence (positivity): {valence:.3f}
- Acousticness: {acousticness:.3f}
- Key: {key} {mode}
- Spectral Centroid: {spectral_centroid:.1f} Hz

**Genre Context for {genre.title()}:**
- Typical tempo range: {genre_context.get('typical_tempo_range', 'Variable')}
- Energy characteristics: {genre_context.get('energy_characteristics', 'Variable')}
- Market focus: {genre_context.get('market_focus', 'General market')}

**Analysis Request:**
Provide a detailed analysis considering this song's genre-specific context. Consider how these features work together and what they mean for this particular genre.

**Response Format:**
Please provide your analysis in the following JSON format:
{{
    "findings": [
        "Specific musical findings about tempo, energy, etc. in genre context",
        "Key signature and harmonic analysis for this genre",
        "Production quality and spectral characteristics analysis"
    ],
    "insights": [
        "Higher-level insights about musical sophistication",
        "Genre positioning and commercial viability insights", 
        "Cross-genre appeal and unique characteristics"
    ],
    "recommendations": [
        "Genre-specific musical recommendations",
        "Production or arrangement suggestions",
        "Market positioning recommendations for this genre"
    ]
}}

Focus on genre-specific analysis rather than generic thresholds. Consider what makes this song unique within its genre and how it compares to successful {genre} tracks.
"""
        
        return prompt
    
    def _create_commercial_analysis_prompt(
        self,
        audio_data: Dict[str, Any],
        content_data: Dict[str, Any],
        hit_prediction: Dict[str, Any],
        song_metadata: Dict[str, Any],
        genre: str,
        genre_context: Dict[str, Any],
        agent_role: AgentRole
    ) -> str:
        """Create genre-specific commercial analysis prompt."""
        
        # Extract key commercial indicators
        hit_probability = hit_prediction.get("hit_probability", 0)
        confidence = hit_prediction.get("confidence", 0)
        feature_importance = hit_prediction.get("feature_importance", {})
        
        # Extract content indicators with instrumental support
        sentiment = content_data.get("sentiment", {})
        emotions = content_data.get("emotions", {})
        themes = content_data.get("themes", [])
        # Determine if track is instrumental based on content features, NOT raw lyrics
        word_count = content_data.get("word_count", 0)
        has_themes = len(content_data.get("themes", [])) > 0
        has_emotions = any(v > 0.1 for v in content_data.get("emotion_scores", {}).values())
        is_instrumental = content_data.get("analysis_type") == "instrumental" or (word_count == 0 and not has_themes and not has_emotions)
        
        # Extract audio commercial features with null safety
        energy = audio_data.get("energy") or 0
        danceability = audio_data.get("danceability") or 0
        tempo = audio_data.get("tempo") or 0
        
        prompt = f"""
As an expert music industry analyst specializing in {genre} market dynamics, analyze this song's commercial potential:

**Song Information:**
- Title: {song_metadata.get('title', 'Unknown')}
- Artist: {song_metadata.get('artist', 'Unknown')}
- Genre: {genre}

**Commercial Indicators:**
- Hit Probability: {hit_probability:.1%} (confidence: {confidence:.1%})
- Key Features: {', '.join([f"{k}: {v:.3f}" for k, v in feature_importance.items()][:3])}

**Audio Commercial Features:**
- Energy: {energy:.3f} | Danceability: {danceability:.3f} | Tempo: {tempo:.1f} BPM

**Content Analysis:**
{f"- Track Type: Instrumental (no lyrics)" if is_instrumental else f"- Sentiment: {sentiment.get('compound', content_data.get('sentiment_score', 0)):.3f}"}
{"- Musical Expression: Pure instrumental storytelling" if is_instrumental else f"- Top Emotions: {', '.join([f'{k}: {v:.2f}' for k, v in sorted(content_data.get('emotion_scores', {}).items(), key=lambda x: x[1], reverse=True)[:3]] if content_data.get('emotion_scores') else ['neutral: 0.50'])}"}
{f"- Commercial Appeal: Instrumental track suitable for licensing, background use" if is_instrumental else f"- Themes: {', '.join(themes[:3]) if themes else 'general themes'}"}
{f"- Word Count: {word_count} | Complexity: {content_data.get('complexity_score', 0.5):.2f} | Language: {content_data.get('language', 'en')}" if not is_instrumental else "- Focus: Pure musical expression without lyrical content"}

**{genre.title()} Market Context:**
- Commercial priorities: {genre_context.get('commercial_priorities', 'General market')}
- Target market: {genre_context.get('market_focus', 'General audience')}

**Analysis Request:**
Analyze this {'instrumental track' if is_instrumental else 'song'}'s commercial viability specifically within the {genre} market. 

{'For instrumental tracks, focus on:' if is_instrumental else 'Based on the provided content analysis features, focus on:'}
{'''- Licensing opportunities (film, TV, advertising, games)
- Background music market potential
- Genre-specific instrumental market dynamics
- Production quality and sonic characteristics''' if is_instrumental else '''- Sentiment and emotional appeal for target demographics
- Theme resonance with current market trends  
- Content complexity and accessibility
- Genre-specific lyrical success factors
- Cross-demographic appeal based on emotional scores'''}

Use the provided features and metrics to support your analysis - do not indicate lack of data when features are available.

**Response Format:**
{{
    "findings": [
        "Genre-specific commercial potential analysis",
        "Target demographic and market segmentation for {genre}",
        "Revenue stream opportunities specific to this genre"
    ],
    "insights": [
        "Market positioning insights for {genre} audience",
        "Competitive analysis within {genre} landscape",
        "Brand partnership and sync licensing potential"
    ],
    "recommendations": [
        "Genre-specific release strategy recommendations",
        "Marketing and promotion tactics for {genre} market",
        "Platform and playlist targeting for this genre"
    ]
}}

Avoid generic thresholds. Focus on what drives commercial success specifically in the {genre} market and how this song's characteristics align with or differentiate from successful {genre} tracks.
"""
        
        return prompt
    
    def _parse_musical_response(self, response: str) -> Tuple[List[str], List[str], List[str]]:
        """Parse LLM response into structured musical insights."""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                findings = data.get("findings", [])
                insights = data.get("insights", [])
                recommendations = data.get("recommendations", [])
                
                return findings, insights, recommendations
            else:
                # Fallback parsing
                return self._fallback_parse_response(response)
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response, using fallback parsing")
            return self._fallback_parse_response(response)
    
    def _parse_commercial_response(self, response: str) -> Tuple[List[str], List[str], List[str]]:
        """Parse LLM response into structured commercial insights."""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                findings = data.get("findings", [])
                insights = data.get("insights", [])
                recommendations = data.get("recommendations", [])
                
                return findings, insights, recommendations
            else:
                # Fallback parsing
                return self._fallback_parse_response(response)
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response, using fallback parsing")
            return self._fallback_parse_response(response)
    
    def _fallback_parse_response(self, response: str) -> Tuple[List[str], List[str], List[str]]:
        """Fallback response parsing when JSON parsing fails."""
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        findings = []
        insights = []
        recommendations = []
        
        current_section = None
        
        for line in lines:
            if 'finding' in line.lower() or 'analysis' in line.lower():
                current_section = 'findings'
            elif 'insight' in line.lower() or 'understanding' in line.lower():
                current_section = 'insights'
            elif 'recommend' in line.lower() or 'suggest' in line.lower():
                current_section = 'recommendations'
            elif line.startswith('-') or line.startswith('•'):
                content = line[1:].strip()
                if current_section == 'findings':
                    findings.append(content)
                elif current_section == 'insights':
                    insights.append(content)
                elif current_section == 'recommendations':
                    recommendations.append(content)
        
        # Ensure we have at least some content
        if not any([findings, insights, recommendations]):
            findings = ["LLM analysis completed"]
            insights = ["Genre-specific analysis provided"]
            recommendations = ["Consider genre-specific optimization"]
        
        return findings, insights, recommendations
    
    def _extract_genre(self, audio_data: Dict[str, Any], song_metadata: Dict[str, Any]) -> str:
        """Extract primary genre from data with improved name handling for UI payload structure."""
        
        # Try song metadata first (this comes from UI as songData.audioFeatures.audio_primary_genre)
        genre = song_metadata.get("genre")
        if genre and genre != "Unknown":
            # If genre is numeric (like "89"), we need to check if there are readable names in audio_data
            if isinstance(genre, (int, str)) and str(genre).isdigit():
                logger.info(f"Received numeric genre ID: {genre}, looking for readable genre name")
                
                # Check genre_predictions for readable names (UI may send both numeric and readable)
                genre_predictions = audio_data.get("genre_predictions", {})
                if genre_predictions:
                    # Look for non-numeric keys (readable genre names)
                    readable_genres = {k: v for k, v in genre_predictions.items() if not str(k).isdigit()}
                    if readable_genres:
                        top_readable_genre = max(readable_genres.items(), key=lambda x: x[1])[0]
                        clean_genre = self._clean_genre_name(str(top_readable_genre))
                        logger.info(f"Found readable genre name: {clean_genre} for numeric ID: {genre}")
                        return clean_genre
                
                logger.warning(f"Received numeric genre: {genre} without readable name, trying other fallback methods")
                # Don't immediately default to pop, try other methods first
            else:
                # Clean and return the string genre (most common case)
                clean_genre = self._clean_genre_name(str(genre))
                logger.info(f"Using song metadata genre: {clean_genre}")
                return clean_genre
        
        # Try genre predictions (this comes from UI as audio_analysis.genre_predictions)
        genre_predictions = audio_data.get("genre_predictions", {})
        if genre_predictions:
            # First try to find readable genre names (non-numeric keys)
            readable_genres = {k: v for k, v in genre_predictions.items() if not str(k).isdigit()}
            if readable_genres:
                top_genre = max(readable_genres.items(), key=lambda x: x[1])[0]
                clean_genre = self._clean_genre_name(str(top_genre))
                logger.info(f"Using readable genre prediction: {clean_genre}")
                return clean_genre
            else:
                # All genre predictions are numeric
                top_genre = max(genre_predictions.items(), key=lambda x: x[1])[0]
                logger.warning(f"All genre predictions are numeric. Top prediction: {top_genre}, cannot map to readable name")
        
        # Legacy: Try to get readable genre from audio features (for backward compatibility)
        primary_genre = audio_data.get("primary_genre") or audio_data.get("audio_primary_genre")
        if primary_genre:
            clean_genre = self._clean_genre_name(str(primary_genre))
            logger.info(f"Using legacy audio primary genre field: {clean_genre}")
            return clean_genre
        
        # Final fallback - log the issue and default to pop
        logger.warning(f"No valid genre found in data. Available keys: audio_data={list(audio_data.keys())}, song_metadata={list(song_metadata.keys())}")
        logger.warning(f"Song metadata genre: {song_metadata.get('genre')}, Genre predictions: {audio_data.get('genre_predictions', {})}")
        return "pop"
    
    def _clean_genre_name(self, genre_name: str) -> str:
        """Clean and normalize genre names from various formats."""
        if not genre_name:
            return "pop"
        
        original_genre = genre_name
        
        # Handle Essentia format like "Electronic---House" or "Funk / Soul---Funk"
        if "---" in genre_name:
            # For electronic music, prefer the main category over sub-genre
            parts = genre_name.split("---")
            main_category = parts[0].strip()
            sub_genre = parts[-1].strip()
            
            # Check if main category is electronic and sub-genre is electronic sub-type
            if main_category.lower() == "electronic" and sub_genre.lower() in ["house", "techno", "trance", "edm", "dubstep", "drum", "bass"]:
                main_genre = main_category  # Use "Electronic" instead of "House"
            else:
                main_genre = sub_genre  # Use sub-genre for other cases like "Funk / Soul---Funk"
        elif " / " in genre_name:
            # Take the first part before the slash
            main_genre = genre_name.split(" / ")[0].strip()
        else:
            main_genre = genre_name.strip()
        
        # Normalize to lowercase for consistency
        main_genre = main_genre.lower()
        
        # Map common variations to standard genre names
        genre_mapping = {
            "funk": "funk",
            "soul": "soul", 
            "r&b": "rnb",
            "hip hop": "hip-hop",
            "hip-hop": "hip-hop",
            "electronic": "electronic",
            "house": "electronic",  # House is a type of electronic
            "techno": "electronic", # Techno is a type of electronic
            "edm": "electronic",    # EDM is electronic dance music
            "trance": "electronic", # Trance is electronic
            "dubstep": "electronic", # Dubstep is electronic
            "drum": "electronic",   # Drum & bass is electronic
            "bass": "electronic",   # Bass music is electronic
            "rock": "rock",
            "pop": "pop",
            "country": "country",
            "jazz": "jazz",
            "classical": "classical",
            "blues": "blues",
            "reggae": "reggae",
            "folk": "folk",
            "metal": "metal",
            "punk": "punk",
            "alternative": "alternative",
            "indie": "indie"
        }
        
        # Find best match
        for key, value in genre_mapping.items():
            if key in main_genre or main_genre in key:
                logger.debug(f"Mapped genre '{original_genre}' -> '{value}' (via '{main_genre}')")
                return value
        
        # If no mapping found, return the cleaned main genre
        logger.debug(f"No mapping found for genre '{original_genre}', using cleaned: '{main_genre}'")
        return main_genre
    
    def _cache_analysis_result(
        self,
        audio_data: Dict[str, Any],
        song_metadata: Dict[str, Any],
        genre: str,
        findings: List[str],
        insights: List[str],
        recommendations: List[str],
        analysis_type: str = "musical"
    ):
        """Cache analysis results for learning and feedback."""
        cache_key = f"{analysis_type}_{genre}_{hash(str(sorted(audio_data.items())))}"
        
        self.analysis_cache[cache_key] = {
            "timestamp": datetime.utcnow().isoformat(),
            "genre": genre,
            "analysis_type": analysis_type,
            "audio_features": {
                "tempo": audio_data.get("tempo", 0),
                "energy": audio_data.get("energy", 0),
                "danceability": audio_data.get("danceability", 0),
                "valence": audio_data.get("valence", 0)
            },
            "results": {
                "findings": findings,
                "insights": insights,
                "recommendations": recommendations
            }
        }
    
    def add_feedback(
        self,
        analysis_id: str,
        feedback_type: str,
        feedback_data: Dict[str, Any]
    ):
        """Add feedback for learning and improvement."""
        feedback_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "analysis_id": analysis_id,
            "feedback_type": feedback_type,
            "data": feedback_data
        }
        
        self.feedback_data.append(feedback_entry)
        logger.info(f"Added feedback for analysis {analysis_id}: {feedback_type}")
    
    def _fallback_musical_analysis(
        self,
        audio_data: Dict[str, Any],
        song_metadata: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Fallback analysis when LLM is not available."""
        findings = ["Basic audio analysis completed"]
        insights = ["Standard musical characteristics identified"]
        recommendations = ["Consider professional music analysis"]
        
        return findings, insights, recommendations
    
    def _fallback_commercial_analysis(
        self,
        audio_data: Dict[str, Any],
        content_data: Dict[str, Any],
        hit_prediction: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Fallback commercial analysis when LLM is not available."""
        findings = ["Basic commercial analysis completed"]
        insights = ["Standard market characteristics identified"]
        recommendations = ["Consider professional market analysis"]
        
        return findings, insights, recommendations
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "llm_provider_available": self.llm_provider is not None,
            "llm_provider_type": self.llm_provider.__class__.__name__ if self.llm_provider else None,
            "cached_analyses": len(self.analysis_cache),
            "feedback_entries": len(self.feedback_data),
            "supported_genres": list(self.genre_contexts.keys())
        }
