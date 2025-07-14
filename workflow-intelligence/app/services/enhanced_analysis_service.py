"""
Enhanced AI Analysis Service for Professional Music Insights

This service implements a sophisticated analysis engine that combines:
- Specialized analysis chains (Musical Meaning, Hit Comparison, Novelty, Production)
- Industry benchmarking with statistical analysis
- Artist-focused recommendations
- Professional report generation
"""

import logging
import json
import statistics
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from ..services.llm_providers import LLMProviderFactory
from ..models.intelligence import InsightGenerationRequest, InsightGenerationResponse

logger = logging.getLogger(__name__)

class AnalysisChainType(Enum):
    MUSICAL_MEANING = "musical_meaning"
    HIT_COMPARISON = "hit_comparison" 
    NOVELTY_ASSESSMENT = "novelty_assessment"
    PRODUCTION_ANALYSIS = "production_analysis"
    COMPREHENSIVE_SYNTHESIS = "comprehensive_synthesis"

@dataclass
class GenreBenchmark:
    """Statistical benchmarks for genre comparison"""
    feature_name: str
    mean: float
    std_dev: float
    min_val: float
    max_val: float
    optimal_range: Tuple[float, float]
    
    def calculate_z_score(self, value: float) -> float:
        """Calculate how many standard deviations away from mean"""
        return (value - self.mean) / self.std_dev if self.std_dev > 0 else 0.0
    
    def get_alignment_assessment(self, value: float) -> str:
        """Get text assessment of value alignment with genre norms"""
        z_score = self.calculate_z_score(value)
        if abs(z_score) < 0.5:
            return f"perfectly aligned with genre norm (mean: {self.mean:.2f})"
        elif abs(z_score) < 1.0:
            return f"slightly {'above' if z_score > 0 else 'below'} genre average ({z_score:+.1f}σ)"
        elif abs(z_score) < 2.0:
            return f"notably {'higher' if z_score > 0 else 'lower'} than typical ({z_score:+.1f}σ)"
        else:
            return f"significantly {'above' if z_score > 0 else 'below'} genre norm ({z_score:+.1f}σ)"

@dataclass
class ProductionStandards:
    """Industry production standards and targets"""
    streaming_lufs_target: float = -14.0
    radio_lufs_target: float = -23.0
    dynamic_range_minimum: float = 6.0
    dynamic_range_optimal: float = 12.0
    frequency_balance_tolerance: float = 3.0

class EnhancedAnalysisService:
    """
    Professional AI analysis service for music industry applications.
    
    Implements specialized analysis chains that provide actionable insights
    for artists, producers, and music industry professionals.
    """
    
    def __init__(self):
        self.llm_provider = LLMProviderFactory.auto_detect_provider()
        self.production_standards = ProductionStandards()
        self.genre_benchmarks = self._load_genre_benchmarks()
        self._initialize_analysis_chains()
    
    def _load_genre_benchmarks(self) -> Dict[str, Dict[str, GenreBenchmark]]:
        """Load statistical benchmarks for different genres"""
        # In production, this would load from a database or data file
        # For now, using representative values based on industry analysis
        benchmarks = {
            "pop": {
                "tempo": GenreBenchmark("tempo", 120.0, 15.0, 80.0, 180.0, (100.0, 140.0)),
                "energy": GenreBenchmark("energy", 0.67, 0.12, 0.2, 1.0, (0.55, 0.85)),
                "danceability": GenreBenchmark("danceability", 0.65, 0.15, 0.2, 1.0, (0.50, 0.80)),
                "valence": GenreBenchmark("valence", 0.55, 0.20, 0.0, 1.0, (0.40, 0.75)),
                "loudness": GenreBenchmark("loudness", -8.5, 2.5, -20.0, -3.0, (-11.0, -6.0)),
            },
            "rock": {
                "tempo": GenreBenchmark("tempo", 125.0, 20.0, 90.0, 200.0, (110.0, 150.0)),
                "energy": GenreBenchmark("energy", 0.78, 0.15, 0.3, 1.0, (0.65, 0.95)),
                "danceability": GenreBenchmark("danceability", 0.55, 0.18, 0.2, 1.0, (0.40, 0.75)),
                "valence": GenreBenchmark("valence", 0.60, 0.22, 0.0, 1.0, (0.45, 0.80)),
                "loudness": GenreBenchmark("loudness", -7.5, 2.0, -15.0, -3.0, (-9.5, -5.5)),
            },
            "electronic": {
                "tempo": GenreBenchmark("tempo", 128.0, 25.0, 70.0, 180.0, (120.0, 140.0)),
                "energy": GenreBenchmark("energy", 0.75, 0.18, 0.3, 1.0, (0.60, 0.90)),
                "danceability": GenreBenchmark("danceability", 0.72, 0.16, 0.3, 1.0, (0.60, 0.85)),
                "valence": GenreBenchmark("valence", 0.58, 0.25, 0.0, 1.0, (0.40, 0.80)),
                "loudness": GenreBenchmark("loudness", -7.0, 2.2, -14.0, -3.0, (-9.0, -5.0)),
            }
        }
        return benchmarks
    
    def _initialize_analysis_chains(self):
        """Initialize specialized analysis prompt templates"""
        self.analysis_prompts = {
            AnalysisChainType.MUSICAL_MEANING: self._create_musical_meaning_prompt(),
            AnalysisChainType.HIT_COMPARISON: self._create_hit_comparison_prompt(),
            AnalysisChainType.NOVELTY_ASSESSMENT: self._create_novelty_assessment_prompt(),
            AnalysisChainType.PRODUCTION_ANALYSIS: self._create_production_analysis_prompt(),
            AnalysisChainType.COMPREHENSIVE_SYNTHESIS: self._create_synthesis_prompt()
        }
    
    def _create_musical_meaning_prompt(self) -> str:
        """Create professional musical meaning analysis prompt"""
        return """
You are a Grammy-winning producer and musicologist analyzing songs for commercial release.

AUDIO FEATURES:
{audio_features}

GENRE: {genre}
SIMILAR HITS: {similar_hits}

ANALYSIS REQUIREMENTS:
Provide a deep musical meaning analysis focusing on:

1. EMOTIONAL LANDSCAPE
   - What emotional journey does this song create?
   - How do tempo, key, and energy interact to create mood?
   - What is the listener's emotional experience from start to finish?

2. SONIC CHARACTER
   - What is this song's sonic personality?
   - How do production choices support the artistic vision?
   - What makes this track distinctive in the sonic landscape?

3. GENRE POSITIONING
   - How does this fit within {genre} expectations?
   - What genre conventions does it follow/break?
   - How does it compare to current {genre} hits?

4. LISTENER EXPERIENCE
   - What scenarios is this song perfect for?
   - Who is the core audience and why?
   - What emotional needs does this fulfill?

5. ARTISTIC STRENGTHS
   - What are the strongest musical elements?
   - What hooks/elements will resonate most?
   - What gives this song commercial potential?

RESPONSE FORMAT:
Return a JSON object with this exact structure:
{{
  "emotional_landscape": {{
    "primary_emotion": "string",
    "emotional_arc": "string",
    "mood_descriptors": ["array", "of", "strings"],
    "listening_context": "string"
  }},
  "sonic_character": {{
    "personality": "string", 
    "production_style": "string",
    "distinctive_elements": ["array", "of", "strings"],
    "reference_artists": ["array", "of", "strings"]
  }},
  "genre_positioning": {{
    "genre_alignment": "string",
    "conventions_followed": ["array"],
    "innovative_elements": ["array"],
    "competitive_landscape": "string"
  }},
  "commercial_assessment": {{
    "target_audience": "string",
    "use_cases": ["array", "of", "scenarios"],
    "playlist_fit": ["array", "of", "playlist", "types"],
    "marketing_angle": "string"
  }}
}}

Focus on actionable insights that help artists and labels make informed decisions.
"""

    def _create_hit_comparison_prompt(self) -> str:
        """Create hit pattern comparison prompt with statistical analysis"""
        return """
You are a data scientist specializing in Hit Song Science with access to industry analytics.

SONG FEATURES:
{audio_features}

GENRE BENCHMARKS:
{genre_benchmarks}

STATISTICAL ANALYSIS:
{statistical_analysis}

MARKET CONTEXT:
Genre: {genre}
Current Chart Trends: {chart_trends}
Historical Success Patterns: {historical_patterns}

ANALYSIS REQUIREMENTS:
Perform statistical hit pattern analysis:

1. FEATURE ALIGNMENT ANALYSIS
   - Compare each feature to genre mean ± standard deviation
   - Calculate Z-scores and statistical significance
   - Identify features that help/hurt commercial potential

2. HIT PROBABILITY ASSESSMENT
   - Based on feature alignment with successful tracks
   - Consider genre-specific success patterns
   - Factor in current market trends

3. COMMERCIAL VIABILITY FACTORS
   - Radio-friendly characteristics
   - Streaming optimization factors
   - Playlist placement potential
   - Demographic appeal indicators

4. RISK/OPPORTUNITY ANALYSIS
   - Features that may limit mainstream appeal
   - Unique elements that could drive viral success
   - Market timing considerations

5. OPTIMIZATION RECOMMENDATIONS
   - Specific feature adjustments for better hit potential
   - Production tweaks aligned with successful patterns
   - Strategic positioning recommendations

RESPONSE FORMAT:
{{
  "statistical_analysis": {{
    "overall_hit_probability": "percentage",
    "confidence_level": "string",
    "key_success_factors": ["array"],
    "risk_factors": ["array"]
  }},
  "feature_analysis": {{
    "aligned_features": {{"feature": "assessment"}},
    "concerning_features": {{"feature": "concern"}},
    "opportunity_features": {{"feature": "opportunity"}}
  }},
  "market_positioning": {{
    "radio_potential": "assessment",
    "streaming_potential": "assessment", 
    "playlist_targets": ["array"],
    "demographic_appeal": "string"
  }},
  "optimization_strategy": {{
    "immediate_actions": ["array"],
    "strategic_positioning": "string",
    "timing_recommendations": "string"
  }}
}}

Base all assessments on statistical analysis and industry data patterns.
"""

    def _create_novelty_assessment_prompt(self) -> str:
        """Create novelty assessment prompt"""
        return """
You are an A&R executive evaluating artistic innovation and commercial differentiation.

MUSICAL DATA:
{audio_features}

LYRICS CONTENT:
{lyrics_content}

MARKET CONTEXT:
Genre: {genre}
Current Trends: {current_trends}
Competitor Analysis: {competitor_analysis}

INNOVATION ASSESSMENT:

1. MUSICAL INNOVATION SCORING (0-1 scale)
   - Melody/Harmony originality vs. genre norms
   - Rhythmic creativity and uniqueness
   - Production technique innovation
   - Arrangement distinctiveness

2. LYRICAL INNOVATION SCORING (0-1 scale)
   - Thematic originality
   - Narrative structure creativity
   - Language/metaphor uniqueness
   - Emotional authenticity

3. COMMERCIAL DIFFERENTIATION
   - What makes this stand out in the market?
   - Competitive advantages vs. similar artists
   - Potential for trendsetting
   - Viral/shareability factors

4. INNOVATION RISK ASSESSMENT
   - Too innovative for mainstream acceptance?
   - Missing familiar elements for accessibility?
   - Innovation-authenticity balance
   - Market readiness for these innovations

RESPONSE FORMAT:
{{
  "innovation_scores": {{
    "musical_innovation": 0.0,
    "lyrical_innovation": 0.0,
    "production_innovation": 0.0,
    "overall_innovation": 0.0
  }},
  "differentiation_analysis": {{
    "unique_selling_points": ["array"],
    "competitive_advantages": ["array"],
    "market_gaps_filled": ["array"],
    "trendsetting_potential": "assessment"
  }},
  "commercial_balance": {{
    "innovation_accessibility_balance": "assessment",
    "mainstream_appeal_factors": ["array"],
    "niche_appeal_factors": ["array"],
    "crossover_potential": "assessment"
  }},
  "strategic_recommendations": {{
    "leverage_innovation": ["array"],
    "accessibility_improvements": ["array"],
    "market_positioning": "string"
  }}
}}
"""

    def _create_production_analysis_prompt(self) -> str:
        """Create production quality analysis prompt"""
        return """
You are a Grammy-winning mixing/mastering engineer analyzing production quality.

TECHNICAL DATA:
{audio_features}

PRODUCTION STANDARDS:
{production_standards}

STREAMING OPTIMIZATION:
Target LUFS: {streaming_target}
Radio Target: {radio_target}
Dynamic Range Standards: {dynamic_standards}

PRODUCTION ANALYSIS:

1. TECHNICAL QUALITY ASSESSMENT
   - Loudness standards compliance (LUFS analysis)
   - Dynamic range evaluation
   - Frequency balance assessment
   - Stereo imaging quality

2. MIX QUALITY EVALUATION
   - Clarity and separation
   - Depth and dimension
   - Frequency spectrum balance
   - Compression/limiting quality

3. PLATFORM OPTIMIZATION
   - Streaming service compatibility
   - Radio broadcast readiness
   - Club/live performance translation
   - Mobile device playback quality

4. INDUSTRY STANDARD COMPARISON
   - How does this compare to chart hits?
   - Professional production benchmark alignment
   - Genre-specific production expectations
   - Commercial release readiness

5. IMPROVEMENT RECOMMENDATIONS
   - Specific mix adjustments needed
   - Mastering optimization suggestions
   - Platform-specific optimizations
   - Professional polish requirements

RESPONSE FORMAT:
{{
  "technical_assessment": {{
    "loudness_compliance": "assessment",
    "dynamic_range_quality": "assessment",
    "frequency_balance": "assessment",
    "overall_technical_grade": "A-F"
  }},
  "mix_evaluation": {{
    "clarity_score": "score/10",
    "depth_score": "score/10", 
    "balance_score": "score/10",
    "professional_polish": "assessment"
  }},
  "platform_readiness": {{
    "streaming_optimized": "yes/no",
    "radio_ready": "yes/no",
    "club_translation": "assessment",
    "mobile_compatibility": "assessment"
  }},
  "improvement_plan": {{
    "critical_fixes": ["array"],
    "enhancement_suggestions": ["array"],
    "mastering_notes": ["array"],
    "estimated_work_hours": "number"
  }}
}}

Provide specific, actionable feedback that engineers can implement.
"""

    def _create_synthesis_prompt(self) -> str:
        """Create comprehensive synthesis prompt"""
        return """
You are a senior A&R executive preparing a comprehensive analysis for label decision-making.

ANALYSIS INPUTS:
Musical Meaning: {musical_meaning}
Hit Comparison: {hit_comparison}  
Novelty Assessment: {novelty_assessment}
Production Analysis: {production_analysis}

SYNTHESIS REQUIREMENTS:

1. EXECUTIVE SUMMARY
   - Overall commercial potential (High/Medium/Low)
   - Key strengths and differentiators
   - Primary concerns and risks
   - Recommended next steps

2. STRATEGIC RECOMMENDATIONS BY CATEGORY
   - PRODUCTION: Specific technical improvements
   - ARRANGEMENT: Musical/structural enhancements  
   - LYRICS: Content and delivery improvements
   - MARKETING: Positioning and promotion strategy

3. MARKET POSITIONING
   - Target demographic and psychographic profiles
   - Competitive positioning strategy
   - Playlist and radio strategy
   - Tour/live performance considerations

4. INVESTMENT RECOMMENDATION
   - Development budget recommendations
   - Timeline for optimization
   - Risk mitigation strategies
   - Success probability assessment

RESPONSE FORMAT:
{{
  "executive_summary": {{
    "commercial_potential": "High/Medium/Low",
    "confidence_level": "percentage",
    "key_strengths": ["array"],
    "primary_concerns": ["array"],
    "investment_recommendation": "string"
  }},
  "action_plan": {{
    "production_priorities": ["array", "in", "priority", "order"],
    "arrangement_improvements": ["array"],
    "lyrical_enhancements": ["array"],
    "timeline_estimate": "string"
  }},
  "market_strategy": {{
    "primary_demographic": "string",
    "positioning_strategy": "string",
    "playlist_targets": ["array"],
    "promotional_angles": ["array"]
  }},
  "financial_assessment": {{
    "development_budget_range": "string",
    "roi_potential": "assessment",
    "risk_factors": ["array"],
    "success_probability": "percentage"
  }}
}}

Focus on actionable business decisions and clear ROI justification.
"""
    
    async def analyze_song(self, request: InsightGenerationRequest) -> Dict[str, Any]:
        """
        Perform comprehensive song analysis using specialized chains
        """
        try:
            # Extract data from request
            audio_features = self._extract_audio_features(request)
            lyrics_content = self._extract_lyrics_content(request)
            genre = self._determine_genre(request)
            
            # Perform statistical analysis
            statistical_analysis = self._perform_statistical_analysis(audio_features, genre)
            
            # Run specialized analysis chains
            results = {}
            
            # Musical Meaning Analysis
            results['musical_meaning'] = await self._run_analysis_chain(
                AnalysisChainType.MUSICAL_MEANING,
                {
                    'audio_features': audio_features,
                    'genre': genre,
                    'similar_hits': statistical_analysis.get('similar_hits', [])
                }
            )
            
            # Hit Comparison Analysis  
            results['hit_comparison'] = await self._run_analysis_chain(
                AnalysisChainType.HIT_COMPARISON,
                {
                    'audio_features': audio_features,
                    'genre_benchmarks': statistical_analysis['benchmarks'],
                    'statistical_analysis': statistical_analysis['analysis'],
                    'genre': genre,
                    'chart_trends': statistical_analysis.get('trends', {}),
                    'historical_patterns': statistical_analysis.get('patterns', {})
                }
            )
            
            # Novelty Assessment
            results['novelty_assessment'] = await self._run_analysis_chain(
                AnalysisChainType.NOVELTY_ASSESSMENT,
                {
                    'audio_features': audio_features,
                    'lyrics_content': lyrics_content,
                    'genre': genre,
                    'current_trends': statistical_analysis.get('trends', {}),
                    'competitor_analysis': statistical_analysis.get('competitors', {})
                }
            )
            
            # Production Analysis
            results['production_analysis'] = await self._run_analysis_chain(
                AnalysisChainType.PRODUCTION_ANALYSIS,
                {
                    'audio_features': audio_features,
                    'production_standards': self._format_production_standards(),
                    'streaming_target': self.production_standards.streaming_lufs_target,
                    'radio_target': self.production_standards.radio_lufs_target,
                    'dynamic_standards': self.production_standards.dynamic_range_optimal
                }
            )
            
            # Comprehensive Synthesis
            results['comprehensive_analysis'] = await self._run_analysis_chain(
                AnalysisChainType.COMPREHENSIVE_SYNTHESIS,
                {
                    'musical_meaning': results['musical_meaning'],
                    'hit_comparison': results['hit_comparison'],
                    'novelty_assessment': results['novelty_assessment'],
                    'production_analysis': results['production_analysis']
                }
            )
            
            # Add metadata
            results['analysis_metadata'] = {
                'analysis_timestamp': datetime.now().isoformat(),
                'genre': genre,
                'statistical_confidence': statistical_analysis.get('confidence', 0.8),
                'analysis_version': '2.0.0'
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in enhanced analysis: {e}")
            raise
    
    def _extract_audio_features(self, request: InsightGenerationRequest) -> Dict[str, Any]:
        """Extract and format audio features from request"""
        if request.audio_features:
            return {
                'tempo': request.audio_features.tempo or 120.0,
                'energy': request.audio_features.energy or 0.5,
                'danceability': request.audio_features.danceability or 0.5,
                'valence': request.audio_features.valence or 0.5,
                'loudness': request.audio_features.loudness or -10.0,
                'acousticness': request.audio_features.acousticness or 0.5,
                'instrumentalness': request.audio_features.instrumentalness or 0.1,
                'liveness': request.audio_features.liveness or 0.1,
                'speechiness': request.audio_features.speechiness or 0.1
            }
        return {}
    
    def _extract_lyrics_content(self, request: InsightGenerationRequest) -> str:
        """Extract lyrics content from request"""
        if request.lyrics_analysis and request.lyrics_analysis.raw_text:
            return request.lyrics_analysis.raw_text
        return ""
    
    def _determine_genre(self, request: InsightGenerationRequest) -> str:
        """Determine genre from request metadata with proper genre cleaning"""
        if request.song_metadata and request.song_metadata.genre:
            return self._clean_genre_name(request.song_metadata.genre)
        return "pop"  # Default fallback
    
    def _perform_statistical_analysis(self, audio_features: Dict[str, Any], genre: str) -> Dict[str, Any]:
        """Perform statistical analysis against genre benchmarks"""
        # Normalize genre and try variations before defaulting to pop
        genre_key = genre.lower().replace(" ", "_").replace("-", "_")
        genre_benchmarks = self.genre_benchmarks.get(genre_key)
        
        if not genre_benchmarks:
            # Try common genre variations before defaulting to pop
            genre_variations = {
                "hip_hop": ["hiphop", "rap"],
                "r_b": ["rnb", "r&b"], 
                "electronic": ["edm", "techno", "house"],
                "alternative": ["alt", "indie"],
                "classical": ["orchestral"],
                "metal": ["heavy_metal", "death_metal"]
            }
            
            for bench_genre, variations in genre_variations.items():
                if genre_key in variations or any(var in genre_key for var in variations):
                    if bench_genre in self.genre_benchmarks:
                        genre_benchmarks = self.genre_benchmarks[bench_genre]
                        logger.info(f"Enhanced analysis: Mapped genre variation '{genre}' -> '{bench_genre}'")
                        break
            
            # Final fallback to pop
            if not genre_benchmarks:
                logger.warning(f"Enhanced analysis: Genre '{genre}' not found, using pop benchmarks")
                genre_benchmarks = self.genre_benchmarks['pop']
        
        analysis = {
            'benchmarks': {},
            'analysis': {},
            'confidence': 0.85
        }
        
        for feature_name, value in audio_features.items():
            if feature_name in genre_benchmarks:
                benchmark = genre_benchmarks[feature_name]
                z_score = benchmark.calculate_z_score(value)
                alignment = benchmark.get_alignment_assessment(value)
                
                analysis['benchmarks'][feature_name] = {
                    'value': value,
                    'genre_mean': benchmark.mean,
                    'std_dev': benchmark.std_dev,
                    'z_score': z_score,
                    'alignment': alignment,
                    'optimal_range': benchmark.optimal_range
                }
        
        return analysis
    
    def _format_production_standards(self) -> str:
        """Format production standards for prompt"""
        return f"""
Streaming LUFS Target: {self.production_standards.streaming_lufs_target}
Radio LUFS Target: {self.production_standards.radio_lufs_target}
Dynamic Range Minimum: {self.production_standards.dynamic_range_minimum}dB
Dynamic Range Optimal: {self.production_standards.dynamic_range_optimal}dB
Frequency Balance Tolerance: ±{self.production_standards.frequency_balance_tolerance}dB
"""
    
    async def _run_analysis_chain(self, chain_type: AnalysisChainType, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific analysis chain with context"""
        try:
            prompt_template = self.analysis_prompts[chain_type]
            prompt = prompt_template.format(**context)
            
            response = await self.llm_provider.generate_insights({
                'prompt': prompt,
                'max_tokens': 2000,
                'temperature': 0.7
            })
            
            # Parse JSON response
            if response and 'content' in response:
                try:
                    return json.loads(response['content'])
                except json.JSONDecodeError:
                    # Fallback to text response
                    return {'raw_response': response['content']}
            
            return {'error': 'No response generated'}
            
        except Exception as e:
            logger.error(f"Error running {chain_type.value} chain: {e}")
            return {'error': str(e)}
    
    def _clean_genre_name(self, genre_name: str) -> str:
        """Clean and normalize genre names from various formats (shared with agent_llm_service)."""
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
                logger.debug(f"Enhanced analysis: Mapped genre '{original_genre}' -> '{value}' (via '{main_genre}')")
                return value
        
        # If no mapping found, return the cleaned main genre
        logger.debug(f"Enhanced analysis: No mapping found for genre '{original_genre}', using cleaned: '{main_genre}'")
        return main_genre