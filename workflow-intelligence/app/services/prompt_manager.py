"""
Prompt Management System for workflow-intelligence

This module handles prompt templates, formatting, and management for AI insights generation.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..models.intelligence import (
    AnalysisType, InsightGenerationRequest, AIInsightResult,
    PromptTemplate, AudioFeaturesInput, LyricsAnalysisInput,
    HitAnalysisInput, SongMetadata
)
from ..config.settings import settings

logger = logging.getLogger(__name__)

class PromptManager:
    """
    Manages AI prompt templates for different analysis types.
    
    Provides template loading, formatting, and caching for AI insights generation.
    """
    
    def __init__(self):
        """Initialize the prompt manager."""
        self.templates: Dict[AnalysisType, str] = {}
        self.custom_templates: Dict[str, PromptTemplate] = {}
        self._load_default_templates()
        
        logger.info("Prompt Manager initialized successfully")
    
    def _load_default_templates(self):
        """Load default prompt templates for each analysis type."""
        
        # Musical Meaning Analysis Template
        self.templates[AnalysisType.MUSICAL_MEANING] = """
You are an expert music analyst specializing in the emotional and artistic interpretation of songs. 
Analyze the provided musical data and generate insights about the song's emotional core and artistic meaning.

Song Information:
{song_metadata}

Audio Features:
{audio_features}

Lyrics Analysis:
{lyrics_analysis}

Please provide a comprehensive musical meaning analysis covering:

1. **Emotional Core**: What is the primary emotional message of this song?
2. **Musical Narrative**: How does the music tell a story or convey meaning?
3. **Mood Progression**: How does the emotional journey unfold throughout the song?
4. **Cultural Context**: What cultural or social themes are present?
5. **Artistic Intent**: What was the likely artistic vision behind this song?
6. **Listener Impact**: How will this song emotionally affect listeners?
7. **Key Strengths**: What are the strongest emotional/artistic elements?
8. **Improvement Areas**: What aspects could enhance the emotional impact?

Respond in JSON format with the following structure:
{{
    "emotional_core": "string",
    "musical_narrative": "string", 
    "mood_progression": ["string"],
    "cultural_context": "string",
    "artistic_intent": "string",
    "listener_impact": "string",
    "key_strengths": ["string"],
    "improvement_areas": ["string"]
}}
"""

        # Hit Comparison Analysis Template
        self.templates[AnalysisType.HIT_COMPARISON] = """
You are a music industry expert specializing in hit song analysis and commercial potential assessment.
Compare this song against successful hits to evaluate its commercial viability using sophisticated analysis.

Song Information:
{song_metadata}

Audio Features:
{audio_features}

Hit Analysis Data:
{hit_analysis}

Lyrics Analysis:
{lyrics_analysis}

HISTORICAL CONTEXT & BENCHMARKS:
- Genre Average Features: Tempo 118.5 BPM, Energy 0.75, Danceability 0.68, Valence 0.62
- Hit Song Characteristics: Tempo 110-140 BPM, Energy 0.6-0.9, Strong hook within 15 seconds
- Commercial Sweet Spots: Radio-friendly 3-4 min duration, streaming-optimized loudness
- Current Market Trends: Pop (32%), Hip-Hop (28%), Rock (18%), Electronic (12%)

SOPHISTICATED COMMERCIAL ANALYSIS REQUIRED:

1. **Hit Alignment Score** (0-1): Calculate based on:
   - Feature proximity to proven hit patterns
   - Genre-specific commercial benchmarks
   - Market positioning potential
   - Cross-demographic appeal

2. **Genre Benchmarks**: Compare against:
   - Average vs. top-performing songs in genre
   - Historical hit patterns and characteristics
   - Current market preferences and trends

3. **Successful References**: Identify similar hits with specific reasoning:
   - Musical feature similarities
   - Commercial performance parallels
   - Market positioning examples

4. **Commercial Strengths**: Analyze technical and artistic factors:
   - Audio feature alignment with hit patterns
   - Production quality against modern standards
   - Hook placement and structure effectiveness
   - Genre authenticity vs. crossover potential

5. **Commercial Weaknesses**: Identify limiting factors:
   - Deviations from successful patterns
   - Production issues affecting commercial appeal
   - Structural elements limiting radio/streaming success
   - Market saturation or trend misalignment

6. **Market Positioning**: Strategic commercial placement:
   - Primary and secondary target markets
   - Optimal release timing and channels
   - Competitive landscape analysis
   - Revenue potential assessment

7. **Target Audience**: Data-driven demographic analysis:
   - Primary age groups and psychographics
   - Genre loyalty and crossover potential
   - Streaming vs. traditional media preferences
   - Geographic and cultural considerations

Provide specific, actionable insights based on actual music industry data and patterns. 
Reference specific feature values and their commercial implications.

Respond in JSON format with the following structure:
{{
    "hit_alignment_score": 0.0,
    "genre_benchmarks": {{"pop": 0.8, "overall": 0.7}},
    "successful_references": ["specific song examples with reasoning"],
    "commercial_strengths": ["specific technical and artistic factors"],
    "commercial_weaknesses": ["specific limiting factors with solutions"],
    "market_positioning": "detailed strategic positioning",
    "target_audience": "data-driven demographic analysis",
    "revenue_potential": "assessment based on market analysis",
    "recommended_strategy": "specific commercial strategy"
}}
"""

        # Novelty Assessment Template
        self.templates[AnalysisType.NOVELTY_ASSESSMENT] = """
You are an innovation expert in music, specializing in identifying unique elements and assessing novelty
against current music trends and historical patterns.

Song Information:
{song_metadata}

Audio Features:
{audio_features}

Lyrics Analysis:
{lyrics_analysis}

INNOVATION ANALYSIS FRAMEWORK:
- Optimal Innovation Rate: 25% innovative, 75% familiar for commercial success
- Genre Deviation Threshold: 30% maximum for mainstream appeal
- Trend Alignment Weight: 40% importance for market timing
- Historical Pattern Matching: Compare against 10,000+ analyzed songs

COMPREHENSIVE NOVELTY ASSESSMENT:

1. **Innovation Score** (0-1): Multi-dimensional analysis:
   - Musical feature uniqueness vs. genre norms
   - Structural innovation vs. proven formats
   - Production technique novelty
   - Lyrical content originality

2. **Unique Elements**: Specific innovative aspects:
   - Audio feature combinations not seen in top hits
   - Structural or arrangement innovations
   - Production techniques and sonic characteristics
   - Lyrical themes, complexity, or delivery methods

3. **Trend Alignment**: Market timing analysis:
   - Emerging genre trends and micro-genres
   - Production style evolution patterns
   - Cultural moment alignment
   - Technology adoption in production

4. **Risk Assessment**: Commercial innovation balance:
   - Market readiness for innovation level
   - Potential audience confusion vs. excitement
   - Radio programmability with novel elements
   - Streaming algorithm compatibility

5. **Differentiation Factors**: Competitive positioning:
   - How this stands out in current market
   - Unique value proposition for listeners
   - Memorable elements for recall and sharing
   - Artist brand alignment with innovation

6. **Market Readiness**: Innovation adoption analysis:
   - Historical precedents for similar innovations
   - Current audience tolerance for experimentation
   - Industry infrastructure support
   - Tastemaker and influencer appeal

PROVIDE SPECIFIC EXAMPLES AND QUANTITATIVE ANALYSIS.

Respond in JSON format with the following structure:
{{
    "innovation_score": 0.0,
    "unique_elements": ["specific innovative aspects with details"],
    "trend_alignment": "detailed market timing analysis",
    "risk_assessment": "balanced commercial innovation evaluation", 
    "differentiation_factors": ["specific competitive advantages"],
    "market_readiness": "innovation adoption potential with precedents",
    "innovation_category": "incremental/moderate/breakthrough",
    "commercial_viability": "assessment of innovation level vs. commercial success"
}}
"""

        # Production Feedback Template
        self.templates[AnalysisType.PRODUCTION_FEEDBACK] = """
You are a professional audio engineer and producer with expertise in modern music production standards.
Analyze the technical aspects of this song against industry benchmarks and provide detailed feedback.

Song Information:
{song_metadata}

Audio Features:
{audio_features}

MODERN PRODUCTION STANDARDS:
- Loudness: -14 LUFS target for streaming, -6 to -8 dB peak for competitive loudness
- Dynamic Range: Minimum 6 dB, optimal 8 dB for musical impact
- Frequency Balance: Controlled sub-bass, clear mids, crisp highs without harshness
- Streaming Optimization: Peak levels -1 to -2 dB, hook within 15 seconds, 3-4 min duration

TECHNICAL PRODUCTION ANALYSIS:

1. **Overall Quality**: Comprehensive production assessment:
   - Professional vs. amateur production indicators
   - Industry standard compliance level
   - Commercial competitiveness rating
   - Technical execution quality

2. **Technical Strengths**: Specific production excellence:
   - Frequency balance and EQ choices
   - Dynamic range and compression effectiveness
   - Stereo imaging and spatial characteristics
   - Loudness optimization for streaming platforms

3. **Technical Issues**: Production problems and solutions:
   - Frequency response issues (muddy bass, harsh highs)
   - Dynamic range problems (over/under compression)
   - Stereo imaging issues or mono compatibility
   - Loudness war victims or insufficient level

4. **Mix Balance**: Element relationship analysis:
   - Vocal clarity and presence in the mix
   - Instrumental balance and separation
   - Rhythm section cohesion and punch
   - Effects usage and spatial placement

5. **Mastering Assessment**: Final stage evaluation:
   - Loudness standards compliance
   - Frequency spectrum optimization
   - Dynamic processing effectiveness
   - Platform compatibility (streaming, radio, club)

6. **Sonic Character**: Unique production signature:
   - Genre-appropriate sound characteristics
   - Artistic vision vs. technical execution balance
   - Innovative production techniques employed
   - Competitive positioning in current market

PROVIDE SPECIFIC TECHNICAL RECOMMENDATIONS AND MEASUREMENTS.

Respond in JSON format with the following structure:
{{
    "overall_quality": "detailed professional assessment",
    "technical_strengths": ["specific production excellences"],
    "technical_issues": ["specific problems with solutions"],
    "mix_balance": "element relationship analysis",
    "mastering_notes": "final stage recommendations",
    "sonic_character": "unique production signature analysis",
    "industry_compliance": "modern standards compliance rating",
    "improvement_priority": "most critical fixes ranked by impact"
}}
"""

        # Strategic Insights Template
        self.templates[AnalysisType.STRATEGIC_INSIGHTS] = """
You are a music industry strategist with expertise in artist development and commercial strategy.
Provide comprehensive strategic business insights for this song's release and promotion.

Song Information:
{song_metadata}

Audio Features:
{audio_features}

Lyrics Analysis:
{lyrics_analysis}

Hit Analysis:
{hit_analysis}

STRATEGIC BUSINESS FRAMEWORK:
- Market Opportunity Analysis: Genre trends, demographic shifts, platform preferences
- Competitive Landscape: Current chart positions, similar artists, market saturation
- Revenue Streams: Streaming, sync, live, merchandise, brand partnerships
- Platform Strategy: TikTok, Spotify, YouTube, radio, sync opportunities

COMPREHENSIVE STRATEGIC ANALYSIS:

1. **Market Opportunity**: Data-driven opportunity assessment:
   - Genre market share and growth trends
   - Demographic expansion opportunities
   - Platform-specific growth potential
   - Geographic market opportunities

2. **Release Strategy**: Tactical release planning:
   - Optimal release timing and calendar positioning
   - Platform sequencing and exclusive opportunities
   - Pre-release campaign strategy and timeline
   - Post-release momentum maintenance plan

3. **Promotional Angles**: Creative marketing positioning:
   - Unique story angles and narrative development
   - Social media content strategy and viral potential
   - Influencer and tastemaker targeting
   - Cross-platform content adaptation

4. **Revenue Optimization**: Multiple income stream development:
   - Streaming platform optimization strategy
   - Sync licensing and placement opportunities
   - Live performance and touring potential
   - Brand partnership and endorsement possibilities

5. **Competitive Analysis**: Market positioning strategy:
   - Similar artist comparison and differentiation
   - Chart competition and timing considerations
   - Playlist placement strategy and curator targeting
   - Media coverage and PR positioning

6. **Risk Mitigation**: Strategic risk management:
   - Market saturation and oversupply concerns
   - Platform algorithm changes and dependencies
   - Cultural sensitivity and controversy potential
   - Long-term career impact considerations

7. **Success Metrics**: KPI definition and tracking:
   - Streaming targets and growth expectations
   - Chart performance goals and benchmarks
   - Social media engagement and follower growth
   - Revenue targets and milestone markers

PROVIDE ACTIONABLE BUSINESS STRATEGIES WITH TIMELINES.

Respond in JSON format with the following structure:
{{
    "market_opportunity": "detailed opportunity analysis with data",
    "release_strategy": "tactical release planning with timeline",
    "promotional_angles": ["creative marketing positioning strategies"],
    "revenue_optimization": "multiple income stream development plan",
    "competitive_analysis": "market positioning strategy",
    "risk_mitigation": "strategic risk management plan",
    "success_metrics": "KPI definition and tracking system",
    "recommended_budget": "strategic investment allocation",
    "timeline": "90-day action plan with milestones"
}}
"""

        # Comprehensive Analysis Template
        self.templates[AnalysisType.COMPREHENSIVE_ANALYSIS] = """
You are a comprehensive music expert with deep knowledge across all aspects of music analysis.
Provide a complete, holistic analysis combining artistic, commercial, and technical perspectives.

Song Information:
{song_metadata}

Audio Features:
{audio_features}

Lyrics Analysis:
{lyrics_analysis}

Hit Analysis:
{hit_analysis}

Please provide a comprehensive analysis that synthesizes insights across all dimensions:
- Artistic and emotional elements
- Commercial potential and market positioning  
- Technical production quality
- Innovation and uniqueness
- Strategic recommendations

Provide a thorough analysis that considers all aspects and their interactions.
Focus on actionable insights and recommendations for the artist/label.
"""

        logger.info(f"Loaded {len(self.templates)} default prompt templates")
    
    async def get_formatted_prompt(
        self, 
        analysis_type: AnalysisType, 
        request: InsightGenerationRequest,
        enhanced_context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Get formatted prompt for a specific analysis type."""
        try:
            template = self.templates.get(analysis_type)
            if not template:
                logger.warning(f"No template found for analysis type: {analysis_type}")
                return None
            
            # Prepare context data for prompt formatting with robust attribute handling
            context_data = {
                "song_metadata": self._format_metadata(getattr(request, 'song_metadata', None)),
                "audio_features": self._format_audio_features(getattr(request, 'audio_features', None)),
                "lyrics_analysis": self._format_lyrics_analysis(getattr(request, 'lyrics_analysis', None)),
                "hit_analysis": self._format_hit_analysis(enhanced_context.get("hit_potential", {}) if enhanced_context else {})
            }
            
            # Add enhanced context if available
            if enhanced_context:
                context_data.update({
                    "historical_analysis": self._format_historical_analysis(enhanced_context.get("historical_analysis", {})),
                    "genre_benchmarks": self._format_genre_benchmarks(enhanced_context.get("genre_benchmarks", {})),
                    "production_standards": self._format_production_standards(enhanced_context.get("production_standards", {})),
                    "market_intelligence": self._format_market_intelligence(enhanced_context.get("market_intelligence", {}))
                })
            
            # Format the template with context data - bypass metadata error
            try:
                formatted_prompt = template.format(**context_data)
            except KeyError as e:
                if 'metadata' in str(e):
                    # Fallback for metadata issue - use song_metadata instead
                    context_data['metadata'] = context_data['song_metadata']
                    formatted_prompt = template.format(**context_data)
                else:
                    raise e
            
            logger.debug(f"Generated prompt for {analysis_type}: {len(formatted_prompt)} characters")
            return formatted_prompt
            
        except Exception as e:
            logger.error(f"Error formatting prompt for {analysis_type}: {e}")
            logger.error(f"Template content: {template}")
            logger.error(f"Context data keys: {list(context_data.keys())}")
            logger.error(f"Request type: {type(request)}")
            logger.error(f"Request attributes: {dir(request)}")
            return None
    
    def _format_historical_analysis(self, historical_data: Dict[str, Any]) -> str:
        """Format historical analysis data for prompt."""
        if not historical_data:
            return "No historical analysis data available"
        
        formatted = []
        
        # Genre alignment
        if "genre_alignment_score" in historical_data:
            formatted.append(f"Genre Alignment Score: {historical_data['genre_alignment_score']:.2f}")
        
        # Benchmark comparisons
        if "benchmark_comparison" in historical_data:
            benchmarks = historical_data["benchmark_comparison"]
            formatted.append("\nBenchmark Comparisons:")
            
            if "vs_genre_average" in benchmarks:
                formatted.append("  vs. Genre Average:")
                for feature, comparison in benchmarks["vs_genre_average"].items():
                    formatted.append(f"    {feature}: {comparison}")
            
            if "vs_hit_patterns" in benchmarks:
                formatted.append("  vs. Hit Patterns:")
                for feature, comparison in benchmarks["vs_hit_patterns"].items():
                    formatted.append(f"    {feature}: {comparison}")
        
        # Commercial indicators
        if "commercial_indicators" in historical_data:
            indicators = historical_data["commercial_indicators"]
            formatted.append("\nCommercial Indicators:")
            for indicator, value in indicators.items():
                formatted.append(f"  {indicator}: {value}")
        
        return "\n".join(formatted)
    
    def _format_genre_benchmarks(self, benchmarks: Dict[str, Any]) -> str:
        """Format genre benchmarks for prompt."""
        if not benchmarks:
            return "No genre benchmarks available"
        
        formatted = []
        for genre, data in benchmarks.items():
            formatted.append(f"{genre.title()} Genre:")
            if "feature_weights" in data:
                formatted.append("  Feature Weights:")
                for feature, weight in data["feature_weights"].items():
                    formatted.append(f"    {feature}: {weight:.2f}")
            if "commercial_factors" in data:
                formatted.append("  Commercial Factors:")
                for factor, value in data["commercial_factors"].items():
                    formatted.append(f"    {factor}: {value}")
        
        return "\n".join(formatted)
    
    def _format_production_standards(self, standards: Dict[str, Any]) -> str:
        """Format production standards for prompt."""
        if not standards:
            return "No production standards available"
        
        formatted = []
        
        if "modern_mastering" in standards:
            mastering = standards["modern_mastering"]
            formatted.append("Modern Mastering Standards:")
            for standard, value in mastering.items():
                formatted.append(f"  {standard}: {value}")
        
        if "streaming_optimization" in standards:
            streaming = standards["streaming_optimization"]
            formatted.append("Streaming Optimization:")
            for standard, value in streaming.items():
                formatted.append(f"  {standard}: {value}")
        
        return "\n".join(formatted)
    
    def _format_market_intelligence(self, intelligence: Dict[str, Any]) -> str:
        """Format market intelligence for prompt."""
        if not intelligence:
            return "No market intelligence available"
        
        formatted = []
        
        if "current_trends" in intelligence:
            trends = intelligence["current_trends"]
            formatted.append("Current Market Trends:")
            for trend, value in trends.items():
                formatted.append(f"  {trend}: {value}")
        
        if "demographic_preferences" in intelligence:
            demographics = intelligence["demographic_preferences"]
            formatted.append("Demographic Preferences:")
            for demo, prefs in demographics.items():
                formatted.append(f"  {demo}: {prefs}")
        
        return "\n".join(formatted)
    
    def _format_hit_analysis(self, hit_data: Dict[str, Any]) -> str:
        """Format hit analysis data for prompt."""
        if not hit_data:
            return "No hit analysis data available"
        
        formatted = []
        
        if "prediction" in hit_data:
            formatted.append(f"Hit Prediction Score: {hit_data['prediction']:.2f}")
        
        if "confidence" in hit_data:
            formatted.append(f"Confidence Level: {hit_data['confidence']:.2f}")
        
        if "model_used" in hit_data:
            formatted.append(f"Model Used: {hit_data['model_used']}")
        
        return "\n".join(formatted)
    
    def _format_metadata(self, metadata: SongMetadata) -> str:
        """Format song metadata for prompt."""
        if not metadata:
            return "No song metadata available"
        
        formatted = []
        if metadata.title:
            formatted.append(f"Title: {metadata.title}")
        if metadata.artist:
            formatted.append(f"Artist: {metadata.artist}")
        if metadata.genre:
            formatted.append(f"Genre: {metadata.genre}")
        if metadata.release_year:
            formatted.append(f"Year: {metadata.release_year}")
        if metadata.duration_seconds:
            duration_min = metadata.duration_seconds / 60
            formatted.append(f"Duration: {duration_min:.1f} minutes")
        
        return "\n".join(formatted)
    
    def _format_audio_features(self, audio_features: AudioFeaturesInput) -> str:
        """Format audio features for prompt."""
        if not audio_features:
            return "No audio features available"
        
        formatted = []
        for field, value in audio_features.dict(exclude_none=True).items():
            if isinstance(value, (int, float)):
                if field in ['tempo']:
                    formatted.append(f"{field.title()}: {value:.1f}")
                elif field in ['loudness']:
                    formatted.append(f"{field.title()}: {value:.1f} dB")
                else:
                    formatted.append(f"{field.title()}: {value:.3f}")
            elif isinstance(value, list) and value:
                formatted.append(f"{field.title()}: {len(value)} values")
            elif isinstance(value, dict) and value:
                formatted.append(f"{field.title()}: {len(value)} features")
        
        return "\n".join(formatted)
    
    def _format_lyrics_analysis(self, lyrics_analysis: LyricsAnalysisInput) -> str:
        """Format lyrics analysis for prompt."""
        if not lyrics_analysis:
            return "No lyrics analysis data available"
        
        formatted = []
        if lyrics_analysis.sentiment_score is not None:
            sentiment = "positive" if lyrics_analysis.sentiment_score > 0 else "negative" if lyrics_analysis.sentiment_score < 0 else "neutral"
            formatted.append(f"Sentiment: {sentiment} ({lyrics_analysis.sentiment_score:.3f})")
        if lyrics_analysis.complexity_score is not None:
            formatted.append(f"Complexity: {lyrics_analysis.complexity_score:.3f}")
        if lyrics_analysis.word_count:
            formatted.append(f"Word count: {lyrics_analysis.word_count}")
        if lyrics_analysis.topics:
            formatted.append(f"Topics: {', '.join(lyrics_analysis.topics)}")
        if lyrics_analysis.emotion_scores:
            emotions = [f"{emotion}: {score:.2f}" for emotion, score in lyrics_analysis.emotion_scores.items()]
            formatted.append(f"Emotions: {', '.join(emotions)}")
        
        return "\n".join(formatted)
    
    async def get_summary_prompt(
        self, 
        insights: AIInsightResult, 
        request: InsightGenerationRequest
    ) -> Optional[str]:
        """Generate executive summary prompt from completed insights."""
        try:
            summary_parts = []
            
            # Collect completed insights
            if insights.musical_meaning:
                summary_parts.append(f"Musical Meaning: {insights.musical_meaning.emotional_core}")
            
            if insights.hit_comparison:
                summary_parts.append(f"Commercial Potential: {insights.hit_comparison.market_positioning}")
            
            if insights.novelty_assessment:
                summary_parts.append(f"Innovation Level: {insights.novelty_assessment.innovation_score}")
            
            if insights.production_feedback:
                summary_parts.append(f"Production Quality: {insights.production_feedback.overall_quality}")
            
            if insights.strategic_insights:
                summary_parts.append(f"Market Opportunity: {insights.strategic_insights.market_opportunity}")
            
            if not summary_parts:
                return None
            
            insights_text = "\n".join(summary_parts)
            
            prompt = f"""
Based on the following analysis insights for this song, provide a concise executive summary 
that synthesizes the key findings and overall assessment:

{insights_text}

Song Context:
{self._format_metadata(request.song_metadata)}

Please provide a 2-3 paragraph executive summary that captures:
1. Overall assessment of the song's potential
2. Key strengths and opportunities
3. Primary recommendations for the artist/label

Focus on actionable insights and strategic direction.
"""
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error generating summary prompt: {e}")
            return None
    
    async def get_recommendations_prompt(
        self, 
        insights: AIInsightResult, 
        request: InsightGenerationRequest
    ) -> Optional[str]:
        """Generate recommendations prompt from completed insights."""
        try:
            # Collect key findings
            findings = []
            
            if insights.musical_meaning:
                if hasattr(insights.musical_meaning, 'key_strengths') and insights.musical_meaning.key_strengths:
                    findings.extend(insights.musical_meaning.key_strengths)
                elif isinstance(insights.musical_meaning, dict) and 'key_strengths' in insights.musical_meaning:
                    findings.extend(insights.musical_meaning['key_strengths'])
            
            if insights.hit_comparison:
                if hasattr(insights.hit_comparison, 'commercial_strengths') and insights.hit_comparison.commercial_strengths:
                    findings.extend(insights.hit_comparison.commercial_strengths)
                elif isinstance(insights.hit_comparison, dict) and 'commercial_strengths' in insights.hit_comparison:
                    findings.extend(insights.hit_comparison['commercial_strengths'])
            
            if insights.strategic_insights:
                if hasattr(insights.strategic_insights, 'promotional_angles') and insights.strategic_insights.promotional_angles:
                    findings.extend(insights.strategic_insights.promotional_angles)
                elif isinstance(insights.strategic_insights, dict) and 'promotional_angles' in insights.strategic_insights:
                    findings.extend(insights.strategic_insights['promotional_angles'])
            
            if not findings:
                return None
            
            findings_text = "\n".join([f"- {finding}" for finding in findings[:10]])
            
            prompt = f"""
Based on the comprehensive analysis of this song, generate 5-7 specific, actionable recommendations:

Key Findings:
{findings_text}

Song Context:
{self._format_metadata(request.song_metadata)}

Please provide specific recommendations covering:
- Artistic/creative development
- Production improvements
- Marketing and promotion strategies
- Target audience engagement
- Commercial positioning

Format as a numbered list of actionable recommendations.
"""
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error generating recommendations prompt: {e}")
            return None
    
    def add_custom_template(self, template: PromptTemplate) -> bool:
        """Add a custom prompt template."""
        try:
            self.custom_templates[template.template_id] = template
            logger.info(f"Added custom template: {template.template_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding custom template: {e}")
            return False
    
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a specific template by ID."""
        return self.custom_templates.get(template_id)
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available templates."""
        templates = []
        
        # Default templates
        for analysis_type in self.templates:
            templates.append({
                "template_id": f"default_{analysis_type.value}",
                "name": f"Default {analysis_type.value.replace('_', ' ').title()}",
                "analysis_type": analysis_type.value,
                "is_default": True
            })
        
        # Custom templates
        for template in self.custom_templates.values():
            templates.append({
                "template_id": template.template_id,
                "name": template.name,
                "analysis_type": template.analysis_type.value,
                "is_default": False
            })
        
        return templates
    
    def update_template(self, analysis_type: AnalysisType, template_text: str) -> bool:
        """Update a default template."""
        try:
            self.templates[analysis_type] = template_text
            logger.info(f"Updated template for {analysis_type}")
            return True
        except Exception as e:
            logger.error(f"Error updating template: {e}")
            return False 