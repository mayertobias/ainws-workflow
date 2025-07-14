"""
Pydantic models for AI intelligence generation
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum

class AnalysisType(str, Enum):
    """Types of AI analysis available"""
    MUSICAL_MEANING = "musical_meaning"
    HIT_COMPARISON = "hit_comparison"
    NOVELTY_ASSESSMENT = "novelty_assessment"
    GENRE_COMPARISON = "genre_comparison"
    PRODUCTION_FEEDBACK = "production_feedback"
    STRATEGIC_INSIGHTS = "strategic_insights"
    COMPREHENSIVE_ANALYSIS = "comprehensive_analysis"

class AIProvider(str, Enum):
    """Supported AI providers"""
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    AUTO = "auto"

class AgentType(str, Enum):
    """Available agent types"""
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    MUSIC_ANALYSIS = "music_analysis"

class InsightStatus(str, Enum):
    """Status of insight generation"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"

class AudioFeaturesInput(BaseModel):
    """Audio features for AI analysis"""
    # Basic audio features
    tempo: Optional[float] = Field(None, ge=40, le=250, description="Tempo in BPM")
    energy: Optional[float] = Field(None, ge=0, le=1, description="Energy level")
    danceability: Optional[float] = Field(None, ge=0, le=1, description="Danceability score")
    valence: Optional[float] = Field(None, ge=0, le=1, description="Valence (positivity)")
    acousticness: Optional[float] = Field(None, ge=0, le=1, description="Acousticness")
    loudness: Optional[float] = Field(None, ge=-60, le=0, description="Loudness in dB")
    instrumentalness: Optional[float] = Field(None, ge=0, le=1, description="Instrumentalness")
    liveness: Optional[float] = Field(None, ge=0, le=1, description="Liveness")
    speechiness: Optional[float] = Field(None, ge=0, le=1, description="Speechiness")
    
    # Advanced audio features
    spectral_centroid: Optional[float] = Field(None, description="Spectral centroid")
    mfcc_mean: Optional[List[float]] = Field(None, description="MFCC mean values")
    chroma_features: Optional[List[float]] = Field(None, description="Chroma features")
    
    # Custom features
    custom_features: Optional[Dict[str, float]] = Field(default_factory=dict, description="Custom audio features")

class LyricsAnalysisInput(BaseModel):
    """Lyrics analysis for AI insights"""
    raw_text: Optional[str] = Field(None, description="Raw lyrics text")
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1, description="Sentiment score")
    complexity_score: Optional[float] = Field(None, ge=0, le=1, description="Complexity score")
    word_count: Optional[int] = Field(None, ge=0, description="Total word count")
    unique_words: Optional[int] = Field(None, ge=0, description="Unique word count")
    readability_score: Optional[float] = Field(None, description="Readability score")
    emotion_scores: Optional[Dict[str, float]] = Field(default_factory=dict, description="Emotion analysis")
    topics: Optional[List[str]] = Field(default_factory=list, description="Extracted topics")

class HitAnalysisInput(BaseModel):
    """Hit song analysis data for comparison"""
    hit_score: Optional[float] = Field(None, ge=0, le=1, description="Predicted hit score")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="Confidence interval")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance")
    model_version: Optional[str] = Field(None, description="Model version used")
    historical_comparison: Optional[Dict[str, Any]] = Field(None, description="Historical hit data")

class SongMetadata(BaseModel):
    """Song metadata for context"""
    title: Optional[str] = Field(None, description="Song title")
    artist: Optional[str] = Field(None, description="Artist name")
    genre: Optional[str] = Field(None, description="Primary genre")
    subgenres: Optional[List[str]] = Field(default_factory=list, description="Sub-genres")
    release_year: Optional[int] = Field(None, ge=1900, le=2030, description="Release year")
    duration_seconds: Optional[float] = Field(None, ge=0, description="Duration in seconds")
    language: Optional[str] = Field(None, description="Lyrics language")
    market_target: Optional[str] = Field(None, description="Target market")

class InsightGenerationRequest(BaseModel):
    """Request for generating AI insights"""
    # Data inputs
    audio_features: Optional[AudioFeaturesInput] = Field(None, description="Audio feature data")
    lyrics_analysis: Optional[LyricsAnalysisInput] = Field(None, description="Lyrics analysis data")
    hit_analysis: Optional[HitAnalysisInput] = Field(None, description="Hit song analysis data")
    song_metadata: Optional[SongMetadata] = Field(None, description="Song metadata")
    
    # Analysis configuration
    analysis_types: List[AnalysisType] = Field(
        default=[AnalysisType.MUSICAL_MEANING, AnalysisType.HIT_COMPARISON],
        description="Types of analysis to perform"
    )
    agent_type: AgentType = Field(default=AgentType.STANDARD, description="Agent type to use")
    ai_provider: AIProvider = Field(default=AIProvider.AUTO, description="AI provider preference")
    
    # Options
    use_cache: bool = Field(default=True, description="Use cached results if available")
    include_raw_output: bool = Field(default=False, description="Include raw AI output")
    max_tokens: Optional[int] = Field(None, ge=100, le=4000, description="Maximum tokens for generation")
    temperature: Optional[float] = Field(None, ge=0, le=1, description="AI generation temperature")
    
    @validator('analysis_types')
    def validate_analysis_types(cls, v):
        if not v:
            raise ValueError('At least one analysis type must be specified')
        return v

class MusicalMeaningInsight(BaseModel):
    """Musical meaning analysis result"""
    emotional_core: Optional[str] = Field(None, description="Core emotional message")
    musical_narrative: Optional[str] = Field(None, description="Musical storytelling analysis")
    mood_progression: Optional[List[str]] = Field(None, description="Mood changes throughout song")
    cultural_context: Optional[str] = Field(None, description="Cultural and social context")
    artistic_intent: Optional[str] = Field(None, description="Inferred artistic intent")
    listener_impact: Optional[str] = Field(None, description="Predicted listener emotional impact")
    key_strengths: Optional[List[str]] = Field(None, description="Key musical strengths")
    improvement_areas: Optional[List[str]] = Field(None, description="Areas for improvement")

class HitComparisonInsight(BaseModel):
    """Hit comparison analysis result"""
    hit_alignment_score: Optional[float] = Field(None, ge=0, le=1, description="Alignment with hit patterns")
    genre_benchmarks: Optional[Dict[str, float]] = Field(None, description="Genre-specific benchmarks")
    successful_references: Optional[List[str]] = Field(None, description="Similar successful songs")
    commercial_strengths: Optional[List[str]] = Field(None, description="Commercial advantages")
    commercial_weaknesses: Optional[List[str]] = Field(None, description="Commercial disadvantages")
    market_positioning: Optional[str] = Field(None, description="Recommended market positioning")
    target_audience: Optional[str] = Field(None, description="Primary target audience")

class NoveltyAssessmentInsight(BaseModel):
    """Novelty assessment analysis result"""
    innovation_score: Optional[float] = Field(None, ge=0, le=1, description="Innovation level")
    unique_elements: Optional[List[str]] = Field(None, description="Unique musical elements")
    trend_alignment: Optional[str] = Field(None, description="Alignment with current trends")
    risk_assessment: Optional[str] = Field(None, description="Risk vs. reward analysis")
    differentiation_factors: Optional[List[str]] = Field(None, description="Key differentiators")
    market_readiness: Optional[str] = Field(None, description="Market readiness assessment")

class ProductionFeedback(BaseModel):
    """Production quality feedback"""
    overall_quality: Optional[str] = Field(None, description="Overall production assessment")
    technical_strengths: Optional[List[str]] = Field(None, description="Technical production strengths")
    technical_issues: Optional[List[str]] = Field(None, description="Technical issues to address")
    mix_balance: Optional[str] = Field(None, description="Mix balance analysis")
    mastering_notes: Optional[str] = Field(None, description="Mastering recommendations")
    sonic_character: Optional[str] = Field(None, description="Sonic character description")

class StrategicInsights(BaseModel):
    """Strategic business insights"""
    market_opportunity: Optional[str] = Field(None, description="Market opportunity analysis")
    competitive_advantage: Optional[str] = Field(None, description="Competitive advantages")
    release_strategy: Optional[str] = Field(None, description="Recommended release strategy")
    promotional_angles: Optional[List[str]] = Field(None, description="Promotional opportunities")
    collaboration_opportunities: Optional[List[str]] = Field(None, description="Collaboration suggestions")
    revenue_potential: Optional[str] = Field(None, description="Revenue potential assessment")

class AIInsightResult(BaseModel):
    """Complete AI insight analysis result"""
    # Core insights
    musical_meaning: Optional[MusicalMeaningInsight] = None
    hit_comparison: Optional[HitComparisonInsight] = None
    novelty_assessment: Optional[NoveltyAssessmentInsight] = None
    production_feedback: Optional[ProductionFeedback] = None
    strategic_insights: Optional[StrategicInsights] = None
    
    # Hit potential data
    hit_potential_score: Optional[float] = Field(None, ge=0, le=1, description="Predicted hit potential score")
    hit_confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence in hit prediction")
    model_used: Optional[str] = Field(None, description="ML model used for hit prediction")
    
    # Summary
    executive_summary: Optional[str] = Field(None, description="Executive summary of all insights")
    key_recommendations: Optional[List[str]] = Field(None, description="Top recommendations")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Overall confidence in analysis")
    
    # Metadata
    analysis_types_completed: List[AnalysisType] = Field(default_factory=list)
    agent_used: Optional[str] = Field(None, description="Agent type used")
    provider_used: Optional[str] = Field(None, description="AI provider used")
    processing_time_ms: Optional[float] = Field(None, description="Processing time")
    token_usage: Optional[Dict[str, int]] = Field(None, description="Token usage statistics")
    raw_outputs: Optional[Dict[str, str]] = Field(None, description="Raw AI outputs")

class InsightGenerationResponse(BaseModel):
    """Response for insight generation"""
    status: InsightStatus = Field(..., description="Generation status")
    insight_id: str = Field(..., description="Unique insight identifier")
    request_id: Optional[str] = Field(None, description="Request identifier")
    insights: Optional[AIInsightResult] = Field(None, description="Generated insights")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")
    cached: bool = Field(default=False, description="Whether result was cached")

class PromptTemplate(BaseModel):
    """AI prompt template"""
    template_id: str = Field(..., description="Template identifier")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    template_text: str = Field(..., description="Prompt template text")
    variables: List[str] = Field(..., description="Template variables")
    analysis_type: AnalysisType = Field(..., description="Associated analysis type")
    version: str = Field(default="1.0", description="Template version")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class PromptTemplateRequest(BaseModel):
    """Request to create or update prompt template"""
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    template_text: str = Field(..., description="Prompt template text")
    variables: List[str] = Field(..., description="Template variables")
    analysis_type: AnalysisType = Field(..., description="Associated analysis type")

class ProviderInfo(BaseModel):
    """Information about an AI provider"""
    provider_name: str = Field(..., description="Provider name")
    model_name: Optional[str] = Field(None, description="Model name")
    available: bool = Field(..., description="Whether provider is available")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Provider configuration")
    capabilities: List[str] = Field(default_factory=list, description="Provider capabilities")
    cost_per_token: Optional[float] = Field(None, description="Estimated cost per token")

class ProviderListResponse(BaseModel):
    """Response listing available AI providers"""
    providers: List[ProviderInfo] = Field(..., description="Available providers")
    default_provider: Optional[str] = Field(None, description="Default provider")
    auto_detected: Optional[str] = Field(None, description="Auto-detected provider")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class InsightMetrics(BaseModel):
    """Intelligence service metrics"""
    total_insights_generated: int = Field(..., description="Total insights generated")
    cache_hit_rate: float = Field(..., description="Cache hit rate")
    average_processing_time_ms: float = Field(..., description="Average processing time")
    provider_usage: Dict[str, int] = Field(..., description="Usage by provider")
    analysis_type_distribution: Dict[str, int] = Field(..., description="Analysis type distribution")
    error_rate: float = Field(..., description="Error rate")
    cost_tracking: Optional[Dict[str, float]] = Field(None, description="Cost tracking by provider")

class CacheStats(BaseModel):
    """Cache statistics"""
    total_cached_items: int = Field(..., description="Total cached items")
    cache_hit_rate: float = Field(..., description="Cache hit rate")
    cache_size_mb: Optional[float] = Field(None, description="Cache size in MB")
    oldest_entry: Optional[datetime] = Field(None, description="Oldest cache entry")
    newest_entry: Optional[datetime] = Field(None, description="Newest cache entry")

class BatchInsightRequest(BaseModel):
    """Request for batch insight generation"""
    requests: List[InsightGenerationRequest] = Field(..., max_items=50, description="Batch requests")
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    priority: int = Field(default=0, ge=0, le=10, description="Processing priority")

class BatchInsightResponse(BaseModel):
    """Response for batch insight generation"""
    batch_id: str = Field(..., description="Batch identifier")
    total_requests: int = Field(..., description="Total requests in batch")
    completed_requests: int = Field(..., description="Completed requests")
    failed_requests: int = Field(..., description="Failed requests")
    results: List[InsightGenerationResponse] = Field(..., description="Individual results")
    processing_time_ms: float = Field(..., description="Total processing time")
    timestamp: datetime = Field(default_factory=datetime.utcnow) 