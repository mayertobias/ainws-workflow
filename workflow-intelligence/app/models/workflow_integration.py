"""
Workflow Integration Models for Intelligence Service

These models define the data structures for receiving comprehensive analysis
from audio, content, and ML prediction services to generate AI insights.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum

# Input Models - Data from other services

class AudioAnalysisResult(BaseModel):
    """Audio analysis results from workflow-audio service"""
    # Basic audio features
    tempo: Optional[float] = Field(None, description="Tempo in BPM")
    energy: Optional[float] = Field(None, ge=0, le=1, description="Energy level")
    danceability: Optional[float] = Field(None, ge=0, le=1, description="Danceability score")
    valence: Optional[float] = Field(None, ge=0, le=1, description="Valence (positivity)")
    acousticness: Optional[float] = Field(None, ge=0, le=1, description="Acousticness")
    loudness: Optional[float] = Field(None, description="Loudness in dB")
    instrumentalness: Optional[float] = Field(None, ge=0, le=1, description="Instrumentalness")
    liveness: Optional[float] = Field(None, ge=0, le=1, description="Liveness")
    speechiness: Optional[float] = Field(None, ge=0, le=1, description="Speechiness")
    
    # Advanced audio features
    spectral_features: Optional[Dict[str, float]] = Field(default_factory=dict, description="Spectral analysis results")
    mfcc_features: Optional[List[float]] = Field(default_factory=list, description="MFCC coefficients")
    chroma_features: Optional[List[float]] = Field(default_factory=list, description="Chroma features")
    rhythm_features: Optional[Dict[str, float]] = Field(default_factory=dict, description="Rhythm analysis")
    harmonic_features: Optional[Dict[str, float]] = Field(default_factory=dict, description="Harmonic analysis")
    
    # Essentia-specific features
    genre_predictions: Optional[Dict[str, float]] = Field(default_factory=dict, description="Genre predictions")
    mood_predictions: Optional[Dict[str, float]] = Field(default_factory=dict, description="Mood predictions")
    
    # Quality metrics
    audio_quality_score: Optional[float] = Field(None, ge=0, le=1, description="Audio quality assessment")
    
    # Metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = Field(None, description="Analysis processing time")
    
class ContentAnalysisResult(BaseModel):
    """Content analysis results from workflow-content service"""
    # Lyrics analysis
    raw_lyrics: Optional[str] = Field(None, description="DEPRECATED: Raw lyrics should NEVER be sent to LLMs for privacy. Use extracted features only.")
    processed_lyrics: Optional[str] = Field(None, description="Cleaned lyrics text")
    
    # Sentiment and emotion analysis
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1, description="Overall sentiment")
    emotion_scores: Optional[Dict[str, float]] = Field(default_factory=dict, description="Emotion analysis results")
    mood_classification: Optional[str] = Field(None, description="Dominant mood")
    
    # Linguistic analysis
    language: Optional[str] = Field(None, description="Detected language")
    complexity_score: Optional[float] = Field(None, ge=0, le=1, description="Lyrical complexity")
    readability_score: Optional[float] = Field(None, description="Readability assessment")
    word_count: Optional[int] = Field(None, ge=0, description="Total word count")
    unique_words: Optional[int] = Field(None, ge=0, description="Unique word count")
    
    # Topic and theme analysis
    topics: Optional[List[str]] = Field(default_factory=list, description="Extracted topics")
    themes: Optional[List[str]] = Field(default_factory=list, description="Identified themes")
    keywords: Optional[List[str]] = Field(default_factory=list, description="Key terms")
    
    # Content categorization
    explicit_content: Optional[bool] = Field(None, description="Contains explicit content")
    content_warnings: Optional[List[str]] = Field(default_factory=list, description="Content warnings")
    target_audience: Optional[str] = Field(None, description="Appropriate audience")
    
    # Metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = Field(None, description="Analysis processing time")

class HitPredictionResult(BaseModel):
    """Hit prediction results from workflow-ml-prediction service"""
    # Core prediction
    hit_probability: Optional[float] = Field(None, ge=0, le=1, description="Probability of being a hit")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Prediction confidence")
    
    # Detailed predictions
    genre_specific_score: Optional[Dict[str, float]] = Field(default_factory=dict, description="Scores by genre")
    market_predictions: Optional[Dict[str, float]] = Field(default_factory=dict, description="Market-specific predictions")
    demographic_scores: Optional[Dict[str, float]] = Field(default_factory=dict, description="Demographic appeal scores")
    
    # Feature importance
    feature_importance: Optional[Dict[str, float]] = Field(default_factory=dict, description="Feature impact on prediction")
    top_contributing_features: Optional[List[str]] = Field(default_factory=list, description="Most important features")
    
    # Comparative analysis
    similar_hits: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Similar successful songs")
    genre_benchmarks: Optional[Dict[str, float]] = Field(default_factory=dict, description="Genre performance benchmarks")
    
    # Risk assessment
    commercial_risk_factors: Optional[List[str]] = Field(default_factory=list, description="Identified risk factors")
    success_factors: Optional[List[str]] = Field(default_factory=list, description="Success indicators")
    
    # Model metadata
    model_version: Optional[str] = Field(None, description="Prediction model version")
    training_data_size: Optional[int] = Field(None, description="Training dataset size")
    model_accuracy: Optional[float] = Field(None, description="Model accuracy on test set")
    
    # Metadata
    prediction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = Field(None, description="Prediction processing time")

class SongMetadata(BaseModel):
    """Song metadata and context information"""
    # Basic info
    title: str = Field(..., description="Song title")
    artist: str = Field(..., description="Artist name")
    album: Optional[str] = Field(None, description="Album name")
    
    # Musical attributes
    genre: Optional[str] = Field(None, description="Primary genre")
    subgenres: Optional[List[str]] = Field(default_factory=list, description="Sub-genres")
    key: Optional[str] = Field(None, description="Musical key")
    time_signature: Optional[str] = Field(None, description="Time signature")
    
    # Production info
    producer: Optional[str] = Field(None, description="Producer name")
    songwriter: Optional[str] = Field(None, description="Songwriter(s)")
    label: Optional[str] = Field(None, description="Record label")
    
    # Release info
    release_date: Optional[datetime] = Field(None, description="Release date")
    release_type: Optional[str] = Field(None, description="Single/Album/EP")
    
    # Technical specs
    duration_seconds: Optional[float] = Field(None, ge=0, description="Duration in seconds")
    sample_rate: Optional[int] = Field(None, description="Audio sample rate")
    bit_depth: Optional[int] = Field(None, description="Audio bit depth")
    
    # Market context
    target_market: Optional[str] = Field(None, description="Target market")
    language: Optional[str] = Field(None, description="Primary language")
    
    # File info
    file_path: Optional[str] = Field(None, description="Audio file path")
    file_size_mb: Optional[float] = Field(None, description="File size in MB")

# Request Models

class ComprehensiveAnalysisRequest(BaseModel):
    """Main request for comprehensive analysis from orchestrator"""
    # Source data
    audio_analysis: AudioAnalysisResult = Field(..., description="Audio analysis results")
    content_analysis: ContentAnalysisResult = Field(..., description="Content analysis results")
    hit_prediction: HitPredictionResult = Field(..., description="Hit prediction results")
    song_metadata: SongMetadata = Field(..., description="Song metadata")
    
    # Analysis configuration
    analysis_depth: str = Field(default="comprehensive", description="Analysis depth level")
    focus_areas: Optional[List[str]] = Field(default_factory=list, description="Specific areas to focus on")
    target_audience: Optional[str] = Field(None, description="Target audience for insights")
    business_context: Optional[str] = Field(None, description="Business context for analysis")
    
    # Request metadata
    request_id: Optional[str] = Field(None, description="Request identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Pre-computed AI insights (optional - if provided, skip re-analysis)
    ai_insights: Optional[Dict[str, Any]] = Field(None, description="Pre-computed AI insights from frontend")

# Response Models

class AnalysisInsight(BaseModel):
    """Individual analysis insight"""
    category: str = Field(..., description="Insight category")
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Detailed description")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in insight")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")

class ComprehensiveAnalysisResponse(BaseModel):
    """Response for comprehensive analysis"""
    # Request context
    request_id: str = Field(..., description="Original request ID")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis performed")
    
    # Executive summary
    executive_summary: str = Field(..., description="High-level summary of findings")
    overall_score: float = Field(..., ge=0, le=1, description="Overall quality/potential score")
    confidence_level: float = Field(..., ge=0, le=1, description="Overall confidence in analysis")
    
    # Individual assessment scores
    overall_assessment: Optional[Dict[str, float]] = Field(None, description="Individual assessment scores (artistic, commercial, innovation, hit_prediction)")
    
    # Detailed insights
    insights: List[AnalysisInsight] = Field(..., description="Detailed analysis insights")
    
    # Strategic recommendations
    strategic_recommendations: List[str] = Field(default_factory=list, description="Strategic business recommendations")
    tactical_recommendations: List[str] = Field(default_factory=list, description="Tactical implementation recommendations")
    
    # Risk and opportunity assessment
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    opportunities: List[str] = Field(default_factory=list, description="Identified opportunities")
    
    # Market insights
    market_positioning: Optional[str] = Field(None, description="Recommended market positioning")
    target_demographics: List[str] = Field(default_factory=list, description="Primary target demographics")
    competitive_analysis: Optional[str] = Field(None, description="Competitive landscape analysis")
    
    # Technical insights
    production_feedback: List[str] = Field(default_factory=list, description="Production quality feedback")
    technical_strengths: List[str] = Field(default_factory=list, description="Technical strengths")
    technical_improvements: List[str] = Field(default_factory=list, description="Suggested improvements")
    
    # Metadata
    processing_time_ms: float = Field(..., description="Total processing time")
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('insights')
    def validate_insights(cls, v):
        if not v:
            raise ValueError('At least one insight is required')
        return v 