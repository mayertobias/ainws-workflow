"""
Pydantic models for lyrics analysis
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
from datetime import datetime

class LyricsAnalysisRequest(BaseModel):
    """Request model for lyrics analysis"""
    text: str = Field(..., description="Lyrics text to analyze", max_length=10000)
    filename: Optional[str] = Field(None, description="Original filename of the lyrics file")
    title: Optional[str] = Field(None, description="Title or name for this analysis")
    language: Optional[str] = Field("en", description="Language of the lyrics")
    analysis_type: Optional[str] = Field("comprehensive", description="Type of analysis to perform")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

class SentimentRequest(BaseModel):
    """Request model for sentiment analysis"""
    text: str = Field(..., description="Text to analyze for sentiment", max_length=10000)
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

class SentimentResponse(BaseModel):
    """Response model for sentiment analysis"""
    polarity: float = Field(..., description="Sentiment polarity (-1 to 1)")
    subjectivity: float = Field(..., description="Subjectivity score (0 to 1)")
    emotional_scores: Dict[str, float] = Field(default_factory=dict, description="Emotional word scores")
    
class ComplexityMetrics(BaseModel):
    """Complexity metrics for lyrics"""
    avg_sentence_length: float
    avg_word_length: float
    lexical_diversity: float

class ThemeAnalysis(BaseModel):
    """Theme analysis results"""
    top_words: List[str]
    main_nouns: List[str]
    main_verbs: List[str]
    entities: List[str]

class Statistics(BaseModel):
    """Basic statistics for lyrics"""
    word_count: int
    unique_words: int
    vocabulary_density: float
    sentence_count: int
    avg_words_per_sentence: float

class LyricsAnalysisResponse(BaseModel):
    """Complete response model for lyrics analysis"""
    sentiment: SentimentResponse
    complexity: ComplexityMetrics
    themes: ThemeAnalysis
    readability: float
    emotional_progression: List[Dict[str, float]]
    narrative_structure: Dict[str, Any]
    key_motifs: List[Dict[str, Any]]
    theme_clusters: List[Dict[str, Any]]
    statistics: Statistics
    
class AnalysisResult(BaseModel):
    """Generic analysis result wrapper"""
    status: str = Field(..., description="Status of the analysis")
    results: Optional[LyricsAnalysisResponse] = Field(None, description="Analysis results")
    error: Optional[str] = Field(None, description="Error message if any")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    analysis_id: Optional[str] = Field(None, description="Unique analysis identifier for database tracking") 