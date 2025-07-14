"""
Pydantic models for ML prediction
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum

class PredictionType(str, Enum):
    """Types of predictions"""
    SINGLE = "single"
    BATCH = "batch"
    STREAMING = "streaming"

class PredictionStatus(str, Enum):
    """Prediction job status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"

class FeatureInput(BaseModel):
    """Audio/content features for prediction"""
    # Audio features
    tempo: Optional[float] = Field(None, ge=40, le=250, description="Tempo in BPM")
    energy: Optional[float] = Field(None, ge=0, le=1, description="Energy level")
    danceability: Optional[float] = Field(None, ge=0, le=1, description="Danceability score")
    valence: Optional[float] = Field(None, ge=0, le=1, description="Valence (positivity)")
    acousticness: Optional[float] = Field(None, ge=0, le=1, description="Acousticness")
    loudness: Optional[float] = Field(None, ge=-60, le=0, description="Loudness in dB")
    instrumentalness: Optional[float] = Field(None, ge=0, le=1, description="Instrumentalness")
    liveness: Optional[float] = Field(None, ge=0, le=1, description="Liveness")
    speechiness: Optional[float] = Field(None, ge=0, le=1, description="Speechiness")
    
    # Content features
    lyrics_sentiment: Optional[float] = Field(None, ge=-1, le=1, description="Lyrics sentiment score")
    lyrics_complexity: Optional[float] = Field(None, ge=0, le=1, description="Lyrics complexity")
    word_count: Optional[int] = Field(None, ge=0, description="Word count")
    unique_words: Optional[int] = Field(None, ge=0, description="Unique word count")
    
    # Custom features
    custom_features: Optional[Dict[str, float]] = Field(default_factory=dict, description="Custom features")

class SinglePredictionRequest(BaseModel):
    """Request for single prediction"""
    model_id: str = Field(..., description="Model identifier to use for prediction")
    features: FeatureInput = Field(..., description="Feature input")
    include_confidence: bool = Field(default=True, description="Include confidence intervals")
    include_feature_importance: bool = Field(default=False, description="Include feature importance")
    use_cache: bool = Field(default=True, description="Use cached results if available")

class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""
    model_id: str = Field(..., description="Model identifier to use for prediction")
    features_list: List[FeatureInput] = Field(..., max_items=1000, description="List of feature inputs")
    include_confidence: bool = Field(default=True, description="Include confidence intervals")
    include_feature_importance: bool = Field(default=False, description="Include feature importance")
    batch_id: Optional[str] = Field(None, description="Optional batch identifier")
    
    @validator('features_list')
    def validate_features_list(cls, v):
        if not v:
            raise ValueError('Features list cannot be empty')
        return v

class PredictionResult(BaseModel):
    """Single prediction result"""
    hit_score: float = Field(..., ge=0, le=1, description="Predicted hit score (0-1)")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="Confidence interval")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")
    model_version: str = Field(..., description="Model version used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class SinglePredictionResponse(BaseModel):
    """Response for single prediction"""
    status: PredictionStatus = Field(..., description="Prediction status")
    prediction: Optional[PredictionResult] = Field(None, description="Prediction result")
    model_id: str = Field(..., description="Model identifier used")
    prediction_id: str = Field(..., description="Unique prediction identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")
    cached: bool = Field(default=False, description="Whether result was cached")

class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    status: PredictionStatus = Field(..., description="Batch prediction status")
    batch_id: str = Field(..., description="Batch identifier")
    total_predictions: int = Field(..., description="Total number of predictions")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    predictions: List[PredictionResult] = Field(..., description="Prediction results")
    model_id: str = Field(..., description="Model identifier used")
    processing_time_ms: float = Field(..., description="Total processing time")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")

class ModelInfo(BaseModel):
    """Information about available models"""
    model_id: str = Field(..., description="Model identifier")
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type")
    version: str = Field(..., description="Model version")
    accuracy_metrics: Dict[str, float] = Field(..., description="Model accuracy metrics")
    feature_columns: List[str] = Field(..., description="Required feature columns")
    last_trained: datetime = Field(..., description="Last training timestamp")
    is_cached: bool = Field(default=False, description="Whether model is cached in memory")

class ModelListResponse(BaseModel):
    """Response for listing available models"""
    models: List[ModelInfo] = Field(..., description="Available models")
    total_count: int = Field(..., description="Total number of models")
    cached_count: int = Field(..., description="Number of cached models")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

class PredictionMetrics(BaseModel):
    """Prediction service metrics"""
    total_predictions: int = Field(..., description="Total predictions made")
    cache_hit_rate: float = Field(..., description="Cache hit rate")
    average_response_time_ms: float = Field(..., description="Average response time")
    models_loaded: int = Field(..., description="Number of models loaded")
    active_sessions: int = Field(..., description="Active prediction sessions")
    error_rate: float = Field(..., description="Error rate")

class FeatureValidationRequest(BaseModel):
    """Request for feature validation"""
    features: FeatureInput = Field(..., description="Features to validate")
    model_id: Optional[str] = Field(None, description="Model to validate against")

class FeatureValidationResponse(BaseModel):
    """Response for feature validation"""
    is_valid: bool = Field(..., description="Whether features are valid")
    missing_features: List[str] = Field(default_factory=list, description="Missing required features")
    invalid_features: List[str] = Field(default_factory=list, description="Invalid feature values")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    feature_count: int = Field(..., description="Number of valid features")

class ModelLoadRequest(BaseModel):
    """Request to load a model"""
    model_id: str = Field(..., description="Model identifier to load")
    force_reload: bool = Field(default=False, description="Force reload even if cached")

class ModelLoadResponse(BaseModel):
    """Response for model loading"""
    model_id: str = Field(..., description="Model identifier")
    status: str = Field(..., description="Load status")
    load_time_ms: float = Field(..., description="Load time in milliseconds")
    model_size_mb: float = Field(..., description="Model size in MB")
    feature_columns: List[str] = Field(..., description="Model feature columns")
    is_cached: bool = Field(..., description="Whether model is now cached")

class ExplainabilityRequest(BaseModel):
    """Request for prediction explainability"""
    model_id: str = Field(..., description="Model identifier")
    features: FeatureInput = Field(..., description="Features to explain")
    explanation_type: str = Field(default="shap", description="Type of explanation (shap, lime)")

class ExplainabilityResponse(BaseModel):
    """Response for prediction explainability"""
    model_id: str = Field(..., description="Model identifier")
    explanation: Dict[str, Any] = Field(..., description="Explanation results")
    feature_contributions: Dict[str, float] = Field(..., description="Feature contributions to prediction")
    base_value: float = Field(..., description="Base prediction value")
    prediction_value: float = Field(..., description="Final prediction value")

class ABTestRequest(BaseModel):
    """Request for A/B testing between models"""
    model_a_id: str = Field(..., description="First model identifier")
    model_b_id: str = Field(..., description="Second model identifier")
    features: FeatureInput = Field(..., description="Features for comparison")
    test_id: Optional[str] = Field(None, description="A/B test identifier")

class ABTestResponse(BaseModel):
    """Response for A/B testing"""
    test_id: str = Field(..., description="A/B test identifier")
    model_a_result: PredictionResult = Field(..., description="Model A prediction")
    model_b_result: PredictionResult = Field(..., description="Model B prediction")
    difference: float = Field(..., description="Prediction difference")
    confidence_comparison: Dict[str, Any] = Field(..., description="Confidence comparison")
    recommendation: str = Field(..., description="Recommended model")

class StreamingPredictionRequest(BaseModel):
    """Request for streaming predictions"""
    model_id: str = Field(..., description="Model identifier")
    stream_id: str = Field(..., description="Stream identifier")
    buffer_size: int = Field(default=100, ge=1, le=1000, description="Buffer size for streaming")

class StreamingPredictionResponse(BaseModel):
    """Response for streaming predictions"""
    stream_id: str = Field(..., description="Stream identifier")
    status: str = Field(..., description="Stream status")
    buffer_size: int = Field(..., description="Current buffer size")
    predictions_processed: int = Field(..., description="Total predictions processed")
    stream_url: str = Field(..., description="Stream endpoint URL") 