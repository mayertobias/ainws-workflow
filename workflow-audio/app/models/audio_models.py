"""
Pydantic models for audio analysis requests and responses
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum

class AnalysisType(str, Enum):
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"

class AudioAnalysisRequest(BaseModel):
    """Request model for audio analysis by file path"""
    file_path: str = Field(..., description="Path to the audio file to analyze")
    analysis_type: Optional[AnalysisType] = Field(AnalysisType.BASIC, description="Type of analysis to perform")
    extractors: Optional[List[str]] = Field(None, description="Specific extractors to use (for comprehensive analysis)")
    force_reanalysis: Optional[bool] = Field(False, description="Force reanalysis even if existing analysis found")

class AudioAnalysisResponse(BaseModel):
    """Response model for audio analysis"""
    status: str = Field(..., description="Status of the analysis")
    filename: str = Field(..., description="Name of the analyzed file")
    analysis_type: str = Field(..., description="Type of analysis performed")
    results: Dict[str, Any] = Field(..., description="Analysis results")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    service: str
    version: str
    capabilities: Dict[str, bool]

class StatusResponse(BaseModel):
    """Detailed status response model"""
    service: str
    audio_analyzer_status: Dict[str, Any]
    supported_formats: List[str]
    features: Dict[str, bool]

class ExtractorInfo(BaseModel):
    """Information about available extractors"""
    available_extractors: List[str]
    extractor_info: Dict[str, str]

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: str
    status_code: int 