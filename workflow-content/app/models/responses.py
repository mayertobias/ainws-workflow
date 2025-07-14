"""
Common response models for workflow-content service
"""

from pydantic import BaseModel, Field
from typing import Any, Optional
from datetime import datetime

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field("1.0.0", description="Service version")
    service: str = Field("workflow-content", description="Service name")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
class SuccessResponse(BaseModel):
    """Generic success response"""
    status: str = Field("success", description="Operation status")
    message: str = Field(..., description="Success message")
    data: Optional[Any] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.utcnow) 