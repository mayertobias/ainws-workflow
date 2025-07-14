"""
Common response models for workflow-orchestrator service
"""

from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class SuccessResponse(BaseModel):
    """Standard success response model."""
    status: str = Field(default="success", description="Response status")
    message: str = Field(description="Success message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    status: str = Field(default="error", description="Response status")
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(description="Health status")
    service: str = Field(description="Service name")
    timestamp: datetime = Field(description="Health check timestamp")
    version: str = Field(default="1.0.0", description="Service version")
    uptime_seconds: Optional[float] = Field(default=None, description="Service uptime") 