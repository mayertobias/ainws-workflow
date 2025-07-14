"""
Pydantic models for workflow-orchestrator service
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskStatus(str, Enum):
    """Individual task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY = "retry"


class WorkflowTemplate(str, Enum):
    """Available workflow templates."""
    COMPREHENSIVE_ANALYSIS = "comprehensive_analysis"
    HIT_PREDICTION = "hit_prediction"
    AUDIO_ONLY_ANALYSIS = "audio_only_analysis"
    AI_INSIGHTS_ONLY = "ai_insights_only"
    PRODUCTION_ANALYSIS = "production_analysis"
    CUSTOM = "custom"


class TaskType(str, Enum):
    """Types of tasks in workflows."""
    AUDIO_ANALYSIS = "audio_analysis"
    CONTENT_ANALYSIS = "content_analysis"
    ML_TRAINING = "ml_training"
    ML_PREDICTION = "ml_prediction"
    AI_INSIGHTS = "ai_insights"
    DATA_VALIDATION = "data_validation"
    NOTIFICATION = "notification"
    STORAGE_UPLOAD = "storage_upload"


class Priority(str, Enum):
    """Task and workflow priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


# Base Models

class TaskDefinition(BaseModel):
    """Definition of a task within a workflow."""
    task_id: str = Field(description="Unique task identifier")
    task_type: TaskType = Field(description="Type of task to execute")
    service_name: str = Field(description="Target microservice name")
    endpoint: str = Field(description="Service endpoint to call")
    method: str = Field(default="POST", description="HTTP method")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    timeout_seconds: Optional[int] = Field(default=None, description="Task timeout")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    required: bool = Field(default=True, description="Whether task is required for workflow success")


class WorkflowDefinition(BaseModel):
    """Definition of a workflow template."""
    template_name: WorkflowTemplate = Field(description="Workflow template name")
    name: str = Field(description="Human-readable workflow name")
    description: str = Field(description="Workflow description")
    version: str = Field(default="1.0", description="Workflow version")
    tasks: List[TaskDefinition] = Field(description="List of tasks in workflow")
    timeout_seconds: int = Field(default=3600, description="Workflow timeout")
    parallel_execution: bool = Field(default=True, description="Allow parallel task execution")
    failure_strategy: str = Field(default="stop_on_failure", description="Strategy for handling failures")


class TaskExecution(BaseModel):
    """Runtime information about task execution."""
    task_id: str = Field(description="Task identifier")
    workflow_id: str = Field(description="Parent workflow ID")
    status: TaskStatus = Field(description="Current task status")
    started_at: Optional[datetime] = Field(default=None, description="Task start time")
    completed_at: Optional[datetime] = Field(default=None, description="Task completion time")
    duration_seconds: Optional[float] = Field(default=None, description="Task duration")
    attempt_number: int = Field(default=1, description="Current attempt number")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Task result data")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    retry_count: int = Field(default=0, description="Number of retries performed")
    logs: List[str] = Field(default_factory=list, description="Task execution logs")


class WorkflowExecution(BaseModel):
    """Runtime information about workflow execution."""
    workflow_id: str = Field(description="Unique workflow identifier")
    template_name: WorkflowTemplate = Field(description="Workflow template used")
    status: WorkflowStatus = Field(description="Current workflow status")
    created_at: datetime = Field(description="Workflow creation time")
    started_at: Optional[datetime] = Field(default=None, description="Workflow start time")
    completed_at: Optional[datetime] = Field(default=None, description="Workflow completion time")
    duration_seconds: Optional[float] = Field(default=None, description="Workflow duration")
    progress_percentage: float = Field(default=0.0, description="Workflow progress percentage")
    tasks: List[TaskExecution] = Field(default_factory=list, description="Task executions")
    input_data: Dict[str, Any] = Field(description="Workflow input data")
    output_data: Optional[Dict[str, Any]] = Field(default=None, description="Workflow output data")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# Request Models

class WorkflowExecutionRequest(BaseModel):
    """Request to execute a workflow."""
    template_name: WorkflowTemplate = Field(description="Workflow template to execute")
    input_data: Dict[str, Any] = Field(description="Input data for workflow")
    priority: Priority = Field(default=Priority.NORMAL, description="Workflow priority")
    timeout_seconds: Optional[int] = Field(default=None, description="Override workflow timeout")
    callback_url: Optional[str] = Field(default=None, description="Callback URL for completion notification")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    tags: List[str] = Field(default_factory=list, description="Workflow tags")


class CustomWorkflowRequest(BaseModel):
    """Request to execute a custom workflow."""
    name: str = Field(description="Custom workflow name")
    description: str = Field(description="Workflow description")
    tasks: List[TaskDefinition] = Field(description="Custom task definitions")
    input_data: Dict[str, Any] = Field(description="Input data for workflow")
    priority: Priority = Field(default=Priority.NORMAL, description="Workflow priority")
    timeout_seconds: int = Field(default=3600, description="Workflow timeout")
    parallel_execution: bool = Field(default=True, description="Allow parallel execution")
    callback_url: Optional[str] = Field(default=None, description="Callback URL")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TaskRetryRequest(BaseModel):
    """Request to retry a failed task."""
    workflow_id: str = Field(description="Workflow identifier")
    task_id: str = Field(description="Task identifier to retry")
    reset_dependencies: bool = Field(default=False, description="Reset dependent tasks")


# Response Models

class WorkflowExecutionResponse(BaseModel):
    """Response from workflow execution request."""
    workflow_id: str = Field(description="Unique workflow identifier")
    status: WorkflowStatus = Field(description="Initial workflow status")
    created_at: datetime = Field(description="Workflow creation time")
    estimated_duration_seconds: Optional[int] = Field(default=None, description="Estimated completion time")
    tracking_url: str = Field(description="URL to track workflow progress")


class WorkflowStatusResponse(BaseModel):
    """Response for workflow status query."""
    workflow: WorkflowExecution = Field(description="Workflow execution details")
    current_tasks: List[TaskExecution] = Field(description="Currently executing tasks")
    next_tasks: List[str] = Field(description="Next tasks to execute")
    completed_tasks_count: int = Field(description="Number of completed tasks")
    total_tasks_count: int = Field(description="Total number of tasks")


class WorkflowListResponse(BaseModel):
    """Response for workflow listing."""
    workflows: List[WorkflowExecution] = Field(description="List of workflows")
    total_count: int = Field(description="Total number of workflows")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Page size")
    has_next: bool = Field(description="Whether there are more pages")


class TaskLogEntry(BaseModel):
    """Log entry for task execution."""
    timestamp: datetime = Field(description="Log timestamp")
    level: str = Field(description="Log level")
    message: str = Field(description="Log message")
    service: str = Field(description="Source service")
    task_id: str = Field(description="Task identifier")


class WorkflowMetrics(BaseModel):
    """Workflow execution metrics."""
    total_workflows: int = Field(description="Total workflows executed")
    active_workflows: int = Field(description="Currently active workflows")
    completed_workflows: int = Field(description="Completed workflows")
    failed_workflows: int = Field(description="Failed workflows")
    average_duration_seconds: float = Field(description="Average workflow duration")
    success_rate: float = Field(description="Workflow success rate")
    popular_templates: List[Dict[str, Any]] = Field(description="Most used workflow templates")
    service_utilization: Dict[str, float] = Field(description="Service utilization metrics")


class ServiceHealthStatus(BaseModel):
    """Health status of a microservice."""
    service_name: str = Field(description="Service name")
    url: str = Field(description="Service URL")
    status: str = Field(description="Health status")
    response_time_ms: Optional[float] = Field(default=None, description="Response time")
    last_checked: datetime = Field(description="Last health check time")
    error_message: Optional[str] = Field(default=None, description="Error message if unhealthy")


class OrchestrationHealthResponse(BaseModel):
    """Overall orchestration service health."""
    status: str = Field(description="Overall health status")
    timestamp: datetime = Field(description="Health check timestamp")
    services: List[ServiceHealthStatus] = Field(description="Individual service health")
    database_connected: bool = Field(description="Database connection status")
    redis_connected: bool = Field(description="Redis connection status")
    active_workflows: int = Field(description="Number of active workflows")
    queue_length: int = Field(description="Task queue length")


# Workflow Template Definitions

class SongAnalysisInput(BaseModel):
    """Input data for song analysis workflows."""
    song_id: Optional[str] = Field(default=None, description="Unique song identifier")
    file_path: Optional[str] = Field(default=None, description="Path to audio file")
    file_url: Optional[str] = Field(default=None, description="URL to audio file")
    song_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Song metadata")
    lyrics: Optional[str] = Field(default=None, description="Song lyrics")
    analysis_options: Dict[str, Any] = Field(default_factory=dict, description="Analysis options")


class ComprehensiveAnalysisOutput(BaseModel):
    """Output from comprehensive analysis workflow."""
    workflow_id: str = Field(description="Workflow identifier")
    song_id: str = Field(description="Song identifier")
    audio_features: Optional[Dict[str, Any]] = Field(default=None, description="Audio analysis results")
    content_analysis: Optional[Dict[str, Any]] = Field(default=None, description="Content analysis results")
    hit_prediction: Optional[Dict[str, Any]] = Field(default=None, description="Hit prediction results")
    ai_insights: Optional[Dict[str, Any]] = Field(default=None, description="AI insights results")
    overall_score: Optional[float] = Field(default=None, description="Overall song score")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    processing_time_seconds: float = Field(description="Total processing time")
    completed_at: datetime = Field(description="Completion timestamp") 