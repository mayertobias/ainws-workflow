"""
Agentic AI Models for Intelligence Service

This module defines the data structures and models for the agentic AI system
including agent definitions, tool interfaces, memory structures, and coordination models.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
from enum import Enum
import uuid

# Core Agent Types and Roles

class AgentRole(str, Enum):
    """Specialized agent roles in the system"""
    MUSIC_ANALYSIS = "music_analysis"
    COMMERCIAL_ANALYSIS = "commercial_analysis" 
    NOVELTY_ASSESSMENT = "novelty_assessment"
    TECHNICAL_ANALYSIS = "technical_analysis"
    STRATEGIC_PLANNING = "strategic_planning"
    QUALITY_ASSURANCE = "quality_assurance"
    SYNTHESIS = "synthesis"
    ORCHESTRATOR = "orchestrator"

class AgentStatus(str, Enum):
    """Agent operational status"""
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"
    OFFLINE = "offline"

class TaskType(str, Enum):
    """Types of tasks agents can perform"""
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    COORDINATION = "coordination"
    TOOL_EXECUTION = "tool_execution"
    SYNTHESIS = "synthesis"
    REVIEW = "review"

class ToolType(str, Enum):
    """Types of tools available to agents"""
    DATA_ANALYSIS = "data_analysis"
    MUSIC_DATABASE = "music_database"
    MARKET_RESEARCH = "market_research"
    VISUALIZATION = "visualization"
    EXTERNAL_API = "external_api"
    CALCULATION = "calculation"
    LLM_REASONING = "llm_reasoning"
    LOGICAL_ANALYSIS = "logical_analysis"
    FACT_CHECKING = "fact_checking"
    STRATEGIC_ANALYSIS = "strategic_analysis"

class CoordinationStrategy(str, Enum):
    """Agent coordination strategies"""
    SEQUENTIAL = "sequential"          # One agent at a time
    PARALLEL = "parallel"              # Multiple agents simultaneously
    COLLABORATIVE = "collaborative"    # Agents work together
    COMPETITIVE = "competitive"        # Agents compete for best result
    HIERARCHICAL = "hierarchical"      # Lead agent coordinates others

# Agent Definition Models

class AgentCapability(BaseModel):
    """Represents a specific capability of an agent"""
    name: str = Field(..., description="Capability name")
    description: str = Field(..., description="Capability description")
    confidence_level: float = Field(..., ge=0, le=1, description="Agent's confidence in this capability")
    required_tools: List[ToolType] = Field(default_factory=list, description="Tools required for this capability")
    complexity_level: int = Field(..., ge=1, le=10, description="Complexity level (1=simple, 10=expert)")

class AgentProfile(BaseModel):
    """Complete profile of an agent"""
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique agent identifier")
    name: str = Field(..., description="Agent name")
    role: AgentRole = Field(..., description="Agent's primary role")
    description: str = Field(..., description="Agent description")
    
    # Capabilities and expertise
    capabilities: List[AgentCapability] = Field(..., description="Agent's capabilities")
    expertise_areas: List[str] = Field(..., description="Areas of expertise")
    specializations: List[str] = Field(default_factory=list, description="Specific specializations")
    
    # Performance metrics
    success_rate: float = Field(default=0.0, ge=0, le=1, description="Historical success rate")
    confidence_score: float = Field(default=0.5, ge=0, le=1, description="Overall confidence score")
    experience_level: int = Field(default=1, ge=1, le=10, description="Experience level")
    
    # Operational settings
    max_concurrent_tasks: int = Field(default=3, ge=1, description="Maximum concurrent tasks")
    preferred_tools: List[ToolType] = Field(default_factory=list, description="Preferred tools")
    communication_style: str = Field(default="collaborative", description="Communication style")
    
    # Status and availability
    status: AgentStatus = Field(default=AgentStatus.IDLE, description="Current status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)

# Tool Definition Models

class ToolParameter(BaseModel):
    """Parameter definition for a tool"""
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    default_value: Optional[Any] = Field(None, description="Default value")
    validation_rules: Optional[Dict[str, Any]] = Field(None, description="Validation rules")

class ToolDefinition(BaseModel):
    """Definition of a tool that agents can use"""
    tool_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique tool identifier")
    name: str = Field(..., description="Tool name")
    type: ToolType = Field(..., description="Tool type")
    description: str = Field(..., description="Tool description")
    
    # Tool functionality
    parameters: List[ToolParameter] = Field(..., description="Tool parameters")
    return_type: str = Field(..., description="Expected return type")
    example_usage: Optional[str] = Field(None, description="Example usage")
    
    # Tool metadata
    complexity: int = Field(..., ge=1, le=10, description="Tool complexity level")
    reliability: float = Field(default=1.0, ge=0, le=1, description="Tool reliability score")
    execution_time_ms: Optional[int] = Field(None, description="Average execution time")
    
    # Access control
    required_permissions: List[str] = Field(default_factory=list, description="Required permissions")
    available_to_roles: List[AgentRole] = Field(default_factory=list, description="Roles that can use this tool")

class ToolExecution(BaseModel):
    """Record of a tool execution"""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Execution identifier")
    tool_id: str = Field(..., description="Tool identifier")
    agent_id: str = Field(..., description="Agent that executed the tool")
    
    # Execution details
    input_parameters: Dict[str, Any] = Field(..., description="Input parameters")
    output_result: Optional[Any] = Field(None, description="Tool output")
    success: bool = Field(..., description="Whether execution succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    # Performance metrics
    execution_time_ms: float = Field(..., description="Actual execution time")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage")
    
    # Context and purpose
    task_context: str = Field(..., description="Context in which tool was used")
    purpose: str = Field(..., description="Purpose of tool execution")
    
    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)

# Task and Coordination Models

class AgentTask(BaseModel):
    """Task assigned to an agent"""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Task identifier")
    agent_id: str = Field(..., description="Assigned agent")
    task_type: TaskType = Field(..., description="Type of task")
    
    # Task definition
    description: str = Field(..., description="Task description")
    input_data: Dict[str, Any] = Field(..., description="Input data for task")
    requirements: List[str] = Field(default_factory=list, description="Task requirements")
    constraints: List[str] = Field(default_factory=list, description="Task constraints")
    
    # Task dependencies and relationships
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    dependent_tasks: List[str] = Field(default_factory=list, description="Tasks that depend on this one")
    parent_task_id: Optional[str] = Field(None, description="Parent task if this is a subtask")
    
    # Execution settings
    priority: int = Field(default=5, ge=1, le=10, description="Task priority")
    max_execution_time: int = Field(default=300, description="Maximum execution time in seconds")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    max_retries: int = Field(default=3, description="Maximum retries allowed")
    
    # Status and results
    status: AgentStatus = Field(default=AgentStatus.IDLE, description="Task status")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence in result")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)

class InterAgentMessage(BaseModel):
    """Message exchanged between agents"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Message identifier")
    sender_agent_id: str = Field(..., description="Sender agent")
    receiver_agent_id: str = Field(..., description="Receiver agent")
    
    # Message content
    message_type: str = Field(..., description="Type of message")
    content: str = Field(..., description="Message content")
    data: Optional[Dict[str, Any]] = Field(None, description="Structured data")
    
    # Message context
    context: str = Field(..., description="Message context")
    priority: int = Field(default=5, ge=1, le=10, description="Message priority")
    requires_response: bool = Field(default=False, description="Whether response is required")
    
    # Message handling
    delivered: bool = Field(default=False, description="Whether message was delivered")
    acknowledged: bool = Field(default=False, description="Whether message was acknowledged")
    response_message_id: Optional[str] = Field(None, description="Response message if any")
    
    # Timestamps
    sent_at: datetime = Field(default_factory=datetime.utcnow)
    delivered_at: Optional[datetime] = Field(None)
    acknowledged_at: Optional[datetime] = Field(None)

# Memory and Learning Models

class MemoryType(str, Enum):
    """Types of memory in the system"""
    SHORT_TERM = "short_term"      # Current session/task memory
    WORKING = "working"            # Active task context
    LONG_TERM = "long_term"        # Cross-session persistent memory
    EPISODIC = "episodic"          # Specific event/experience memory
    SEMANTIC = "semantic"          # Knowledge and facts memory

class MemoryEntry(BaseModel):
    """Entry in agent memory"""
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Memory entry identifier")
    agent_id: str = Field(..., description="Agent that owns this memory")
    memory_type: MemoryType = Field(..., description="Type of memory")
    
    # Memory content
    content: str = Field(..., description="Memory content")
    structured_data: Optional[Dict[str, Any]] = Field(None, description="Structured memory data")
    tags: List[str] = Field(default_factory=list, description="Memory tags for categorization")
    
    # Memory metadata
    importance: float = Field(default=0.5, ge=0, le=1, description="Memory importance score")
    confidence: float = Field(default=1.0, ge=0, le=1, description="Confidence in memory accuracy")
    access_count: int = Field(default=0, description="Number of times accessed")
    
    # Context and associations
    context: str = Field(..., description="Context when memory was formed")
    associated_task_id: Optional[str] = Field(None, description="Associated task")
    related_memories: List[str] = Field(default_factory=list, description="Related memory entries")
    
    # Lifecycle management
    expires_at: Optional[datetime] = Field(None, description="When memory expires")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    last_modified: datetime = Field(default_factory=datetime.utcnow)

class LearningEvent(BaseModel):
    """Record of a learning event"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Learning event identifier")
    agent_id: str = Field(..., description="Agent that learned")
    
    # Learning content
    learning_type: str = Field(..., description="Type of learning")
    description: str = Field(..., description="What was learned")
    knowledge_gained: str = Field(..., description="Knowledge gained")
    
    # Learning context
    trigger_event: str = Field(..., description="What triggered the learning")
    success_outcome: bool = Field(..., description="Whether learning led to success")
    performance_improvement: Optional[float] = Field(None, description="Measured performance improvement")
    
    # Application and validation
    applied_in_tasks: List[str] = Field(default_factory=list, description="Tasks where learning was applied")
    validation_results: Optional[Dict[str, Any]] = Field(None, description="Validation of learning")
    
    # Timestamps
    occurred_at: datetime = Field(default_factory=datetime.utcnow)

# Coordination and Orchestration Models

class AgentCoordination(BaseModel):
    """Coordination configuration for multi-agent tasks"""
    coordination_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Coordination identifier")
    strategy: CoordinationStrategy = Field(..., description="Coordination strategy")
    
    # Participating agents
    lead_agent_id: Optional[str] = Field(None, description="Lead agent if hierarchical")
    participating_agents: List[str] = Field(..., description="All participating agents")
    
    # Coordination rules
    communication_protocol: str = Field(default="broadcast", description="Communication protocol")
    decision_making_method: str = Field(default="consensus", description="How decisions are made")
    conflict_resolution: str = Field(default="majority_vote", description="How conflicts are resolved")
    
    # Performance tracking
    coordination_quality: Optional[float] = Field(None, ge=0, le=1, description="Quality of coordination")
    efficiency_score: Optional[float] = Field(None, ge=0, le=1, description="Coordination efficiency")
    
    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)

class AgentPerformanceMetrics(BaseModel):
    """Performance metrics for an agent"""
    agent_id: str = Field(..., description="Agent identifier")
    
    # Task performance
    tasks_completed: int = Field(default=0, description="Total tasks completed")
    tasks_successful: int = Field(default=0, description="Successful tasks")
    success_rate: float = Field(default=0.0, ge=0, le=1, description="Success rate")
    average_completion_time: float = Field(default=0.0, description="Average task completion time")
    
    # Quality metrics
    average_confidence: float = Field(default=0.0, ge=0, le=1, description="Average confidence in results")
    quality_score: float = Field(default=0.0, ge=0, le=1, description="Overall quality score")
    innovation_score: float = Field(default=0.0, ge=0, le=1, description="Innovation/creativity score")
    
    # Collaboration metrics
    messages_sent: int = Field(default=0, description="Messages sent to other agents")
    messages_received: int = Field(default=0, description="Messages received from other agents")
    collaboration_rating: float = Field(default=0.0, ge=0, le=1, description="Collaboration effectiveness")
    
    # Learning metrics
    learning_events: int = Field(default=0, description="Number of learning events")
    knowledge_gained: float = Field(default=0.0, description="Amount of knowledge gained")
    adaptation_rate: float = Field(default=0.0, ge=0, le=1, description="Rate of adaptation to new situations")
    
    # Timestamps
    measurement_period_start: datetime = Field(..., description="Start of measurement period")
    measurement_period_end: datetime = Field(..., description="End of measurement period")
    last_updated: datetime = Field(default_factory=datetime.utcnow)

# Request and Response Models for Agentic Operations

class AgenticAnalysisRequest(BaseModel):
    """Request for agentic analysis"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Request identifier")
    
    # Analysis configuration
    coordination_strategy: CoordinationStrategy = Field(default=CoordinationStrategy.COLLABORATIVE, description="How agents should coordinate")
    required_agent_roles: List[AgentRole] = Field(..., description="Required agent roles for analysis")
    optional_agent_roles: List[AgentRole] = Field(default_factory=list, description="Optional agent roles")
    
    # Analysis parameters
    analysis_depth: str = Field(default="comprehensive", description="Depth of analysis")
    focus_areas: List[str] = Field(default_factory=list, description="Specific areas to focus on")
    constraints: List[str] = Field(default_factory=list, description="Analysis constraints")
    
    # Tool and capability requirements
    required_tools: List[ToolType] = Field(default_factory=list, description="Required tools")
    enable_learning: bool = Field(default=True, description="Whether agents should learn from this analysis")
    enable_memory: bool = Field(default=True, description="Whether to use agent memory")
    
    # Performance requirements
    max_execution_time: int = Field(default=300, description="Maximum execution time in seconds")
    quality_threshold: float = Field(default=0.7, ge=0, le=1, description="Minimum quality threshold")
    
    # Input data (inherited from workflow integration)
    input_data: Dict[str, Any] = Field(..., description="Input data for analysis")
    
    # Timestamps
    submitted_at: datetime = Field(default_factory=datetime.utcnow)

class AgentContribution(BaseModel):
    """Individual agent's contribution to analysis"""
    agent_id: str = Field(..., description="Agent identifier")
    agent_role: AgentRole = Field(..., description="Agent role")
    
    # Contribution content
    findings: List[str] = Field(..., description="Key findings from this agent")
    insights: List[str] = Field(..., description="Insights provided")
    recommendations: List[str] = Field(..., description="Recommendations from this agent")
    evidence: List[str] = Field(..., description="Supporting evidence")
    
    # Analysis methodology
    methodology: str = Field(..., description="Methodology used")
    tools_used: List[str] = Field(..., description="Tools used by agent")
    confidence_level: float = Field(..., ge=0, le=1, description="Agent's confidence in contribution")
    
    # Reasoning and transparency
    reasoning_chain: List[str] = Field(..., description="Step-by-step reasoning")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions made")
    limitations: List[str] = Field(default_factory=list, description="Acknowledged limitations")
    
    # Performance metrics
    processing_time_ms: float = Field(..., description="Time taken for this contribution")
    memory_entries_accessed: int = Field(default=0, description="Memory entries accessed")
    new_knowledge_gained: List[str] = Field(default_factory=list, description="New knowledge gained")
    
    # Timestamps
    started_at: datetime = Field(..., description="When agent started working")
    completed_at: datetime = Field(..., description="When agent completed work")

class AgenticAnalysisResponse(BaseModel):
    """Response from agentic analysis"""
    request_id: str = Field(..., description="Original request identifier")
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Analysis identifier")
    
    # Coordination results
    coordination_strategy_used: CoordinationStrategy = Field(..., description="Coordination strategy used")
    agents_involved: List[str] = Field(..., description="Agents that participated")
    coordination_quality: float = Field(..., ge=0, le=1, description="Quality of agent coordination")
    
    # Individual agent contributions
    agent_contributions: List[AgentContribution] = Field(..., description="Contributions from each agent")
    
    # Synthesized results
    executive_summary: str = Field(..., description="Executive summary from lead agent")
    consensus_findings: List[str] = Field(..., description="Findings all agents agree on")
    conflicting_views: List[str] = Field(default_factory=list, description="Areas where agents disagree")
    
    # Evidence and reasoning
    evidence_base: List[str] = Field(..., description="All evidence supporting conclusions")
    reasoning_transparency: List[str] = Field(..., description="High-level reasoning steps")
    cross_validation_results: Dict[str, float] = Field(..., description="Cross-validation between agents")
    
    # Quality and confidence metrics
    overall_confidence: float = Field(..., ge=0, le=1, description="Overall confidence in analysis")
    quality_score: float = Field(..., ge=0, le=1, description="Analysis quality score")
    innovation_score: float = Field(..., ge=0, le=1, description="Innovation/creativity score")
    
    # Performance metrics
    total_processing_time_ms: float = Field(..., description="Total processing time")
    coordination_efficiency: float = Field(..., ge=0, le=1, description="Coordination efficiency")
    tools_usage_summary: Dict[str, int] = Field(..., description="Summary of tools used")
    
    # Learning and adaptation
    learning_events_recorded: int = Field(..., description="Learning events during analysis")
    memory_entries_created: int = Field(..., description="New memory entries created")
    knowledge_improvements: List[str] = Field(default_factory=list, description="Knowledge improvements made")
    
    # Timestamps
    analysis_started_at: datetime = Field(..., description="When analysis started")
    analysis_completed_at: datetime = Field(..., description="When analysis completed") 