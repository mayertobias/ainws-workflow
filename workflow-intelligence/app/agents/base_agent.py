"""
Base Agent Class for Agentic AI System

This module provides the foundational BaseAgent class that all specialized agents inherit from.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Set

from ..models.agentic_models import (
    AgentProfile, AgentStatus, AgentRole, ToolType, ToolExecution,
    AgentTask, InterAgentMessage, AgentContribution, ToolDefinition
)

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Base class for all intelligent agents in the system.
    
    Provides core functionality including:
    - Task execution and management
    - Tool usage and coordination
    - Inter-agent communication
    - Memory management
    """
    
    def __init__(self, profile: AgentProfile):
        """Initialize the base agent."""
        self.profile = profile
        self.current_tasks: Dict[str, AgentTask] = {}
        self.tool_executions: List[ToolExecution] = []
        
        # Performance tracking
        self.tasks_completed = 0
        self.tasks_successful = 0
        self.total_processing_time = 0.0
        
        logger.info(f"Initialized {self.profile.role.value} agent: {self.profile.name}")
    
    # Abstract Methods - Must be implemented by specialized agents
    
    @abstractmethod
    async def analyze_task(self, request: Dict[str, Any], task: AgentTask) -> AgentContribution:
        """Perform specialized analysis based on agent's role."""
        pass
    
    @abstractmethod
    def get_expertise_areas(self) -> List[str]:
        """Get the agent's areas of expertise."""
        pass
    
    @abstractmethod
    def get_preferred_tools(self) -> List[ToolType]:
        """Get the agent's preferred tools."""
        pass
    
    # Core Agent Functionality
    
    async def execute_task(self, task: AgentTask, request: Dict[str, Any]) -> AgentContribution:
        """Execute a task assigned to this agent."""
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Agent {self.profile.name} starting task: {task.task_id}")
            
            # Update agent and task status
            self.profile.status = AgentStatus.WORKING
            task.status = AgentStatus.WORKING
            task.started_at = start_time
            
            # Store current task
            self.current_tasks[task.task_id] = task
            
            # Perform the specialized analysis
            contribution = await self.analyze_task(request, task)
            
            # Update task status
            task.status = AgentStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.result = contribution.dict()
            
            # Update performance metrics
            self.tasks_completed += 1
            self.tasks_successful += 1
            processing_time = (task.completed_at - task.started_at).total_seconds() * 1000
            self.total_processing_time += processing_time
            
            return contribution
            
        except Exception as e:
            logger.error(f"Agent {self.profile.name} failed task {task.task_id}: {e}")
            
            # Update task status
            task.status = AgentStatus.ERROR
            task.completed_at = datetime.utcnow()
            self.tasks_completed += 1
            
            # Create error contribution
            error_contribution = AgentContribution(
                agent_id=self.profile.agent_id,
                agent_role=self.profile.role,
                findings=[f"Task failed with error: {str(e)}"],
                insights=[],
                recommendations=[],
                evidence=[],
                methodology="error_handling",
                tools_used=[],
                confidence_level=0.0,
                reasoning_chain=[f"Error occurred: {str(e)}"],
                processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                started_at=start_time,
                completed_at=datetime.utcnow()
            )
            
            return error_contribution
            
        finally:
            # Clean up
            self.profile.status = AgentStatus.IDLE
            self.profile.last_active = datetime.utcnow()
            if task.task_id in self.current_tasks:
                del self.current_tasks[task.task_id]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        success_rate = self.tasks_successful / max(self.tasks_completed, 1)
        avg_processing_time = self.total_processing_time / max(self.tasks_completed, 1)
        
        return {
            "agent_id": self.profile.agent_id,
            "agent_name": self.profile.name,
            "agent_role": self.profile.role.value,
            "tasks_completed": self.tasks_completed,
            "tasks_successful": self.tasks_successful,
            "success_rate": success_rate,
            "average_processing_time_ms": avg_processing_time,
            "tools_used": len(self.tool_executions),
            "status": self.profile.status.value,
            "last_active": self.profile.last_active
        }
