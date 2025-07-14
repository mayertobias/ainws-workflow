"""
Workflow API endpoints for workflow-orchestrator service
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse

from ..models.workflow import (
    WorkflowExecutionRequest, WorkflowExecutionResponse, CustomWorkflowRequest,
    WorkflowStatusResponse, WorkflowListResponse, TaskRetryRequest, WorkflowStatus,
    WorkflowMetrics, WorkflowDefinition, WorkflowExecution, TaskExecution
)
from ..models.responses import SuccessResponse, ErrorResponse
from ..services.orchestrator import WorkflowOrchestrator
from ..config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Global orchestrator instance (in production, use dependency injection)
orchestrator = None

async def get_orchestrator() -> WorkflowOrchestrator:
    """Dependency to get orchestrator instance."""
    global orchestrator
    if orchestrator is None:
        orchestrator = WorkflowOrchestrator()
        await orchestrator.start_workers()
    return orchestrator

@router.post("/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(
    request: WorkflowExecutionRequest,
    background_tasks: BackgroundTasks,
    orch: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """
    Execute a workflow using a predefined template.
    
    Supports multiple workflow templates including comprehensive analysis,
    hit prediction, audio-only analysis, AI insights, and production analysis.
    """
    try:
        logger.info(f"Executing workflow template: {request.template_name}")
        
        # Validate input data
        if not request.input_data:
            raise HTTPException(
                status_code=400,
                detail="Input data is required for workflow execution"
            )
        
        # Execute workflow
        workflow_id = await orch.execute_workflow(request)
        
        # Get workflow for response
        workflow = orch.get_workflow_status(workflow_id)
        if not workflow:
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve workflow after creation"
            )
        
        # Estimate duration based on template
        estimated_duration = _estimate_workflow_duration(request.template_name)
        
        response = WorkflowExecutionResponse(
            workflow_id=workflow_id,
            status=workflow.status,
            created_at=workflow.created_at,
            estimated_duration_seconds=estimated_duration,
            tracking_url=f"/workflows/{workflow_id}/status"
        )
        
        # Log workflow execution in background
        background_tasks.add_task(_log_workflow_execution, request, workflow_id)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute workflow: {str(e)}"
        )

@router.post("/execute/custom", response_model=WorkflowExecutionResponse)
async def execute_custom_workflow(
    request: CustomWorkflowRequest,
    background_tasks: BackgroundTasks,
    orch: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """
    Execute a custom workflow with user-defined tasks.
    
    Allows for flexible workflow definitions with custom task sequences,
    dependencies, and execution strategies.
    """
    try:
        logger.info(f"Executing custom workflow: {request.name}")
        
        # Validate custom workflow
        if not request.tasks:
            raise HTTPException(
                status_code=400,
                detail="At least one task must be defined for custom workflow"
            )
        
        # Validate task dependencies
        task_ids = {task.task_id for task in request.tasks}
        for task in request.tasks:
            for dep in task.dependencies:
                if dep not in task_ids:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Task '{task.task_id}' depends on unknown task '{dep}'"
                    )
        
        # Execute custom workflow
        workflow_id = await orch.execute_custom_workflow(request)
        
        # Get workflow for response
        workflow = orch.get_workflow_status(workflow_id)
        if not workflow:
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve custom workflow after creation"
            )
        
        response = WorkflowExecutionResponse(
            workflow_id=workflow_id,
            status=workflow.status,
            created_at=workflow.created_at,
            estimated_duration_seconds=request.timeout_seconds,
            tracking_url=f"/workflows/{workflow_id}/status"
        )
        
        # Log custom workflow execution
        background_tasks.add_task(_log_custom_workflow_execution, request, workflow_id)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing custom workflow: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute custom workflow: {str(e)}"
        )

@router.get("/{workflow_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(
    workflow_id: str,
    orch: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """Get detailed status of a specific workflow."""
    try:
        workflow = orch.get_workflow_status(workflow_id)
        if not workflow:
            raise HTTPException(
                status_code=404,
                detail=f"Workflow {workflow_id} not found"
            )
        
        # Get current and next tasks
        current_tasks = [task for task in workflow.tasks if task.status == "running"]
        
        # Determine next tasks (pending tasks with satisfied dependencies)
        next_tasks = []
        for task in workflow.tasks:
            if task.status == "pending":
                # Check if dependencies are satisfied
                deps_satisfied = True
                for dep_task_id in _get_task_dependencies(workflow, task.task_id):
                    dep_task = next((t for t in workflow.tasks if t.task_id == dep_task_id), None)
                    if not dep_task or dep_task.status != "completed":
                        deps_satisfied = False
                        break
                
                if deps_satisfied:
                    next_tasks.append(task.task_id)
        
        # Count completed tasks
        completed_tasks = [task for task in workflow.tasks if task.status == "completed"]
        
        return WorkflowStatusResponse(
            workflow=workflow,
            current_tasks=current_tasks,
            next_tasks=next_tasks,
            completed_tasks_count=len(completed_tasks),
            total_tasks_count=len(workflow.tasks)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get workflow status: {str(e)}"
        )

@router.get("/", response_model=WorkflowListResponse)
async def list_workflows(
    status: Optional[WorkflowStatus] = Query(default=None, description="Filter by workflow status"),
    limit: int = Query(default=50, le=100, description="Maximum number of workflows to return"),
    page: int = Query(default=1, ge=1, description="Page number for pagination"),
    orch: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """List workflows with optional filtering and pagination."""
    try:
        # Calculate offset
        offset = (page - 1) * limit
        
        # Get workflows from orchestrator
        all_workflows = orch.list_workflows(status=status, limit=limit + offset)
        
        # Apply pagination
        workflows = all_workflows[offset:offset + limit]
        
        # Determine if there are more pages
        has_next = len(all_workflows) > offset + limit
        
        return WorkflowListResponse(
            workflows=workflows,
            total_count=len(all_workflows),
            page=page,
            page_size=limit,
            has_next=has_next
        )
        
    except Exception as e:
        logger.error(f"Error listing workflows: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list workflows: {str(e)}"
        )

@router.post("/{workflow_id}/cancel")
async def cancel_workflow(
    workflow_id: str,
    orch: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """Cancel a running or pending workflow."""
    try:
        success = await orch.cancel_workflow(workflow_id)
        
        if success:
            return SuccessResponse(
                message=f"Workflow {workflow_id} cancelled successfully",
                data={"workflow_id": workflow_id, "status": "cancelled"}
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel workflow {workflow_id}. It may not exist or already be completed."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling workflow: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel workflow: {str(e)}"
        )

@router.post("/{workflow_id}/retry")
async def retry_failed_task(
    workflow_id: str,
    request: TaskRetryRequest,
    orch: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """Retry a failed task in a workflow."""
    try:
        if request.workflow_id != workflow_id:
            raise HTTPException(
                status_code=400,
                detail="Workflow ID in URL and request body must match"
            )
        
        success = await orch.retry_failed_task(workflow_id, request.task_id)
        
        if success:
            return SuccessResponse(
                message=f"Task {request.task_id} queued for retry",
                data={
                    "workflow_id": workflow_id,
                    "task_id": request.task_id,
                    "reset_dependencies": request.reset_dependencies
                }
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot retry task {request.task_id}. It may not exist or not be in failed state."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrying task: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retry task: {str(e)}"
        )

@router.get("/templates")
async def list_workflow_templates(
    orch: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """List all available workflow templates."""
    try:
        templates = orch.get_workflow_templates()
        
        template_list = []
        for template in templates:
            template_info = {
                "template_name": template.template_name.value,
                "name": template.name,
                "description": template.description,
                "version": template.version,
                "timeout_seconds": template.timeout_seconds,
                "parallel_execution": template.parallel_execution,
                "task_count": len(template.tasks),
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "task_type": task.task_type.value,
                        "service_name": task.service_name,
                        "endpoint": task.endpoint,
                        "dependencies": task.dependencies,
                        "required": task.required
                    }
                    for task in template.tasks
                ]
            }
            template_list.append(template_info)
        
        return {
            "templates": template_list,
            "total_templates": len(template_list),
            "default_templates": settings.DEFAULT_WORKFLOW_TEMPLATES
        }
        
    except Exception as e:
        logger.error(f"Error listing workflow templates: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list workflow templates: {str(e)}"
        )

@router.get("/metrics", response_model=WorkflowMetrics)
async def get_workflow_metrics(
    orch: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """Get workflow execution metrics and statistics."""
    try:
        workflows = orch.list_workflows(limit=1000)  # Get more for metrics
        
        # Calculate metrics
        total_workflows = len(workflows)
        active_workflows = len([w for w in workflows if w.status in ["pending", "running"]])
        completed_workflows = len([w for w in workflows if w.status == "completed"])
        failed_workflows = len([w for w in workflows if w.status == "failed"])
        
        # Calculate average duration for completed workflows
        completed_with_duration = [w for w in workflows if w.status == "completed" and w.duration_seconds]
        avg_duration = (
            sum(w.duration_seconds for w in completed_with_duration) / len(completed_with_duration)
            if completed_with_duration else 0
        )
        
        # Calculate success rate
        success_rate = completed_workflows / total_workflows if total_workflows > 0 else 0
        
        # Popular templates
        template_counts = {}
        for workflow in workflows:
            template_name = workflow.template_name.value
            template_counts[template_name] = template_counts.get(template_name, 0) + 1
        
        popular_templates = [
            {"template": template, "count": count}
            for template, count in sorted(template_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # Service utilization (simplified)
        service_utilization = {
            "audio": 0.7,  # Would calculate from actual task executions
            "content": 0.8,
            "ml_prediction": 0.6,
            "intelligence": 0.5
        }
        
        return WorkflowMetrics(
            total_workflows=total_workflows,
            active_workflows=active_workflows,
            completed_workflows=completed_workflows,
            failed_workflows=failed_workflows,
            average_duration_seconds=avg_duration,
            success_rate=success_rate,
            popular_templates=popular_templates,
            service_utilization=service_utilization
        )
        
    except Exception as e:
        logger.error(f"Error getting workflow metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get workflow metrics: {str(e)}"
        )

@router.delete("/cleanup")
async def cleanup_old_workflows(
    retention_hours: int = Query(default=24, ge=1, le=168, description="Hours to retain completed workflows"),
    orch: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """Clean up old completed workflows."""
    try:
        removed_count = await orch.cleanup_completed_workflows(retention_hours)
        
        return SuccessResponse(
            message=f"Cleaned up {removed_count} old workflows",
            data={
                "removed_count": removed_count,
                "retention_hours": retention_hours
            }
        )
        
    except Exception as e:
        logger.error(f"Error cleaning up workflows: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clean up workflows: {str(e)}"
        )

# Helper functions

def _estimate_workflow_duration(template_name) -> int:
    """Estimate workflow duration based on template."""
    duration_estimates = {
        "comprehensive_analysis": 1800,  # 30 minutes
        "hit_prediction": 600,           # 10 minutes
        "audio_only_analysis": 300,      # 5 minutes
        "ai_insights_only": 300,         # 5 minutes
        "production_analysis": 600       # 10 minutes
    }
    
    return duration_estimates.get(template_name.value, 900)  # Default 15 minutes

def _get_task_dependencies(workflow: WorkflowExecution, task_id: str) -> List[str]:
    """Get dependencies for a specific task."""
    # In a real implementation, this would come from the workflow template
    # For now, return empty list as we don't store template info with execution
    return []

async def _log_workflow_execution(request: WorkflowExecutionRequest, workflow_id: str):
    """Log workflow execution for analytics."""
    try:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "workflow_id": workflow_id,
            "template_name": request.template_name.value,
            "priority": request.priority.value,
            "user_id": request.user_id,
            "tags": request.tags,
            "input_data_keys": list(request.input_data.keys()) if request.input_data else []
        }
        
        logger.info(f"Workflow execution logged: {log_data}")
        
    except Exception as e:
        logger.error(f"Error logging workflow execution: {e}")

async def _log_custom_workflow_execution(request: CustomWorkflowRequest, workflow_id: str):
    """Log custom workflow execution for analytics."""
    try:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "workflow_id": workflow_id,
            "workflow_name": request.name,
            "task_count": len(request.tasks),
            "parallel_execution": request.parallel_execution,
            "timeout_seconds": request.timeout_seconds,
            "priority": request.priority.value
        }
        
        logger.info(f"Custom workflow execution logged: {log_data}")
        
    except Exception as e:
        logger.error(f"Error logging custom workflow execution: {e}") 