"""
Core workflow orchestration service
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import httpx
from collections import defaultdict

from ..models.workflow import (
    WorkflowDefinition, WorkflowExecution, TaskExecution, WorkflowStatus, TaskStatus,
    WorkflowTemplate, TaskType, Priority, WorkflowExecutionRequest, CustomWorkflowRequest,
    TaskDefinition, SongAnalysisInput, ComprehensiveAnalysisOutput
)
from ..config.settings import settings

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """
    Core orchestration service for managing workflow execution.
    
    Handles workflow templates, task coordination, service communication,
    and execution state management.
    """
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        self.workflow_templates: Dict[WorkflowTemplate, WorkflowDefinition] = {}
        self.service_clients: Dict[str, httpx.AsyncClient] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.worker_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        
        # Initialize workflow templates
        self._initialize_workflow_templates()
        
        # Initialize service clients
        self._initialize_service_clients()
        
        logger.info("Workflow orchestrator initialized successfully")
    
    def _initialize_workflow_templates(self):
        """Initialize default workflow templates."""
        
        # Comprehensive Analysis Workflow
        comprehensive_tasks = [
            TaskDefinition(
                task_id="audio_analysis",
                task_type=TaskType.AUDIO_ANALYSIS,
                service_name="workflow-audio",
                endpoint="/analyze/comprehensive",
                parameters={"include_advanced_features": True},
                timeout_seconds=settings.AUDIO_ANALYSIS_TIMEOUT,
                dependencies=[]
            ),
            TaskDefinition(
                task_id="content_analysis",
                task_type=TaskType.CONTENT_ANALYSIS,
                service_name="workflow-content",
                endpoint="/analyze/lyrics",
                parameters={"include_sentiment": True, "include_features": True},
                timeout_seconds=settings.SERVICE_TIMEOUT_SECONDS,
                dependencies=[]
            ),
            TaskDefinition(
                task_id="hit_prediction",
                task_type=TaskType.ML_PREDICTION,
                service_name="workflow-ml-prediction",
                endpoint="/predict/single",
                parameters={"model_type": "hit_prediction", "include_explanation": True},
                timeout_seconds=settings.SERVICE_TIMEOUT_SECONDS,
                dependencies=["audio_analysis", "content_analysis"]
            ),
            TaskDefinition(
                task_id="ai_insights",
                task_type=TaskType.AI_INSIGHTS,
                service_name="workflow-intelligence",
                endpoint="/analyze/from-orchestrator",
                parameters={
                    "analysis_config": {
                        "depth": "comprehensive",
                        "focus_areas": ["musical_analysis", "commercial_potential", "strategic_recommendations"],
                        "business_context": "hit_song_science_analysis"
                    }
                },
                timeout_seconds=settings.AI_INSIGHTS_TIMEOUT,
                dependencies=["audio_analysis", "content_analysis", "hit_prediction"]
            )
        ]
        
        self.workflow_templates[WorkflowTemplate.COMPREHENSIVE_ANALYSIS] = WorkflowDefinition(
            template_name=WorkflowTemplate.COMPREHENSIVE_ANALYSIS,
            name="Comprehensive Song Analysis",
            description="Complete analysis including audio, content, ML prediction, and AI insights",
            tasks=comprehensive_tasks,
            timeout_seconds=1800,  # 30 minutes
            parallel_execution=True
        )
        
        # Hit Prediction Workflow
        hit_prediction_tasks = [
            TaskDefinition(
                task_id="audio_analysis",
                task_type=TaskType.AUDIO_ANALYSIS,
                service_name="workflow-audio",
                endpoint="/analyze/audio",
                parameters={},
                timeout_seconds=settings.AUDIO_ANALYSIS_TIMEOUT,
                dependencies=[]
            ),
            TaskDefinition(
                task_id="content_analysis",
                task_type=TaskType.CONTENT_ANALYSIS,
                service_name="workflow-content",
                endpoint="/analyze/features/hss",
                parameters={},
                timeout_seconds=settings.SERVICE_TIMEOUT_SECONDS,
                dependencies=[]
            ),
            TaskDefinition(
                task_id="hit_prediction",
                task_type=TaskType.ML_PREDICTION,
                service_name="workflow-ml-prediction",
                endpoint="/predict/single",
                parameters={"model_type": "hit_prediction", "include_confidence": True},
                timeout_seconds=settings.SERVICE_TIMEOUT_SECONDS,
                dependencies=["audio_analysis", "content_analysis"]
            )
        ]
        
        self.workflow_templates[WorkflowTemplate.HIT_PREDICTION] = WorkflowDefinition(
            template_name=WorkflowTemplate.HIT_PREDICTION,
            name="Hit Song Prediction",
            description="Predict commercial potential using audio and content analysis",
            tasks=hit_prediction_tasks,
            timeout_seconds=600,  # 10 minutes
            parallel_execution=True
        )
        
        # Audio Only Analysis
        audio_only_tasks = [
            TaskDefinition(
                task_id="audio_analysis",
                task_type=TaskType.AUDIO_ANALYSIS,
                service_name="workflow-audio",
                endpoint="/analyze/comprehensive",
                parameters={"include_advanced_features": True, "include_spectral": True},
                timeout_seconds=settings.AUDIO_ANALYSIS_TIMEOUT,
                dependencies=[]
            )
        ]
        
        self.workflow_templates[WorkflowTemplate.AUDIO_ONLY_ANALYSIS] = WorkflowDefinition(
            template_name=WorkflowTemplate.AUDIO_ONLY_ANALYSIS,
            name="Audio-Only Analysis",
            description="Comprehensive audio feature extraction and analysis",
            tasks=audio_only_tasks,
            timeout_seconds=300,  # 5 minutes
            parallel_execution=False
        )
        
        # AI Insights Only
        ai_insights_tasks = [
            TaskDefinition(
                task_id="ai_insights",
                task_type=TaskType.AI_INSIGHTS,
                service_name="workflow-intelligence",
                endpoint="/insights/generate",
                parameters={
                    "analysis_types": ["musical_meaning", "novelty_assessment", "production_feedback"],
                    "agent_type": "standard"
                },
                timeout_seconds=settings.AI_INSIGHTS_TIMEOUT,
                dependencies=[]
            )
        ]
        
        self.workflow_templates[WorkflowTemplate.AI_INSIGHTS_ONLY] = WorkflowDefinition(
            template_name=WorkflowTemplate.AI_INSIGHTS_ONLY,
            name="AI Insights Generation",
            description="Generate AI-powered insights and recommendations",
            tasks=ai_insights_tasks,
            timeout_seconds=300,  # 5 minutes
            parallel_execution=False
        )
        
        # Production Analysis
        production_tasks = [
            TaskDefinition(
                task_id="audio_analysis",
                task_type=TaskType.AUDIO_ANALYSIS,
                service_name="workflow-audio",
                endpoint="/analyze/comprehensive",
                parameters={"focus": "production_quality"},
                timeout_seconds=settings.AUDIO_ANALYSIS_TIMEOUT,
                dependencies=[]
            ),
            TaskDefinition(
                task_id="production_feedback",
                task_type=TaskType.AI_INSIGHTS,
                service_name="workflow-intelligence",
                endpoint="/insights/generate",
                parameters={
                    "analysis_types": ["production_feedback"],
                    "agent_type": "comprehensive"
                },
                timeout_seconds=settings.AI_INSIGHTS_TIMEOUT,
                dependencies=["audio_analysis"]
            )
        ]
        
        self.workflow_templates[WorkflowTemplate.PRODUCTION_ANALYSIS] = WorkflowDefinition(
            template_name=WorkflowTemplate.PRODUCTION_ANALYSIS,
            name="Production Quality Analysis",
            description="Analyze and provide feedback on production quality",
            tasks=production_tasks,
            timeout_seconds=600,  # 10 minutes
            parallel_execution=False
        )
        
        logger.info(f"Initialized {len(self.workflow_templates)} workflow templates")
    
    def _initialize_service_clients(self):
        """Initialize HTTP clients for microservices."""
        service_urls = settings.get_service_urls()
        timeouts = settings.get_service_timeouts()
        
        for service_name, url in service_urls.items():
            timeout = timeouts.get(service_name, settings.SERVICE_TIMEOUT_SECONDS)
            self.service_clients[service_name] = httpx.AsyncClient(
                base_url=url,
                timeout=httpx.Timeout(timeout),
                follow_redirects=True
            )
        
        logger.info(f"Initialized {len(self.service_clients)} service clients")
    
    async def start_workers(self, num_workers: int = 5):
        """Start background workers for task execution."""
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker_loop(worker_id=i))
            self.worker_tasks.append(worker)
        
        logger.info(f"Started {num_workers} workflow workers")
    
    async def stop_workers(self):
        """Stop all background workers."""
        self.shutdown_event.set()
        
        for task in self.worker_tasks:
            task.cancel()
        
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        
        logger.info("Stopped all workflow workers")
    
    async def _worker_loop(self, worker_id: int):
        """Background worker loop for processing workflows."""
        logger.info(f"Worker {worker_id} started")
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Get next workflow to process
                    workflow_id = await asyncio.wait_for(
                        self.task_queue.get(), 
                        timeout=1.0
                    )
                    
                    if workflow_id in self.active_workflows:
                        await self._execute_workflow(workflow_id)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                    
        except asyncio.CancelledError:
            pass
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def execute_workflow(self, request: WorkflowExecutionRequest) -> str:
        """Execute a workflow from a template."""
        try:
            # Get workflow template
            template = self.workflow_templates.get(request.template_name)
            if not template:
                raise ValueError(f"Unknown workflow template: {request.template_name}")
            
            # Create workflow execution
            workflow_id = str(uuid.uuid4())
            workflow = WorkflowExecution(
                workflow_id=workflow_id,
                template_name=request.template_name,
                status=WorkflowStatus.PENDING,
                created_at=datetime.utcnow(),
                input_data=request.input_data,
                metadata=request.metadata
            )
            
            # Create task executions
            for task_def in template.tasks:
                task_execution = TaskExecution(
                    task_id=task_def.task_id,
                    workflow_id=workflow_id,
                    status=TaskStatus.PENDING
                )
                workflow.tasks.append(task_execution)
            
            # Store workflow
            self.active_workflows[workflow_id] = workflow
            
            # Queue for execution
            await self.task_queue.put(workflow_id)
            
            logger.info(f"Queued workflow {workflow_id} for execution")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            raise
    
    async def execute_custom_workflow(self, request: CustomWorkflowRequest) -> str:
        """Execute a custom workflow."""
        try:
            # Create custom workflow definition
            custom_template = WorkflowDefinition(
                template_name=WorkflowTemplate.CUSTOM,
                name=request.name,
                description=request.description,
                tasks=request.tasks,
                timeout_seconds=request.timeout_seconds,
                parallel_execution=request.parallel_execution
            )
            
            # Create workflow execution
            workflow_id = str(uuid.uuid4())
            workflow = WorkflowExecution(
                workflow_id=workflow_id,
                template_name=WorkflowTemplate.CUSTOM,
                status=WorkflowStatus.PENDING,
                created_at=datetime.utcnow(),
                input_data=request.input_data,
                metadata=request.metadata
            )
            
            # Create task executions
            for task_def in custom_template.tasks:
                task_execution = TaskExecution(
                    task_id=task_def.task_id,
                    workflow_id=workflow_id,
                    status=TaskStatus.PENDING
                )
                workflow.tasks.append(task_execution)
            
            # Store workflow and template
            self.active_workflows[workflow_id] = workflow
            self.workflow_templates[WorkflowTemplate.CUSTOM] = custom_template
            
            # Queue for execution
            await self.task_queue.put(workflow_id)
            
            logger.info(f"Queued custom workflow {workflow_id} for execution")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Error executing custom workflow: {e}")
            raise
    
    async def _execute_workflow(self, workflow_id: str):
        """Execute a workflow."""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            logger.error(f"Workflow {workflow_id} not found")
            return
        
        try:
            logger.info(f"Starting workflow execution: {workflow_id}")
            
            # Update workflow status
            workflow.status = WorkflowStatus.RUNNING
            workflow.started_at = datetime.utcnow()
            
            # Get workflow template
            template = self.workflow_templates.get(workflow.template_name)
            if not template:
                raise ValueError(f"Template not found: {workflow.template_name}")
            
            # Execute tasks based on dependencies
            if template.parallel_execution:
                await self._execute_tasks_parallel(workflow, template)
            else:
                await self._execute_tasks_sequential(workflow, template)
            
            # Update final status
            failed_tasks = [t for t in workflow.tasks if t.status == TaskStatus.FAILED]
            if failed_tasks:
                workflow.status = WorkflowStatus.FAILED
                workflow.error_message = f"Failed tasks: {[t.task_id for t in failed_tasks]}"
            else:
                workflow.status = WorkflowStatus.COMPLETED
                await self._compile_workflow_output(workflow)
            
            workflow.completed_at = datetime.utcnow()
            workflow.duration_seconds = (
                workflow.completed_at - workflow.started_at
            ).total_seconds()
            
            # Calculate progress
            completed_tasks = [t for t in workflow.tasks if t.status == TaskStatus.COMPLETED]
            workflow.progress_percentage = (len(completed_tasks) / len(workflow.tasks)) * 100
            
            logger.info(f"Workflow {workflow_id} completed with status: {workflow.status}")
            
        except Exception as e:
            logger.error(f"Error executing workflow {workflow_id}: {e}")
            workflow.status = WorkflowStatus.FAILED
            workflow.error_message = str(e)
            workflow.completed_at = datetime.utcnow()
    
    async def _execute_tasks_parallel(self, workflow: WorkflowExecution, template: WorkflowDefinition):
        """Execute tasks in parallel based on dependencies."""
        task_map = {task.task_id: task for task in workflow.tasks}
        template_map = {task.task_id: task for task in template.tasks}
        
        # Track task dependencies
        dependency_graph = defaultdict(set)
        dependents = defaultdict(set)
        
        for task_def in template.tasks:
            for dep in task_def.dependencies:
                dependency_graph[task_def.task_id].add(dep)
                dependents[dep].add(task_def.task_id)
        
        completed_tasks = set()
        running_tasks = set()
        
        while len(completed_tasks) < len(workflow.tasks):
            # Find tasks ready to run
            ready_tasks = []
            for task_id, deps in dependency_graph.items():
                if (task_id not in completed_tasks and 
                    task_id not in running_tasks and
                    deps.issubset(completed_tasks)):
                    ready_tasks.append(task_id)
            
            if not ready_tasks:
                # Check if we're deadlocked
                remaining_tasks = set(task_map.keys()) - completed_tasks - running_tasks
                if remaining_tasks and not running_tasks:
                    # Deadlock detected
                    for task_id in remaining_tasks:
                        task_map[task_id].status = TaskStatus.FAILED
                        task_map[task_id].error_message = "Dependency deadlock"
                    break
                
                # Wait for running tasks to complete
                await asyncio.sleep(0.1)
                continue
            
            # Execute ready tasks
            task_coroutines = []
            for task_id in ready_tasks:
                running_tasks.add(task_id)
                coro = self._execute_task(workflow, task_map[task_id], template_map[task_id])
                task_coroutines.append(coro)
            
            # Wait for at least one task to complete
            if task_coroutines:
                done, pending = await asyncio.wait(
                    task_coroutines, 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for task_coro in done:
                    try:
                        task_result = await task_coro
                        task_id = task_result[0] if task_result else None
                        if task_id:
                            completed_tasks.add(task_id)
                            running_tasks.discard(task_id)
                    except Exception as e:
                        logger.error(f"Task execution error: {e}")
    
    async def _execute_tasks_sequential(self, workflow: WorkflowExecution, template: WorkflowDefinition):
        """Execute tasks sequentially."""
        task_map = {task.task_id: task for task in workflow.tasks}
        
        for task_def in template.tasks:
            task_execution = task_map[task_def.task_id]
            try:
                await self._execute_task(workflow, task_execution, task_def)
                
                # Stop on failure if required
                if (task_execution.status == TaskStatus.FAILED and 
                    task_def.required and 
                    template.failure_strategy == "stop_on_failure"):
                    break
                    
            except Exception as e:
                logger.error(f"Task {task_def.task_id} execution error: {e}")
                task_execution.status = TaskStatus.FAILED
                task_execution.error_message = str(e)
                
                if task_def.required:
                    break
    
    async def _execute_task(self, workflow: WorkflowExecution, task: TaskExecution, task_def: TaskDefinition) -> Tuple[str, bool]:
        """Execute a single task."""
        try:
            logger.info(f"Executing task {task.task_id} for workflow {workflow.workflow_id}")
            
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            
            # Get service client
            client = self.service_clients.get(task_def.service_name.replace("workflow-", ""))
            if not client:
                raise ValueError(f"No client for service: {task_def.service_name}")
            
            # Prepare request data
            request_data = self._prepare_task_request(workflow, task_def)
            
            # Execute service call
            if task_def.method.upper() == "GET":
                response = await client.get(task_def.endpoint, params=request_data)
            else:
                response = await client.post(task_def.endpoint, json=request_data)
            
            response.raise_for_status()
            result = response.json()
            
            # Store result
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.duration_seconds = (task.completed_at - task.started_at).total_seconds()
            
            logger.info(f"Task {task.task_id} completed successfully")
            return task.task_id, True
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
            task.duration_seconds = (task.completed_at - task.started_at).total_seconds()
            
            # Retry logic
            if task.retry_count < task_def.retry_attempts:
                task.retry_count += 1
                task.status = TaskStatus.RETRY
                logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count})")
                
                # Wait before retry
                await asyncio.sleep(settings.RETRY_DELAY_SECONDS)
                return await self._execute_task(workflow, task, task_def)
            
            return task.task_id, False
    
    def _prepare_task_request(self, workflow: WorkflowExecution, task_def: TaskDefinition) -> Dict[str, Any]:
        """Prepare request data for a task."""
        request_data = dict(task_def.parameters)
        
        # Special handling for AI insights task (intelligence service integration)
        if task_def.task_id == "ai_insights" and task_def.service_name == "workflow-intelligence":
            return self._prepare_intelligence_request(workflow, task_def)
        
        # Standard task preparation
        # Add workflow input data
        request_data.update(workflow.input_data)
        
        # Add results from dependent tasks
        for dep_task_id in task_def.dependencies:
            dep_task = next((t for t in workflow.tasks if t.task_id == dep_task_id), None)
            if dep_task and dep_task.result:
                request_data[f"{dep_task_id}_result"] = dep_task.result
        
        return request_data
    
    def _prepare_intelligence_request(self, workflow: WorkflowExecution, task_def: TaskDefinition) -> Dict[str, Any]:
        """Prepare request data specifically for the intelligence service."""
        try:
            # Get results from dependent tasks
            audio_result = None
            content_result = None
            prediction_result = None
            
            for dep_task_id in task_def.dependencies:
                dep_task = next((t for t in workflow.tasks if t.task_id == dep_task_id), None)
                if dep_task and dep_task.result:
                    if dep_task_id == "audio_analysis":
                        audio_result = dep_task.result
                    elif dep_task_id == "content_analysis":
                        content_result = dep_task.result
                    elif dep_task_id == "hit_prediction":
                        prediction_result = dep_task.result
            
            # Extract song metadata from workflow input
            song_metadata = {
                "title": workflow.input_data.get("title", "Unknown"),
                "artist": workflow.input_data.get("artist", "Unknown"),
                "album": workflow.input_data.get("album"),
                "genre": workflow.input_data.get("genre"),
                "duration_seconds": workflow.input_data.get("duration"),
                "file_path": workflow.input_data.get("file_path"),
                "language": workflow.input_data.get("language"),
            }
            
            # Format audio analysis data
            audio_analysis = self._format_audio_analysis(audio_result) if audio_result else {}
            
            # Format content analysis data
            content_analysis = self._format_content_analysis(content_result) if content_result else {}
            
            # Format hit prediction data
            hit_prediction = self._format_hit_prediction(prediction_result) if prediction_result else {}
            
            # Prepare the intelligence service request
            intelligence_request = {
                "audio_analysis": audio_analysis,
                "content_analysis": content_analysis,
                "hit_prediction": hit_prediction,
                "song_metadata": song_metadata,
                "analysis_config": task_def.parameters.get("analysis_config", {})
            }
            
            logger.info(f"Prepared intelligence request for workflow {workflow.workflow_id}")
            return intelligence_request
            
        except Exception as e:
            logger.error(f"Error preparing intelligence request: {e}")
            # Return minimal request structure to avoid complete failure
            return {
                "audio_analysis": {},
                "content_analysis": {},
                "hit_prediction": {},
                "song_metadata": {
                    "title": workflow.input_data.get("title", "Unknown"),
                    "artist": workflow.input_data.get("artist", "Unknown")
                },
                "analysis_config": task_def.parameters.get("analysis_config", {})
            }
    
    def _format_audio_analysis(self, audio_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format audio analysis result for intelligence service."""
        # Map the audio service response to the expected format
        return {
            "tempo": audio_result.get("tempo"),
            "energy": audio_result.get("energy"),
            "danceability": audio_result.get("danceability"),
            "valence": audio_result.get("valence"),
            "acousticness": audio_result.get("acousticness"),
            "loudness": audio_result.get("loudness"),
            "instrumentalness": audio_result.get("instrumentalness"),
            "liveness": audio_result.get("liveness"),
            "speechiness": audio_result.get("speechiness"),
            "spectral_features": audio_result.get("spectral_features", {}),
            "mfcc_features": audio_result.get("mfcc_features", []),
            "chroma_features": audio_result.get("chroma_features", []),
            "rhythm_features": audio_result.get("rhythm_features", {}),
            "harmonic_features": audio_result.get("harmonic_features", {}),
            "genre_predictions": audio_result.get("genre_predictions", {}),
            "mood_predictions": audio_result.get("mood_predictions", {}),
            "audio_quality_score": audio_result.get("audio_quality_score"),
            "analysis_timestamp": audio_result.get("timestamp", datetime.utcnow().isoformat()),
            "processing_time_ms": audio_result.get("processing_time_ms")
        }
    
    def _format_content_analysis(self, content_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format content analysis result for intelligence service."""
        # Map the content service response to the expected format
        return {
            "raw_lyrics": content_result.get("lyrics", content_result.get("raw_lyrics")),
            "processed_lyrics": content_result.get("processed_lyrics"),
            "sentiment_score": content_result.get("sentiment_score"),
            "emotion_scores": content_result.get("emotion_scores", {}),
            "mood_classification": content_result.get("mood_classification"),
            "language": content_result.get("language"),
            "complexity_score": content_result.get("complexity_score"),
            "readability_score": content_result.get("readability_score"),
            "word_count": content_result.get("word_count"),
            "unique_words": content_result.get("unique_words"),
            "topics": content_result.get("topics", []),
            "themes": content_result.get("themes", []),
            "keywords": content_result.get("keywords", []),
            "explicit_content": content_result.get("explicit_content"),
            "content_warnings": content_result.get("content_warnings", []),
            "target_audience": content_result.get("target_audience"),
            "analysis_timestamp": content_result.get("timestamp", datetime.utcnow().isoformat()),
            "processing_time_ms": content_result.get("processing_time_ms")
        }
    
    def _format_hit_prediction(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format hit prediction result for intelligence service."""
        # Map the ML prediction service response to the expected format
        return {
            "hit_probability": prediction_result.get("hit_probability", prediction_result.get("probability")),
            "confidence_score": prediction_result.get("confidence_score", prediction_result.get("confidence")),
            "genre_specific_score": prediction_result.get("genre_specific_score", {}),
            "market_predictions": prediction_result.get("market_predictions", {}),
            "demographic_scores": prediction_result.get("demographic_scores", {}),
            "feature_importance": prediction_result.get("feature_importance", {}),
            "top_contributing_features": prediction_result.get("top_contributing_features", []),
            "similar_hits": prediction_result.get("similar_hits", []),
            "genre_benchmarks": prediction_result.get("genre_benchmarks", {}),
            "commercial_risk_factors": prediction_result.get("commercial_risk_factors", []),
            "success_factors": prediction_result.get("success_factors", []),
            "model_version": prediction_result.get("model_version"),
            "training_data_size": prediction_result.get("training_data_size"),
            "model_accuracy": prediction_result.get("model_accuracy"),
            "prediction_timestamp": prediction_result.get("timestamp", datetime.utcnow().isoformat()),
            "processing_time_ms": prediction_result.get("processing_time_ms")
        }
    
    async def _compile_workflow_output(self, workflow: WorkflowExecution):
        """Compile final output from all completed tasks."""
        output = {
            "workflow_id": workflow.workflow_id,
            "template_name": workflow.template_name.value,
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
            "duration_seconds": workflow.duration_seconds,
            "task_results": {}
        }
        
        for task in workflow.tasks:
            if task.status == TaskStatus.COMPLETED and task.result:
                output["task_results"][task.task_id] = task.result
        
        workflow.output_data = output
    
    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution status."""
        return self.active_workflows.get(workflow_id)
    
    def list_workflows(self, status: Optional[WorkflowStatus] = None, limit: int = 100) -> List[WorkflowExecution]:
        """List workflows with optional filtering."""
        workflows = list(self.active_workflows.values())
        
        if status:
            workflows = [w for w in workflows if w.status == status]
        
        # Sort by creation time (newest first)
        workflows.sort(key=lambda w: w.created_at, reverse=True)
        
        return workflows[:limit]
    
    def get_workflow_templates(self) -> List[WorkflowDefinition]:
        """Get all available workflow templates."""
        return list(self.workflow_templates.values())
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return False
        
        if workflow.status in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]:
            workflow.status = WorkflowStatus.CANCELLED
            workflow.completed_at = datetime.utcnow()
            workflow.error_message = "Workflow cancelled by user"
            
            # Cancel running tasks
            for task in workflow.tasks:
                if task.status == TaskStatus.RUNNING:
                    task.status = TaskStatus.FAILED
                    task.error_message = "Cancelled"
                    task.completed_at = datetime.utcnow()
            
            logger.info(f"Cancelled workflow {workflow_id}")
            return True
        
        return False
    
    async def retry_failed_task(self, workflow_id: str, task_id: str) -> bool:
        """Retry a failed task."""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return False
        
        task = next((t for t in workflow.tasks if t.task_id == task_id), None)
        if not task or task.status != TaskStatus.FAILED:
            return False
        
        # Reset task status
        task.status = TaskStatus.PENDING
        task.retry_count = 0
        task.error_message = None
        task.started_at = None
        task.completed_at = None
        
        # Queue workflow for re-execution
        await self.task_queue.put(workflow_id)
        
        logger.info(f"Retrying task {task_id} in workflow {workflow_id}")
        return True
    
    async def cleanup_completed_workflows(self, retention_hours: int = 24):
        """Clean up old completed workflows."""
        cutoff_time = datetime.utcnow() - timedelta(hours=retention_hours)
        
        to_remove = []
        for workflow_id, workflow in self.active_workflows.items():
            if (workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED] and
                workflow.completed_at and workflow.completed_at < cutoff_time):
                to_remove.append(workflow_id)
        
        for workflow_id in to_remove:
            del self.active_workflows[workflow_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old workflows")
        
        return len(to_remove)
    
    async def close(self):
        """Clean up resources."""
        await self.stop_workers()
        
        for client in self.service_clients.values():
            await client.aclose()
        
        logger.info("Orchestrator closed") 