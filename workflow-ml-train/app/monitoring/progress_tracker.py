"""
Progress Tracker for real-time pipeline monitoring.

Provides real-time tracking of:
- Pipeline execution progress
- Stage-level progress  
- Performance metrics
- Resource utilization
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class ProgressTracker:
    """
    Real-time progress tracking for ML training pipelines.
    
    Tracks progress at both pipeline and stage levels with 
    performance metrics and resource monitoring.
    """
    
    def __init__(self):
        """Initialize the progress tracker"""
        self.pipeline_progress: Dict[str, Dict[str, Any]] = {}
        self.stage_progress: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics: Dict[str, Any] = {}
        self.resource_usage: Dict[str, Any] = {}
        
        # Performance tracking
        self.start_time = datetime.now()
        self.pipelines_completed = 0
        self.pipelines_failed = 0
        
        logger.info("📊 Progress Tracker initialized")
    
    def start_pipeline_tracking(
        self, 
        pipeline_id: str, 
        strategy: str, 
        total_stages: int = 5
    ):
        """Start tracking a new pipeline"""
        self.pipeline_progress[pipeline_id] = {
            "pipeline_id": pipeline_id,
            "strategy": strategy,
            "status": "starting",
            "start_time": datetime.now(),
            "total_stages": total_stages,
            "completed_stages": 0,
            "current_stage": None,
            "overall_progress": 0.0,
            "estimated_duration": 600,  # 10 minutes default
            "elapsed_time": 0,
            "eta": None
        }
        
        logger.info(f"📈 Started tracking pipeline {pipeline_id}")
    
    def update_pipeline_stage(
        self, 
        pipeline_id: str, 
        stage: str, 
        status: str = "running"
    ):
        """Update current pipeline stage"""
        if pipeline_id in self.pipeline_progress:
            progress = self.pipeline_progress[pipeline_id]
            progress["current_stage"] = stage
            progress["status"] = status
            
            # Update stage tracking
            stage_key = f"{pipeline_id}_{stage}"
            if stage_key not in self.stage_progress:
                self.stage_progress[stage_key] = {
                    "pipeline_id": pipeline_id,
                    "stage": stage,
                    "status": "running",
                    "progress_percent": 0.0,
                    "start_time": datetime.now(),
                    "messages": []
                }
            
            logger.debug(f"📊 Pipeline {pipeline_id}: Stage updated to '{stage}' ({status})")
    
    def update_stage_progress(
        self, 
        pipeline_id: str, 
        stage: str, 
        progress_percent: float,
        message: Optional[str] = None
    ):
        """Update progress for a specific stage"""
        stage_key = f"{pipeline_id}_{stage}"
        
        if stage_key in self.stage_progress:
            stage_progress = self.stage_progress[stage_key]
            stage_progress["progress_percent"] = progress_percent
            
            if message:
                stage_progress["messages"].append({
                    "timestamp": datetime.now(),
                    "message": message
                })
            
            # Update overall pipeline progress
            self._update_overall_progress(pipeline_id)
            
            logger.debug(f"📊 Pipeline {pipeline_id}, Stage {stage}: {progress_percent}%")
    
    def complete_stage(self, pipeline_id: str, stage: str):
        """Mark a stage as completed"""
        stage_key = f"{pipeline_id}_{stage}"
        
        if stage_key in self.stage_progress:
            self.stage_progress[stage_key]["status"] = "completed"
            self.stage_progress[stage_key]["progress_percent"] = 100.0
            self.stage_progress[stage_key]["end_time"] = datetime.now()
        
        # Update pipeline progress
        if pipeline_id in self.pipeline_progress:
            progress = self.pipeline_progress[pipeline_id]
            progress["completed_stages"] += 1
            self._update_overall_progress(pipeline_id)
        
        logger.info(f"✅ Pipeline {pipeline_id}: Stage '{stage}' completed")
    
    def fail_stage(self, pipeline_id: str, stage: str, error_message: str):
        """Mark a stage as failed"""
        stage_key = f"{pipeline_id}_{stage}"
        
        if stage_key in self.stage_progress:
            self.stage_progress[stage_key]["status"] = "failed"
            self.stage_progress[stage_key]["error_message"] = error_message
            self.stage_progress[stage_key]["end_time"] = datetime.now()
        
        # Update pipeline progress
        if pipeline_id in self.pipeline_progress:
            self.pipeline_progress[pipeline_id]["status"] = "failed"
            self.pipeline_progress[pipeline_id]["error_message"] = error_message
            self.pipelines_failed += 1
        
        logger.error(f"❌ Pipeline {pipeline_id}: Stage '{stage}' failed - {error_message}")
    
    def complete_pipeline(self, pipeline_id: str):
        """Mark a pipeline as completed"""
        if pipeline_id in self.pipeline_progress:
            progress = self.pipeline_progress[pipeline_id]
            progress["status"] = "completed" 
            progress["end_time"] = datetime.now()
            progress["overall_progress"] = 100.0
            
            # Calculate duration
            if "start_time" in progress:
                duration = progress["end_time"] - progress["start_time"]
                progress["total_duration"] = duration.total_seconds()
            
            self.pipelines_completed += 1
        
        logger.info(f"🎉 Pipeline {pipeline_id} completed successfully")
    
    def _update_overall_progress(self, pipeline_id: str):
        """Update overall pipeline progress based on stage completion"""
        if pipeline_id not in self.pipeline_progress:
            return
        
        progress = self.pipeline_progress[pipeline_id]
        total_stages = progress["total_stages"]
        completed_stages = progress["completed_stages"]
        
        # Calculate progress from completed stages
        base_progress = (completed_stages / total_stages) * 100
        
        # Add progress from current stage
        current_stage = progress.get("current_stage")
        if current_stage:
            stage_key = f"{pipeline_id}_{current_stage}"
            if stage_key in self.stage_progress:
                current_stage_progress = self.stage_progress[stage_key]["progress_percent"]
                stage_weight = (1 / total_stages) * 100
                current_contribution = (current_stage_progress / 100) * stage_weight
                overall_progress = base_progress + current_contribution
            else:
                overall_progress = base_progress
        else:
            overall_progress = base_progress
        
        progress["overall_progress"] = min(overall_progress, 100.0)
        
        # Update ETA
        if overall_progress > 0:
            elapsed = (datetime.now() - progress["start_time"]).total_seconds()
            estimated_total = (elapsed / overall_progress) * 100
            eta_seconds = estimated_total - elapsed
            progress["eta"] = datetime.now() + timedelta(seconds=eta_seconds)
            progress["elapsed_time"] = elapsed
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive pipeline status"""
        if pipeline_id not in self.pipeline_progress:
            return None
        
        progress = self.pipeline_progress[pipeline_id].copy()
        
        # Add stage details
        stages = {}
        for stage_key, stage_data in self.stage_progress.items():
            if stage_data["pipeline_id"] == pipeline_id:
                stage_name = stage_data["stage"]
                stages[stage_name] = {
                    "status": stage_data["status"],
                    "progress": stage_data["progress_percent"],
                    "start_time": stage_data.get("start_time"),
                    "end_time": stage_data.get("end_time"),
                    "messages": stage_data.get("messages", [])[-5:]  # Last 5 messages
                }
        
        progress["stages"] = stages
        return progress
    
    def list_active_pipelines(self) -> List[Dict[str, Any]]:
        """List all actively tracked pipelines"""
        active = []
        for pipeline_id, progress in self.pipeline_progress.items():
            if progress["status"] in ["starting", "running"]:
                active.append({
                    "pipeline_id": pipeline_id,
                    "strategy": progress["strategy"], 
                    "status": progress["status"],
                    "current_stage": progress.get("current_stage"),
                    "overall_progress": progress["overall_progress"],
                    "start_time": progress["start_time"],
                    "eta": progress.get("eta")
                })
        return active
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall performance metrics"""
        total_pipelines = self.pipelines_completed + self.pipelines_failed
        success_rate = (self.pipelines_completed / total_pipelines * 100) if total_pipelines > 0 else 0
        
        # Calculate average duration for completed pipelines
        completed_durations = []
        for progress in self.pipeline_progress.values():
            if progress["status"] == "completed" and "total_duration" in progress:
                completed_durations.append(progress["total_duration"])
        
        avg_duration = sum(completed_durations) / len(completed_durations) if completed_durations else 0
        
        return {
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "total_pipelines": total_pipelines,
            "completed_pipelines": self.pipelines_completed,
            "failed_pipelines": self.pipelines_failed,
            "success_rate_percent": round(success_rate, 1),
            "average_duration_seconds": round(avg_duration, 1),
            "active_pipelines": len(self.list_active_pipelines())
        }
    
    def cleanup_completed_pipelines(self, max_age_hours: int = 24):
        """Clean up old pipeline tracking data"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Clean up old pipeline progress
        to_remove = []
        for pipeline_id, progress in self.pipeline_progress.items():
            if (progress["status"] in ["completed", "failed"] and 
                progress.get("end_time", datetime.now()) < cutoff_time):
                to_remove.append(pipeline_id)
        
        for pipeline_id in to_remove:
            del self.pipeline_progress[pipeline_id]
            
            # Clean up associated stage progress
            stage_keys_to_remove = [
                key for key in self.stage_progress.keys() 
                if key.startswith(f"{pipeline_id}_")
            ]
            for key in stage_keys_to_remove:
                del self.stage_progress[key]
        
        if to_remove:
            logger.info(f"🧹 Cleaned up {len(to_remove)} old pipeline records")
    
    async def start_metrics_collection(self):
        """Start background metrics collection"""
        logger.info("📊 Starting metrics collection")
        
        while True:
            try:
                # Update resource usage metrics
                self.resource_usage = {
                    "timestamp": datetime.now(),
                    "memory_usage_mb": 512,  # Mock data
                    "cpu_usage_percent": 45,
                    "disk_io_mb_per_sec": 10.2,
                    "network_io_mb_per_sec": 5.1
                }
                
                # Clean up old data periodically
                await asyncio.sleep(300)  # Every 5 minutes
                self.cleanup_completed_pipelines()
                
            except Exception as e:
                logger.error(f"❌ Metrics collection error: {e}")
                await asyncio.sleep(60)  # Wait before retrying 