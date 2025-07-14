#!/usr/bin/env python3
"""
Comprehensive Multi-Training Monitor

This script monitors multiple training jobs simultaneously:
- Enhanced training jobs
- Smart training jobs  
- Database activity
- Feature extraction progress
- Model performance metrics

Usage:
    python scripts/monitor_all_training.py
"""

import requests
import time
import json
from datetime import datetime
from typing import Dict, List, Any
import sys

class MultiTrainingMonitor:
    def __init__(self, base_url: str = "http://localhost:8003"):
        self.base_url = base_url
        self.start_time = datetime.now()
        self.tracked_jobs = {
            'enhanced': [],
            'smart': []
        }

    def add_enhanced_training(self, training_id: str, description: str = ""):
        """Add an enhanced training job to monitor"""
        self.tracked_jobs['enhanced'].append({
            'training_id': training_id,
            'description': description,
            'added_at': datetime.now(),
            'last_status': None
        })

    def add_smart_training(self, training_id: str, description: str = ""):
        """Add a smart training job to monitor"""
        self.tracked_jobs['smart'].append({
            'training_id': training_id,
            'description': description,
            'added_at': datetime.now(),
            'last_status': None
        })

    def get_enhanced_status(self, training_id: str) -> Dict[str, Any]:
        """Get status of enhanced training job"""
        try:
            response = requests.get(f"{self.base_url}/enhanced/status/{training_id}", timeout=10)
            if response.status_code == 200:
                return response.json()
            return {"status": "error", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_smart_status(self, training_id: str) -> Dict[str, Any]:
        """Get status of smart training job"""
        try:
            response = requests.get(f"{self.base_url}/smart/status/{training_id}", timeout=10)
            if response.status_code == 200:
                return response.json()
            return {"status": "error", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_database_activity(self) -> Dict[str, Any]:
        """Get current database activity using our progress tracker"""
        from training_progress_tracker import TrainingProgressTracker
        tracker = TrainingProgressTracker()
        return tracker.get_database_status()

    def check_models_directory(self) -> Dict[str, Any]:
        """Check what models have been saved"""
        try:
            response = requests.get(f"{self.base_url}/enhanced/models", timeout=10)
            if response.status_code == 200:
                return response.json()
            return {"available_models": [], "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"available_models": [], "error": str(e)}

    def print_comprehensive_status(self):
        """Print comprehensive status of all training jobs"""
        print("\n" + "="*100)
        print("🎯 COMPREHENSIVE TRAINING MONITOR")
        print("="*100)
        print(f"📅 Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Current: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Enhanced Training Jobs
        if self.tracked_jobs['enhanced']:
            print(f"\n🚀 ENHANCED TRAINING JOBS ({len(self.tracked_jobs['enhanced'])} active):")
            print("-" * 80)
            
            for job in self.tracked_jobs['enhanced']:
                training_id = job['training_id']
                status = self.get_enhanced_status(training_id)
                current_status = status.get('status', 'unknown')
                
                # Only print if status changed
                if current_status != job['last_status']:
                    print(f"\n  📦 {training_id}")
                    print(f"     📝 Description: {job['description']}")
                    print(f"     📊 Status: {current_status}")
                    print(f"     ⏱️  Added: {job['added_at'].strftime('%H:%M:%S')}")
                    
                    if 'current_stage' in status:
                        print(f"     🎯 Current Stage: {status['current_stage']}")
                    if 'progress' in status:
                        progress = status['progress'] * 100
                        print(f"     📈 Progress: {progress:.1f}%")
                    if 'dataset_path' in status:
                        print(f"     📁 Dataset: {status['dataset_path'].split('/')[-1]}")
                    
                    # Check for completion details
                    if current_status == 'completed':
                        if 'models_path' in status:
                            print(f"     🤖 Models Saved: {status['models_path']}")
                        if 'registry_path' in status:
                            print(f"     📋 Registry: {status['registry_path']}")
                    elif current_status == 'failed':
                        print(f"     ❌ Error: {status.get('error', 'Unknown error')}")
                
                job['last_status'] = current_status

        # Smart Training Jobs
        if self.tracked_jobs['smart']:
            print(f"\n🧠 SMART TRAINING JOBS ({len(self.tracked_jobs['smart'])} active):")
            print("-" * 80)
            
            for job in self.tracked_jobs['smart']:
                training_id = job['training_id']
                status = self.get_smart_status(training_id)
                current_status = status.get('status', 'unknown')
                
                # Only print if status changed
                if current_status != job['last_status']:
                    print(f"\n  🧠 {training_id}")
                    print(f"     📝 Description: {job['description']}")
                    print(f"     📊 Status: {current_status}")
                    print(f"     ⏱️  Added: {job['added_at'].strftime('%H:%M:%S')}")
                    
                    if 'strategy' in status:
                        print(f"     🎯 Strategy: {status['strategy']}")
                    if 'models_trained' in status:
                        models = status['models_trained']
                        print(f"     🤖 Models Trained: {list(models.keys()) if models else 'None'}")
                    if 'performance_metrics' in status:
                        metrics = status['performance_metrics']
                        if metrics:
                            print(f"     📈 Performance:")
                            for model, perf in metrics.items():
                                r2 = perf.get('r2_score', 'N/A')
                                print(f"        {model}: R² = {r2:.4f}" if isinstance(r2, (int, float)) else f"        {model}: R² = {r2}")
                    
                    # Check for completion details
                    if current_status in ['completed', 'COMPLETED']:
                        print(f"     ✅ Training completed successfully!")
                    elif current_status in ['failed', 'FAILED']:
                        print(f"     ❌ Error: {status.get('error', 'Unknown error')}")
                
                job['last_status'] = current_status

        # Database Activity
        print(f"\n💾 DATABASE ACTIVITY:")
        print("-" * 80)
        db_status = self.get_database_activity()
        
        for db_type, status in db_status.items():
            if status["connection"] == "healthy":
                if db_type == "audio":
                    analysis = status["analysis_results"]
                    print(f"  🎵 Audio DB: {analysis['total']} total, {analysis['completed']} completed, {analysis['processing']} processing")
                    print(f"      💾 Cached Features: {status['cached_features']}")
                elif db_type == "content":
                    lyrics = status["lyrics_analysis"]
                    print(f"  📝 Content DB: {lyrics['total']} total, {lyrics['completed']} completed")
            else:
                print(f"  ❌ {db_type.upper()} DB: Connection {status['connection']}")

        # Available Models
        print(f"\n🤖 TRAINED MODELS:")
        print("-" * 80)
        models_info = self.check_models_directory()
        if 'available_models' in models_info and models_info['available_models']:
            for model in models_info['available_models']:
                print(f"  📦 {model.get('name', 'Unknown')}")
                if 'r2_score' in model:
                    print(f"      📈 R² Score: {model['r2_score']:.4f}")
                if 'created_at' in model:
                    print(f"      📅 Created: {model['created_at']}")
        else:
            print("  ⏳ No models available yet or still training...")

    def monitor_continuously(self, interval: int = 30):
        """Monitor all training jobs continuously"""
        print("🔄 Starting continuous monitoring...")
        print(f"📊 Tracking {len(self.tracked_jobs['enhanced'])} enhanced + {len(self.tracked_jobs['smart'])} smart training jobs")
        
        completed_jobs = set()
        
        try:
            while True:
                self.print_comprehensive_status()
                
                # Check if all jobs are completed
                all_completed = True
                for job_type in ['enhanced', 'smart']:
                    for job in self.tracked_jobs[job_type]:
                        job_id = f"{job_type}_{job['training_id']}"
                        if job_id not in completed_jobs:
                            status_func = self.get_enhanced_status if job_type == 'enhanced' else self.get_smart_status
                            status = status_func(job['training_id'])
                            current_status = status.get('status', 'unknown')
                            
                            if current_status in ['completed', 'COMPLETED']:
                                completed_jobs.add(job_id)
                                print(f"\n🎉 {job_type.upper()} training {job['training_id']} completed!")
                            elif current_status in ['failed', 'FAILED']:
                                completed_jobs.add(job_id)
                                print(f"\n❌ {job_type.upper()} training {job['training_id']} failed!")
                            else:
                                all_completed = False
                
                if all_completed and len(completed_jobs) == (len(self.tracked_jobs['enhanced']) + len(self.tracked_jobs['smart'])):
                    print(f"\n🎉 All training jobs completed!")
                    break
                
                print(f"\n⏳ Waiting {interval}s before next check...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")

def main():
    monitor = MultiTrainingMonitor()
    
    # Add current training jobs that we started
    monitor.add_enhanced_training(
        "filtered_comprehensive_ensemble_training_8", 
        "Audio-only training (467 songs) with Random Forest + XGBoost"
    )
    
    monitor.add_enhanced_training(
        "filtered_multimodal_ensemble_training_9",
        "Multimodal training (400 songs) with audio + lyrics features"
    )
    
    monitor.add_smart_training(
        "smart_training_4",
        "Smart ensemble training using filtered dataset"
    )
    
    print("🎯 Comprehensive Training Monitor Started")
    print(f"📊 Monitoring {len(monitor.tracked_jobs['enhanced'])} enhanced + {len(monitor.tracked_jobs['smart'])} smart training jobs")
    
    # Start continuous monitoring
    monitor.monitor_continuously(interval=30)

if __name__ == "__main__":
    main() 