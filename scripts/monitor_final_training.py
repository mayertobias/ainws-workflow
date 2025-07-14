#!/usr/bin/env python3
"""
Final Comprehensive Training Monitor

Monitors the final comprehensive training jobs:
- Audio-only training (467 songs)
- Multimodal training (400 songs)  
- Smart ensemble training
- Database activity tracking
- Model performance validation

Usage:
    python scripts/monitor_final_training.py
"""

import requests
import time
import json
from datetime import datetime
from typing import Dict, List, Any
import sys

class FinalTrainingMonitor:
    def __init__(self, base_url: str = "http://localhost:8003"):
        self.base_url = base_url
        self.start_time = datetime.now()
        
        # Current training jobs
        self.training_jobs = {
            'enhanced': [
                {
                    'training_id': 'filtered_comprehensive_audio_training_10',
                    'description': 'Audio-only ensemble (467 songs) - Random Forest',
                    'dataset_type': 'audio_only',
                    'expected_songs': 467,
                    'last_status': None
                },
                {
                    'training_id': 'filtered_comprehensive_multimodal_training_11',
                    'description': 'Multimodal ensemble (400 songs) - Audio + Lyrics',
                    'dataset_type': 'multimodal',
                    'expected_songs': 400,
                    'last_status': None
                }
            ],
            'smart': [
                {
                    'training_id': 'smart_training_5',
                    'description': 'Smart ensemble (467 songs) - Auto-delegation',
                    'dataset_type': 'smart_filtered',
                    'expected_songs': 467,
                    'last_status': None
                }
            ]
        }

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
        """Get current database activity"""
        from training_progress_tracker import TrainingProgressTracker
        tracker = TrainingProgressTracker()
        return tracker.get_database_status()

    def check_shared_models(self) -> Dict[str, Any]:
        """Check models in shared volume"""
        try:
            response = requests.get(f"{self.base_url}/enhanced/models", timeout=10)
            if response.status_code == 200:
                return response.json()
            return {"available_models": [], "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"available_models": [], "error": str(e)}

    def print_comprehensive_status(self):
        """Print comprehensive status of all training jobs"""
        print("\n" + "="*120)
        print("🎯 FINAL COMPREHENSIVE TRAINING MONITOR - ENSEMBLE ML MODELS")
        print("="*120)
        print(f"📅 Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Current: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Target: 467 audio-only + 400 multimodal analysis → Random Forest + XGBoost models")
        
        total_expected = 0
        total_completed = 0
        
        # Enhanced Training Jobs
        if self.training_jobs['enhanced']:
            print(f"\n🚀 ENHANCED TRAINING JOBS ({len(self.training_jobs['enhanced'])} active):")
            print("-" * 100)
            
            for job in self.training_jobs['enhanced']:
                training_id = job['training_id']
                status = self.get_enhanced_status(training_id)
                current_status = status.get('status', 'unknown')
                
                print(f"\n  📦 {training_id}")
                print(f"     📝 {job['description']}")
                print(f"     🎯 Target: {job['expected_songs']} songs ({job['dataset_type']})")
                print(f"     📊 Status: {current_status}")
                
                total_expected += job['expected_songs']
                
                if 'current_stage' in status:
                    print(f"     🎪 Stage: {status['current_stage']}")
                if 'progress' in status:
                    progress = status['progress'] * 100
                    print(f"     📈 Progress: {progress:.1f}%")
                
                # Show completion details
                if current_status == 'completed':
                    total_completed += job['expected_songs']
                    print(f"     ✅ COMPLETED SUCCESSFULLY!")
                    
                    if 'models_path' in status:
                        print(f"     🤖 Models: {status['models_path']}")
                    if 'final_training_id' in status:
                        print(f"     🔑 Final ID: {status['final_training_id']}")
                        
                elif current_status == 'failed':
                    print(f"     ❌ FAILED: {status.get('error', 'Unknown error')}")
                    
                elif current_status == 'processing':
                    print(f"     🔄 Processing...")
                
                job['last_status'] = current_status

        # Smart Training Jobs
        if self.training_jobs['smart']:
            print(f"\n🧠 SMART TRAINING JOBS ({len(self.training_jobs['smart'])} active):")
            print("-" * 100)
            
            for job in self.training_jobs['smart']:
                training_id = job['training_id']
                status = self.get_smart_status(training_id)
                current_status = status.get('status', 'unknown')
                
                print(f"\n  🧠 {training_id}")
                print(f"     📝 {job['description']}")
                print(f"     🎯 Target: {job['expected_songs']} songs")
                print(f"     📊 Status: {current_status}")
                
                if 'strategy' in status:
                    print(f"     🎪 Strategy: {status['strategy']}")
                if 'models_trained' in status:
                    models = status['models_trained']
                    print(f"     🤖 Models: {list(models.keys()) if models else 'None'}")
                if 'performance_metrics' in status:
                    metrics = status['performance_metrics']
                    if metrics:
                        print(f"     📈 Performance:")
                        for model, perf in metrics.items():
                            r2 = perf.get('r2_score', 'N/A')
                            if isinstance(r2, (int, float)):
                                print(f"        {model}: R² = {r2:.4f}")
                            else:
                                print(f"        {model}: R² = {r2}")
                
                # Show completion details
                if current_status in ['completed', 'COMPLETED']:
                    print(f"     ✅ COMPLETED SUCCESSFULLY!")
                elif current_status in ['failed', 'FAILED']:
                    print(f"     ❌ FAILED: {status.get('error', 'Unknown error')}")
                
                job['last_status'] = current_status

        # Database Activity Summary
        print(f"\n💾 DATABASE ACTIVITY SUMMARY:")
        print("-" * 100)
        db_status = self.get_database_activity()
        
        total_analysis = 0
        total_features = 0
        
        for db_type, status in db_status.items():
            if status["connection"] == "healthy":
                if db_type == "audio":
                    analysis = status["analysis_results"]
                    total_analysis += analysis['completed']
                    total_features += status['cached_features']
                    print(f"  🎵 Audio Analysis: {analysis['completed']}/{analysis['total']} completed")
                    print(f"      💾 Features Cached: {status['cached_features']}")
                    if analysis['failed'] > 0:
                        print(f"      ❌ Failed: {analysis['failed']}")
                elif db_type == "content":
                    lyrics = status["lyrics_analysis"]
                    print(f"  📝 Lyrics Analysis: {lyrics['completed']}/{lyrics['total']} completed")
                    if lyrics['failed'] > 0:
                        print(f"      ❌ Failed: {lyrics['failed']}")
            else:
                print(f"  ❌ {db_type.upper()} DB: {status['connection']}")

        # Model Performance Summary
        print(f"\n🤖 TRAINED MODELS & SHARED VOLUME:")
        print("-" * 100)
        models_info = self.check_shared_models()
        
        if 'available_models' in models_info and models_info['available_models']:
            print(f"  ✅ Models Available: {len(models_info['available_models'])}")
            for model in models_info['available_models']:
                print(f"    📦 {model.get('name', 'Unknown')}")
                if 'r2_score' in model:
                    print(f"        📈 R² Score: {model['r2_score']:.4f}")
                if 'model_type' in model:
                    print(f"        🎯 Type: {model['model_type']}")
                if 'created_at' in model:
                    print(f"        📅 Created: {model['created_at']}")
        else:
            print("  ⏳ No models available yet...")

        # Progress Summary
        print(f"\n📊 COMPREHENSIVE PROGRESS SUMMARY:")
        print("-" * 100)
        print(f"  🎯 Expected Analysis: {total_expected} songs total")
        print(f"  ✅ Completed Analysis: {total_analysis} songs")
        print(f"  💾 Features Extracted: {total_features} cached")
        print(f"  📈 Progress Rate: {(total_analysis/total_expected*100):.1f}%" if total_expected > 0 else "  📈 Progress Rate: 0%")
        
        completion_rate = (total_completed / total_expected * 100) if total_expected > 0 else 0
        print(f"  🏁 Training Completion: {completion_rate:.1f}%")

    def monitor_until_completion(self, check_interval: int = 30):
        """Monitor until all training jobs complete"""
        print("🔄 Starting comprehensive monitoring until completion...")
        print(f"📊 Monitoring {len(self.training_jobs['enhanced'])} enhanced + {len(self.training_jobs['smart'])} smart training jobs")
        
        completed_jobs = set()
        
        try:
            while True:
                self.print_comprehensive_status()
                
                # Check completion status
                all_completed = True
                for job_type in ['enhanced', 'smart']:
                    for job in self.training_jobs[job_type]:
                        job_id = f"{job_type}_{job['training_id']}"
                        
                        if job_id not in completed_jobs:
                            status_func = self.get_enhanced_status if job_type == 'enhanced' else self.get_smart_status
                            status = status_func(job['training_id'])
                            current_status = status.get('status', 'unknown')
                            
                            if current_status in ['completed', 'COMPLETED']:
                                completed_jobs.add(job_id)
                                print(f"\n🎉 {job_type.upper()} training {job['training_id']} COMPLETED!")
                                
                                # Show final metrics for completed jobs
                                if job_type == 'enhanced' and 'final_training_id' in status:
                                    print(f"   🔑 Final Training ID: {status['final_training_id']}")
                                elif job_type == 'smart' and 'performance_metrics' in status:
                                    metrics = status['performance_metrics']
                                    if metrics:
                                        print(f"   📈 Final Performance:")
                                        for model, perf in metrics.items():
                                            r2 = perf.get('r2_score', 'N/A')
                                            if isinstance(r2, (int, float)):
                                                print(f"      {model}: R² = {r2:.4f}")
                                
                            elif current_status in ['failed', 'FAILED']:
                                completed_jobs.add(job_id)
                                print(f"\n❌ {job_type.upper()} training {job['training_id']} FAILED!")
                                print(f"   Error: {status.get('error', 'Unknown error')}")
                            else:
                                all_completed = False
                
                if all_completed and len(completed_jobs) >= (len(self.training_jobs['enhanced']) + len(self.training_jobs['smart'])):
                    print(f"\n🎉🎉🎉 ALL COMPREHENSIVE TRAINING COMPLETED! 🎉🎉🎉")
                    
                    # Final summary
                    db_status = self.get_database_activity()
                    models_info = self.check_shared_models()
                    
                    print(f"\n📋 FINAL RESULTS:")
                    print("-" * 100)
                    for db_type, status in db_status.items():
                        if status["connection"] == "healthy" and db_type == "audio":
                            analysis = status["analysis_results"]
                            print(f"  🎵 Total Audio Analysis: {analysis['completed']} completed")
                            print(f"  💾 Total Features Cached: {status['cached_features']}")
                    
                    if 'available_models' in models_info and models_info['available_models']:
                        print(f"  🤖 Total Models Trained: {len(models_info['available_models'])}")
                        for model in models_info['available_models']:
                            if 'r2_score' in model:
                                print(f"     📈 {model.get('name', 'Model')}: R² = {model['r2_score']:.4f}")
                    
                    break
                
                print(f"\n⏳ Next check in {check_interval}s...")
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")

def main():
    monitor = FinalTrainingMonitor()
    
    print("🎯 FINAL COMPREHENSIVE TRAINING MONITOR")
    print("="*80)
    print("📊 Monitoring comprehensive ensemble training:")
    print("   • Audio-only training: 467 songs → Random Forest models")
    print("   • Multimodal training: 400 songs → Audio + Lyrics ensemble")
    print("   • Smart training: Auto-delegation with ensemble strategies")
    print("   • Target: Achieve R² > 0.7 for audio-only, R² > 0.8 for multimodal")
    
    # Start monitoring until completion
    monitor.monitor_until_completion(check_interval=30)

if __name__ == "__main__":
    main() 