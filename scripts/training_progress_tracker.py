#!/usr/bin/env python3
"""
Comprehensive Training Progress Tracker for Workflow ML Training Service

This script provides real-time monitoring of:
- Database status and record counts
- Training job progress
- Model performance metrics
- Feature extraction progress
- Error tracking and recovery suggestions

Usage:
    python scripts/training_progress_tracker.py [options]
    
Options:
    --training-id ID        Monitor specific training job
    --continuous           Continuous monitoring mode
    --export-report        Export progress report to JSON
    --clean-mode          Clean all databases before starting
"""

import requests
import time
import json
import argparse
import psycopg2
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sys
import os

class TrainingProgressTracker:
    def __init__(self, base_url: str = "http://localhost:8003"):
        self.base_url = base_url
        self.start_time = datetime.now()
        self.tracking_history = []
        
        # Database connection configs
        self.db_configs = {
            'audio': {
                'host': 'localhost',
                'port': 5435,
                'user': 'postgres',
                'password': 'postgres',
                'database': 'workflow_audio'
            },
            'content': {
                'host': 'localhost',
                'port': 5433,
                'user': 'postgres', 
                'password': 'postgres',
                'database': 'workflow_content'
            }
        }

    def get_db_connection(self, db_type: str):
        """Get database connection"""
        try:
            config = self.db_configs[db_type]
            return psycopg2.connect(**config)
        except Exception as e:
            print(f"❌ Failed to connect to {db_type} database: {e}")
            return None

    def check_service_health(self) -> Dict[str, Any]:
        """Check if the ML training service is running"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "service_info": response.json()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "unreachable",
                "error": str(e)
            }

    def get_database_status(self) -> Dict[str, Any]:
        """Get comprehensive database status"""
        db_status = {}
        
        for db_type in ['audio', 'content']:
            conn = self.get_db_connection(db_type)
            if conn:
                try:
                    cursor = conn.cursor()
                    
                    if db_type == 'audio':
                        # Audio analysis results
                        try:
                            cursor.execute("""
                                SELECT 
                                    COUNT(*) as total_records,
                                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed,
                                    COUNT(CASE WHEN status = 'processing' THEN 1 END) as processing
                                FROM audio_analysis_results
                            """)
                            analysis_stats = cursor.fetchone()
                        except Exception:
                            # Table might not exist or have different schema
                            analysis_stats = (0, 0, 0, 0)
                        
                        # Feature cache status
                        try:
                            cursor.execute("SELECT COUNT(*) FROM feature_cache")
                            cached_features = cursor.fetchone()[0]
                        except Exception:
                            cached_features = 0
                        
                        # Training jobs status
                        try:
                            cursor.execute("""
                                SELECT 
                                    COUNT(*) as total_jobs,
                                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_jobs,
                                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_jobs,
                                    COUNT(CASE WHEN status = 'running' THEN 1 END) as running_jobs
                                FROM training_jobs 
                                WHERE created_at >= NOW() - INTERVAL '24 hours'
                            """)
                            training_stats = cursor.fetchone()
                        except Exception:
                            # Table might not exist or have different schema
                            training_stats = (0, 0, 0, 0)
                        
                        db_status[db_type] = {
                            "connection": "healthy",
                            "analysis_results": {
                                "total": analysis_stats[0],
                                "completed": analysis_stats[1],
                                "failed": analysis_stats[2],
                                "processing": analysis_stats[3]
                            },
                            "cached_features": cached_features,
                            "training_jobs_24h": {
                                "total": training_stats[0],
                                "completed": training_stats[1], 
                                "failed": training_stats[2],
                                "running": training_stats[3]
                            }
                        }
                        
                    elif db_type == 'content':
                        # Lyrics analysis results
                        try:
                            cursor.execute("""
                                SELECT 
                                    COUNT(*) as total_records,
                                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed
                                FROM lyrics_analysis_results
                            """)
                            lyrics_stats = cursor.fetchone()
                        except Exception:
                            # Table might not exist or have different schema
                            lyrics_stats = (0, 0, 0)
                        
                        db_status[db_type] = {
                            "connection": "healthy",
                            "lyrics_analysis": {
                                "total": lyrics_stats[0],
                                "completed": lyrics_stats[1],
                                "failed": lyrics_stats[2]
                            }
                        }
                        
                except Exception as e:
                    db_status[db_type] = {
                        "connection": "error",
                        "error": str(e)
                    }
                finally:
                    conn.close()
            else:
                db_status[db_type] = {
                    "connection": "failed",
                    "error": "Could not establish connection"
                }
        
        return db_status

    def clean_all_databases(self) -> Dict[str, Any]:
        """Clean all training-related data from databases"""
        print("🧹 Starting database cleanup...")
        cleanup_results = {}
        
        for db_type in ['audio', 'content']:
            conn = self.get_db_connection(db_type)
            if conn:
                try:
                    cursor = conn.cursor()
                    
                    if db_type == 'audio':
                        # Clean audio database tables
                        tables_to_clean = [
                            'audio_analysis_results',
                            'feature_cache', 
                            'training_jobs',
                            'model_registry',
                            'training_logs'
                        ]
                        
                        for table in tables_to_clean:
                            try:
                                cursor.execute(f"DELETE FROM {table}")
                                deleted_count = cursor.rowcount
                                print(f"  ✅ Cleaned {table}: {deleted_count} records deleted")
                            except Exception as e:
                                print(f"  ⚠️  Warning cleaning {table}: {e}")
                        
                    elif db_type == 'content':
                        # Clean content database tables
                        tables_to_clean = [
                            'lyrics_analysis_results',
                            'sentiment_cache',
                            'content_features'
                        ]
                        
                        for table in tables_to_clean:
                            try:
                                cursor.execute(f"DELETE FROM {table}")
                                deleted_count = cursor.rowcount
                                print(f"  ✅ Cleaned {table}: {deleted_count} records deleted")
                            except Exception as e:
                                print(f"  ⚠️  Warning cleaning {table}: {e}")
                    
                    conn.commit()
                    cleanup_results[db_type] = {"status": "success", "message": "Database cleaned successfully"}
                    
                except Exception as e:
                    conn.rollback()
                    cleanup_results[db_type] = {"status": "error", "error": str(e)}
                    print(f"❌ Error cleaning {db_type} database: {e}")
                finally:
                    conn.close()
            else:
                cleanup_results[db_type] = {"status": "failed", "error": "Could not connect to database"}
        
        return cleanup_results

    def start_smart_training(self, strategy: str = "smart_ensemble") -> Dict[str, Any]:
        """Start a new smart training job with ensemble approach"""
        print(f"🚀 Starting smart training with strategy: {strategy}")
        
        training_config = {
            "data_path": "/app/data/filtered/filtered_audio_only_r4a_song_data_training_20250613_113215.csv",
            "strategy": strategy,
            "config": {
                "enable_parallel_training": True,
                "use_load_balancing": True,
                "max_epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "ensemble_models": ["random_forest", "xgboost"],
                "cross_validation_folds": 5,
                "feature_selection": True,
                "hyperparameter_tuning": True
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/smart/train",
                json=training_config,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Training started successfully: {result.get('training_id')}")
                return result
            else:
                error_msg = f"Failed to start training: HTTP {response.status_code}"
                print(f"❌ {error_msg}")
                return {"status": "error", "error": error_msg}
                
        except Exception as e:
            error_msg = f"Error starting training: {str(e)}"
            print(f"❌ {error_msg}")
            return {"status": "error", "error": error_msg}

    def monitor_training(self, training_id: str) -> Dict[str, Any]:
        """Monitor specific training job progress"""
        try:
            response = requests.get(f"{self.base_url}/smart/status/{training_id}", timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "status": "error",
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e)
            }

    def get_available_strategies(self) -> List[Dict[str, Any]]:
        """Get available training strategies"""
        try:
            response = requests.get(f"{self.base_url}/smart/strategies", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return []
        except Exception as e:
            print(f"❌ Error getting strategies: {e}")
            return []

    def print_status_report(self, include_db: bool = True):
        """Print comprehensive status report"""
        print("\n" + "="*80)
        print("🎯 WORKFLOW ML TRAINING - PROGRESS TRACKER")
        print("="*80)
        print(f"📅 Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Current: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Service health
        print("\n🔍 SERVICE HEALTH:")
        health = self.check_service_health()
        if health["status"] == "healthy":
            print("  ✅ ML Training Service: HEALTHY")
            service_info = health.get("service_info", {})
            print(f"     Version: {service_info.get('version', 'Unknown')}")
            print(f"     Status: {service_info.get('status', 'Unknown')}")
        else:
            print(f"  ❌ ML Training Service: {health['status'].upper()}")
            print(f"     Error: {health.get('error', 'Unknown')}")
        
        if include_db:
            # Database status
            print("\n💾 DATABASE STATUS:")
            db_status = self.get_database_status()
            
            for db_type, status in db_status.items():
                print(f"\n  📊 {db_type.upper()} DATABASE:")
                if status["connection"] == "healthy":
                    print("    ✅ Connection: HEALTHY")
                    
                    if db_type == "audio":
                        analysis = status["analysis_results"]
                        print(f"    🎵 Analysis Results: {analysis['total']} total")
                        print(f"       ✅ Completed: {analysis['completed']}")
                        print(f"       ❌ Failed: {analysis['failed']}")
                        print(f"       ⏳ Processing: {analysis['processing']}")
                        print(f"    💾 Cached Features: {status['cached_features']}")
                        
                        training = status["training_jobs_24h"]
                        print(f"    🏃 Training Jobs (24h): {training['total']} total")
                        print(f"       ✅ Completed: {training['completed']}")
                        print(f"       ❌ Failed: {training['failed']}")
                        print(f"       🔄 Running: {training['running']}")
                        
                    elif db_type == "content":
                        lyrics = status["lyrics_analysis"]
                        print(f"    📝 Lyrics Analysis: {lyrics['total']} total")
                        print(f"       ✅ Completed: {lyrics['completed']}")
                        print(f"       ❌ Failed: {lyrics['failed']}")
                        
                else:
                    print(f"    ❌ Connection: {status['connection'].upper()}")
                    print(f"       Error: {status.get('error', 'Unknown')}")

    def continuous_monitoring(self, training_id: Optional[str] = None, interval: int = 30):
        """Continuous monitoring mode"""
        print(f"🔄 Starting continuous monitoring (interval: {interval}s)")
        if training_id:
            print(f"🎯 Tracking specific training: {training_id}")
        
        try:
            while True:
                self.print_status_report(include_db=True)
                
                if training_id:
                    print(f"\n🎯 TRAINING JOB: {training_id}")
                    training_status = self.monitor_training(training_id)
                    
                    if training_status.get("status") == "completed":
                        print("✅ Training completed successfully!")
                        print(f"📊 Results: {json.dumps(training_status, indent=2)}")
                        break
                    elif training_status.get("status") == "failed":
                        print("❌ Training failed!")
                        print(f"Error: {training_status.get('error', 'Unknown')}")
                        break
                    else:
                        print(f"🔄 Status: {training_status.get('status', 'Unknown')}")
                        if 'progress' in training_status:
                            print(f"📈 Progress: {training_status['progress']}")
                
                print(f"\n⏳ Waiting {interval}s before next check...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")

    def export_report(self, filename: Optional[str] = None) -> str:
        """Export comprehensive report to JSON"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_progress_report_{timestamp}.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "service_health": self.check_service_health(),
            "database_status": self.get_database_status(),
            "available_strategies": self.get_available_strategies(),
            "tracking_history": self.tracking_history
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Report exported to: {filename}")
        return filename

def main():
    parser = argparse.ArgumentParser(description="Training Progress Tracker")
    parser.add_argument("--training-id", help="Monitor specific training job")
    parser.add_argument("--continuous", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    parser.add_argument("--export-report", action="store_true", help="Export progress report")
    parser.add_argument("--clean-databases", action="store_true", help="Clean all databases")
    parser.add_argument("--start-training", choices=["smart_ensemble", "audio_only", "multimodal"], 
                       help="Start new training with specified strategy")
    
    args = parser.parse_args()
    
    tracker = TrainingProgressTracker()
    
    # Clean databases if requested
    if args.clean_databases:
        cleanup_results = tracker.clean_all_databases()
        print("\n🧹 DATABASE CLEANUP RESULTS:")
        for db_type, result in cleanup_results.items():
            print(f"  {db_type}: {result['status']}")
            if result['status'] == 'error':
                print(f"    Error: {result['error']}")
        print("\n✅ Database cleanup completed")
    
    # Start training if requested
    if args.start_training:
        training_result = tracker.start_smart_training(args.start_training)
        if training_result.get("status") != "error":
            training_id = training_result.get("training_id")
            print(f"🎯 Training started with ID: {training_id}")
        else:
            print(f"❌ Failed to start training: {training_result.get('error')}")
            sys.exit(1)
    
    # Export report if requested
    if args.export_report:
        tracker.export_report()
    
    # Continuous monitoring
    if args.continuous:
        tracker.continuous_monitoring(args.training_id, args.interval)
    else:
        # Single status check
        tracker.print_status_report()

if __name__ == "__main__":
    main() 