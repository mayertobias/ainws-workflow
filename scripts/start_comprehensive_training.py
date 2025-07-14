#!/usr/bin/env python3
"""
Comprehensive Training Starter Script

This script starts fresh smart ensemble training with:
- Random Forest + XGBoost ensemble
- Both audio-only and multimodal models
- Comprehensive feature extraction
- Progress monitoring
- Detailed reporting

Usage:
    python scripts/start_comprehensive_training.py [options]
"""

import requests
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

class ComprehensiveTrainingStarter:
    def __init__(self, base_url: str = "http://localhost:8003"):
        self.base_url = base_url
        self.training_jobs = []
        self.start_time = datetime.now()

    def check_service_health(self) -> bool:
        """Check if ML training service is healthy"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def get_available_strategies(self) -> list:
        """Get available training strategies"""
        try:
            response = requests.get(f"{self.base_url}/smart/strategies", timeout=10)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception:
            return []

    def get_available_datasets(self) -> list:
        """Get available filtered datasets"""
        try:
            response = requests.get(f"{self.base_url}/datasets/filtered", timeout=10)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception:
            return []

    def create_filtered_dataset(self) -> Optional[str]:
        """Create a fresh filtered dataset"""
        print("📊 Creating fresh filtered dataset...")
        
        create_config = {
            "source_csv": "r4a_song_data_training.csv",
            "description": "Fresh comprehensive training dataset with audio+multimodal features"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/datasets/create-filtered",
                json=create_config,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                creation_id = result.get("creation_id")
                print(f"✅ Dataset creation started: {creation_id}")
                
                # Monitor dataset creation
                while True:
                    status_response = requests.get(
                        f"{self.base_url}/datasets/filter-status/{creation_id}",
                        timeout=10
                    )
                    
                    if status_response.status_code == 200:
                        status = status_response.json()
                        if status.get("status") == "completed":
                            dataset_path = status.get("output_path")
                            print(f"✅ Dataset created successfully: {dataset_path}")
                            return dataset_path
                        elif status.get("status") == "failed":
                            print(f"❌ Dataset creation failed: {status.get('error')}")
                            return None
                        else:
                            print(f"⏳ Dataset creation in progress: {status.get('status')}")
                            time.sleep(10)
                    else:
                        print("❌ Failed to check dataset creation status")
                        return None
                        
            else:
                print(f"❌ Failed to create dataset: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error creating dataset: {e}")
            return None

    def start_smart_ensemble_training(self, dataset_path: str) -> Dict[str, Any]:
        """Start comprehensive smart ensemble training"""
        print("🚀 Starting comprehensive smart ensemble training...")
        
        # Enhanced configuration for comprehensive training
        training_config = {
            "data_path": dataset_path,
            "strategy": "smart_ensemble",
            "config": {
                "enable_parallel_training": True,
                "use_load_balancing": True,
                "max_epochs": 150,
                "batch_size": 64,
                "learning_rate": 0.0005,
                "ensemble_models": ["random_forest", "xgboost"],
                "model_configs": {
                    "random_forest": {
                        "n_estimators": 300,
                        "max_depth": 15,
                        "min_samples_split": 5,
                        "min_samples_leaf": 2,
                        "max_features": "sqrt",
                        "bootstrap": True,
                        "n_jobs": -1
                    },
                    "xgboost": {
                        "n_estimators": 200,
                        "max_depth": 10,
                        "learning_rate": 0.1,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "reg_alpha": 0.1,
                        "reg_lambda": 0.1,
                        "n_jobs": -1
                    }
                },
                "cross_validation_folds": 10,
                "feature_selection": True,
                "hyperparameter_tuning": True,
                "comprehensive_analysis": True,
                "generate_reports": True,
                "model_types": ["audio_only", "multimodal"],
                "feature_engineering": {
                    "audio_features": {
                        "spectral_features": True,
                        "rhythmic_features": True,
                        "harmonic_features": True,
                        "timbral_features": True,
                        "temporal_features": True
                    },
                    "lyrics_features": {
                        "sentiment_analysis": True,
                        "emotion_detection": True,
                        "topic_modeling": True,
                        "linguistic_features": True,
                        "semantic_features": True
                    }
                }
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/smart/train",
                json=training_config,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                training_id = result.get("training_id")
                print(f"✅ Comprehensive training started: {training_id}")
                
                # Store training job info
                self.training_jobs.append({
                    "training_id": training_id,
                    "start_time": datetime.now(),
                    "strategy": "smart_ensemble",
                    "dataset_path": dataset_path,
                    "status": "started"
                })
                
                return result
            else:
                error_msg = f"Failed to start training: HTTP {response.status_code}"
                print(f"❌ {error_msg}")
                return {"status": "error", "error": error_msg}
                
        except Exception as e:
            error_msg = f"Error starting training: {str(e)}"
            print(f"❌ {error_msg}")
            return {"status": "error", "error": error_msg}

    def monitor_training_progress(self, training_id: str, continuous: bool = True):
        """Monitor training progress with detailed reporting"""
        print(f"📈 Monitoring training progress: {training_id}")
        
        try:
            last_status = None
            
            while True:
                try:
                    response = requests.get(
                        f"{self.base_url}/smart/status/{training_id}",
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        status = response.json()
                        current_status = status.get("status")
                        
                        # Only print if status changed
                        if current_status != last_status:
                            print(f"\n🔄 Training Status: {current_status}")
                            print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                            
                            if "progress" in status:
                                print(f"📊 Progress: {status['progress']}")
                            
                            if "models_trained" in status:
                                print(f"🤖 Models Trained: {status['models_trained']}")
                            
                            if "current_stage" in status:
                                print(f"🎯 Current Stage: {status['current_stage']}")
                            
                            if "performance_metrics" in status:
                                metrics = status["performance_metrics"]
                                print(f"📈 Performance Metrics:")
                                for model, perf in metrics.items():
                                    print(f"   {model}: R² = {perf.get('r2', 'N/A'):.4f}")
                            
                            last_status = current_status
                        
                        # Check if training completed
                        if current_status == "completed":
                            print("\n🎉 Training completed successfully!")
                            print("📊 Final Results:")
                            print(json.dumps(status, indent=2))
                            break
                        elif current_status == "failed":
                            print(f"\n❌ Training failed: {status.get('error', 'Unknown error')}")
                            break
                        elif not continuous:
                            print(f"📊 Current status: {current_status}")
                            break
                        
                        time.sleep(30)  # Check every 30 seconds
                        
                    else:
                        print(f"❌ Failed to get training status: HTTP {response.status_code}")
                        if not continuous:
                            break
                        time.sleep(30)
                        
                except Exception as e:
                    print(f"⚠️ Error checking training status: {e}")
                    if not continuous:
                        break
                    time.sleep(30)
                    
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")

    def run_comprehensive_training(self, use_existing_dataset: bool = False):
        """Run the complete comprehensive training pipeline"""
        print("🎯 Starting Comprehensive Training Pipeline")
        print("="*60)
        print(f"📅 Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Check service health
        print("\n🔍 Checking service health...")
        if not self.check_service_health():
            print("❌ ML Training service is not healthy. Please check the service.")
            return False
        print("✅ ML Training service is healthy")
        
        # Step 2: Get available strategies
        print("\n📋 Getting available training strategies...")
        strategies = self.get_available_strategies()
        if strategies:
            print("✅ Available strategies:")
            for strategy in strategies:
                print(f"   - {strategy.get('name', 'Unknown')}: {strategy.get('description', 'No description')}")
        
        # Step 3: Prepare dataset
        dataset_path = None
        if use_existing_dataset:
            print("\n📊 Checking for existing filtered datasets...")
            datasets = self.get_available_datasets()
            if datasets:
                # Use the most recent dataset
                dataset_path = datasets[0].get("path")
                print(f"✅ Using existing dataset: {dataset_path}")
            else:
                print("⚠️ No existing datasets found, creating new one...")
                dataset_path = self.create_filtered_dataset()
        else:
            dataset_path = self.create_filtered_dataset()
        
        if not dataset_path:
            print("❌ Failed to prepare dataset")
            return False
        
        # Step 4: Start training
        print(f"\n🚀 Starting comprehensive training with dataset: {dataset_path}")
        training_result = self.start_smart_ensemble_training(dataset_path)
        
        if training_result.get("status") == "error":
            print(f"❌ Failed to start training: {training_result.get('error')}")
            return False
        
        training_id = training_result.get("training_id")
        if not training_id:
            print("❌ No training ID received")
            return False
        
        # Step 5: Monitor training
        print(f"\n📈 Starting continuous monitoring for training: {training_id}")
        self.monitor_training_progress(training_id, continuous=True)
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Training Starter")
    parser.add_argument("--use-existing-dataset", action="store_true", 
                       help="Use existing filtered dataset if available")
    parser.add_argument("--monitor-only", help="Only monitor existing training job")
    parser.add_argument("--base-url", default="http://localhost:8003",
                       help="Base URL for ML training service")
    
    args = parser.parse_args()
    
    starter = ComprehensiveTrainingStarter(base_url=args.base_url)
    
    if args.monitor_only:
        print(f"📈 Monitoring existing training: {args.monitor_only}")
        starter.monitor_training_progress(args.monitor_only, continuous=True)
    else:
        success = starter.run_comprehensive_training(use_existing_dataset=args.use_existing_dataset)
        if success:
            print("\n🎉 Comprehensive training pipeline completed successfully!")
        else:
            print("\n❌ Comprehensive training pipeline failed!")

if __name__ == "__main__":
    main() 