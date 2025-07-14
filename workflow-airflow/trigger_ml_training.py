#!/usr/bin/env python3
"""
Script to trigger ML Training using Airflow with the filtered dataset.
Uses ensemble technique for both audio-only and multimodal models.
"""

import requests
import json
import base64
from datetime import datetime

# Airflow configuration
AIRFLOW_URL = "http://localhost:8080"
USERNAME = "admin"
PASSWORD = "admin"

# Create basic auth header
credentials = base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()
headers = {
    "Authorization": f"Basic {credentials}",
    "Content-Type": "application/json"
}

def trigger_ml_training_with_dataset():
    """
    Trigger ML training with the filtered dataset using ensemble techniques
    for both audio-only and multimodal models.
    """
    dag_id = "ml_training_smart_ensemble"
    url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns"
    
    # Generate unique training ID
    training_id = f"ensemble_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Training configuration for the filtered dataset
    training_config = {
        # Required parameters
        "dataset_path": "/opt/airflow/data/training_data/filtered/filtered_audio_only_corrected_20250621_180350.csv",
        "training_id": training_id,
        
        # Dataset configuration
        "dataset_name": "filtered_audio_only_corrected_20250621_180350",
        "batch_size": 50,
        "max_concurrent_requests": 25,
        "timeout_per_song": 300,
        
        # Model configuration
        "training_strategy": "smart_ensemble",
        "model_types": [
            "audio_only_ensemble",    # Audio-only models ensemble
            "multimodal_ensemble"     # Multimodal (audio + lyrics) ensemble
        ],
        
        # Audio-only ensemble configuration
        "audio_only_config": {
            "models": [
                "random_forest",
                "gradient_boosting", 
                "neural_network",
                "svm",
                "logistic_regression"
            ],
            "feature_types": ["audio_features", "spectral_features", "advanced_features"],
            "cross_validation_folds": 5,
            "hyperparameter_tuning": True
        },
        
        # Multimodal ensemble configuration  
        "multimodal_config": {
            "models": [
                "random_forest",
                "gradient_boosting",
                "neural_network", 
                "svm",
                "logistic_regression"
            ],
            "feature_types": ["audio_features", "content_features", "combined_features"],
            "cross_validation_folds": 5,
            "hyperparameter_tuning": True,
            "feature_fusion_strategy": "late_fusion"
        },
        
        # Training parameters
        "validation_split": 0.2,
        "test_split": 0.1,
        "random_state": 42,
        "n_jobs": -1,  # Use all available cores
        
        # Ensemble parameters
        "ensemble_method": "voting",  # voting or stacking
        "voting_strategy": "soft",    # soft or hard voting
        "meta_learner": "logistic_regression",  # for stacking
        
        # Output configuration
        "model_name": f"ensemble_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "save_individual_models": True,
        "save_ensemble_model": True,
        "generate_evaluation_report": True,
        
        # Performance optimization
        "enable_early_stopping": True,
        "patience": 10,
        "min_delta": 0.001,
        
        # Logging and monitoring
        "log_level": "INFO",
        "save_training_logs": True,
        "track_metrics": True
    }
    
    payload = {
        "conf": training_config,
        "dag_run_id": training_id
    }
    
    print("🚀 Triggering ML Training with Ensemble Technique")
    print("=" * 60)
    print(f"📊 Dataset: {training_config['dataset_name']}")
    print(f"🆔 Training ID: {training_id}")
    print(f"🎵 Audio-only Models: {', '.join(training_config['audio_only_config']['models'])}")
    print(f"🎭 Multimodal Models: {', '.join(training_config['multimodal_config']['models'])}")
    print(f"🔧 Ensemble Method: {training_config['ensemble_method']}")
    print(f"📈 Validation Split: {training_config['validation_split']}")
    print(f"⚙️  Batch Size: {training_config['batch_size']}")
    print("=" * 60)
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ ML Training triggered successfully!")
            print(f"   DAG Run ID: {result['dag_run_id']}")
            print(f"   State: {result['state']}")
            print(f"   Execution Date: {result['execution_date']}")
            
            # Monitor the training progress
            print(f"\n📊 Monitor training progress:")
            print(f"   • Airflow UI: {AIRFLOW_URL}/dags/{dag_id}/grid")
            print(f"   • Flower Monitor: http://localhost:5555")
            
            return result
        else:
            print(f"❌ Failed to trigger ML training: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error triggering ML training: {e}")
        return None

def get_training_status(dag_run_id):
    """Get the status of the training DAG run."""
    dag_id = "ml_training_smart_ensemble"
    url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}"
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            result = response.json()
            print(f"\n📊 Training Status:")
            print(f"   DAG ID: {result['dag_id']}")
            print(f"   Run ID: {result['dag_run_id']}")
            print(f"   State: {result['state']}")
            print(f"   Start Date: {result['start_date']}")
            print(f"   End Date: {result.get('end_date', 'Still running...')}")
            
            if result['state'] == 'success':
                print(f"   🎉 Training completed successfully!")
            elif result['state'] == 'failed':
                print(f"   ❌ Training failed")
            elif result['state'] == 'running':
                print(f"   🔄 Training in progress...")
            
            return result
        else:
            print(f"❌ Failed to get training status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting training status: {e}")
        return None

def list_recent_training_runs(limit=5):
    """List recent training runs."""
    dag_id = "ml_training_smart_ensemble"
    url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns?limit={limit}&order_by=-start_date"
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            result = response.json()
            print(f"\n📋 Recent Training Runs:")
            for run in result['dag_runs']:
                status_emoji = {
                    'success': '✅',
                    'failed': '❌', 
                    'running': '🔄',
                    'queued': '⏳'
                }.get(run['state'], '❓')
                print(f"   {status_emoji} {run['dag_run_id']} - {run['state']} ({run['start_date']})")
            return result
        else:
            print(f"❌ Failed to list training runs: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error listing training runs: {e}")
        return None

if __name__ == "__main__":
    print("🎯 ML Training with Ensemble Technique - Filtered Dataset")
    print("Dataset: filtered_audio_only_corrected_20250621_180350.csv")
    print("Models: Audio-only + Multimodal Ensembles")
    print("")
    
    # Show recent training runs first
    list_recent_training_runs()
    
    # Trigger new training
    result = trigger_ml_training_with_dataset()
    
    if result:
        dag_run_id = result['dag_run_id']
        print(f"\n⏳ Waiting a moment for training to start...")
        import time
        time.sleep(10)
        
        # Check initial status
        get_training_status(dag_run_id)
        
        print(f"\n📝 Next Steps:")
        print(f"   1. Monitor progress in Airflow UI: {AIRFLOW_URL}")
        print(f"   2. Check Celery workers: http://localhost:5555")
        print(f"   3. Training logs will be available in the task logs")
        print(f"   4. Models will be saved with timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        print(f"\n💡 Training Configuration Summary:")
        print(f"   • Dataset: 402 songs with audio + lyrics")
        print(f"   • Audio-only ensemble: 5 models (RF, GB, NN, SVM, LR)")
        print(f"   • Multimodal ensemble: 5 models with feature fusion")
        print(f"   • Cross-validation: 5-fold")
        print(f"   • Hyperparameter tuning: Enabled")
        print(f"   • Validation split: 20%")
        print(f"   • Test split: 10%")
    else:
        print(f"\n❌ Failed to trigger training. Please check the logs and try again.")