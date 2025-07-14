#!/usr/bin/env python3
"""
Simplified script to trigger ML Training using Airflow.
This version just passes parameters to the ML training service without 
Airflow doing any dataset processing.
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

def trigger_simplified_ml_training():
    """
    Trigger simplified ML training - just pass parameters to the ML training service.
    """
    dag_id = "ml_training_smart_ensemble_simplified"
    url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns"
    
    # Generate unique training ID
    training_id = f"ensemble_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Simple configuration - just pass everything to the ML training service
    training_config = {
        # Required parameters
        "dataset_path": "/opt/airflow/data/training_data/filtered/filtered_audio_only_corrected_20250621_180350.csv",
        "training_id": training_id,
        
        # Training strategy and configuration
        "training_strategy": "smart_ensemble",
        "model_types": [
            "audio_only_ensemble",
            "multimodal_ensemble"
        ],
        
        # Audio-only ensemble configuration
        "audio_only_config": {
            "models": ["random_forest", "gradient_boosting", "neural_network", "svm", "logistic_regression"],
            "feature_types": ["audio_features", "spectral_features", "advanced_features"],
            "cross_validation_folds": 5,
            "hyperparameter_tuning": True
        },
        
        # Multimodal ensemble configuration  
        "multimodal_config": {
            "models": ["random_forest", "gradient_boosting", "neural_network", "svm", "logistic_regression"],
            "feature_types": ["audio_features", "content_features", "combined_features"],
            "cross_validation_folds": 5,
            "hyperparameter_tuning": True,
            "feature_fusion_strategy": "late_fusion"
        },
        
        # Processing parameters
        "batch_size": 50,
        "max_concurrent_requests": 25,
        "timeout_per_song": 300,
        "validation_split": 0.2,
        "test_split": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        
        # Ensemble parameters
        "ensemble_method": "voting",
        "voting_strategy": "soft",
        "meta_learner": "logistic_regression",
        
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
    
    print("🚀 Triggering Simplified ML Training")
    print("=" * 60)
    print(f"📊 Dataset: {training_config['dataset_path'].split('/')[-1]}")
    print(f"🆔 Training ID: {training_id}")
    print(f"🎯 Strategy: {training_config['training_strategy']}")
    print(f"📦 Batch Size: {training_config['batch_size']}")
    print(f"🔧 Ensemble Method: {training_config['ensemble_method']}")
    print("=" * 60)
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ ML Training triggered successfully!")
            print(f"   DAG Run ID: {result['dag_run_id']}")
            print(f"   State: {result['state']}")
            print(f"   Execution Date: {result['execution_date']}")
            
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
    dag_id = "ml_training_smart_ensemble_simplified"
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
            
            return result
        else:
            print(f"❌ Failed to get training status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting training status: {e}")
        return None

if __name__ == "__main__":
    print("🎯 Simplified ML Training Trigger")
    print("Just passes parameters to workflow-ml-training service")
    print("No dataset processing in Airflow - much simpler!")
    print("")
    
    # Trigger new training
    result = trigger_simplified_ml_training()
    
    if result:
        dag_run_id = result['dag_run_id']
        print(f"\n⏳ Waiting a moment for training to start...")
        import time
        time.sleep(10)
        
        # Check initial status
        get_training_status(dag_run_id)
        
        print(f"\n📝 Benefits of Simplified Approach:")
        print(f"   ✅ No dataset processing in Airflow")
        print(f"   ✅ ML training service handles everything")
        print(f"   ✅ Simpler DAG with fewer failure points")
        print(f"   ✅ Better separation of concerns")
        print(f"   ✅ Faster execution (no unnecessary steps)")
    else:
        print(f"\n❌ Failed to trigger training. Please check the logs and try again.") 