"""
🚀 Audio Ensemble Training Trigger (Random Forest + XGBoost + SHAP)
==================================================================

This DAG automatically triggers ensemble training with:
- Audio-only features
- Random Forest + XGBoost ensemble
- SHAP explainability analysis
- MLflow experiment tracking

🎯 Features: Audio tempo, energy, valence, danceability, etc.
🧠 Models: Ensemble (Random Forest + XGBoost)
🔍 Explainability: SHAP analysis with feature importance
📊 Tracking: MLflow experiments and model registry
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': datetime(2025, 6, 28),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Configuration for audio-only ensemble training
# FIXED: Flexible dataset path that will be resolved by the ML service
AUDIO_ENSEMBLE_CONFIG = {
    "services": ["audio"],
    "dataset_path": "filtered_audio_only_corrected_20250621_180350.csv",  # Relative path - will be resolved by ML service
    "training_id": f"audio_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "strategy": "audio_only",
    "model_types": ["ensemble"],  # This will use Random Forest + XGBoost
    "enable_shap": True,
    "agreed_features": {
        "audio": [
            "audio_tempo",
            "audio_energy", 
            "audio_valence",
            "audio_danceability",
            "audio_loudness",
            "audio_speechiness",
            "audio_acousticness",
            "audio_instrumentalness",
            "audio_liveness",
            "audio_key",
            "audio_mode",
            "audio_brightness",
            "audio_complexity",
            "audio_warmth",
            "audio_harmonic_strength",
            "audio_mood_happy",
            "audio_mood_sad",
            "audio_mood_aggressive",
            "audio_mood_relaxed",
            "audio_mood_party",
            "audio_mood_electronic",
            "audio_primary_genre",
            "audio_top_genre_1_prob",
            "audio_top_genre_2_prob"
        ]
    },
    "ensemble_config": {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42
        },
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": "binary:logistic",
            "random_state": 42
        },
        "voting": "soft"  # Use probability voting for better performance
    },
    "shap_config": {
        "sample_size": 100,  # Number of samples for SHAP analysis
        "generate_plots": True,
        "save_artifacts": True
    }
}

dag = DAG(
    'trigger_audio_ensemble_training',
    default_args=default_args,
    description='🎵 Trigger Audio Ensemble Training (RF + XGBoost + SHAP)',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['audio', 'ensemble', 'training', 'shap', 'random-forest', 'xgboost']
)

def log_ensemble_configuration():
    """Log the ensemble configuration that will be used"""
    print("🎯 Starting Audio Ensemble Training (Random Forest + XGBoost + SHAP)")
    print("📋 Ensemble Configuration:")
    
    audio_features = AUDIO_ENSEMBLE_CONFIG['agreed_features']['audio']
    print(f"   🎵 Audio Features: {len(audio_features)} features")
    print(f"   🧠 Model Type: Ensemble (Random Forest + XGBoost)")
    print(f"   🔍 SHAP Analysis: {'Enabled' if AUDIO_ENSEMBLE_CONFIG['enable_shap'] else 'Disabled'}")
    print(f"   📊 Dataset: {AUDIO_ENSEMBLE_CONFIG['dataset_path']} (will be resolved by ML service)")
    print(f"   🆔 Training ID: {AUDIO_ENSEMBLE_CONFIG['training_id']}")
    
    print("\n🎵 Audio Features:")
    for i, feature in enumerate(audio_features, 1):
        print(f"   {i:2d}. {feature}")
    
    print("\n🧠 Ensemble Configuration:")
    rf_config = AUDIO_ENSEMBLE_CONFIG['ensemble_config']['random_forest']
    xgb_config = AUDIO_ENSEMBLE_CONFIG['ensemble_config']['xgboost']
    print(f"   Random Forest: {rf_config['n_estimators']} trees, max_depth={rf_config['max_depth']}")
    print(f"   XGBoost: {xgb_config['n_estimators']} rounds, max_depth={xgb_config['max_depth']}, lr={xgb_config['learning_rate']}")
    print(f"   Voting: {AUDIO_ENSEMBLE_CONFIG['ensemble_config']['voting']}")
    
    print("\n🔍 SHAP Configuration:")
    shap_config = AUDIO_ENSEMBLE_CONFIG['shap_config']
    print(f"   Sample Size: {shap_config['sample_size']} samples")
    print(f"   Generate Plots: {shap_config['generate_plots']}")
    print(f"   Save Artifacts: {shap_config['save_artifacts']}")
    
    print("\n📊 Expected Outputs:")
    print("   ✅ Ensemble model (Random Forest + XGBoost)")
    print("   ✅ SHAP feature importance analysis")
    print("   ✅ SHAP summary plots and bar plots")
    print("   ✅ Feature importance comparison (RF vs XGBoost vs SHAP)")
    print("   ✅ MLflow experiment tracking")
    print("   ✅ Model registry with metadata")
    
    print("\n🔗 Access Points:")
    print("   📊 MLflow UI: http://localhost:5001")
    print("   🎯 Training Service: http://localhost:8005")
    print("   📋 Model Registry: /app/models/ml-training/model_registry.json")
    
    print("\n⚠️ Note: Dataset path will be automatically resolved by ML service.")
    print("   The service will search for the dataset in appropriate directories.")
    
    print("\n✅ Ensemble configuration validated, triggering training pipeline...")

log_config_task = PythonOperator(
    task_id='log_ensemble_configuration',
    python_callable=log_ensemble_configuration,
    dag=dag
)

trigger_ensemble_pipeline = TriggerDagRunOperator(
    task_id='trigger_audio_ensemble_pipeline',
    trigger_dag_id='audio_ensemble_training_pipeline',
    conf=AUDIO_ENSEMBLE_CONFIG,
    wait_for_completion=False,
    dag=dag
)

log_config_task >> trigger_ensemble_pipeline 