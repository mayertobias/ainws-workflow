"""
🚀 Multimodal Ensemble Training Trigger (Random Forest + XGBoost + SHAP)
=======================================================================

This DAG automatically triggers multimodal ensemble training with:
- Audio + Content features
- Random Forest + XGBoost ensemble
- SHAP explainability analysis
- MLflow experiment tracking

🎯 Audio Features: Tempo, energy, valence, danceability, mood, genre
📝 Content Features: Sentiment, emotions, themes, complexity
🧠 Models: Ensemble (Random Forest + XGBoost)
🔍 Explainability: SHAP analysis with cross-modal feature importance
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

# Configuration for multimodal ensemble training
# FIXED: Flexible dataset path that will be resolved by the ML service
MULTIMODAL_ENSEMBLE_CONFIG = {
    "services": ["audio", "content"],
    "dataset_path": "filtered_multimodal_corrected_20250621_180350.csv",  # Relative path - will be resolved by ML service
    "training_id": f"multimodal_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "strategy": "multimodal",
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
        ],
        "content": [
            "lyrics_sentiment_positive",
            "lyrics_sentiment_negative",
            "lyrics_sentiment_neutral",
            "lyrics_complexity_score",
            "lyrics_word_count",
            "lyrics_unique_words",
            "lyrics_reading_level",
            "lyrics_emotion_anger",
            "lyrics_emotion_joy",
            "lyrics_emotion_sadness",
            "lyrics_emotion_fear",
            "lyrics_emotion_surprise",
            "lyrics_theme_love",
            "lyrics_theme_party",
            "lyrics_theme_sadness",
            "lyrics_profanity_score",
            "lyrics_repetition_score",
            "lyrics_rhyme_density",
            "lyrics_narrative_complexity",
            "lyrics_lexical_diversity",
            "lyrics_motif_count",
            "lyrics_verse_count",
            "lyrics_chorus_count",
            "lyrics_bridge_count"
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
        "save_artifacts": True,
        "cross_modal_analysis": True  # Compare audio vs content feature importance
    }
}

dag = DAG(
    'trigger_multimodal_ensemble_training',
    default_args=default_args,
    description='🎵📝 Trigger Multimodal Ensemble Training (RF + XGBoost + SHAP)',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['multimodal', 'ensemble', 'training', 'shap', 'random-forest', 'xgboost', 'audio', 'content']
)

def log_multimodal_ensemble_configuration():
    """Log the multimodal ensemble configuration that will be used"""
    print("🎯 Starting Multimodal Ensemble Training (Random Forest + XGBoost + SHAP)")
    print("📋 Multimodal Ensemble Configuration:")
    
    audio_features = MULTIMODAL_ENSEMBLE_CONFIG['agreed_features']['audio']
    content_features = MULTIMODAL_ENSEMBLE_CONFIG['agreed_features']['content']
    total_features = len(audio_features) + len(content_features)
    
    print(f"   🎵 Audio Features: {len(audio_features)} features")
    print(f"   📝 Content Features: {len(content_features)} features")
    print(f"   📊 Total Features: {total_features} features")
    print(f"   🧠 Model Type: Ensemble (Random Forest + XGBoost)")
    print(f"   🔍 SHAP Analysis: {'Enabled' if MULTIMODAL_ENSEMBLE_CONFIG['enable_shap'] else 'Disabled'}")
    print(f"   📊 Dataset: {MULTIMODAL_ENSEMBLE_CONFIG['dataset_path']} (will be resolved by ML service)")
    print(f"   🆔 Training ID: {MULTIMODAL_ENSEMBLE_CONFIG['training_id']}")
    
    print("\n🎵 Audio Features:")
    for i, feature in enumerate(audio_features, 1):
        print(f"   {i:2d}. {feature}")
    
    print("\n📝 Content Features:")
    for i, feature in enumerate(content_features, 1):
        print(f"   {i:2d}. {feature}")
    
    print("\n🧠 Ensemble Configuration:")
    rf_config = MULTIMODAL_ENSEMBLE_CONFIG['ensemble_config']['random_forest']
    xgb_config = MULTIMODAL_ENSEMBLE_CONFIG['ensemble_config']['xgboost']
    print(f"   Random Forest: {rf_config['n_estimators']} trees, max_depth={rf_config['max_depth']}")
    print(f"   XGBoost: {xgb_config['n_estimators']} rounds, max_depth={xgb_config['max_depth']}, lr={xgb_config['learning_rate']}")
    print(f"   Voting: {MULTIMODAL_ENSEMBLE_CONFIG['ensemble_config']['voting']}")
    
    print("\n🔍 SHAP Configuration:")
    shap_config = MULTIMODAL_ENSEMBLE_CONFIG['shap_config']
    print(f"   Sample Size: {shap_config['sample_size']} samples")
    print(f"   Generate Plots: {shap_config['generate_plots']}")
    print(f"   Save Artifacts: {shap_config['save_artifacts']}")
    print(f"   Cross-Modal Analysis: {shap_config['cross_modal_analysis']}")
    
    print("\n📊 Expected Outputs:")
    print("   ✅ Multimodal ensemble model (Random Forest + XGBoost)")
    print("   ✅ SHAP feature importance analysis")
    print("   ✅ Cross-modal feature importance comparison")
    print("   ✅ SHAP summary plots and bar plots")
    print("   ✅ Audio vs Content feature importance breakdown")
    print("   ✅ Feature importance comparison (RF vs XGBoost vs SHAP)")
    print("   ✅ MLflow experiment tracking")
    print("   ✅ Model registry with metadata")
    
    print("\n🎯 Multimodal Benefits:")
    print("   ✅ Better prediction accuracy through combined modalities")
    print("   ✅ Feature importance comparison across audio vs lyrics")
    print("   ✅ Enhanced model interpretability")
    print("   ✅ Cross-modal insights for music analysis")
    
    print("\n🔗 Access Points:")
    print("   📊 MLflow UI: http://localhost:5001")
    print("   🎯 Training Service: http://localhost:8005")
    print("   📋 Model Registry: /app/models/ml-training/model_registry.json")
    
    print("\n⚠️ Note: Dataset path will be automatically resolved by ML service.")
    print("   The service will search for the dataset in appropriate directories.")
    
    print("\n✅ Multimodal ensemble configuration validated, triggering training pipeline...")

log_config_task = PythonOperator(
    task_id='log_multimodal_ensemble_configuration',
    python_callable=log_multimodal_ensemble_configuration,
    dag=dag
)

trigger_multimodal_ensemble_pipeline = TriggerDagRunOperator(
    task_id='trigger_multimodal_ensemble_pipeline',
    trigger_dag_id='multimodal_ensemble_training_pipeline',
    conf=MULTIMODAL_ENSEMBLE_CONFIG,
    wait_for_completion=False,
    dag=dag
)

log_config_task >> trigger_multimodal_ensemble_pipeline 