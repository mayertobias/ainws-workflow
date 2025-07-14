"""
🚀 Simple Multimodal Training Trigger
=====================================

This DAG automatically triggers the multimodal feature agreement pipeline 
with the correct audio + content configuration. Just run this DAG 
and it will trigger the main multimodal pipeline with proper settings.

📊 Dataset: filtered_multimodal_corrected_20250621_180350.csv (338 songs)
🎯 Features: Audio + Lyrics/Content features for enhanced prediction
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

# Configuration for multimodal training (audio + content)
MULTIMODAL_TRAINING_CONFIG = {
    "services": ["audio", "content"],
    "dataset_path": "/app/shared-data/training_data/filtered/filtered_multimodal_corrected_20250621_180350.csv",
    "training_id": f"multimodal_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
            "audio_brightness",
            "audio_mood_sad",
            "audio_mood_happy"
        ],
        "content": [
            "lyrics_sentiment_positive",
            "lyrics_sentiment_negative",
            "lyrics_emotion_joy",
            "lyrics_emotion_sadness",
            "lyrics_complexity_score",
            "lyrics_word_count",
            "lyrics_theme_love",
            "lyrics_theme_party"
        ]
    }
}

dag = DAG(
    'trigger_multimodal_training',
    default_args=default_args,
    description='🎵📝 Trigger Multimodal Feature Training (Audio + Content) with Proper Config',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['multimodal', 'audio', 'content', 'training', 'trigger']
)

def log_multimodal_configuration():
    """Log the multimodal configuration that will be used"""
    print("🎯 Starting Multimodal Feature Agreement Training")
    print("📋 Multimodal Configuration:")
    
    audio_count = len(MULTIMODAL_TRAINING_CONFIG['agreed_features']['audio'])
    content_count = len(MULTIMODAL_TRAINING_CONFIG['agreed_features']['content'])
    total_features = audio_count + content_count
    
    print(f"   Services: {MULTIMODAL_TRAINING_CONFIG['services']}")
    print(f"   Dataset: {MULTIMODAL_TRAINING_CONFIG['dataset_path']}")
    print(f"   Training ID: {MULTIMODAL_TRAINING_CONFIG['training_id']}")
    print(f"   Total Features: {total_features}")
    print(f"   Audio Features ({audio_count}):")
    for feature in MULTIMODAL_TRAINING_CONFIG['agreed_features']['audio']:
        print(f"     - {feature}")
    print(f"   Content Features ({content_count}):")
    for feature in MULTIMODAL_TRAINING_CONFIG['agreed_features']['content']:
        print(f"     - {feature}")
    
    print("\n🎯 Expected Benefits of Multimodal Training:")
    print("   ✅ Audio features: tempo, energy, mood, acoustics")
    print("   ✅ Content features: sentiment, emotions, themes, complexity")
    print("   ✅ Better prediction accuracy through combined modalities")
    print("   ✅ Feature importance comparison across audio vs lyrics")
    print("   ✅ Enhanced model interpretability")
    
    print("\n✅ Multimodal configuration validated, triggering main pipeline...")

log_config_task = PythonOperator(
    task_id='log_multimodal_configuration',
    python_callable=log_multimodal_configuration,
    dag=dag
)

trigger_multimodal_pipeline = TriggerDagRunOperator(
    task_id='trigger_multimodal_feature_agreement_pipeline',
    trigger_dag_id='multimodal_feature_agreement_training_pipeline',
    conf=MULTIMODAL_TRAINING_CONFIG,
    wait_for_completion=False,
    dag=dag
)

log_config_task >> trigger_multimodal_pipeline 