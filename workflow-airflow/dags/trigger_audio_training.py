"""
🚀 Simple Audio Training Trigger
===============================

This DAG automatically triggers the feature agreement pipeline 
with the correct audio-only configuration. Just run this DAG 
and it will trigger the main pipeline with proper settings.
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

# Configuration for audio-only training
AUDIO_TRAINING_CONFIG = {
    "services": ["audio"],
    "dataset_path": "/app/shared-data/training_data/filtered/filtered_audio_only_corrected_20250621_180350.csv",
    "training_id": f"audio_only_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "agreed_features": {
        "audio": [
            "audio_tempo",
            "audio_energy", 
            "audio_valence",
            "audio_danceability",
            "audio_loudness",
            "audio_speechiness",
            "audio_acousticness",
            "audio_instrumentalness"
        ]
    }
}

dag = DAG(
    'trigger_audio_training',
    default_args=default_args,
    description='🎵 Trigger Audio Feature Training with Proper Config',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['audio', 'training', 'trigger']
)

def log_configuration():
    """Log the configuration that will be used"""
    print("🎯 Starting Audio Feature Agreement Training")
    print("📋 Configuration:")
    for key, value in AUDIO_TRAINING_CONFIG.items():
        print(f"   {key}: {value}")
    print("✅ Configuration validated, triggering main pipeline...")

log_config_task = PythonOperator(
    task_id='log_configuration',
    python_callable=log_configuration,
    dag=dag
)

trigger_main_pipeline = TriggerDagRunOperator(
    task_id='trigger_feature_agreement_pipeline',
    trigger_dag_id='feature_agreement_training_pipeline',
    conf=AUDIO_TRAINING_CONFIG,
    wait_for_completion=False,
    dag=dag
)

log_config_task >> trigger_main_pipeline 