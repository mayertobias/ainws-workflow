"""
ML Training DAG for HSS Workflow system.
This DAG handles smart ensemble training and replaces the ML training orchestration 
from the current orchestrator system.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator

# Import custom operators
import sys
sys.path.append('/opt/airflow/plugins')
from workflow_operators import (
    DatasetProcessingOperator,
    MLTrainingOperator,
    WorkflowResultsAggregatorOperator
)

# Default arguments for the DAG
default_args = {
    'owner': 'hss-ml-training',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,  # Training tasks typically shouldn't retry automatically
    'retry_delay': timedelta(minutes=10),
    'max_active_runs': 1,  # Only one training at a time
}

# Create the DAG
dag = DAG(
    'ml_training_smart_ensemble',
    default_args=default_args,
    description='Smart ensemble ML training workflow for hit prediction models',
    schedule_interval=None,  # Triggered manually or via API
    catchup=False,
    max_active_runs=1,  # Ensure only one training runs at a time
    tags=['ml-training', 'smart-ensemble', 'hit-prediction'],
)

def validate_training_config(**context):
    """Validate training configuration parameters."""
    params = context.get('params', {})
    
    # Check required parameters
    required_params = ['dataset_path', 'training_id']
    missing_params = [p for p in required_params if p not in params]
    
    if missing_params:
        raise ValueError(f"Missing required parameters: {missing_params}")
    
    # Validate dataset path
    dataset_path = params['dataset_path']
    if not dataset_path.endswith('.csv'):
        raise ValueError("Dataset path must be a CSV file")
    
    print(f"Training configuration validated successfully")
    print(f"Dataset: {dataset_path}")
    print(f"Training ID: {params['training_id']}")
    
    return params

# Start task
start_task = DummyOperator(
    task_id='start_ml_training',
    dag=dag,
)

# Validate training configuration
validate_config = PythonOperator(
    task_id='validate_training_config',
    python_callable=validate_training_config,
    dag=dag,
)

# Process training dataset
process_dataset = DatasetProcessingOperator(
    task_id='process_training_dataset',
    dataset_path='{{ params.dataset_path }}',
    output_key='training_songs',
    batch_size='{{ params.get("batch_size", None) }}',
    dag=dag,
)

# Smart Ensemble Training Task
smart_ensemble_training = MLTrainingOperator(
    task_id='smart_ensemble_training',
    training_strategy='smart_ensemble',
    dataset_path='{{ params.dataset_path }}',
    training_id='{{ params.training_id }}',
    model_config={
        'ensemble_models': ['random_forest', 'gradient_boosting', 'svm', 'neural_network', 'voting_classifier'],
        'cross_validation_folds': 5,
        'feature_types': ['audio', 'lyrics'],
        'batch_size': '{{ params.get("batch_size", 100) }}',
        'max_concurrent_requests': '{{ params.get("max_concurrent_requests", 50) }}',
        'timeout_per_song': '{{ params.get("timeout_per_song", 300) }}',
    },
    timeout=7200,  # 2 hours timeout for training
    retries=0,  # Don't retry training automatically
    dag=dag,
)

# Training Results Aggregation
aggregate_training_results = WorkflowResultsAggregatorOperator(
    task_id='aggregate_training_results',
    result_keys=['training_id', 'model_paths', 'training_songs'],
    output_path='/opt/airflow/shared-data/training_results/{{ params.training_id }}_results.json',
    dag=dag,
)

# End task
end_task = DummyOperator(
    task_id='end_ml_training',
    dag=dag,
)

# Define task dependencies
start_task >> validate_config >> process_dataset >> smart_ensemble_training >> aggregate_training_results >> end_task 