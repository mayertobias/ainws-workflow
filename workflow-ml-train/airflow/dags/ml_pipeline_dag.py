"""
Airflow DAG for ML Training Pipeline Visualization

Provides visual representation of the ML training pipeline stages
with real-time status updates and monitoring capabilities.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Default arguments for all DAGs
default_args = {
    'owner': 'workflow-ml-train',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
    'catchup': False
}

def create_ml_pipeline_dag(strategy: str = "audio_only"):
    """
    Dynamic DAG creation for different training strategies.
    
    Args:
        strategy: Training strategy (audio_only, multimodal, custom)
    """
    
    dag_id = f'ml_pipeline_{strategy}'
    
    dag = DAG(
        dag_id,
        default_args=default_args,
        description=f'ML Training Pipeline for {strategy} strategy',
        schedule_interval=None,  # Manual trigger only
        tags=['ml-pipeline', strategy, 'training'],
        max_active_runs=3,
        doc_md=f"""
        # ML Training Pipeline - {strategy.upper()}
        
        This DAG visualizes the ML training pipeline execution for the **{strategy}** strategy.
        
        ## Pipeline Stages
        
        1. **Service Discovery**: Auto-discover available features from services
        2. **Feature Agreement**: Select and validate feature vector
        3. **Feature Extraction**: Extract features from training data
        4. **Model Training**: Train ML models using extracted features
        5. **Model Registry**: Register trained models in MLflow
        
        ## Strategy Details
        
        **{strategy}**: {'Audio features only' if strategy == 'audio_only' else 'Audio + Content features' if strategy == 'multimodal' else 'User-defined features'}
        
        ## Usage
        
        This DAG is triggered programmatically by the workflow-ml-train service.
        Monitor progress via the Airflow UI or the service's real-time monitoring endpoints.
        """
    )
    
    # Start task
    start_task = DummyOperator(
        task_id='start_pipeline',
        dag=dag,
        doc_md=f"Start {strategy} training pipeline"
    )
    
    # Stage 1: Service Discovery
    service_discovery_task = PythonOperator(
        task_id='service_discovery',
        python_callable=run_service_discovery,
        op_kwargs={'strategy': strategy},
        dag=dag,
        doc_md="""
        **Service Discovery Stage**
        
        - Query /features endpoints from audio and content services
        - Build comprehensive feature catalog
        - Validate service availability
        - Estimate total available features
        """
    )
    
    # Stage 2: Feature Agreement
    feature_agreement_task = PythonOperator(
        task_id='feature_agreement',
        python_callable=run_feature_agreement,
        op_kwargs={'strategy': strategy},
        dag=dag,
        doc_md="""
        **Feature Agreement Stage**
        
        - Select features based on strategy
        - Create feature vector agreement
        - Validate feature selection
        - Test feature extraction on samples
        """
    )
    
    # Stage 3: Feature Extraction
    feature_extraction_task = PythonOperator(
        task_id='feature_extraction',
        python_callable=run_feature_extraction,
        op_kwargs={'strategy': strategy},
        dag=dag,
        doc_md="""
        **Feature Extraction Stage**
        
        - Extract features from all songs in dataset
        - Call audio/content services for each song
        - Aggregate features into training matrix
        - Validate feature completeness
        """
    )
    
    # Stage 4: Model Training
    model_training_task = PythonOperator(
        task_id='model_training',
        python_callable=run_model_training,
        op_kwargs={'strategy': strategy},
        dag=dag,
        doc_md="""
        **Model Training Stage**
        
        - Train multiple model types (Random Forest, XGBoost)
        - Perform cross-validation
        - Select best performing model
        - Calculate feature importance
        """
    )
    
    # Stage 5: Model Registry
    model_registry_task = PythonOperator(
        task_id='model_registry',
        python_callable=run_model_registry,
        op_kwargs={'strategy': strategy},
        dag=dag,
        doc_md="""
        **Model Registry Stage**
        
        - Register trained models in MLflow
        - Save model artifacts and metadata
        - Create model version and tags
        - Update model deployment status
        """
    )
    
    # End task
    end_task = DummyOperator(
        task_id='pipeline_complete',
        dag=dag,
        doc_md=f"Complete {strategy} training pipeline"
    )
    
    # Define task dependencies
    start_task >> service_discovery_task
    service_discovery_task >> feature_agreement_task
    feature_agreement_task >> feature_extraction_task
    feature_extraction_task >> model_training_task
    model_training_task >> model_registry_task
    model_registry_task >> end_task
    
    return dag

# Task implementations
def run_service_discovery(strategy: str, **context):
    """Execute service discovery stage"""
    logger.info(f"🔍 Running service discovery for strategy: {strategy}")
    
    # This would call the actual workflow-ml-train service
    # For now, just simulate the process
    
    import time
    time.sleep(5)  # Simulate processing
    
    logger.info(f"✅ Service discovery completed for {strategy}")
    return {"stage": "service_discovery", "status": "completed", "strategy": strategy}

def run_feature_agreement(strategy: str, **context):
    """Execute feature agreement stage"""
    logger.info(f"📝 Running feature agreement for strategy: {strategy}")
    
    import time
    time.sleep(3)  # Simulate processing
    
    logger.info(f"✅ Feature agreement completed for {strategy}")
    return {"stage": "feature_agreement", "status": "completed", "strategy": strategy}

def run_feature_extraction(strategy: str, **context):
    """Execute feature extraction stage"""
    logger.info(f"⚡ Running feature extraction for strategy: {strategy}")
    
    import time
    # Simulate longer processing for feature extraction
    for i in range(10):
        time.sleep(2)
        logger.info(f"Feature extraction progress: {(i+1)*10}%")
    
    logger.info(f"✅ Feature extraction completed for {strategy}")
    return {"stage": "feature_extraction", "status": "completed", "strategy": strategy}

def run_model_training(strategy: str, **context):
    """Execute model training stage"""
    logger.info(f"🧠 Running model training for strategy: {strategy}")
    
    import time
    # Simulate model training
    for i in range(8):
        time.sleep(2)
        logger.info(f"Model training progress: {(i+1)*12.5}%")
    
    logger.info(f"✅ Model training completed for {strategy}")
    return {"stage": "model_training", "status": "completed", "strategy": strategy}

def run_model_registry(strategy: str, **context):
    """Execute model registry stage"""
    logger.info(f"📦 Running model registry for strategy: {strategy}")
    
    import time
    time.sleep(3)  # Simulate registry process
    
    logger.info(f"✅ Model registry completed for {strategy}")
    return {"stage": "model_registry", "status": "completed", "strategy": strategy}

# Create DAGs for each strategy
audio_only_dag = create_ml_pipeline_dag("audio_only")
multimodal_dag = create_ml_pipeline_dag("multimodal")
custom_dag = create_ml_pipeline_dag("custom")

# Expose DAGs to Airflow
globals()['ml_pipeline_audio_only'] = audio_only_dag
globals()['ml_pipeline_multimodal'] = multimodal_dag
globals()['ml_pipeline_custom'] = custom_dag 