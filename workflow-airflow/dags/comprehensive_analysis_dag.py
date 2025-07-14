"""
Comprehensive Analysis DAG for HSS Workflow system.
This DAG replaces the comprehensive workflow template from the current orchestrator.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.dummy import DummyOperator

# Import custom operators
import sys
sys.path.append('/opt/airflow/plugins')
from workflow_operators import (
    AudioAnalysisOperator,
    ContentAnalysisOperator,
    MLPredictionOperator,
    IntelligenceOperator,
    WorkflowResultsAggregatorOperator
)

# Default arguments for the DAG
default_args = {
    'owner': 'hss-workflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 5,
}

# Create the DAG
dag = DAG(
    'comprehensive_analysis',
    default_args=default_args,
    description='Comprehensive music analysis workflow with audio, content, prediction, and AI insights',
    schedule_interval=None,  # Triggered manually or via API
    catchup=False,
    max_active_runs=5,
    tags=['music-analysis', 'comprehensive', 'ml-prediction', 'ai-insights'],
)

# Start task
start_task = DummyOperator(
    task_id='start_comprehensive_analysis',
    dag=dag,
)

# Audio Analysis Task
audio_analysis = AudioAnalysisOperator(
    task_id='audio_analysis',
    include_advanced_features=True,
    include_spectral=True,
    retries=3,
    retry_delay=timedelta(minutes=2),
    dag=dag,
)

# Content Analysis Task (runs in parallel with audio analysis)
content_analysis = ContentAnalysisOperator(
    task_id='content_analysis',
    include_sentiment=True,
    include_features=True,
    retries=3,
    retry_delay=timedelta(minutes=1),
    dag=dag,
)

# Hit Prediction Task (depends on both audio and content analysis)
hit_prediction = MLPredictionOperator(
    task_id='hit_prediction',
    model_type='hit_prediction',
    include_explanation=True,
    retries=2,
    retry_delay=timedelta(minutes=1),
    dag=dag,
)

# AI Insights Task (depends on all previous tasks)
ai_insights = IntelligenceOperator(
    task_id='ai_insights',
    analysis_types=['musical_meaning', 'hit_comparison', 'strategic_insights'],
    agent_type='comprehensive',
    retries=2,
    retry_delay=timedelta(minutes=1),
    dag=dag,
)

# Results Aggregation Task
aggregate_results = WorkflowResultsAggregatorOperator(
    task_id='aggregate_results',
    result_keys=['audio_features', 'content_features', 'prediction', 'insights'],
    output_path='/opt/airflow/shared-data/results/comprehensive_{{ ds }}_{{ ts_nodash }}.json',
    dag=dag,
)

# End task
end_task = DummyOperator(
    task_id='end_comprehensive_analysis',
    dag=dag,
)

# Define task dependencies (DAG structure)
start_task >> [audio_analysis, content_analysis]
[audio_analysis, content_analysis] >> hit_prediction
hit_prediction >> ai_insights
ai_insights >> aggregate_results
aggregate_results >> end_task 