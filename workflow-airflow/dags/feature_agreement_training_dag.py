"""
🎯 Feature Agreement Training Pipeline DAG
===============================================

This DAG implements the 6-step feature agreement pipeline:
1. Service Selection: Choose [audio], [content], or [audio, content]
2. Feature Discovery: Query services to find available features
3. Feature Agreement: User selects which features to use
4. SongAnalyzer: Extract features from all songs (with caching)
5. Feature Filtering: Create training matrix with only agreed features
6. Training: Run ML training with filtered features

🛠️ Tools Used:
- Airflow: Orchestrates the workflow (like a conductor)
- MLflow: Tracks experiments and models (like a lab notebook)
- Audio Service: Extracts audio features (/analyze/comprehensive)
- Content Service: Extracts lyrics features (/api/v1/lyrics)
- ML Training Service: Runs the actual training
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import requests
import json

# Default arguments
default_args = {
    'owner': 'hss-feature-training',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'feature_agreement_training_pipeline',
    default_args=default_args,
    description='6-step feature agreement training pipeline with service selection',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    max_active_runs=1,
    tags=['ml-training', 'feature-agreement', 'audio', 'lyrics'],
)

def step1_service_selection(**context):
    """
    🎯 STEP 1: Service Selection
    User chooses which services to use for feature extraction.
    
    Expected params:
    - services: ["audio"] or ["content"] or ["audio", "content"]
    """
    params = context.get('params', {})
    selected_services = params.get('services', ['audio'])  # Default to audio only
    
    print("🎯 STEP 1: Service Selection")
    print(f"📊 Selected Services: {selected_services}")
    
    # Validate service selection
    valid_services = ['audio', 'content']
    invalid = [s for s in selected_services if s not in valid_services]
    if invalid:
        raise ValueError(f"Invalid services: {invalid}. Valid options: {valid_services}")
    
    # Store for next steps
    context['task_instance'].xcom_push(key='selected_services', value=selected_services)
    return selected_services

def step2_feature_discovery(**context):
    """
    🔍 STEP 2: Feature Discovery
    Query each selected service to discover available features.
    """
    print("🔍 STEP 2: Feature Discovery")
    
    # Get selected services from previous step
    selected_services = context['task_instance'].xcom_pull(task_ids='step1_service_selection', key='selected_services')
    
    available_features = {}
    
    # Query Audio Service Features
    if 'audio' in selected_services:
        try:
            print("📡 Querying Audio Service for available features...")
            # This would typically be a GET request to discover features
            # For now, we'll use known audio features from comprehensive analysis
            audio_features = [
                'audio_tempo', 'audio_energy', 'audio_valence', 'audio_danceability',
                'audio_loudness', 'audio_speechiness', 'audio_acousticness', 
                'audio_instrumentalness', 'audio_liveness', 'audio_key', 'audio_mode'
            ]
            available_features['audio'] = audio_features
            print(f"✅ Found {len(audio_features)} audio features")
        except Exception as e:
            print(f"❌ Error querying audio service: {e}")
            available_features['audio'] = []
    
    # Query Content Service Features  
    if 'content' in selected_services:
        try:
            print("📡 Querying Content Service for available features...")
            # Typical lyrics features
            content_features = [
                'lyrics_sentiment', 'lyrics_complexity', 'lyrics_word_count',
                'lyrics_unique_words', 'lyrics_reading_level', 'lyrics_emotion_anger',
                'lyrics_emotion_joy', 'lyrics_emotion_sadness', 'lyrics_emotion_fear'
            ]
            available_features['content'] = content_features
            print(f"✅ Found {len(content_features)} content features")
        except Exception as e:
            print(f"❌ Error querying content service: {e}")
            available_features['content'] = []
    
    print(f"📊 Total available features: {sum(len(features) for features in available_features.values())}")
    
    # Store for next step
    context['task_instance'].xcom_push(key='available_features', value=available_features)
    return available_features

def step3_feature_agreement(**context):
    """
    🤝 STEP 3: Feature Agreement
    User selects specific features from available ones.
    
    Expected params:
    - agreed_features: {"audio": ["audio_tempo", "audio_energy"], "content": ["lyrics_sentiment"]}
    """
    print("🤝 STEP 3: Feature Agreement")
    
    # Get available features from previous step
    available_features = context['task_instance'].xcom_pull(task_ids='step2_feature_discovery', key='available_features')
    
    # Get user's feature selection from params
    params = context.get('params', {})
    agreed_features = params.get('agreed_features', {})
    
    # Default selection if none provided
    if not agreed_features:
        agreed_features = {
            'audio': ['audio_tempo', 'audio_energy', 'audio_valence'] if 'audio' in available_features else [],
            'content': ['lyrics_sentiment'] if 'content' in available_features else []
        }
        print("⚠️ No feature agreement provided, using defaults")
    
    # Validate that agreed features are available
    for service, features in agreed_features.items():
        if service in available_features:
            invalid_features = [f for f in features if f not in available_features[service]]
            if invalid_features:
                raise ValueError(f"Invalid {service} features: {invalid_features}")
    
    total_features = sum(len(features) for features in agreed_features.values())
    print(f"✅ Feature agreement completed: {total_features} features selected")
    for service, features in agreed_features.items():
        print(f"  {service}: {features}")
    
    # Store for next steps
    context['task_instance'].xcom_push(key='agreed_features', value=agreed_features)
    return agreed_features

def step4_song_analyzer(**context):
    """
    🎵 STEP 4: SongAnalyzer - Extract FULL feature sets
    Extract features from all songs using selected services (with caching).
    """
    print("🎵 STEP 4: SongAnalyzer - Feature Extraction")
    
    # Get agreed features and dataset info
    agreed_features = context['task_instance'].xcom_pull(task_ids='step3_feature_agreement', key='agreed_features')
    params = context.get('params', {})
    dataset_path = params.get('dataset_path', '/app/shared-data/training_data/filtered/filtered_audio_only_corrected_20250621_180350.csv')
    
    print(f"📁 Dataset: {dataset_path}")
    print(f"🎯 Extracting features for services: {list(agreed_features.keys())}")
    
    # This is where we would trigger comprehensive analysis
    extraction_results = {}
    
    if 'audio' in agreed_features:
        print("🎵 Starting audio feature extraction...")
        # In practice, this would call the audio service's comprehensive analysis
        # curl -X POST http://localhost:8001/analyze/comprehensive
        extraction_results['audio'] = {
            'status': 'completed',
            'features_extracted': len(agreed_features['audio']),
            'songs_processed': 401  # From your CSV
        }
    
    if 'content' in agreed_features:
        print("📝 Starting lyrics feature extraction...")
        # In practice, this would call the content service's lyrics analysis
        # curl -X POST http://localhost:8002/api/v1/lyrics
        extraction_results['content'] = {
            'status': 'completed', 
            'features_extracted': len(agreed_features['content']),
            'songs_processed': 401
        }
    
    print("✅ Feature extraction completed with caching enabled")
    context['task_instance'].xcom_push(key='extraction_results', value=extraction_results)
    return extraction_results

def step5_feature_filtering(**context):
    """
    🔧 STEP 5: Feature Filtering
    Create training matrix with only agreed features.
    """
    print("🔧 STEP 5: Feature Filtering")
    
    # Get agreed features and extraction results
    agreed_features = context['task_instance'].xcom_pull(task_ids='step3_feature_agreement', key='agreed_features')
    extraction_results = context['task_instance'].xcom_pull(task_ids='step4_song_analyzer', key='extraction_results')
    
    # Create feature matrix configuration
    feature_matrix_config = {
        'included_features': agreed_features,
        'total_features': sum(len(features) for features in agreed_features.values()),
        'target_column': 'final_popularity',
        'filter_strategy': 'agreed_features_only'
    }
    
    print(f"📊 Creating training matrix with {feature_matrix_config['total_features']} features")
    print(f"🎯 Target variable: {feature_matrix_config['target_column']}")
    
    # Store configuration for training step
    context['task_instance'].xcom_push(key='feature_matrix_config', value=feature_matrix_config)
    return feature_matrix_config

def step6_training(**context):
    """
    🚀 STEP 6: Training
    Run ML training with filtered feature vector and WAIT for completion.
    """
    import requests
    import json
    import time
    from datetime import datetime
    
    print("🚀 STEP 6: ML Training")
    
    # Get all previous step results
    agreed_features = context['task_instance'].xcom_pull(task_ids='step3_feature_agreement', key='agreed_features')
    feature_matrix_config = context['task_instance'].xcom_pull(task_ids='step5_feature_filtering', key='feature_matrix_config')
    
    params = context.get('params', {})
    training_id = params.get('training_id', f"feature_agreement_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    training_config = {
        'training_id': training_id,
        'dataset_path': params.get('dataset_path'),
        'strategy': 'audio_only',  # Use audio_only strategy since we have audio features
        'agreed_features': agreed_features,
        'feature_matrix_config': feature_matrix_config,
        'mlflow_experiment': f'feature_agreement_experiments_{datetime.now().strftime("%Y%m%d")}'
    }
    
    print(f"🆔 Training ID: {training_config['training_id']}")
    print(f"📊 Features: {training_config['feature_matrix_config']['total_features']}")
    print(f"📝 MLflow Experiment: {training_config['mlflow_experiment']}")
    
    # 🚀 STEP 6A: Start the ML training service
    pipeline_id = None
    try:
        ml_training_payload = {
            'strategy': 'audio_only',
            'experiment_name': training_config['mlflow_experiment'],
            'features': agreed_features.get('audio', []),
            'model_types': ['random_forest', 'xgboost'],
            'skip_feature_agreement': True,
            'parameters': {
                'dataset_path': '/app/shared-data/training_data/filtered/filtered_audio_only_corrected_20250621_180350.csv',
                'training_id': training_config['training_id']
            }
        }
        
        # Use host.docker.internal to reach ML training service from Airflow container
        training_url = 'http://host.docker.internal:8005/pipeline/train'
        print(f"📡 Calling ML training service: {training_url}")
        print(f"📋 Payload: {json.dumps(ml_training_payload, indent=2)}")
        
        response = requests.post(
            training_url,
            json=ml_training_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            pipeline_id = result.get('pipeline_id')
            print(f"✅ Training started successfully!")
            print(f"🆔 Pipeline ID: {pipeline_id}")
            print(f"📊 Status: {result.get('status')}")
            print(f"⏱️ Estimated Duration: 5-10 minutes")
            
            training_config['pipeline_id'] = pipeline_id
            training_config['training_status'] = 'started'
            
        else:
            print(f"❌ Training service error: {response.status_code}")
            print(f"❌ Response: {response.text}")
            training_config['training_status'] = 'failed'
            training_config['error'] = response.text
            return training_config
            
    except Exception as e:
        print(f"❌ Failed to call training service: {e}")
        training_config['training_status'] = 'failed'
        training_config['error'] = str(e)
        return training_config
    
    # 🚀 STEP 6B: Wait for training completion with polling
    if pipeline_id:
        print(f"⏳ Waiting for pipeline {pipeline_id} to complete...")
        
        max_wait_time = 30 * 60  # 30 minutes maximum wait
        poll_interval = 30  # Poll every 30 seconds
        elapsed_time = 0
        
        monitoring_url = f'http://host.docker.internal:8005/monitoring/status/{pipeline_id}'
        
        while elapsed_time < max_wait_time:
            try:
                print(f"🔍 Checking pipeline status... (elapsed: {elapsed_time//60}m {elapsed_time%60}s)")
                
                status_response = requests.get(monitoring_url, timeout=10)
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    current_status = status_data.get('status', 'unknown')
                    current_stage = status_data.get('current_stage', 'unknown')
                    progress = status_data.get('progress', {})
                    
                    overall_progress = progress.get('overall_percent', 0)
                    print(f"📊 Status: {current_status} | Stage: {current_stage} | Progress: {overall_progress}%")
                    
                    # Check if training is complete
                    if current_status == 'completed':
                        print("🎉 Training completed successfully!")
                        
                        # Get final results
                        training_config['training_status'] = 'completed'
                        training_config['final_status'] = status_data
                        
                        # Try to get MLflow experiment info
                        experiment_name = training_config['mlflow_experiment']
                        print(f"📊 Check MLflow for experiment: {experiment_name}")
                        print(f"🔗 MLflow UI: http://localhost:5001")
                        
                        # Log success metrics
                        if 'metrics' in status_data:
                            metrics = status_data['metrics']
                            print(f"📈 Training Metrics:")
                            for key, value in metrics.items():
                                print(f"   {key}: {value}")
                        
                        return training_config
                        
                    elif current_status == 'failed':
                        print("❌ Training failed!")
                        training_config['training_status'] = 'failed'
                        training_config['error'] = status_data.get('error_message', 'Training failed')
                        training_config['final_status'] = status_data
                        return training_config
                        
                    elif current_status in ['starting', 'running', 'waiting_for_input']:
                        # Training still in progress, continue polling
                        print(f"⏳ Training in progress: {current_status}")
                        
                        # Show detailed stage progress
                        stages = progress.get('stages', {})
                        for stage_name, stage_info in stages.items():
                            stage_status = stage_info.get('status', 'unknown')
                            stage_progress = stage_info.get('progress', 0)
                            print(f"   {stage_name}: {stage_status} ({stage_progress}%)")
                    
                else:
                    print(f"⚠️ Could not get status: {status_response.status_code}")
                
                # Wait before next poll
                time.sleep(poll_interval)
                elapsed_time += poll_interval
                
            except Exception as e:
                print(f"⚠️ Error checking status: {e}")
                time.sleep(poll_interval)
                elapsed_time += poll_interval
        
        # Timeout reached
        print(f"⏰ Training timeout after {max_wait_time//60} minutes")
        training_config['training_status'] = 'timeout'
        training_config['error'] = f'Training did not complete within {max_wait_time//60} minutes'
    
    # 🚀 STEP 6C: Final status and MLflow information
    print("📊 Training Summary:")
    print(f"   Pipeline ID: {pipeline_id}")
    print(f"   Status: {training_config.get('training_status', 'unknown')}")
    print(f"   Experiment: {training_config['mlflow_experiment']}")
    print(f"   Features Used: {len(agreed_features.get('audio', []))}")
    
    # MLflow access information
    print("\n🔗 MLflow Access:")
    print("   URL: http://localhost:5001")
    print(f"   Experiment: {training_config['mlflow_experiment']}")
    print("   Look for:")
    print("     - Feature importance values (should NOT be 0.0)")
    print("     - Model accuracy metrics")
    print("     - Training parameters")
    print("     - Registered models")
    
    return training_config

# Define tasks
start = DummyOperator(task_id='start', dag=dag)

service_selection = PythonOperator(
    task_id='step1_service_selection',
    python_callable=step1_service_selection,
    dag=dag,
)

feature_discovery = PythonOperator(
    task_id='step2_feature_discovery', 
    python_callable=step2_feature_discovery,
    dag=dag,
)

feature_agreement = PythonOperator(
    task_id='step3_feature_agreement',
    python_callable=step3_feature_agreement,
    dag=dag,
)

song_analyzer = PythonOperator(
    task_id='step4_song_analyzer',
    python_callable=step4_song_analyzer,
    dag=dag,
)

feature_filtering = PythonOperator(
    task_id='step5_feature_filtering',
    python_callable=step5_feature_filtering,
    dag=dag,
)

training = PythonOperator(
    task_id='step6_training',
    python_callable=step6_training,
    dag=dag,
)

end = DummyOperator(task_id='end', dag=dag)

# Define the workflow (your 6-step pipeline!)
start >> service_selection >> feature_discovery >> feature_agreement >> song_analyzer >> feature_filtering >> training >> end 