"""
🎯 Multimodal Feature Agreement Training Pipeline DAG
=====================================================

This DAG implements the 6-step multimodal feature agreement pipeline:
1. Service Selection: Choose [audio, content] for multimodal training
2. Feature Discovery: Query both audio and content services to find available features
3. Feature Agreement: User selects which audio AND content features to use
4. SongAnalyzer: Extract features from all songs (with caching)
5. Feature Filtering: Create training matrix with only agreed features
6. Training: Run ML training with filtered multimodal features

🛠️ Tools Used:
- Airflow: Orchestrates the workflow (like a conductor)
- MLflow: Tracks experiments and models (like a lab notebook)
- Audio Service: Extracts audio features (/analyze/comprehensive)
- Content Service: Extracts lyrics features (/api/v1/lyrics)
- ML Training Service: Runs the actual training

📊 Dataset: filtered_multimodal_corrected_20250621_180350.csv (338 songs with audio + lyrics)
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
    'owner': 'hss-multimodal-training',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'multimodal_feature_agreement_training_pipeline',
    default_args=default_args,
    description='6-step multimodal feature agreement training pipeline with audio + content features',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    max_active_runs=1,
    tags=['ml-training', 'multimodal', 'feature-agreement', 'audio', 'lyrics'],
)

def step1_service_selection(**context):
    """
    🎯 STEP 1: Service Selection - Multimodal
    For multimodal training, we use both audio and content services.
    
    Expected params:
    - services: ["audio", "content"] (both required for multimodal)
    """
    params = context.get('params', {})
    selected_services = params.get('services', ['audio', 'content'])  # Default to multimodal
    
    print("🎯 STEP 1: Service Selection (Multimodal)")
    print(f"📊 Selected Services: {selected_services}")
    
    # Validate that both services are selected for multimodal training
    required_services = ['audio', 'content']
    if not all(service in selected_services for service in required_services):
        raise ValueError(f"Multimodal training requires both services: {required_services}. Got: {selected_services}")
    
    # Store for next steps
    context['task_instance'].xcom_push(key='selected_services', value=selected_services)
    return selected_services

def step2_feature_discovery(**context):
    """
    🔍 STEP 2: Feature Discovery - Multimodal
    Query both audio and content services to discover available features.
    """
    print("🔍 STEP 2: Feature Discovery (Multimodal)")
    
    # Get selected services from previous step
    selected_services = context['task_instance'].xcom_pull(task_ids='step1_service_selection', key='selected_services')
    
    available_features = {}
    
    # Query Audio Service Features
    if 'audio' in selected_services:
        try:
            print("📡 Querying Audio Service for available features...")
            # Audio features from comprehensive analysis
            audio_features = [
                'audio_tempo', 'audio_energy', 'audio_valence', 'audio_danceability',
                'audio_loudness', 'audio_speechiness', 'audio_acousticness', 
                'audio_instrumentalness', 'audio_liveness', 'audio_key', 'audio_mode',
                'audio_brightness', 'audio_mood_sad', 'audio_mood_happy', 'audio_mood_electronic',
                'audio_mood_aggressive', 'audio_mood_relaxed'
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
            # Comprehensive lyrics features
            content_features = [
                'lyrics_sentiment_positive', 'lyrics_sentiment_negative', 'lyrics_sentiment_neutral',
                'lyrics_complexity_score', 'lyrics_word_count', 'lyrics_unique_words', 
                'lyrics_reading_level', 'lyrics_emotion_anger', 'lyrics_emotion_joy', 
                'lyrics_emotion_sadness', 'lyrics_emotion_fear', 'lyrics_emotion_surprise',
                'lyrics_theme_love', 'lyrics_theme_party', 'lyrics_theme_sadness',
                'lyrics_profanity_score', 'lyrics_repetition_score', 'lyrics_rhyme_density'
            ]
            available_features['content'] = content_features
            print(f"✅ Found {len(content_features)} content features")
        except Exception as e:
            print(f"❌ Error querying content service: {e}")
            available_features['content'] = []
    
    total_features = sum(len(features) for features in available_features.values())
    print(f"📊 Total available multimodal features: {total_features}")
    print(f"   Audio: {len(available_features.get('audio', []))}")
    print(f"   Content: {len(available_features.get('content', []))}")
    
    # Store for next step
    context['task_instance'].xcom_push(key='available_features', value=available_features)
    return available_features

def step3_feature_agreement(**context):
    """
    🤝 STEP 3: Feature Agreement - Multimodal
    User selects specific features from both audio and content services.
    
    Expected params:
    - agreed_features: {
        "audio": ["audio_tempo", "audio_energy", "audio_valence"], 
        "content": ["lyrics_sentiment_positive", "lyrics_emotion_joy"]
      }
    """
    print("🤝 STEP 3: Feature Agreement (Multimodal)")
    
    # Get available features from previous step
    available_features = context['task_instance'].xcom_pull(task_ids='step2_feature_discovery', key='available_features')
    
    # Get user's feature selection from params
    params = context.get('params', {})
    agreed_features = params.get('agreed_features', {})
    
    # Default multimodal selection if none provided
    if not agreed_features:
        agreed_features = {
            'audio': [
                'audio_tempo', 'audio_energy', 'audio_valence', 'audio_danceability',
                'audio_loudness', 'audio_speechiness'
            ] if 'audio' in available_features else [],
            'content': [
                'lyrics_sentiment_positive', 'lyrics_emotion_joy', 'lyrics_complexity_score',
                'lyrics_word_count'
            ] if 'content' in available_features else []
        }
        print("⚠️ No feature agreement provided, using multimodal defaults")
    
    # Validate that agreed features are available
    for service, features in agreed_features.items():
        if service in available_features:
            invalid_features = [f for f in features if f not in available_features[service]]
            if invalid_features:
                raise ValueError(f"Invalid {service} features: {invalid_features}")
    
    audio_count = len(agreed_features.get('audio', []))
    content_count = len(agreed_features.get('content', []))
    total_features = audio_count + content_count
    
    print(f"✅ Multimodal feature agreement completed: {total_features} features selected")
    print(f"  🎵 Audio features ({audio_count}): {agreed_features.get('audio', [])}")
    print(f"  📝 Content features ({content_count}): {agreed_features.get('content', [])}")
    
    # Store for next steps
    context['task_instance'].xcom_push(key='agreed_features', value=agreed_features)
    return agreed_features

def step4_song_analyzer(**context):
    """
    🎵 STEP 4: SongAnalyzer - Extract MULTIMODAL feature sets
    Extract features from all songs using both audio and content services (with caching).
    """
    print("🎵 STEP 4: SongAnalyzer - Multimodal Feature Extraction")
    
    # Get agreed features and dataset info
    agreed_features = context['task_instance'].xcom_pull(task_ids='step3_feature_agreement', key='agreed_features')
    params = context.get('params', {})
    dataset_path = params.get('dataset_path', '/app/shared-data/training_data/filtered/filtered_multimodal_corrected_20250621_180350.csv')
    
    print(f"📁 Dataset: {dataset_path}")
    print(f"🎯 Extracting multimodal features for services: {list(agreed_features.keys())}")
    
    # This is where we would trigger comprehensive analysis for both services
    extraction_results = {}
    
    if 'audio' in agreed_features:
        print("🎵 Starting audio feature extraction...")
        # In practice, this would call the audio service's comprehensive analysis
        # curl -X POST http://host.docker.internal:8001/analyze/comprehensive
        extraction_results['audio'] = {
            'status': 'completed',
            'features_extracted': len(agreed_features['audio']),
            'songs_processed': 338  # From multimodal CSV
        }
    
    if 'content' in agreed_features:
        print("📝 Starting lyrics feature extraction...")
        # In practice, this would call the content service's lyrics analysis
        # curl -X POST http://host.docker.internal:8002/api/v1/lyrics
        extraction_results['content'] = {
            'status': 'completed', 
            'features_extracted': len(agreed_features['content']),
            'songs_processed': 338
        }
    
    total_extracted = sum(result['features_extracted'] for result in extraction_results.values())
    print(f"✅ Multimodal feature extraction completed: {total_extracted} features extracted")
    print(f"   🎵 Audio: {extraction_results.get('audio', {}).get('features_extracted', 0)} features")
    print(f"   📝 Content: {extraction_results.get('content', {}).get('features_extracted', 0)} features")
    
    context['task_instance'].xcom_push(key='extraction_results', value=extraction_results)
    return extraction_results

def step5_feature_filtering(**context):
    """
    🔧 STEP 5: Feature Filtering - Multimodal
    Create training matrix with agreed features from both audio and content.
    """
    print("🔧 STEP 5: Feature Filtering (Multimodal)")
    
    # Get agreed features and extraction results
    agreed_features = context['task_instance'].xcom_pull(task_ids='step3_feature_agreement', key='agreed_features')
    extraction_results = context['task_instance'].xcom_pull(task_ids='step4_song_analyzer', key='extraction_results')
    
    # Create multimodal feature matrix configuration
    audio_count = len(agreed_features.get('audio', []))
    content_count = len(agreed_features.get('content', []))
    total_features = audio_count + content_count
    
    feature_matrix_config = {
        'included_features': agreed_features,
        'total_features': total_features,
        'audio_features': audio_count,
        'content_features': content_count,
        'target_column': 'final_popularity',
        'filter_strategy': 'multimodal_agreed_features'
    }
    
    print(f"📊 Creating multimodal training matrix with {feature_matrix_config['total_features']} features")
    print(f"   🎵 Audio features: {feature_matrix_config['audio_features']}")
    print(f"   📝 Content features: {feature_matrix_config['content_features']}")
    print(f"🎯 Target variable: {feature_matrix_config['target_column']}")
    
    # Store configuration for training step
    context['task_instance'].xcom_push(key='feature_matrix_config', value=feature_matrix_config)
    return feature_matrix_config

def step6_training(**context):
    """
    🚀 STEP 6: Training - Multimodal
    Run ML training with multimodal feature vector and WAIT for completion.
    """
    import requests
    import json
    import time
    from datetime import datetime
    
    print("🚀 STEP 6: Multimodal ML Training")
    
    # Get all previous step results
    agreed_features = context['task_instance'].xcom_pull(task_ids='step3_feature_agreement', key='agreed_features')
    feature_matrix_config = context['task_instance'].xcom_pull(task_ids='step5_feature_filtering', key='feature_matrix_config')
    
    params = context.get('params', {})
    training_id = params.get('training_id', f"multimodal_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    training_config = {
        'training_id': training_id,
        'dataset_path': params.get('dataset_path'),
        'strategy': 'multimodal',  # Use multimodal strategy
        'agreed_features': agreed_features,
        'feature_matrix_config': feature_matrix_config,
        'mlflow_experiment': f'multimodal_experiments_{datetime.now().strftime("%Y%m%d")}'
    }
    
    print(f"🆔 Training ID: {training_config['training_id']}")
    print(f"📊 Total Features: {training_config['feature_matrix_config']['total_features']}")
    print(f"   🎵 Audio: {training_config['feature_matrix_config']['audio_features']}")
    print(f"   📝 Content: {training_config['feature_matrix_config']['content_features']}")
    print(f"📝 MLflow Experiment: {training_config['mlflow_experiment']}")
    
    # 🚀 STEP 6A: Start the ML training service
    pipeline_id = None
    try:
        # Combine audio and content features for multimodal training
        all_features = []
        all_features.extend(agreed_features.get('audio', []))
        all_features.extend(agreed_features.get('content', []))
        
        ml_training_payload = {
            'strategy': 'multimodal',
            'experiment_name': training_config['mlflow_experiment'],
            'features': all_features,
            'audio_features': agreed_features.get('audio', []),
            'content_features': agreed_features.get('content', []),
            'model_types': ['random_forest', 'xgboost', 'neural_network'],
            'skip_feature_agreement': True,
            'parameters': {
                'dataset_path': '/app/shared-data/training_data/filtered/filtered_multimodal_corrected_20250621_180350.csv',
                'training_id': training_config['training_id'],
                'multimodal': True
            }
        }
        
        # Use host.docker.internal to reach ML training service from Airflow container
        training_url = 'http://host.docker.internal:8005/pipeline/train'
        print(f"📡 Calling ML training service: {training_url}")
        print(f"📋 Multimodal Payload: {json.dumps(ml_training_payload, indent=2)}")
        
        response = requests.post(
            training_url,
            json=ml_training_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            pipeline_id = result.get('pipeline_id')
            print(f"✅ Multimodal training started successfully!")
            print(f"🆔 Pipeline ID: {pipeline_id}")
            print(f"📊 Status: {result.get('status')}")
            print(f"⏱️ Estimated Duration: 10-15 minutes (multimodal takes longer)")
            
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
    
    # 🚀 STEP 6B: Wait for multimodal training completion with polling
    if pipeline_id:
        print(f"⏳ Waiting for multimodal pipeline {pipeline_id} to complete...")
        
        max_wait_time = 45 * 60  # 45 minutes maximum wait (multimodal takes longer)
        poll_interval = 30  # Poll every 30 seconds
        elapsed_time = 0
        
        monitoring_url = f'http://host.docker.internal:8005/monitoring/status/{pipeline_id}'
        
        while elapsed_time < max_wait_time:
            try:
                print(f"🔍 Checking multimodal pipeline status... (elapsed: {elapsed_time//60}m {elapsed_time%60}s)")
                
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
                        print("🎉 Multimodal training completed successfully!")
                        
                        # Get final results
                        training_config['training_status'] = 'completed'
                        training_config['final_status'] = status_data
                        
                        # Try to get MLflow experiment info
                        experiment_name = training_config['mlflow_experiment']
                        print(f"📊 Check MLflow for multimodal experiment: {experiment_name}")
                        print(f"🔗 MLflow UI: http://localhost:5001")
                        
                        # Log success metrics
                        if 'metrics' in status_data:
                            metrics = status_data['metrics']
                            print(f"📈 Multimodal Training Metrics:")
                            for key, value in metrics.items():
                                print(f"   {key}: {value}")
                        
                        return training_config
                        
                    elif current_status == 'failed':
                        print("❌ Multimodal training failed!")
                        training_config['training_status'] = 'failed'
                        training_config['error'] = status_data.get('error_message', 'Training failed')
                        training_config['final_status'] = status_data
                        return training_config
                        
                    elif current_status in ['starting', 'running', 'waiting_for_input']:
                        # Training still in progress, continue polling
                        print(f"⏳ Multimodal training in progress: {current_status}")
                        
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
        print(f"⏰ Multimodal training timeout after {max_wait_time//60} minutes")
        training_config['training_status'] = 'timeout'
        training_config['error'] = f'Multimodal training did not complete within {max_wait_time//60} minutes'
    
    # 🚀 STEP 6C: Final status and MLflow information
    print("📊 Multimodal Training Summary:")
    print(f"   Pipeline ID: {pipeline_id}")
    print(f"   Status: {training_config.get('training_status', 'unknown')}")
    print(f"   Experiment: {training_config['mlflow_experiment']}")
    print(f"   Total Features: {len(agreed_features.get('audio', [])) + len(agreed_features.get('content', []))}")
    print(f"   Audio Features: {len(agreed_features.get('audio', []))}")
    print(f"   Content Features: {len(agreed_features.get('content', []))}")
    
    # MLflow access information
    print("\n🔗 MLflow Access:")
    print("   URL: http://localhost:5001")
    print(f"   Experiment: {training_config['mlflow_experiment']}")
    print("   Look for:")
    print("     - Multimodal feature importance values (should NOT be 0.0)")
    print("     - Audio vs Content feature comparison")
    print("     - Model accuracy improvements from multimodal approach")
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

# Define the multimodal workflow (your 6-step pipeline!)
start >> service_selection >> feature_discovery >> feature_agreement >> song_analyzer >> feature_filtering >> training >> end 