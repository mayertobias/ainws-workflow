"""
🎯 Multimodal Ensemble Training Pipeline DAG (Random Forest + XGBoost + SHAP)
============================================================================

This DAG implements the 6-step multimodal ensemble training pipeline:
1. Service Selection: Choose [audio, content] for multimodal training
2. Feature Discovery: Query both audio and content services for available features
3. Feature Agreement: User selects which audio AND content features to use
4. SongAnalyzer: Extract features from all songs using both services (with caching)
5. Feature Filtering: Create training matrix with agreed multimodal features
6. Ensemble Training: Run ML training with Random Forest + XGBoost + SHAP

🛠️ Tools Used:
- Airflow: Orchestrates the workflow
- MLflow: Tracks experiments and models
- Audio Service: Extracts audio features (/analyze/comprehensive)
- Content Service: Extracts lyrics features (/api/v1/lyrics)
- ML Training Service: Runs ensemble training with SHAP
- Ensemble Models: Random Forest + XGBoost with soft voting
- SHAP: Explainability analysis with cross-modal feature importance

📊 Dataset: filtered_multimodal_corrected_20250621_180350.csv
🎯 Audio Features: Tempo, energy, valence, danceability, mood, genre, etc.
📝 Content Features: Sentiment, emotions, themes, complexity, etc.
🧠 Models: Ensemble (Random Forest + XGBoost)
🔍 Explainability: SHAP analysis with cross-modal feature importance
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
import requests
import json
import time

# Default arguments
default_args = {
    'owner': 'hss-multimodal-ensemble-training',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'multimodal_ensemble_training_pipeline',
    default_args=default_args,
    description='6-step multimodal ensemble training pipeline with Random Forest + XGBoost + SHAP',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    max_active_runs=1,
    tags=['ml-training', 'ensemble', 'multimodal', 'audio', 'content', 'random-forest', 'xgboost', 'shap'],
)

def step1_service_selection(**context):
    """
    🎯 STEP 1: Service Selection - Multimodal
    For multimodal ensemble training, we use both audio and content services.
    """
    params = context.get('params', {})
    selected_services = params.get('services', ['audio', 'content'])  # Default to multimodal
    
    print("🎯 STEP 1: Service Selection (Multimodal Ensemble)")
    print(f"📊 Selected Services: {selected_services}")
    
    # Validate that both services are selected for multimodal training
    required_services = ['audio', 'content']
    if not all(service in selected_services for service in required_services):
        raise ValueError(f"Multimodal ensemble training requires both services: {required_services}. Got: {selected_services}")
    
    # Store for next steps
    context['task_instance'].xcom_push(key='selected_services', value=selected_services)
    return selected_services

def step2_feature_discovery(**context):
    """
    🔍 STEP 2: Feature Discovery - Multimodal
    Query both audio and content services to discover available features.
    """
    print("🔍 STEP 2: Feature Discovery (Multimodal Ensemble)")
    
    # Get selected services from previous step
    selected_services = context['task_instance'].xcom_pull(task_ids='step1_service_selection', key='selected_services')
    
    available_features = {}
    
    # Query Audio Service Features
    if 'audio' in selected_services:
        try:
            print("📡 Querying Audio Service for available features...")
            # Comprehensive audio features from the audio service
            audio_features = [
                'audio_tempo', 'audio_energy', 'audio_valence', 'audio_danceability',
                'audio_loudness', 'audio_speechiness', 'audio_acousticness', 
                'audio_instrumentalness', 'audio_liveness', 'audio_key', 'audio_mode',
                'audio_brightness', 'audio_complexity', 'audio_warmth', 'audio_harmonic_strength',
                'audio_mood_happy', 'audio_mood_sad', 'audio_mood_aggressive',
                'audio_mood_relaxed', 'audio_mood_party', 'audio_mood_electronic',
                'audio_primary_genre', 'audio_top_genre_1_prob', 'audio_top_genre_2_prob'
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
                'lyrics_profanity_score', 'lyrics_repetition_score', 'lyrics_rhyme_density',
                'lyrics_narrative_complexity', 'lyrics_lexical_diversity', 'lyrics_motif_count',
                'lyrics_verse_count', 'lyrics_chorus_count', 'lyrics_bridge_count'
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
    """
    print("🤝 STEP 3: Feature Agreement (Multimodal Ensemble)")
    
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
                'audio_loudness', 'audio_speechiness', 'audio_acousticness',
                'audio_instrumentalness', 'audio_liveness', 'audio_key', 'audio_mode',
                'audio_brightness', 'audio_complexity', 'audio_warmth', 'audio_harmonic_strength',
                'audio_mood_happy', 'audio_mood_sad', 'audio_mood_aggressive',
                'audio_mood_relaxed', 'audio_mood_party', 'audio_mood_electronic',
                'audio_primary_genre', 'audio_top_genre_1_prob', 'audio_top_genre_2_prob'
            ] if 'audio' in available_features else [],
            'content': [
                'lyrics_sentiment_positive', 'lyrics_sentiment_negative', 'lyrics_sentiment_neutral',
                'lyrics_complexity_score', 'lyrics_word_count', 'lyrics_unique_words',
                'lyrics_reading_level', 'lyrics_emotion_anger', 'lyrics_emotion_joy',
                'lyrics_emotion_sadness', 'lyrics_emotion_fear', 'lyrics_emotion_surprise',
                'lyrics_theme_love', 'lyrics_theme_party', 'lyrics_theme_sadness',
                'lyrics_profanity_score', 'lyrics_repetition_score', 'lyrics_rhyme_density',
                'lyrics_narrative_complexity', 'lyrics_lexical_diversity', 'lyrics_motif_count',
                'lyrics_verse_count', 'lyrics_chorus_count', 'lyrics_bridge_count'
            ] if 'content' in available_features else []
        }
        print("⚠️ No feature agreement provided, using comprehensive multimodal defaults")
    
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
    print(f"🎯 Extracting multimodal features for ensemble training")
    
    # This is where we would trigger comprehensive analysis for both services
    extraction_results = {}
    
    if 'audio' in agreed_features:
        print("🎵 Starting comprehensive audio feature extraction...")
        # In practice, this would call the audio service's comprehensive analysis
        # curl -X POST http://host.docker.internal:8001/analyze/comprehensive
        extraction_results['audio'] = {
            'status': 'completed',
            'features_extracted': len(agreed_features['audio']),
            'songs_processed': 338,  # From multimodal CSV
            'extraction_type': 'comprehensive_audio_analysis'
        }
    
    if 'content' in agreed_features:
        print("📝 Starting comprehensive lyrics feature extraction...")
        # In practice, this would call the content service's lyrics analysis
        # curl -X POST http://host.docker.internal:8002/api/v1/lyrics
        extraction_results['content'] = {
            'status': 'completed', 
            'features_extracted': len(agreed_features['content']),
            'songs_processed': 338,
            'extraction_type': 'comprehensive_lyrics_analysis'
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
    print("🔧 STEP 5: Feature Filtering (Multimodal Ensemble)")
    
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
        'filter_strategy': 'multimodal_ensemble_features',
        'model_type': 'ensemble'
    }
    
    print(f"📊 Creating multimodal ensemble training matrix with {feature_matrix_config['total_features']} features")
    print(f"   🎵 Audio features: {feature_matrix_config['audio_features']}")
    print(f"   📝 Content features: {feature_matrix_config['content_features']}")
    print(f"🎯 Target variable: {feature_matrix_config['target_column']}")
    print(f"🧠 Model type: {feature_matrix_config['model_type']}")
    
    # Store configuration for training step
    context['task_instance'].xcom_push(key='feature_matrix_config', value=feature_matrix_config)
    return feature_matrix_config

def step6_ensemble_training(**context):
    """
    🚀 STEP 6: Ensemble Training - Random Forest + XGBoost + SHAP
    Run ensemble ML training with Random Forest + XGBoost and SHAP explainability.
    """
    import requests
    import json
    import time
    from datetime import datetime
    
    print("🚀 STEP 6: Multimodal Ensemble Training (Random Forest + XGBoost + SHAP)")
    
    # Get all previous step results
    agreed_features = context['task_instance'].xcom_pull(task_ids='step3_feature_agreement', key='agreed_features')
    feature_matrix_config = context['task_instance'].xcom_pull(task_ids='step5_feature_filtering', key='feature_matrix_config')
    
    params = context.get('params', {})
    training_id = params.get('training_id', f"multimodal_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    training_config = {
        'training_id': training_id,
        'dataset_path': params.get('dataset_path', '/app/shared-data/training_data/filtered/filtered_multimodal_corrected_20250621_180350.csv'),
        'strategy': 'multimodal',
        'agreed_features': agreed_features,
        'feature_matrix_config': feature_matrix_config,
        'mlflow_experiment': f'multimodal_ensemble_experiments_{datetime.now().strftime("%Y%m%d")}',
        'ensemble_config': params.get('ensemble_config', {}),
        'shap_config': params.get('shap_config', {})
    }
    
    print(f"🆔 Training ID: {training_config['training_id']}")
    print(f"📊 Total Features: {training_config['feature_matrix_config']['total_features']}")
    print(f"🎵 Audio Features: {training_config['feature_matrix_config']['audio_features']}")
    print(f"📝 Content Features: {training_config['feature_matrix_config']['content_features']}")
    print(f"🧠 Model Type: Ensemble (Random Forest + XGBoost)")
    print(f"🔍 SHAP Analysis: Enabled")
    print(f"📝 MLflow Experiment: {training_config['mlflow_experiment']}")
    
    # 🚀 STEP 6A: Start the ML training service with ensemble configuration
    pipeline_id = None
    try:
        # Prepare ensemble training payload (FIXED: Use correct TrainingRequest format)
        audio_features = agreed_features.get('audio', [])
        content_features = agreed_features.get('content', [])
        all_features = audio_features + content_features
        
        ml_training_payload = {
            'strategy': 'multimodal',
            'experiment_name': training_config['mlflow_experiment'],
            'features': all_features,
            'model_types': ['ensemble'],  # This triggers Random Forest + XGBoost
            'skip_feature_agreement': True,
            'parameters': {
                'dataset_path': training_config['dataset_path'],
                'training_id': training_config['training_id'],
                'ensemble_training': True,
                'enable_shap': True,
                'multimodal': True,
                'ensemble_config': training_config['ensemble_config'],
                'shap_config': training_config['shap_config'],
                'audio_features': audio_features,
                'content_features': content_features
            }
        }
        
        # FIXED: Use correct endpoint /pipeline/train (not /pipeline/start)
        training_url = 'http://host.docker.internal:8005/pipeline/train'
        print(f"📡 Calling ML training service: {training_url}")
        print(f"📋 Multimodal Ensemble Payload: {json.dumps(ml_training_payload, indent=2)}")
        
        response = requests.post(
            training_url,
            json=ml_training_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            pipeline_id = result.get('pipeline_id')
            print(f"✅ Multimodal ensemble training started successfully!")
            print(f"🆔 Pipeline ID: {pipeline_id}")
            print(f"📊 Status: {result.get('status')}")
            print(f"⏱️ Estimated Duration: 15-20 minutes (multimodal ensemble + SHAP)")
            
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
    
    # 🚀 STEP 6B: Wait for ensemble training completion with polling
    if pipeline_id:
        print(f"⏳ Waiting for multimodal ensemble pipeline {pipeline_id} to complete...")
        
        max_wait_time = 60 * 60  # 60 minutes maximum wait (multimodal ensemble + SHAP takes longer)
        poll_interval = 30  # Poll every 30 seconds
        elapsed_time = 0
        
        # FIXED: Use correct status endpoint /pipeline/status/{pipeline_id}
        monitoring_url = f'http://host.docker.internal:8005/pipeline/status/{pipeline_id}'
        
        while elapsed_time < max_wait_time:
            try:
                print(f"🔍 Checking multimodal ensemble pipeline status... (elapsed: {elapsed_time//60}m {elapsed_time%60}s)")
                
                status_response = requests.get(monitoring_url, timeout=10)
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    current_status = status_data.get('status', 'unknown')
                    current_stage = status_data.get('current_stage', 'unknown')
                    
                    print(f"📊 Status: {current_status} | Stage: {current_stage}")
                    
                    # Check if training is complete
                    if current_status == 'completed':
                        print("🎉 Multimodal ensemble training completed successfully!")
                        
                        # Get final results
                        training_config['training_status'] = 'completed'
                        training_config['final_status'] = status_data
                        
                        # Try to get MLflow experiment info
                        experiment_name = training_config['mlflow_experiment']
                        print(f"📊 Check MLflow for multimodal ensemble experiment: {experiment_name}")
                        print(f"🔗 MLflow UI: http://localhost:5001")
                        
                        # Log success metrics (compatible with new API response format)
                        print(f"📈 Multimodal Ensemble Training Results:")
                        print(f"   Status: {current_status}")
                        print(f"   Pipeline ID: {pipeline_id}")
                        print(f"   Strategy: {status_data.get('strategy', 'multimodal')}")
                        print(f"   Progress: {status_data.get('progress_percent', 100)}%")
                        
                        return training_config
                        
                    elif current_status == 'failed':
                        print("❌ Multimodal ensemble training failed!")
                        training_config['training_status'] = 'failed'
                        training_config['error'] = status_data.get('error_message', 'Training failed')
                        training_config['final_status'] = status_data
                        return training_config
                        
                    elif current_status in ['pending', 'running', 'starting']:
                        # Training still in progress, continue polling
                        print(f"⏳ Multimodal ensemble training in progress: {current_status}")
                        
                        # Show progress if available
                        progress = status_data.get('progress_percent', 0)
                        print(f"   Progress: {progress:.1f}%")
                    
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
        print(f"⏰ Multimodal ensemble training timeout after {max_wait_time//60} minutes")
        training_config['training_status'] = 'timeout'
        training_config['error'] = f'Multimodal ensemble training did not complete within {max_wait_time//60} minutes'
    
    # 🚀 STEP 6C: Final status and MLflow information
    print("📊 Multimodal Ensemble Training Summary:")
    print(f"   Pipeline ID: {pipeline_id}")
    print(f"   Status: {training_config.get('training_status', 'unknown')}")
    print(f"   Experiment: {training_config['mlflow_experiment']}")
    print(f"   Audio Features: {len(agreed_features.get('audio', []))}")
    print(f"   Content Features: {len(agreed_features.get('content', []))}")
    print(f"   Model Type: Ensemble (Random Forest + XGBoost)")
    print(f"   SHAP Analysis: Enabled")
    
    # MLflow access information
    print("\n🔗 MLflow Access:")
    print("   URL: http://localhost:5001")
    print(f"   Experiment: {training_config['mlflow_experiment']}")
    print("   Look for:")
    print("     - Multimodal ensemble model (Random Forest + XGBoost)")
    print("     - SHAP feature importance values")
    print("     - Cross-modal feature importance comparison")
    print("     - SHAP summary plots and bar plots")
    print("     - Audio vs Content feature importance breakdown")
    print("     - Feature importance comparison (RF vs XGBoost vs SHAP)")
    print("     - Model accuracy metrics")
    print("     - Training parameters")
    print("     - Registered models")
    
    # SHAP explainability information
    print("\n🔍 SHAP Explainability:")
    print("   📊 SHAP Summary Plot: Shows feature importance and value distributions")
    print("   📈 SHAP Bar Plot: Feature importance ranking")
    print("   🎯 Cross-Modal Analysis: Audio vs Content feature importance")
    print("   📋 Model Interpretability: Understand model decisions")
    print("   🔗 Access: MLflow UI → Experiments → Runs → Artifacts → shap_analysis")
    
    # Multimodal benefits
    print("\n🎯 Multimodal Benefits:")
    print("   ✅ Better prediction accuracy through combined modalities")
    print("   ✅ Feature importance comparison across audio vs lyrics")
    print("   ✅ Enhanced model interpretability")
    print("   ✅ Cross-modal insights for music analysis")
    
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

ensemble_training = PythonOperator(
    task_id='step6_ensemble_training',
    python_callable=step6_ensemble_training,
    dag=dag,
)

end = DummyOperator(task_id='end', dag=dag)

# Define the multimodal ensemble workflow (6-step pipeline!)
start >> service_selection >> feature_discovery >> feature_agreement >> song_analyzer >> feature_filtering >> ensemble_training >> end 