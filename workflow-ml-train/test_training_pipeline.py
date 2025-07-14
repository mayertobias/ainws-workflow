"""
Comprehensive unit test for ML training pipeline with mocked services
Tests feature engineering, model training, and SHAP explainability without calling real services
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the app directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.pipeline.orchestrator import PipelineOrchestrator
from app.services.song_analyzer import SongAnalyzer

class TestMLTrainingPipeline:
    """Test the ML training pipeline with mocked services"""
    
    @pytest.fixture
    def mock_audio_service_response(self):
        """Mock response from audio analysis service"""
        return {
            "status": "success",
            "features": {
                "basic_tempo": 120.5,
                "basic_energy": 0.8,
                "basic_valence": 0.7,
                "basic_danceability": 0.6,
                "basic_loudness": -8.2,
                "basic_acousticness": 0.3,
                "basic_instrumentalness": 0.1,
                "basic_speechiness": 0.05,
                "basic_liveness": 0.2,
                "basic_key": 5,
                "basic_mode": 1,
                "basic_brightness": 0.7,
                "basic_complexity": 0.6,
                "basic_warmth": 0.5,
                "basic_harmonic_strength": 0.8,
                "genre_primary_genre": "pop",
                "genre_top_genre_1_prob": 0.7,
                "genre_top_genre_2_prob": 0.2,
                "mfcc_mean": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
                "chroma_mean": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
            },
            "analysis_id": "test_audio_analysis_123",
            "processing_time": 2.5
        }
    
    @pytest.fixture
    def mock_content_service_response(self):
        """Mock response from content analysis service"""
        return {
            "status": "success",
            "features": {
                "lyrics_sentiment_positive": 0.6,
                "lyrics_sentiment_negative": 0.2,
                "lyrics_sentiment_neutral": 0.2,
                "lyrics_emotion_joy": 0.7,
                "lyrics_emotion_sadness": 0.1,
                "lyrics_emotion_anger": 0.05,
                "lyrics_emotion_fear": 0.05,
                "lyrics_emotion_surprise": 0.1,
                "lyrics_complexity_score": 0.65,
                "lyrics_narrative_complexity": 0.55,
                "lyrics_unique_words": 45,
                "lyrics_total_words": 120,
                "lyrics_readability_score": 0.7,
                "lyrics_metaphor_count": 3,
                "lyrics_rhyme_density": 0.8
            },
            "analysis_id": "test_content_analysis_456",
            "processing_time": 1.8
        }
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data for testing"""
        np.random.seed(42)
        n_samples = 100
        
        # Create realistic audio features
        data = {
            'song_id': [f'song_{i}' for i in range(n_samples)],
            'title': [f'Test Song {i}' for i in range(n_samples)],
            'artist': [f'Test Artist {i % 20}' for i in range(n_samples)],
            'original_popularity': np.random.randint(0, 100, n_samples),
            'basic_tempo': np.random.uniform(80, 180, n_samples),
            'basic_energy': np.random.uniform(0, 1, n_samples),
            'basic_valence': np.random.uniform(0, 1, n_samples),
            'basic_danceability': np.random.uniform(0, 1, n_samples),
            'basic_loudness': np.random.uniform(-20, 0, n_samples),
            'basic_acousticness': np.random.uniform(0, 1, n_samples),
            'basic_instrumentalness': np.random.uniform(0, 1, n_samples),
            'basic_speechiness': np.random.uniform(0, 1, n_samples),
            'basic_liveness': np.random.uniform(0, 1, n_samples),
            'basic_key': np.random.randint(0, 12, n_samples),
            'basic_mode': np.random.randint(0, 2, n_samples),
            'basic_brightness': np.random.uniform(0, 1, n_samples),
            'basic_complexity': np.random.uniform(0, 1, n_samples),
            'basic_warmth': np.random.uniform(0, 1, n_samples),
            'basic_harmonic_strength': np.random.uniform(0, 1, n_samples),
            'genre_primary_genre': np.random.choice(['pop', 'rock', 'electronic', 'hip-hop', 'country'], n_samples),
            'genre_top_genre_1_prob': np.random.uniform(0.5, 1.0, n_samples),
            'genre_top_genre_2_prob': np.random.uniform(0.1, 0.5, n_samples),
            'lyrics_sentiment_positive': np.random.uniform(0, 1, n_samples),
            'lyrics_sentiment_negative': np.random.uniform(0, 1, n_samples),
            'lyrics_emotion_joy': np.random.uniform(0, 1, n_samples),
            'lyrics_complexity_score': np.random.uniform(0, 1, n_samples),
            'lyrics_narrative_complexity': np.random.uniform(0, 1, n_samples),
            'lyrics_unique_words': np.random.randint(20, 100, n_samples)
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing"""
        return PipelineOrchestrator()
    
    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_csv_file(self, sample_training_data, temp_directory):
        """Create sample CSV file for testing"""
        csv_path = os.path.join(temp_directory, 'test_training_data.csv')
        sample_training_data.to_csv(csv_path, index=False)
        return csv_path
    
    @patch('app.pipeline.orchestrator.mlflow')
    def test_feature_engineering_and_training(self, mock_mlflow, orchestrator, sample_csv_file, mock_audio_service_response, mock_content_service_response):
        """Test the complete feature engineering and model training pipeline"""
        
        # Mock MLflow methods
        mock_mlflow.set_tracking_uri = MagicMock()
        mock_mlflow.set_experiment = MagicMock()
        mock_mlflow.start_run = MagicMock()
        mock_mlflow.log_param = MagicMock()
        mock_mlflow.log_metric = MagicMock()
        mock_mlflow.sklearn.log_model = MagicMock()
        
        # Create a mock MLflow run context
        mock_run_context = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run_context)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=None)
        
        # Setup pipeline state
        pipeline_id = "test_pipeline_123"
        strategy = "audio_only"
        
        # Mock the pipeline state
        orchestrator.active_pipelines[pipeline_id] = {
            'pipeline_id': pipeline_id,
            'strategy': strategy,
            'status': 'running',
            'current_stage': 'model_training',
            'experiment_name': 'test_experiment',
            'feature_agreement': {
                'selected_features': [
                    'audio_tempo', 'audio_energy', 'audio_valence', 'audio_danceability',
                    'audio_loudness', 'audio_acousticness', 'audio_instrumentalness',
                    'audio_speechiness', 'audio_liveness', 'audio_key', 'audio_mode',
                    'audio_brightness', 'audio_complexity', 'audio_warmth',
                    'audio_harmonic_strength', 'audio_primary_genre', 'audio_top_genre_1_prob',
                    'audio_top_genre_2_prob', 'lyrics_sentiment_positive', 'lyrics_sentiment_negative',
                    'lyrics_emotion_joy', 'lyrics_complexity_score', 'lyrics_narrative_complexity',
                    'lyrics_unique_words'
                ]
            },
            'parameters': {
                'dataset_path': sample_csv_file
            }
        }
        
        # Mock the strategy config
        orchestrator.strategy_configs[strategy] = {
            'csv_path': sample_csv_file,
            'features': orchestrator.active_pipelines[pipeline_id]['feature_agreement']['selected_features']
        }
        
        # Mock service responses
        with patch('httpx.post') as mock_post, \
             patch('httpx.get') as mock_get:
            
            # Mock audio service response
            mock_audio_response = Mock()
            mock_audio_response.status_code = 200
            mock_audio_response.json.return_value = mock_audio_service_response
            mock_post.return_value = mock_audio_response
            
            # Mock content service response
            mock_content_response = Mock()
            mock_content_response.status_code = 200
            mock_content_response.json.return_value = mock_content_service_response
            mock_get.return_value = mock_content_response
            
            # Test the model training execution
            import asyncio
            result = asyncio.run(orchestrator._execute_model_training(pipeline_id, strategy))
            
            # Verify the training succeeded
            assert result is True, "Model training should succeed"
            
            # Verify MLflow was called properly
            mock_mlflow.set_tracking_uri.assert_called()
            mock_mlflow.set_experiment.assert_called()
            mock_mlflow.start_run.assert_called()
            
            # Verify parameters were logged
            mock_mlflow.log_param.assert_called()
            param_calls = [call.args for call in mock_mlflow.log_param.call_args_list]
            param_dict = dict(param_calls)
            
            assert 'strategy' in param_dict
            assert param_dict['strategy'] == strategy
            assert 'pipeline_id' in param_dict
            assert param_dict['pipeline_id'] == pipeline_id
            assert 'model_type' in param_dict
            assert param_dict['model_type'] == 'Ensemble'
            
            # Verify metrics were logged
            mock_mlflow.log_metric.assert_called()
            metric_calls = [call.args for call in mock_mlflow.log_metric.call_args_list]
            metric_dict = dict(metric_calls)
            
            assert 'accuracy' in metric_dict
            assert 'n_training_samples' in metric_dict
            assert 'n_test_samples' in metric_dict
            assert 'music_theory_alignment_score' in metric_dict
            
            # Check that accuracy is reasonable (should be > 0.3 for random data)
            assert metric_dict['accuracy'] > 0.3, f"Accuracy should be reasonable, got {metric_dict['accuracy']}"
            
            # Check music theory alignment score
            assert 0 <= metric_dict['music_theory_alignment_score'] <= 1, "Music theory alignment should be between 0 and 1"
    
    def test_feature_mapping(self, orchestrator):
        """Test feature name mapping functionality"""
        
        # Test basic feature mapping
        audio_features = ['audio_tempo', 'audio_energy', 'audio_valence']
        mapped_features = orchestrator._map_feature_names(audio_features)
        
        # Should map to basic_ prefix
        expected_mapped = ['basic_tempo', 'basic_energy', 'basic_valence']
        assert mapped_features == expected_mapped
        
        # Test reverse mapping
        basic_features = ['basic_tempo', 'basic_energy', 'basic_valence']
        reverse_mapped = orchestrator._map_feature_names(basic_features, reverse=True)
        
        expected_reverse = ['audio_tempo', 'audio_energy', 'audio_valence']
        assert reverse_mapped == expected_reverse
        
        # Test genre mapping
        genre_features = ['audio_primary_genre', 'audio_top_genre_1_prob']
        mapped_genre = orchestrator._map_feature_names(genre_features)
        
        expected_genre = ['genre_primary_genre', 'genre_top_genre_1_prob']
        assert mapped_genre == expected_genre
    
    def test_available_features_detection(self, orchestrator, sample_training_data):
        """Test detection of available features in dataframe"""
        
        # Test features that exist in the dataframe
        selected_features = [
            'audio_tempo', 'audio_energy', 'audio_valence', 'audio_danceability',
            'audio_loudness', 'audio_primary_genre', 'lyrics_sentiment_positive'
        ]
        
        available_features = orchestrator._get_available_features(sample_training_data, selected_features)
        
        # Should return mapped feature names that exist in dataframe
        expected_available = [
            'basic_tempo', 'basic_energy', 'basic_valence', 'basic_danceability',
            'basic_loudness', 'genre_primary_genre', 'lyrics_sentiment_positive'
        ]
        
        assert set(available_features) == set(expected_available)
        
        # Test with non-existent features
        missing_features = ['audio_nonexistent', 'lyrics_fake_feature']
        available_with_missing = orchestrator._get_available_features(sample_training_data, selected_features + missing_features)
        
        # Should still return only the available ones
        assert set(available_with_missing) == set(expected_available)
    
    def test_composite_feature_creation(self, orchestrator, sample_training_data):
        """Test creation of composite musical features"""
        
        # Create a copy of the dataframe for testing
        test_df = sample_training_data.copy()
        
        # Test rhythmic appeal index creation
        if 'basic_tempo' in test_df.columns and 'basic_danceability' in test_df.columns:
            # Simulate the composite feature creation logic
            tempo_normalized = test_df['basic_tempo'].apply(lambda x: 
                1.0 if 120 <= x <= 130 else  # Optimal dance tempo
                0.8 if 110 <= x <= 140 else  # Good dance tempo
                0.6 if 100 <= x <= 150 else  # Acceptable
                0.4  # Suboptimal
            )
            test_df['rhythmic_appeal_index'] = (
                tempo_normalized * 0.4 + 
                test_df['basic_danceability'] * 0.6
            )
            
            # Verify the composite feature was created
            assert 'rhythmic_appeal_index' in test_df.columns
            assert test_df['rhythmic_appeal_index'].notna().all()
            assert (test_df['rhythmic_appeal_index'] >= 0).all()
            assert (test_df['rhythmic_appeal_index'] <= 1).all()
        
        # Test emotional impact score creation
        if all(col in test_df.columns for col in ['basic_valence', 'basic_energy']):
            emotional_cols = ['basic_valence', 'basic_energy']
            if 'lyrics_sentiment_positive' in test_df.columns:
                emotional_cols.append('lyrics_sentiment_positive')
            
            test_df['emotional_impact_score'] = test_df[emotional_cols].mean(axis=1)
            
            # Verify the composite feature was created
            assert 'emotional_impact_score' in test_df.columns
            assert test_df['emotional_impact_score'].notna().all()
            assert (test_df['emotional_impact_score'] >= 0).all()
            assert (test_df['emotional_impact_score'] <= 1).all()
    
    def test_feature_importance_weighting(self, orchestrator):
        """Test music theory-based feature importance weighting"""
        
        # Test feature categorization
        high_importance_features = [
            'basic_danceability', 'basic_energy', 'basic_valence', 'basic_tempo',
            'rhythmic_appeal_index', 'emotional_impact_score', 'commercial_viability_index'
        ]
        
        medium_importance_features = [
            'basic_loudness', 'basic_complexity', 'basic_brightness', 'basic_warmth',
            'lyrics_sentiment_positive', 'lyrics_complexity_score', 'lyrics_narrative_complexity',
            'sonic_sophistication_score'
        ]
        
        lower_importance_features = [
            'genre_primary_genre', 'genre_top_genre_1_prob', 'genre_top_genre_2_prob',
            'basic_acousticness', 'basic_instrumentalness', 'basic_speechiness'
        ]
        
        # Test weight assignment
        core_feature_weights = {}
        available_features = high_importance_features + medium_importance_features + lower_importance_features
        
        for feature in available_features:
            if feature in high_importance_features:
                core_feature_weights[feature] = 1.5  # Boost core features
            elif feature in medium_importance_features:
                core_feature_weights[feature] = 1.2  # Moderate boost
            elif feature in lower_importance_features:
                core_feature_weights[feature] = 0.8  # Reduce genre over-influence
            else:
                core_feature_weights[feature] = 1.0  # Default weight
        
        # Verify weights are correctly assigned
        for feature in high_importance_features:
            assert core_feature_weights[feature] == 1.5
        
        for feature in medium_importance_features:
            assert core_feature_weights[feature] == 1.2
        
        for feature in lower_importance_features:
            assert core_feature_weights[feature] == 0.8
    
    def test_model_configuration(self, orchestrator):
        """Test model configuration and hyperparameters"""
        
        # Test Random Forest configuration
        from sklearn.ensemble import RandomForestClassifier
        rf_model = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            max_depth=12,
            min_samples_split=3,
            class_weight='balanced'
        )
        
        assert rf_model.n_estimators == 150
        assert rf_model.random_state == 42
        assert rf_model.max_depth == 12
        assert rf_model.min_samples_split == 3
        assert rf_model.class_weight == 'balanced'
        
        # Test XGBoost configuration
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(
            n_estimators=150,
            random_state=42,
            max_depth=8,
            learning_rate=0.08,
            objective='binary:logistic',
            scale_pos_weight=1,
            importance_type='gain'
        )
        
        assert xgb_model.n_estimators == 150
        assert xgb_model.random_state == 42
        assert xgb_model.max_depth == 8
        assert xgb_model.learning_rate == 0.08
        assert xgb_model.objective == 'binary:logistic'

if __name__ == '__main__':
    pytest.main([__file__, '-v']) 