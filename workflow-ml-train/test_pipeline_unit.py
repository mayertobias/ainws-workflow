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

class TestMLTrainingPipeline:
    """Test the ML training pipeline with mocked services"""
    
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

    @patch('app.pipeline.orchestrator.mlflow')
    def test_model_training_execution(self, mock_mlflow, orchestrator, sample_csv_file):
        """Test the complete model training execution without external service calls"""
        
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

if __name__ == '__main__':
    pytest.main([__file__, '-v']) 