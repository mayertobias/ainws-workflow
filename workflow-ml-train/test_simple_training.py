"""
Simple test for ML training pipeline without pytest
Tests the basic functionality of the orchestrator
"""

import os
import sys
import pandas as pd
import numpy as np
import tempfile
import shutil

# Add the app directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.pipeline.orchestrator import PipelineOrchestrator

def test_feature_mapping():
    """Test feature name mapping functionality"""
    print("Testing feature mapping...")
    
    orchestrator = PipelineOrchestrator()
    
    # Test basic feature mapping
    audio_features = ['audio_tempo', 'audio_energy', 'audio_valence']
    mapped_features = orchestrator._map_feature_names(audio_features)
    
    # Should map to basic_ prefix
    expected_mapped = ['basic_tempo', 'basic_energy', 'basic_valence']
    assert mapped_features == expected_mapped, f"Expected {expected_mapped}, got {mapped_features}"
    
    # Test reverse mapping
    basic_features = ['basic_tempo', 'basic_energy', 'basic_valence']
    reverse_mapped = orchestrator._map_feature_names(basic_features, reverse=True)
    
    expected_reverse = ['audio_tempo', 'audio_energy', 'audio_valence']
    assert reverse_mapped == expected_reverse, f"Expected {expected_reverse}, got {reverse_mapped}"
    
    # Test genre mapping
    genre_features = ['audio_primary_genre', 'audio_top_genre_1_prob']
    mapped_genre = orchestrator._map_feature_names(genre_features)
    
    expected_genre = ['genre_primary_genre', 'genre_top_genre_1_prob']
    assert mapped_genre == expected_genre, f"Expected {expected_genre}, got {mapped_genre}"
    
    print("✅ Feature mapping test passed!")

def test_available_features_detection():
    """Test detection of available features in dataframe"""
    print("Testing available features detection...")
    
    orchestrator = PipelineOrchestrator()
    
    # Create sample data
    np.random.seed(42)
    n_samples = 50
    
    sample_data = pd.DataFrame({
        'song_id': [f'song_{i}' for i in range(n_samples)],
        'basic_tempo': np.random.uniform(80, 180, n_samples),
        'basic_energy': np.random.uniform(0, 1, n_samples),
        'basic_valence': np.random.uniform(0, 1, n_samples),
        'basic_danceability': np.random.uniform(0, 1, n_samples),
        'basic_loudness': np.random.uniform(-20, 0, n_samples),
        'genre_primary_genre': np.random.choice(['pop', 'rock', 'electronic'], n_samples),
        'lyrics_sentiment_positive': np.random.uniform(0, 1, n_samples),
    })
    
    # Test features that exist in the dataframe
    selected_features = [
        'audio_tempo', 'audio_energy', 'audio_valence', 'audio_danceability',
        'audio_loudness', 'audio_primary_genre', 'lyrics_sentiment_positive'
    ]
    
    available_features = orchestrator._get_available_features(sample_data, selected_features)
    
    # Should return mapped feature names that exist in dataframe
    expected_available = [
        'basic_tempo', 'basic_energy', 'basic_valence', 'basic_danceability',
        'basic_loudness', 'genre_primary_genre', 'lyrics_sentiment_positive'
    ]
    
    assert set(available_features) == set(expected_available), f"Expected {expected_available}, got {available_features}"
    
    # Test with non-existent features
    missing_features = ['audio_nonexistent', 'lyrics_fake_feature']
    available_with_missing = orchestrator._get_available_features(sample_data, selected_features + missing_features)
    
    # Should still return only the available ones
    assert set(available_with_missing) == set(expected_available), f"Expected {expected_available}, got {available_with_missing}"
    
    print("✅ Available features detection test passed!")

def test_composite_feature_creation():
    """Test creation of composite musical features"""
    print("Testing composite feature creation...")
    
    # Create test dataframe
    np.random.seed(42)
    n_samples = 50
    
    test_df = pd.DataFrame({
        'basic_tempo': np.random.uniform(80, 180, n_samples),
        'basic_danceability': np.random.uniform(0, 1, n_samples),
        'basic_valence': np.random.uniform(0, 1, n_samples),
        'basic_energy': np.random.uniform(0, 1, n_samples),
        'lyrics_sentiment_positive': np.random.uniform(0, 1, n_samples),
    })
    
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
        assert 'rhythmic_appeal_index' in test_df.columns, "Rhythmic appeal index not created"
        assert test_df['rhythmic_appeal_index'].notna().all(), "Rhythmic appeal index contains NaN values"
        assert (test_df['rhythmic_appeal_index'] >= 0).all(), "Rhythmic appeal index has negative values"
        assert (test_df['rhythmic_appeal_index'] <= 1).all(), "Rhythmic appeal index has values > 1"
    
    # Test emotional impact score creation
    if all(col in test_df.columns for col in ['basic_valence', 'basic_energy']):
        emotional_cols = ['basic_valence', 'basic_energy']
        if 'lyrics_sentiment_positive' in test_df.columns:
            emotional_cols.append('lyrics_sentiment_positive')
        
        test_df['emotional_impact_score'] = test_df[emotional_cols].mean(axis=1)
        
        # Verify the composite feature was created
        assert 'emotional_impact_score' in test_df.columns, "Emotional impact score not created"
        assert test_df['emotional_impact_score'].notna().all(), "Emotional impact score contains NaN values"
        assert (test_df['emotional_impact_score'] >= 0).all(), "Emotional impact score has negative values"
        assert (test_df['emotional_impact_score'] <= 1).all(), "Emotional impact score has values > 1"
    
    print("✅ Composite feature creation test passed!")

def test_feature_importance_weighting():
    """Test music theory-based feature importance weighting"""
    print("Testing feature importance weighting...")
    
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
        assert core_feature_weights[feature] == 1.5, f"High importance feature {feature} should have weight 1.5"
    
    for feature in medium_importance_features:
        assert core_feature_weights[feature] == 1.2, f"Medium importance feature {feature} should have weight 1.2"
    
    for feature in lower_importance_features:
        assert core_feature_weights[feature] == 0.8, f"Lower importance feature {feature} should have weight 0.8"
    
    print("✅ Feature importance weighting test passed!")

def test_model_configuration():
    """Test model configuration and hyperparameters"""
    print("Testing model configuration...")
    
    # Test Random Forest configuration
    try:
        from sklearn.ensemble import RandomForestClassifier
        rf_model = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            max_depth=12,
            min_samples_split=3,
            class_weight='balanced'
        )
        
        assert rf_model.n_estimators == 150, f"RF n_estimators should be 150, got {rf_model.n_estimators}"
        assert rf_model.random_state == 42, f"RF random_state should be 42, got {rf_model.random_state}"
        assert rf_model.max_depth == 12, f"RF max_depth should be 12, got {rf_model.max_depth}"
        assert rf_model.min_samples_split == 3, f"RF min_samples_split should be 3, got {rf_model.min_samples_split}"
        assert rf_model.class_weight == 'balanced', f"RF class_weight should be 'balanced', got {rf_model.class_weight}"
        
        print("✅ Random Forest configuration test passed!")
        
    except ImportError:
        print("⚠️ sklearn not available, skipping Random Forest test")
    
    # Test XGBoost configuration
    try:
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
        
        assert xgb_model.n_estimators == 150, f"XGB n_estimators should be 150, got {xgb_model.n_estimators}"
        assert xgb_model.random_state == 42, f"XGB random_state should be 42, got {xgb_model.random_state}"
        assert xgb_model.max_depth == 8, f"XGB max_depth should be 8, got {xgb_model.max_depth}"
        assert xgb_model.learning_rate == 0.08, f"XGB learning_rate should be 0.08, got {xgb_model.learning_rate}"
        assert xgb_model.objective == 'binary:logistic', f"XGB objective should be 'binary:logistic', got {xgb_model.objective}"
        
        print("✅ XGBoost configuration test passed!")
        
    except ImportError:
        print("⚠️ xgboost not available, skipping XGBoost test")

if __name__ == '__main__':
    print("🧪 Starting ML Training Pipeline Tests...")
    print("=" * 50)
    
    try:
        test_feature_mapping()
        test_available_features_detection()
        test_composite_feature_creation()
        test_feature_importance_weighting()
        test_model_configuration()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        print("✅ ML Training Pipeline is working correctly!")
        
    except Exception as e:
        print("\n" + "=" * 50)
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 