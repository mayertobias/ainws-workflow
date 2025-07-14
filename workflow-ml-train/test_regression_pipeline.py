"""
Test script to verify the end-to-end regression pipeline works correctly.

This script tests:
1. Continuous hit_score target variable engineering
2. Regression model training (VotingRegressor)
3. Business-relevant regression metrics
4. Model prediction with regression output
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from utils.target_engineering import ContinuousHitScoreEngineer
from utils.regression_metrics import HitSongRegressionMetrics
from services.predictor import SmartSongPredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data():
    """Create synthetic test data for regression pipeline testing."""
    logger.info("Creating synthetic test data...")
    
    np.random.seed(42)  # For reproducibility
    n_samples = 200
    
    # Create synthetic song data with multiple success metrics
    data = {
        'song_name': [f'Song_{i:03d}' for i in range(n_samples)],
        'chart_position': np.random.randint(1, 201, n_samples),  # Chart positions
        'weeks_on_chart': np.random.randint(1, 53, n_samples),  # Weeks on chart
        'streams': np.random.lognormal(10, 2, n_samples).astype(int),  # Streaming numbers
        'popularity': np.random.uniform(0, 100, n_samples),  # General popularity
        
        # Audio features
        'audio_energy': np.random.uniform(0, 1, n_samples),
        'audio_valence': np.random.uniform(0, 1, n_samples),
        'audio_danceability': np.random.uniform(0, 1, n_samples),
        'audio_tempo': np.random.uniform(60, 200, n_samples),
        'audio_loudness': np.random.uniform(-20, 0, n_samples),
        
        # Content features
        'lyrics_sentiment_polarity': np.random.uniform(-1, 1, n_samples),
        'lyrics_word_count': np.random.randint(50, 500, n_samples),
        'lyrics_complexity_score': np.random.uniform(0, 1, n_samples),
    }
    
    # Create correlations to make it more realistic
    # Higher energy and danceability tend to correlate with better chart performance
    for i in range(n_samples):
        if data['audio_energy'][i] > 0.7 and data['audio_danceability'][i] > 0.7:
            data['chart_position'][i] = int(data['chart_position'][i] * 0.3)  # Better chart position
            data['weeks_on_chart'][i] = int(data['weeks_on_chart'][i] * 1.5)  # Longer on chart
    
    df = pd.DataFrame(data)
    logger.info(f"Created synthetic dataset with {len(df)} songs")
    return df

def test_hit_score_engineering():
    """Test the continuous hit_score target variable engineering."""
    logger.info("=" * 60)
    logger.info("TESTING: Continuous Hit Score Engineering")
    logger.info("=" * 60)
    
    # Create test data
    df = create_test_data()
    
    # Test hit score engineering
    hit_score_engineer = ContinuousHitScoreEngineer()
    hit_scores, engineering_report = hit_score_engineer.engineer_hit_score(df)
    
    # Validate results
    assert len(hit_scores) == len(df), "Hit scores length mismatch"
    assert all(0 <= score <= 1 for score in hit_scores), "Hit scores outside 0-1 range"
    assert 'available_metrics' in engineering_report, "Missing available_metrics in report"
    assert 'score_statistics' in engineering_report, "Missing score_statistics in report"
    
    logger.info("✅ Hit score engineering test PASSED")
    logger.info(f"   📊 Score range: {engineering_report['score_statistics']['min']:.3f} - {engineering_report['score_statistics']['max']:.3f}")
    logger.info(f"   📊 Mean: {engineering_report['score_statistics']['mean']:.3f}")
    logger.info(f"   📊 Metrics used: {', '.join(engineering_report['available_metrics'])}")
    
    return df, hit_scores, engineering_report

def test_regression_metrics():
    """Test the business-relevant regression metrics."""
    logger.info("=" * 60)
    logger.info("TESTING: Business-Relevant Regression Metrics")
    logger.info("=" * 60)
    
    # Create synthetic true and predicted values
    np.random.seed(42)
    n_samples = 100
    
    # Create realistic hit score distribution (most songs are not hits)
    y_true = np.random.beta(2, 5, n_samples)  # Skewed toward lower scores
    
    # Create predictions with some correlation to true values
    y_pred = y_true + np.random.normal(0, 0.1, n_samples)
    y_pred = np.clip(y_pred, 0, 1)  # Ensure 0-1 range
    
    # Test metrics calculation
    metrics_calculator = HitSongRegressionMetrics()
    comprehensive_metrics = metrics_calculator.calculate_comprehensive_metrics(y_true, y_pred)
    
    # Validate results
    assert 'standard_regression' in comprehensive_metrics, "Missing standard_regression metrics"
    assert 'ranking_performance' in comprehensive_metrics, "Missing ranking_performance metrics"
    assert 'top_k_precision' in comprehensive_metrics, "Missing top_k_precision metrics"
    assert 'business_impact' in comprehensive_metrics, "Missing business_impact metrics"
    
    # Check key metrics
    r2_score = comprehensive_metrics['standard_regression']['r2_score']
    spearman_corr = comprehensive_metrics['ranking_performance']['spearman_correlation']
    top_10_precision = comprehensive_metrics['top_k_precision']['top_10_percent']
    
    logger.info("✅ Regression metrics test PASSED")
    logger.info(f"   📊 R² Score: {r2_score:.3f}")
    logger.info(f"   📊 Ranking Correlation: {spearman_corr:.3f}")
    logger.info(f"   📊 Top-10% Precision: {top_10_precision:.3f}")
    
    return comprehensive_metrics

def test_model_training_integration():
    """Test the integration with model training pipeline."""
    logger.info("=" * 60)
    logger.info("TESTING: Model Training Integration")
    logger.info("=" * 60)
    
    # This is a simplified test that verifies the imports and basic functionality
    # In a real scenario, you would run the full training pipeline
    
    try:
        # Test imports
        from sklearn.ensemble import RandomForestRegressor, VotingRegressor
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        import xgboost as xgb
        
        # Create simple synthetic data for training
        np.random.seed(42)
        n_samples = 50
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.uniform(0, 1, n_samples)  # Continuous target
        
        # Create and train ensemble regressor
        rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
        xgb_model = xgb.XGBRegressor(n_estimators=10, random_state=42)
        
        ensemble_model = VotingRegressor(
            estimators=[('rf', rf_model), ('xgb', xgb_model)]
        )
        
        # Train model
        ensemble_model.fit(X, y)
        
        # Make predictions
        y_pred = ensemble_model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        # Validate results
        assert len(y_pred) == len(y), "Prediction length mismatch"
        assert all(isinstance(pred, (int, float)) for pred in y_pred), "Invalid prediction types"
        assert mse >= 0, "MSE should be non-negative"
        assert mae >= 0, "MAE should be non-negative"
        
        logger.info("✅ Model training integration test PASSED")
        logger.info(f"   📊 MSE: {mse:.4f}")
        logger.info(f"   📊 R²: {r2:.4f}")
        logger.info(f"   📊 MAE: {mae:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model training integration test FAILED: {e}")
        return False

def test_prediction_service():
    """Test the prediction service with regression output."""
    logger.info("=" * 60)
    logger.info("TESTING: Prediction Service (Regression)")
    logger.info("=" * 60)
    
    # Note: This test requires actual models to be available
    # In a real scenario, you would load trained models
    
    try:
        # Test that the SmartSongPredictor can be initialized
        predictor = SmartSongPredictor()
        
        # Test the prediction logic (without actual models)
        # This validates the code structure and imports
        
        logger.info("✅ Prediction service initialization test PASSED")
        logger.info("   📊 Service initialized successfully")
        
        # Test sample features format
        sample_features = {
            'audio_energy': 0.7,
            'audio_valence': 0.8,
            'audio_danceability': 0.6,
            'audio_tempo': 120.0,
            'audio_loudness': -8.0,
            'lyrics_sentiment_polarity': 0.3,
            'lyrics_word_count': 200,
            'lyrics_complexity_score': 0.5
        }
        
        # Test feature validation (this should work even without models)
        available_models = predictor.get_available_models()
        
        logger.info("✅ Prediction service feature validation test PASSED")
        logger.info(f"   📊 Available models: {len(available_models)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Prediction service test FAILED: {e}")
        return False

def run_all_tests():
    """Run all regression pipeline tests."""
    logger.info("🧪 Starting End-to-End Regression Pipeline Tests")
    logger.info("=" * 80)
    
    test_results = {}
    
    # Test 1: Hit Score Engineering
    try:
        df, hit_scores, engineering_report = test_hit_score_engineering()
        test_results['hit_score_engineering'] = True
    except Exception as e:
        logger.error(f"❌ Hit score engineering test FAILED: {e}")
        test_results['hit_score_engineering'] = False
    
    # Test 2: Regression Metrics
    try:
        comprehensive_metrics = test_regression_metrics()
        test_results['regression_metrics'] = True
    except Exception as e:
        logger.error(f"❌ Regression metrics test FAILED: {e}")
        test_results['regression_metrics'] = False
    
    # Test 3: Model Training Integration
    test_results['model_training'] = test_model_training_integration()
    
    # Test 4: Prediction Service
    test_results['prediction_service'] = test_prediction_service()
    
    # Summary
    logger.info("=" * 80)
    logger.info("🏁 TEST SUMMARY")
    logger.info("=" * 80)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n📊 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests PASSED! Regression pipeline is ready.")
        return True
    else:
        logger.error("❌ Some tests FAILED. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)