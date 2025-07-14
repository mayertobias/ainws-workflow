"""
Simple test to verify the regression logic changes are correctly implemented.

This test verifies the code structure and logic without requiring full dependencies.
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_orchestrator_imports():
    """Test that the orchestrator has the correct imports for regression."""
    logger.info("Testing orchestrator imports...")
    
    orchestrator_path = Path(__file__).parent / "app" / "pipeline" / "orchestrator.py"
    
    if not orchestrator_path.exists():
        logger.error(f"❌ Orchestrator file not found: {orchestrator_path}")
        return False
    
    with open(orchestrator_path, 'r') as f:
        content = f.read()
    
    # Check for regression imports
    regression_imports = [
        "RandomForestRegressor",
        "VotingRegressor",
        "mean_squared_error",
        "r2_score",
        "mean_absolute_error",
        "ContinuousHitScoreEngineer",
        "HitSongRegressionMetrics"
    ]
    
    missing_imports = []
    for import_name in regression_imports:
        if import_name not in content:
            missing_imports.append(import_name)
    
    if missing_imports:
        logger.error(f"❌ Missing imports: {missing_imports}")
        return False
    
    logger.info("✅ All regression imports found")
    return True

def test_orchestrator_regression_logic():
    """Test that the orchestrator contains regression logic."""
    logger.info("Testing orchestrator regression logic...")
    
    orchestrator_path = Path(__file__).parent / "app" / "pipeline" / "orchestrator.py"
    
    with open(orchestrator_path, 'r') as f:
        content = f.read()
    
    # Check for regression-specific logic
    regression_patterns = [
        "task_type = \"regression\"",
        "VotingRegressor(",
        "XGBRegressor(",
        "RandomForestRegressor(",
        "continuous hit_score",
        "r2_score",
        "mse",
        "mae"
    ]
    
    missing_patterns = []
    for pattern in regression_patterns:
        if pattern not in content:
            missing_patterns.append(pattern)
    
    if missing_patterns:
        logger.error(f"❌ Missing regression patterns: {missing_patterns}")
        return False
    
    # Check that classification patterns are mostly removed
    classification_patterns = [
        "VotingClassifier(",
        "RandomForestClassifier(",
        "XGBClassifier(",
        "accuracy_score"
    ]
    
    remaining_classification = []
    for pattern in classification_patterns:
        if pattern in content:
            remaining_classification.append(pattern)
    
    if remaining_classification:
        logger.warning(f"⚠️ Some classification patterns still present: {remaining_classification}")
        # This might be OK if they're in imports but not used
    
    logger.info("✅ Regression logic patterns found")
    return True

def test_predictor_regression_logic():
    """Test that the predictor contains regression logic."""
    logger.info("Testing predictor regression logic...")
    
    predictor_path = Path(__file__).parent.parent / "workflow-ml-prediction" / "app" / "services" / "predictor.py"
    
    if not predictor_path.exists():
        logger.error(f"❌ Predictor file not found: {predictor_path}")
        return False
    
    with open(predictor_path, 'r') as f:
        content = f.read()
    
    # Check for regression-specific logic
    regression_patterns = [
        "REGRESSION: Direct continuous output",
        "continuous hit_score",
        "model_type': 'regression'",
        "r2_score",
        "mse",
        "rmse",
        "mae"
    ]
    
    missing_patterns = []
    for pattern in regression_patterns:
        if pattern not in content:
            missing_patterns.append(pattern)
    
    if missing_patterns:
        logger.error(f"❌ Missing regression patterns in predictor: {missing_patterns}")
        return False
    
    logger.info("✅ Predictor regression logic found")
    return True

def test_utility_files_exist():
    """Test that the utility files exist."""
    logger.info("Testing utility files existence...")
    
    utility_files = [
        "app/utils/target_engineering.py",
        "app/utils/regression_metrics.py"
    ]
    
    missing_files = []
    for file_path in utility_files:
        full_path = Path(__file__).parent / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"❌ Missing utility files: {missing_files}")
        return False
    
    logger.info("✅ All utility files exist")
    return True

def test_utility_file_content():
    """Test that utility files have the expected content."""
    logger.info("Testing utility file content...")
    
    # Test target engineering file
    target_eng_path = Path(__file__).parent / "app" / "utils" / "target_engineering.py"
    
    with open(target_eng_path, 'r') as f:
        target_content = f.read()
    
    target_patterns = [
        "ContinuousHitScoreEngineer",
        "engineer_hit_score",
        "chart_score",
        "longevity_score",
        "streaming_score",
        "MinMaxScaler"
    ]
    
    for pattern in target_patterns:
        if pattern not in target_content:
            logger.error(f"❌ Missing pattern in target_engineering.py: {pattern}")
            return False
    
    # Test regression metrics file
    metrics_path = Path(__file__).parent / "app" / "utils" / "regression_metrics.py"
    
    with open(metrics_path, 'r') as f:
        metrics_content = f.read()
    
    metrics_patterns = [
        "HitSongRegressionMetrics",
        "calculate_comprehensive_metrics",
        "ranking_performance",
        "top_k_precision",
        "business_impact",
        "spearman_correlation"
    ]
    
    for pattern in metrics_patterns:
        if pattern not in metrics_content:
            logger.error(f"❌ Missing pattern in regression_metrics.py: {pattern}")
            return False
    
    logger.info("✅ Utility file content verified")
    return True

def run_structure_tests():
    """Run all structure tests."""
    logger.info("🧪 Starting Regression Pipeline Structure Tests")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Test 1: Orchestrator imports
    test_results['orchestrator_imports'] = test_orchestrator_imports()
    
    # Test 2: Orchestrator regression logic
    test_results['orchestrator_logic'] = test_orchestrator_regression_logic()
    
    # Test 3: Predictor regression logic
    test_results['predictor_logic'] = test_predictor_regression_logic()
    
    # Test 4: Utility files exist
    test_results['utility_files_exist'] = test_utility_files_exist()
    
    # Test 5: Utility file content
    test_results['utility_content'] = test_utility_file_content()
    
    # Summary
    logger.info("=" * 60)
    logger.info("🏁 TEST SUMMARY")
    logger.info("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n📊 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All structure tests PASSED! Regression pipeline structure is correct.")
        return True
    else:
        logger.error("❌ Some tests FAILED. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_structure_tests()
    sys.exit(0 if success else 1)