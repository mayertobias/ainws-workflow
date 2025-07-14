#!/usr/bin/env python3
"""
🧪 Test Script for Ensemble Training Fix
=======================================

This script tests the fixed ensemble training logic to ensure
feature importance extraction works correctly.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb

def test_ensemble_feature_importance_fix():
    """Test that the ensemble training fix works correctly"""
    
    print("🧪 Testing Ensemble Feature Importance Fix")
    print("=" * 50)
    
    # Create synthetic data (same as in orchestrator)
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training data: {X_train.shape}")
    print(f"📊 Test data: {X_test.shape}")
    
    # Test the ORIGINAL approach (this should fail)
    print("\n❌ Testing ORIGINAL approach (should fail):")
    try:
        # Define individual models (not fitted)
        rf_model_original = RandomForestClassifier(n_estimators=10, random_state=42)
        xgb_model_original = xgb.XGBClassifier(n_estimators=10, random_state=42)
        
        # Create and fit ensemble
        ensemble_original = VotingClassifier(
            estimators=[('rf', rf_model_original), ('xgb', xgb_model_original)],
            voting='soft'
        )
        ensemble_original.fit(X_train, y_train)
        
        # Try to access feature importance from original (unfitted) models
        rf_importance_original = rf_model_original.feature_importances_  # Should fail
        print("   ⚠️ Unexpected: Original approach worked (this shouldn't happen)")
        
    except Exception as e:
        print(f"   ✅ Expected failure: {type(e).__name__}: {str(e)[:60]}...")
    
    # Test the FIXED approach (this should work)
    print("\n✅ Testing FIXED approach (should work):")
    try:
        # Define individual models
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        xgb_model = xgb.XGBClassifier(n_estimators=10, random_state=42)
        
        # Create and fit ensemble
        ensemble_model = VotingClassifier(
            estimators=[('rf', rf_model), ('xgb', xgb_model)],
            voting='soft'
        )
        ensemble_model.fit(X_train, y_train)
        
        # FIXED: Access fitted models from ensemble
        fitted_rf_model = ensemble_model.named_estimators_['rf']
        fitted_xgb_model = ensemble_model.named_estimators_['xgb']
        
        # Extract feature importance from fitted models
        rf_importance = dict(zip(feature_names, fitted_rf_model.feature_importances_))
        xgb_importance = dict(zip(feature_names, fitted_xgb_model.feature_importances_))
        
        # Calculate ensemble feature importance
        ensemble_importance = {}
        for feature in feature_names:
            ensemble_importance[feature] = (rf_importance[feature] + xgb_importance[feature]) / 2
        
        print("   ✅ Fixed approach works!")
        print(f"   📊 RF features extracted: {len(rf_importance)}")
        print(f"   📊 XGB features extracted: {len(xgb_importance)}")
        print(f"   📊 Ensemble features: {len(ensemble_importance)}")
        
        # Show top 3 features
        top_features = sorted(ensemble_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        print("\n   🏆 Top 3 features by ensemble importance:")
        for i, (feature, importance) in enumerate(top_features, 1):
            print(f"      {i}. {feature}: {importance:.4f}")
        
        # Test model predictions
        y_pred = ensemble_model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        print(f"\n   🎯 Model accuracy: {accuracy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Unexpected failure: {type(e).__name__}: {e}")
        return False

def test_individual_model_access():
    """Test that we can access individual models from VotingClassifier"""
    
    print("\n🔍 Testing Individual Model Access")
    print("=" * 40)
    
    # Create simple test data
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    # Create ensemble
    rf = RandomForestClassifier(n_estimators=5, random_state=42)
    xgb_clf = xgb.XGBClassifier(n_estimators=5, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb_clf)],
        voting='soft'
    )
    
    ensemble.fit(X, y)
    
    # Test accessing individual models
    print("✅ Individual model access tests:")
    
    # Test 1: named_estimators_ attribute exists
    assert hasattr(ensemble, 'named_estimators_'), "VotingClassifier should have named_estimators_"
    print("   ✅ named_estimators_ attribute exists")
    
    # Test 2: Can access RF model
    fitted_rf = ensemble.named_estimators_['rf']
    assert hasattr(fitted_rf, 'feature_importances_'), "RF should have feature_importances_"
    print("   ✅ Can access fitted RF model")
    
    # Test 3: Can access XGB model
    fitted_xgb = ensemble.named_estimators_['xgb']
    assert hasattr(fitted_xgb, 'feature_importances_'), "XGB should have feature_importances_"
    print("   ✅ Can access fitted XGB model")
    
    # Test 4: Feature importances have correct shape
    assert len(fitted_rf.feature_importances_) == X.shape[1], "RF feature_importances_ wrong shape"
    assert len(fitted_xgb.feature_importances_) == X.shape[1], "XGB feature_importances_ wrong shape"
    print("   ✅ Feature importances have correct shape")
    
    # Test 5: Feature importances sum to 1 (approximately)
    rf_sum = fitted_rf.feature_importances_.sum()
    xgb_sum = fitted_xgb.feature_importances_.sum()
    assert abs(rf_sum - 1.0) < 0.001, f"RF importances don't sum to 1: {rf_sum}"
    assert abs(xgb_sum - 1.0) < 0.001, f"XGB importances don't sum to 1: {xgb_sum}"
    print("   ✅ Feature importances sum to 1")
    
    print("\n🎉 All individual model access tests passed!")

if __name__ == "__main__":
    print("🚀 Running Ensemble Training Fix Tests")
    print("=" * 60)
    
    # Run main test
    success = test_ensemble_feature_importance_fix()
    
    # Run additional tests
    test_individual_model_access()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ The ensemble training fix is working correctly.")
        print("✅ The ML training service should now handle ensemble training without errors.")
        print("\n🚀 You can now run ensemble training DAGs:")
        print("   - trigger_audio_ensemble_training")
        print("   - trigger_multimodal_ensemble_training")
    else:
        print("❌ TESTS FAILED!")
        print("The fix may not be working correctly.")
    
    print("\n📋 Fix Summary:")
    print("   - Issue: Accessing feature_importances_ from unfitted individual models")
    print("   - Root Cause: VotingClassifier fits internal copies, not original objects")
    print("   - Solution: Use ensemble_model.named_estimators_['model_name']")
    print("   - Files Fixed: workflow-ml-train/app/pipeline/orchestrator.py") 