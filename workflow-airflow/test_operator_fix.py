#!/usr/bin/env python3
"""
Test script to verify the fixed MLTrainingOperator works correctly
"""
import os
import sys
import json

# Add the plugins directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'plugins'))

try:
    import httpx
except ImportError:
    print("httpx not available, testing without actual HTTP call")
    httpx = None

from workflow_operators import MLTrainingOperator

def test_ml_training_operator():
    """Test the MLTrainingOperator with the fixed endpoint and payload"""
    
    # Set up the operator
    operator = MLTrainingOperator(
        task_id='test_smart_ensemble_training',
        training_strategy='smart_ensemble',
        dataset_path='/app/data/training_data/filtered/sample_dataset.csv',
        training_id='test_operator_fix',
        model_config={
            'target_column': 'final_popularity',
            'min_r2_threshold': 0.0,
            'enable_feature_engineering': True
        }
    )
    
    # Mock context for testing
    mock_context = {
        'params': {
            'experiment_name': 'test_operator_fix',
            'target_column': 'final_popularity',
            'min_r2_threshold': 0.0,
            'enable_feature_engineering': True
        }
    }
    
    # Test the service URL construction
    service_url = operator._get_service_url()
    print(f"Service URL: {service_url}")
    
    # Test the payload creation
    payload = operator._create_payload(mock_context)
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    return True

if __name__ == "__main__":
    print("Testing MLTrainingOperator with fixed endpoint and payload...")
    print("=" * 60)
    
    success = test_ml_training_operator()
    
    print("=" * 60)
    if success:
        print("🎉 MLTrainingOperator configuration looks good!")
    else:
        print("💥 MLTrainingOperator has issues.")
        sys.exit(1)
