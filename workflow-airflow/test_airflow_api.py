#!/usr/bin/env python3
"""
Test script to demonstrate Airflow API usage for triggering workflows.
This replaces the previous orchestrator API calls.
"""

import requests
import json
import base64
from datetime import datetime

# Airflow configuration
AIRFLOW_URL = "http://localhost:8080"
USERNAME = "admin"
PASSWORD = "admin"

# Create basic auth header
credentials = base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()
headers = {
    "Authorization": f"Basic {credentials}",
    "Content-Type": "application/json"
}

def trigger_comprehensive_analysis(song_data):
    """
    Trigger comprehensive analysis workflow (replaces /api/workflow/comprehensive)
    """
    dag_id = "comprehensive_analysis"
    url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns"
    
    payload = {
        "conf": song_data,
        "dag_run_id": f"manual__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Comprehensive analysis triggered successfully!")
            print(f"   DAG Run ID: {result['dag_run_id']}")
            print(f"   State: {result['state']}")
            return result
        else:
            print(f"❌ Failed to trigger comprehensive analysis: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error triggering comprehensive analysis: {e}")
        return None

def trigger_ml_training(training_config):
    """
    Trigger ML training workflow (replaces /api/ml/train)
    """
    dag_id = "ml_training_smart_ensemble"
    url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns"
    
    payload = {
        "conf": training_config,
        "dag_run_id": f"training__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ ML training triggered successfully!")
            print(f"   DAG Run ID: {result['dag_run_id']}")
            print(f"   State: {result['state']}")
            return result
        else:
            print(f"❌ Failed to trigger ML training: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error triggering ML training: {e}")
        return None

def get_dag_run_status(dag_id, dag_run_id):
    """
    Get the status of a DAG run (replaces /api/workflow/status/{task_id})
    """
    url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}"
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            result = response.json()
            print(f"📊 DAG Run Status:")
            print(f"   DAG ID: {result['dag_id']}")
            print(f"   Run ID: {result['dag_run_id']}")
            print(f"   State: {result['state']}")
            print(f"   Start Date: {result['start_date']}")
            print(f"   End Date: {result.get('end_date', 'Still running')}")
            return result
        else:
            print(f"❌ Failed to get DAG run status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting DAG run status: {e}")
        return None

def list_dag_runs(dag_id, limit=5):
    """
    List recent DAG runs
    """
    url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns?limit={limit}&order_by=-start_date"
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            result = response.json()
            print(f"📋 Recent DAG Runs for {dag_id}:")
            for run in result['dag_runs']:
                print(f"   • {run['dag_run_id']} - {run['state']} ({run['start_date']})")
            return result
        else:
            print(f"❌ Failed to list DAG runs: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error listing DAG runs: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Testing Airflow API Integration")
    print("=" * 50)
    
    # Test 1: Trigger comprehensive analysis
    print("\n1. Testing Comprehensive Analysis Trigger")
    song_data = {
        "song_id": "test_song_123",
        "title": "Test Song",
        "artist": "Test Artist",
        "audio_file": "test_audio.mp3",
        "lyrics_file": "test_lyrics.txt"
    }
    
    result = trigger_comprehensive_analysis(song_data)
    if result:
        dag_run_id = result['dag_run_id']
        print(f"\n   Checking status...")
        get_dag_run_status("comprehensive_analysis", dag_run_id)
    
    # Test 2: List recent runs
    print("\n2. Listing Recent Comprehensive Analysis Runs")
    list_dag_runs("comprehensive_analysis")
    
    # Test 3: Trigger ML training
    print("\n3. Testing ML Training Trigger")
    training_config = {
        "dataset_path": "/opt/airflow/shared-data/training_dataset.csv",
        "model_name": "test_ensemble_model",
        "validation_split": 0.2,
        "epochs": 10
    }
    
    result = trigger_ml_training(training_config)
    if result:
        dag_run_id = result['dag_run_id']
        print(f"\n   Checking status...")
        get_dag_run_status("ml_training_smart_ensemble", dag_run_id)
    
    print("\n✨ API testing completed!")
    print("\nNext steps:")
    print("• Update your gateway service to use these API calls")
    print("• Replace orchestrator endpoints with Airflow API calls")
    print("• Monitor workflows at: http://localhost:8080")
    print("• Monitor Celery workers at: http://localhost:5555") 