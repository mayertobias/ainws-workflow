#!/usr/bin/env python3
"""
Comprehensive API validation test for workflow-intelligence service
This will help us debug the 422 validation errors systematically
"""

import json
import subprocess
from datetime import datetime
from typing import Dict, Any

def test_minimal_request():
    """Test with absolute minimal valid request"""
    print("=== Testing Minimal Request ===")
    
    minimal_request = {
        "song_metadata": {
            "title": "Test Song",
            "artist": "Test Artist"
        },
        "audio_analysis": {
            "analysis_timestamp": datetime.utcnow().isoformat() + "Z"
        },
        "content_analysis": {
            "analysis_timestamp": datetime.utcnow().isoformat() + "Z"
        },
        "hit_prediction": {
            "prediction_timestamp": datetime.utcnow().isoformat() + "Z"
        }
    }
    
    return test_request(minimal_request, "minimal")

def test_comprehensive_request():
    """Test with all required fields properly filled"""
    print("=== Testing Comprehensive Request ===")
    
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    comprehensive_request = {
        "song_metadata": {
            "title": "Test Song",
            "artist": "Test Artist",
            "genre": "Pop",
            "duration_seconds": 180.0
        },
        "audio_analysis": {
            "tempo": 120.0,
            "energy": 0.7,
            "danceability": 0.8,
            "valence": 0.6,
            "acousticness": 0.3,
            "loudness": -8.5,
            "instrumentalness": 0.1,
            "liveness": 0.2,
            "speechiness": 0.05,
            "spectral_features": {},
            "mfcc_features": [],
            "chroma_features": [],
            "rhythm_features": {},
            "harmonic_features": {},
            "genre_predictions": {"pop": 0.8},
            "mood_predictions": {"happy": 0.7},
            "audio_quality_score": 0.9,
            "analysis_timestamp": timestamp,
            "processing_time_ms": 1500.0
        },
        "content_analysis": {
            "raw_lyrics": "Test lyrics for validation",
            "processed_lyrics": "Test lyrics for validation",
            "sentiment_score": 0.2,
            "emotion_scores": {"happy": 0.7},
            "mood_classification": "upbeat",
            "language": "en",
            "complexity_score": 0.5,
            "readability_score": 0.8,
            "word_count": 25,
            "unique_words": 20,
            "topics": ["love", "life"],
            "themes": ["positivity"],
            "keywords": ["test", "song"],
            "explicit_content": False,
            "content_warnings": [],
            "target_audience": "general",
            "analysis_timestamp": timestamp,
            "processing_time_ms": 500.0
        },
        "hit_prediction": {
            "hit_probability": 0.75,
            "confidence_score": 0.85,
            "genre_specific_score": {"pop": 0.8},
            "market_predictions": {"us": 0.7},
            "demographic_scores": {"18-25": 0.8},
            "feature_importance": {"tempo": 0.3, "energy": 0.4},
            "top_contributing_features": ["tempo", "energy"],
            "similar_hits": [],
            "genre_benchmarks": {"pop": 0.6},
            "commercial_risk_factors": [],
            "success_factors": ["strong melody"],
            "model_version": "v1.0",
            "training_data_size": 10000,
            "model_accuracy": 0.85,
            "prediction_timestamp": timestamp,
            "processing_time_ms": 800.0
        },
        "analysis_depth": "comprehensive",
        "focus_areas": ["commercial", "musical"],
        "target_audience": "general",
        "business_context": "independent artist",
        "request_id": "test_request_001",
        "user_id": "test_user",
        "timestamp": timestamp
    }
    
    return test_request(comprehensive_request, "comprehensive")

def test_frontend_request():
    """Test with the exact format our frontend sends"""
    print("=== Testing Frontend Request Format (OLD - Should Fail) ===")
    
    frontend_request = {
        "song_metadata": {
            "title": "Untitled",
            "artist": "Unknown Artist",
            "genre": "audio_ensemble_experiments_20250707_audio_only_ensemble_pipeline_ce3c4568",
            "duration_seconds": None
        },
        "audio_analysis": {
            "tempo": 31.0,
            "energy": 5.0,
            "danceability": 5.0,
            "valence": 6.0,
            "acousticness": 6.0,
            "loudness": 6.0,
            "instrumentalness": 5.0,
            "liveness": 5.0,
            "speechiness": 5.0,
            "spectral_features": {},
            "mfcc_features": [],
            "chroma_features": [],
            "rhythm_features": {},
            "harmonic_features": {},
            "genre_predictions": {},
            "mood_predictions": {},
            "audio_quality_score": None,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "processing_time_ms": 100
        },
        "content_analysis": {
            "raw_lyrics": None,
            "processed_lyrics": None,
            "sentiment_score": None,
            "emotion_scores": {},
            "mood_classification": None,
            "language": "en",
            "complexity_score": None,
            "readability_score": None,
            "word_count": 0,
            "unique_words": None,
            "topics": [],
            "themes": [],
            "keywords": [],
            "explicit_content": None,
            "content_warnings": [],
            "target_audience": None,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "processing_time_ms": 50
        },
        "hit_prediction": {
            "hit_probability": 0.31,
            "confidence_score": 0.8,
            "genre_specific_score": {},
            "market_predictions": {},
            "demographic_scores": {},
            "feature_importance": {},
            "top_contributing_features": [],
            "similar_hits": [],
            "genre_benchmarks": {},
            "commercial_risk_factors": [],
            "success_factors": [],
            "model_version": "audio_ensemble_experiments_20250707_audio_only_ensemble_pipeline_ce3c4568",
            "training_data_size": None,
            "model_accuracy": None,
            "prediction_timestamp": datetime.utcnow().isoformat(),
            "processing_time_ms": 200
        },
        "analysis_depth": "comprehensive",
        "focus_areas": [],
        "target_audience": None,
        "business_context": None,
        "request_id": f"req_{int(datetime.now().timestamp() * 1000)}",
        "user_id": None,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return test_request(frontend_request, "frontend")

def test_frontend_request_normalized():
    """Test with normalized frontend format (should pass)"""
    print("=== Testing Frontend Request Format (NORMALIZED - Should Pass) ===")
    
    # Helper function to normalize features
    def normalize_feature(value, max_value=10):
        if value is None or value == 0:
            return 0
        num_value = float(value)
        if 0 <= num_value <= 1:
            return num_value
        return min(max(num_value / max_value, 0), 1)
    
    frontend_request_normalized = {
        "song_metadata": {
            "title": "Untitled",
            "artist": "Unknown Artist",
            "genre": "884",  # String conversion of numeric genre
            "duration_seconds": None
        },
        "audio_analysis": {
            "tempo": 31.0,
            "energy": normalize_feature(5.0),  # 0.5
            "danceability": normalize_feature(5.0),  # 0.5
            "valence": normalize_feature(6.0),  # 0.6
            "acousticness": normalize_feature(6.0),  # 0.6
            "loudness": 6.0,  # Loudness can be > 1 (dB)
            "instrumentalness": normalize_feature(5.0),  # 0.5
            "liveness": normalize_feature(5.0),  # 0.5
            "speechiness": normalize_feature(5.0),  # 0.5
            "spectral_features": {},
            "mfcc_features": [],
            "chroma_features": [],
            "rhythm_features": {},
            "harmonic_features": {},
            "genre_predictions": {},
            "mood_predictions": {},
            "audio_quality_score": None,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "processing_time_ms": 100
        },
        "content_analysis": {
            "raw_lyrics": None,
            "processed_lyrics": None,
            "sentiment_score": None,
            "emotion_scores": {},
            "mood_classification": None,
            "language": "en",
            "complexity_score": None,
            "readability_score": None,
            "word_count": 0,
            "unique_words": None,
            "topics": [],
            "themes": [],
            "keywords": [],
            "explicit_content": None,
            "content_warnings": [],
            "target_audience": None,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "processing_time_ms": 50
        },
        "hit_prediction": {
            "hit_probability": 0.31,
            "confidence_score": 0.8,
            "genre_specific_score": {},
            "market_predictions": {},
            "demographic_scores": {},
            "feature_importance": {},
            "top_contributing_features": [],
            "similar_hits": [],
            "genre_benchmarks": {},
            "commercial_risk_factors": [],
            "success_factors": [],
            "model_version": "audio_ensemble_experiments_20250707_audio_only_ensemble_pipeline_ce3c4568",
            "training_data_size": None,
            "model_accuracy": None,
            "prediction_timestamp": datetime.utcnow().isoformat(),
            "processing_time_ms": 200
        },
        "analysis_depth": "comprehensive",
        "focus_areas": [],
        "target_audience": None,
        "business_context": None,
        "request_id": f"req_{int(datetime.now().timestamp() * 1000)}",
        "user_id": None,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return test_request(frontend_request_normalized, "frontend-normalized")

def test_actual_ui_request():
    """Test with the exact request format from the actual UI error"""
    print("=== Testing Actual UI Request Format ===")
    
    actual_ui_request = {
        "song_metadata": {
            "title": "94. Addicted",
            "artist": "Unknown Artist", 
            "genre": "884",  # Fixed: Convert number to string
            "duration_seconds": 303.129
        },
        "audio_analysis": {
            "tempo": 156.2366943359375,
            "energy": 0.23049171231687066,  # Already normalized
            "danceability": 0.48945069238543515,  # Already normalized
            "valence": 0.76717473,  # Already normalized
            "acousticness": 0.8198251193743897,  # Already normalized
            "loudness": -7.345268249511719,
            "instrumentalness": None,  # Keep as None if null
            "liveness": 0.01800355295401311,  # Already normalized
            "speechiness": 0.021145730960231424,  # Already normalized
            "spectral_features": {},
            "mfcc_features": [],
            "chroma_features": [],
            "rhythm_features": {},
            "harmonic_features": {},
            "genre_predictions": {"884": 1},  # String key
            "mood_predictions": {"happy": 1, "sad": 1, "energetic": 1},
            "audio_quality_score": None,
            "analysis_timestamp": "2025-07-07T13:17:18.082Z",
            "processing_time_ms": 100
        },
        "content_analysis": {
            "raw_lyrics": None,
            "processed_lyrics": None,
            "sentiment_score": None,
            "emotion_scores": {},
            "mood_classification": None,
            "language": "en",
            "complexity_score": None,
            "readability_score": None,
            "word_count": 0,
            "unique_words": None,
            "topics": [],
            "themes": [],
            "keywords": [],
            "explicit_content": None,
            "content_warnings": [],
            "target_audience": None,
            "analysis_timestamp": "2025-07-07T13:17:18.082Z",
            "processing_time_ms": 50
        },
        "hit_prediction": {
            "hit_probability": 0.3096023428428997,
            "confidence_score": 0.8,
            "genre_specific_score": {},
            "market_predictions": {},
            "demographic_scores": {},
            "feature_importance": {},
            "top_contributing_features": [],
            "similar_hits": [],
            "genre_benchmarks": {},
            "commercial_risk_factors": [],
            "success_factors": [],
            "model_version": "audio_ensemble_experiments_20250707_audio_only_ensemble_pipeline_ce3c4568",
            "training_data_size": None,
            "model_accuracy": None,
            "prediction_timestamp": "2025-07-07T13:17:18.082Z",
            "processing_time_ms": 200
        },
        "analysis_depth": "comprehensive",
        "focus_areas": [],
        "target_audience": None,
        "business_context": None,
        "request_id": "req_1751894238082",
        "user_id": None,
        "timestamp": "2025-07-07T13:17:18.082Z"
    }
    
    return test_request(actual_ui_request, "actual-ui")

def test_request(request_data: Dict[str, Any], test_name: str) -> bool:
    """Send a test request and analyze the response"""
    url = "http://localhost:8007/workflow/analyze/comprehensive"
    
    try:
        print(f"Sending {test_name} request...")
        
        # Use curl via subprocess
        json_data = json.dumps(request_data)
        
        result = subprocess.run([
            'curl', '-s', '-X', 'POST', url,
            '-H', 'Content-Type: application/json',
            '-d', json_data,
            '-w', '%{http_code}'
        ], capture_output=True, text=True, timeout=30)
        
        # Extract status code (last 3 characters)
        response_body = result.stdout[:-3]
        status_code = result.stdout[-3:]
        
        print(f"Status Code: {status_code}")
        
        if status_code == "200":
            print("✅ SUCCESS!")
            try:
                result_json = json.loads(response_body)
                print(f"Response has {len(result_json.get('insights', []))} insights")
                print(f"Executive summary: {result_json.get('executive_summary', 'Missing')[:100]}...")
            except:
                print("Response received but couldn't parse JSON")
            return True
        else:
            print("❌ FAILED!")
            try:
                error_detail = json.loads(response_body)
                print("Error details:")
                print(json.dumps(error_detail, indent=2))
            except:
                print(f"Raw error response: {response_body}")
            return False
            
    except Exception as e:
        print(f"❌ REQUEST FAILED: {e}")
        return False

def test_service_health():
    """Test if the service is running and healthy"""
    print("=== Testing Service Health ===")
    try:
        result = subprocess.run([
            'curl', '-s', 'http://localhost:8007/workflow/health',
            '-w', '%{http_code}'
        ], capture_output=True, text=True, timeout=5)
        
        response_body = result.stdout[:-3]
        status_code = result.stdout[-3:]
        
        if status_code == "200":
            health_data = json.loads(response_body)
            print("✅ Service is healthy!")
            print(f"Service: {health_data.get('service')}")
            print(f"Version: {health_data.get('version')}")
            return True
        else:
            print(f"❌ Service unhealthy: {status_code}")
            return False
    except Exception as e:
        print(f"❌ Service not responding: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🔍 WORKFLOW-INTELLIGENCE API VALIDATION TESTS")
    print("=" * 60)
    
    # Test service health first
    if not test_service_health():
        print("Service is not running properly. Please check Docker containers.")
        return
    
    print()
    
    # Run validation tests
    tests = [
        test_minimal_request,
        test_comprehensive_request,
        test_frontend_request,
        test_frontend_request_normalized,
        test_actual_ui_request
    ]
    
    results = []
    for test_func in tests:
        success = test_func()
        results.append(success)
        print()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY:")
    test_names = ["Minimal", "Comprehensive", "Frontend (Old)", "Frontend (Normalized)", "Actual UI"]
    for i, (name, success) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name}: {status}")
    
    total_passed = sum(results)
    print(f"\nOverall: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("🚨 Some tests failed. Check the error details above.")

if __name__ == "__main__":
    main()