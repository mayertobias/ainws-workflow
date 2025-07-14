#!/usr/bin/env python3
"""
Integration Test for Workflow Intelligence Service

This script tests the critical workflow integration functionality
to ensure the intelligence service can receive and process data
from other microservices.
"""

import asyncio
import httpx
import json
from datetime import datetime
from typing import Dict, Any

# Test data that simulates real service responses
SAMPLE_AUDIO_ANALYSIS = {
    "tempo": 128.5,
    "energy": 0.85,
    "danceability": 0.73,
    "valence": 0.64,
    "acousticness": 0.12,
    "loudness": -5.2,
    "instrumentalness": 0.001,
    "liveness": 0.15,
    "speechiness": 0.04,
    "spectral_features": {
        "spectral_centroid": 2156.3,
        "spectral_rolloff": 4230.1,
        "zero_crossing_rate": 0.089
    },
    "mfcc_features": [23.1, -12.4, 8.7, -3.2, 5.1],
    "genre_predictions": {
        "pop": 0.72,
        "rock": 0.18,
        "electronic": 0.10
    },
    "mood_predictions": {
        "happy": 0.65,
        "energetic": 0.78,
        "relaxed": 0.25
    },
    "audio_quality_score": 0.88,
    "analysis_timestamp": datetime.utcnow().isoformat(),
    "processing_time_ms": 2340.5
}

SAMPLE_CONTENT_ANALYSIS = {
    "raw_lyrics": """I've been walking down this road so long
Looking for the light to guide me home
Every step I take, I'm getting strong
Never gonna give up, never gonna roam""",
    "processed_lyrics": "walking road long looking light guide home step take getting strong never give up never roam",
    "sentiment_score": 0.72,
    "emotion_scores": {
        "joy": 0.45,
        "hope": 0.78,
        "determination": 0.85,
        "sadness": 0.12
    },
    "mood_classification": "uplifting",
    "language": "en",
    "complexity_score": 0.65,
    "readability_score": 85.2,
    "word_count": 32,
    "unique_words": 26,
    "topics": ["journey", "perseverance", "hope"],
    "themes": ["personal growth", "determination"],
    "keywords": ["walking", "road", "light", "home", "strong"],
    "explicit_content": False,
    "content_warnings": [],
    "target_audience": "general",
    "analysis_timestamp": datetime.utcnow().isoformat(),
    "processing_time_ms": 1250.3
}

SAMPLE_HIT_PREDICTION = {
    "hit_probability": 0.78,
    "confidence_score": 0.82,
    "genre_specific_score": {
        "pop": 0.81,
        "rock": 0.72,
        "alternative": 0.68
    },
    "market_predictions": {
        "us_market": 0.75,
        "uk_market": 0.73,
        "global_market": 0.77
    },
    "demographic_scores": {
        "18-25": 0.82,
        "25-35": 0.75,
        "35-50": 0.65
    },
    "feature_importance": {
        "energy": 0.18,
        "danceability": 0.15,
        "sentiment": 0.12,
        "tempo": 0.11
    },
    "top_contributing_features": ["energy", "danceability", "sentiment_score", "tempo"],
    "similar_hits": [
        {"title": "Similar Hit 1", "artist": "Artist A", "similarity": 0.85},
        {"title": "Similar Hit 2", "artist": "Artist B", "similarity": 0.79}
    ],
    "commercial_risk_factors": ["market_saturation"],
    "success_factors": ["strong_energy", "positive_sentiment", "catchy_tempo"],
    "model_version": "hit_predictor_v2.1",
    "model_accuracy": 0.87,
    "prediction_timestamp": datetime.utcnow().isoformat(),
    "processing_time_ms": 890.2
}

SAMPLE_SONG_METADATA = {
    "title": "Walking Home",
    "artist": "Test Artist",
    "album": "Test Album",
    "genre": "Pop",
    "duration_seconds": 210.5,
    "language": "en",
    "file_path": "/test/walking_home.mp3"
}

async def test_health_check():
    """Test the health check endpoint."""
    print("🏥 Testing health check endpoint...")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8005/workflow/health")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Health check passed: {result['status']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

async def test_comprehensive_analysis():
    """Test the comprehensive analysis endpoint."""
    print("🧠 Testing comprehensive analysis endpoint...")
    
    request_data = {
        "audio_analysis": SAMPLE_AUDIO_ANALYSIS,
        "content_analysis": SAMPLE_CONTENT_ANALYSIS,
        "hit_prediction": SAMPLE_HIT_PREDICTION,
        "song_metadata": SAMPLE_SONG_METADATA,
        "analysis_config": {
            "depth": "comprehensive",
            "focus_areas": ["musical_analysis", "commercial_potential"],
            "business_context": "test_integration"
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8005/workflow/analyze/comprehensive",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Comprehensive analysis successful!")
                print(f"📊 Overall Score: {result.get('overall_score', 'N/A'):.2f}")
                print(f"🎯 Confidence Level: {result.get('confidence_level', 'N/A'):.2f}")
                print(f"📝 Executive Summary: {result.get('executive_summary', 'N/A')[:100]}...")
                print(f"💡 Insights Count: {len(result.get('insights', []))}")
                print(f"⏱️  Processing Time: {result.get('processing_time_ms', 'N/A'):.0f}ms")
                
                # Show sample insights
                insights = result.get('insights', [])
                if insights:
                    print("\n🔍 Sample Insights:")
                    for i, insight in enumerate(insights[:3]):
                        print(f"  {i+1}. {insight.get('category', 'Unknown')}: {insight.get('title', 'No title')}")
                        print(f"     {insight.get('description', 'No description')[:80]}...")
                
                return True
            else:
                print(f"❌ Comprehensive analysis failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Comprehensive analysis error: {e}")
        return False

async def test_orchestrator_endpoint():
    """Test the orchestrator integration endpoint."""
    print("🎭 Testing orchestrator integration endpoint...")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8005/workflow/analyze/from-orchestrator",
                json={
                    "audio_analysis": SAMPLE_AUDIO_ANALYSIS,
                    "content_analysis": SAMPLE_CONTENT_ANALYSIS,
                    "hit_prediction": SAMPLE_HIT_PREDICTION,
                    "song_metadata": SAMPLE_SONG_METADATA,
                    "analysis_config": {
                        "depth": "comprehensive",
                        "business_context": "orchestrator_test"
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Orchestrator integration successful!")
                print(f"📊 Request ID: {result.get('request_id', 'N/A')}")
                print(f"🎯 Analysis Type: {result.get('analysis_type', 'N/A')}")
                return True
            else:
                print(f"❌ Orchestrator integration failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Orchestrator integration error: {e}")
        return False

async def test_service_status():
    """Test the service status endpoint."""
    print("📊 Testing service status endpoint...")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8005/workflow/status")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Service status retrieved!")
                print(f"🔧 Service: {result.get('service_config', {}).get('service_name', 'N/A')}")
                print(f"🤖 AI Provider: {result.get('service_config', {}).get('ai_provider', 'N/A')}")
                print(f"💾 Cache Enabled: {result.get('service_config', {}).get('cache_enabled', 'N/A')}")
                return True
            else:
                print(f"❌ Service status failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Service status error: {e}")
        return False

async def run_integration_tests():
    """Run all integration tests."""
    print("🚀 Starting Workflow Intelligence Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Service Status", test_service_status),
        ("Orchestrator Endpoint", test_orchestrator_endpoint),
        ("Comprehensive Analysis", test_comprehensive_analysis),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        
        print("-" * 40)
    
    # Summary
    print("\n📋 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! Workflow integration is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the service configuration and try again.")
        return False

if __name__ == "__main__":
    print("Workflow Intelligence Service - Integration Test")
    print("Make sure the intelligence service is running on localhost:8005")
    print()
    
    # Run tests
    success = asyncio.run(run_integration_tests())
    
    exit(0 if success else 1) 