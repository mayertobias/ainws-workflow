#!/usr/bin/env python3
"""
Test script for workflow-content microservice
"""

import asyncio
import httpx
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8002"

# Sample lyrics for testing
SAMPLE_LYRICS = """
Verse 1:
Walking down the street tonight
Feeling like everything's alright
The stars are shining bright above
This feeling must be love

Chorus:
Dancing in the moonlight
Everything feels so right
Hold me close and don't let go
This is all I need to know

Verse 2:
Time keeps moving fast it seems
But tonight we're living dreams
Together we can face the world
Our love story will unfold
"""

async def test_health_endpoint():
    """Test the health check endpoint"""
    print("Testing health endpoint...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/health")
            print(f"Health Status: {response.status_code}")
            print(f"Response: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

async def test_lyrics_analysis():
    """Test lyrics analysis endpoint"""
    print("\nTesting lyrics analysis...")
    
    request_data = {
        "text": SAMPLE_LYRICS,
        "language": "en",
        "analysis_type": "comprehensive"
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            start_time = time.time()
            response = await client.post(
                f"{BASE_URL}/analyze/lyrics",
                json=request_data
            )
            end_time = time.time()
            
            print(f"Analysis Status: {response.status_code}")
            print(f"Processing Time: {(end_time - start_time) * 1000:.2f}ms")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Analysis completed successfully!")
                print(f"Sentiment Polarity: {result['results']['sentiment']['polarity']:.3f}")
                print(f"Word Count: {result['results']['statistics']['word_count']}")
                print(f"Readability Score: {result['results']['readability']:.3f}")
                print(f"Theme Clusters: {len(result['results']['theme_clusters'])}")
                return True
            else:
                print(f"Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"Lyrics analysis failed: {e}")
            return False

async def test_sentiment_analysis():
    """Test sentiment analysis endpoint"""
    print("\nTesting sentiment analysis...")
    
    request_data = {
        "text": "I am feeling so happy and excited about this amazing day!"
    }
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.post(
                f"{BASE_URL}/analyze/sentiment",
                json=request_data
            )
            
            print(f"Sentiment Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Sentiment analysis completed!")
                print(f"Polarity: {result['results']['polarity']:.3f}")
                print(f"Subjectivity: {result['results']['subjectivity']:.3f}")
                return True
            else:
                print(f"Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"Sentiment analysis failed: {e}")
            return False

async def test_hss_features():
    """Test HSS features extraction"""
    print("\nTesting HSS features extraction...")
    
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            response = await client.get(
                f"{BASE_URL}/analyze/features/hss",
                params={"text": SAMPLE_LYRICS}
            )
            
            print(f"HSS Features Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"HSS features extracted successfully!")
                features = result['features']
                print(f"Sentiment Polarity: {features['sentiment_polarity']:.3f}")
                print(f"Lexical Diversity: {features['lexical_diversity']:.3f}")
                print(f"Theme Diversity: {features['theme_diversity']}")
                print(f"Readability: {features['readability']:.3f}")
                return True
            else:
                print(f"Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"HSS features extraction failed: {e}")
            return False

async def test_error_handling():
    """Test error handling with invalid input"""
    print("\nTesting error handling...")
    
    # Test with empty text
    request_data = {"text": ""}
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(
                f"{BASE_URL}/analyze/lyrics",
                json=request_data
            )
            
            print(f"Error handling test status: {response.status_code}")
            
            if response.status_code == 422:  # Validation error
                print("Error handling working correctly!")
                return True
            else:
                print(f"Unexpected response: {response.text}")
                return False
                
        except Exception as e:
            print(f"Error handling test failed: {e}")
            return False

async def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("WORKFLOW-CONTENT SERVICE TESTS")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Lyrics Analysis", test_lyrics_analysis),
        ("Sentiment Analysis", test_sentiment_analysis),
        ("HSS Features", test_hss_features),
        ("Error Handling", test_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = await test_func()
            if success:
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print(f"{'='*50}")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1) 