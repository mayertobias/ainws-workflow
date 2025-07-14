#!/usr/bin/env python3
"""
Test script for lyrics API endpoint
Tests the actual FastAPI endpoint and saves response as JSON
"""

import sys
import os
import json
import asyncio
from pathlib import Path
from datetime import datetime

# Add the parent directory to Python path to fix import issues
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from app.api.lyrics import router, get_lyrics_analyzer
    from app.models.lyrics import LyricsAnalysisRequest
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    print("✓ Successfully imported API components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from workflow-content directory and dependencies are installed")
    sys.exit(1)

# Sample lyrics for testing (generic, non-copyrighted)
# Replace this with your own lyrics text for testing
SAMPLE_LYRICS = """
Verse 1:
Walking down the street at night
City lights are burning bright
Dreams are calling out my name
Nothing ever feels the same

Chorus:
We're dancing through the rain
Washing away the pain
Tomorrow's another day
We'll find another way

Verse 2:
Music playing in my head
All the words I should have said
Time keeps moving like a song
Hope will carry us along

Bridge:
Stars are shining up above
All we need is hope and love
Keep on moving, don't look back
Stay together on this track
"""

def create_test_app():
    """Create a test FastAPI app with the lyrics router"""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app

def save_response_to_json(response_data, filename_prefix="lyrics_api_response"):
    """Save the response data to a JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"
    filepath = Path(__file__).parent / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(response_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Response saved to: {filepath}")
    return str(filepath)

def test_lyrics_endpoint():
    """Test the lyrics analysis endpoint"""
    print("🧪 Testing Lyrics API Endpoint")
    print("=" * 50)
    
    # Create test app and client
    app = create_test_app()
    client = TestClient(app)
    
    # Prepare test request
    test_request = {
        "text": SAMPLE_LYRICS.strip(),
        "filename": "test_song.txt",
        "title": "Test Song"
    }
    
    print(f"📝 Testing with lyrics ({len(test_request['text'])} characters)")
    print(f"📁 Filename: {test_request['filename']}")
    print(f"🎵 Title: {test_request['title']}")
    print()
    
    try:
        # Make the API request
        print("🚀 Making API request...")
        response = client.post(
            "/api/v1/lyrics",
            json=test_request,
            headers={
                "X-Session-ID": "test-session-123",
                "User-Agent": "Test-Client/1.0"
            }
        )
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print("✅ API request successful!")
            
            # Print key response details
            print("\n📋 Response Summary:")
            print(f"  Status: {response_data.get('status', 'unknown')}")
            print(f"  Processing Time: {response_data.get('processing_time_ms', 0):.2f}ms")
            
            if 'results' in response_data:
                results = response_data['results']
                
                # Sentiment analysis
                if 'sentiment' in results:
                    sentiment = results['sentiment']
                    print(f"  Sentiment Polarity: {sentiment.get('polarity', 0):.3f}")
                    print(f"  Sentiment Subjectivity: {sentiment.get('subjectivity', 0):.3f}")
                    print(f"  Sentiment Label: {sentiment.get('label', 'unknown')}")
                
                # Statistics
                if 'statistics' in results:
                    stats = results['statistics']
                    print(f"  Word Count: {stats.get('word_count', 0)}")
                    print(f"  Unique Words: {stats.get('unique_words', 0)}")
                    print(f"  Vocabulary Density: {stats.get('vocabulary_density', 0):.3f}")
                
                # Complexity
                if 'complexity' in results:
                    complexity = results['complexity']
                    print(f"  Avg Sentence Length: {complexity.get('avg_sentence_length', 0):.2f}")
                    print(f"  Lexical Diversity: {complexity.get('lexical_diversity', 0):.3f}")
                
                # Themes
                if 'themes' in results:
                    themes = results['themes']
                    print(f"  Top Themes: {themes.get('top_words', [])[:5]}")
                
                # Readability
                if 'readability' in results:
                    print(f"  Readability Score: {results['readability']:.2f}")
            
            # Save response to JSON file
            print("\n💾 Saving response to JSON...")
            saved_file = save_response_to_json(response_data)
            
            return response_data, saved_file
            
        else:
            print(f"❌ API request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error during API test: {e}")
        return None, None

def test_sentiment_endpoint():
    """Test the sentiment analysis endpoint"""
    print("\n🧪 Testing Sentiment Endpoint")
    print("=" * 50)
    
    app = create_test_app()
    client = TestClient(app)
    
    test_request = {
        "text": "This is a beautiful day filled with hope and joy!"
    }
    
    try:
        response = client.post("/api/v1/sentiment", json=test_request)
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print("✅ Sentiment API request successful!")
            
            if 'results' in response_data:
                results = response_data['results']
                print(f"  Polarity: {results.get('polarity', 0):.3f}")
                print(f"  Subjectivity: {results.get('subjectivity', 0):.3f}")
                print(f"  Label: {results.get('label', 'unknown')}")
            
            # Save sentiment response
            saved_file = save_response_to_json(response_data, "sentiment_api_response")
            return response_data, saved_file
        else:
            print(f"❌ Sentiment API request failed: {response.text}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error during sentiment test: {e}")
        return None, None

def test_hss_features_endpoint():
    """Test the HSS features endpoint"""
    print("\n🧪 Testing HSS Features Endpoint")
    print("=" * 50)
    
    app = create_test_app()
    client = TestClient(app)
    
    try:
        response = client.get(
            "/api/v1/features/hss",
            params={"text": SAMPLE_LYRICS.strip()},
            headers={"X-Session-ID": "test-session-123"}
        )
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print("✅ HSS Features API request successful!")
            
            if 'features' in response_data:
                features = response_data['features']
                print(f"  Features extracted: {len(features)}")
                print("  Key features:")
                for key, value in list(features.items())[:5]:
                    print(f"    {key}: {value}")
            
            # Save HSS features response
            saved_file = save_response_to_json(response_data, "hss_features_api_response")
            return response_data, saved_file
        else:
            print(f"❌ HSS Features API request failed: {response.text}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error during HSS features test: {e}")
        return None, None

def main():
    """Run all API tests"""
    print("🎵 Lyrics API Test Suite")
    print("=" * 60)
    print(f"📂 Working directory: {os.getcwd()}")
    print(f"📁 Test file location: {__file__}")
    print()
    
    results = {}
    
    # Test main lyrics endpoint
    lyrics_result, lyrics_file = test_lyrics_endpoint()
    results['lyrics_analysis'] = {
        'success': lyrics_result is not None,
        'file': lyrics_file
    }
    
    # Test sentiment endpoint
    sentiment_result, sentiment_file = test_sentiment_endpoint()
    results['sentiment_analysis'] = {
        'success': sentiment_result is not None,
        'file': sentiment_file
    }
    
    # Test HSS features endpoint
    hss_result, hss_file = test_hss_features_endpoint()
    results['hss_features'] = {
        'success': hss_result is not None,
        'file': hss_file
    }
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 60)
    for test_name, result in results.items():
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result['file']:
            print(f"  📄 Response saved: {result['file']}")
    
    successful_tests = sum(1 for r in results.values() if r['success'])
    total_tests = len(results)
    print(f"\n🎯 Results: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests == total_tests:
        print("🎉 All tests passed successfully!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()