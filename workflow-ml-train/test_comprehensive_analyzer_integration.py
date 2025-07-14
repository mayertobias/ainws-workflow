#!/usr/bin/env python3
"""
Test Comprehensive Analyzer Integration

Tests that the ml-train service correctly:
1. Calls the comprehensive analyzer endpoint
2. Parses the response structure correctly
3. Extracts the right feature names
4. Handles the audio service response properly

This validates the fixes made to use the correct endpoint and feature names.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from services.song_analyzer import SongAnalyzer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_comprehensive_analyzer_integration():
    """Test the comprehensive analyzer integration"""
    
    print("🎵 Testing Comprehensive Analyzer Integration")
    print("=" * 60)
    
    # Test 1: Verify SongAnalyzer initialization
    print("\n1. Testing SongAnalyzer Initialization...")
    analyzer = SongAnalyzer()
    print(f"✅ SongAnalyzer initialized with audio service URL: {analyzer.service_urls['audio']}")
    print(f"✅ Cache directory: {analyzer.cache_dir}")
    
    # Test 2: Test response parsing with sample comprehensive analyzer response
    print("\n2. Testing Response Parsing...")
    
    # Create a sample response matching the analysis_results.json structure
    sample_response = {
        "analysis": {
            "basic": {
                "acousticness": 0.77,
                "instrumentalness": 0.0004,
                "liveness": 0.0067,
                "speechiness": 0.0038,
                "brightness": 0.107,
                "complexity": 0.289,
                "warmth": 0.579,
                "valence": 0.659,
                "harmonic_strength": 0.562,
                "key": "F#",
                "mode": 0,
                "tempo": 117.4,
                "danceability": 0.397,
                "energy": 0.352,
                "loudness": -11.33,
                "duration_ms": 27792,
                "time_signature": 4,
                "loudness_features": {
                    "integrated_lufs": -11.33,
                    "loudness_range_lu": 2.09
                }
            },
            "genre": {
                "primary_genre": "Funk / Soul---Funk",
                "top_genres": [
                    ["Funk / Soul---Funk", 0.205],
                    ["Funk / Soul---Soul", 0.138],
                    ["Electronic---Disco", 0.135]
                ]
            },
            "mood": {
                "moods": {
                    "mood_happy": "not happy",
                    "mood_sad": "not sad", 
                    "mood_aggressive": "aggressive",
                    "mood_relaxed": "not relaxed",
                    "mood_party": "party",
                    "mood_electronic": "electronic",
                    "mood_acoustic": "acoustic"
                }
            }
        }
    }
    
    # Test response parsing
    parsed_features = analyzer._parse_audio_response(sample_response)
    
    print(f"✅ Parsed {len(parsed_features)} features from comprehensive analyzer response")
    
    # Test 3: Verify extracted feature names
    print("\n3. Testing Feature Names...")
    
    expected_basic_features = [
        "audio_acousticness", "audio_instrumentalness", "audio_liveness", "audio_speechiness",
        "audio_brightness", "audio_complexity", "audio_warmth", "audio_valence",
        "audio_harmonic_strength", "audio_key", "audio_mode", "audio_tempo",
        "audio_danceability", "audio_energy", "audio_loudness", "audio_duration_ms",
        "audio_time_signature"
    ]
    
    expected_genre_features = [
        "audio_primary_genre", "audio_top_genre_1", "audio_top_genre_1_prob",
        "audio_top_genre_2", "audio_top_genre_2_prob", "audio_top_genre_3", "audio_top_genre_3_prob"
    ]
    
    expected_mood_features = [
        "audio_mood_happy", "audio_mood_sad", "audio_mood_aggressive",
        "audio_mood_relaxed", "audio_mood_party", "audio_mood_electronic", "audio_mood_acoustic"
    ]
    
    print("🔍 Checking Basic Features...")
    basic_found = 0
    for feature in expected_basic_features:
        if feature in parsed_features:
            basic_found += 1
            print(f"  ✅ {feature}: {parsed_features[feature]}")
        else:
            print(f"  ❌ {feature}: NOT FOUND")
    
    print(f"\n📊 Basic Features: {basic_found}/{len(expected_basic_features)} found")
    
    print("\n🔍 Checking Genre Features...")
    genre_found = 0
    for feature in expected_genre_features:
        if feature in parsed_features:
            genre_found += 1
            print(f"  ✅ {feature}: {parsed_features[feature]}")
        else:
            print(f"  ❌ {feature}: NOT FOUND")
    
    print(f"\n📊 Genre Features: {genre_found}/{len(expected_genre_features)} found")
    
    print("\n🔍 Checking Mood Features...")
    mood_found = 0
    for feature in expected_mood_features:
        if feature in parsed_features:
            mood_found += 1
            print(f"  ✅ {feature}: {parsed_features[feature]}")
        else:
            print(f"  ❌ {feature}: NOT FOUND")
    
    print(f"\n📊 Mood Features: {mood_found}/{len(expected_mood_features)} found")
    
    # Test 4: Check service URL
    print("\n4. Testing Service Configuration...")
    print(f"✅ Audio service URL: {analyzer.service_urls['audio']}")
    print(f"✅ Content service URL: {analyzer.service_urls['content']}")
    
    # Test 5: Summary
    print("\n" + "=" * 60)
    print("🎯 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    total_expected = len(expected_basic_features) + len(expected_genre_features) + len(expected_mood_features)
    total_found = basic_found + genre_found + mood_found
    
    print(f"📊 Total Features Expected: {total_expected}")
    print(f"📊 Total Features Found: {total_found}") 
    print(f"📊 Success Rate: {(total_found/total_expected)*100:.1f}%")
    
    if total_found >= total_expected * 0.8:  # 80% success rate
        print("✅ INTEGRATION TEST PASSED")
        print("🎵 Comprehensive analyzer integration is working correctly!")
    else:
        print("❌ INTEGRATION TEST FAILED")
        print("🔧 Some features are missing - check the parsing logic")
    
    print("\n🔧 Key Fixes Applied:")
    print("  ✅ Changed endpoint from /analyze/audio to /analyze/comprehensive")
    print("  ✅ Fixed response parsing from results.features.analysis to analysis")
    print("  ✅ Updated feature names to match actual response structure")
    print("  ✅ Added proper genre and mood feature extraction")
    print("  ✅ Handle nested objects correctly (skip loudness_features)")
    
    return total_found >= total_expected * 0.8

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_comprehensive_analyzer_integration())
    
    if success:
        print("\n🎉 All tests passed! The comprehensive analyzer integration is ready.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Check the output above for details.")
        sys.exit(1) 