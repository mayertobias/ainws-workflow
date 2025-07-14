#!/usr/bin/env python3
"""
Test Actual Feature Extraction

Tests the actual audio and content services to see what features they return,
so we can identify inconsistencies with our feature vector implementation.

Uses "Billie Jean" as a test case since it has both audio and lyrics files.
"""

import asyncio
import aiohttp
import json
import logging
from pathlib import Path
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Service URLs (adjust if needed)
AUDIO_SERVICE_URL = "http://localhost:8001"
CONTENT_SERVICE_URL = "http://localhost:8002"

# Test song details (from training data)
TEST_SONG = {
    "name": "Billie Jean",
    "audio_file": "/app/songs/popularity_9/Billie Jean.mp3",
    "lyrics_file": "/app/lyrics/popularity_9/Billie Jean.txt"
}

async def test_audio_features():
    """Test audio feature extraction"""
    logger.info("🎵 Testing Audio Service Feature Extraction...")
    
    try:
        # Prepare the audio file for upload
        # Note: The actual file paths are internal to the services
        # We'll use a test endpoint to analyze the file
        
        async with aiohttp.ClientSession() as session:
            # Method 1: Try direct analysis if the service supports file paths
            try:
                audio_response = await session.post(
                    f"{AUDIO_SERVICE_URL}/analyze-file",
                    json={
                        "file_path": TEST_SONG["audio_file"],
                        "analysis_type": "comprehensive"
                    },
                    timeout=60
                )
                
                if audio_response.status == 200:
                    audio_data = await audio_response.json()
                    logger.info("✅ Audio service responded successfully")
                    return audio_data
                else:
                    logger.warning(f"⚠️ Audio service returned status {audio_response.status}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Direct file analysis failed: {e}")
            
            # Method 2: Try health check to see service structure
            try:
                health_response = await session.get(f"{AUDIO_SERVICE_URL}/health")
                if health_response.status == 200:
                    health_data = await health_response.json()
                    logger.info("✅ Audio service is healthy")
                    logger.info(f"📋 Audio service info: {health_data}")
                else:
                    logger.error(f"❌ Audio service health check failed: {health_response.status}")
                    
            except Exception as e:
                logger.error(f"❌ Audio service health check error: {e}")
            
            # Method 3: Try features endpoint to see available features
            try:
                features_response = await session.get(f"{AUDIO_SERVICE_URL}/features")
                if features_response.status == 200:
                    features_data = await features_response.json()
                    logger.info("✅ Got audio service features")
                    return {"service_features": features_data, "type": "feature_schema"}
                else:
                    logger.warning(f"⚠️ Audio features endpoint returned {features_response.status}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Audio features endpoint error: {e}")
                
        return None
        
    except Exception as e:
        logger.error(f"❌ Audio service test failed: {e}")
        return None

async def test_content_features():
    """Test content feature extraction"""
    logger.info("📝 Testing Content Service Feature Extraction...")
    
    try:
        async with aiohttp.ClientSession() as session:
            # Method 1: Try direct analysis if the service supports file paths
            try:
                content_response = await session.post(
                    f"{CONTENT_SERVICE_URL}/analyze-lyrics",
                    json={
                        "lyrics_file": TEST_SONG["lyrics_file"],
                        "analysis_type": "comprehensive"
                    },
                    timeout=30
                )
                
                if content_response.status == 200:
                    content_data = await content_response.json()
                    logger.info("✅ Content service responded successfully")
                    return content_data
                else:
                    logger.warning(f"⚠️ Content service returned status {content_response.status}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Direct lyrics analysis failed: {e}")
            
            # Method 2: Try health check
            try:
                health_response = await session.get(f"{CONTENT_SERVICE_URL}/health")
                if health_response.status == 200:
                    health_data = await health_response.json()
                    logger.info("✅ Content service is healthy")
                    logger.info(f"📋 Content service info: {health_data}")
                else:
                    logger.error(f"❌ Content service health check failed: {health_response.status}")
                    
            except Exception as e:
                logger.error(f"❌ Content service health check error: {e}")
            
            # Method 3: Try features endpoint
            try:
                features_response = await session.get(f"{CONTENT_SERVICE_URL}/features")
                if features_response.status == 200:
                    features_data = await features_response.json()
                    logger.info("✅ Got content service features")
                    return {"service_features": features_data, "type": "feature_schema"}
                else:
                    logger.warning(f"⚠️ Content features endpoint returned {features_response.status}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Content features endpoint error: {e}")
                
        return None
        
    except Exception as e:
        logger.error(f"❌ Content service test failed: {e}")
        return None

def save_feature_response(service_name: str, response_data: dict):
    """Save feature extraction response to JSON file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"actual_{service_name}_features_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(response_data, f, indent=2)
        
        logger.info(f"💾 Saved {service_name} response to {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"❌ Failed to save {service_name} response: {e}")
        return None

def analyze_audio_features(audio_data: dict):
    """Analyze actual audio features vs our assumptions"""
    logger.info("🔍 Analyzing Audio Feature Structure...")
    
    analysis = {
        "assumptions_vs_reality": {},
        "actual_structure": {},
        "missing_features": [],
        "unexpected_features": [],
        "recommendations": []
    }
    
    try:
        if "results" in audio_data and "features" in audio_data["results"]:
            features = audio_data["results"]["features"]
            
            if "analysis" in features:
                analysis_data = features["analysis"]
                
                # Check each extractor
                for extractor, data in analysis_data.items():
                    if isinstance(data, dict) and "error" not in data:
                        logger.info(f"✅ {extractor} extractor working: {len(data)} features")
                        analysis["actual_structure"][extractor] = list(data.keys())
                    elif isinstance(data, dict) and "error" in data:
                        logger.warning(f"⚠️ {extractor} extractor failed: {data['error']}")
                        analysis["missing_features"].append(extractor)
                    else:
                        logger.info(f"ℹ️ {extractor} has data: {type(data)}")
                
                # Check our assumptions vs reality
                our_assumptions = [
                    "genre_rock", "genre_pop", "genre_electronic",
                    "basic_energy", "basic_valence", "basic_danceability",
                    "rhythm_bpm", "tonal_key", "mood_happy"
                ]
                
                actual_features = []
                if "basic" in analysis_data:
                    basic_features = analysis_data["basic"]
                    actual_features.extend([f"basic_{k}" for k in basic_features.keys()])
                
                if "genre" in analysis_data:
                    genre_data = analysis_data["genre"]
                    if "primary_genre" in genre_data:
                        actual_features.append("genre_primary")
                    if "genre_probabilities" in genre_data:
                        # Genre probabilities are dynamic based on detected genres
                        for genre in genre_data["genre_probabilities"].keys():
                            actual_features.append(f"genre_{genre}")
                
                # Compare assumptions vs reality
                for assumption in our_assumptions:
                    if assumption not in actual_features:
                        analysis["assumptions_vs_reality"][assumption] = "NOT_FOUND"
                
                # Find unexpected features
                for actual in actual_features:
                    if actual not in our_assumptions:
                        analysis["unexpected_features"].append(actual)
                
                # Recommendations
                if "genre" in analysis_data:
                    analysis["recommendations"].append(
                        "Use 'primary_genre' instead of specific genre flags like 'genre_rock'"
                    )
                
                if "basic" in analysis_data:
                    analysis["recommendations"].append(
                        "Use 'basic_*' prefix for all basic features like 'basic_energy'"
                    )
        
        logger.info("📊 Audio Feature Analysis Complete")
        return analysis
        
    except Exception as e:
        logger.error(f"❌ Audio feature analysis failed: {e}")
        return analysis

def analyze_content_features(content_data: dict):
    """Analyze actual content features vs our assumptions"""
    logger.info("🔍 Analyzing Content Feature Structure...")
    
    analysis = {
        "assumptions_vs_reality": {},
        "actual_structure": {},
        "missing_features": [],
        "unexpected_features": [],
        "recommendations": []
    }
    
    try:
        # Our assumptions about content features
        our_assumptions = [
            "sentiment_compound", "sentiment_positive", "sentiment_negative",
            "themes_love", "themes_party", "themes_sadness",
            "language_word_count", "language_complexity",
            "structure_verse_count"
        ]
        
        # Analyze actual structure based on response
        if content_data and "type" == "feature_schema":
            # This is a feature schema response
            if "service_features" in content_data:
                features_info = content_data["service_features"]
                analysis["actual_structure"] = features_info
                logger.info("📋 Got content service feature schema")
        else:
            # This would be an actual analysis response
            logger.info("📋 Got actual content analysis response")
            analysis["actual_structure"] = content_data
        
        logger.info("📊 Content Feature Analysis Complete")
        return analysis
        
    except Exception as e:
        logger.error(f"❌ Content feature analysis failed: {e}")
        return analysis

async def main():
    """Main test function"""
    logger.info("🚀 Starting Actual Feature Extraction Test")
    logger.info(f"🎵 Test Song: {TEST_SONG['name']}")
    logger.info(f"📁 Audio File: {TEST_SONG['audio_file']}")
    logger.info(f"📄 Lyrics File: {TEST_SONG['lyrics_file']}")
    
    # Test audio features
    logger.info("\n" + "="*60)
    audio_data = await test_audio_features()
    
    if audio_data:
        audio_file = save_feature_response("audio", audio_data)
        audio_analysis = analyze_audio_features(audio_data)
        
        if audio_analysis:
            analysis_file = save_feature_response("audio_analysis", audio_analysis)
    
    # Test content features  
    logger.info("\n" + "="*60)
    content_data = await test_content_features()
    
    if content_data:
        content_file = save_feature_response("content", content_data)
        content_analysis = analyze_content_features(content_data)
        
        if content_analysis:
            analysis_file = save_feature_response("content_analysis", content_analysis)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📋 FEATURE EXTRACTION TEST SUMMARY")
    logger.info("="*60)
    
    if audio_data:
        logger.info("✅ Audio service tested successfully")
    else:
        logger.error("❌ Audio service test failed")
    
    if content_data:
        logger.info("✅ Content service tested successfully")
    else:
        logger.error("❌ Content service test failed")
    
    logger.info("")
    logger.info("🔍 CHECK THE GENERATED JSON FILES:")
    
    # List generated files
    for file in Path(".").glob("actual_*_features_*.json"):
        logger.info(f"📄 {file}")
    
    logger.info("")
    logger.info("🎯 NEXT STEPS:")
    logger.info("1. Review the actual feature structures in the JSON files")
    logger.info("2. Compare with our feature vector assumptions")
    logger.info("3. Update feature vector implementation if needed")
    logger.info("4. Fix any naming inconsistencies")

if __name__ == "__main__":
    asyncio.run(main()) 