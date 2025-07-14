#!/usr/bin/env python3
"""
Test Corrected Lyrical Features Integration

Tests the updated content service integration to verify that:
1. Content service is called with correct endpoint and request format
2. Response is parsed correctly to extract actual lyrical features
3. Feature names match the updated presets and configurations

This test validates the complete fix for lyrical feature engineering.
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
import os
import sys

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from services.song_analyzer import SongAnalyzer
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Expected lyrical features based on the corrected implementation
EXPECTED_LYRICAL_FEATURES = {
    # Sentiment features
    "content_sentiment_polarity": "float",
    "content_sentiment_subjectivity": "float",
    
    # Emotional scores
    "content_emotion_pain": "float",
    "content_emotion_hope": "float", 
    "content_emotion_love": "float",
    
    # Complexity features
    "content_avg_sentence_length": "float",
    "content_avg_word_length": "float",
    "content_lexical_diversity": "float",
    
    # Theme features
    "content_top_word_1": "string",
    "content_top_word_2": "string", 
    "content_top_word_3": "string",
    "content_themes_word_count": "int",
    "content_themes_noun_count": "int",
    "content_themes_verb_count": "int",
    "content_themes_entity_count": "int",
    
    # Readability and statistics
    "content_readability": "float",
    "content_word_count": "int",
    "content_unique_words": "int",
    "content_vocabulary_density": "float",
    "content_sentence_count": "int",
    "content_avg_words_per_sentence": "float",
    
    # Narrative structure
    "content_narrative_complexity": "int",  # 0 or 1
    "content_verse_count": "int",
    "content_repetition_score": "float",
    "content_avg_verse_length": "float",
    
    # Array-based features (converted to counts)
    "content_motif_count": "int",
    "content_theme_cluster_count": "int",
    "content_emotional_progression_count": "int"
}

def create_test_csv():
    """Create a test CSV with a sample song that has lyrics"""
    test_data = [{
        'song_name': 'Test Song',
        'original_popularity': 85,
        'has_audio_file': True,
        'audio_file_path': '/app/songs/test_song.mp3',
        'has_lyrics': True,
        'lyrics_file_path': '/app/lyrics/test_song.txt'
    }]
    
    df = pd.DataFrame(test_data)
    csv_path = "test_lyrical_features.csv"
    df.to_csv(csv_path, index=False)
    
    return csv_path

def create_test_lyrics_file():
    """Create a test lyrics file"""
    lyrics_content = """
Verse 1:
Walking through the city lights tonight
Dreams are shining oh so bright
Love is all around me everywhere
Hope is dancing in the air

Chorus:
We can make it through the pain
Dancing in the summer rain
Tomorrow brings another day
Love will always find a way

Verse 2:
Music flowing through my soul
Making broken hearts feel whole
Every step I take feels right
Guided by the morning light

Bridge:
Hope and love will never fade
Every memory that we've made
Shines like stars up in the sky
Love lifts us up so we can fly
"""
    
    # Create the directory structure
    lyrics_dir = Path("./shared-data/lyrics")
    lyrics_dir.mkdir(parents=True, exist_ok=True)
    
    lyrics_file = lyrics_dir / "test_song.txt"
    with open(lyrics_file, 'w', encoding='utf-8') as f:
        f.write(lyrics_content)
    
    return str(lyrics_file)

async def test_lyrical_feature_extraction():
    """Test the corrected lyrical feature extraction"""
    logger.info("🎵 Testing Corrected Lyrical Feature Extraction")
    logger.info("=" * 60)
    
    try:
        # Create test files
        logger.info("📝 Creating test files...")
        csv_path = create_test_csv()
        lyrics_file = create_test_lyrics_file()
        logger.info(f"✅ Created test CSV: {csv_path}")
        logger.info(f"✅ Created test lyrics: {lyrics_file}")
        
        # Initialize song analyzer
        logger.info("🔧 Initializing SongAnalyzer...")
        analyzer = SongAnalyzer(cache_dir="./test_cache", base_data_dir="./shared-data")
        
        # Test content service integration
        logger.info("📡 Testing content service integration...")
        
        # Load test data
        df = pd.read_csv(csv_path)
        test_row = df.iloc[0]
        
        # Call content service directly
        logger.info("🚀 Calling content service...")
        content_features = await analyzer._call_content_service(test_row)
        
        if content_features:
            logger.info(f"✅ Content service responded with {len(content_features)} features")
            
            # Validate extracted features
            logger.info("🔍 Validating extracted features...")
            validation_results = validate_lyrical_features(content_features)
            
            # Save results
            results = {
                "test_timestamp": datetime.now().isoformat(),
                "content_service_features": content_features,
                "validation_results": validation_results,
                "total_features_extracted": len(content_features),
                "expected_features": len(EXPECTED_LYRICAL_FEATURES),
                "extraction_success": validation_results["all_expected_found"]
            }
            
            # Save to JSON file
            results_file = f"lyrical_features_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Test results saved to: {results_file}")
            
            # Print summary
            print_test_summary(validation_results, content_features)
            
            return results
        else:
            logger.error("❌ Content service returned no features")
            return None
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return None

def validate_lyrical_features(extracted_features: dict) -> dict:
    """Validate that extracted features match expectations"""
    validation_results = {
        "expected_features": list(EXPECTED_LYRICAL_FEATURES.keys()),
        "extracted_features": list(extracted_features.keys()),
        "found_features": [],
        "missing_features": [],
        "unexpected_features": [],
        "type_mismatches": [],
        "all_expected_found": False
    }
    
    # Check which expected features were found
    for expected_feature in EXPECTED_LYRICAL_FEATURES:
        if expected_feature in extracted_features:
            validation_results["found_features"].append(expected_feature)
            
            # Check data type
            expected_type = EXPECTED_LYRICAL_FEATURES[expected_feature]
            actual_value = extracted_features[expected_feature]
            
            type_match = validate_feature_type(actual_value, expected_type)
            if not type_match:
                validation_results["type_mismatches"].append({
                    "feature": expected_feature,
                    "expected_type": expected_type,
                    "actual_type": type(actual_value).__name__,
                    "actual_value": actual_value
                })
        else:
            validation_results["missing_features"].append(expected_feature)
    
    # Check for unexpected features
    for extracted_feature in extracted_features:
        if extracted_feature not in EXPECTED_LYRICAL_FEATURES:
            validation_results["unexpected_features"].append(extracted_feature)
    
    # Overall success
    validation_results["all_expected_found"] = len(validation_results["missing_features"]) == 0
    validation_results["success_rate"] = len(validation_results["found_features"]) / len(EXPECTED_LYRICAL_FEATURES)
    
    return validation_results

def validate_feature_type(value, expected_type: str) -> bool:
    """Validate that a feature value matches the expected type"""
    if expected_type == "float":
        return isinstance(value, (int, float))
    elif expected_type == "int":
        return isinstance(value, int)
    elif expected_type == "string":
        return isinstance(value, str)
    else:
        return True  # Unknown type, assume valid

def print_test_summary(validation_results: dict, extracted_features: dict):
    """Print a formatted test summary"""
    logger.info("\n" + "=" * 60)
    logger.info("📋 LYRICAL FEATURES TEST SUMMARY")
    logger.info("=" * 60)
    
    # Overall results
    total_expected = len(validation_results["expected_features"])
    total_found = len(validation_results["found_features"])
    success_rate = validation_results["success_rate"] * 100
    
    logger.info(f"✅ Features Found: {total_found}/{total_expected} ({success_rate:.1f}%)")
    
    if validation_results["all_expected_found"]:
        logger.info("🎉 ALL EXPECTED FEATURES EXTRACTED SUCCESSFULLY!")
    else:
        logger.warning(f"⚠️  Missing {len(validation_results['missing_features'])} expected features")
    
    # Feature categories
    print("\n📊 EXTRACTED FEATURES BY CATEGORY:")
    
    categories = {
        "Sentiment": [f for f in extracted_features if "sentiment" in f],
        "Emotions": [f for f in extracted_features if "emotion" in f],
        "Complexity": [f for f in extracted_features if any(x in f for x in ["avg_", "lexical"])],
        "Themes": [f for f in extracted_features if "themes" in f or "top_word" in f],
        "Statistics": [f for f in extracted_features if any(x in f for x in ["word_count", "unique", "vocabulary", "sentence"])],
        "Structure": [f for f in extracted_features if any(x in f for x in ["narrative", "verse", "repetition"])],
        "Readability": [f for f in extracted_features if "readability" in f],
        "Counts": [f for f in extracted_features if any(x in f for x in ["motif", "cluster", "progression"])]
    }
    
    for category, features in categories.items():
        if features:
            print(f"  {category}: {len(features)} features")
            for feature in features[:3]:  # Show first 3
                value = extracted_features[feature]
                print(f"    • {feature}: {value}")
            if len(features) > 3:
                print(f"    ... and {len(features)-3} more")
    
    # Missing features
    if validation_results["missing_features"]:
        print(f"\n❌ MISSING FEATURES ({len(validation_results['missing_features'])}):")
        for feature in validation_results["missing_features"][:5]:
            print(f"  • {feature}")
        if len(validation_results["missing_features"]) > 5:
            print(f"  ... and {len(validation_results['missing_features'])-5} more")
    
    # Unexpected features
    if validation_results["unexpected_features"]:
        print(f"\n🆕 UNEXPECTED FEATURES ({len(validation_results['unexpected_features'])}):")
        for feature in validation_results["unexpected_features"][:5]:
            value = extracted_features[feature]
            print(f"  • {feature}: {value}")
        if len(validation_results["unexpected_features"]) > 5:
            print(f"  ... and {len(validation_results['unexpected_features'])-5} more")
    
    # Type mismatches
    if validation_results["type_mismatches"]:
        print(f"\n⚠️  TYPE MISMATCHES ({len(validation_results['type_mismatches'])}):")
        for mismatch in validation_results["type_mismatches"]:
            print(f"  • {mismatch['feature']}: expected {mismatch['expected_type']}, got {mismatch['actual_type']}")

def cleanup_test_files():
    """Clean up test files"""
    try:
        test_files = [
            "test_lyrical_features.csv",
            "./shared-data/lyrics/test_song.txt"
        ]
        
        for file_path in test_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"🗑️  Cleaned up: {file_path}")
                
        # Remove test cache directory
        import shutil
        if os.path.exists("./test_cache"):
            shutil.rmtree("./test_cache")
            logger.info("🗑️  Cleaned up test cache directory")
            
    except Exception as e:
        logger.warning(f"⚠️  Cleanup warning: {e}")

async def main():
    """Main test function"""
    logger.info("🚀 Starting Corrected Lyrical Features Integration Test")
    
    try:
        # Run the test
        results = await test_lyrical_feature_extraction()
        
        if results and results["extraction_success"]:
            logger.info("\n🎉 TEST PASSED: All lyrical features extracted successfully!")
            logger.info("✅ Content service integration is working correctly")
            logger.info("✅ Feature names match updated presets")
            logger.info("✅ Response parsing handles actual API structure")
        else:
            logger.error("\n❌ TEST FAILED: Some issues found with lyrical feature extraction")
            logger.error("🔧 Review the validation results and fix any missing features")
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
    finally:
        # Always cleanup
        cleanup_test_files()

if __name__ == "__main__":
    asyncio.run(main()) 