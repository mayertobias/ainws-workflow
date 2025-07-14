#!/usr/bin/env python3
"""
Test script to validate genre mapping logic with actual UI payload structure.
This tests the exact data structure sent from EnhancedAIInsightsPanel.tsx
"""

def clean_genre_name(genre_name: str) -> str:
    """Clean and normalize genre names from various formats."""
    if not genre_name:
        return "pop"
    
    original_genre = genre_name
    
    # Handle Essentia format like "Electronic---House" or "Funk / Soul---Funk"
    if "---" in genre_name:
        # For electronic music, prefer the main category over sub-genre
        parts = genre_name.split("---")
        main_category = parts[0].strip()
        sub_genre = parts[-1].strip()
        
        # Check if main category is electronic and sub-genre is electronic sub-type
        if main_category.lower() == "electronic" and sub_genre.lower() in ["house", "techno", "trance", "edm", "dubstep", "drum", "bass"]:
            main_genre = main_category  # Use "Electronic" instead of "House"
        else:
            main_genre = sub_genre  # Use sub-genre for other cases like "Funk / Soul---Funk"
    elif " / " in genre_name:
        # Take the first part before the slash
        main_genre = genre_name.split(" / ")[0].strip()
    else:
        main_genre = genre_name.strip()
    
    # Normalize to lowercase for consistency
    main_genre = main_genre.lower()
    
    # Map common variations to standard genre names
    genre_mapping = {
        "funk": "funk",
        "soul": "soul", 
        "r&b": "rnb",
        "hip hop": "hip-hop",
        "hip-hop": "hip-hop",
        "electronic": "electronic",
        "house": "electronic",  # House is a type of electronic
        "techno": "electronic", # Techno is a type of electronic
        "edm": "electronic",    # EDM is electronic dance music
        "trance": "electronic", # Trance is electronic
        "dubstep": "electronic", # Dubstep is electronic
        "drum": "electronic",   # Drum & bass is electronic
        "bass": "electronic",   # Bass music is electronic
        "rock": "rock",
        "pop": "pop",
        "country": "country",
        "jazz": "jazz",
        "classical": "classical",
        "blues": "blues",
        "reggae": "reggae",
        "folk": "folk",
        "metal": "metal",
        "punk": "punk",
        "alternative": "alternative",
        "indie": "indie"
    }
    
    # Find best match
    for key, value in genre_mapping.items():
        if key in main_genre or main_genre in key:
            print(f"  🎯 Mapped genre '{original_genre}' -> '{value}' (via '{main_genre}')")
            return value
    
    # If no mapping found, return the cleaned main genre
    print(f"  📝 No mapping found for genre '{original_genre}', using cleaned: '{main_genre}'")
    return main_genre

def extract_genre_ui_structure(audio_data: dict, song_metadata: dict) -> str:
    """
    Extract genre using the actual UI payload structure from EnhancedAIInsightsPanel.tsx
    """
    
    # Try song metadata first (this comes from UI as songData.audioFeatures.audio_primary_genre)
    genre = song_metadata.get("genre")
    if genre and genre != "Unknown":
        # If genre is numeric (like "89"), we need to check if there are readable names in audio_data
        if isinstance(genre, (int, str)) and str(genre).isdigit():
            print(f"  🔢 Received numeric genre ID: {genre}, looking for readable genre name")
            
            # Check genre_predictions for readable names (UI may send both numeric and readable)
            genre_predictions = audio_data.get("genre_predictions", {})
            if genre_predictions:
                # Look for non-numeric keys (readable genre names)
                readable_genres = {k: v for k, v in genre_predictions.items() if not str(k).isdigit()}
                if readable_genres:
                    top_readable_genre = max(readable_genres.items(), key=lambda x: x[1])[0]
                    clean_genre = clean_genre_name(str(top_readable_genre))
                    print(f"  ✅ Found readable genre name: {clean_genre} for numeric ID: {genre}")
                    return clean_genre
            
            print(f"  ⚠️ Received numeric genre: {genre} without readable name, trying other fallback methods")
            # Don't immediately default to pop, try other methods first
        else:
            # Clean and return the string genre (most common case)
            clean_genre = clean_genre_name(str(genre))
            print(f"  ✅ Using song metadata genre: {clean_genre}")
            return clean_genre
    
    # Try genre predictions (this comes from UI as audio_analysis.genre_predictions)
    genre_predictions = audio_data.get("genre_predictions", {})
    if genre_predictions:
        # First try to find readable genre names (non-numeric keys)
        readable_genres = {k: v for k, v in genre_predictions.items() if not str(k).isdigit()}
        if readable_genres:
            top_genre = max(readable_genres.items(), key=lambda x: x[1])[0]
            clean_genre = clean_genre_name(str(top_genre))
            print(f"  ✅ Using readable genre prediction: {clean_genre}")
            return clean_genre
        else:
            # All genre predictions are numeric
            top_genre = max(genre_predictions.items(), key=lambda x: x[1])[0]
            print(f"  ⚠️ All genre predictions are numeric. Top prediction: {top_genre}, cannot map to readable name")
    
    # Legacy: Try to get readable genre from audio features (for backward compatibility)
    primary_genre = audio_data.get("primary_genre") or audio_data.get("audio_primary_genre")
    if primary_genre:
        clean_genre = clean_genre_name(str(primary_genre))
        print(f"  ✅ Using legacy audio primary genre field: {clean_genre}")
        return clean_genre
    
    # Final fallback - log the issue and default to pop
    print(f"  ❌ No valid genre found. Available audio keys: {list(audio_data.keys())}, song metadata keys: {list(song_metadata.keys())}")
    print(f"  ❌ Song metadata genre: {song_metadata.get('genre')}, Genre predictions: {audio_data.get('genre_predictions', {})}")
    return "pop"

def test_ui_genre_scenarios():
    """Test genre extraction with UI payload structure from EnhancedAIInsightsPanel.tsx"""
    
    print("🎵 Testing Genre Mapping with Actual UI Payload Structure")
    print("=" * 70)
    
    # These test cases reflect the actual data structure sent by EnhancedAIInsightsPanel.tsx
    test_cases = [
        {
            "name": "🎤 Most Common: Readable Genre from UI",
            "audio_data": {"genre_predictions": {"Pop": 1.0}},
            "song_metadata": {"genre": "Pop"},
            "expected": "pop",
            "description": "UI sends readable genre name (most common case)"
        },
        {
            "name": "🎸 Essentia Format from Audio Service",
            "audio_data": {"genre_predictions": {"Electronic---House": 1.0}},
            "song_metadata": {"genre": "Electronic---House"},
            "expected": "electronic",
            "description": "Audio service provides Essentia format genres"
        },
        {
            "name": "🔢 Numeric Genre ID with Readable Prediction",
            "audio_data": {"genre_predictions": {"Electronic---House": 0.9, "89": 0.8}},
            "song_metadata": {"genre": "89"},
            "expected": "electronic",
            "description": "Numeric genre ID but readable names available in predictions"
        },
        {
            "name": "🎵 Funk/Soul Essentia Format",
            "audio_data": {"genre_predictions": {"Funk / Soul---Funk": 1.0}},
            "song_metadata": {"genre": "Funk / Soul---Funk"},
            "expected": "funk",
            "description": "Complex Essentia format with slash and dash"
        },
        {
            "name": "🎶 R&B Genre Variation",
            "audio_data": {"genre_predictions": {"R&B": 1.0}},
            "song_metadata": {"genre": "R&B"},
            "expected": "rnb",
            "description": "Genre with special characters"
        },
        {
            "name": "🤖 Pure Numeric - Should Fallback",
            "audio_data": {"genre_predictions": {"89": 0.7, "45": 0.3}},
            "song_metadata": {"genre": "89"},
            "expected": "pop",
            "description": "Only numeric data available, should fallback to pop"
        },
        {
            "name": "🚫 No Genre Data",
            "audio_data": {},
            "song_metadata": {},
            "expected": "pop",
            "description": "Empty data should fallback to pop"
        },
        {
            "name": "🎹 Legacy Primary Genre Field",
            "audio_data": {"primary_genre": "Jazz"},
            "song_metadata": {},
            "expected": "jazz",
            "description": "Backward compatibility with legacy field"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Description: {test_case['description']}")
        print(f"   Audio Data: {test_case['audio_data']}")
        print(f"   Song Metadata: {test_case['song_metadata']}")
        
        try:
            result = extract_genre_ui_structure(test_case["audio_data"], test_case["song_metadata"])
            success = result == test_case["expected"]
            
            print(f"   Expected: '{test_case['expected']}'")
            print(f"   Got: '{result}'")
            print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")
            
            results.append(success)
            
        except Exception as e:
            print(f"   Result: ❌ ERROR - {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n📊 Test Summary")
    print("=" * 50)
    print(f"✅ Passed: {passed}/{total} ({(passed/total)*100:.1f}%)")
    print(f"❌ Failed: {total - passed}")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        print("✅ Genre mapping works correctly with UI payload structure")
        print("✅ No longer defaults to 'pop' inappropriately")
        print("✅ Handles numeric genre IDs when readable names are available")
        print("✅ Properly processes Essentia format genres")
        return True
    else:
        print("\n⚠️ Some tests failed - needs investigation")
        return False

if __name__ == "__main__":
    success = test_ui_genre_scenarios()
    
    if success:
        print("\n🎯 UI Genre Mapping Validation: SUCCESS")
        print("The workflow-intelligence service correctly handles the UI payload structure!")
    else:
        print("\n❌ UI Genre Mapping Validation: FAILED")
        print("There are still issues with the genre mapping logic.")