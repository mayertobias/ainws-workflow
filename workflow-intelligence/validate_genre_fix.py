#!/usr/bin/env python3
"""
Simple validation script to test genre mapping logic without service dependencies.
This validates that the genre extraction logic works correctly.
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

def extract_genre_logic(audio_data: dict, song_metadata: dict) -> str:
    """Simplified version of the genre extraction logic."""
    
    # Try song metadata first
    genre = song_metadata.get("genre")
    if genre:
        # If genre is numeric (like "89"), try to find the readable name
        if isinstance(genre, (int, str)) and str(genre).isdigit():
            # Check if there's a corresponding name field or try audio analysis
            genre_name = (song_metadata.get("genre_name") or 
                         audio_data.get("audio_primary_genre_name") or
                         audio_data.get("primary_genre"))
            if genre_name:
                # Clean genre name (remove sub-genre parts like "Funk / Soul---Funk" -> "Funk")
                clean_genre = clean_genre_name(genre_name)
                print(f"  ✅ Found readable genre name: {clean_genre} for encoded value: {genre}")
                return clean_genre
            else:
                print(f"  ⚠️ Received numeric genre: {genre} without readable name, trying fallback methods")
                # Don't immediately default to pop, try other methods first
        else:
            # Clean and return the string genre
            clean_genre = clean_genre_name(str(genre))
            print(f"  ✅ Using song metadata genre: {clean_genre}")
            return clean_genre
    
    # Try to get readable genre from audio features (correct field name)
    primary_genre = audio_data.get("primary_genre") or audio_data.get("audio_primary_genre")
    if primary_genre:
        clean_genre = clean_genre_name(str(primary_genre))
        print(f"  ✅ Using audio primary genre: {clean_genre}")
        return clean_genre
    
    # Try genre predictions (most reliable fallback)
    genre_predictions = audio_data.get("genre_predictions", {})
    if genre_predictions:
        top_genre = max(genre_predictions.items(), key=lambda x: x[1])[0]
        # Check if this is numeric
        if str(top_genre).isdigit():
            print(f"  ⚠️ Received numeric genre prediction: {top_genre}, cannot map to readable name")
        else:
            clean_genre = clean_genre_name(str(top_genre))
            print(f"  ✅ Using top genre prediction: {clean_genre}")
            return clean_genre
    
    # Final fallback - log the issue and default to pop
    print(f"  ❌ No valid genre found in data. Available keys: audio_data={list(audio_data.keys())}, song_metadata={list(song_metadata.keys())}")
    return "pop"

def test_genre_scenarios():
    """Test various genre mapping scenarios."""
    
    print("🎵 Genre Mapping Validation")
    print("=" * 50)
    
    test_cases = [
        {
            "name": "🎸 Essentia Format - Funk/Soul",
            "audio_data": {"primary_genre": "Funk / Soul---Funk"},
            "song_metadata": {},
            "expected": "funk"
        },
        {
            "name": "🔢 Numeric Genre with Primary Genre",
            "audio_data": {"primary_genre": "Electronic---House"},
            "song_metadata": {"genre": "89"},
            "expected": "electronic"
        },
        {
            "name": "🎯 Genre Predictions",
            "audio_data": {"genre_predictions": {"rock": 0.8, "pop": 0.2}},
            "song_metadata": {},
            "expected": "rock"
        },
        {
            "name": "🎤 Direct String Genre - Hip-Hop",
            "audio_data": {},
            "song_metadata": {"genre": "Hip-Hop"},
            "expected": "hip-hop"
        },
        {
            "name": "🎶 R&B Variation",
            "audio_data": {"primary_genre": "R&B---Contemporary R&B"},
            "song_metadata": {},
            "expected": "rnb"
        },
        {
            "name": "🤖 Numeric-only predictions (should default)",
            "audio_data": {"genre_predictions": {"89": 0.7, "45": 0.3}},
            "song_metadata": {},
            "expected": "pop"
        },
        {
            "name": "🚫 No Genre Data (fallback)",
            "audio_data": {},
            "song_metadata": {},
            "expected": "pop"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Input: audio_data={test_case['audio_data']}")
        print(f"   Input: song_metadata={test_case['song_metadata']}")
        
        try:
            result = extract_genre_logic(test_case["audio_data"], test_case["song_metadata"])
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
    
    print("\n📊 Validation Summary")
    print("=" * 50)
    print(f"✅ Passed: {passed}/{total} ({(passed/total)*100:.1f}%)")
    print(f"❌ Failed: {total - passed}")
    
    if passed == total:
        print("\n🎉 All genre mapping tests passed!")
        print("✅ The fix correctly handles various genre formats")
        print("✅ No longer defaults to 'pop' inappropriately")
        return True
    else:
        print("\n⚠️ Some tests failed - needs investigation")
        return False

if __name__ == "__main__":
    success = test_genre_scenarios()
    
    if success:
        print("\n🎯 Genre mapping fix validation: SUCCESS")
        print("The workflow-intelligence service should now correctly map genres!")
    else:
        print("\n❌ Genre mapping fix validation: FAILED")
        print("There are still issues with the genre mapping logic.")