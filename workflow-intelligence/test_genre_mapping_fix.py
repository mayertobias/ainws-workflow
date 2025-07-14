#!/usr/bin/env python3
"""
Test script to validate genre mapping fixes in workflow-intelligence service.
This ensures that numeric genre IDs and various genre formats are correctly
mapped instead of defaulting to "pop".
"""

import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.agent_llm_service import AgentLLMService

def test_genre_extraction():
    """Test genre extraction with various data formats."""
    
    print("🧪 Testing Genre Mapping Fixes\n")
    
    # Create a mock AgentLLMService instance
    service = AgentLLMService()
    
    # Test cases
    test_cases = [
        {
            "name": "Essentia Format - Funk/Soul",
            "audio_data": {"primary_genre": "Funk / Soul---Funk"},
            "song_metadata": {},
            "expected": "funk"
        },
        {
            "name": "Numeric Genre with Readable Genre Prediction (UI Structure)",
            "audio_data": {"genre_predictions": {"Electronic---House": 1.0, "89": 0.8}},
            "song_metadata": {"genre": "89"},
            "expected": "electronic"
        },
        {
            "name": "Genre Predictions",
            "audio_data": {"genre_predictions": {"rock": 0.8, "pop": 0.2}},
            "song_metadata": {},
            "expected": "rock"
        },
        {
            "name": "Direct String Genre",
            "audio_data": {},
            "song_metadata": {"genre": "Hip-Hop"},
            "expected": "hip-hop"
        },
        {
            "name": "R&B Variation",
            "audio_data": {"primary_genre": "R&B---Contemporary R&B"},
            "song_metadata": {},
            "expected": "rnb"
        },
        {
            "name": "Numeric Genre Prediction (should still default)",
            "audio_data": {"genre_predictions": {"89": 0.7, "45": 0.3}},
            "song_metadata": {},
            "expected": "pop"  # Should still default since no readable names
        },
        {
            "name": "No Genre Data (fallback)",
            "audio_data": {},
            "song_metadata": {},
            "expected": "pop"
        },
        {
            "name": "UI Payload - Readable Genre from UI (Most Common Case)",
            "audio_data": {"genre_predictions": {"Pop": 1.0}},
            "song_metadata": {"genre": "Pop"},
            "expected": "pop"
        },
        {
            "name": "UI Payload - Essentia Format Genre from UI",
            "audio_data": {"genre_predictions": {"Electronic---House": 1.0}},
            "song_metadata": {"genre": "Electronic---House"},
            "expected": "electronic"
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"Testing: {test_case['name']}")
        
        try:
            result = service._extract_genre(test_case["audio_data"], test_case["song_metadata"])
            success = result == test_case["expected"]
            
            print(f"  Expected: {test_case['expected']}")
            print(f"  Got: {result}")
            print(f"  Status: {'✅ PASS' if success else '❌ FAIL'}")
            
            results.append({
                "test": test_case["name"],
                "expected": test_case["expected"],
                "actual": result,
                "passed": success
            })
            
        except Exception as e:
            print(f"  Status: ❌ ERROR - {e}")
            results.append({
                "test": test_case["name"],
                "expected": test_case["expected"],
                "actual": f"ERROR: {e}",
                "passed": False
            })
        
        print()
    
    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    
    print("📊 Test Summary")
    print("=" * 50)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! Genre mapping is working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the implementation.")
        print("\nFailed tests:")
        for result in results:
            if not result["passed"]:
                print(f"  - {result['test']}: Expected '{result['expected']}', got '{result['actual']}'")
        return False

def test_genre_cleaning():
    """Test the genre cleaning functionality."""
    
    print("\n🧹 Testing Genre Cleaning\n")
    
    service = AgentLLMService()
    
    test_cases = [
        ("Funk / Soul---Funk", "funk"),
        ("Electronic---House", "electronic"), 
        ("Hip-Hop", "hip-hop"),
        ("R&B", "rnb"),
        ("Pop", "pop"),
        ("Alternative Rock", "alternative"),
        ("Heavy Metal", "metal"),
        ("", "pop"),  # Empty string fallback
        ("Unknown Genre", "unknown genre")  # Unknown genre
    ]
    
    results = []
    
    for input_genre, expected in test_cases:
        try:
            result = service._clean_genre_name(input_genre)
            success = result == expected
            
            print(f"Input: '{input_genre}' -> Output: '{result}' (Expected: '{expected}') {'✅' if success else '❌'}")
            
            results.append(success)
            
        except Exception as e:
            print(f"Input: '{input_genre}' -> ERROR: {e} ❌")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nGenre Cleaning Results: {passed}/{total} passed ({(passed/total)*100:.1f}%)")
    
    return passed == total

if __name__ == "__main__":
    print("🎵 Workflow Intelligence - Genre Mapping Test Suite")
    print("=" * 60)
    
    # Run tests
    genre_extraction_passed = test_genre_extraction()
    genre_cleaning_passed = test_genre_cleaning()
    
    # Overall result
    if genre_extraction_passed and genre_cleaning_passed:
        print("\n🎯 Overall Result: ALL TESTS PASSED")
        print("Genre mapping is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Overall Result: SOME TESTS FAILED")
        print("Please check the implementation and fix the issues.")
        sys.exit(1)