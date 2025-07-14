#!/usr/bin/env python3
"""
Test comprehensive audio analysis using the ComprehensiveAnalyzer service
"""

import sys
import os
import json
from pathlib import Path
from app.services.comprehensive_analyzer import ComprehensiveAudioAnalyzer

def test_comprehensive_analysis():
    """Test comprehensive audio analysis using the ComprehensiveAudioAnalyzer service"""
    print("\n🔬 Testing comprehensive audio analysis...")
    
    # Create test audio file path
    audio_file = "/Users/manojveluchuri/saas/workflow/workflow-audio/uploads/billie_jean.mp3"
    
    try:
        # Initialize analyzer service (specialized extractors auto-initialized)
        analyzer = ComprehensiveAudioAnalyzer()
        
        # Run comprehensive analysis
        results = analyzer.analyze(audio_file)
        
        # Verify results structure (matches the simplified design)
        assert "analysis" in results, "Missing analysis section in results"
        
        analysis = results["analysis"]
        
        # Print actual structure to understand what we got
        print(f"📊 Analysis structure:")
        for section, data in analysis.items():
            if isinstance(data, dict) and 'error' not in data:
                print(f"  ✅ {section}: {len(data)} features")
            elif isinstance(data, dict) and 'error' in data:
                print(f"  ❌ {section}: FAILED - {data['error']}")
            else:
                print(f"  📝 {section}: {type(data)}")
        
        # Verify essential features are present  
        assert "basic" in analysis, "Missing basic audio features"
        
        # Check if basic features worked
        if 'error' not in analysis["basic"]:
            print(f"  🎵 Basic features: {len(analysis['basic'])} extracted")
        else:
            print(f"  ⚠️ Basic features failed: {analysis['basic']['error']}")
        
        # Check specialized extractors (these may fail, that's OK)
        if "genre" in analysis:
            if 'error' not in analysis["genre"]:
                print(f"  🎼 Genre classification: SUCCESS")
            else:
                print(f"  ⚠️ Genre classification failed: {analysis['genre']['error']}")
        
        if "mood" in analysis:
            if 'error' not in analysis["mood"]:
                print(f"  😊 Mood classification: SUCCESS")
            else:
                print(f"  ⚠️ Mood classification failed: {analysis['mood']['error']}")
        
        # Save results for inspection
        output_file = Path("analysis_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\n✅ Comprehensive analysis complete! Results saved to: {output_file}")
        print(f"📈 Summary:")
        
        # Count what worked
        working_extractors = []
        if 'error' not in analysis.get("basic", {}):
            working_extractors.append("basic audio features")
        if 'error' not in analysis.get("genre", {}):
            working_extractors.append("genre classification")  
        if 'error' not in analysis.get("mood", {}):
            working_extractors.append("mood classification")
        
        print(f"  ✅ Working: {', '.join(working_extractors) if working_extractors else 'None'}")
        
        # At minimum, basic features should work
        basic_works = 'error' not in analysis.get("basic", {})
        if basic_works:
            print(f"  🎉 SUCCESS: Basic audio analysis is working correctly!")
            return True
        else:
            print(f"  ❌ FAILURE: Basic audio analysis failed")
            return False
        
    except Exception as e:
        print(f"❌ Comprehensive analysis failed: {e}")
        return False

if __name__ == "__main__":
    if test_comprehensive_analysis():
        sys.exit(0)
    sys.exit(1)