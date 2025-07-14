#!/usr/bin/env python3

"""
Debug script to see what features Essentia MusicExtractor actually provides
"""

import essentia.standard as es
from pathlib import Path

def list_essentia_features(audio_path: str):
    """List all features that Essentia MusicExtractor provides"""
    print(f"🔍 Analyzing features from: {audio_path}")
    
    try:
        # Initialize MusicExtractor
        extractor = es.MusicExtractor()
        
        # Extract features
        features_pool, features_frames_pool = extractor(str(audio_path))
        
        # Get all available descriptor names
        available_keys = sorted(features_pool.descriptorNames())
        
        print(f"\n📊 Total features available: {len(available_keys)}")
        print("\n🎵 Available Essentia Features:")
        print("=" * 50)
        
        # Group by category
        categories = {}
        for key in available_keys:
            category = key.split('.')[0] if '.' in key else 'root'
            if category not in categories:
                categories[category] = []
            categories[category].append(key)
        
        for category, keys in sorted(categories.items()):
            print(f"\n📁 {category.upper()}:")
            for key in keys:
                try:
                    value = features_pool[key]
                    value_type = type(value).__name__
                    if hasattr(value, '__len__') and len(value) > 0:
                        if hasattr(value, 'shape'):
                            shape_info = f" (shape: {value.shape})"
                        else:
                            shape_info = f" (len: {len(value)})"
                    else:
                        shape_info = ""
                    print(f"  ✓ {key} -> {value_type}{shape_info}")
                except Exception as e:
                    print(f"  ✗ {key} -> ERROR: {e}")
        
        print("\n" + "=" * 50)
        
        # Check specific features the extractors are looking for
        missing_features = [
            'lowlevel.spectral_rolloff',
            'tonal.chords_changes_rate',
            'lowlevel.silence_rate_30dB',
        ]
        
        print("\n🔍 Checking for missing features used by extractors:")
        for feature in missing_features:
            if feature in available_keys:
                print(f"  ✓ {feature} - FOUND")
            else:
                print(f"  ✗ {feature} - NOT FOUND")
                # Try to find similar features
                similar = [k for k in available_keys if any(part in k for part in feature.split('.'))]
                if similar:
                    print(f"    💡 Similar features: {similar[:3]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    audio_file = "/Users/manojveluchuri/saas/workflow/workflow-audio/uploads/billie_jean.mp3"
    if Path(audio_file).exists():
        list_essentia_features(audio_file)
    else:
        print(f"❌ Audio file not found: {audio_file}")