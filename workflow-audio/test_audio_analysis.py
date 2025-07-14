#!/usr/bin/env python3
"""
Test script for audio analysis using Essentia in the virtual environment
"""

import sys
import os
import numpy as np
import tempfile
import wave
from pathlib import Path

def create_test_audio():
    """Create a simple test audio file"""
    print("🎵 Creating test audio file...")
    
    sample_rate = 44100
    duration = 2  # seconds
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.5
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Create uploads directory
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    # Save as WAV file
    test_file = uploads_dir / "test_audio.wav"
    with wave.open(str(test_file), 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    print(f"✅ Created test audio file: {test_file}")
    return str(test_file)

def test_music_extractor(audio_file):
    """Test Essentia MusicExtractor"""
    print(f"\n🎼 Testing MusicExtractor on {Path(audio_file).name}...")
    
    try:
        import essentia.standard as es
        
        extractor = es.MusicExtractor()
        features, features_frames = extractor(audio_file)
        
        print(f"✅ Extracted {len(features.descriptorNames())} features")
        
        # Show key features
        key_features = {
            'BPM': features.get('rhythm.bpm', 0),
            'Key': features.get('tonal.key_krumhansl.key', 'Unknown'),
            'Scale': features.get('tonal.key_krumhansl.scale', 'Unknown'),
            'Loudness': features.get('lowlevel.loudness_ebu128.integrated', 0),
            'Spectral Centroid': features.get('lowlevel.spectral_centroid.mean', 0),
            'Dynamic Complexity': features.get('lowlevel.dynamic_complexity', 0),
        }
        
        print("   Key analysis results:")
        for name, value in key_features.items():
            if isinstance(value, (int, float)):
                print(f"     {name}: {value:.3f}")
            else:
                print(f"     {name}: {value}")
        
        return features
        
    except Exception as e:
        print(f"❌ MusicExtractor failed: {e}")
        return None

def test_audio_analyzer_service(audio_file):
    """Test the AudioAnalyzer service"""
    print(f"\n🛠️ Testing AudioAnalyzer service...")
    
    try:
        sys.path.insert(0, '/Users/manojveluchuri/saas/workflow/workflow-audio/app')
        from services.audio_analyzer import AudioAnalyzer
        
        analyzer = AudioAnalyzer()
        results = analyzer.analyze_audio(audio_file)
        
        print("✅ AudioAnalyzer service working!")
        print("   High-level features:")
        
        # Show high-level features
        hlf_features = [
            'tempo', 'energy', 'danceability', 'valence', 
            'acousticness', 'instrumentalness', 'liveness', 
            'speechiness', 'loudness', 'key', 'mode'
        ]
        
        for feature in hlf_features:
            if feature in results:
                value = results[feature]
                if isinstance(value, (int, float)):
                    print(f"     {feature}: {value:.3f}")
                else:
                    print(f"     {feature}: {value}")
        
        return results
        
    except Exception as e:
        print(f"⚠️  AudioAnalyzer service failed: {e}")
        return None

def test_existing_service():
    """Test the existing test_service.py"""
    print(f"\n🧪 Testing existing service...")
    
    try:
        # Run the existing test
        os.system("python test_service.py")
        return True
    except Exception as e:
        print(f"⚠️  Existing service test failed: {e}")
        return False

def main():
    """Run audio analysis tests"""
    print("🎼 Audio Analysis Test with Essentia Virtual Environment")
    print("=" * 65)
    
    # Create test audio
    test_audio_file = create_test_audio()
    
    # Test MusicExtractor
    features = test_music_extractor(test_audio_file)
    
    # Test AudioAnalyzer service
    service_results = test_audio_analyzer_service(test_audio_file)
    
    # Test existing service
    print(f"\n{'='*40}")
    print("Testing existing service endpoints...")
    print(f"{'='*40}")
    
    print("\n📝 To test the web service, run:")
    print("   1. In one terminal: source essentia_env/bin/activate && python -m uvicorn app.main:app --port 8001")
    print("   2. In another terminal: source essentia_env/bin/activate && python test_service.py")
    
    print(f"\n✅ Audio analysis testing completed!")
    print(f"   Test file available at: {test_audio_file}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 