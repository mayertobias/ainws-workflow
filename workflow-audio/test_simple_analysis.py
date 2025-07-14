#!/usr/bin/env python3
"""
Simple test for Essentia MusicExtractor with proper Pool handling
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
    duration = 3  # seconds
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.5
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Create uploads directory
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    # Save as WAV file
    test_file = uploads_dir / "test_simple.wav"
    with wave.open(str(test_file), 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    print(f"✅ Created test audio file: {test_file}")
    return str(test_file)

def test_music_extractor_detailed(audio_file):
    """Test Essentia MusicExtractor with proper Pool handling"""
    print(f"\n🎼 Testing MusicExtractor analysis...")
    
    try:
        import essentia.standard as es
        
        extractor = es.MusicExtractor()
        features, features_frames = extractor(audio_file)
        
        print(f"✅ Successfully extracted features!")
        print(f"   Total descriptors: {len(features.descriptorNames())}")
        
        # Access features using proper Pool syntax
        descriptor_names = features.descriptorNames()
        
        # Key features to display
        key_features = [
            'rhythm.bpm',
            'tonal.key_krumhansl.key', 
            'tonal.key_krumhansl.scale',
            'lowlevel.loudness_ebu128.integrated',
            'lowlevel.spectral_centroid.mean',
            'lowlevel.dynamic_complexity',
            'lowlevel.spectral_energy.mean'
        ]
        
        print("\n   Key Features:")
        for feature_name in key_features:
            if feature_name in descriptor_names:
                try:
                    value = features[feature_name]
                    if isinstance(value, (int, float)):
                        print(f"     {feature_name}: {value:.3f}")
                    else:
                        print(f"     {feature_name}: {value}")
                except Exception as e:
                    print(f"     {feature_name}: Error - {e}")
        
        # Show some categories
        categories = {}
        for name in descriptor_names:
            category = name.split('.')[0]
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        
        print(f"\n   Feature Categories:")
        for category, count in categories.items():
            print(f"     {category}: {count} features")
        
        return features
        
    except Exception as e:
        print(f"❌ MusicExtractor failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_individual_algorithms(audio_file):
    """Test individual Essentia algorithms"""
    print(f"\n🔧 Testing individual algorithms...")
    
    try:
        import essentia.standard as es
        
        # Load audio
        loader = es.MonoLoader(filename=audio_file, sampleRate=44100)
        audio = loader()
        
        print(f"   Audio loaded: {len(audio)} samples ({len(audio)/44100:.2f}s)")
        
        # Test individual algorithms
        results = {}
        
        # 1. Rhythm Analysis
        print("   Testing rhythm analysis...")
        rhythm_extractor = es.RhythmExtractor2013()
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
        results['bpm'] = float(bpm)
        results['beats_confidence'] = float(beats_confidence)
        print(f"     BPM: {bpm:.1f}, Confidence: {beats_confidence:.3f}")
        
        # 2. Key Detection
        print("   Testing key detection...")
        key_extractor = es.KeyExtractor()
        key, scale, key_strength = key_extractor(audio)
        results['key'] = key
        results['scale'] = scale
        results['key_strength'] = float(key_strength)
        print(f"     Key: {key} {scale}, Strength: {key_strength:.3f}")
        
        # 3. Loudness
        print("   Testing loudness...")
        loudness_extractor = es.Loudness()
        loudness = loudness_extractor(audio)
        results['loudness'] = float(loudness)
        print(f"     Loudness: {loudness:.3f} dB")
        
        # 4. Spectral features
        print("   Testing spectral features...")
        windowing = es.Windowing(type='hann')
        spectrum = es.Spectrum()
        centroid = es.Centroid(range=22050)
        rolloff = es.RollOff()
        
        centroids = []
        rolloffs = []
        
        for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024):
            windowed = windowing(frame)
            spec = spectrum(windowed)
            centroids.append(centroid(spec))
            rolloffs.append(rolloff(spec))
        
        results['spectral_centroid'] = float(np.mean(centroids))
        results['spectral_rolloff'] = float(np.mean(rolloffs))
        print(f"     Spectral Centroid: {np.mean(centroids):.1f} Hz")
        print(f"     Spectral Rolloff: {np.mean(rolloffs):.1f} Hz")
        
        print("✅ Individual algorithms working!")
        return results
        
    except Exception as e:
        print(f"❌ Individual algorithms failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run simple audio analysis tests"""
    print("🎼 Simple Essentia Audio Analysis Test")
    print("=" * 50)
    
    # Create test audio
    test_audio_file = create_test_audio()
    
    # Test MusicExtractor
    features = test_music_extractor_detailed(test_audio_file)
    
    # Test individual algorithms
    individual_results = test_individual_algorithms(test_audio_file)
    
    print(f"\n{'='*50}")
    print("🎯 Testing Summary:")
    print(f"   ✅ Test audio created: {test_audio_file}")
    print(f"   ✅ MusicExtractor: {'Working' if features else 'Failed'}")
    print(f"   ✅ Individual algorithms: {'Working' if individual_results else 'Failed'}")
    
    print(f"\n📋 Next Steps:")
    print(f"   1. Start the web service:")
    print(f"      source essentia_env/bin/activate")
    print(f"      python -m uvicorn app.main:app --port 8001 --reload")
    print(f"   2. Test the service:")
    print(f"      python test_service.py")
    
    return 0

if __name__ == "__main__":
    exit(main()) 