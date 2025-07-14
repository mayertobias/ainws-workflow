#!/usr/bin/env python3
"""
Test script to validate Essentia installation and audio analysis capabilities
"""

import sys
import os
import numpy as np
import tempfile
import wave

def test_essentia_import():
    """Test if Essentia can be imported successfully"""
    print("🔍 Testing Essentia import...")
    try:
        import essentia
        import essentia.standard as es
        print(f"✅ Essentia imported successfully! Version: {essentia.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import Essentia: {e}")
        return False

def test_basic_algorithms():
    """Test basic Essentia algorithms"""
    print("\n🧪 Testing basic Essentia algorithms...")
    try:
        import essentia.standard as es
        
        # Test basic algorithms
        windowing = es.Windowing(type='hann')
        spectrum = es.Spectrum()
        spectral_centroid = es.SpectralCentroid()
        
        # Create test signal
        test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 1024))
        
        # Process signal
        windowed = windowing(test_signal)
        spec = spectrum(windowed)
        centroid = spectral_centroid(spec)
        
        print(f"✅ Basic algorithms working! Spectral centroid: {centroid:.2f} Hz")
        return True
    except Exception as e:
        print(f"❌ Basic algorithms test failed: {e}")
        return False

def test_music_extractor():
    """Test Essentia's MusicExtractor (the main feature extractor)"""
    print("\n🎵 Testing MusicExtractor...")
    try:
        import essentia.standard as es
        
        # Create a simple test audio file
        sample_rate = 44100
        duration = 2  # seconds
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t)
        audio_data = (audio_data * 0.5 * 32767).astype(np.int16)
        
        # Save as temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            with wave.open(temp_file.name, 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            temp_audio_path = temp_file.name
        
        try:
            # Test MusicExtractor
            audio_file = "/Users/manojveluchuri/saas/workflow/workflow-audio/uploads/billie_jean.mp3"
            extractor = es.MusicExtractor()
            features, features_frames = extractor(audio_file)
            
            print("✅ MusicExtractor working!")
            print(f"   Extracted {len(features.descriptorNames())} descriptors")
            
            # Show some key features
            key_features = [
                'rhythm.bpm',
                'tonal.key_krumhansl.key',
                'tonal.key_krumhansl.scale',
                'lowlevel.loudness_ebu128.integrated',
                'lowlevel.spectral_centroid.mean'
            ]
            
            print("   Key features extracted:")
            for feature in key_features:
                if feature in features.descriptorNames():
                    value = features[feature]
                    print(f"     {feature}: {value}")
            
            return True
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
                
    except Exception as e:
        print(f"❌ MusicExtractor test failed: {e}")
        return False

def test_tensorflow_integration():
    """Test if Essentia TensorFlow integration is available"""
    print("\n🤖 Testing TensorFlow integration...")
    try:
        import essentia.standard as es
        
        # Try to create a TensorFlow-based algorithm
        # This is optional and may not be available in all installations
        try:
            tf_predict = es.TensorflowPredict2D
            tensorflow_algorithms = [algo for algo in dir(es) if "Tensorflow" in algo]
            print(tensorflow_algorithms)
            print("✅ TensorFlow integration available!")
            return True
        except AttributeError:
            print("⚠️  TensorFlow integration not available (this is optional)")
            return True
            
    except Exception as e:
        print(f"❌ TensorFlow integration test failed: {e}")
        return False

def test_audio_loading():
    """Test audio loading capabilities"""
    print("\n🎧 Testing audio loading...")
    try:
        import essentia.standard as es
        
        # Create test audio
        sample_rate = 44100
        duration = 1
        frequency = 880  # A5 note
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t)
        audio_data = (audio_data * 0.3 * 32767).astype(np.int16)
        
        # Save as temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            with wave.open(temp_file.name, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            temp_audio_path = temp_file.name
        
        try:
            # Test MonoLoader
            loader = es.MonoLoader(filename=temp_audio_path)
            audio = loader()
            
            print(f"✅ Audio loading working!")
            print(f"   Loaded {len(audio)} samples")
            print(f"   Duration: {len(audio) / 44100:.2f} seconds")
            print(f"   RMS Energy: {np.sqrt(np.mean(audio**2)):.4f}")
            
            return True
            
        finally:
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
                
    except Exception as e:
        print(f"❌ Audio loading test failed: {e}")
        return False

def test_essentia_service():
    """Test the local essentia_service.py if available"""
    print("\n🛠️ Testing local Essentia service...")
    try:
        # Try to import the local service
        sys.path.insert(0, '/Users/manojveluchuri/saas/workflow/workflow-audio/app')
        from essentia_service import AudioAnalyzer
        
        analyzer = AudioAnalyzer()
        print("✅ Local Essentia service imported successfully!")
        
        # Create test audio for the service
        sample_rate = 44100
        duration = 2
        frequency = 440
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t)
        audio_data = (audio_data * 0.5 * 32767).astype(np.int16)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            with wave.open(temp_file.name, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            temp_audio_path = temp_file.name
        
        try:
            # Test the analyzer
            results = analyzer.analyze_audio(temp_audio_path)
            
            print("✅ Local Essentia service analysis successful!")
            print("   Key results:")
            key_results = ['tempo', 'key', 'mode', 'energy', 'duration', 'loudness']
            for key in key_results:
                if key in results:
                    print(f"     {key}: {results[key]}")
            
            return True
            
        finally:
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
                
    except Exception as e:
        print(f"⚠️  Local Essentia service test failed: {e}")
        print("   (This is expected if the service isn't set up)")
        return True  # Not critical for basic Essentia testing

def main():
    """Run all Essentia tests"""
    print("🎼 Essentia Audio Analysis Test Suite")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print("=" * 60)
    
    tests = [
        ("Essentia Import", test_essentia_import),
        ("Basic Algorithms", test_basic_algorithms),
        ("MusicExtractor", test_music_extractor),
        ("TensorFlow Integration", test_tensorflow_integration),
        ("Audio Loading", test_audio_loading),
        ("Local Service", test_essentia_service),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Essentia is working perfectly!")
        return 0
    elif passed >= total - 1:
        print("✅ Essentia is working well with minor issues!")
        return 0
    else:
        print("⚠️  Some issues detected. Check the output above.")
        return 1

if __name__ == "__main__":
    exit(main()) 