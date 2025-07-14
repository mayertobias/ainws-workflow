#!/usr/bin/env python3
"""
Test script for Workflow Audio Analysis Microservice
Run this to verify your audio microservice is working correctly.
"""

import requests
import json
import time
import numpy as np
import wave
from pathlib import Path

def create_test_audio():
    """Create a simple test audio file"""
    print("🎵 Creating test audio file...")
    
    # Create a simple sine wave
    sample_rate = 44100
    duration = 2  # seconds
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit integers
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

def test_health():
    """Test service health"""
    print("🏥 Testing service health...")
    try:
        response = requests.get("http://localhost:8001/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service is healthy: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to service: {e}")
        return False

def test_status():
    """Test service status"""
    print("📊 Testing service status...")
    try:
        response = requests.get("http://localhost:8001/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service status retrieved")
            print(f"   Service: {data.get('service')}")
            print(f"   Features: {data.get('features', {})}")
            return True
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Status check error: {e}")
        return False

def test_extractors():
    """Test extractors endpoint"""
    print("🔧 Testing extractors endpoint...")
    try:
        response = requests.get("http://localhost:8001/extractors", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Extractors available: {data.get('available_extractors', [])}")
            return True
        else:
            print(f"❌ Extractors check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Extractors check error: {e}")
        return False

def test_basic_analysis(audio_file):
    """Test basic audio analysis"""
    print("🎼 Testing basic audio analysis...")
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': ('test_audio.wav', f, 'audio/wav')}
            response = requests.post(
                "http://localhost:8001/analyze/basic",
                files=files,
                timeout=60
            )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Basic analysis successful!")
            print(f"   Filename: {data.get('filename')}")
            print(f"   Analysis type: {data.get('analysis_type')}")
            
            results = data.get('results', {})
            if results:
                print("   Key features:")
                for key in ['tempo', 'key', 'mode', 'energy', 'duration']:
                    if key in results:
                        print(f"     {key}: {results[key]}")
            return True
        else:
            print(f"❌ Basic analysis failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Basic analysis error: {e}")
        return False

def test_comprehensive_analysis(audio_file):
    """Test comprehensive audio analysis"""
    print("🎯 Testing comprehensive audio analysis...")
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': ('test_audio.wav', f, 'audio/wav')}
            response = requests.post(
                "http://localhost:8001/analyze/comprehensive",
                files=files,
                timeout=120
            )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Comprehensive analysis successful!")
            print(f"   Filename: {data.get('filename')}")
            print(f"   Analysis type: {data.get('analysis_type')}")
            
            results = data.get('results', {})
            if 'metadata' in results:
                metadata = results['metadata']
                print(f"   Processing time: {metadata.get('processing_time_seconds', 0):.2f}s")
                print(f"   Extractors used: {metadata.get('extractors_used', [])}")
                print(f"   Successful: {metadata.get('extractors_successful', [])}")
                if metadata.get('extractors_failed'):
                    print(f"   Failed: {metadata.get('extractors_failed', [])}")
            return True
        else:
            print(f"❌ Comprehensive analysis failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Comprehensive analysis error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Workflow Audio Microservice Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Service health
    total_tests += 1
    if test_health():
        tests_passed += 1
    
    # Test 2: Service status
    total_tests += 1
    if test_status():
        tests_passed += 1
    
    # Test 3: Extractors endpoint
    total_tests += 1
    if test_extractors():
        tests_passed += 1
    
    # Test 4: Create test audio and analyze
    test_audio_file = create_test_audio()
    if test_audio_file:
        # Test 5: Basic analysis
        total_tests += 1
        if test_basic_analysis(test_audio_file):
            tests_passed += 1
        
        # Test 6: Comprehensive analysis
        total_tests += 1
        if test_comprehensive_analysis(test_audio_file):
            tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your audio microservice is working perfectly!")
        print("\n📚 Next steps:")
        print("   • Visit http://localhost:8001/docs for API documentation")
        print("   • Try uploading your own audio files")
        print("   • Integrate with your main application")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("\n💡 Troubleshooting tips:")
        print("   1. Make sure Docker is running: docker info")
        print("   2. Start the services: ./start.sh")
        print("   3. Check service logs: docker-compose logs workflow-audio")
        return 1

if __name__ == "__main__":
    exit(main()) 