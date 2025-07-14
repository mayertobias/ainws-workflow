#!/usr/bin/env python3
"""
Script to check available Essentia TensorFlow algorithms and models
"""

import sys
import os
from pathlib import Path

def check_tensorflow_algorithms():
    """Check what TensorFlow algorithms are available in Essentia"""
    print("🤖 Checking Essentia TensorFlow Algorithms")
    print("=" * 60)
    
    try:
        import essentia
        import essentia.standard as es
        print(f"✅ Essentia version: {essentia.__version__}")
        
        # List of known TensorFlow-based algorithms in Essentia
        tensorflow_algorithms = [
            'TensorflowPredict',
            'TensorflowPredict2D',
            'TensorflowPredictVGGish',
            'TensorflowPredictMusiCNN',
            'TensorflowPredictTempoCNN',
            'TensorflowPredictEffnetDiscogs',
            'TensorflowPredictMAEST',
        ]
        
        print("\n🔍 Available TensorFlow Algorithms:")
        available_algorithms = []
        
        for algorithm in tensorflow_algorithms:
            try:
                # Try to get the algorithm class
                algo_class = getattr(es, algorithm, None)
                if algo_class is not None:
                    print(f"  ✅ {algorithm}: Available")
                    available_algorithms.append(algorithm)
                    
                    # Try to get algorithm info if possible
                    try:
                        # Some algorithms might have documentation or info
                        if hasattr(algo_class, '__doc__') and algo_class.__doc__:
                            doc_first_line = algo_class.__doc__.split('\n')[0].strip()
                            if doc_first_line:
                                print(f"     📝 {doc_first_line}")
                    except:
                        pass
                        
                else:
                    print(f"  ❌ {algorithm}: Not available")
            except Exception as e:
                print(f"  ❌ {algorithm}: Error checking - {e}")
        
        return available_algorithms
        
    except ImportError as e:
        print(f"❌ Failed to import Essentia: {e}")
        return []

def check_tensorflow_models():
    """Check for available TensorFlow model files"""
    print(f"\n🗂️ Checking for TensorFlow Model Files")
    print("=" * 40)
    
    # Common model paths
    model_paths = [
        '/app/models/essentia',
        './models/essentia',
        './models',
        os.path.expanduser('~/.essentia/models'),
        '/usr/local/share/essentia/models'
    ]
    
    model_extensions = ['.pb', '.json', '.h5', '.onnx']
    found_models = []
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"📁 Checking: {model_path}")
            try:
                for file_path in Path(model_path).rglob('*'):
                    if file_path.is_file() and file_path.suffix in model_extensions:
                        print(f"  ✅ {file_path.name} ({file_path.suffix})")
                        found_models.append(str(file_path))
            except Exception as e:
                print(f"  ❌ Error scanning {model_path}: {e}")
        else:
            print(f"📁 {model_path}: Not found")
    
    if not found_models:
        print("⚠️  No model files found in common locations")
    
    return found_models

def test_tensorflow_algorithms():
    """Test available TensorFlow algorithms with dummy data"""
    print(f"\n🧪 Testing TensorFlow Algorithms")
    print("=" * 40)
    
    try:
        import essentia.standard as es
        import numpy as np
        
        # Test TensorflowPredict2D if available
        if hasattr(es, 'TensorflowPredict2D'):
            print("🔬 Testing TensorflowPredict2D...")
            try:
                # This will fail without a model file, but we can see the error
                predictor = es.TensorflowPredict2D()
                print("  ✅ TensorflowPredict2D can be instantiated")
            except Exception as e:
                print(f"  ⚠️  TensorflowPredict2D error (expected without model): {e}")
        
        # Test TensorflowPredictMusiCNN if available
        if hasattr(es, 'TensorflowPredictMusiCNN'):
            print("🔬 Testing TensorflowPredictMusiCNN...")
            try:
                predictor = es.TensorflowPredictMusiCNN()
                print("  ✅ TensorflowPredictMusiCNN can be instantiated")
            except Exception as e:
                print(f"  ⚠️  TensorflowPredictMusiCNN error (expected without model): {e}")
        
        # Test other algorithms similarly
        algorithms_to_test = ['TensorflowPredictVGGish', 'TensorflowPredictTempoCNN']
        
        for algo_name in algorithms_to_test:
            if hasattr(es, algo_name):
                print(f"🔬 Testing {algo_name}...")
                try:
                    predictor = getattr(es, algo_name)()
                    print(f"  ✅ {algo_name} can be instantiated")
                except Exception as e:
                    print(f"  ⚠️  {algo_name} error (expected without model): {e}")
        
    except Exception as e:
        print(f"❌ Error testing algorithms: {e}")

def check_essentia_tensorflow_version():
    """Check TensorFlow version used by Essentia"""
    print(f"\n🔧 TensorFlow Integration Details")
    print("=" * 40)
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Check if TensorFlow can see any GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🎮 GPU devices available: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        else:
            print("💻 Running on CPU (no GPU devices found)")
            
    except ImportError:
        print("❌ TensorFlow not available in Python environment")
    except Exception as e:
        print(f"❌ Error checking TensorFlow: {e}")

def download_models_info():
    """Provide information about downloading Essentia models"""
    print(f"\n📥 Model Download Information")
    print("=" * 40)
    
    print("To use TensorFlow algorithms, you need model files:")
    print()
    print("1. 🌐 Official Essentia Models:")
    print("   https://essentia.upf.edu/models/")
    print()
    print("2. 📋 Common Models for Music Analysis:")
    print("   • MusiCNN models (genre, mood, instrument classification)")
    print("   • EffnetDiscogs models (music style classification)")
    print("   • TempoCNN models (tempo estimation)")
    print("   • VGGish models (audio embeddings)")
    print()
    print("3. 📁 Model Installation:")
    print("   • Create directory: mkdir -p ./models/essentia")
    print("   • Download .pb files to this directory")
    print("   • Some models also need .json metadata files")
    print()
    print("4. 🔧 Environment Variable:")
    print("   export ESSENTIA_MODELS_PATH=/path/to/models")

def main():
    """Main function to check all TensorFlow capabilities"""
    print("🎼 Essentia TensorFlow Capabilities Check")
    print("=" * 60)
    
    # Check available algorithms
    available_algorithms = check_tensorflow_algorithms()
    
    # Check for model files
    found_models = check_tensorflow_models()
    
    # Test algorithms
    test_tensorflow_algorithms()
    
    # Check TensorFlow version
    check_essentia_tensorflow_version()
    
    # Provide download info
    download_models_info()
    
    # Summary
    print(f"\n{'='*60}")
    print("🎯 Summary:")
    print(f"   • Available TF algorithms: {len(available_algorithms)}")
    print(f"   • Found model files: {len(found_models)}")
    
    if available_algorithms:
        print(f"   • Ready for: {', '.join(available_algorithms)}")
    
    if not found_models:
        print("   ⚠️  No models found - download models to use TF algorithms")
    
    print(f"\n💡 Next Steps:")
    if available_algorithms and not found_models:
        print("   1. Download model files from https://essentia.upf.edu/models/")
        print("   2. Place them in ./models/essentia/ directory")
        print("   3. Test with actual audio files")
    elif available_algorithms and found_models:
        print("   1. Test TensorFlow algorithms with real audio files")
        print("   2. Integrate into your audio analysis pipeline")
    else:
        print("   1. Check Essentia installation")
        print("   2. Verify TensorFlow integration")

if __name__ == "__main__":
    exit(main()) 