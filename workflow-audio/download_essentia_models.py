#!/usr/bin/env python3
"""
Download and test popular Essentia TensorFlow models
"""

import os
import urllib.request
import sys
import tempfile
import wave
import numpy as np
from pathlib import Path

def create_models_directory():
    """Create the models directory"""
    models_dir = Path("./models/essentia")
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created models directory: {models_dir}")
    return models_dir

def download_model(url, filename, models_dir):
    """Download a model file"""
    model_path = models_dir / filename
    
    if model_path.exists():
        print(f"✅ {filename} already exists")
        return str(model_path)
    
    print(f"⬇️  Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, str(model_path))
        print(f"✅ Downloaded {filename}")
        return str(model_path)
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return None

def download_popular_models():
    """Download some popular Essentia models"""
    print("📥 Downloading Popular Essentia Models")
    print("=" * 50)
    
    models_dir = create_models_directory()
    
    # Popular models from Essentia
    models = [
        {
            'name': 'MusiCNN MSD',
            'url': 'https://essentia.upf.edu/models/feature-extractors/msd/msd-musicnn-1.pb',
            'filename': 'msd-musicnn-1.pb',
            'description': 'MusiCNN embeddings trained on Million Song Dataset'
        },
        {
            'name': 'Genre Discogs400',
            'url': 'https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb',
            'filename': 'genre_discogs400-discogs-effnet-1.pb', 
            'description': 'Genre classification with 400 genres'
        },
        {
            'name': 'Genre Discogs400 Labels',
            'url': 'https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json',
            'filename': 'genre_discogs400-discogs-effnet-1.json',
            'description': 'Genre labels for the genre classification model'
        },
        {
            'name': 'Mood Happy',
            'url': 'https://essentia.upf.edu/models/classification-heads/mood_happy/mood_happy-msd-musicnn-1.pb',
            'filename': 'mood_happy-msd-musicnn-1.pb',
            'description': 'Happy mood classification'
        },
        {
            'name': 'Mood Sad',
            'url': 'https://essentia.upf.edu/models/classification-heads/mood_sad/mood_sad-msd-musicnn-1.pb',
            'filename': 'mood_sad-msd-musicnn-1.pb',
            'description': 'Sad mood classification'
        }
    ]
    
    downloaded_models = []
    
    for model in models:
        print(f"\n🎯 {model['name']}")
        print(f"   📝 {model['description']}")
        model_path = download_model(model['url'], model['filename'], models_dir)
        if model_path:
            downloaded_models.append(model_path)
    
    return downloaded_models

def test_musicnn_model(model_path, audio_file):
    """Test MusiCNN model with audio file"""
    print(f"\n🎵 Testing MusiCNN model...")
    
    try:
        import essentia.standard as es
        
        # Load audio
        loader = es.MonoLoader(filename=audio_file, sampleRate=16000)
        audio = loader()
        
        print(f"   Audio loaded: {len(audio)} samples")
        
        # Create MusiCNN predictor
        musicnn = es.TensorflowPredictMusiCNN(
            graphFilename=model_path,
            output='model/dense/BiasAdd'
        )
        
        # Get embeddings
        embeddings = musicnn(audio)
        
        print(f"✅ MusiCNN embeddings extracted!")
        print(f"   Embedding shape: {embeddings.shape}")
        print(f"   First 5 values: {embeddings[:5]}")
        
        return embeddings
        
    except Exception as e:
        print(f"❌ MusiCNN test failed: {e}")
        return None

def test_genre_classification(model_path, json_path, audio_file):
    """Test genre classification"""
    print(f"\n🎪 Testing Genre Classification...")
    
    try:
        import essentia.standard as es
        import json
        
        # Load genre labels
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                genre_labels = json.load(f)
            print(f"   Loaded {len(genre_labels)} genre labels")
        else:
            print("   ⚠️  Genre labels not found, using indices")
            genre_labels = None
        
        # Load audio
        loader = es.MonoLoader(filename=audio_file, sampleRate=16000)
        audio = loader()
        
        # Get MusiCNN embeddings first
        embedding_model = es.TensorflowPredictMusiCNN(
            graphFilename='./models/essentia/msd-musicnn-1.pb',
            output='model/dense/BiasAdd'
        )
        embeddings = embedding_model(audio)
        
        # Genre classification
        genre_predictor = es.TensorflowPredict2D(
            graphFilename=model_path,
            input='serving_default_model_Placeholder',
            output='PartitionedCall'
        )
        
        predictions = genre_predictor(embeddings)
        
        print(f"✅ Genre classification complete!")
        print(f"   Predictions shape: {predictions.shape}")
        
        # Show top predictions
        if predictions.size > 0:
            # Get top 5 predictions
            top_indices = np.argsort(predictions.flatten())[-5:][::-1]
            
            print("   🎯 Top 5 Genre Predictions:")
            for i, idx in enumerate(top_indices):
                prob = predictions.flatten()[idx]
                if genre_labels and idx < len(genre_labels):
                    genre = genre_labels[idx]
                    print(f"     {i+1}. {genre}: {prob:.3f}")
                else:
                    print(f"     {i+1}. Genre {idx}: {prob:.3f}")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Genre classification failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_mood_classification(model_path, audio_file, mood_name):
    """Test mood classification"""
    print(f"\n😊 Testing {mood_name} Mood Classification...")
    
    try:
        import essentia.standard as es
        
        # Load audio
        loader = es.MonoLoader(filename=audio_file, sampleRate=16000)
        audio = loader()
        
        # Get MusiCNN embeddings first
        embedding_model = es.TensorflowPredictMusiCNN(
            graphFilename='./models/essentia/msd-musicnn-1.pb',
            output='model/dense/BiasAdd'
        )
        embeddings = embedding_model(audio)
        
        # Mood classification
        mood_predictor = es.TensorflowPredict2D(
            graphFilename=model_path,
            input='serving_default_model_Placeholder',
            output='PartitionedCall'
        )
        
        predictions = mood_predictor(embeddings)
        
        print(f"✅ {mood_name} mood classification complete!")
        
        if predictions.size > 0:
            mood_score = predictions.flatten()[0] if len(predictions.flatten()) > 0 else 0
            print(f"   🎯 {mood_name} Score: {mood_score:.3f}")
            
            if mood_score > 0.5:
                print(f"   🎉 Audio classified as {mood_name.lower()}!")
            else:
                print(f"   🎵 Audio not strongly {mood_name.lower()}")
        
        return predictions
        
    except Exception as e:
        print(f"❌ {mood_name} mood classification failed: {e}")
        return None

def create_test_audio():
    """Create test audio for model testing"""
    print("🎵 Creating test audio for model testing...")
    
    sample_rate = 16000  # Most models expect 16kHz
    duration = 10  # 10 seconds for better analysis
    
    # Create a more complex audio signal
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Combine multiple frequencies for richer audio
    audio_data = (
        np.sin(2 * np.pi * 440 * t) * 0.3 +  # A4
        np.sin(2 * np.pi * 554.37 * t) * 0.2 +  # C#5
        np.sin(2 * np.pi * 659.25 * t) * 0.2 +  # E5
        np.sin(2 * np.pi * 880 * t) * 0.1       # A5
    )
    
    # Add some envelope and dynamics
    envelope = np.exp(-t * 0.1)  # Decay envelope
    audio_data = audio_data * envelope
    
    # Create uploads directory
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    # Save as WAV file
    test_file = uploads_dir / "test_model_audio.wav"
    audio_data_int = (audio_data * 32767).astype(np.int16)
    
    with wave.open(str(test_file), 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data_int.tobytes())
    
    print(f"✅ Created test audio: {test_file}")
    return str(test_file)

def main():
    """Main function"""
    print("🤖 Essentia TensorFlow Models Download & Test")
    print("=" * 60)
    
    # Download models
    downloaded_models = download_popular_models()
    
    if downloaded_models:
        print(f"\n✅ Downloaded {len(downloaded_models)} models")
        
        # Create test audio
        test_audio = create_test_audio()
        
        # Test models
        print(f"\n🧪 Testing Models with Audio")
        print("=" * 40)
        
        # Test MusiCNN embeddings
        musicnn_path = "./models/essentia/msd-musicnn-1.pb"
        if os.path.exists(musicnn_path):
            test_musicnn_model(musicnn_path, test_audio)
        
        # Test genre classification
        genre_model_path = "./models/essentia/genre_discogs400-discogs-effnet-1.pb"
        genre_labels_path = "./models/essentia/genre_discogs400-discogs-effnet-1.json"
        if os.path.exists(genre_model_path):
            test_genre_classification(genre_model_path, genre_labels_path, test_audio)
        
        # Test mood classification
        mood_models = [
            ("./models/essentia/mood_happy-msd-musicnn-1.pb", "Happy"),
            ("./models/essentia/mood_sad-msd-musicnn-1.pb", "Sad")
        ]
        
        for mood_path, mood_name in mood_models:
            if os.path.exists(mood_path):
                test_mood_classification(mood_path, test_audio, mood_name)
        
        print(f"\n🎯 Testing Complete!")
        print("   You can now use these models in your audio analysis pipeline")
        
    else:
        print("❌ No models were downloaded successfully")
    
    return 0

if __name__ == "__main__":
    exit(main()) 