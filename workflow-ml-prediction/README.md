# Workflow ML Prediction Service

## Overview

The Workflow ML Prediction Service provides intelligent song popularity prediction using trained models from the ML Training Service. It features smart model selection, comprehensive explanations, and high-performance prediction capabilities.

## 🎯 Key Features

### Smart Prediction
- **Auto-Model Selection**: Automatically selects the best model based on available features
- **Multimodal Support**: Uses audio-only or multimodal models based on data availability
- **Confidence Scoring**: Provides prediction confidence based on model performance and feature coverage
- **Detailed Explanations**: Explains predictions with feature importance and contributing factors

### Performance & Scalability
- **Model Registry Integration**: Seamlessly loads models from training service registry
- **Intelligent Caching**: Caches models and predictions for optimal performance
- **Batch Processing**: Efficient batch prediction with summary analytics
- **Health Monitoring**: Comprehensive health checks and model status monitoring

### Production Ready
- **API Versioning**: Both legacy and smart API endpoints for compatibility
- **Error Handling**: Graceful degradation when models or features are unavailable
- **Async Processing**: High-performance async prediction pipeline
- **Model Hot-Loading**: Automatic model updates without service restart

## 🚀 Quick Start

### 1. Start the Service
```bash
cd workflow-ml-prediction
python -m uvicorn app.main:app --host 0.0.0.0 --port 8004 --reload
```

### 2. Access the API Documentation
- Swagger UI: http://localhost:8004/docs
- ReDoc: http://localhost:8004/redoc

### 3. Make Your First Prediction
```bash
# Demo prediction with sample data
curl -X POST http://localhost:8004/predict/smart/demo/sample-prediction
```

## 📊 API Endpoints

### Smart Prediction
- `POST /predict/smart/single` - Smart prediction for single song
- `POST /predict/smart/batch` - Smart batch predictions
- `GET /predict/smart/models` - Get smart model information
- `GET /predict/smart/health` - Smart prediction service health
- `POST /predict/smart/update-models` - Force model registry update

### Model Management
- `GET /predict/smart/features/{model_type}` - Get required features for model
- `POST /predict/smart/validate-features` - Validate song features
- `GET /predict/smart/registry` - Get model registry information
- `POST /predict/smart/demo/sample-prediction` - Demo prediction

### Legacy Prediction (Backwards Compatibility)
- `POST /predict/single` - Traditional single prediction
- `POST /predict/batch` - Traditional batch prediction
- `GET /predict/models` - List available models
- `POST /predict/validate-features` - Validate features

## 🧠 Smart Prediction Examples

### Single Song Prediction
```python
import httpx
import asyncio

async def predict_song():
    # Song features (mix of audio and lyrics)
    song_features = {
        # Audio features
        "tempo": 128.0,
        "energy": 0.8,
        "danceability": 0.9,
        "valence": 0.7,
        "loudness": -5.0,
        "speechiness": 0.1,
        "acousticness": 0.2,
        "instrumentalness": 0.05,
        "liveness": 0.3,
        "genre_confidence": 0.85,
        
        # Lyrics features (if available)
        "lyrics_word_count": 120,
        "lyrics_sentiment_score": 0.6,
        "lyrics_complexity_score": 0.5,
        "lyrics_theme_diversity": 0.7
    }
    
    response = await httpx.post(
        "http://localhost:8004/predict/smart/single",
        json={
            "song_features": song_features,
            "explain_prediction": True
        }
    )
    
    result = response.json()
    print(f"Prediction: {result['prediction']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Model used: {result['model_used']}")
    
    # Print explanation
    if result.get('explanation'):
        print("\nTop influencing features:")
        for feature in result['explanation']['top_features']:
            print(f"  - {feature['feature']}: {feature['value']} "
                  f"(importance: {feature['importance']:.3f})")

asyncio.run(predict_song())
```

### Batch Prediction
```python
async def predict_multiple_songs():
    songs_data = [
        {
            "file_path": "song1.mp3",
            "artist": "Artist 1",
            "title": "Hit Song",
            "tempo": 120,
            "energy": 0.9,
            "danceability": 0.8,
            "valence": 0.7,
            "lyrics_sentiment_score": 0.5
        },
        {
            "file_path": "song2.mp3", 
            "artist": "Artist 2",
            "title": "Ballad",
            "tempo": 70,
            "energy": 0.3,
            "danceability": 0.2,
            "valence": 0.3,
            # No lyrics features - will use audio-only model
        }
    ]
    
    response = await httpx.post(
        "http://localhost:8004/predict/smart/batch",
        json={
            "songs_data": songs_data,
            "explain_predictions": False  # Faster for batch
        }
    )
    
    result = response.json()
    
    print(f"Batch Summary:")
    print(f"  Total songs: {result['summary']['total_songs']}")
    print(f"  Successful: {result['summary']['successful_predictions']}")
    print(f"  Average prediction: {result['summary']['average_prediction']}")
    print(f"  Models used: {result['summary']['models_used']}")
    
    for i, prediction in enumerate(result['predictions']):
        if 'error' not in prediction:
            song_info = prediction['song_info']
            print(f"\n{song_info['title']} by {song_info['artist']}:")
            print(f"  Prediction: {prediction['prediction']:.3f}")
            print(f"  Model: {prediction['model_used']}")

asyncio.run(predict_multiple_songs())
```

### Feature Validation
```python
async def validate_features():
    # Check if your features are sufficient for prediction
    features = {
        "tempo": 120,
        "energy": 0.8,
        "danceability": 0.7
        # Missing some features
    }
    
    response = await httpx.post(
        "http://localhost:8004/predict/smart/validate-features",
        json=features
    )
    
    result = response.json()
    
    print(f"Can predict: {result['can_predict']}")
    print(f"Usable models: {result['usable_models']}")
    print(f"Recommended model: {result['recommended_model']}")
    
    # Check what's missing for each model
    for model, validation in result['validation_results'].items():
        if not validation['valid']:
            print(f"{model} missing: {validation['missing_features']}")

asyncio.run(validate_features())
```

## 🔧 Configuration

### Environment Variables
```bash
# Service Configuration
PORT=8004
DEBUG=true

# Model Registry
MODELS_DIR=/app/models
MODEL_REGISTRY_PATH=/app/models/model_registry.json

# Performance Settings
MAX_CACHED_MODELS=10
PREDICTION_TIMEOUT_SECONDS=5
MAX_BATCH_SIZE=1000
MAX_CONCURRENT_PREDICTIONS=100

# Service Discovery
WORKFLOW_ML_TRAINING_URL=http://workflow-ml-training:8003

# Caching
REDIS_URL=redis://localhost:6379/3
CACHE_TTL_SECONDS=3600
```

### Model Registry Integration
The service automatically loads models from the training service registry:

```json
{
  "latest_training": "smart_training_20241201_143022",
  "available_models": {
    "audio_only": {
      "model_path": "/app/models/audio_only_20241201_143022/model.joblib",
      "metadata_path": "/app/models/audio_only_20241201_143022/metadata.json",
      "features_path": "/app/models/audio_only_20241201_143022/features.json",
      "performance": {
        "r2_score": 0.75,
        "mse": 0.12
      }
    },
    "multimodal": {
      "model_path": "/app/models/multimodal_20241201_143022/model.joblib",
      "performance": {
        "r2_score": 0.85,
        "mse": 0.08
      }
    }
  }
}
```

## 🎵 Feature Requirements

### Audio Features (Always Required)
| Feature | Description | Range |
|---------|-------------|-------|
| `tempo` | Song tempo (BPM) | 50-200 |
| `energy` | Energy level | 0-1 |
| `danceability` | Danceability score | 0-1 |
| `valence` | Musical positivity | 0-1 |
| `loudness` | Loudness in dB | -60 to 0 |
| `speechiness` | Speech content | 0-1 |
| `acousticness` | Acoustic ratio | 0-1 |
| `instrumentalness` | Vocal absence | 0-1 |
| `liveness` | Live audience | 0-1 |
| `genre_confidence` | Genre certainty | 0-1 |

### Lyrics Features (Optional - Enables Multimodal)
| Feature | Description | Range |
|---------|-------------|-------|
| `lyrics_word_count` | Number of words | 0+ |
| `lyrics_sentiment_score` | Emotional sentiment | -1 to 1 |
| `lyrics_complexity_score` | Linguistic complexity | 0-1 |
| `lyrics_theme_diversity` | Theme variety | 0-1 |
| `lyrics_positive_emotion` | Positive emotion | 0-1 |
| `lyrics_love_theme` | Love theme presence | 0-1 |
| `lyrics_party_theme` | Party theme presence | 0-1 |

## 🏗️ Smart Model Selection Logic

### Selection Algorithm
```python
def select_best_model(song_features):
    # 1. Check feature availability for each model
    usable_models = []
    for model_type in available_models:
        if has_required_features(song_features, model_type):
            performance = get_model_performance(model_type)
            usable_models.append((model_type, performance))
    
    # 2. Select highest performing model
    if usable_models:
        return max(usable_models, key=lambda x: x[1])
    else:
        raise ValueError("No compatible models found")
```

### Model Priority
1. **Multimodal Model** (if lyrics features available and performs better)
2. **Audio-Only Model** (fallback, works with any song)

### Confidence Calculation
```python
confidence = base_confidence * feature_coverage
# Where:
# base_confidence = model's R² score
# feature_coverage = available_features / required_features
```

## 📈 Performance & Monitoring

### Health Checks
```bash
# Service health
curl http://localhost:8004/health

# Smart prediction health
curl http://localhost:8004/predict/smart/health

# Model information
curl http://localhost:8004/predict/smart/models
```

### Performance Metrics
```bash
# Get prediction metrics
curl http://localhost:8004/predict/metrics

# Cache statistics
curl http://localhost:8004/predict/cache/stats
```

### Model Updates
```bash
# Force model reload from registry
curl -X POST http://localhost:8004/predict/smart/update-models
```

## 🔍 Prediction Explanations

### Feature Importance
Each prediction includes feature importance scores:
```json
{
  "top_influencing_features": [
    {
      "feature": "energy",
      "importance": 0.245,
      "value": 0.8,
      "description": "Energy level of the song (0-1)"
    },
    {
      "feature": "danceability", 
      "importance": 0.189,
      "value": 0.9,
      "description": "How suitable the song is for dancing (0-1)"
    }
  ]
}
```

### Prediction Factors
Contextual analysis of what drove the prediction:
```json
{
  "prediction_factors": {
    "energy": "High energy contributes positively to prediction",
    "mood": "Positive mood/valence boosts prediction", 
    "danceability": "High danceability increases appeal"
  }
}
```

## 🚨 Error Handling

### Graceful Degradation
- **No Models Available**: Returns clear error message
- **Missing Features**: Suggests compatible models and missing features
- **Model Loading Errors**: Falls back to cached models or alternative models
- **Prediction Timeouts**: Returns partial results with warnings

### Common Error Scenarios
```python
# Insufficient features
{
  "error": "Cannot use multimodal model. Missing features: ['lyrics_sentiment_score']",
  "available_models": ["audio_only"],
  "recommendation": "Use audio_only model or provide lyrics features"
}

# No compatible models
{
  "error": "No models can be used with features: ['tempo', 'energy']", 
  "required_features": {
    "audio_only": ["tempo", "energy", "danceability", "valence", "loudness"],
    "multimodal": ["tempo", "energy", "danceability", "valence", "loudness", "lyrics_sentiment_score"]
  }
}
```

## 🔄 Integration Examples

### Complete Workflow Integration
```python
async def full_prediction_workflow():
    # 1. Get song features from analysis services
    audio_features = await get_audio_features("song.mp3")
    lyrics_features = await get_lyrics_features("lyrics.txt")
    
    # 2. Combine features
    combined_features = {**audio_features, **lyrics_features}
    
    # 3. Validate features
    validation = await httpx.post(
        "http://localhost:8004/predict/smart/validate-features",
        json=combined_features
    )
    
    if not validation.json()["can_predict"]:
        print("Insufficient features for prediction")
        return
    
    # 4. Make prediction
    prediction = await httpx.post(
        "http://localhost:8004/predict/smart/single",
        json={
            "song_features": combined_features,
            "explain_prediction": True
        }
    )
    
    return prediction.json()
```

### Real-time Prediction Pipeline
```python
async def realtime_prediction_pipeline(song_queue):
    async for song_data in song_queue:
        try:
            prediction = await httpx.post(
                "http://localhost:8004/predict/smart/single",
                json={"song_features": song_data},
                timeout=2.0  # Fast timeout for real-time
            )
            
            result = prediction.json()
            await send_to_analytics(song_data, result)
            
        except httpx.TimeoutException:
            await handle_prediction_timeout(song_data)
        except Exception as e:
            await handle_prediction_error(song_data, e)
```

## 🛠️ Troubleshooting

### Common Issues

1. **Models not loading**
   ```bash
   # Check model registry
   curl http://localhost:8004/predict/smart/registry
   
   # Force model update
   curl -X POST http://localhost:8004/predict/smart/update-models
   ```

2. **Low prediction confidence**
   - Ensure all required features are provided
   - Check feature quality and realistic ranges
   - Verify model performance in training service

3. **Slow predictions**
   - Enable Redis caching
   - Use batch endpoints for multiple predictions
   - Check model loading times

### Performance Optimization

```bash
# Optimize for high throughput
export MAX_CACHED_MODELS=20
export MAX_CONCURRENT_PREDICTIONS=200
export REDIS_URL=redis://redis-cluster:6379

# Optimize for low latency  
export PREDICTION_TIMEOUT_SECONDS=1
export CACHE_TTL_SECONDS=7200
```

## 📊 Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Mount models directory
VOLUME ["/app/models"]

EXPOSE 8004
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
```

### Health Check Configuration
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8004/predict/smart/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

## 📄 License

This project is part of the Workflow microservices ecosystem. 