# 🛠️ Ensemble Training DAGs - Fixes and Usage Guide

## 🔍 Issues Fixed

### 1. **API Endpoint Mismatches** ❌➡️✅
- **Problem**: DAGs were calling non-existent endpoints
- **Fixed Endpoints**:
  - `POST /pipeline/start` ➡️ `POST /pipeline/train`
  - `GET /pipeline/{pipeline_id}/status` ➡️ `GET /pipeline/status/{pipeline_id}`

### 2. **Payload Format Issues** ❌➡️✅
- **Problem**: Request payload didn't match `TrainingRequest` model
- **Fixed**: Restructured payload to match actual API schema
- **Key Changes**:
  - Moved complex parameters into `parameters` field
  - Simplified top-level structure
  - Added proper `skip_feature_agreement` flag

### 3. **Hardcoded Dataset Paths** ❌➡️✅
- **Problem**: Absolute paths that may not exist
- **Fixed**: Use relative paths resolved by ML service
- **Benefits**: Automatic path resolution across environments

### 4. **Status Response Handling** ❌➡️✅
- **Problem**: Expected old response format
- **Fixed**: Compatible with new `PipelineStatus` model
- **Improvements**: Better error handling and progress tracking

## 📋 Fixed DAGs

### 1. `multimodal_ensemble_training_pipeline.py`
- ✅ Fixed API endpoints
- ✅ Corrected payload format
- ✅ Flexible dataset path handling
- ✅ Improved status monitoring

### 2. `audio_ensemble_training_pipeline.py`
- ✅ Fixed API endpoints
- ✅ Corrected payload format
- ✅ Flexible dataset path handling
- ✅ Improved status monitoring

### 3. `trigger_multimodal_ensemble_training.py`
- ✅ Updated configuration format
- ✅ Flexible dataset path
- ✅ Better documentation

### 4. `trigger_audio_ensemble_training.py`
- ✅ Updated configuration format
- ✅ Flexible dataset path
- ✅ Better documentation

## 🚀 How to Use

### Quick Start

1. **Start the ML Training Service**:
   ```bash
   cd workflow-ml-train
   docker-compose up -d
   ```

2. **Verify Service Health**:
   ```bash
   curl http://localhost:8005/health
   ```

3. **Trigger Audio Ensemble Training**:
   ```bash
   # In Airflow UI
   # Navigate to: trigger_audio_ensemble_training
   # Click "Trigger DAG"
   ```

4. **Trigger Multimodal Ensemble Training**:
   ```bash
   # In Airflow UI
   # Navigate to: trigger_multimodal_ensemble_training
   # Click "Trigger DAG"
   ```

### Advanced Configuration

#### Custom Dataset Path
```python
# In DAG configuration
CUSTOM_CONFIG = {
    "dataset_path": "your_custom_dataset.csv",  # Will be auto-resolved
    "training_id": "custom_training_001",
    # ... other parameters
}
```

#### Custom Features
```python
# Modify agreed_features in trigger DAGs
"agreed_features": {
    "audio": [
        "audio_tempo",
        "audio_energy",
        "audio_valence"
        # Add/remove features as needed
    ],
    "content": [
        "lyrics_sentiment_positive",
        "lyrics_complexity_score"
        # Add/remove features as needed
    ]
}
```

#### Ensemble Configuration
```python
"ensemble_config": {
    "random_forest": {
        "n_estimators": 200,  # More trees
        "max_depth": 15,      # Deeper trees
        "random_state": 42
    },
    "xgboost": {
        "n_estimators": 150,
        "max_depth": 8,
        "learning_rate": 0.05  # Slower learning
    }
}
```

## 📊 Expected Outputs

### MLflow Experiments
- **Location**: http://localhost:5001
- **Experiments**: 
  - `audio_ensemble_experiments_YYYYMMDD`
  - `multimodal_ensemble_experiments_YYYYMMDD`

### Model Artifacts
- Random Forest model
- XGBoost model
- Ensemble voting classifier
- SHAP analysis plots
- Feature importance rankings

### SHAP Analysis
- **Summary Plots**: Feature importance distributions
- **Bar Plots**: Feature importance rankings
- **Waterfall Plots**: Individual prediction explanations
- **Cross-Modal Analysis**: Audio vs Content feature comparison (multimodal only)

## 🔧 Troubleshooting

### Common Issues

#### 1. Dataset Not Found
```
Error: Dataset 'filtered_xxx.csv' not found
```
**Solution**: 
- Check if dataset exists in `shared-data/training_data/filtered/`
- Use dataset creation service: `POST /datasets/create-filtered`
- Verify path resolver configuration

#### 2. ML Service Unavailable
```
Error: ML training service not available
```
**Solution**:
- Start ML training service: `docker-compose up -d`
- Check service health: `curl http://localhost:8005/health`
- Verify Docker network connectivity

#### 3. Training Timeout
```
Error: Training did not complete within 60 minutes
```
**Solution**:
- Check training progress: `GET /pipeline/status/{pipeline_id}`
- Increase timeout in DAG configuration
- Check MLflow for partial results

#### 4. Feature Agreement Issues
```
Error: Invalid features specified
```
**Solution**:
- Use `skip_feature_agreement: true` (default in fixed DAGs)
- Verify feature names match service capabilities
- Check feature discovery results

### Debug Commands

```bash
# Check ML service health
curl http://localhost:8005/health

# List available strategies
curl http://localhost:8005/pipeline/strategies

# Check pipeline status
curl http://localhost:8005/pipeline/status/{pipeline_id}

# List experiments
curl http://localhost:8005/pipeline/experiments

# Get SHAP analysis
curl http://localhost:8005/pipeline/{pipeline_id}/shap-analysis
```

## 📈 Performance Tips

1. **Dataset Size**: Start with smaller datasets for testing
2. **Feature Selection**: Use fewer features for faster training
3. **Parallel Processing**: Ensemble training uses multiple cores
4. **Caching**: Audio/content features are cached for reuse
5. **Memory**: Ensure sufficient memory for large datasets

## 🔗 Related Services

- **Audio Service**: http://localhost:8001
- **Content Service**: http://localhost:8002
- **ML Training Service**: http://localhost:8005
- **MLflow UI**: http://localhost:5001
- **Airflow UI**: http://localhost:8080

## 📚 Next Steps

1. **Monitor Training**: Use MLflow UI to track progress
2. **Analyze Results**: Review SHAP analysis for insights
3. **Model Deployment**: Use trained models for predictions
4. **Feature Engineering**: Experiment with different feature combinations
5. **Hyperparameter Tuning**: Optimize ensemble configurations 