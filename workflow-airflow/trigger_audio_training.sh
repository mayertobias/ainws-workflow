#!/bin/bash
# Trigger Audio-Only Feature Agreement Training

echo "🚀 Triggering Feature Agreement Training Pipeline..."

curl -X POST "http://localhost:8080/api/v1/dags/feature_agreement_training_pipeline/dagRuns" \
  -H "Content-Type: application/json" \
  -d '{
    "conf": {
      "services": ["audio"],
      "dataset_path": "/app/shared-data/training_data/filtered/filtered_audio_only_corrected_20250621_180350.csv",
      "training_id": "audio_only_feature_agreement_'$(date +%Y%m%d_%H%M%S)'",
      "agreed_features": {
        "audio": [
          "audio_tempo",
          "audio_energy", 
          "audio_valence",
          "audio_danceability",
          "audio_loudness"
        ]
      }
    }
  }'

echo "✅ Training pipeline triggered!"
echo "🌐 Monitor progress at: http://localhost:8080"
echo "📊 Track experiments at: http://localhost:5001"
