#!/bin/bash
echo "🚀 Triggering Feature Agreement Pipeline via API..."

curl -X POST "http://localhost:8080/api/v1/dags/feature_agreement_training_pipeline/dagRuns" \
  -H "Content-Type: application/json" \
  -u "admin:admin123" \
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

echo -e "\n✅ Pipeline triggered!"
echo "🌐 Monitor at: http://localhost:8080"
