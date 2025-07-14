"""
Feature Vector Manager

Handles loading, saving, and validating pre-existing feature vectors
from shared volumes (like training data location).

Supports:
- Loading feature vectors from JSON/YAML files
- Validating feature vector structure
- Managing feature vector presets
- Converting between different feature vector formats
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class FeatureVectorManager:
    """Manages pre-existing feature vectors from shared storage"""
    
    def __init__(self, shared_data_path: str = "./shared-data"):
        self.shared_data_path = Path(shared_data_path)
        self.feature_vectors_path = self.shared_data_path / "feature_vectors"
        self.presets_path = self.feature_vectors_path / "presets"
        
        # Create directories if they don't exist
        self.feature_vectors_path.mkdir(parents=True, exist_ok=True)
        self.presets_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Feature Vector Manager initialized with path: {self.feature_vectors_path}")
    
    def load_feature_vector(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load feature vector from file.
        
        Args:
            file_path: Relative path to feature vector file (JSON or YAML)
            
        Returns:
            Feature vector dictionary or None if not found
        """
        try:
            # Handle relative paths
            if not file_path.startswith('/'):
                full_path = self.feature_vectors_path / file_path
            else:
                full_path = Path(file_path)
            
            if not full_path.exists():
                logger.warning(f"⚠️ Feature vector file not found: {full_path}")
                return None
            
            # Load based on file extension
            if full_path.suffix.lower() == '.json':
                with open(full_path, 'r') as f:
                    feature_vector = json.load(f)
            elif full_path.suffix.lower() in ['.yaml', '.yml']:
                with open(full_path, 'r') as f:
                    feature_vector = yaml.safe_load(f)
            else:
                logger.error(f"❌ Unsupported feature vector format: {full_path.suffix}")
                return None
            
            logger.info(f"✅ Loaded feature vector from {full_path}")
            return feature_vector
            
        except Exception as e:
            logger.error(f"❌ Failed to load feature vector from {file_path}: {e}")
            return None
    
    def save_feature_vector(self, feature_vector: Dict[str, Any], file_name: str) -> bool:
        """
        Save feature vector to file.
        
        Args:
            feature_vector: Feature vector dictionary
            file_name: File name (with .json or .yaml extension)
            
        Returns:
            True if saved successfully
        """
        try:
            file_path = self.feature_vectors_path / file_name
            
            # Add metadata
            feature_vector_with_metadata = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0",
                    "description": feature_vector.get("description", "User-defined feature vector"),
                    "total_features": len(feature_vector.get("selected_features", []))
                },
                **feature_vector
            }
            
            # Save based on file extension
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'w') as f:
                    json.dump(feature_vector_with_metadata, f, indent=2)
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                with open(file_path, 'w') as f:
                    yaml.safe_dump(feature_vector_with_metadata, f, indent=2)
            else:
                logger.error(f"❌ Unsupported file format: {file_path.suffix}")
                return False
            
            logger.info(f"✅ Saved feature vector to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save feature vector: {e}")
            return False
    
    def load_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a feature vector preset by name.
        
        Args:
            preset_name: Name of the preset (without extension)
            
        Returns:
            Feature vector dictionary or None if not found
        """
        try:
            # Try JSON first, then YAML
            for ext in ['.json', '.yaml', '.yml']:
                preset_path = self.presets_path / f"{preset_name}{ext}"
                if preset_path.exists():
                    return self.load_feature_vector(f"presets/{preset_name}{ext}")
            
            logger.warning(f"⚠️ Preset '{preset_name}' not found")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to load preset '{preset_name}': {e}")
            return None
    
    def save_preset(self, preset_name: str, feature_vector: Dict[str, Any]) -> bool:
        """Save a feature vector as a reusable preset"""
        try:
            preset_path = f"presets/{preset_name}.json"
            return self.save_feature_vector(feature_vector, preset_path)
        except Exception as e:
            logger.error(f"❌ Failed to save preset '{preset_name}': {e}")
            return False
    
    def list_available_presets(self) -> List[Dict[str, Any]]:
        """List all available feature vector presets"""
        try:
            presets = []
            
            for preset_file in self.presets_path.glob("*"):
                if preset_file.suffix.lower() in ['.json', '.yaml', '.yml']:
                    preset_name = preset_file.stem
                    preset_data = self.load_preset(preset_name)
                    
                    if preset_data:
                        presets.append({
                            "name": preset_name,
                            "file": preset_file.name,
                            "description": preset_data.get("description", "No description"),
                            "strategy": preset_data.get("strategy", "unknown"),
                            "total_features": len(preset_data.get("selected_features", [])),
                            "created_at": preset_data.get("metadata", {}).get("created_at"),
                            "services": list(preset_data.get("selected_features_by_service", {}).keys())
                        })
            
            logger.info(f"📋 Found {len(presets)} feature vector presets")
            return presets
            
        except Exception as e:
            logger.error(f"❌ Failed to list presets: {e}")
            return []
    
    def validate_feature_vector(self, feature_vector: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate feature vector structure and content.
        
        Args:
            feature_vector: Feature vector to validate
            
        Returns:
            Validation result with status and details
        """
        try:
            validation_result = {
                "valid": True,
                "issues": [],
                "warnings": [],
                "summary": {}
            }
            
            # Required fields
            required_fields = ["strategy", "selected_features"]
            for field in required_fields:
                if field not in feature_vector:
                    validation_result["valid"] = False
                    validation_result["issues"].append(f"Missing required field: {field}")
            
            # Validate selected_features
            selected_features = feature_vector.get("selected_features", [])
            if isinstance(selected_features, list):
                if len(selected_features) == 0:
                    validation_result["valid"] = False
                    validation_result["issues"].append("No features selected")
                elif len(selected_features) > 100:
                    validation_result["warnings"].append(f"Large feature set ({len(selected_features)} features) may impact performance")
            elif isinstance(selected_features, dict):
                # Dictionary format: {"audio": [...], "content": [...]}
                total_features = sum(len(features) for features in selected_features.values())
                if total_features == 0:
                    validation_result["valid"] = False
                    validation_result["issues"].append("No features selected")
            else:
                validation_result["valid"] = False
                validation_result["issues"].append("Invalid selected_features format (must be list or dict)")
            
            # Validate strategy
            valid_strategies = ["audio_only", "multimodal", "custom"]
            strategy = feature_vector.get("strategy")
            if strategy not in valid_strategies:
                validation_result["warnings"].append(f"Unknown strategy '{strategy}'. Valid strategies: {valid_strategies}")
            
            # Summary
            validation_result["summary"] = {
                "strategy": strategy,
                "total_features": len(selected_features) if isinstance(selected_features, list) else sum(len(f) for f in selected_features.values()),
                "services": list(selected_features.keys()) if isinstance(selected_features, dict) else ["unknown"],
                "valid": validation_result["valid"]
            }
            
            if validation_result["valid"]:
                logger.info(f"✅ Feature vector validation passed: {validation_result['summary']}")
            else:
                logger.error(f"❌ Feature vector validation failed: {validation_result['issues']}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Feature vector validation error: {e}")
            return {
                "valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "warnings": [],
                "summary": {}
            }
    
    def convert_to_flat_features(self, feature_vector: Dict[str, Any]) -> List[str]:
        """
        Convert feature vector to flat list of feature names.
        
        Handles both formats:
        - List: ["audio_energy", "content_sentiment"]
        - Dict: {"audio": ["energy"], "content": ["sentiment"]}
        """
        try:
            selected_features = feature_vector.get("selected_features", [])
            
            if isinstance(selected_features, list):
                # Already flat
                return selected_features
            elif isinstance(selected_features, dict):
                # Convert dict to flat list with service prefixes
                flat_features = []
                for service, features in selected_features.items():
                    for feature in features:
                        # Add service prefix if not already present
                        if not feature.startswith(f"{service}_"):
                            flat_features.append(f"{service}_{feature}")
                        else:
                            flat_features.append(feature)
                return flat_features
            else:
                logger.error(f"❌ Invalid selected_features format: {type(selected_features)}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Failed to convert feature vector to flat list: {e}")
            return []
    
    def create_default_presets(self):
        """Create default feature vector presets if they don't exist"""
        try:
            default_presets = {
                "audio_basic": {
                    "strategy": "audio_only",
                    "description": "Essential audio features for quick training (verified working)",
                    "selected_features_by_service": {
                        "audio": ["audio_energy", "audio_valence", "audio_danceability", "audio_tempo", "audio_loudness"]
                    },
                    "estimated_accuracy": "0.70-0.80",
                    "training_time": "2-3 minutes",
                    "verified_features": True
                },
                "audio_comprehensive": {
                    "strategy": "audio_only",
                    "description": "Comprehensive audio feature set (only working extractors)",
                    "selected_features_by_service": {
                        "audio": [
                            "audio_energy", "audio_valence", "audio_danceability", "audio_tempo",
                            "audio_acousticness", "audio_instrumentalness", "audio_liveness",
                            "audio_speechiness", "audio_brightness", "audio_complexity",
                            "audio_warmth", "audio_harmonic_strength", "audio_key", "audio_mode",
                            "audio_loudness", "audio_duration_ms", "audio_time_signature",
                            "audio_primary_genre"
                        ]
                    },
                    "estimated_accuracy": "0.75-0.85",
                    "training_time": "4-6 minutes",
                    "verified_features": True
                },
                "multimodal_balanced": {
                    "strategy": "multimodal", 
                    "description": "Balanced mix of verified audio and content features",
                    "selected_features_by_service": {
                        "audio": ["audio_energy", "audio_valence", "audio_danceability", "audio_tempo", "audio_primary_genre"],
                        "content": ["lyrics_sentiment_polarity", "lyrics_sentiment_subjectivity", "lyrics_word_count", "lyrics_readability", "lyrics_lexical_diversity", "lyrics_unique_words"]
                    },
                    "estimated_accuracy": "0.80-0.90",
                    "training_time": "5-7 minutes",
                    "notes": "Audio and content features updated to match actual API responses"
                },
                "sentiment_focused": {
                    "strategy": "multimodal", 
                    "description": "Focus on verified emotion and sentiment features",
                    "selected_features_by_service": {
                        "audio": ["audio_valence", "audio_energy", "audio_danceability"],
                        "content": ["lyrics_sentiment_polarity", "lyrics_sentiment_subjectivity", "lyrics_word_count", "lyrics_lexical_diversity", "lyrics_readability", "lyrics_verse_count", "lyrics_avg_word_length"]
                    },
                    "estimated_accuracy": "0.75-0.85",
                    "training_time": "3-5 minutes",
                    "notes": "Updated to use actual lyrical analysis features from content service"
                }
            }
            
            created_count = 0
            for preset_name, preset_data in default_presets.items():
                preset_file = self.presets_path / f"{preset_name}.json"
                if not preset_file.exists():
                    # Convert to flat features format
                    preset_data["selected_features"] = self.convert_to_flat_features(preset_data)
                    
                    if self.save_preset(preset_name, preset_data):
                        created_count += 1
                        logger.info(f"✅ Created default preset: {preset_name}")
            
            if created_count > 0:
                logger.info(f"🎯 Created {created_count} default feature vector presets")
            else:
                logger.info("📋 All default presets already exist")
            
        except Exception as e:
            logger.error(f"❌ Failed to create default presets: {e}")

# Global instance
feature_vector_manager = FeatureVectorManager() 