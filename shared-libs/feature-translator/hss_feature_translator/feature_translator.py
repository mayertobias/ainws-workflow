import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

class FeatureTranslator:
    # Musical key to numeric mapping (chromatic scale)
    KEY_MAPPING = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
        'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }
    
    # Mood to numeric mapping (binary classification)
    MOOD_MAPPING = {
        'happy': 1, 'not happy': 0,
        'sad': 1, 'not sad': 0,
        'aggressive': 1, 'not aggressive': 0,
        'relaxed': 1, 'not relaxed': 0,
        'party': 1, 'not party': 0,
        'electronic': 1, 'not electronic': 0,
        'acoustic': 1, 'not acoustic': 0
    }
    
    def __init__(self, schema_path: Optional[str] = None):
        # Try different paths for schema file (package data first, then fallbacks)
        package_schema_path = os.path.join(os.path.dirname(__file__), "feature_registry_v1.yaml")
        possible_paths = [
            schema_path,
            package_schema_path,                             # Package bundled schema (preferred)
            "/app/shared/schemas/feature_registry_v1.yaml",  # Docker volume path (fallback)
            "schemas/feature_registry_v1.yaml",              # Local development (fallback)
            os.path.join(os.path.dirname(__file__), "..", "schemas", "feature_registry_v1.yaml")  # Legacy path
        ]
        
        schema_loaded = False
        for path in possible_paths:
            if path and os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        self.schema = yaml.safe_load(f)
                    schema_loaded = True
                    print(f"✅ Loaded feature schema from: {path}")
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load schema from {path}: {e}")
                    continue
        
        if not schema_loaded:
            raise FileNotFoundError(
                f"Could not load feature schema from any of these paths: {possible_paths}"
            )
    
    def audio_producer_to_consumer(self, raw_audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """Convert audio service output to ML consumer format"""
        translated = {}
        
        for feature_name, config in self.schema['audio_features'].items():
            producer_path = config['producer_output']
            consumer_name = config['consumer_expected']
            
            # Navigate nested dictionary path
            value = self._get_nested_value(raw_audio_features, producer_path)
            
            if value is not None:
                # Apply data type conversions for ML compatibility
                if feature_name == 'audio_key' and isinstance(value, str):
                    # Convert musical key string to numeric value
                    value = self.KEY_MAPPING.get(value, 0)  # Default to C if unknown
                elif 'mood_' in feature_name and isinstance(value, str):
                    # Convert mood classification to binary numeric
                    value = self.MOOD_MAPPING.get(value, 0)  # Default to 0 if unknown
                elif feature_name == 'audio_primary_genre' and isinstance(value, str):
                    # For genre, use hash-based encoding
                    value = self._encode_genre(value)
                
                translated[consumer_name] = value
            elif config.get('validation_required', False):
                raise ValueError(f"Required feature {feature_name} not found in audio service output")
                
        return translated
    
    def content_producer_to_consumer(self, raw_content_features: Dict[str, Any]) -> Dict[str, Any]:
        """Convert content service output to ML consumer format"""
        translated = {}
        
        for feature_name, config in self.schema['content_features'].items():
            producer_output = config['producer_output']
            consumer_name = config['consumer_expected']
            
            # Use nested value extraction for complex paths
            value = self._get_nested_value(raw_content_features, producer_output)
            
            if value is not None:
                translated[consumer_name] = value
            elif config.get('validation_required', False):
                raise ValueError(f"Required feature {consumer_name} not found in content service output")
                
        return translated
    
    def validate_consumer_features(self, features: Dict[str, Any], feature_type: str) -> Dict[str, Any]:
        """Validate features against schema constraints"""
        schema_features = self.schema[f'{feature_type}_features']
        validated = {}
        
        for feature_name, value in features.items():
            # Find matching feature config by consumer_expected name
            feature_config = None
            for config in schema_features.values():
                if config['consumer_expected'] == feature_name:
                    feature_config = config
                    break
            
            if feature_config:
                # Type validation
                expected_type = feature_config['type']
                if expected_type == 'float' and not isinstance(value, (int, float)):
                    raise ValueError(f"{feature_name}: Expected float, got {type(value)}")
                
                if expected_type == 'integer' and not isinstance(value, int):
                    raise ValueError(f"{feature_name}: Expected integer, got {type(value)}")
                
                # Range validation
                if 'range' in feature_config:
                    min_val, max_val = feature_config['range']
                    if not (min_val <= value <= max_val):
                        raise ValueError(f"{feature_name}: Value {value} outside range [{min_val}, {max_val}]")
                
                validated[feature_name] = float(value) if expected_type == 'float' else value
                
        return validated
    
    def _get_nested_value(self, data: Dict, path: str) -> Optional[Any]:
        """Navigate nested dictionary using dot notation and array indexing"""
        import re
        keys = path.split('.')
        current = data
        
        for key in keys:
            # Handle array indexing like "top_genres[0][1]"
            if '[' in key and ']' in key:
                # Extract base key and indices
                match = re.match(r'(\w+)(\[.+\])', key)
                if match:
                    base_key, indices_str = match.groups()
                    
                    # Get the base object
                    if isinstance(current, dict) and base_key in current:
                        current = current[base_key]
                    else:
                        return None
                    
                    # Apply each index
                    indices = re.findall(r'\[(\d+)\]', indices_str)
                    for index in indices:
                        try:
                            if isinstance(current, (list, tuple)) and int(index) < len(current):
                                current = current[int(index)]
                            else:
                                return None
                        except (ValueError, IndexError):
                            return None
                else:
                    return None
            else:
                # Regular dictionary key access
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return None
                
        return current
    
    def get_required_features(self, feature_type: str) -> List[str]:
        """Get list of required features for validation"""
        schema_features = self.schema[f'{feature_type}_features']
        return [
            config['consumer_expected'] 
            for config in schema_features.values() 
            if config.get('validation_required', False)
        ]
    
    def get_required_features_for_model_type(self, model_type: str) -> List[str]:
        """
        Get required features based on model type for context-aware validation
        
        Args:
            model_type: Type of model - 'audio_only', 'multimodal', or 'custom'
            
        Returns:
            List of required feature names for the specific model type
        """
        if model_type == 'audio_only':
            # For audio-only models, only require audio features
            return self.get_required_features('audio')
        elif model_type == 'multimodal':
            # For multimodal models, require both audio and content features
            return (self.get_required_features('audio') + 
                   self.get_required_features('content'))
        elif model_type == 'custom':
            # For custom models, require both by default (can be enhanced with model metadata)
            return (self.get_required_features('audio') + 
                   self.get_required_features('content'))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def detect_model_type_from_name(self, model_name: str) -> str:
        """
        Determine model type from model name for smart feature extraction
        
        Args:
            model_name: Full model name
            
        Returns:
            Detected model type: 'audio_only', 'multimodal', or 'custom'
        """
        lower_name = model_name.lower()
        
        if 'audio_only' in lower_name or 'audio-only' in lower_name:
            return 'audio_only'
        elif 'multimodal' in lower_name or 'multi-modal' in lower_name:
            return 'multimodal'
        else:
            # Default to custom for unknown patterns
            return 'custom'
    
    def _encode_genre(self, genre: str) -> int:
        """
        Simple hash-based encoding for genre strings
        Ensures consistent numeric mapping for the same genre
        """
        hash_value = hash(genre)
        # Return positive number between 0 and 999
        return abs(hash_value) % 1000

# Test validation
if __name__ == "__main__":
    translator = FeatureTranslator()
    
    # Test audio translation
    test_audio = {
        "results": {
            "features": {
                "analysis": {
                    "basic": {
                        "energy": 0.7,
                        "valence": 0.6,
                        "tempo": 120.5,
                        "danceability": 0.8
                    }
                }
            }
        }
    }
    
    translated = translator.audio_producer_to_consumer(test_audio)
    print("Audio translation test:", translated)
    
    # Test content translation
    test_content = {
        "sentiment_polarity": 0.3,
        "word_count": 45
    }
    
    translated_content = translator.content_producer_to_consumer(test_content)
    print("Content translation test:", translated_content)