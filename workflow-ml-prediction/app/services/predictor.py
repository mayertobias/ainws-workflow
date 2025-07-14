"""
ML Prediction Service for workflow-ml-prediction microservice

Handles real-time model predictions, model loading, and caching.
"""

import asyncio
import time
import uuid
import hashlib
import json
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import redis
from minio import Minio
from minio.error import S3Error

# Import feature translator and derived features calculator as proper packages
from hss_feature_translator import FeatureTranslator
from hss_derived_features import DerivedFeaturesCalculator

from ..config.settings import settings
from ..models.prediction import (
    FeatureInput, PredictionResult, PredictionStatus,
    ModelInfo, PredictionMetrics
)

logger = logging.getLogger(__name__)

class ModelCache:
    """In-memory model cache with LRU eviction."""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, datetime] = {}
        self.load_times: Dict[str, float] = {}
        
    def get(self, model_id: str) -> Optional[Any]:
        """Get model from cache."""
        if model_id in self.cache:
            self.access_times[model_id] = datetime.utcnow()
            return self.cache[model_id]
        return None
    
    def put(self, model_id: str, model: Any, load_time: float) -> None:
        """Put model in cache with LRU eviction."""
        if len(self.cache) >= self.max_size and model_id not in self.cache:
            # Evict least recently used
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            self.evict(lru_key)
        
        self.cache[model_id] = model
        self.access_times[model_id] = datetime.utcnow()
        self.load_times[model_id] = load_time
        
    def evict(self, model_id: str) -> None:
        """Evict model from cache."""
        if model_id in self.cache:
            del self.cache[model_id]
            del self.access_times[model_id]
            del self.load_times[model_id]
            logger.info(f"Evicted model {model_id} from cache")
    
    def clear(self) -> None:
        """Clear all models from cache."""
        self.cache.clear()
        self.access_times.clear()
        self.load_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cached_models': len(self.cache),
            'max_size': self.max_size,
            'cache_keys': list(self.cache.keys()),
            'total_load_time': sum(self.load_times.values()),
            'average_load_time': np.mean(list(self.load_times.values())) if self.load_times else 0
        }

class SmartSongPredictor:
    """Smart prediction service that selects the best model based on available features."""
    
    def __init__(self, model_registry_path: Optional[str] = None):
        """Initialize the prediction service with model registry."""
        if model_registry_path:
            self.model_registry_path = model_registry_path
        else:
            # Try to find the model registry in different locations
            potential_paths = [
                "/Users/manojveluchuri/saas/workflow/shared-models/ml-training/model_registry.json",  # Local development
                "../shared-models/ml-training/model_registry.json",  # Relative path
                "/app/models/ml-training/model_registry.json",  # Container path
                settings.MODEL_REGISTRY_PATH  # Settings default
            ]
            
            self.model_registry_path = None
            for path in potential_paths:
                if Path(path).exists():
                    self.model_registry_path = path
                    break
            
            if not self.model_registry_path:
                self.model_registry_path = settings.MODEL_REGISTRY_PATH  # Fallback to settings
        self.models = {}
        self.model_metadata = {}
        self.features_info = {}
        self.registry = {}
        self.last_registry_check = None
        
        # Initialize derived features calculator (shared library)
        self.derived_features_calculator = DerivedFeaturesCalculator()
        logger.info("✅ Initialized derived features calculator for multimodal features")
        
        # Load models on initialization
        asyncio.create_task(self._load_models_async())
        
        # Start 24-hour model registry update scheduler
        asyncio.create_task(self._start_model_registry_scheduler())
    
    async def _load_models_async(self):
        """Load models asynchronously."""
        try:
            await self.load_models()
        except Exception as e:
            logger.error(f"Error loading models during initialization: {e}")
    
    async def _start_model_registry_scheduler(self):
        """Start scheduler to automatically update model registry."""
        startup_delay = settings.MODEL_REGISTRY_STARTUP_DELAY_MINUTES * 60
        update_interval = settings.MODEL_REGISTRY_UPDATE_INTERVAL_HOURS * 3600
        
        logger.info(f"🕐 Model registry scheduler starting with {settings.MODEL_REGISTRY_UPDATE_INTERVAL_HOURS}h interval")
        logger.info(f"   ⏳ Startup delay: {settings.MODEL_REGISTRY_STARTUP_DELAY_MINUTES} minutes")
        
        # Wait before first scheduled check (default: 1 hour)
        await asyncio.sleep(startup_delay)
        
        while True:
            try:
                old_model_count = len(self.models)
                logger.info(f"🕐 Starting scheduled model registry update ({settings.MODEL_REGISTRY_UPDATE_INTERVAL_HOURS}h cycle)")
                
                await self.load_models()
                
                new_model_count = len(self.models)
                if new_model_count != old_model_count:
                    logger.info(f"🔄 Model registry updated via scheduler: {old_model_count} → {new_model_count} models")
                    
                    # Log new models for visibility
                    new_models = set(self.models.keys())
                    for model_name in new_models:
                        model_metadata = self.model_metadata.get(model_name, {})
                        r2_score = model_metadata.get('r2_score', 0.0)
                        logger.info(f"   📊 {model_name}: R² = {r2_score:.4f}")
                else:
                    logger.info("✅ Model registry checked - no changes detected")
                
            except Exception as e:
                logger.error(f"❌ Error in scheduled model registry update: {e}")
            
            # Wait for configured interval (default: 24 hours)
            await asyncio.sleep(update_interval)
    
    async def load_models(self):
        """Load all available models from the registry."""
        
        logger.info(f"Loading models from registry: {self.model_registry_path}")
        
        registry_path = Path(self.model_registry_path)
        if not registry_path.exists():
            logger.warning(f"Model registry not found: {self.model_registry_path}")
            return
        
        # Check if registry has been updated
        if self.last_registry_check:
            stat = registry_path.stat()
            if stat.st_mtime <= self.last_registry_check:
                return  # No update needed
        
        try:
            with open(registry_path, 'r') as f:
                self.registry = json.load(f)
            
            # Clear existing models
            self.models.clear()
            self.model_metadata.clear()
            self.features_info.clear()
            
            for model_type, model_info in self.registry.get('available_models', {}).items():
                try:
                    # Load the actual model
                    model_path = self._resolve_model_path(model_info['model_path'])
                    if not Path(model_path).exists():
                        logger.warning(f"Model file not found: {model_path}")
                        continue
                    
                    # Load model using consistent scikit-learn version
                    model = joblib.load(model_path)
                    self.models[model_type] = model
                    
                    # Load metadata
                    metadata_path = self._resolve_model_path(model_info['metadata_path'])
                    if Path(metadata_path).exists():
                        with open(metadata_path, 'r') as f:
                            self.model_metadata[model_type] = json.load(f)
                    
                    # FIXED: Use actual trained features from metadata instead of hardcoded features.json
                    # Load the real selected_features that the model was trained with
                    if model_type in self.model_metadata and 'selected_features' in self.model_metadata[model_type]:
                        actual_features = self.model_metadata[model_type]['selected_features']
                        self.features_info[model_type] = {
                            'all_features': actual_features,
                            'audio_features': [f for f in actual_features if f.startswith('audio_')],
                            'lyrics_features': [f for f in actual_features if f.startswith('lyrics_') or f.startswith('content_')]
                        }
                        logger.info(f"✅ Using actual trained features for {model_type}: {len(actual_features)} features")
                    else:
                        # Fallback to features.json if metadata is not available (backwards compatibility)
                        features_path = self._resolve_model_path(model_info['features_path'])
                        if Path(features_path).exists():
                            with open(features_path, 'r') as f:
                                self.features_info[model_type] = json.load(f)
                            logger.warning(f"⚠️ Using fallback features.json for {model_type} - consider updating to use metadata")
                    
                    logger.info(f"✅ Loaded {model_type} model:")
                    logger.info(f"      - R² Score: {model_info['performance']['r2_score']:.3f}")
                    logger.info(f"      - Features: {len(self.features_info[model_type]['all_features'])}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load {model_type} model: {e}")
                    logger.error(f"   Model path: {model_path}")
                    logger.error(f"   Error type: {type(e).__name__}")
                    # Continue with other models even if one fails
            
            self.last_registry_check = registry_path.stat().st_mtime
            logger.info(f"Loaded {len(self.models)} models successfully")
            
        except Exception as e:
            logger.error(f"Error loading model registry: {e}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available model types."""
        return list(self.models.keys())
    
    def validate_features(self, song_features: Dict[str, Any], model_type: str) -> Tuple[bool, List[str]]:
        """Validate that required features are available for the model."""
        
        if model_type not in self.features_info:
            logger.error(f"Model {model_type} not found in features_info. Available models: {list(self.features_info.keys())}")
            return False, [f"Model {model_type} not found"]
        
        required_features = self.features_info[model_type]['all_features']
        available_features = set(song_features.keys())
        
        # Log detailed feature information
        logger.info(f"🔍 Feature validation for model '{model_type}':")
        logger.info(f"   📋 Required features ({len(required_features)}): {required_features[:10]}...")
        logger.info(f"   📦 Available features ({len(available_features)}): {list(available_features)}")
        
        # Check if we have all required features
        missing_features = set(required_features) - available_features
        
        if missing_features:
            logger.warning(f"   ❌ Missing {len(missing_features)} features: {list(missing_features)[:10]}...")
            return False, list(missing_features)
        
        logger.info(f"   ✅ All required features available for model '{model_type}'")
        return True, []
    
    def select_best_model(self, song_features: Dict[str, Any]) -> str:
        """Select the best model based on available features and performance."""
        
        # Check if any features are provided
        if not song_features or len(song_features) == 0:
            logger.warning("No features provided for model selection")
            # Return first available model as fallback
            if self.models:
                return list(self.models.keys())[0]
            else:
                raise ValueError("No features provided and no models available")
        
        # Check which models can be used (including model-specific derived features)
        usable_models = []
        
        for model_type in self.models.keys():
            # Calculate model-specific features for validation
            model_features = self._calculate_model_specific_derived_features(song_features, model_type)
            is_valid, missing = self.validate_features(model_features, model_type)
            if is_valid:
                performance = self.model_metadata.get(model_type, {}).get('r2_score', 0.0)
                usable_models.append((model_type, performance))
                logger.info(f"✅ Model '{model_type}' is usable (R²={performance:.3f})")
            else:
                logger.info(f"❌ Model '{model_type}' missing features: {missing[:5]}...")
        
        if not usable_models:
            available_features = list(song_features.keys())
            logger.warning(f"No models can be used with features: {available_features}")
            # Return best available model as fallback
            if self.models:
                return list(self.models.keys())[0]
            else:
                raise ValueError(f"No models can be used with features: {available_features}")
        
        # Select model with best performance
        best_model = max(usable_models, key=lambda x: x[1])
        return best_model[0]
    
    async def predict_single_song(self, song_features: Dict[str, Any], 
                                 model_type: Optional[str] = None,
                                 explain_prediction: bool = True) -> Dict[str, Any]:
        """Make a prediction for a single song."""
        
        # Ensure models are loaded
        await self.load_models()
        
        if not self.models:
            raise ValueError("No models available for prediction")
        
        # Auto-select model if not specified
        if model_type is None:
            model_type = self.select_best_model(song_features)
            logger.info(f"🎯 Auto-selected model: {model_type}")
        
        # IMPORTANT: Add only derived features that this specific model expects
        enhanced_features = self._calculate_model_specific_derived_features(song_features, model_type)
        
        # Validate features (using enhanced features)
        is_valid, missing_features = self.validate_features(enhanced_features, model_type)
        if not is_valid:
            raise ValueError(f"Cannot use {model_type} model. Missing features: {missing_features}")
        
        # Use enhanced features for the rest of the prediction
        song_features = enhanced_features
        
        # Prepare features in correct order (FIXED: Use DataFrame with feature names)
        required_features = self.features_info[model_type]['all_features']
        feature_values = [song_features[feature] for feature in required_features]
        
        # Create DataFrame with proper column names to avoid sklearn warnings
        import pandas as pd
        feature_df = pd.DataFrame([feature_values], columns=required_features)
        feature_array = feature_df  # Use DataFrame instead of numpy array
        
        # Make prediction (REGRESSION: Direct continuous output)
        model = self.models[model_type]
        
        # Regression model - direct prediction (0-1 range)
        raw_prediction = model.predict(feature_df)[0]
        logger.info(f"📈 Raw regression prediction: {raw_prediction}")
        
        # Models are trained on continuous hit_score (0-1 range)
        # Convert to percentage (0-100) for user display
        prediction = float(raw_prediction) * 100
        
        # Log for debugging
        if raw_prediction < 0 or raw_prediction > 1:
            logger.warning(f"⚠️ Model prediction outside expected 0-1 range: {raw_prediction}")
            prediction = max(0.0, min(100.0, prediction))
            
        logger.info(f"📈 Final hit score prediction: {prediction}%")
        
        # Ensure prediction is in valid range (0-100)
        prediction = max(0.0, min(100.0, prediction))
        
        # Calculate confidence (simplified approach)
        confidence = self._calculate_confidence(model_type, song_features)
        
        # Safely get model performance metrics (regression)
        model_metadata = self.model_metadata.get(model_type, {})
        
        result = {
            'prediction': float(prediction),
            'confidence': confidence,
            'model_used': model_type,
            'model_type': 'regression',
            'model_performance': {
                'r2_score': model_metadata.get('r2_score', 0.5),
                'mse': model_metadata.get('mse', 0.25),
                'rmse': model_metadata.get('rmse', 0.5),
                'mae': model_metadata.get('mae', 0.2),
                'mape': model_metadata.get('mape', 20.0)
            },
            # Include enhanced features with readable genre names for UI consumption
            'enhanced_features': self._enhance_features_with_readable_names(song_features)
        }
        
        # Add explanation if requested
        if explain_prediction:
            explanation = self._generate_explanation(model_type, song_features, prediction)
            result['explanation'] = explanation
            result['top_influencing_features'] = explanation['top_features']
        
        return result
    
    def _calculate_confidence(self, model_type: str, song_features: Dict[str, Any]) -> float:
        """Calculate prediction confidence based on regression model performance and feature coverage."""
        
        try:
            # Get regression model performance metrics
            model_metadata = self.model_metadata.get(model_type, {})
            r2_score = model_metadata.get('r2_score', 0.5)
            mse = model_metadata.get('mse', 0.25)
            
            # Convert R² to confidence (optimized for regression)
            if r2_score < 0:
                # Negative R² means model performs worse than mean baseline
                performance_confidence = 0.2
                logger.info(f"📊 Negative R² ({r2_score:.3f}) indicates poor regression model performance")
            elif r2_score > 0.9:
                # Very high R² likely indicates overfitting, reduce confidence
                performance_confidence = 0.8
                logger.info(f"📊 Very high R² ({r2_score:.3f}) suggests possible overfitting")
            else:
                # Normal case: R² directly translates to confidence
                performance_confidence = float(r2_score)
                logger.info(f"📊 Regression R² score: {r2_score:.3f}")
            
            # Feature coverage factor
            features_info = self.features_info.get(model_type, {})
            required_features = features_info.get('all_features', [])
            
            if required_features:
                available_count = len([f for f in required_features if f in song_features])
                feature_coverage = available_count / len(required_features)
                logger.info(f"📊 Feature coverage: {available_count}/{len(required_features)} ({feature_coverage:.1%})")
            else:
                feature_coverage = 1.0  # Assume full coverage if no info
            
            # Final confidence combines performance and feature availability (regression)
            final_confidence = performance_confidence * feature_coverage
            
            # Adjust bounds for regression (more conservative)
            final_confidence = max(0.2, min(0.9, final_confidence))
            
            logger.info(f"📊 Regression Confidence: performance={performance_confidence:.3f} × coverage={feature_coverage:.3f} = {final_confidence:.3f}")
            return float(final_confidence)
            
        except Exception as e:
            logger.warning(f"Error calculating confidence for {model_type}: {e}")
            return 0.6  # Default middle-ground confidence
    
    def _generate_explanation(self, model_type: str, song_features: Dict[str, Any], 
                             prediction: float) -> Dict[str, Any]:
        """Generate explanation for the prediction."""
        
        try:
            # Safely get feature importance data
            model_metadata = self.model_metadata.get(model_type, {})
            feature_importance = model_metadata.get('feature_importance', {})
            
            if not feature_importance:
                logger.warning(f"No feature importance data available for model {model_type}")
                # Create a basic explanation without feature importance
                return {
                    'prediction_value': float(prediction),
                    'model_used': model_type,
                    'top_features': [],
                    'prediction_factors': {},
                    'model_confidence': model_metadata.get('accuracy', model_metadata.get('r2_score', 0.5))
                }
            
            # FIXED: Handle nested feature importance structure
            # Extract flat feature importance from nested structure
            flat_importance = {}
            
            if isinstance(feature_importance, dict):
                # Check if it has nested structure (ensemble, random_forest, etc.)
                if any(isinstance(v, dict) for v in feature_importance.values()):
                    # Use ensemble importance if available, otherwise first nested dict
                    if 'ensemble' in feature_importance:
                        flat_importance = feature_importance['ensemble']
                    elif feature_importance:
                        # Take first nested dict
                        first_key = next(iter(feature_importance.keys()))
                        flat_importance = feature_importance[first_key]
                        logger.info(f"Using {first_key} feature importance for explanation")
                else:
                    # Already flat structure
                    flat_importance = feature_importance
            
            # Convert all values to float for sorting
            numeric_importance = {}
            for k, v in flat_importance.items():
                try:
                    numeric_importance[k] = float(v)
                except (ValueError, TypeError):
                    logger.warning(f"Skipping non-numeric importance value for {k}: {v}")
            
            if not numeric_importance:
                logger.warning(f"No valid numeric feature importance found for model {model_type}")
                top_features = []
            else:
                # Get top influencing features
                top_features = sorted(numeric_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Create explanation
            explanation = {
                'prediction_value': float(prediction),
                'model_used': model_type,
                'top_features': [
                    {
                        'feature': feature_name,
                        'importance': float(importance),
                        'value': song_features.get(feature_name, 'N/A'),
                        'description': self._get_feature_description(feature_name)
                    }
                    for feature_name, importance in top_features
                ],
                'prediction_factors': self._analyze_prediction_factors(song_features, top_features),
                'model_confidence': model_metadata.get('accuracy', model_metadata.get('r2_score', 0.5))
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation for {model_type}: {e}")
            # Return a basic explanation if anything fails
            return {
                'prediction_value': float(prediction),
                'model_used': model_type,
                'top_features': [],
                'prediction_factors': {},
                'model_confidence': 0.5,
                'explanation_error': str(e)
            }
    
    def _calculate_model_specific_derived_features(self, song_features: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """Calculate only the derived features that the specific model expects.
        
        Uses the model's metadata to determine which derived features were used during training.
        This prevents feature count mismatches between models trained with different feature sets.
        """
        enhanced_features = song_features.copy()
        
        try:
            # Get the features this specific model expects
            model_features = self.features_info.get(model_type, {}).get('all_features', [])
            
            # Identify which derived features this model expects
            expected_derived_features = [
                feature for feature in model_features 
                if feature in self.derived_features_calculator.derived_feature_names
            ]
            
            if not expected_derived_features:
                logger.info(f"📊 Model '{model_type}' was trained without derived features")
                return enhanced_features
            
            logger.info(f"📊 Model '{model_type}' expects {len(expected_derived_features)} derived features: {expected_derived_features}")
            
            # Calculate all available derived features using shared library
            all_derived_features = self.derived_features_calculator.calculate_derived_features(song_features)
            
            # Only add the derived features this model expects
            model_specific_derived = {
                feature: all_derived_features[feature] 
                for feature in expected_derived_features 
                if feature in all_derived_features
            }
            
            # Add model-specific derived features to the feature set
            enhanced_features.update(model_specific_derived)
            
            logger.info(f"✅ Added {len(model_specific_derived)} model-specific derived features: {list(model_specific_derived.keys())}")
            
            # Check if any expected derived features are missing
            missing_derived = set(expected_derived_features) - set(model_specific_derived.keys())
            if missing_derived:
                logger.warning(f"⚠️ Could not calculate some expected derived features: {missing_derived}")
                # Add default values for missing derived features
                for feature in missing_derived:
                    enhanced_features[feature] = 0.5  # Safe default value
                    logger.info(f"   Using default value 0.5 for {feature}")
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate model-specific derived features for {model_type}: {e}")
            # Fall back to original features without derived features
            
        return enhanced_features
    
    def _enhance_features_with_readable_names(self, song_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance features with readable genre names for UI consumption.
        
        This method ensures that readable genre names are available in the response
        so the UI can display them properly instead of numeric IDs.
        """
        enhanced_features = song_features.copy()
        
        try:
            # Check if we already have readable genre names (from frontend)
            if 'audio_primary_genre_name' in song_features:
                logger.info(f"📋 Found existing readable genre name: {song_features['audio_primary_genre_name']}")
                return enhanced_features
            
            # If we have the encoded genre but no readable name, try to preserve any
            # original genre information that might be available
            if 'audio_primary_genre' in song_features:
                encoded_genre = song_features['audio_primary_genre']
                logger.info(f"📋 Found encoded genre: {encoded_genre}")
                
                # For now, we don't have a reverse mapping table, so we'll indicate
                # that this is an encoded value. In a future enhancement, we could
                # maintain a bidirectional mapping table.
                enhanced_features['audio_primary_genre_name'] = f"Genre_{encoded_genre}"
                logger.info(f"📋 Generated genre name placeholder: Genre_{encoded_genre}")
            
            # Check for other feature name patterns that might benefit from enhancement
            for feature_key, feature_value in song_features.items():
                if feature_key.endswith('_name') and isinstance(feature_value, str):
                    # Preserve any readable names that already exist
                    enhanced_features[feature_key] = feature_value
                    logger.debug(f"📋 Preserved readable feature: {feature_key} = {feature_value}")
            
            return enhanced_features
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to enhance features with readable names: {e}")
            # Return original features if enhancement fails
            return song_features
    
    def _get_feature_description(self, feature_name: str) -> str:
        """Get human-readable description of a feature."""
        
        descriptions = {
            'tempo': 'Song tempo (beats per minute)',
            'energy': 'Energy level of the song (0-1)',
            'danceability': 'How suitable the song is for dancing (0-1)',
            'valence': 'Musical positivity/happiness (0-1)',
            'loudness': 'Overall loudness in decibels',
            'speechiness': 'Presence of speech-like vocals (0-1)',
            'acousticness': 'Acoustic vs electric instrumentation (0-1)',
            'instrumentalness': 'Absence of vocals (0-1)',
            'liveness': 'Presence of live audience (0-1)',
            'lyrics_sentiment_score': 'Overall emotional sentiment of lyrics',
            'lyrics_word_count': 'Number of words in lyrics',
            'lyrics_complexity_score': 'Linguistic complexity of lyrics',
            'lyrics_theme_diversity': 'Variety of themes in lyrics',
            'rhythmic_appeal_index': 'Composite rhythmic and dance appeal score',
            'emotional_impact_score': 'Composite emotional impact measurement',
            'commercial_viability_index': 'Composite commercial appeal score',
            'sonic_sophistication_score': 'Composite musical sophistication score'
        }
        
        return descriptions.get(feature_name, f"Audio/lyrics feature: {feature_name}")
    
    def _analyze_prediction_factors(self, song_features: Dict[str, Any], 
                                   top_features: List[tuple]) -> Dict[str, str]:
        """Analyze what factors contributed to the prediction."""
        
        factors = {}
        
        for feature_name, importance in top_features[:3]:
            value = song_features.get(feature_name, 0)
            
            if feature_name == 'energy' and value > 0.7:
                factors['energy'] = "High energy contributes positively to prediction"
            elif feature_name == 'valence' and value > 0.6:
                factors['mood'] = "Positive mood/valence boosts prediction"
            elif feature_name == 'danceability' and value > 0.7:
                factors['danceability'] = "High danceability increases appeal"
            elif feature_name.startswith('lyrics_') and 'sentiment' in feature_name and value > 0:
                factors['lyrics_sentiment'] = "Positive lyrics sentiment adds value"
        
        return factors
    
    async def predict_batch(self, songs_data: Union[List[Dict[str, Any]], pd.DataFrame]) -> List[Dict[str, Any]]:
        """Make predictions for multiple songs."""
        
        if isinstance(songs_data, pd.DataFrame):
            songs_list = songs_data.to_dict('records')
        else:
            songs_list = songs_data
        
        predictions = []
        
        for i, song_data in enumerate(songs_list):
            try:
                # Extract features (remove metadata columns)
                song_features = {k: v for k, v in song_data.items() 
                               if k not in ['file_path', 'artist', 'title', 'has_lyrics', 'popularity_score']}
                
                prediction = await self.predict_single_song(song_features)
                
                # Add song metadata
                prediction['song_info'] = {
                    'file_path': song_data.get('file_path', ''),
                    'artist': song_data.get('artist', ''),
                    'title': song_data.get('title', ''),
                    'has_lyrics': song_data.get('has_lyrics', False)
                }
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Failed to predict for song {i}: {e}")
                predictions.append({
                    'error': str(e),
                    'song_info': {
                        'file_path': song_data.get('file_path', ''),
                        'artist': song_data.get('artist', ''),
                        'title': song_data.get('title', '')
                    }
                })
        
        return predictions
    
    async def get_model_info(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        """Get information about available models."""
        
        await self.load_models()
        
        if model_type:
            if model_type not in self.models:
                raise ValueError(f"Model {model_type} not found")
            
            return {
                'model_type': model_type,
                'performance': self.model_metadata[model_type],
                'features': self.features_info[model_type],
                'last_updated': self.registry.get('created_at', 'Unknown')
            }
        
        # Return info for all models
        return {
            'available_models': list(self.models.keys()),
            'model_details': {
                model_type: {
                    'performance': {
                        'accuracy': self.model_metadata[model_type].get('accuracy', 0.5),
                        'r2_score': self.model_metadata[model_type].get('r2_score', self.model_metadata[model_type].get('accuracy', 0.5)),
                        'mse': self.model_metadata[model_type].get('mse', 0.5)
                    },
                    'feature_count': len(self.features_info[model_type]['all_features']),
                    'model_type': model_type,
                    'training_timestamp': self.model_metadata[model_type].get('training_timestamp'),
                    'strategy': self.model_metadata[model_type].get('strategy'),
                    'experiment_name': self.model_metadata[model_type].get('experiment_name')
                }
                for model_type in self.models.keys()
            },
            'prediction_strategy': self.registry.get('prediction_strategy', 'smart_selection'),
            'last_updated': self.registry.get('created_at', 'Unknown')
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the prediction service."""
        
        try:
            await self.load_models()
            
            status = {
                'status': 'healthy',
                'models_loaded': len(self.models),
                'available_models': list(self.models.keys()),
                'registry_path': self.model_registry_path,
                'registry_exists': Path(self.model_registry_path).exists(),
                'last_registry_check': self.last_registry_check,
                'prediction_capability': 'ready' if self.models else 'no_models'
            }
            
            # Test a simple prediction if models are available
            if self.models:
                test_features = self._generate_test_features()
                try:
                    test_prediction = await self.predict_single_song(test_features, explain_prediction=False)
                    status['test_prediction'] = 'success'
                except Exception as e:
                    status['test_prediction'] = f'failed: {str(e)}'
            
            return status
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'models_loaded': len(self.models),
                'prediction_capability': 'error'
            }
    
    def _generate_test_features(self) -> Dict[str, Any]:
        """Generate test features for health check."""
        
        # Get features from the first available model
        if not self.features_info:
            raise ValueError("No model features available")
        
        first_model = list(self.features_info.keys())[0]
        required_features = self.features_info[first_model]['all_features']
        
        # Generate reasonable test values
        test_features = {}
        for feature in required_features:
            if feature.startswith('lyrics_'):
                if 'count' in feature or 'words' in feature:
                    test_features[feature] = 100
                elif 'sentiment' in feature:
                    test_features[feature] = 0.5
                else:
                    test_features[feature] = 0.5
            else:
                # Audio features
                if feature == 'tempo':
                    test_features[feature] = 120.0
                elif feature == 'loudness':
                    test_features[feature] = -10.0
                else:
                    test_features[feature] = 0.5
        
        return test_features
    
    async def update_models(self) -> Dict[str, Any]:
        """Force update of models from registry."""
        
        self.last_registry_check = None  # Force reload
        await self.load_models()
        
        return {
            'status': 'updated',
            'models_loaded': len(self.models),
            'available_models': list(self.models.keys()),
            'timestamp': datetime.now().isoformat()
        }
    
    def validate_prediction_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Validate features for prediction using schema"""
        validated_features = {}
        
        # Separate audio and content features
        audio_features = {k: v for k, v in features.items() if k.startswith('audio_')}
        content_features = {k: v for k, v in features.items() if k.startswith('content_')}
        
        # Validate each feature type
        if audio_features:
            validated_audio = self.feature_translator.validate_consumer_features(
                audio_features, 'audio'
            )
            validated_features.update(validated_audio)
            
        if content_features:
            validated_content = self.feature_translator.validate_consumer_features(
                content_features, 'content'
            )
            validated_features.update(validated_content)
        
        # Check for minimum required features
        required_audio = self.feature_translator.get_required_features('audio')
        required_content = self.feature_translator.get_required_features('content')
        
        # For prediction, we need at least some audio OR some content features
        has_audio = any(f in validated_features for f in required_audio)
        has_content = any(f in validated_features for f in required_content)
        
        if not has_audio and not has_content:
            raise ValueError(
                f"Insufficient features for prediction. "
                f"Need at least one of audio features {required_audio} "
                f"or content features {required_content}. "
                f"Received: {list(features.keys())}"
            )
        
        return validated_features
    
    def _fix_monotonic_cst_model(self, model):
        """Fix models with monotonic_cst compatibility issues."""
        import copy
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.pipeline import Pipeline
        
        logger.info("Applying comprehensive monotonic_cst compatibility fix...")
        logger.info(f"Model type: {type(model)}")
        
        try:
            # Create a deep copy to avoid modifying the original
            fixed_model = copy.deepcopy(model)
            
            # ALTERNATIVE APPROACH: Rebuild the problematic estimators
            if hasattr(fixed_model, 'named_steps'):
                logger.info("Pipeline detected - checking each step...")
                for step_name, step_estimator in fixed_model.named_steps.items():
                    logger.info(f"  Step '{step_name}': {type(step_estimator)}")
                    if hasattr(step_estimator, 'estimators_'):
                        logger.info(f"    Has estimators_: {len(step_estimator.estimators_)}")
                        for i, est in enumerate(step_estimator.estimators_):
                            logger.info(f"      Estimator {i}: {type(est)}")
                            if hasattr(est, 'monotonic_cst'):
                                logger.info(f"        Found monotonic_cst in estimator {i}")
            
            # Apply the comprehensive fix to all nested estimators
            total_removed = self._remove_monotonic_cst_from_estimator(fixed_model)
            
            logger.info(f"Successfully applied monotonic_cst compatibility fix - removed {total_removed} instances")
            return fixed_model
            
        except Exception as e:
            logger.error(f"Failed to apply monotonic_cst fix: {e}")
            raise e
    
    def _remove_monotonic_cst_from_estimator(self, estimator):
        """Remove monotonic_cst attribute from an estimator and all nested estimators."""
        removed_count = 0
        
        # Remove from current estimator
        if hasattr(estimator, 'monotonic_cst'):
            try:
                delattr(estimator, 'monotonic_cst')
                logger.info(f"Removed monotonic_cst from {type(estimator).__name__}")
                removed_count += 1
            except:
                pass
        
        # SPECIFIC FIX FOR RANDOM FOREST: Check individual trees
        if hasattr(estimator, 'estimators_') and estimator.estimators_ is not None:
            logger.info(f"Found {len(estimator.estimators_)} estimators in {type(estimator).__name__}")
            for i, tree in enumerate(estimator.estimators_):
                # More aggressive approach: remove ALL attributes containing "monotonic"
                attributes_to_remove = []
                for attr_name in dir(tree):
                    if 'monotonic' in attr_name.lower():
                        attributes_to_remove.append(attr_name)
                
                for attr_name in attributes_to_remove:
                    try:
                        if hasattr(tree, attr_name):
                            delattr(tree, attr_name)
                            logger.info(f"Removed {attr_name} from tree {i} in {type(estimator).__name__}")
                            removed_count += 1
                    except Exception as e:
                        logger.debug(f"Could not remove {attr_name} from tree {i}: {e}")
                
                # Also check nested estimators recursively
                removed_count += self._remove_monotonic_cst_from_estimator(tree)
        
        # Handle all possible nested estimator attributes
        nested_attributes = [
            'estimators_',           # RandomForest, VotingClassifier, etc.
            'base_estimator_',       # AdaBoost, etc. (deprecated)
            'estimator_',            # AdaBoost, etc. (new)
            'estimators',            # Some ensemble variations
            'base_estimator',        # Some ensemble variations
            'named_estimators_',     # VotingClassifier
            'estimators_features_',  # RandomForest subsampling
        ]
        
        for attr_name in nested_attributes:
            if hasattr(estimator, attr_name):
                nested_attr = getattr(estimator, attr_name)
                
                if nested_attr is not None:
                    # Handle different types of nested attributes
                    if isinstance(nested_attr, (list, tuple)):
                        # List/tuple of estimators
                        for nested_estimator in nested_attr:
                            if nested_estimator is not None:
                                removed_count += self._remove_monotonic_cst_from_estimator(nested_estimator)
                    
                    elif isinstance(nested_attr, dict):
                        # Dictionary of estimators (named_estimators_)
                        for name, nested_estimator in nested_attr.items():
                            if nested_estimator is not None:
                                removed_count += self._remove_monotonic_cst_from_estimator(nested_estimator)
                    
                    elif hasattr(nested_attr, '__dict__'):
                        # Single estimator object
                        removed_count += self._remove_monotonic_cst_from_estimator(nested_attr)
        
        # Handle Pipeline steps
        if hasattr(estimator, 'named_steps'):
            for step_name, step_estimator in estimator.named_steps.items():
                if step_estimator is not None:
                    removed_count += self._remove_monotonic_cst_from_estimator(step_estimator)
        
        if hasattr(estimator, 'steps'):
            for step_name, step_estimator in estimator.steps:
                if step_estimator is not None:
                    removed_count += self._remove_monotonic_cst_from_estimator(step_estimator)
        
        # ENHANCED: More aggressive approach - check all attributes recursively
        try:
            # Get all attributes of the estimator
            for attr_name in dir(estimator):
                if not attr_name.startswith('_'):  # Skip private attributes
                    try:
                        attr_value = getattr(estimator, attr_name)
                        
                        # Check if it's a scikit-learn estimator
                        if hasattr(attr_value, 'fit') and hasattr(attr_value, 'predict'):
                            removed_count += self._remove_monotonic_cst_from_estimator(attr_value)
                        
                        # Check if it's a list/tuple of estimators
                        elif isinstance(attr_value, (list, tuple)):
                            for item in attr_value:
                                if item is not None and hasattr(item, 'fit') and hasattr(item, 'predict'):
                                    removed_count += self._remove_monotonic_cst_from_estimator(item)
                        
                        # Check if it's a dict of estimators
                        elif isinstance(attr_value, dict):
                            for key, item in attr_value.items():
                                if item is not None and hasattr(item, 'fit') and hasattr(item, 'predict'):
                                    removed_count += self._remove_monotonic_cst_from_estimator(item)
                    
                    except (AttributeError, TypeError):
                        # Skip attributes that can't be accessed
                        continue
        except Exception as e:
            logger.debug(f"Error in aggressive attribute search: {e}")
        
        return removed_count
    
    def _load_compatible_model(self, model_path: str):
        """Load a model with maximum compatibility."""
        import sklearn
        from sklearn import __version__ as sklearn_version
        
        logger.info(f"Loading model with compatibility mode. sklearn version: {sklearn_version}")
        
        try:
            # Try loading with different sklearn configurations
            with sklearn.config_context(assume_finite=True):
                model = joblib.load(model_path)
                
            # Apply comprehensive fix
            model = self._fix_monotonic_cst_model(model)
            
            # Test prediction capability with standard feature size
            # Use common feature count based on typical ML models
            dummy_features = np.zeros((1, 50))  # Standard test size
            try:
                _ = model.predict(dummy_features)
                logger.info("Model compatibility test passed")
            except Exception as test_error:
                logger.warning(f"Model test failed: {test_error}")
                raise test_error
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load compatible model: {e}")
            raise e
    
    def _resolve_model_path(self, original_path: str) -> str:
        """Resolve model path from container format to local format if needed."""
        if Path(original_path).exists():
            return original_path
        
        # Try to convert container paths to local paths
        if original_path.startswith('/app/models/'):
            # Convert from container path to local path
            relative_path = original_path.replace('/app/models/', '')
            local_alternatives = [
                f"/Users/manojveluchuri/saas/workflow/shared-models/{relative_path}",
                f"../shared-models/{relative_path}",
                f"./models/{relative_path}",
                f"/tmp/models/{relative_path}"
            ]
            
            for alt_path in local_alternatives:
                if Path(alt_path).exists():
                    logger.info(f"Resolved model path: {original_path} -> {alt_path}")
                    return alt_path
        
        # Return original path if no alternatives found
        return original_path

# Legacy predictor for backwards compatibility
class Predictor:
    """Legacy predictor class for backwards compatibility."""
    
    def __init__(self):
        self.smart_predictor = SmartSongPredictor()
    
    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction using the smart predictor."""
        return await self.smart_predictor.predict_single_song(features)
    
    async def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make batch predictions."""
        return await self.smart_predictor.predict_batch(features_list)
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return await self.smart_predictor.get_model_info()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return await self.smart_predictor.health_check()

class MLPredictorService:
    """
    Handles ML model predictions for Hit Song Science.
    
    This service provides:
    - Real-time single predictions
    - Batch predictions
    - Model loading and caching
    - Prediction caching
    - Feature validation
    """
    
    def __init__(self):
        """Initialize the ML prediction service."""
        # Initialize feature translator
        try:
            # Let FeatureTranslator find schema using its fallback mechanism
            self.feature_translator = FeatureTranslator()
        except Exception as e:
            logger.error(f"Failed to initialize feature translator: {e}")
            raise ValueError(f"Cannot proceed with predictions - feature translator issues: {e}")
        
        # Initialize model cache
        self.model_cache = ModelCache(max_size=settings.MAX_CACHED_MODELS)
        
        # Initialize Redis for prediction caching
        try:
            self.redis_client = redis.from_url(settings.REDIS_URL)
            self.redis_client.ping()
            logger.info("Redis connection established for prediction caching")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Predictions will not be cached.")
            self.redis_client = None
        
        # Initialize MinIO for model storage
        try:
            self.minio_client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE
            )
            logger.info("MinIO connection established for model storage")
        except Exception as e:
            logger.warning(f"MinIO connection failed: {e}. Model loading may be limited.")
            self.minio_client = None
        
        # Prediction metrics
        self.metrics = {
            'total_predictions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_response_time': 0.0,
            'errors': 0,
            'models_loaded': 0
        }
        
        logger.info("ML Prediction Service initialized successfully with schema-based feature validation")
    
    async def predict_single(self, 
                           model_id: str,
                           features: FeatureInput,
                           include_confidence: bool = True,
                           include_feature_importance: bool = False,
                           use_cache: bool = True) -> Tuple[PredictionResult, bool]:
        """
        Make a single prediction.
        
        Args:
            model_id: Model identifier
            features: Feature input
            include_confidence: Include confidence intervals
            include_feature_importance: Include feature importance
            use_cache: Use cached predictions if available
            
        Returns:
            Tuple of (prediction result, was_cached)
        """
        start_time = time.time()
        prediction_id = str(uuid.uuid4())
        
        try:
            # Check cache first
            cached_result = None
            if use_cache and self.redis_client:
                cache_key = self._generate_cache_key(model_id, features)
                cached_result = await self._get_cached_prediction(cache_key)
                
                if cached_result:
                    self.metrics['cache_hits'] += 1
                    self.metrics['total_predictions'] += 1
                    processing_time = (time.time() - start_time) * 1000
                    
                    cached_result['processing_time_ms'] = processing_time
                    return PredictionResult(**cached_result), True
            
            self.metrics['cache_misses'] += 1
            
            # Load model
            model, model_metadata = await self.load_model(model_id)
            
            # Validate and prepare features
            feature_array, feature_names = await self._prepare_features(
                features, model_metadata['feature_columns']
            )
            
            # Make prediction
            prediction = model.predict(feature_array)[0]
            
            # Calculate confidence interval if requested
            confidence_interval = None
            if include_confidence:
                confidence_interval = await self._calculate_confidence_interval(
                    model, feature_array, prediction
                )
            
            # Get feature importance if requested
            feature_importance = None
            if include_feature_importance:
                feature_importance = await self._get_feature_importance(
                    model, feature_names
                )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Create result
            result = PredictionResult(
                hit_score=float(np.clip(prediction, 0, 1)),
                confidence_interval=confidence_interval,
                feature_importance=feature_importance,
                model_version=model_metadata.get('version', '1.0'),
                processing_time_ms=processing_time
            )
            
            # Cache result
            if use_cache and self.redis_client:
                cache_key = self._generate_cache_key(model_id, features)
                await self._cache_prediction(cache_key, result.dict())
            
            # Update metrics
            self.metrics['total_predictions'] += 1
            self.metrics['total_response_time'] += processing_time
            
            return result, False
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error in single prediction: {e}")
            raise
    
    async def predict_batch(self,
                          model_id: str,
                          features_list: List[FeatureInput],
                          include_confidence: bool = True,
                          include_feature_importance: bool = False) -> List[PredictionResult]:
        """
        Make batch predictions.
        
        Args:
            model_id: Model identifier
            features_list: List of feature inputs
            include_confidence: Include confidence intervals
            include_feature_importance: Include feature importance
            
        Returns:
            List of prediction results
        """
        start_time = time.time()
        
        try:
            # Load model
            model, model_metadata = await self.load_model(model_id)
            
            # Prepare all features
            feature_arrays = []
            for features in features_list:
                feature_array, feature_names = await self._prepare_features(
                    features, model_metadata['feature_columns']
                )
                feature_arrays.append(feature_array[0])  # Remove single-item dimension
            
            # Convert to numpy array for batch prediction
            batch_features = np.array(feature_arrays)
            
            # Make batch predictions
            predictions = model.predict(batch_features)
            
            # Process results
            results = []
            for i, prediction in enumerate(predictions):
                # Calculate confidence interval if requested
                confidence_interval = None
                if include_confidence:
                    single_feature = batch_features[i:i+1]
                    confidence_interval = await self._calculate_confidence_interval(
                        model, single_feature, prediction
                    )
                
                # Get feature importance if requested (only for first prediction to save time)
                feature_importance = None
                if include_feature_importance and i == 0:
                    feature_importance = await self._get_feature_importance(
                        model, feature_names
                    )
                
                result = PredictionResult(
                    hit_score=float(np.clip(prediction, 0, 1)),
                    confidence_interval=confidence_interval,
                    feature_importance=feature_importance if i == 0 else None,
                    model_version=model_metadata.get('version', '1.0'),
                    processing_time_ms=0  # Will be set later
                )
                results.append(result)
            
            # Set processing time for all results
            total_processing_time = (time.time() - start_time) * 1000
            avg_processing_time = total_processing_time / len(results)
            
            for result in results:
                result.processing_time_ms = avg_processing_time
            
            # Update metrics
            self.metrics['total_predictions'] += len(results)
            self.metrics['total_response_time'] += total_processing_time
            
            return results
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error in batch prediction: {e}")
            raise
    
    async def load_model(self, model_id: str, force_reload: bool = False) -> Tuple[Any, Dict[str, Any]]:
        """
        Load model from cache or storage.
        
        Args:
            model_id: Model identifier
            force_reload: Force reload from storage
            
        Returns:
            Tuple of (model, metadata)
        """
        # Check cache first
        if not force_reload:
            cached_model = self.model_cache.get(model_id)
            if cached_model:
                model, metadata = cached_model
                return model, metadata
        
        # Load from storage
        start_time = time.time()
        
        try:
            # Download model artifacts
            model_dir = Path(f"/tmp/models/{model_id}")
            await self._download_model_artifacts(model_id, model_dir)
            
            # Load model
            model_path = model_dir / "model.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model = joblib.load(model_path)
            
            # Load metadata
            metadata_path = model_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                # Create basic metadata
                metadata = {
                    'model_id': model_id,
                    'version': '1.0',
                    'feature_columns': self._infer_feature_columns(model)
                }
            
            load_time = time.time() - start_time
            
            # Cache model
            self.model_cache.put(model_id, (model, metadata), load_time)
            self.metrics['models_loaded'] += 1
            
            logger.info(f"Model {model_id} loaded successfully in {load_time:.2f}s")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            raise
    
    async def _download_model_artifacts(self, model_id: str, local_dir: Path) -> None:
        """Download model artifacts from MinIO."""
        if not self.minio_client:
            # Try local storage fallback
            local_model_dir = Path(f"/tmp/models/{model_id}")
            if local_model_dir.exists():
                return
            raise RuntimeError("No storage client available and no local model found")
        
        local_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"models/{model_id}/"
        
        try:
            # List and download all model files
            objects = self.minio_client.list_objects(
                settings.MINIO_BUCKET,
                prefix=prefix,
                recursive=True
            )
            
            for obj in objects:
                # Calculate local file path
                relative_path = obj.object_name[len(prefix):]
                local_file_path = local_dir / relative_path
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download file
                self.minio_client.fget_object(
                    settings.MINIO_BUCKET,
                    obj.object_name,
                    str(local_file_path)
                )
                
        except S3Error as e:
            if e.code == 'NoSuchKey':
                raise FileNotFoundError(f"Model {model_id} not found in storage")
            raise
    
    async def _prepare_features(self, 
                              features: FeatureInput, 
                              required_columns: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for prediction."""
        # Convert features to dictionary
        feature_dict = features.dict(exclude_none=True)
        
        # Add custom features
        if features.custom_features:
            feature_dict.update(features.custom_features)
        
        # Create feature array in the correct order
        feature_values = []
        feature_names = []
        
        for col in required_columns:
            if col in feature_dict:
                feature_values.append(feature_dict[col])
                feature_names.append(col)
            else:
                # Schema-based approach: Raise error instead of using hardcoded defaults
                raise ValueError(
                    f"Required feature '{col}' is missing. "
                    f"Schema-based feature validation requires all features to be properly extracted. "
                    f"Please ensure audio/content analysis is complete."
                )
        
        # Convert to numpy array
        feature_array = np.array(feature_values).reshape(1, -1)
        
        return feature_array, feature_names
    
    async def _calculate_confidence_interval(self,
                                           model: Any,
                                           features: np.ndarray,
                                           prediction: float) -> Dict[str, float]:
        """Calculate confidence interval for prediction."""
        try:
            # Simple confidence interval based on model type
            if hasattr(model, 'predict_proba'):
                # For probabilistic models
                probabilities = model.predict_proba(features)[0]
                std = np.std(probabilities)
            else:
                # For deterministic models, use a simple heuristic
                std = 0.1  # Default standard deviation
            
            # 95% confidence interval
            margin = 1.96 * std
            
            return {
                'lower': max(0.0, prediction - margin),
                'upper': min(1.0, prediction + margin),
                'margin': margin,
                'confidence_level': 0.95
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate confidence interval: {e}")
            return {
                'lower': max(0.0, prediction - 0.1),
                'upper': min(1.0, prediction + 0.1),
                'margin': 0.1,
                'confidence_level': 0.95
            }
    
    async def _get_feature_importance(self,
                                    model: Any,
                                    feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Get feature importance from model."""
        try:
            # Check if model has feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('model'), 'feature_importances_'):
                importances = model.named_steps['model'].feature_importances_
            else:
                return None
            
            # Create importance dictionary
            importance_dict = {}
            for name, importance in zip(feature_names, importances):
                importance_dict[name] = float(importance)
            
            return importance_dict
            
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
            return None
    
    def _generate_cache_key(self, model_id: str, features: FeatureInput) -> str:
        """Generate cache key for prediction."""
        # Create a hash of model_id and features
        feature_dict = features.dict(exclude_none=True)
        cache_data = {'model_id': model_id, 'features': feature_dict}
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    async def _get_cached_prediction(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction result."""
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Error getting cached prediction: {e}")
        return None
    
    async def _cache_prediction(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache prediction result."""
        try:
            self.redis_client.setex(
                cache_key,
                settings.CACHE_TTL_SECONDS,
                json.dumps(result, default=str)
            )
        except Exception as e:
            logger.warning(f"Error caching prediction: {e}")
    
    def _infer_feature_columns(self, model: Any) -> List[str]:
        """Infer feature columns from model."""
        # Try to get feature names from model
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        elif hasattr(model, 'named_steps'):
            # For pipelines, try to get from the last step
            last_step = list(model.named_steps.values())[-1]
            if hasattr(last_step, 'feature_names_in_'):
                return list(last_step.feature_names_in_)
        
        # Default feature columns
        return settings.REQUIRED_FEATURES
    
    async def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models."""
        models = []
        
        try:
            # Get models from MinIO
            if self.minio_client:
                model_ids = set()
                objects = self.minio_client.list_objects(
                    settings.MINIO_BUCKET,
                    prefix="models/",
                    recursive=True
                )
                
                for obj in objects:
                    path_parts = obj.object_name.split('/')
                    if len(path_parts) >= 2:
                        model_ids.add(path_parts[1])
                
                for model_id in model_ids:
                    try:
                        # Try to get metadata
                        metadata = await self._get_model_metadata(model_id)
                        is_cached = self.model_cache.get(model_id) is not None
                        
                        model_info = ModelInfo(
                            model_id=model_id,
                            model_name=metadata.get('model_name', model_id),
                            model_type=metadata.get('model_type', 'unknown'),
                            version=metadata.get('version', '1.0'),
                            accuracy_metrics=metadata.get('training_metrics', {}),
                            feature_columns=metadata.get('feature_columns', []),
                            last_trained=datetime.fromisoformat(metadata.get('created_at', datetime.utcnow().isoformat())),
                            is_cached=is_cached
                        )
                        models.append(model_info)
                        
                    except Exception as e:
                        logger.warning(f"Could not get metadata for model {model_id}: {e}")
            
            return models
            
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
    
    async def _get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """Get model metadata from storage."""
        try:
            if self.minio_client:
                # Download metadata from MinIO
                metadata_object = f"models/{model_id}/metadata.json"
                temp_path = f"/tmp/{model_id}_metadata.json"
                
                self.minio_client.fget_object(
                    settings.MINIO_BUCKET,
                    metadata_object,
                    temp_path
                )
                
                with open(temp_path, 'r') as f:
                    metadata = json.load(f)
                
                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)
                
                return metadata
            else:
                # Try local storage
                local_metadata_path = Path(f"/tmp/models/{model_id}/metadata.json")
                if local_metadata_path.exists():
                    with open(local_metadata_path, 'r') as f:
                        return json.load(f)
        
        except Exception as e:
            logger.warning(f"Could not load metadata for model {model_id}: {e}")
        
        # Return basic metadata
        return {
            'model_id': model_id,
            'version': '1.0',
            'feature_columns': settings.REQUIRED_FEATURES
        }
    
    async def validate_features(self,
                              features: FeatureInput,
                              model_id: Optional[str] = None) -> Dict[str, Any]:
        """Validate feature input."""
        result = {
            'is_valid': True,
            'missing_features': [],
            'invalid_features': [],
            'warnings': [],
            'feature_count': 0
        }
        
        # Convert features to dict
        feature_dict = features.dict(exclude_none=True)
        result['feature_count'] = len(feature_dict)
        
        # Check required features
        for required_feature in settings.REQUIRED_FEATURES:
            if required_feature not in feature_dict:
                result['missing_features'].append(required_feature)
                result['is_valid'] = False
        
        # If model specified, check model-specific requirements
        if model_id:
            try:
                _, metadata = await self.load_model(model_id)
                model_features = metadata.get('feature_columns', [])
                
                for feature in model_features:
                    if feature not in feature_dict:
                        result['warnings'].append(f"Model {model_id} expects feature '{feature}' but it's not provided")
                        
            except Exception as e:
                result['warnings'].append(f"Could not validate against model {model_id}: {str(e)}")
        
        # Check feature value ranges
        for feature, value in feature_dict.items():
            if isinstance(value, (int, float)):
                if feature in ['energy', 'danceability', 'valence', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']:
                    if not 0 <= value <= 1:
                        result['invalid_features'].append(f"{feature} should be between 0 and 1")
                        result['is_valid'] = False
                elif feature == 'tempo':
                    if not 40 <= value <= 250:
                        result['warnings'].append(f"Tempo {value} is outside typical range (40-250 BPM)")
                elif feature == 'loudness':
                    if not -60 <= value <= 0:
                        result['warnings'].append(f"Loudness {value} is outside typical range (-60 to 0 dB)")
        
        return result
    
    async def get_metrics(self) -> PredictionMetrics:
        """Get prediction service metrics."""
        cache_hit_rate = 0.0
        if self.metrics['total_predictions'] > 0:
            cache_hit_rate = self.metrics['cache_hits'] / self.metrics['total_predictions']
        
        avg_response_time = 0.0
        if self.metrics['total_predictions'] > 0:
            avg_response_time = self.metrics['total_response_time'] / self.metrics['total_predictions']
        
        error_rate = 0.0
        if self.metrics['total_predictions'] > 0:
            error_rate = self.metrics['errors'] / self.metrics['total_predictions']
        
        return PredictionMetrics(
            total_predictions=self.metrics['total_predictions'],
            cache_hit_rate=cache_hit_rate,
            average_response_time_ms=avg_response_time,
            models_loaded=len(self.model_cache.cache),
            active_sessions=1,  # Single service instance
            error_rate=error_rate
        ) 