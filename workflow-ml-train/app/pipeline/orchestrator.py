"""
Pipeline Orchestrator for coordinating ML training workflows.

Manages the execution of training pipelines with:
- Dynamic strategy selection
- Service discovery and feature agreement
- Stage coordination and error handling  
- MLflow integration for experiment tracking
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
from pathlib import Path
import json
import os
import sys
import httpx
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import mlflow
import mlflow.sklearn
import joblib
import time

# Import feature translator and derived features calculator as proper packages
from hss_feature_translator import FeatureTranslator
from hss_derived_features import DerivedFeaturesCalculator

# Import our services
from ..services.song_analyzer import SongAnalyzer
from ..api.pipeline import PipelineStatus
from ..utils.target_engineering import ContinuousHitScoreEngineer
from ..utils.regression_metrics import HitSongRegressionMetrics

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """
    Main orchestrator for ML training pipelines.
    
    Coordinates the execution of training workflows across multiple stages
    with dynamic strategy selection and comprehensive tracking.
    """
    
    def __init__(self):
        """Initialize the orchestrator"""
        self.active_pipelines: Dict[str, Dict[str, Any]] = {}
        self.pipeline_history: List[Dict[str, Any]] = []
        
        # Initialize feature translator and derived features calculator
        try:
            # Let FeatureTranslator find schema using its fallback mechanism
            self.feature_translator = FeatureTranslator()
            # Initialize derived features calculator (shared library)
            self.derived_features_calculator = DerivedFeaturesCalculator()
        except Exception as e:
            logger.error(f"Failed to initialize feature translator or derived features calculator: {e}")
            raise ValueError(f"Cannot proceed with training - feature translator/derived features issues: {e}")
        
        # Initialize services
        self.song_analyzer = SongAnalyzer(
            cache_dir="./cache",
            base_data_dir="./shared-data"
        )
        
        # Pipeline stages
        self.stages = [
            "service_discovery",
            "feature_agreement", 
            "feature_extraction",
            "model_training",
            "model_registry"
        ]
        
        # Interactive pipeline state
        self.pending_agreements: Dict[str, Dict[str, Any]] = {}  # Pipelines waiting for feature agreement
        
        # Strategy configurations
        self.strategy_configs = {
            "audio_only": {
                "services": ["audio"],
                "csv_path": "/app/data/training_data/filtered/filtered_audio_only_corrected_20250621_180350.csv",
                "features": "auto_select_audio"
            },
            "multimodal": {
                "services": ["audio", "content"],
                "csv_path": "/app/data/training_data/filtered/filtered_multimodal_corrected_20250621_180350.csv",
                "features": "auto_select_all"
            },
            "custom": {
                "services": [],  # To be set by user
                "csv_path": "/app/data/training_data/filtered/filtered_audio_only_corrected_20250621_180350.csv",
                "features": []  # To be set by user
            }
        }
        
        # Remove all hardcoded mappings - use schema-based translation
        
        logger.info("🎼 Pipeline Orchestrator initialized with schema-based feature translation")
    
    async def process_song_features(self, song_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process song features using schema-based translation"""
        processed_features = {}
        
        # Extract and translate audio features
        if 'audio_analysis' in song_data:
            try:
                audio_features = self.feature_translator.audio_producer_to_consumer(
                    song_data['audio_analysis']
                )
                validated_audio = self.feature_translator.validate_consumer_features(
                    audio_features, 'audio'
                )
                processed_features.update(validated_audio)
                
            except Exception as e:
                logger.error(f"Audio feature translation failed: {e}")
                raise ValueError(f"Cannot proceed with training - audio feature issues: {e}")
        
        # Extract and translate content features  
        if 'content_analysis' in song_data:
            try:
                content_features = self.feature_translator.content_producer_to_consumer(
                    song_data['content_analysis']
                )
                validated_content = self.feature_translator.validate_consumer_features(
                    content_features, 'content'
                )
                processed_features.update(validated_content)
                
            except Exception as e:
                logger.error(f"Content feature translation failed: {e}")
                raise ValueError(f"Cannot proceed with training - content feature issues: {e}")
        
        # Verify minimum features for training
        required_audio = self.feature_translator.get_required_features('audio')
        required_content = self.feature_translator.get_required_features('content')
        
        missing_features = []
        missing_features.extend([f for f in required_audio if f not in processed_features])
        missing_features.extend([f for f in required_content if f not in processed_features])
        
        if missing_features:
            raise ValueError(
                f"Cannot train model - missing required features: {missing_features}. "
                f"Available features: {list(processed_features.keys())}"
            )
        
        logger.info(f"Successfully processed {len(processed_features)} features for training")
        return processed_features
    
    def _get_available_features(self, df: pd.DataFrame, selected_features: List[str]) -> List[str]:
        """Get list of features that are actually available in the dataframe
        
        Args:
            df: DataFrame containing the data
            selected_features: List of requested feature names (should be consumer names)
            
        Returns:
            List of features that exist in the dataframe
        """
        # selected_features should already be in consumer format (audio_*, content_*)
        # Filter to only include features that exist in the dataframe
        available_features = [f for f in selected_features if f in df.columns]
        
        if len(available_features) != len(selected_features):
            missing_features = set(selected_features) - set(available_features)
            logger.warning(f"⚠️ Missing features in dataframe: {missing_features}")
            logger.info(f"📊 Available features: {len(available_features)}/{len(selected_features)}")
            logger.info(f"📊 DataFrame columns: {list(df.columns)[:10]}...")  # Show first 10 columns
        
        return available_features
    
    async def start_pipeline(
        self,
        pipeline_id: str,
        strategy: str,
        experiment_name: str,
        features: Optional[List[str]] = None,
        model_types: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Start a new training pipeline with the specified strategy.
        
        Args:
            pipeline_id: Unique pipeline identifier
            strategy: Training strategy (audio_only, multimodal, custom)
            experiment_name: MLflow experiment name
            features: Custom feature selection (for custom strategy)
            model_types: Model types to train
            parameters: Additional parameters
            
        Returns:
            Pipeline execution details
        """
        try:
            logger.info(f"🚀 Starting pipeline {pipeline_id} with strategy '{strategy}'")
            
            # Initialize pipeline state
            pipeline_state = {
                "pipeline_id": pipeline_id,
                "strategy": strategy,
                "experiment_name": experiment_name,
                "features": features or [],
                "model_types": model_types or ["random_forest"],
                "parameters": parameters or {},
                "status": "starting",
                "current_stage": None,
                "start_time": datetime.now(),
                "end_time": None,
                "stages": {stage: {"status": "pending", "start_time": None, "end_time": None} 
                          for stage in self.stages},
                "error_message": None
            }
            
            # Store pipeline state
            self.active_pipelines[pipeline_id] = pipeline_state
            
            # Start pipeline execution in background
            asyncio.create_task(self._execute_pipeline(pipeline_id))
            
            return {
                "pipeline_id": pipeline_id,
                "status": "starting",
                "message": f"Pipeline started with strategy '{strategy}'"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to start pipeline {pipeline_id}: {e}")
            raise
    
    async def _execute_pipeline(self, pipeline_id: str):
        """Execute the pipeline stages in sequence"""
        try:
            pipeline_state = self.active_pipelines[pipeline_id]
            pipeline_state["status"] = "running"
            
            logger.info(f"⚙️ Executing pipeline {pipeline_id}")
            
            # Execute each stage in sequence
            for stage in self.stages:
                pipeline_state["current_stage"] = stage
                pipeline_state["stages"][stage]["status"] = "running"
                pipeline_state["stages"][stage]["start_time"] = datetime.now()
                
                logger.info(f"🔄 Pipeline {pipeline_id}: Starting stage '{stage}'")
                
                # Execute stage
                result = await self._execute_stage(pipeline_id, stage)
                
                if result == "waiting_for_input":
                    # Special case: stage is waiting for user input, pause pipeline
                    pipeline_state["stages"][stage]["status"] = "waiting_for_input"
                    pipeline_state["status"] = "waiting_for_input"
                    logger.info(f"⏸️ Pipeline {pipeline_id}: Stage '{stage}' waiting for user input")
                    break
                elif result:
                    pipeline_state["stages"][stage]["status"] = "completed"
                    pipeline_state["stages"][stage]["end_time"] = datetime.now()
                    logger.info(f"✅ Pipeline {pipeline_id}: Completed stage '{stage}'")
                else:
                    pipeline_state["stages"][stage]["status"] = "failed"
                    pipeline_state["stages"][stage]["end_time"] = datetime.now()
                    pipeline_state["status"] = "failed"
                    pipeline_state["error_message"] = f"Stage '{stage}' failed"
                    logger.error(f"❌ Pipeline {pipeline_id}: Stage '{stage}' failed")
                    break
            
            # Mark pipeline as completed if all stages succeeded
            if pipeline_state["status"] == "running":
                pipeline_state["status"] = "completed"
                pipeline_state["end_time"] = datetime.now()
                logger.info(f"🎉 Pipeline {pipeline_id} completed successfully")
            
            # Move to history
            self.pipeline_history.append(pipeline_state.copy())
            
        except Exception as e:
            logger.error(f"❌ Pipeline {pipeline_id} execution failed: {e}")
            if pipeline_id in self.active_pipelines:
                self.active_pipelines[pipeline_id]["status"] = "failed"
                self.active_pipelines[pipeline_id]["error_message"] = str(e)
                self.active_pipelines[pipeline_id]["end_time"] = datetime.now()
    
    async def _execute_stage(self, pipeline_id: str, stage: str) -> bool:
        """Execute a specific pipeline stage"""
        try:
            pipeline_state = self.active_pipelines[pipeline_id]
            strategy = pipeline_state["strategy"]
            
            # Simulate stage execution based on stage type
            if stage == "service_discovery":
                return await self._execute_service_discovery(pipeline_id, strategy)
            elif stage == "feature_agreement":
                return await self._execute_feature_agreement(pipeline_id, strategy)
            elif stage == "feature_extraction":
                return await self._execute_feature_extraction(pipeline_id, strategy)
            elif stage == "model_training":
                return await self._execute_model_training(pipeline_id, strategy)
            elif stage == "model_registry":
                return await self._execute_model_registry(pipeline_id, strategy)
            else:
                logger.warning(f"⚠️ Unknown stage: {stage}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Stage {stage} execution failed: {e}")
            return False
    
    async def _execute_service_discovery(self, pipeline_id: str, strategy: str) -> bool:
        """Execute service discovery stage"""
        logger.info(f"🔍 Pipeline {pipeline_id}: Discovering services for strategy '{strategy}'")
        
        try:
            pipeline_state = self.active_pipelines[pipeline_id]
            
            # Get strategy configuration
            strategy_config = self.strategy_configs.get(strategy, {})
            if strategy == "custom":
                services_to_use = pipeline_state.get("custom_services", ["audio"])
            else:
                services_to_use = strategy_config.get("services", ["audio"])
            
            logger.info(f"🔍 Discovering features from services: {services_to_use}")
            
            # Use SongAnalyzer's service discovery
            await self.song_analyzer._discover_service_features(services_to_use)
            
            # Store discovered features in pipeline state
            discovered_features = {}
            total_features = 0
            
            for service in services_to_use:
                if service in self.song_analyzer.feature_schemas:
                    schema = self.song_analyzer.feature_schemas[service]
                    capabilities = schema.get('capabilities', {})
                    feature_count = capabilities.get('total_features', 0)
                    
                    discovered_features[service] = {
                        'total_features': feature_count,
                        'categories': list(capabilities.get('categories', {}).keys()),
                        'status': 'available'
                    }
                    total_features += feature_count
                    
                    logger.info(f"✅ {service} service: {feature_count} features available")
                else:
                    discovered_features[service] = {
                        'total_features': 0,
                        'categories': [],
                        'status': 'unavailable'
                    }
                    logger.warning(f"⚠️ {service} service: unavailable")
            
            pipeline_state["discovered_services"] = discovered_features
            pipeline_state["total_available_features"] = total_features
            
            logger.info(f"🎉 Service discovery completed: {total_features} total features from {len([s for s in discovered_features.values() if s['status'] == 'available'])} services")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Service discovery failed: {e}")
            pipeline_state = self.active_pipelines.get(pipeline_id, {})
            pipeline_state["discovery_error"] = str(e)
            return False
    
    async def _execute_feature_agreement(self, pipeline_id: str, strategy: str) -> bool:
        """Execute feature agreement stage - handles pre-existing vectors, auto mode, or interactive mode"""
        logger.info(f"📝 Pipeline {pipeline_id}: Creating feature agreement for strategy '{strategy}'")
        
        try:
            pipeline_state = self.active_pipelines[pipeline_id]
            parameters = pipeline_state.get("parameters", {})
            
            # Debug logging for parameters
            logger.info(f"🔍 DEBUG: Pipeline {pipeline_id} parameters: {parameters}")
            
            # Check if pre-existing feature vector is provided
            feature_vector = parameters.get("feature_vector")
            skip_agreement = parameters.get("skip_feature_agreement", False)
            auto_mode = parameters.get("auto_feature_selection", False)
            
            logger.info(f"🔍 DEBUG: feature_vector={feature_vector}, skip_agreement={skip_agreement}, auto_mode={auto_mode}")
            
            if feature_vector:
                # Use pre-existing feature vector
                logger.info(f"📄 Using pre-existing feature vector")
                return await self._execute_predefined_feature_agreement(pipeline_id, strategy, feature_vector)
            elif skip_agreement or auto_mode:
                # Automatic feature selection (legacy mode)
                logger.info(f"🤖 Using automatic feature selection (skip={skip_agreement}, auto={auto_mode})")
                return await self._execute_automatic_feature_agreement(pipeline_id, strategy)
            else:
                # Interactive feature agreement - wait for user input
                logger.info(f"👤 Using interactive feature agreement")
                return await self._execute_interactive_feature_agreement(pipeline_id, strategy)
                
        except Exception as e:
            logger.error(f"❌ Feature agreement failed: {e}")
            pipeline_state = self.active_pipelines.get(pipeline_id, {})
            pipeline_state["agreement_error"] = str(e)
            return False
    
    async def _execute_predefined_feature_agreement(self, pipeline_id: str, strategy: str, feature_vector: Dict[str, Any]) -> bool:
        """Execute feature agreement using pre-existing feature vector"""
        logger.info(f"📄 Pipeline {pipeline_id}: Using pre-existing feature vector")
        
        try:
            pipeline_state = self.active_pipelines[pipeline_id]
            
            # Import feature vector manager
            from ..utils.feature_vector_manager import feature_vector_manager
            
            # Convert feature vector to flat list
            selected_features = feature_vector_manager.convert_to_flat_features(feature_vector)
            
            if not selected_features:
                logger.error(f"❌ No features found in pre-existing feature vector")
                return False
            
            # Override strategy if specified in feature vector
            if "strategy" in feature_vector and feature_vector["strategy"] != strategy:
                logger.info(f"🔄 Overriding strategy from '{strategy}' to '{feature_vector['strategy']}'")
                pipeline_state["strategy"] = feature_vector["strategy"]
                strategy = feature_vector["strategy"]
            
            # Create feature agreement
            feature_agreement = {
                "agreement_id": f"{pipeline_id}_agreement",
                "strategy": strategy,
                "selected_features": selected_features,
                "feature_count": len(selected_features),
                "created_at": datetime.now().isoformat(),
                "mode": "predefined",
                "source": "pre_existing_feature_vector",
                "original_feature_vector": feature_vector
            }
            
            pipeline_state["feature_agreement"] = feature_agreement
            
            logger.info(f"✅ Pre-existing feature vector applied: {len(selected_features)} features")
            logger.info(f"📋 Features: {', '.join(selected_features[:10])}{'...' if len(selected_features) > 10 else ''}")
            logger.info(f"🎯 Strategy: {strategy}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to apply pre-existing feature vector: {e}")
            return False
    
    async def _execute_automatic_feature_agreement(self, pipeline_id: str, strategy: str) -> bool:
        """Automatic feature selection (no user interaction)"""
        pipeline_state = self.active_pipelines[pipeline_id]
        parameters = pipeline_state.get("parameters", {})
        
        # Get strategy configuration
        strategy_config = self.strategy_configs.get(strategy, {})
        
        # Determine feature selection based on strategy and user input
        ui_features = parameters.get("features", [])  # Features selected in UI
        
        if ui_features:
            # User provided features in UI - use those
            selected_features = ui_features
            logger.info(f"🎯 Using UI-selected features: {len(selected_features)} features")
        elif strategy == "custom":
            # For custom strategy, require features
            selected_features = pipeline_state.get("features", [])
            if not selected_features:
                logger.warning("⚠️ No features selected for custom strategy")
                return False
        else:
            # Auto-select features based on strategy
            selected_features = self._auto_select_features(strategy, pipeline_state)
            logger.info(f"🤖 Auto-selected features for '{strategy}' strategy")
        
        # Store feature agreement
        feature_agreement = {
            "agreement_id": f"{pipeline_id}_agreement",
            "strategy": strategy,
            "selected_features": selected_features,
            "feature_count": len(selected_features),
            "created_at": datetime.now().isoformat(),
            "mode": "automatic" if not ui_features else "ui_selected"
        }
        
        pipeline_state["feature_agreement"] = feature_agreement
        
        logger.info(f"✅ Feature agreement created: {len(selected_features)} features selected")
        logger.info(f"📋 Selected features: {', '.join(selected_features[:10])}{'...' if len(selected_features) > 10 else ''}")
        
        return True
    
    async def _execute_interactive_feature_agreement(self, pipeline_id: str, strategy: str) -> bool:
        """Interactive feature agreement - waits for user input"""
        pipeline_state = self.active_pipelines[pipeline_id]
        
        # Create feature agreement request
        discovered_services = pipeline_state.get("discovered_services", {})
        
        # Prepare available features for user selection
        available_features = {}
        for service_name, service_info in discovered_services.items():
            if service_info.get("status") == "available":
                # Get feature schema from song analyzer
                if service_name in self.song_analyzer.feature_schemas:
                    schema = self.song_analyzer.feature_schemas[service_name]
                    features = schema.get("features", {})
                    available_features[service_name] = features
        
        # Create pending agreement
        agreement_request = {
            "pipeline_id": pipeline_id,
            "strategy": strategy,
            "available_features": available_features,
            "services_discovered": list(discovered_services.keys()),
            "status": "waiting_for_user_input",
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now().timestamp() + 3600),  # 1 hour timeout
            "feature_agreement_url": f"/features/agreement/{pipeline_id}",
            "ui_url": f"/ui/feature-selection/{pipeline_id}"
        }
        
        # Store pending agreement
        self.pending_agreements[pipeline_id] = agreement_request
        pipeline_state["feature_agreement_status"] = "pending_user_input"
        pipeline_state["agreement_request"] = agreement_request
        
        logger.info(f"⏸️ Pipeline {pipeline_id} waiting for feature selection")
        logger.info(f"🔗 Feature agreement URL: /features/agreement/{pipeline_id}")
        logger.info(f"🖥️ UI URL: /ui/feature-selection/{pipeline_id}")
        
        # Wait for user input (this will be completed by external API call)
        return "waiting_for_input"  # Special return value indicating pipeline should pause
    
    def complete_feature_agreement(self, pipeline_id: str, selected_features: List[str], agreement_data: Dict[str, Any]) -> bool:
        """Complete feature agreement when user makes selection"""
        try:
            if pipeline_id not in self.pending_agreements:
                logger.error(f"❌ No pending agreement found for pipeline {pipeline_id}")
                return False
            
            pipeline_state = self.active_pipelines.get(pipeline_id)
            if not pipeline_state:
                logger.error(f"❌ Pipeline {pipeline_id} not found")
                return False
            
            # Create feature agreement
            feature_agreement = {
                "agreement_id": f"{pipeline_id}_agreement",
                "strategy": pipeline_state["strategy"],
                "selected_features": selected_features,
                "feature_count": len(selected_features),
                "created_at": datetime.now().isoformat(),
                "mode": "interactive",
                "user_agreement_data": agreement_data
            }
            
            # Store agreement and remove from pending
            pipeline_state["feature_agreement"] = feature_agreement
            pipeline_state["feature_agreement_status"] = "completed"
            del self.pending_agreements[pipeline_id]
            
            logger.info(f"✅ Feature agreement completed for pipeline {pipeline_id}: {len(selected_features)} features selected")
            
            # Resume pipeline execution
            asyncio.create_task(self._resume_pipeline_after_agreement(pipeline_id))
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to complete feature agreement: {e}")
            return False
    
    async def _resume_pipeline_after_agreement(self, pipeline_id: str):
        """Resume pipeline execution after feature agreement is completed"""
        try:
            pipeline_state = self.active_pipelines.get(pipeline_id)
            if not pipeline_state:
                return
            
            logger.info(f"▶️ Resuming pipeline {pipeline_id} after feature agreement")
            
            # Continue with remaining stages
            current_stage_idx = self.stages.index("feature_agreement")
            remaining_stages = self.stages[current_stage_idx + 1:]
            
            for stage in remaining_stages:
                pipeline_state["current_stage"] = stage
                pipeline_state["stages"][stage]["status"] = "running"
                pipeline_state["stages"][stage]["start_time"] = datetime.now()
                
                logger.info(f"🔄 Pipeline {pipeline_id}: Starting stage '{stage}'")
                
                # Execute stage
                success = await self._execute_stage(pipeline_id, stage)
                
                if success:
                    pipeline_state["stages"][stage]["status"] = "completed"
                    pipeline_state["stages"][stage]["end_time"] = datetime.now()
                    logger.info(f"✅ Pipeline {pipeline_id}: Completed stage '{stage}'")
                else:
                    pipeline_state["stages"][stage]["status"] = "failed"
                    pipeline_state["stages"][stage]["end_time"] = datetime.now()
                    pipeline_state["status"] = "failed"
                    pipeline_state["error_message"] = f"Stage '{stage}' failed"
                    logger.error(f"❌ Pipeline {pipeline_id}: Stage '{stage}' failed")
                    break
            
            # Mark pipeline as completed if all stages succeeded
            if pipeline_state["status"] == "running":
                pipeline_state["status"] = "completed"
                pipeline_state["end_time"] = datetime.now()
                logger.info(f"🎉 Pipeline {pipeline_id} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to resume pipeline {pipeline_id}: {e}")
            if pipeline_id in self.active_pipelines:
                self.active_pipelines[pipeline_id]["status"] = "failed"
                self.active_pipelines[pipeline_id]["error_message"] = f"Resume failed: {str(e)}"
    
    def _auto_select_features(self, strategy: str, pipeline_state: Dict[str, Any]) -> List[str]:
        """Auto-select features based on strategy"""
        try:
            discovered_services = pipeline_state.get("discovered_services", {})
            
            # Define feature selection patterns for each strategy
            if strategy == "audio_only":
                # Select key audio features from comprehensive analyzer
                audio_features = [
                    "audio_energy", "audio_valence", "audio_danceability",
                    "audio_tempo", "audio_acousticness", "audio_instrumentalness",
                    "audio_liveness", "audio_loudness", "audio_speechiness",
                    "audio_brightness", "audio_complexity", "audio_warmth",
                    "audio_key", "audio_mode", "audio_harmonic_strength",
                    "audio_mood_happy", "audio_mood_sad", "audio_mood_aggressive",
                    "audio_mood_relaxed", "audio_mood_party", "audio_mood_electronic",
                    "audio_primary_genre", "audio_top_genre_1_prob", "audio_top_genre_2_prob"
                ]
                return audio_features
                
            elif strategy == "multimodal":
                # COMPREHENSIVE BALANCED: Raw features only - derived features added later by DerivedFeaturesCalculator
                multimodal_features = [
                    # === AUDIO FEATURES (24 features) ===
                    "audio_tempo",
                    "audio_energy", 
                    "audio_valence",
                    "audio_danceability",
                    "audio_loudness",
                    "audio_speechiness",
                    "audio_acousticness",
                    "audio_instrumentalness", 
                    "audio_liveness",
                    "audio_key",
                    "audio_mode",
                    "audio_brightness",
                    "audio_complexity",
                    "audio_warmth",
                    "audio_harmonic_strength",
                    "audio_mood_happy",
                    "audio_mood_sad",
                    "audio_mood_aggressive",
                    "audio_mood_relaxed",
                    "audio_mood_party",
                    "audio_mood_electronic", 
                    "audio_primary_genre",
                    "audio_top_genre_1_prob",
                    "audio_top_genre_2_prob",
                    
                    # === LYRICAL FEATURES (27 features) ===
                    "lyrics_sentiment_positive",
                    "lyrics_sentiment_negative",
                    "lyrics_sentiment_neutral",
                    "lyrics_complexity_score",
                    "lyrics_word_count",
                    "lyrics_unique_words",
                    "lyrics_reading_level",
                    "lyrics_emotion_anger",
                    "lyrics_emotion_joy",
                    "lyrics_emotion_sadness",
                    "lyrics_emotion_fear",
                    "lyrics_emotion_surprise",
                    "lyrics_theme_love",
                    "lyrics_theme_party",
                    "lyrics_theme_sadness",
                    "lyrics_profanity_score",
                    "lyrics_repetition_score",
                    "lyrics_rhyme_density",
                    "lyrics_narrative_complexity",
                    "lyrics_lexical_diversity",
                    "lyrics_motif_count",
                    "lyrics_verse_count",
                    "lyrics_chorus_count",
                    "lyrics_bridge_count"
                ]
                
                # NOTE: Derived features (rhythmic_appeal_index, emotional_impact_score, 
                # commercial_viability_index, sonic_sophistication_score) will be added 
                # automatically later by the DerivedFeaturesCalculator
                
                return multimodal_features
            
            else:
                # Default to audio features
                return ["audio_energy", "audio_valence", "audio_tempo"]
                
        except Exception as e:
            logger.error(f"❌ Auto feature selection failed: {e}")
            return ["audio_energy", "audio_valence", "audio_tempo"]
    
    async def _execute_feature_extraction(self, pipeline_id: str, strategy: str) -> bool:
        """Execute feature extraction stage"""
        logger.info(f"🎵 Pipeline {pipeline_id}: Extracting features for strategy '{strategy}'")
        
        try:
            pipeline_state = self.active_pipelines[pipeline_id]
            
            # Get strategy configuration
            strategy_config = self.strategy_configs.get(strategy, {})
            if strategy == "custom":
                # For custom strategy, use user-provided services and features
                services_to_use = pipeline_state.get("custom_services", ["audio"])
                csv_path = pipeline_state.get("custom_csv_path", strategy_config["csv_path"])
            else:
                services_to_use = strategy_config.get("services", ["audio"])
                # FIXED: Check for dataset_path parameter override from Airflow DAG
                csv_path = pipeline_state.get("parameters", {}).get("dataset_path", strategy_config["csv_path"])
            
            logger.info(f"📊 Using services: {services_to_use}")
            logger.info(f"📄 CSV path: {csv_path}")
            
            # Check if CSV exists
            if not Path(csv_path).exists():
                logger.error(f"❌ CSV file not found: {csv_path}")
                return False
            
            # Extract features using SongAnalyzer
            agreement_id = f"{pipeline_id}_agreement"
            force_extract = pipeline_state.get("parameters", {}).get("force_extract", False)
            
            logger.info(f"🔄 Starting feature extraction with SongAnalyzer...")
            raw_features_df, extraction_report = await self.song_analyzer.analyze_songs_from_csv(
                csv_path=csv_path,
                services_to_use=services_to_use,
                agreement_id=agreement_id,
                force_extract=force_extract
            )
            
            # Store extraction results in pipeline state
            pipeline_state["extraction_report"] = {
                "total_songs": extraction_report["total_songs"],
                "successful_extractions": extraction_report["successful_extractions"],
                "failed_extractions": extraction_report["failed_extractions"],
                "extraction_errors": extraction_report["extraction_errors"],
                "services_used": extraction_report["services_used"],
                "duration_seconds": extraction_report.get("total_duration", 0)
            }
            
            pipeline_state["raw_features_shape"] = raw_features_df.shape
            pipeline_state["available_features"] = list(raw_features_df.columns)
            
            # Create training matrix based on feature agreement
            feature_agreement = pipeline_state.get("feature_agreement", {})
            selected_features = feature_agreement.get("selected_features", [])
            agreement_id = feature_agreement.get("agreement_id", f"{pipeline_id}_agreement")
            
            if selected_features:
                logger.info(f"🔧 Creating training matrix with {len(selected_features)} selected features")
                training_matrix = self.song_analyzer.create_training_matrix(
                    raw_features_df=raw_features_df,
                    selected_features=selected_features,
                    agreement_id=agreement_id
                )
                
                pipeline_state["training_matrix_shape"] = training_matrix.shape
                pipeline_state["final_features"] = list(training_matrix.columns)
                pipeline_state["training_matrix_cache_key"] = f"{pipeline_id}_training_matrix"
                
                logger.info(f"✅ Training matrix created: {training_matrix.shape}")
            else:
                logger.warning("⚠️ No feature agreement found, using raw features")
                training_matrix = raw_features_df
                pipeline_state["training_matrix_shape"] = training_matrix.shape
                pipeline_state["final_features"] = list(training_matrix.columns)
            
            # Cache the raw features DataFrame for later stages
            pipeline_state["raw_features_cache_key"] = f"{pipeline_id}_raw_features"
            
            logger.info(f"✅ Feature extraction completed: {extraction_report['successful_extractions']}/{extraction_report['total_songs']} songs")
            logger.info(f"📊 Raw features shape: {raw_features_df.shape}")
            logger.info(f"🎯 Final training matrix shape: {training_matrix.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            pipeline_state = self.active_pipelines.get(pipeline_id, {})
            pipeline_state["extraction_error"] = str(e)
            return False
    
    async def _execute_model_training(self, pipeline_id: str, strategy: str) -> bool:
        """Execute model training stage with ensemble models (Random Forest + XGBoost) and SHAP explainability"""
        logger.info(f"🧠 Pipeline {pipeline_id}: Training ensemble models for strategy '{strategy}'")
        
        try:
            import mlflow
            import mlflow.sklearn
            import pandas as pd
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier, VotingClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            import xgboost as xgb
            import shap
            import os
            import json
            from pathlib import Path
            
            pipeline_state = self.active_pipelines[pipeline_id]
            
            # Set MLflow tracking URI - use host.docker.internal for cross-container access
            mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://host.docker.internal:5001')
            mlflow.set_tracking_uri(mlflow_uri)
            logger.info(f"🔗 MLflow tracking URI: {mlflow_uri}")
            
            # Get experiment name
            experiment_name = pipeline_state.get("experiment_name", f"pipeline_{strategy}")
            
            # Set or create experiment
            try:
                experiment = mlflow.set_experiment(experiment_name)
                logger.info(f"📊 Using MLflow experiment: {experiment_name}")
            except Exception as e:
                logger.warning(f"⚠️ Could not set experiment, using default: {e}")
                experiment = mlflow.set_experiment("Default")
            
            # Get training data (assuming it was cached during feature extraction)
            training_matrix_key = pipeline_state.get("training_matrix_cache_key")
            if not training_matrix_key:
                logger.error("❌ No training matrix found in pipeline state")
                return False
            
            # For this implementation, we'll use the CSV data and selected features
            strategy_config = self.strategy_configs.get(strategy, {})
            # FIXED: Check for dataset_path parameter override from Airflow DAG
            csv_path = pipeline_state.get("parameters", {}).get("dataset_path", strategy_config.get("csv_path", "/app/data/training_data/filtered/filtered_audio_only_corrected_20250621_180350.csv"))
            
            # Get selected features from feature agreement
            feature_agreement = pipeline_state.get("feature_agreement", {})
            selected_features = feature_agreement.get("selected_features", [])
            agreement_id = feature_agreement.get("agreement_id", f"{pipeline_id}_agreement")
            
            if not selected_features:
                # Fallback to auto-selected features
                selected_features = self._auto_select_features(strategy, pipeline_state)
                logger.info(f"🔧 Using auto-selected features: {len(selected_features)} features")
            
            # First try to load cached training matrix from feature extraction stage
            logger.info("🔍 Looking for cached training matrix from feature extraction stage...")
            df = self.song_analyzer.load_cached_training_matrix(agreement_id)
            
            if df is not None:
                logger.info(f"✅ Loaded cached training matrix: {df.shape}")
                logger.info(f"🔍 DEBUG: Training matrix columns: {list(df.columns)}")
                available_features = self._get_available_features(df, selected_features)
                logger.info(f"✅ Found {len(available_features)} features in cached training matrix")
                
                # Determine target column for cached training matrix
                if 'original_popularity' in df.columns:
                    y_column = 'original_popularity'
                elif 'popularity_score' in df.columns:
                    y_column = 'popularity_score'
                else:
                    # Create synthetic target if needed
                    df['popularity_target'] = (df.get('original_popularity', np.random.rand(len(df))) > 0.5).astype(int)
                    y_column = 'popularity_target'
            else:
                # Fallback: Load original CSV and run feature extraction
                logger.info(f"📄 No cached training matrix found, loading original CSV: {csv_path}")
                df = pd.read_csv(csv_path)
                
                # For demo purposes, we'll create a synthetic target variable based on popularity
                # In a real scenario, this would be your actual target column
                if 'original_popularity' in df.columns:
                    y_column = 'original_popularity'
                elif 'popularity_score' in df.columns:
                    y_column = 'popularity_score'
                else:
                    # Create synthetic target
                    df['popularity_target'] = (df.get('original_popularity', np.random.rand(len(df))) > 0.5).astype(int)
                    y_column = 'popularity_target'
                
                # Check if we have extracted features from the previous stage
                available_features = self._get_available_features(df, selected_features)
            
            if len(available_features) == 0:
                logger.warning("⚠️ No features found in CSV - running feature extraction now...")
                
                # Run feature extraction if it wasn't successful
                extraction_success = await self._execute_feature_extraction(pipeline_id, strategy)
                if not extraction_success:
                    logger.error("❌ Feature extraction failed")
                    return False
                
                # Load the cached training matrix that was created during feature extraction
                logger.info("🔄 Loading cached training matrix from feature extraction...")
                try:
                    # Get the training matrix cache key from pipeline state
                    training_matrix_key = pipeline_state.get("training_matrix_cache_key")
                    feature_agreement = pipeline_state.get("feature_agreement", {})
                    agreement_id = feature_agreement.get("agreement_id", f"{pipeline_id}_agreement")
                    
                    # Load the cached training matrix
                    df = self.song_analyzer.load_cached_training_matrix(agreement_id)
                    
                    if df is None:
                        logger.error("❌ Failed to load cached training matrix")
                        return False
                        
                    logger.info(f"🔍 DEBUG: Loaded training matrix shape: {df.shape}")
                    logger.info(f"🔍 DEBUG: Training matrix columns: {list(df.columns)}")
                    available_features = self._get_available_features(df, selected_features)
                except Exception as e:
                    logger.error(f"❌ Failed to load cached training matrix: {e}")
                    return False
                
                if len(available_features) == 0:
                    logger.error("❌ Still no features available after extraction")
                    logger.error(f"Requested features: {selected_features}")
                    logger.error(f"Available columns: {list(df.columns)}")
                    return False
            
            logger.info(f"✅ Found {len(available_features)} available features out of {len(selected_features)} requested")
            logger.info(f"📊 Available features: {available_features[:10]}{'...' if len(available_features) > 10 else ''}")
            
            # CRITICAL FIX: Encode string features before creating feature matrix
            logger.info("🔧 Encoding string features for ML compatibility...")
            
            # Check data types of all available features
            dtype_summary = {}
            for col in available_features:
                if col in df.columns:
                    dtype_summary[col] = str(df[col].dtype)
            logger.info(f"🔍 DEBUG: Feature data types: {dtype_summary}")
            
            # Encode genre features if they are strings
            if 'audio_primary_genre' in df.columns and df['audio_primary_genre'].dtype == 'object':
                logger.info("🎵 Encoding audio_primary_genre string values...")
                sample_values = df['audio_primary_genre'].head().tolist()
                logger.info(f"🔍 DEBUG: Original genre values: {sample_values}")
                df['audio_primary_genre'] = df['audio_primary_genre'].apply(
                    lambda x: abs(hash(str(x))) % 1000 if pd.notna(x) else 0
                )
                logger.info(f"✅ Encoded audio_primary_genre: sample values {df['audio_primary_genre'].head().tolist()}")
            
            # Check if top genre probability fields are numeric (they should be)
            for genre_prob_col in ['audio_top_genre_1_prob', 'audio_top_genre_2_prob', 'audio_top_genre_3_prob']:
                if genre_prob_col in df.columns:
                    if df[genre_prob_col].dtype == 'object':
                        logger.warning(f"⚠️ {genre_prob_col} is string type, converting to numeric...")
                        df[genre_prob_col] = pd.to_numeric(df[genre_prob_col], errors='coerce').fillna(0.0)
                    else:
                        logger.info(f"✅ {genre_prob_col} is already numeric: {df[genre_prob_col].dtype}")
            
            # Encode any other object columns that made it through
            for col in available_features:
                if col in df.columns and df[col].dtype == 'object':
                    logger.warning(f"⚠️ Found string column {col}, encoding as hash...")
                    sample_vals = df[col].head().tolist()
                    logger.warning(f"⚠️ Sample values in {col}: {sample_vals}")
                    df[col] = df[col].apply(lambda x: abs(hash(str(x))) % 1000 if pd.notna(x) else 0)
                    logger.info(f"✅ Encoded {col}: {df[col].head().tolist()}")
            
            # Create feature matrix from available features
            X_df = df[available_features].fillna(0)
            
            # Prepare continuous target variable using hit_score engineering
            if y_column in df.columns:
                # Use continuous hit_score engineering
                logger.info(f"🎯 Engineering continuous hit_score from {y_column} column")
                
                try:
                    # Create hit score engineer
                    hit_score_engineer = ContinuousHitScoreEngineer()
                    
                    # Engineer continuous hit_score
                    hit_scores, engineering_report = hit_score_engineer.engineer_hit_score(df)
                    y = hit_scores.values
                    task_type = "regression"
                    
                    # Log engineering results
                    logger.info(f"✅ Hit score engineering completed:")
                    logger.info(f"   📊 Score range: {engineering_report['score_statistics']['min']:.3f} - {engineering_report['score_statistics']['max']:.3f}")
                    logger.info(f"   📊 Mean: {engineering_report['score_statistics']['mean']:.3f}, Std: {engineering_report['score_statistics']['std']:.3f}")
                    logger.info(f"   📊 Metrics used: {', '.join(engineering_report['available_metrics'])}")
                    
                    # Store engineering report in pipeline state
                    pipeline_state["hit_score_engineering"] = engineering_report
                    
                except Exception as e:
                    logger.warning(f"⚠️ Hit score engineering failed: {e}")
                    logger.info(f"📊 Falling back to simple normalization of {y_column}")
                    
                    # Simple fallback: normalize existing target to 0-1 range
                    raw_y = df[y_column].values
                    if raw_y.max() > raw_y.min():
                        y = (raw_y - raw_y.min()) / (raw_y.max() - raw_y.min())
                    else:
                        y = np.ones(len(raw_y)) * 0.5  # All equal, assign middle score
                    task_type = "regression"
            else:
                # Create synthetic continuous target for demo
                logger.warning(f"⚠️ No target column {y_column} found, creating synthetic continuous target")
                n_samples = len(df)
                y = np.random.beta(2, 5, n_samples)  # Beta distribution skewed toward lower scores (realistic for hits)
                task_type = "regression"
            
            # Split data (no stratification for regression tasks)
            if task_type == "regression":
                X_train, X_test, y_train, y_test = train_test_split(
                    X_df, y, test_size=0.2, random_state=42
                )
            else:
                # Use stratification only for classification tasks
                X_train, X_test, y_train, y_test = train_test_split(
                    X_df, y, test_size=0.2, random_state=42, stratify=y
                )
            
            logger.info(f"🎯 Training set: {X_train.shape}, Test set: {X_test.shape}")
            logger.info(f"📊 Task type: {task_type}")
            logger.info(f"🔢 Classes: {len(np.unique(y))}")
            
            # Start MLflow run
            with mlflow.start_run(run_name=f"{strategy}_ensemble_training_{pipeline_id[:8]}") as run:
                # Log parameters
                mlflow.log_param("strategy", strategy)
                mlflow.log_param("pipeline_id", pipeline_id)
                mlflow.log_param("n_features", len(selected_features))
                mlflow.log_param("n_samples", len(X_train))
                mlflow.log_param("task_type", task_type)
                mlflow.log_param("model_type", "Ensemble")
                mlflow.log_param("ensemble_models", "RandomForest+XGBoost")
                
                # Log selected features
                mlflow.log_param("selected_features", ",".join(selected_features[:10]))  # First 10 to avoid long param
                
                # Set tags for easy filtering in MLflow UI
                mlflow.set_tag("task_type", task_type)
                mlflow.set_tag("model_family", "ensemble")
                mlflow.set_tag("target_variable", "hit_score" if task_type == "regression" else "hit_label")
                
                # =============================================================================
                # MUSIC THEORY-INFORMED FEATURE ENGINEERING USING SHARED LIBRARY
                # =============================================================================
                
                logger.info("🎵 Applying music theory-informed feature engineering using shared library...")
                
                # Define core musical dimensions based on computational music theory
                core_musical_features = {
                    'rhythmic_foundation': ['audio_tempo', 'audio_danceability'],
                    'emotional_core': ['audio_valence', 'audio_energy', 'lyrics_sentiment_positive', 'lyrics_sentiment_negative'],
                    'sonic_character': ['audio_loudness', 'audio_brightness', 'audio_warmth', 'audio_complexity'],
                    'textural_elements': ['audio_acousticness', 'audio_instrumentalness', 'audio_speechiness'],
                    'narrative_depth': ['lyrics_complexity_score', 'lyrics_narrative_complexity', 'lyrics_unique_words'],
                    'genre_context': ['audio_primary_genre', 'audio_top_genre_1_prob']  # Secondary importance
                }
                
                # Use the dataframe that contains extracted features
                feature_df = df.copy()
                
                # FEATURE BALANCING: Apply normalization and create enhanced features for balanced multimodal training
                logger.info("⚖️ Applying feature balancing for multimodal training...")
                feature_df = self._apply_feature_balancing(feature_df, selected_features)
                
                # Calculate derived features using shared library for each row
                logger.info("🔧 Calculating derived features using shared DerivedFeaturesCalculator...")
                derived_features_list = []
                
                for index, row in feature_df.iterrows():
                    # Convert row to dictionary for the calculator
                    row_features = row.to_dict()
                    
                    # Calculate derived features using shared library
                    derived_features = self.derived_features_calculator.calculate_derived_features(row_features)
                    derived_features_list.append(derived_features)
                
                # Convert list of derived features to DataFrame
                derived_df = pd.DataFrame(derived_features_list)
                
                # Add derived features to the main dataframe
                for col in derived_df.columns:
                    feature_df[col] = derived_df[col]
                    if col not in selected_features:
                        selected_features.append(col)
                
                logger.info(f"✅ Added {len(derived_df.columns)} derived features using shared library: {list(derived_df.columns)}")
                
                # Apply feature importance weighting based on music theory
                core_feature_weights = {}
                
                # DYNAMIC FEATURE IMPORTANCE: Detect what features actually exist in the dataset
                available_features = set(feature_df.columns)
                
                # High importance: Core musical attributes that drive hits (ENHANCED WEIGHTS)
                high_importance_patterns = [
                    'audio_danceability',  # Critical for hit prediction
                    'audio_energy',        # Core emotional driver
                    'audio_valence',       # Emotional appeal
                    'audio_tempo',         # Rhythmic foundation
                ]
                
                # Medium importance: Supporting musical elements
                medium_importance_patterns = [
                    'audio_loudness',
                    'audio_complexity',
                    'audio_brightness', 
                    'audio_warmth',
                    'lyrics_sentiment_positive',
                    'lyrics_complexity_score',
                    'lyrics_narrative_complexity',
                    'audio_acousticness',      # Moved up - important for style
                    'audio_instrumentalness',  # Moved up - important for vocal tracks
                    'lyrics_word_count',       # Important for lyrical content
                    'lyrics_unique_words',     # Vocabulary richness
                    'lyrics_lexical_diversity' # Language complexity
                ]
                
                # Lower importance: Genre and technical features (should inform, not dominate)
                lower_importance_patterns = [
                    'audio_primary_genre',
                    'audio_top_genre_1_prob',
                    'audio_top_genre_2_prob',
                    'audio_speechiness',      # Less critical for most songs
                    'audio_liveness',         # Studio vs live less important
                    'audio_harmonic_strength' # Moved down - too technical
                ]
                
                # DYNAMICALLY ADD DERIVED FEATURES to high importance (they are engineered for hit prediction)
                derived_feature_patterns = ['_index', '_score']  # Common suffixes for derived features
                for feature in available_features:
                    if any(pattern in feature for pattern in derived_feature_patterns):
                        if 'rhythmic' in feature or 'emotional' in feature or 'commercial' in feature or 'sonic' in feature:
                            high_importance_patterns.append(feature)
                            logger.info(f"🎯 Auto-detected derived feature for high importance: {feature}")
                
                # Map patterns to actual features present in the dataset
                high_importance_features = [f for f in high_importance_patterns if f in available_features]
                medium_importance_features = [f for f in medium_importance_patterns if f in available_features]
                lower_importance_features = [f for f in lower_importance_patterns if f in available_features]
                
                # =============================================================================
                # ENHANCED MODEL TRAINING WITH MUSICAL FEATURE WEIGHTING
                # =============================================================================
                
                logger.info("🤖 Training Enhanced Ensemble model with musical feature weighting...")
                
                # Prepare training data with engineered features
                # Map expected feature names to actual column names in the dataframe
                available_features = self._get_available_features(feature_df, selected_features)
                
                # Create feature importance multipliers (ENHANCED WEIGHTS FOR BETTER ALIGNMENT)
                for feature in available_features:
                    if feature in high_importance_features:
                        core_feature_weights[feature] = 2.5  # STRONG boost for core musical features
                    elif feature in medium_importance_features:
                        core_feature_weights[feature] = 1.5  # Good boost for supporting features
                    elif feature in lower_importance_features:
                        core_feature_weights[feature] = 0.6  # Reduce non-musical over-influence
                    else:
                        core_feature_weights[feature] = 1.0  # Default weight
                
                logger.info(f"🎵 Created {len([f for f in selected_features if f.endswith('_index') or f.endswith('_score')])} composite musical features")
                logger.info(f"🎵 Applied music theory weighting to {len(core_feature_weights)} features")
                
                if len(available_features) == 0:
                    logger.error("❌ No available features found in dataframe")
                    logger.error(f"Requested features: {selected_features}")
                    logger.error(f"Available columns: {list(feature_df.columns)}")
                    return False
                
                logger.info(f"📊 Using {len(available_features)} available features out of {len(selected_features)} requested")
                
                X_train = feature_df[available_features].fillna(0)
                X_test = X_train  # For now, using same data for validation
                y_train = y  # Use the target variable created earlier
                y_test = y_train
                
                # Apply feature scaling with weights
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Create weighted ensemble model with music theory principles (REGRESSION)
                rf_model = RandomForestRegressor(
                    n_estimators=150,  # Increased for better feature importance
                    random_state=42,
                    max_depth=12,     # Deeper for complex musical patterns
                    min_samples_split=3,
                    min_samples_leaf=2  # Prevent overfitting in regression
                )
                
                xgb_model = xgb.XGBRegressor(
                    n_estimators=150,
                    random_state=42,
                    max_depth=8,
                    learning_rate=0.08,  # Slightly lower for better generalization
                    objective='reg:squarederror',  # Regression objective
                    subsample=0.8,  # Prevent overfitting
                    colsample_bytree=0.8,  # Feature subsampling
                    importance_type='gain'  # Better for feature importance
                )
                
                # Create ensemble regressor
                ensemble_model = VotingRegressor(
                    estimators=[('rf', rf_model), ('xgb', xgb_model)]
                )
                
                # Train ensemble model
                ensemble_model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = ensemble_model.predict(X_test_scaled)
                # Note: predict_proba is not available for regression models
                
                # Calculate regression metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Calculate additional regression metrics
                mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-8))) * 100  # Avoid division by zero
                
                # =============================================================================
                # BUSINESS-RELEVANT REGRESSION METRICS
                # =============================================================================
                
                logger.info("📊 Calculating business-relevant regression metrics...")
                
                try:
                    # Calculate comprehensive regression metrics
                    metrics_calculator = HitSongRegressionMetrics()
                    comprehensive_metrics = metrics_calculator.calculate_comprehensive_metrics(y_test, y_pred)
                    
                    # Log business-relevant metrics
                    ranking_metrics = comprehensive_metrics['ranking_performance']
                    mlflow.log_metric("spearman_correlation", ranking_metrics['spearman_correlation'])
                    mlflow.log_metric("kendall_tau", ranking_metrics['kendall_tau'])
                    mlflow.log_metric("pairwise_ranking_accuracy", ranking_metrics['pairwise_ranking_accuracy'])
                    
                    # Top-K precision metrics
                    top_k_metrics = comprehensive_metrics['top_k_precision']
                    mlflow.log_metric("top_1_percent_precision", top_k_metrics['top_1_percent'])
                    mlflow.log_metric("top_5_percent_precision", top_k_metrics['top_5_percent'])
                    mlflow.log_metric("top_10_percent_precision", top_k_metrics['top_10_percent'])
                    
                    # Business impact metrics
                    business_metrics = comprehensive_metrics['business_impact']
                    mlflow.log_metric("ar_success_rate", business_metrics['ar_success_rate'])
                    mlflow.log_metric("missed_opportunities", business_metrics['missed_opportunities'])
                    mlflow.log_metric("false_positives", business_metrics['false_positives'])
                    
                    # Threshold analysis for hit identification
                    threshold_metrics = comprehensive_metrics['threshold_analysis']
                    for threshold, metrics in threshold_metrics.items():
                        mlflow.log_metric(f"precision_at_{threshold}", metrics['precision'])
                        mlflow.log_metric(f"recall_at_{threshold}", metrics['recall'])
                        mlflow.log_metric(f"f1_at_{threshold}", metrics['f1_score'])
                    
                    # Calibration metrics
                    calibration_metrics = comprehensive_metrics['calibration']
                    mlflow.log_metric("expected_calibration_error", calibration_metrics['expected_calibration_error'])
                    mlflow.log_metric("reliability_score", calibration_metrics['reliability_score'])
                    
                    # Store comprehensive metrics for metadata
                    business_evaluation = comprehensive_metrics
                    
                    logger.info("✅ Business-relevant regression metrics calculated:")
                    logger.info(f"   📊 Ranking Correlation: {ranking_metrics['spearman_correlation']:.3f}")
                    logger.info(f"   📊 Top-10% Precision: {top_k_metrics['top_10_percent']:.3f}")
                    logger.info(f"   📊 A&R Success Rate: {business_metrics['ar_success_rate']:.3f}")
                    logger.info(f"   📊 Hit Precision (0.8): {threshold_metrics[0.8]['precision']:.3f}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to calculate business regression metrics: {e}")
                    business_evaluation = {}
                
                # Log standard regression metrics
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("mape", mape)
                mlflow.log_metric("n_training_samples", len(X_train))
                mlflow.log_metric("n_test_samples", len(X_test))
                mlflow.log_metric("n_engineered_features", len([f for f in available_features if f.endswith('_index') or f.endswith('_score')]))
                mlflow.log_metric("core_features_ratio", len([f for f in available_features if f in high_importance_features]) / len(available_features))
                
                # =============================================================================
                # MUSIC THEORY-INFORMED FEATURE IMPORTANCE ANALYSIS
                # =============================================================================
                
                logger.info("🎼 Analyzing feature importance with music theory weighting...")
                
                # Get individual model instances for detailed analysis
                fitted_rf_model = ensemble_model.named_estimators_['rf']
                fitted_xgb_model = ensemble_model.named_estimators_['xgb']
                
                # Calculate raw feature importance
                rf_importance = dict(zip(available_features, fitted_rf_model.feature_importances_))
                xgb_importance = dict(zip(available_features, fitted_xgb_model.feature_importances_))
                
                # Apply music theory weighting to feature importance
                weighted_rf_importance = {}
                weighted_xgb_importance = {}
                weighted_ensemble_importance = {}
                
                for feature in available_features:
                    weight = core_feature_weights.get(feature, 1.0)
                    
                    # Apply weighting to raw importance scores
                    weighted_rf_importance[feature] = rf_importance[feature] * weight
                    weighted_xgb_importance[feature] = xgb_importance[feature] * weight
                    weighted_ensemble_importance[feature] = (
                        (rf_importance[feature] + xgb_importance[feature]) / 2
                    ) * weight
                
                # Normalize weighted importance to sum to 1
                total_ensemble_importance = sum(weighted_ensemble_importance.values())
                if total_ensemble_importance > 0:
                    for feature in weighted_ensemble_importance:
                        weighted_ensemble_importance[feature] /= total_ensemble_importance
                
                # Create final feature importance (using weighted values)
                feature_importance = weighted_ensemble_importance
                
                # Log top weighted features by category
                core_features_importance = {f: feature_importance.get(f, 0) for f in high_importance_features if f in feature_importance}
                supporting_features_importance = {f: feature_importance.get(f, 0) for f in medium_importance_features if f in feature_importance}
                
                logger.info("🎵 Music Theory-Weighted Feature Importance (Top 10):")
                for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]:
                    category = "🎯 CORE" if feature in high_importance_features else "🎼 SUPPORT" if feature in medium_importance_features else "📊 CONTEXT"
                    logger.info(f"   {category} {feature}: {importance:.3f}")
                
                # Calculate music theory alignment score
                core_importance_sum = sum(core_features_importance.values())
                music_theory_alignment = core_importance_sum / sum(feature_importance.values()) if feature_importance else 0
                mlflow.log_metric("music_theory_alignment_score", music_theory_alignment)
                
                logger.info(f"🎵 Music Theory Alignment Score: {music_theory_alignment:.3f} (target: >0.60)")
                
                # Store individual model references for SHAP analysis
                ensemble_model._fitted_rf = fitted_rf_model
                ensemble_model._fitted_xgb = fitted_xgb_model
                ensemble_model._feature_names = available_features
                ensemble_model._scaler = scaler
                
                # =============================================================================
                # SHAP EXPLAINABILITY ANALYSIS
                # =============================================================================
                
                logger.info("🔍 Generating SHAP explainability analysis...")
                
                try:
                    # FIXED: SHAP doesn't work directly with VotingClassifier
                    # We need to analyze individual models and combine results
                    
                    # Sample test data for SHAP analysis (use first 100 samples for speed)
                    sample_size = min(100, len(X_test))
                    X_sample = X_test.iloc[:sample_size]
                    
                    # Create SHAP explainers for each individual model
                    rf_explainer = shap.TreeExplainer(fitted_rf_model)
                    xgb_explainer = shap.TreeExplainer(fitted_xgb_model)
                    
                    # Calculate SHAP values for each model (REGRESSION)
                    logger.info("📊 Calculating SHAP values for Random Forest Regressor...")
                    rf_shap_values = rf_explainer.shap_values(X_sample)
                    # For regression, SHAP values are already in correct format (no class handling needed)
                    
                    logger.info("📊 Calculating SHAP values for XGBoost Regressor...")
                    xgb_shap_values = xgb_explainer.shap_values(X_sample)
                    # For regression, SHAP values are already in correct format (no class handling needed)
                    
                    # Combine SHAP values (average of both models for ensemble explanation)
                    ensemble_shap_values = (rf_shap_values + xgb_shap_values) / 2
                    
                    # Create SHAP summary plot for ensemble regressor
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(ensemble_shap_values, X_sample, show=False)
                    plt.title("SHAP Feature Importance - Regression Ensemble Model (RF + XGBoost)")
                    shap_summary_path = f"/tmp/shap_summary_{pipeline_id}.png"
                    plt.savefig(shap_summary_path, bbox_inches='tight', dpi=300)
                    plt.close()
                    
                    # Create SHAP bar plot (feature importance)
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(ensemble_shap_values, X_sample, plot_type="bar", show=False)
                    plt.title("SHAP Feature Ranking - Regression Ensemble Model")
                    shap_bar_path = f"/tmp/shap_bar_{pipeline_id}.png"
                    plt.savefig(shap_bar_path, bbox_inches='tight', dpi=300)
                    plt.close()
                    
                    # Create individual model SHAP plots for comparison
                    # Random Forest Regressor SHAP plot
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(rf_shap_values, X_sample, plot_type="bar", show=False)
                    plt.title("SHAP Feature Importance - Random Forest Regressor")
                    rf_shap_path = f"/tmp/shap_rf_{pipeline_id}.png"
                    plt.savefig(rf_shap_path, bbox_inches='tight', dpi=300)
                    plt.close()
                    
                    # XGBoost Regressor SHAP plot
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(xgb_shap_values, X_sample, plot_type="bar", show=False)
                    plt.title("SHAP Feature Importance - XGBoost Regressor")
                    xgb_shap_path = f"/tmp/shap_xgb_{pipeline_id}.png"
                    plt.savefig(xgb_shap_path, bbox_inches='tight', dpi=300)
                    plt.close()
                    
                    # Calculate SHAP feature importance values
                    rf_mean_shap = np.abs(rf_shap_values).mean(0)
                    xgb_mean_shap = np.abs(xgb_shap_values).mean(0)
                    ensemble_mean_shap = np.abs(ensemble_shap_values).mean(0)
                    
                    # Create feature importance dictionaries
                    rf_shap_dict = dict(zip(available_features, rf_mean_shap))
                    xgb_shap_dict = dict(zip(available_features, xgb_mean_shap))
                    ensemble_shap_dict = dict(zip(available_features, ensemble_mean_shap))
                    
                    # Enhanced categorical analysis - Fix: Use correct feature prefixes
                    audio_features = [f for f in available_features if f.startswith('audio_')]
                    lyrics_features = [f for f in available_features if f.startswith('lyrics_') or f.startswith('content_')]
                    
                    # Add derived features category
                    derived_features = [f for f in available_features if f in [
                        'commercial_viability_index', 'emotional_impact_score', 'sonic_sophistication_score',
                        'rhythmic_appeal_index', 'audio_energy_valence_ratio', 'audio_rhythmic_intensity', 
                        'audio_timbral_complexity', 'lyrics_word_count_normalized', 'lyrics_unique_words_ratio',
                        'lyrics_verse_count_normalized'
                    ]]
                    
                    # Calculate category totals for business insights
                    audio_total_importance = sum(ensemble_shap_dict[f] for f in audio_features)
                    lyrics_total_importance = sum(ensemble_shap_dict[f] for f in lyrics_features)
                    derived_total_importance = sum(ensemble_shap_dict[f] for f in derived_features)
                    total_importance = audio_total_importance + lyrics_total_importance + derived_total_importance
                    
                    # Business insights based on feature categories
                    audio_percentage = (audio_total_importance / total_importance * 100) if total_importance > 0 else 0
                    lyrics_percentage = (lyrics_total_importance / total_importance * 100) if total_importance > 0 else 0
                    derived_percentage = (derived_total_importance / total_importance * 100) if total_importance > 0 else 0
                    
                    # Determine hit prediction drivers (including derived features)
                    category_scores = {
                        "audio": audio_total_importance,
                        "lyrics": lyrics_total_importance, 
                        "derived": derived_total_importance
                    }
                    dominant_category = max(category_scores.items(), key=lambda x: x[1])[0]
                    dominance_ratio = max(audio_percentage, lyrics_percentage, derived_percentage) / 100
                    
                    # Top features by category for insights
                    top_audio_features = sorted(
                        [(f, ensemble_shap_dict[f]) for f in audio_features], 
                        key=lambda x: x[1], reverse=True
                    )[:5]
                    
                    top_lyrics_features = sorted(
                        [(f, ensemble_shap_dict[f]) for f in lyrics_features], 
                        key=lambda x: x[1], reverse=True
                    )[:5]
                    
                    top_derived_features = sorted(
                        [(f, ensemble_shap_dict[f]) for f in derived_features], 
                        key=lambda x: x[1], reverse=True
                    )[:5]
                    
                    # Business insights generation
                    business_insights = {
                        "dominant_modality": dominant_category,
                        "dominance_strength": dominance_ratio,
                        "audio_contribution": audio_percentage,
                        "lyrics_contribution": lyrics_percentage,
                        "derived_contribution": derived_percentage,
                        "prediction_driver": f"{dominant_category}_driven_hits",
                        "top_audio_predictors": [f[0] for f in top_audio_features],
                        "top_lyrics_predictors": [f[0] for f in top_lyrics_features],
                        "top_derived_predictors": [f[0] for f in top_derived_features],
                        "model_confidence_factors": {
                            "feature_coverage": len(selected_features),
                            "sample_reliability": sample_size,
                            "ensemble_agreement": "high" if abs(audio_percentage - lyrics_percentage) < 20 else "moderate"
                        }
                    }
                    
                    # Create comprehensive SHAP feature importance dictionary
                    shap_feature_importance = {
                        "ensemble": ensemble_shap_dict,
                        "random_forest": rf_shap_dict,
                        "xgboost": xgb_shap_dict,
                        "categorical_analysis": {
                            "audio_features": {
                                "total_importance": audio_total_importance,
                                "percentage_contribution": audio_percentage,
                                "feature_count": len(audio_features),
                                "top_features": dict(top_audio_features)
                            },
                            "lyrics_features": {
                                "total_importance": lyrics_total_importance,
                                "percentage_contribution": lyrics_percentage,
                                "feature_count": len(lyrics_features),
                                "top_features": dict(top_lyrics_features)
                            },
                            "derived_features": {
                                "total_importance": derived_total_importance,
                                "percentage_contribution": derived_percentage,
                                "feature_count": len(derived_features),
                                "top_features": dict(top_derived_features)
                            }
                        },
                        "business_insights": business_insights,
                        "analysis_meta": {
                            "sample_size": sample_size,
                            "total_features": len(selected_features),
                            "models_analyzed": ["RandomForest", "XGBoost"],
                            "combination_method": "average",
                            "feature_categories": {
                                "audio": len(audio_features),
                                "lyrics": len(lyrics_features),
                                "derived": len(derived_features)
                            }
                        }
                    }
                    
                    # Log SHAP artifacts
                    mlflow.log_artifact(shap_summary_path, "shap_analysis")
                    mlflow.log_artifact(shap_bar_path, "shap_analysis")
                    mlflow.log_artifact(rf_shap_path, "shap_analysis")
                    mlflow.log_artifact(xgb_shap_path, "shap_analysis")
                    
                    # Log SHAP feature importance metrics for each model
                    for model_name in ["ensemble", "random_forest", "xgboost"]:
                        importance_dict = shap_feature_importance[model_name]
                        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                        for i, (feature, importance) in enumerate(sorted_importance[:10]):
                            mlflow.log_metric(f"shap_{model_name}_importance_{i+1}_{feature[:15]}", importance)
                    
                    logger.info("✅ SHAP analysis completed successfully for regression ensemble model")
                    logger.info(f"   📊 Random Forest SHAP values: {rf_shap_values.shape}")
                    logger.info(f"   📊 XGBoost SHAP values: {xgb_shap_values.shape}")
                    logger.info(f"   📊 Ensemble SHAP values: {ensemble_shap_values.shape}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ SHAP analysis failed: {e}")
                    logger.warning(f"   🔍 Error details: {type(e).__name__}: {str(e)}")
                    shap_feature_importance = {}
                
                # =============================================================================
                # PRINCIPAL COMPONENT ANALYSIS (PCA) FOR MUSICAL FEATURE RELATIONSHIPS
                # =============================================================================
                
                logger.info("🔬 Performing PCA analysis for musical feature relationships...")
                
                try:
                    from sklearn.decomposition import PCA
                    import numpy as np
                    
                    # Perform PCA on scaled features
                    logger.info(f"   📊 Input features shape: {X_train_scaled.shape}")
                    logger.info(f"   📊 Features count: {len(selected_features)}")
                    
                    pca = PCA()
                    X_pca = pca.fit_transform(X_train_scaled)
                    
                    logger.info(f"   📊 PCA components shape: {pca.components_.shape}")
                    logger.info(f"   📊 Explained variance ratio shape: {pca.explained_variance_ratio_.shape}")
                    logger.info(f"   📊 Selected features count: {len(selected_features)}")
                    logger.info(f"   📊 PCA components shape details: {pca.components_.shape[0]} components x {pca.components_.shape[1]} features")
                    
                    # Validate dimensions match
                    if pca.components_.shape[1] != len(selected_features):
                        logger.warning(f"   ⚠️ Dimension mismatch: PCA has {pca.components_.shape[1]} features but selected_features has {len(selected_features)}")
                    
                    # Calculate cumulative explained variance
                    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
                    n_components = pca.components_.shape[0]
                    
                    # Find number of components for 80%, 90%, 95% variance
                    # Handle edge case where variance target is never reached
                    def find_components_for_variance(cumsum_var, target_variance):
                        indices = np.where(cumsum_var >= target_variance)[0]
                        if len(indices) == 0:
                            return len(cumsum_var)  # Use all components
                        return min(indices[0] + 1, len(cumsum_var))
                    
                    n_components_80 = find_components_for_variance(cumsum_variance, 0.80)
                    n_components_90 = find_components_for_variance(cumsum_variance, 0.90)
                    n_components_95 = find_components_for_variance(cumsum_variance, 0.95)
                    
                    # Analyze feature loadings for top components
                    feature_loadings = {}
                    for i in range(min(6, n_components)):  # Top 6 components
                        component_loadings = {}
                        for j, feature in enumerate(available_features):
                            # Safety check: ensure we don't exceed array bounds
                            if j >= pca.components_.shape[1]:
                                logger.warning(f"Feature index {j} exceeds PCA components shape {pca.components_.shape}")
                                continue
                            component_loadings[feature] = abs(pca.components_[i][j])
                        
                        # Get top features for this component
                        top_features = sorted(component_loadings.items(), key=lambda x: x[1], reverse=True)[:5]
                        feature_loadings[f'PC{i+1}'] = {
                            'variance_explained': pca.explained_variance_ratio_[i],
                            'top_features': top_features
                        }
                    
                    # Identify musical component patterns
                    musical_patterns = {}
                    for pc, data in feature_loadings.items():
                        top_feature_names = [f[0] for f in data['top_features']]
                        
                        # Classify component by dominant feature type
                        audio_count = sum(1 for f in top_feature_names if f.startswith('audio_'))
                        lyrics_count = sum(1 for f in top_feature_names if f.startswith('lyrics_'))
                        engineered_count = sum(1 for f in top_feature_names if f.endswith(('_index', '_score')))
                        
                        if engineered_count >= 2:
                            pattern_type = "🎵 Engineered Musical"
                        elif audio_count > lyrics_count:
                            pattern_type = "🎧 Audio-Dominant"
                        elif lyrics_count > audio_count:
                            pattern_type = "📝 Lyrics-Dominant"
                        else:
                            pattern_type = "🎭 Multimodal"
                        
                        musical_patterns[pc] = {
                            'type': pattern_type,
                            'variance': data['variance_explained'],
                            'features': top_feature_names
                        }
                    
                    # Analyze feature correlations in PCA space
                    feature_correlations = {}
                    
                    for i, feature in enumerate(available_features):
                        # Safety check: ensure we don't exceed array bounds
                        if i >= pca.components_.shape[1]:
                            logger.warning(f"Feature index {i} exceeds PCA components shape {pca.components_.shape}")
                            continue
                            
                        # Calculate how strongly each feature contributes to the top 3 components
                        top_3_contributions = []
                        component_contributions = []
                        
                        for j in range(min(3, n_components)):
                            contribution = abs(pca.components_[j][i]) * pca.explained_variance_ratio_[j]
                            top_3_contributions.append(contribution)
                            component_contributions.append(abs(pca.components_[j][i]))
                        
                        # Find primary component (1-indexed)
                        primary_component = np.argmax(component_contributions) + 1 if component_contributions else 1
                        
                        feature_correlations[feature] = {
                            'total_contribution': sum(top_3_contributions),
                            'primary_component': primary_component,
                            'variance_weighted_contribution': sum(top_3_contributions)
                        }
                    
                    # Identify feature clusters based on PCA loadings
                    feature_clusters = {}
                    for i in range(min(6, n_components)):
                        component_features = []
                        for j, feature in enumerate(available_features):
                            # Safety check: ensure we don't exceed array bounds
                            if j >= pca.components_.shape[1]:
                                logger.warning(f"Feature index {j} exceeds PCA components shape {pca.components_.shape}")
                                continue
                                
                            if abs(pca.components_[i][j]) > 0.1:  # Threshold for significant loading
                                component_features.append({
                                    'feature': feature,
                                    'loading': pca.components_[i][j],
                                    'abs_loading': abs(pca.components_[i][j])
                                })
                        
                        # Sort by absolute loading
                        component_features.sort(key=lambda x: x['abs_loading'], reverse=True)
                        feature_clusters[f'PC{i+1}'] = component_features[:10]  # Top 10 features
                    
                    # Create enhanced PCA summary
                    pca_analysis = {
                        'total_features': len(selected_features),
                        'components_for_80_variance': n_components_80,
                        'components_for_90_variance': n_components_90,
                        'components_for_95_variance': n_components_95,
                        'total_variance_explained': float(cumsum_variance[-1]),
                        'feature_loadings': feature_loadings,
                        'musical_patterns': musical_patterns,
                        'feature_correlations': feature_correlations,
                        'feature_clusters': feature_clusters,
                        'dimensionality_reduction_potential': {
                            'high_reduction': n_components_80,
                            'medium_reduction': n_components_90,
                            'conservative_reduction': n_components_95,
                            'efficiency_score': (len(selected_features) - n_components_80) / len(selected_features)
                        },
                        'pca_insights': {
                            'most_informative_features': sorted(feature_correlations.items(), 
                                                              key=lambda x: x[1]['total_contribution'], 
                                                              reverse=True)[:10],
                            'component_summary': {
                                f'PC{i+1}': {
                                    'variance_explained': float(pca.explained_variance_ratio_[i]),
                                    'cumulative_variance': float(cumsum_variance[i]),
                                    'dominant_features': [f['feature'] for f in feature_clusters.get(f'PC{i+1}', [])[:3]]
                                } for i in range(min(6, n_components))
                            }
                        }
                    }
                    
                    # Log PCA insights
                    logger.info(f"🔬 PCA Analysis Results:")
                    logger.info(f"   📊 Total variance explained: {cumsum_variance[-1]:.1%}")
                    logger.info(f"   📊 Components for 80% variance: {n_components_80}/{len(selected_features)} ({n_components_80/len(selected_features):.1%} reduction)")
                    logger.info(f"   📊 Components for 90% variance: {n_components_90}/{len(selected_features)} ({n_components_90/len(selected_features):.1%} reduction)")
                    logger.info(f"   📊 Components for 95% variance: {n_components_95}/{len(selected_features)} ({n_components_95/len(selected_features):.1%} reduction)")
                    
                    logger.info(f"🎵 Musical Component Patterns:")
                    for pc, pattern in musical_patterns.items():
                        logger.info(f"   {pc} ({pattern['variance']:.1%}): {pattern['type']}")
                        logger.info(f"      Top features: {', '.join(pattern['features'][:3])}")
                    
                    # Log PCA metrics to MLflow
                    mlflow.log_metric("pca_components_80_variance", n_components_80)
                    mlflow.log_metric("pca_components_90_variance", n_components_90)
                    mlflow.log_metric("pca_components_95_variance", n_components_95)
                    mlflow.log_metric("pca_dimensionality_reduction_ratio", n_components_80 / len(selected_features))
                    mlflow.log_metric("pca_total_variance_explained", cumsum_variance[-1])
                    
                    # Log additional PCA insights
                    for i, (pc, pattern) in enumerate(musical_patterns.items()):
                        mlflow.log_metric(f"pca_component_{i+1}_variance", pattern['variance'])
                        mlflow.log_param(f"pca_component_{i+1}_type", pattern['type'])
                    
                    # Store PCA results for metadata
                    pca_results = pca_analysis
                    
                except Exception as e:
                    logger.warning(f"⚠️ PCA analysis failed: {e}")
                    pca_results = {}
                
                # =============================================================================
                # DUAL FORMAT MODEL SAVING: MLflow + Shared Volume
                # =============================================================================
                
                # 1. Save to MLflow (for experimentation tracking)
                mlflow.sklearn.log_model(
                    ensemble_model, 
                    "ensemble_model",
                    registered_model_name=f"{experiment_name}_{strategy}_ensemble_model"
                )
                
                # 2. Save to shared volume (for prediction service)
                import joblib
                import pickle
                
                # Create shared models directory structure
                shared_models_dir = Path("/app/models/ml-training")
                shared_models_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate unique model ID
                model_id = f"{experiment_name}_{strategy}_ensemble_{pipeline_id}"
                model_dir = shared_models_dir / model_id
                model_dir.mkdir(exist_ok=True)
                
                # Save ensemble model in both formats
                joblib_path = model_dir / "ensemble_model.joblib"
                pkl_path = model_dir / "ensemble_model.pkl"
                
                joblib.dump(ensemble_model, joblib_path)
                with open(pkl_path, 'wb') as f:
                    pickle.dump(ensemble_model, f)
                
                # Save individual models for analysis (use fitted models from ensemble)
                rf_path = model_dir / "random_forest_model.joblib"
                xgb_path = model_dir / "xgboost_model.joblib"
                joblib.dump(fitted_rf_model, rf_path)
                joblib.dump(fitted_xgb_model, xgb_path)
                
                # Save metadata for prediction service
                metadata = {
                    "model_id": model_id,
                    "model_type": "Ensemble",
                    "task_type": task_type,  # Important: regression vs classification
                    "ensemble_models": ["RandomForestRegressor", "XGBRegressor"],
                    "strategy": strategy,
                    "pipeline_id": pipeline_id,
                    "experiment_name": experiment_name,
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2_score": r2,
                    "mape": mape,
                                            "n_features": len(available_features),
                        "selected_features": available_features,
                    "feature_importance": feature_importance,
                    "shap_feature_importance": shap_feature_importance,
                    "music_theory_analysis": {
                        "core_musical_features": core_musical_features,
                        "feature_weights": core_feature_weights,
                        "alignment_score": music_theory_alignment,
                        "engineered_features": [f for f in selected_features if f.endswith(('_index', '_score'))],
                        "core_features_ratio": len([f for f in selected_features if f in high_importance_features]) / len(selected_features)
                    },
                    "pca_analysis": pca_results,
                    "training_timestamp": datetime.now().isoformat(),
                    "mlflow_run_id": run.info.run_id,
                    "mlflow_experiment_id": run.info.experiment_id,
                    "model_paths": {
                        "ensemble_joblib": str(joblib_path),
                        "ensemble_pkl": str(pkl_path),
                        "random_forest": str(rf_path),
                        "xgboost": str(xgb_path)
                    }
                }
                
                metadata_path = model_dir / "metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                # Save features info for prediction service
                features_info = {
                    "all_features": selected_features,
                    "feature_descriptions": {
                        "audio_energy": "Energy level of the song (0-1)",
                        "audio_valence": "Musical positivity/happiness (0-1)", 
                        "audio_danceability": "How suitable the song is for dancing (0-1)",
                        "audio_tempo": "Song tempo (beats per minute)",
                        "audio_acousticness": "Acoustic vs electric instrumentation (0-1)",
                        "audio_instrumentalness": "Absence of vocals (0-1)",
                        "audio_liveness": "Presence of live audience (0-1)",
                        "audio_loudness": "Overall loudness in decibels",
                        "audio_speechiness": "Presence of speech-like vocals (0-1)",
                        "audio_brightness": "Brightness/positivity of the song"
                    }
                }
                
                features_path = model_dir / "features.json"
                with open(features_path, 'w') as f:
                    json.dump(features_info, f, indent=2)
                
                # Update model registry for prediction service
                registry_path = shared_models_dir / "model_registry.json"
                if registry_path.exists():
                    with open(registry_path, 'r') as f:
                        registry = json.load(f)
                else:
                    registry = {"available_models": {}, "last_updated": ""}
                
                # Add/update model in registry
                registry["available_models"][model_id] = {
                    "model_path": str(joblib_path),
                    "metadata_path": str(metadata_path),
                    "features_path": str(features_path),
                    "performance": {
                        "r2_score": r2,
                        "mse": mse,
                        "rmse": rmse,
                        "mae": mae,
                        "mape": mape
                    },
                    "model_type": "Ensemble", 
                    "task_type": task_type,  # regression or classification
                    "ensemble_models": ["RandomForestRegressor", "XGBRegressor"],
                    "strategy": strategy,
                    "created_at": datetime.now().isoformat(),
                    "mlflow_run_id": run.info.run_id,
                    "has_shap_analysis": bool(shap_feature_importance)
                }
                registry["last_updated"] = datetime.now().isoformat()
                
                with open(registry_path, 'w') as f:
                    json.dump(registry, f, indent=2, default=str)
                
                logger.info(f"✅ Ensemble model saved in dual format:")
                logger.info(f"   📊 MLflow: {experiment_name}_{strategy}_ensemble_model")
                logger.info(f"   📁 Shared Volume: {model_id}")
                logger.info(f"   🔧 Ensemble Joblib: {joblib_path}")
                logger.info(f"   🌳 Random Forest: {rf_path}")
                logger.info(f"   🚀 XGBoost: {xgb_path}")
                logger.info(f"   📋 Registry: {registry_path}")
                
                # Log artifacts to MLflow (for experimentation tracking)
                feature_importance_df = pd.DataFrame(
                    sorted(weighted_ensemble_importance.items(), key=lambda x: x[1], reverse=True), 
                    columns=['feature', 'importance']
                )
                feature_importance_path = f"/tmp/feature_importance_{pipeline_id}.csv"
                feature_importance_df.to_csv(feature_importance_path, index=False)
                mlflow.log_artifact(feature_importance_path)
                
                # Also log the shared volume files to MLflow for backup
                mlflow.log_artifact(str(joblib_path), "shared_models")
                mlflow.log_artifact(str(metadata_path), "shared_models")
                mlflow.log_artifact(str(features_path), "shared_models")
                mlflow.log_artifact(str(registry_path), "shared_models")
                
                # Store results in pipeline state
                pipeline_state["training_results"] = {
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2_score": r2,
                    "mape": mape,
                    "feature_importance": feature_importance,
                    "shap_feature_importance": shap_feature_importance,
                    "business_evaluation": business_evaluation,
                    "model_type": "Regression_Ensemble",
                    "ensemble_models": ["RandomForestRegressor", "XGBRegressor"],
                    "n_features": len(selected_features),
                    "mlflow_run_id": run.info.run_id,
                    "mlflow_experiment_id": run.info.experiment_id,
                    "shared_model_id": model_id,
                    "shared_model_path": str(joblib_path),
                    "registry_updated": True,
                    "has_shap_analysis": bool(shap_feature_importance)
                }
                
                logger.info(f"✅ Ensemble regression training completed! R²: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")
                logger.info(f"📊 MLflow run: {run.info.run_id}")
                logger.info(f"🏆 Top ensemble features: {[f[0] for f in sorted(weighted_ensemble_importance.items(), key=lambda x: x[1], reverse=True)[:5]]}")
                if shap_feature_importance:
                    logger.info(f"🔍 SHAP analysis completed with {len(shap_feature_importance)} features")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Ensemble model training failed: {e}")
            import traceback
            traceback.print_exc()
            pipeline_state = self.active_pipelines.get(pipeline_id, {})
            pipeline_state["training_error"] = str(e)
            return False
    
    async def _execute_model_registry(self, pipeline_id: str, strategy: str) -> bool:
        """Execute model registry stage with MLflow model registration"""
        logger.info(f"📦 Pipeline {pipeline_id}: Registering models for strategy '{strategy}'")
        
        try:
            import mlflow
            import os
            
            pipeline_state = self.active_pipelines[pipeline_id]
            training_results = pipeline_state.get("training_results", {})
            
            if not training_results:
                logger.error("❌ No training results found for model registry")
                return False
            
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
            
            run_id = training_results.get("mlflow_run_id")
            experiment_name = pipeline_state.get("experiment_name", f"pipeline_{strategy}")
            
            if run_id:
                try:
                    # Get the model URI
                    model_uri = f"runs:/{run_id}/model"
                    model_name = f"{experiment_name}_{strategy}_model"
                    
                    # Register model version
                    model_version = mlflow.register_model(model_uri, model_name)
                    
                    # Add model version tags using MLflow client
                    client = mlflow.MlflowClient()
                    client.set_model_version_tag(
                        model_name, 
                        model_version.version,
                        "strategy", 
                        strategy
                    )
                    client.set_model_version_tag(
                        model_name,
                        model_version.version, 
                        "pipeline_id",
                        pipeline_id
                    )
                    client.set_model_version_tag(
                        model_name,
                        model_version.version,
                        "accuracy",
                        str(training_results.get("accuracy", 0))
                    )
                    
                    pipeline_state["model_registry_results"] = {
                        "model_name": model_name,
                        "model_version": model_version.version,
                        "model_uri": model_uri,
                        "registration_timestamp": model_version.creation_timestamp
                    }
                    
                    logger.info(f"✅ Model registered: {model_name} v{model_version.version}")
                    return True
                    
                except Exception as e:
                    logger.error(f"❌ Model registration failed: {e}")
                    return False
            else:
                logger.error("❌ No MLflow run ID found for model registration")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model registry stage failed: {e}")
            pipeline_state = self.active_pipelines.get(pipeline_id, {})
            pipeline_state["registry_error"] = str(e)
            return False
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a pipeline"""
        if pipeline_id in self.active_pipelines:
            return self.active_pipelines[pipeline_id]
        
        # Check history
        for pipeline in self.pipeline_history:
            if pipeline["pipeline_id"] == pipeline_id:
                return pipeline
        
        return None
    
    def list_active_pipelines(self) -> List[Dict[str, Any]]:
        """List all currently active pipelines"""
        return list(self.active_pipelines.values())
    
    def list_pipeline_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List pipeline execution history"""
        return self.pipeline_history[-limit:]
    
    async def stop_pipeline(self, pipeline_id: str) -> bool:
        """Stop a running pipeline"""
        try:
            if pipeline_id in self.active_pipelines:
                pipeline_state = self.active_pipelines[pipeline_id]
                pipeline_state["status"] = "stopped"
                pipeline_state["end_time"] = datetime.now()
                
                # Move to history
                self.pipeline_history.append(pipeline_state.copy())
                del self.active_pipelines[pipeline_id]
                
                logger.info(f"🛑 Pipeline {pipeline_id} stopped")
                return True
            else:
                logger.warning(f"⚠️ Pipeline {pipeline_id} not found or not active")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to stop pipeline {pipeline_id}: {e}")
            return False
    
    def _apply_feature_balancing(self, df: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
        """
        Apply feature balancing to address lyrical feature dominance in multimodal training
        
        This method:
        1. Normalizes count-based lyrical features to prevent scale dominance
        2. Creates enhanced audio ratio features for better discrimination
        3. Adds missing discriminative audio features
        4. Ensures balanced representation between audio and lyrical modalities
        """
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        
        logger.info("⚖️ Implementing feature balancing for multimodal training...")
        
        # Create a copy to avoid modifying original
        balanced_df = df.copy()
        
        # 1. NORMALIZE COUNT-BASED LYRICAL FEATURES
        logger.info("📝 Normalizing lyrical count features to prevent scale dominance...")
        
        # Handle lyrics_word_count -> lyrics_word_count_normalized
        if 'lyrics_word_count' in balanced_df.columns:
            # Log normalize to reduce extreme values dominance
            word_count = balanced_df['lyrics_word_count'].fillna(0)
            balanced_df['lyrics_word_count_normalized'] = np.log1p(word_count) / np.log1p(word_count.max())
            logger.info(f"   ✅ Normalized lyrics_word_count: {word_count.min():.0f}-{word_count.max():.0f} → 0.0-1.0")
        
        # Handle lyrics_unique_words -> lyrics_unique_words_ratio  
        if 'lyrics_word_count' in balanced_df.columns and 'lyrics_unique_words' in balanced_df.columns:
            word_count = balanced_df['lyrics_word_count'].fillna(1)
            unique_words = balanced_df['lyrics_unique_words'].fillna(0)
            # Create lexical diversity ratio (0-1)
            balanced_df['lyrics_unique_words_ratio'] = unique_words / (word_count + 1)
            logger.info(f"   ✅ Created lyrics_unique_words_ratio from raw counts")
        
        # Handle lyrics_sentence_count -> lyrics_sentence_density
        if 'lyrics_sentence_count' in balanced_df.columns and 'lyrics_word_count' in balanced_df.columns:
            sentence_count = balanced_df['lyrics_sentence_count'].fillna(1)
            word_count = balanced_df['lyrics_word_count'].fillna(1)
            # Create sentence density (sentences per 100 words)
            balanced_df['lyrics_sentence_density'] = (sentence_count / (word_count + 1)) * 100
            # Normalize to 0-1 scale
            max_density = balanced_df['lyrics_sentence_density'].max()
            balanced_df['lyrics_sentence_density'] = balanced_df['lyrics_sentence_density'] / (max_density + 1e-8)
            logger.info(f"   ✅ Created normalized lyrics_sentence_density")
        
        # Handle lyrics_verse_count -> lyrics_verse_count_normalized
        if 'lyrics_verse_count' in balanced_df.columns:
            verse_count = balanced_df['lyrics_verse_count'].fillna(0)
            if verse_count.max() > 0:
                balanced_df['lyrics_verse_count_normalized'] = verse_count / (verse_count.max() + 1e-8)
            else:
                balanced_df['lyrics_verse_count_normalized'] = 0.0
            logger.info(f"   ✅ Normalized lyrics_verse_count to 0-1 scale")
        
        # 2. CREATE ENHANCED AUDIO RATIO FEATURES
        logger.info("🎧 Creating enhanced audio discrimination features...")
        
        # Energy-to-valence ratio (emotional intensity vs positivity)
        if 'audio_energy' in balanced_df.columns and 'audio_valence' in balanced_df.columns:
            energy = balanced_df['audio_energy'].fillna(0.5)
            valence = balanced_df['audio_valence'].fillna(0.5)
            balanced_df['audio_energy_valence_ratio'] = energy / (valence + 0.1)
            logger.info(f"   ✅ Created audio_energy_valence_ratio")
        
        # Danceability-tempo index (rhythmic appeal normalized by tempo)
        if 'audio_danceability' in balanced_df.columns and 'audio_tempo' in balanced_df.columns:
            danceability = balanced_df['audio_danceability'].fillna(0.5)
            tempo = balanced_df['audio_tempo'].fillna(120)
            # Normalize tempo to 0-1 (typical range 60-200 BPM)
            tempo_normalized = np.clip((tempo - 60) / 140, 0, 1)
            balanced_df['audio_rhythmic_intensity'] = danceability * tempo_normalized
            logger.info(f"   ✅ Created audio_rhythmic_intensity")
        
        # Timbral complexity score (brightness + complexity + warmth)
        timbral_features = ['audio_brightness', 'audio_complexity', 'audio_warmth']
        available_timbral = [f for f in timbral_features if f in balanced_df.columns]
        if len(available_timbral) >= 2:
            timbral_sum = sum(balanced_df[f].fillna(0.5) for f in available_timbral)
            balanced_df['audio_timbral_complexity'] = timbral_sum / len(available_timbral)
            logger.info(f"   ✅ Created audio_timbral_complexity from {len(available_timbral)} features")
        
        # 3. ADD MISSING AUDIO FEATURES TO SELECTED FEATURES LIST
        enhanced_audio_features = [
            'audio_energy_valence_ratio', 'audio_rhythmic_intensity', 'audio_timbral_complexity'
        ]
        balanced_lyrical_features = [
            'lyrics_word_count_normalized', 'lyrics_unique_words_ratio', 
            'lyrics_sentence_density', 'lyrics_verse_count_normalized'
        ]
        
        # Add new features to selected_features if they were created
        for feature in enhanced_audio_features + balanced_lyrical_features:
            if feature in balanced_df.columns and feature not in selected_features:
                selected_features.append(feature)
        
        # 4. FEATURE SCALING FOR REMAINING PROBLEMATIC FEATURES
        logger.info("📊 Applying standardization to ensure feature balance...")
        
        # Identify features that might still need scaling
        features_to_scale = []
        for feature in selected_features:
            if feature in balanced_df.columns:
                feature_values = balanced_df[feature].dropna()
                if len(feature_values) > 0 and feature_values.std() > 2.0:  # High variance features
                    features_to_scale.append(feature)
        
        if features_to_scale:
            scaler = StandardScaler()
            balanced_df[features_to_scale] = scaler.fit_transform(balanced_df[features_to_scale])
            logger.info(f"   ✅ Standardized {len(features_to_scale)} high-variance features")
        
        # 5. SUMMARY STATISTICS
        audio_features_count = sum(1 for f in selected_features if f.startswith('audio_'))
        lyrics_features_count = sum(1 for f in selected_features if f.startswith('lyrics_'))
        derived_features_count = len([f for f in selected_features if not f.startswith(('audio_', 'lyrics_'))])
        
        logger.info(f"📊 Feature balancing completed:")
        logger.info(f"   🎧 Audio features: {audio_features_count}")
        logger.info(f"   📝 Lyrical features: {lyrics_features_count}")
        logger.info(f"   🎯 Derived features: {derived_features_count}")
        logger.info(f"   📊 Total features: {len(selected_features)}")
        logger.info(f"   ⚖️ Audio/Lyrics ratio: {audio_features_count/(lyrics_features_count+1e-8):.2f}")
        
        return balanced_df

    async def cleanup(self):
        """Cleanup orchestrator resources"""
        logger.info("🧹 Cleaning up Pipeline Orchestrator")
        
        # Stop all active pipelines
        for pipeline_id in list(self.active_pipelines.keys()):
            await self.stop_pipeline(pipeline_id)
        
        logger.info("✅ Pipeline Orchestrator cleanup completed") 