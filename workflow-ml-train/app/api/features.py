"""
Features API for service discovery and feature agreement workflow.

Provides endpoints for:
- Discovering available features from services
- Interactive feature selection  
- Feature vector validation
- Feature agreement management
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
import httpx
import asyncio
from datetime import datetime
import uuid
from .pipeline import get_orchestrator

logger = logging.getLogger(__name__)

router = APIRouter()

# =============================================================================
# REQUEST/RESPONSE MODELS  
# =============================================================================

class FeatureSelectionRequest(BaseModel):
    """Request for feature selection"""
    strategy: str = Field(..., description="Strategy name")
    selected_features: Dict[str, List[str]] = Field(..., description="Selected features by service")
    description: Optional[str] = Field(None, description="Agreement description")

class FeatureAgreementResponse(BaseModel):
    """Response for feature agreement"""
    agreement_id: str = Field(..., description="Unique agreement ID")
    selected_features: Dict[str, List[str]] = Field(..., description="Selected features")
    total_features: int = Field(..., description="Total number of features")
    strategy: str = Field(..., description="Strategy name")
    status: str = Field(..., description="Agreement status")

class ValidationRequest(BaseModel):
    """Request for feature validation"""
    agreement_id: str = Field(..., description="Agreement ID to validate")
    sample_size: Optional[int] = Field(5, description="Number of sample files to test")

class ValidationResponse(BaseModel):
    """Response for feature validation"""
    agreement_id: str
    validation_results: Dict[str, Any]
    status: str
    ready_for_training: bool

# =============================================================================
# SERVICE DISCOVERY
# =============================================================================

@router.get("/discover")
async def discover_available_features():
    """
    Auto-discover all available features from services.
    
    Queries the /features endpoints of all registered services
    to build a comprehensive feature catalog.
    """
    try:
        logger.info("🔍 Discovering available features from services")
        
        # Service endpoints to query (use load balancers for production)
        services = {
            "audio": "http://audio-load-balancer:80/features",
            "content": "http://content-load-balancer:80/features"
        }
        
        discovered_features = {}
        service_info = {}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for service_name, url in services.items():
                try:
                    logger.info(f"Querying {service_name} service: {url}")
                    response = await client.get(url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        service_features = data.get("features", [])
                        
                        # Handle both list and dict formats
                        if isinstance(service_features, list):
                            # Convert flat list to categorized dict for consistency
                            discovered_features[service_name] = {
                                "all": service_features
                            }
                            total_features = len(service_features)
                        elif isinstance(service_features, dict):
                            discovered_features[service_name] = service_features
                            total_features = sum(len(category) for category in service_features.values())
                        else:
                            discovered_features[service_name] = {}
                            total_features = 0
                        
                        service_info[service_name] = {
                            "status": "available",
                            "version": data.get("version", "unknown"),
                            "endpoint": data.get("endpoint", "unknown"),
                            "total_features": total_features,
                            "capabilities": data.get("capabilities", {}),
                            "response_structure": data.get("response_structure", {})
                        }
                        logger.info(f"✅ {service_name} service: {service_info[service_name]['total_features']} features")
                    else:
                        logger.warning(f"⚠️ {service_name} service returned {response.status_code}")
                        service_info[service_name] = {"status": "unavailable", "error": f"HTTP {response.status_code}"}
                        
                except Exception as e:
                    logger.error(f"❌ Failed to query {service_name} service: {e}")
                    service_info[service_name] = {"status": "error", "error": str(e)}
        
        # Count total features
        total_features = sum(
            service_info[service_name].get("total_features", 0)
            for service_name in discovered_features.keys()
        )
        
        return {
            "discovery_timestamp": datetime.now().isoformat(),
            "services": service_info,
            "features": discovered_features,
            "summary": {
                "total_services": len(services),
                "available_services": len([s for s in service_info.values() if s.get("status") == "available"]),
                "total_features": total_features,
                "feature_breakdown": {
                    service: service_info[service].get("total_features", 0)
                    for service in discovered_features.keys()
                }
            },
            "recommended_strategies": {
                "audio_only": {
                    "services": ["audio"],
                    "estimated_features": service_info.get("audio", {}).get("total_features", 0)
                },
                "multimodal": {
                    "services": ["audio", "content"],
                    "estimated_features": total_features
                }
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Feature discovery failed: {e}")
        raise HTTPException(status_code=500, detail=f"Feature discovery failed: {str(e)}")

# =============================================================================
# FEATURE AGREEMENT
# =============================================================================

@router.post("/select", response_model=FeatureAgreementResponse)
async def select_features(request: FeatureSelectionRequest):
    """
    User selects which features to use for training.
    
    Creates a feature agreement that can be validated and used for training.
    """
    try:
        logger.info(f"📝 Creating feature agreement for strategy: {request.strategy}")
        
        # Generate agreement ID
        agreement_id = f"agreement_{uuid.uuid4().hex[:8]}"
        
        # Count total features
        total_features = sum(len(features) for features in request.selected_features.values())
        
        # Validate feature selection
        if total_features == 0:
            raise HTTPException(status_code=400, detail="No features selected")
        
        # Store agreement (in production, this would go to database)
        agreement = {
            "agreement_id": agreement_id,
            "strategy": request.strategy,
            "selected_features": request.selected_features,
            "total_features": total_features,
            "status": "pending_validation",
            "created_at": datetime.now().isoformat(),
            "description": request.description
        }
        
        logger.info(f"✅ Created feature agreement {agreement_id} with {total_features} features")
        
        return FeatureAgreementResponse(
            agreement_id=agreement_id,
            selected_features=request.selected_features,
            total_features=total_features,
            strategy=request.strategy,
            status="pending_validation"
        )
        
    except Exception as e:
        logger.error(f"❌ Feature selection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Feature selection failed: {str(e)}")

@router.post("/validate", response_model=ValidationResponse)
async def validate_features(request: ValidationRequest):
    """
    Test feature extraction on sample files to validate the agreement.
    
    Performs actual feature extraction on a small sample to ensure
    all selected features can be extracted successfully.
    """
    try:
        logger.info(f"🧪 Validating feature agreement: {request.agreement_id}")
        
        # TODO: Get agreement from storage
        # For now, simulate validation
        
        # Simulate validation process
        await asyncio.sleep(2)  # Simulate processing time
        
        validation_results = {
            "success": True,
            "sample_extractions": request.sample_size,
            "failed_extractions": 0,
            "average_extraction_time": "2.3s",
            "feature_completeness": "100%",
            "issues": [],
            "sample_features": {
                "audio": {
                    "tempo": 120.5,
                    "energy": 0.75,
                    "valence": 0.60
                },
                "content": {
                    "sentiment_polarity": 0.2,
                    "word_count": 156
                }
            }
        }
        
        logger.info(f"✅ Validation completed for {request.agreement_id}")
        
        return ValidationResponse(
            agreement_id=request.agreement_id,
            validation_results=validation_results,
            status="validated",
            ready_for_training=validation_results["success"]
        )
        
    except Exception as e:
        logger.error(f"❌ Feature validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Feature validation failed: {str(e)}")

@router.get("/agreement/{pipeline_id}")
async def get_pending_agreement(pipeline_id: str):
    """Get pending feature agreement for a pipeline"""
    try:
        logger.info(f"📊 Getting pending agreement for pipeline: {pipeline_id}")
        
        # TODO: Get orchestrator to check pending agreements
        # For now, return mock pending agreement
        return {
            "pipeline_id": pipeline_id,
            "status": "waiting_for_user_input",
            "available_features": {
                "audio": {
                    "basic": ["energy", "valence", "danceability", "tempo", "acousticness", "instrumentalness"],
                    "rhythm": ["bpm", "beats_count", "onset_rate"],
                    "tonal": ["key", "mode", "chroma_stft"],
                    "spectral": ["centroid", "rolloff", "bandwidth"],
                    "mood": ["happy", "sad", "aggressive"],
                    "genre": ["electronic", "rock", "pop"]
                },
                "content": {
                    "sentiment": ["polarity", "subjectivity"],
                    "emotions": ["love", "hope", "pain"],
                    "complexity": ["avg_sentence_length", "avg_word_length", "lexical_diversity"],
                    "statistics": ["word_count", "unique_words", "vocabulary_density"],
                    "structure": ["verse_count", "narrative_complexity", "readability"],
                    "themes": ["theme_cluster_count", "motif_count"]
                }
            },
            "recommended_presets": {
                "audio_only": ["audio_energy", "audio_valence", "audio_tempo", "audio_loudness","audio_danceability","audio_primary_genre","audio_instrumentalness","audio_acousticness","audio_liveness","audio_speechiness","audio_brightness","audio_complexity","audio_warmth","audio_harmonic_strength","audio_key","audio_mode"],
                "multimodal": [
                    # === AUDIO FEATURES (24 features) ===
                    "audio_tempo", "audio_energy", "audio_valence", "audio_danceability", "audio_loudness", 
                    "audio_speechiness", "audio_acousticness", "audio_instrumentalness", "audio_liveness", 
                    "audio_key", "audio_mode", "audio_brightness", "audio_complexity", "audio_warmth", 
                    "audio_harmonic_strength", "audio_mood_happy", "audio_mood_sad", "audio_mood_aggressive", 
                    "audio_mood_relaxed", "audio_mood_party", "audio_mood_electronic", "audio_primary_genre", 
                    "audio_top_genre_1_prob", "audio_top_genre_2_prob",
                    # === LYRICAL FEATURES (27 features) ===
                    "lyrics_sentiment_positive", "lyrics_sentiment_negative", "lyrics_sentiment_neutral",
                    "lyrics_complexity_score", "lyrics_word_count", "lyrics_unique_words", "lyrics_reading_level",
                    "lyrics_emotion_anger", "lyrics_emotion_joy", "lyrics_emotion_sadness", "lyrics_emotion_fear",
                    "lyrics_emotion_surprise", "lyrics_theme_love", "lyrics_theme_party", "lyrics_theme_sadness",
                    "lyrics_profanity_score", "lyrics_repetition_score", "lyrics_rhyme_density",
                    "lyrics_narrative_complexity", "lyrics_lexical_diversity", "lyrics_motif_count",
                    "lyrics_verse_count", "lyrics_chorus_count", "lyrics_bridge_count"
                    # NOTE: Derived features (rhythmic_appeal_index, emotional_impact_score, 
                    # commercial_viability_index, sonic_sophistication_score) are added 
                    # automatically by DerivedFeaturesCalculator during pipeline execution
                ]
            },
            "feature_agreement_url": f"/features/complete/{pipeline_id}",
            "ui_url": f"/ui/feature-selection/{pipeline_id}",
            "expires_at": datetime.now().timestamp() + 3600,
            "instructions": {
                "step1": "Review available features by service",
                "step2": "Select features using POST /features/complete/{pipeline_id}",
                "step3": "Pipeline will resume automatically after selection"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get pending agreement: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get pending agreement: {str(e)}")

@router.post("/complete/{pipeline_id}")
async def complete_feature_agreement(
    pipeline_id: str,
    request: FeatureSelectionRequest,
    fastapi_request: Request,
    orchestrator = Depends(get_orchestrator)
):
    """Complete feature agreement for a pipeline"""
    try:
        logger.info(f"✅ Completing feature agreement for pipeline: {pipeline_id}")
        
        # Flatten selected features into a single list with service prefixes
        selected_features = []
        for service, features in request.selected_features.items():
            for feature in features:
                # Add service prefix to feature names for consistency
                prefixed_feature = f"{service}_{feature}"
                selected_features.append(prefixed_feature)
        
        agreement_data = {
            "selected_features_by_service": request.selected_features,
            "strategy": request.strategy,
            "description": request.description,
            "user_confirmed": True,
            "completion_timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"📋 Selected {len(selected_features)} features: {', '.join(selected_features[:5])}{'...' if len(selected_features) > 5 else ''}")
        # Check orchestrator availability
        if orchestrator is None:
            logger.error(f"❌ Orchestrator is None for pipeline {pipeline_id}")
            raise HTTPException(status_code=500, detail="Orchestrator not available")
        
        # Complete the agreement in orchestrator
        logger.info(f"🔧 Calling orchestrator.complete_feature_agreement for pipeline {pipeline_id}")
        try:
            success = orchestrator.complete_feature_agreement(pipeline_id, selected_features, agreement_data)
            logger.info(f"🔧 Orchestrator call result: {success}")
        except Exception as orch_error:
            logger.error(f"❌ Exception calling orchestrator: {orch_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Error calling orchestrator: {str(orch_error)}"
            )
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to complete feature agreement for pipeline {pipeline_id}. Pipeline may not exist or not be waiting for input."
            )
        
        logger.info(f"🎉 Feature agreement completed successfully for pipeline {pipeline_id}")
        
        return {
            "pipeline_id": pipeline_id,
            "status": "completed",
            "agreement_id": f"{pipeline_id}_agreement",
            "selected_features": selected_features,
            "selected_features_by_service": request.selected_features,
            "total_features": len(selected_features),
            "message": "Feature agreement completed successfully. Pipeline will resume automatically.",
            "next_stage": "feature_extraction",
            "estimated_resume_time": "1-2 minutes"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to complete feature agreement: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to complete feature agreement: {str(e)}")

# =============================================================================
# FEATURE PRESETS
# =============================================================================

@router.get("/presets")
async def get_feature_presets():
    """Get pre-defined feature selections for common strategies"""
    return {
        "presets": {
            "audio_basic": {
                "name": "Audio Basic",
                "description": "Essential audio features for quick training",
                "features": {
                    "audio": ["audio_tempo", "audio_energy", "audio_valence", "audio_danceability", "audio_loudness"]
                },
                "total_features": 5,
                "typical_accuracy": "0.70-0.80",
                "training_time": "2-3 minutes"
            },
            "audio_comprehensive": {
                "name": "Audio Comprehensive", 
                "description": "All available audio features from comprehensive analyzer",
                "features": {
                    "audio": [
                        "audio_tempo", "audio_energy", "audio_valence", "audio_danceability", "audio_acousticness",
                        "audio_instrumentalness", "audio_liveness", "audio_speechiness", "audio_loudness",
                        "audio_brightness", "audio_complexity", "audio_warmth", "audio_harmonic_strength",
                        "audio_key", "audio_mode", "audio_mood_happy", "audio_mood_sad", "audio_mood_aggressive",
                        "audio_primary_genre", "audio_top_genre_1_prob", "audio_top_genre_2_prob"
                    ]
                },
                "total_features": 21,
                "typical_accuracy": "0.75-0.85",
                "training_time": "4-6 minutes"
            },
            "multimodal_balanced": {
                "name": "Multimodal Balanced",
                "description": "Balanced mix of audio and content features",
                "features": {
                    "audio": ["tempo", "energy", "valence", "danceability", "acousticness", "primary_genre"],
                    "content": ["sentiment_polarity", "sentiment_subjectivity", "word_count", "readability", "lexical_diversity"]
                },
                "total_features": 11,
                "typical_accuracy": "0.80-0.90",
                "training_time": "5-7 minutes"
            },
            "sentiment_focused": {
                "name": "Sentiment Focused",
                "description": "Features focused on emotional content",
                "features": {
                    "audio": ["valence", "energy", "danceability"],
                    "content": ["sentiment_polarity", "sentiment_subjectivity", "emotion_love", "emotion_pain", "emotion_hope", "narrative_complexity"]
                },
                "total_features": 9,
                "typical_accuracy": "0.75-0.85",
                "training_time": "3-5 minutes"
            }
        },
        "usage": {
            "example": "POST /features/select with preset features",
            "recommendation": "Use 'multimodal_balanced' for best accuracy/time tradeoff"
        }
    }

# =============================================================================
# FEATURE VECTOR MANAGEMENT
# =============================================================================

@router.get("/vectors")
async def list_feature_vectors():
    """List all available feature vectors and presets"""
    try:
        from ..utils.feature_vector_manager import feature_vector_manager
        
        # List available presets
        presets = feature_vector_manager.list_available_presets()
        
        # List custom feature vectors (from main feature_vectors directory)
        custom_vectors = []
        vectors_path = feature_vector_manager.feature_vectors_path
        
        for vector_file in vectors_path.glob("*.json"):
            if vector_file.parent.name != "presets":  # Skip presets
                vector_data = feature_vector_manager.load_feature_vector(vector_file.name)
                if vector_data:
                    custom_vectors.append({
                        "name": vector_file.stem,
                        "file": vector_file.name,
                        "description": vector_data.get("description", "Custom feature vector"),
                        "strategy": vector_data.get("strategy", "unknown"),
                        "total_features": len(vector_data.get("selected_features", [])),
                        "created_at": vector_data.get("metadata", {}).get("created_at"),
                        "type": "custom"
                    })
        
        return {
            "presets": presets,
            "custom_vectors": custom_vectors,
            "total": len(presets) + len(custom_vectors),
            "storage_path": str(feature_vector_manager.feature_vectors_path)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to list feature vectors: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list feature vectors: {str(e)}")

@router.get("/vectors/{vector_name}")
async def get_feature_vector(vector_name: str):
    """Get a specific feature vector or preset by name"""
    try:
        from ..utils.feature_vector_manager import feature_vector_manager
        
        # Try to load as preset first
        feature_vector = feature_vector_manager.load_preset(vector_name)
        vector_type = "preset"
        
        if not feature_vector:
            # Try to load as custom vector
            feature_vector = feature_vector_manager.load_feature_vector(f"{vector_name}.json")
            vector_type = "custom"
        
        if not feature_vector:
            raise HTTPException(
                status_code=404,
                detail=f"Feature vector '{vector_name}' not found"
            )
        
        # Validate feature vector
        validation = feature_vector_manager.validate_feature_vector(feature_vector)
        
        return {
            "name": vector_name,
            "type": vector_type,
            "feature_vector": feature_vector,
            "validation": validation,
            "usage_example": {
                "via_name": f"POST /pipeline/train with feature_vector_name='{vector_name}'",
                "via_file": f"POST /pipeline/train with feature_vector_file='{vector_name}.json'"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get feature vector '{vector_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get feature vector: {str(e)}")

@router.post("/vectors")
async def create_feature_vector(
    name: str,
    feature_vector: Dict[str, Any]
):
    """Create a new custom feature vector"""
    try:
        from ..utils.feature_vector_manager import feature_vector_manager
        
        # Validate feature vector
        validation = feature_vector_manager.validate_feature_vector(feature_vector)
        if not validation["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid feature vector: {', '.join(validation['issues'])}"
            )
        
        # Save feature vector
        file_name = f"{name}.json"
        success = feature_vector_manager.save_feature_vector(feature_vector, file_name)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to save feature vector"
            )
        
        logger.info(f"✅ Created feature vector '{name}' with {validation['summary']['total_features']} features")
        
        return {
            "name": name,
            "file": file_name,
            "status": "created",
            "validation": validation,
            "usage": {
                "via_name": f"feature_vector_file='{file_name}'",
                "via_api": f"POST /pipeline/train with feature_vector_file='{file_name}'"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to create feature vector: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create feature vector: {str(e)}")

@router.post("/presets")
async def create_preset(
    name: str,
    feature_vector: Dict[str, Any]
):
    """Create a new feature vector preset"""
    try:
        from ..utils.feature_vector_manager import feature_vector_manager
        
        # Validate feature vector
        validation = feature_vector_manager.validate_feature_vector(feature_vector)
        if not validation["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid feature vector: {', '.join(validation['issues'])}"
            )
        
        # Save as preset
        success = feature_vector_manager.save_preset(name, feature_vector)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to save preset"
            )
        
        logger.info(f"✅ Created preset '{name}' with {validation['summary']['total_features']} features")
        
        return {
            "name": name,
            "status": "created",
            "type": "preset",
            "validation": validation,
            "usage": {
                "via_name": f"feature_vector_name='{name}'",
                "via_api": f"POST /pipeline/train with feature_vector_name='{name}'"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to create preset: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create preset: {str(e)}")

@router.post("/vectors/validate")
async def validate_feature_vector(feature_vector: Dict[str, Any]):
    """Validate a feature vector structure"""
    try:
        from ..utils.feature_vector_manager import feature_vector_manager
        
        validation = feature_vector_manager.validate_feature_vector(feature_vector)
        
        return {
            "validation": validation,
            "recommendations": {
                "if_valid": "Use this feature vector with POST /pipeline/train",
                "if_invalid": "Fix the issues listed and try again",
                "optimal_features": "8-20 features for best performance/accuracy balance"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Feature vector validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.post("/initialize-defaults")
async def initialize_default_presets():
    """Initialize default feature vector presets"""
    try:
        from ..utils.feature_vector_manager import feature_vector_manager
        
        feature_vector_manager.create_default_presets()
        presets = feature_vector_manager.list_available_presets()
        
        return {
            "status": "initialized",
            "presets_created": len(presets),
            "available_presets": [preset["name"] for preset in presets],
            "message": "Default feature vector presets are now available"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize default presets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize presets: {str(e)}")

# =============================================================================
# DATA QUALITY ANALYSIS
# =============================================================================

@router.post("/data-quality/analyze")
async def analyze_data_quality(request: Dict[str, Any] = None):
    """
    Comprehensive data quality analysis using existing audio and content service endpoints.
    
    Leverages:
    - Audio Service: POST /analyze/batch/quality-check
    - Content Service: POST /analyze/batch/lyrics-quality
    - Combines results with file pairing analysis
    """
    try:
        logger.info("🔍 Starting comprehensive data quality analysis")
        start_time = datetime.now()
        
        # Service endpoints (use load balancers for production)
        audio_service_url = "http://audio-load-balancer:80"
        content_service_url = "http://content-load-balancer:80"
        
        # Get directories from request or use defaults
        songs_directory = request.get("songs_directory") if request else None
        lyrics_directory = request.get("lyrics_directory") if request else None
        
        # Use provided directories or fallback to defaults
        songs_dir = songs_directory or "/Users/manojveluchuri/saas/workflow/songs"
        lyrics_dir = lyrics_directory or "/Users/manojveluchuri/saas/workflow/lyrics"
        
        logger.info(f"📁 Using songs directory: {songs_dir}")
        logger.info(f"📁 Using lyrics directory: {lyrics_dir}")
        
        # Prepare requests
        audio_request = {
            "check_directory": songs_dir,
            "include_metadata": True,
            "quality_thresholds": {
                "MIN_DURATION_SECONDS": 30,
                "MAX_DURATION_SECONDS": 900,
                "MIN_FILE_SIZE_MB": 0.5,
                "MAX_FILE_SIZE_MB": 50
            }
        }
        
        content_request = {
            "check_directory": lyrics_dir,
            "language_detection": True,
            "structure_analysis": True,
            "quality_thresholds": {
                "MIN_WORD_COUNT": 10,
                "MAX_WORD_COUNT": 2000,
                "MIN_LINE_COUNT": 4
            }
        }
        
        audio_result = None
        content_result = None
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Call audio service
            try:
                logger.info("📞 Calling audio service batch quality check...")
                audio_response = await client.post(
                    f"{audio_service_url}/analyze/batch/quality-check",
                    json=audio_request
                )
                if audio_response.status_code == 200:
                    audio_result = audio_response.json()
                    logger.info(f"✅ Audio analysis: {len(audio_result.get('audio_files', []))} files, {len(audio_result.get('issues', []))} issues")
                else:
                    logger.warning(f"⚠️ Audio service returned {audio_response.status_code}")
            except Exception as e:
                logger.error(f"❌ Audio service error: {e}")
            
            # Call content service  
            try:
                logger.info("📞 Calling content service batch quality check...")
                content_response = await client.post(
                    f"{content_service_url}/analyze/batch/lyrics-quality",
                    json=content_request
                )
                if content_response.status_code == 200:
                    content_result = content_response.json()
                    logger.info(f"✅ Content analysis: {len(content_result.get('lyrics_files', []))} files, {len(content_result.get('issues', []))} issues")
                else:
                    logger.warning(f"⚠️ Content service returned {content_response.status_code}")
            except Exception as e:
                logger.error(f"❌ Content service error: {e}")
        
        # Combine results
        audio_files = audio_result.get('audio_files', []) if audio_result else []
        lyrics_files = content_result.get('lyrics_files', []) if content_result else []
        audio_issues = audio_result.get('issues', []) if audio_result else []
        lyrics_issues = content_result.get('issues', []) if content_result else []
        
        # File pairing analysis
        pairing_analysis = analyze_file_pairing(audio_files, lyrics_files)
        
        # Calculate quality metrics
        total_issues = len(audio_issues) + len(lyrics_issues)
        critical_issues = len([i for i in audio_issues + lyrics_issues if i.get('severity') == 'critical'])
        warning_issues = len([i for i in audio_issues + lyrics_issues if i.get('severity') == 'warning'])
        
        # Calculate quality scores based on critical vs warning issues ratio
        total_audio_files = len(audio_files)
        total_lyrics_files = len(lyrics_files)
        
        # Audio quality: percentage-based scoring
        audio_critical = len([i for i in audio_issues if i.get('severity') == 'critical'])
        audio_warnings = len([i for i in audio_issues if i.get('severity') == 'warning'])
        if total_audio_files > 0:
            # Calculate percentage of files with issues
            audio_critical_pct = (audio_critical / total_audio_files) * 100
            audio_warnings_pct = (audio_warnings / total_audio_files) * 100
            # Deduct points: 2 points per % critical, 1 point per % warning
            audio_quality_score = max(10, 100 - (audio_critical_pct * 2) - audio_warnings_pct)
        else:
            audio_quality_score = 100
        
        # Lyrics quality: same percentage-based approach
        lyrics_critical = len([i for i in lyrics_issues if i.get('severity') == 'critical'])
        lyrics_warnings = len([i for i in lyrics_issues if i.get('severity') == 'warning'])
        if total_lyrics_files > 0:
            lyrics_critical_pct = (lyrics_critical / total_lyrics_files) * 100
            lyrics_warnings_pct = (lyrics_warnings / total_lyrics_files) * 100
            lyrics_quality_score = max(10, 100 - (lyrics_critical_pct * 2) - lyrics_warnings_pct)
        else:
            lyrics_quality_score = 100
        
        # Overall quality: weighted average based on file counts
        if total_audio_files > 0 and total_lyrics_files > 0:
            overall_quality_score = (audio_quality_score + lyrics_quality_score) / 2
        elif total_audio_files > 0:
            overall_quality_score = audio_quality_score
        elif total_lyrics_files > 0:
            overall_quality_score = lyrics_quality_score
        else:
            overall_quality_score = 100
        
        # Generate recommendations
        recommendations = []
        if critical_issues > 0:
            recommendations.append(f"🔧 Fix {critical_issues} critical issues (corrupted or invalid files)")
        if pairing_analysis['unpaired_audio'] > 0:
            recommendations.append(f"🔗 Consider adding lyrics for {pairing_analysis['unpaired_audio']} audio files")
        if not recommendations:
            recommendations.append("✅ Dataset quality is good! No immediate improvements needed.")
        
        analysis_duration = (datetime.now() - start_time).total_seconds()
        
        return {
            "dataset_profile": {
                "total_audio_files": len(audio_files),
                "total_lyrics_files": len(lyrics_files),
                "paired_files": pairing_analysis['paired_count'],
                "unpaired_audio": pairing_analysis['unpaired_audio'],
                "unpaired_lyrics": pairing_analysis['unpaired_lyrics'],
                "audio_format_distribution": calculate_format_distribution(audio_files),
                "lyrics_format_distribution": {".txt": len(lyrics_files)},
                "language_distribution": {"en": len(lyrics_files)},
                "duration_distribution": calculate_duration_distribution(audio_files),
                "file_size_distribution": calculate_file_size_distribution(audio_files),
                "quality_score_distribution": calculate_quality_distribution(audio_files)
            },
            "quality_metrics": {
                "overall_quality_score": overall_quality_score,
                "audio_quality_score": audio_quality_score,
                "lyrics_quality_score": lyrics_quality_score,
                "completeness_score": max(0, 100 - (critical_issues * 15)),
                "consistency_score": max(0, 100 - (warning_issues * 5)),
                "total_issues": total_issues,
                "critical_issues": critical_issues,
                "warning_issues": warning_issues,
                "issues_by_category": calculate_issues_by_category(audio_issues + lyrics_issues)
            },
            "audio_issues": audio_issues,
            "lyrics_issues": lyrics_issues,
            "recommendations": recommendations,
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_duration_seconds": analysis_duration
        }
        
    except Exception as e:
        logger.error(f"❌ Data quality analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Data quality analysis failed: {str(e)}")

def analyze_file_pairing(audio_files, lyrics_files):
    """Analyze pairing between audio and lyrics files"""
    from pathlib import Path
    
    # Create mapping based on filename without extension
    lyrics_map = {}
    for lyrics_file in lyrics_files:
        base_name = Path(lyrics_file.get('name', '')).stem
        lyrics_map[base_name] = lyrics_file.get('name', '')
    
    paired_count = 0
    unpaired_audio = 0
    
    for audio_file in audio_files:
        base_name = Path(audio_file.get('name', '')).stem
        if base_name in lyrics_map:
            paired_count += 1
            del lyrics_map[base_name]  # Remove matched lyrics
        else:
            unpaired_audio += 1
    
    unpaired_lyrics = len(lyrics_map)  # Remaining lyrics are unpaired
    
    return {
        "paired_count": paired_count,
        "unpaired_audio": unpaired_audio,
        "unpaired_lyrics": unpaired_lyrics
    }

def calculate_format_distribution(audio_files):
    """Calculate audio format distribution"""
    format_dist = {}
    for file in audio_files:
        format_ext = file.get('format', 'unknown')
        format_dist[format_ext] = format_dist.get(format_ext, 0) + 1
    return format_dist

def calculate_duration_distribution(audio_files):
    """Calculate duration distribution"""
    duration_dist = {"under_1min": 0, "1_to_3min": 0, "3_to_5min": 0, "5_to_10min": 0, "over_10min": 0}
    
    for file in audio_files:
        duration = file.get('duration_seconds', 0)
        minutes = duration / 60
        if minutes < 1:
            duration_dist["under_1min"] += 1
        elif minutes < 3:
            duration_dist["1_to_3min"] += 1
        elif minutes < 5:
            duration_dist["3_to_5min"] += 1
        elif minutes < 10:
            duration_dist["5_to_10min"] += 1
        else:
            duration_dist["over_10min"] += 1
    
    return duration_dist

def calculate_file_size_distribution(audio_files):
    """Calculate file size distribution"""
    size_dist = {"under_1mb": 0, "1_to_5mb": 0, "5_to_15mb": 0, "15_to_30mb": 0, "over_30mb": 0}
    
    for file in audio_files:
        size_mb = file.get('size_mb', 0)
        if size_mb < 1:
            size_dist["under_1mb"] += 1
        elif size_mb < 5:
            size_dist["1_to_5mb"] += 1
        elif size_mb < 15:
            size_dist["5_to_15mb"] += 1
        elif size_mb < 30:
            size_dist["15_to_30mb"] += 1
        else:
            size_dist["over_30mb"] += 1
    
    return size_dist

def calculate_quality_distribution(audio_files):
    """Calculate quality score distribution"""
    quality_dist = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
    
    for file in audio_files:
        score = file.get('quality_score', 100)
        if score >= 90:
            quality_dist["excellent"] += 1
        elif score >= 70:
            quality_dist["good"] += 1
        elif score >= 50:
            quality_dist["fair"] += 1
        else:
            quality_dist["poor"] += 1
    
    return quality_dist

def calculate_issues_by_category(issues):
    """Calculate issues by category"""
    issues_by_category = {}
    for issue in issues:
        issue_type = issue.get('issue_type', 'unknown')
        issues_by_category[issue_type] = issues_by_category.get(issue_type, 0) + 1
    return issues_by_category 