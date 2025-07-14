"""
Workflow Audio Analysis Microservice

This microservice handles all audio analysis functionality including:
- Audio feature extraction using Essentia
- High-level feature computation
- Audio classification and analysis
- Integration with various audio extractors
- PERSISTENT STORAGE and data lineage tracking (NEW)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import uvicorn
from contextlib import asynccontextmanager
import uuid
import time
from datetime import datetime
import numpy as np

from .services.audio_analyzer import AudioAnalyzer
from .services.comprehensive_analyzer import ComprehensiveAudioAnalyzer
from .services.persistent_audio_analyzer import persistent_analyzer
from .models.audio_models import AudioAnalysisRequest, AudioAnalysisResponse
from .config.settings import get_settings
from .api.history import router as history_router
from .services.database_service import DatabaseService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global analyzers and settings
audio_analyzer = None
comprehensive_analyzer = None
settings = None
db_service = None

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events"""
    global audio_analyzer, comprehensive_analyzer, settings, db_service
    
    # Startup
    logger.info("Starting Workflow Audio Analysis Service with Persistent Storage")
    from .config.settings import ensure_directories
    ensure_directories()
    
    # Initialize persistent analyzer (CRITICAL FOR DATA PERSISTENCE)
    await persistent_analyzer.initialize()
    
    # Initialize services
    settings = get_settings()
    audio_analyzer = AudioAnalyzer()
    comprehensive_analyzer = ComprehensiveAudioAnalyzer()
    db_service = DatabaseService()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Workflow Audio Analysis Service")
    await persistent_analyzer.cleanup()

app = FastAPI(
    title="Workflow Audio Analysis Service",
    description="Microservice for comprehensive audio analysis with persistent storage and data lineage",
    version="2.0.0",  # Updated version to reflect persistent storage
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(history_router, prefix="/history", tags=["history"])

# =============================================================================
# SERVICE DISCOVERY ENDPOINT
# =============================================================================

@app.get("/features")
async def list_features():
    """
    Returns the actual feature names extracted by the audio analysis service
    
    This endpoint provides a flat list of feature names that ML training services
    can use for feature discovery and model training.
    """
    return {
        "service": "audio",
        "version": "2.0.0",
        "features": [
            # Basic audio features
            "acousticness",
            "instrumentalness", 
            "liveness",
            "speechiness",
            "brightness",
            "complexity",
            "warmth",
            "valence",
            "harmonic_strength",
            "key",
            "mode",
            "tempo",
            "danceability", 
            "energy",
            "loudness",
            "duration_ms",
            "time_signature",
            
            # Loudness features
            "integrated_lufs",
            "loudness_range_lu",
            "max_momentary_lufs",
            "max_short_term_lufs",
            "true_peak_dbtp",
            
            # Spectral features
            "spectral_centroid_mean",
            "spectral_centroid_var",
            "spectral_rolloff_mean",
            "spectral_rolloff_var",
            "spectral_spread_mean",
            "spectral_entropy_mean",
            "spectral_flux_mean",
            "spectral_kurtosis_mean",
            "spectral_skewness_mean",
            "spectral_energy_mean",
            
            # Rhythm features
            "bpm",
            "bpm_histogram_first_peak",
            "bpm_histogram_second_peak",
            "onset_rate",
            "beats_loudness_mean",
            "beats_position",
            
            # Tonal features
            "key_krumhansl_key",
            "key_krumhansl_strength",
            "key_temperley_key", 
            "key_temperley_strength",
            "chords_changes_rate",
            "chords_strength_mean",
            "chords_strength_var",
            "hpcp_mean",
            "hpcp_var",
            
            # Dynamics features
            "dynamic_complexity",
            "loudness_ebu128_integrated",
            "loudness_ebu128_range",
            "silence_rate_20db_mean",
            "silence_rate_30db_mean",
            "silence_rate_60db_mean",
            
            # MFCC features
            "mfcc_mean",
            
            # Genre features
            "primary_genre",
            "top_genre_confidence",
            
            # Mood features
            "mood_happy",
            "mood_sad", 
            "mood_aggressive",
            "mood_relaxed",
            "mood_party",
            "mood_electronic",
            "mood_acoustic"
        ],
        "feature_count": 64,
        "extractors": ["basic", "rhythm", "tonal", "timbre", "dynamics", "mood", "genre"],
        "description": "Audio features extracted using Essentia ML models with comprehensive analysis"
    }

@app.get("/features/legacy")
async def list_features_legacy():
    """
    [LEGACY] Original features documentation endpoint
    
    This is kept for backward compatibility. Use /features for the new format.
    """
    return {
        "service": "audio",
        "version": "2.0.0",
        "endpoint": "/analyze/audio",
        "description": "Audio analysis service providing comprehensive audio features",
        "capabilities": {
            "extractors": ["basic", "rhythm", "tonal", "timbre", "dynamics", "mood", "genre"],
            "total_features": 34,
            "analysis_time_avg": "3-5 seconds per song"
        },
        "features": {
            "basic": {
                "tempo": {"type": "float", "range": "60-200", "unit": "BPM", "description": "Beats per minute"},
                "energy": {"type": "float", "range": "0-1", "description": "Overall energy level"},
                "valence": {"type": "float", "range": "0-1", "description": "Musical positivity/happiness"},
                "danceability": {"type": "float", "range": "0-1", "description": "How suitable for dancing"},
                "acousticness": {"type": "float", "range": "0-1", "description": "Acoustic vs electronic"},
                "instrumentalness": {"type": "float", "range": "0-1", "description": "Vocal vs instrumental"},
                "liveness": {"type": "float", "range": "0-1", "description": "Live performance vs studio"},
                "speechiness": {"type": "float", "range": "0-1", "description": "Spoken word content"},
                "loudness": {"type": "float", "range": "-60-0", "unit": "dB", "description": "Overall loudness"}
            },
            "rhythm": {
                "beats_confidence": {"type": "float", "range": "0-1", "description": "Beat detection confidence"},
                "rhythm_stability": {"type": "float", "range": "0-1", "description": "Tempo consistency"},
                "time_signature": {"type": "integer", "range": "3-7", "description": "Beats per measure"}
            },
            "tonal": {
                "key": {"type": "string", "values": ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"], "description": "Musical key"},
                "mode": {"type": "string", "values": ["major", "minor"], "description": "Musical mode"},
                "key_strength": {"type": "float", "range": "0-1", "description": "Key detection confidence"}
            },
            "spectral": {
                "spectral_centroid": {"type": "float", "unit": "Hz", "description": "Brightness measure"},
                "spectral_rolloff": {"type": "float", "unit": "Hz", "description": "High frequency rolloff"},
                "zero_crossing_rate": {"type": "float", "description": "Rate of sign changes"},
                "mfcc": {"type": "array", "shape": [13], "description": "Mel frequency coefficients"}
            },
            "mood": {
                "mood_happy": {"type": "float", "range": "0-1", "description": "Happiness probability"},
                "mood_sad": {"type": "float", "range": "0-1", "description": "Sadness probability"},
                "mood_relaxed": {"type": "float", "range": "0-1", "description": "Relaxation probability"},
                "mood_aggressive": {"type": "float", "range": "0-1", "description": "Aggressiveness probability"}
            },
            "genre": {
                "genre_rock": {"type": "float", "range": "0-1", "description": "Rock genre probability"},
                "genre_pop": {"type": "float", "range": "0-1", "description": "Pop genre probability"},
                "genre_electronic": {"type": "float", "range": "0-1", "description": "Electronic genre probability"},
                "genre_classical": {"type": "float", "range": "0-1", "description": "Classical genre probability"}
            }
        },
        "response_structure": {
            "description": "Nested structure returned by /analyze/audio endpoint",
            "path_to_features": "results.features.analysis.basic",
            "structure": {
                "results": {
                    "features": {
                        "analysis": {
                            "basic": {
                                "tempo": "float",
                                "energy": "float", 
                                "valence": "float",
                                "danceability": "float",
                                "acousticness": "float",
                                "instrumentalness": "float",
                                "liveness": "float",
                                "speechiness": "float",
                                "loudness": "float"
                            },
                            "rhythm": {
                                "beats_confidence": "float",
                                "rhythm_stability": "float", 
                                "time_signature": "int"
                            },
                            "tonal": {
                                "key": "string",
                                "mode": "string",
                                "key_strength": "float"
                            },
                            "spectral": {
                                "spectral_centroid": "float",
                                "spectral_rolloff": "float",
                                "zero_crossing_rate": "float",
                                "mfcc": "array"
                            },
                            "mood": {
                                "mood_happy": "float",
                                "mood_sad": "float",
                                "mood_relaxed": "float",
                                "mood_aggressive": "float"
                            },
                            "genre": {
                                "genre_rock": "float",
                                "genre_pop": "float",
                                "genre_electronic": "float",
                                "genre_classical": "float"
                            }
                        }
                    }
                }
            }
        },
        "usage": {
            "example_call": "POST /analyze/audio with audio file",
            "parsing_example": "features = response['results']['features']['analysis']['basic']",
            "key_features_for_ml": ["tempo", "energy", "valence", "danceability", "acousticness", "mood_happy", "genre_rock"]
        }
    }

# =============================================================================
# SIMPLIFIED ANALYSIS ENDPOINTS - Consistent with content service
# =============================================================================

@app.post("/analyze/audio")
async def analyze_audio_simple(
    file: UploadFile = File(...),
    user_session: Optional[str] = Header(None, alias="X-Session-ID"),
    workflow_id: Optional[str] = Header(None, alias="Workflow-ID"),
    correlation_id: Optional[str] = Header(None, alias="Correlation-ID"),
    force_reanalysis: bool = False  # Add parameter to force reanalysis if needed
):
    """
    Comprehensive audio analysis endpoint with server-side deduplication
    
    This endpoint implements proper deduplication architecture:
    1. Checks for existing analysis by filename first (server-side deduplication)
    2. Returns existing analysis if found and not forcing reanalysis
    3. Only performs new analysis if no existing analysis found or force_reanalysis=True
    4. Automatically persists results and returns analysis_id for frontend reference
    
    This endpoint always performs comprehensive analysis with all extractors:
    - Basic Essentia features
    - Rhythm analysis (tempo, beats, rhythm patterns)
    - Tonal analysis (key, scale, harmonic analysis)
    - Timbre analysis (MFCC, spectral characteristics)
    - Dynamics analysis (loudness, dynamic range)
    - Mood analysis (emotional characteristics)
    - Genre classification
    """
    if not file.content_type or not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    original_filename = file.filename or "unknown.mp3"
    
    # STEP 1: Server-side deduplication check (unless forcing reanalysis)
    if not force_reanalysis:
        try:
            existing_analysis = await db_service.get_analysis_by_filename(original_filename)
            
            if existing_analysis and existing_analysis.get("status") == "completed":
                logger.info(f"✅ Server-side deduplication: Found existing analysis for {original_filename}")
                
                # Track reuse for audit purposes
                await db_service._log_event(
                    aggregate_id=existing_analysis["analysis_id"],
                    event_type="analysis_reused",
                    event_data={
                        "original_filename": original_filename,
                        "reused_by_session": user_session,
                        "workflow_id": workflow_id,
                        "correlation_id": correlation_id,
                        "endpoint": "/analyze/audio"
                    },
                    correlation_id=correlation_id
                )
                
                # Return existing analysis in the expected format
                return {
                    "status": "success",
                    "analysis_id": existing_analysis["analysis_id"],
                    "database_id": existing_analysis.get("database_id"),
                    "cached": True,  # Indicate this was served from cache/database
                    "analysis_type": "comprehensive",
                    "processing_time_ms": 0,  # No processing time since it was cached
                    "filename": original_filename,
                    "results": {
                        "features": existing_analysis.get("features", {}),
                        "extractor_types": ["audio", "rhythm", "tonal", "timbre", "dynamics", "mood", "genre"],
                        "metadata": {
                            "original_analysis_date": existing_analysis.get("completed_at"),
                            "original_processing_time_ms": existing_analysis.get("processing_time_ms", 0)
                        },
                        "extractors_used": ["audio", "rhythm", "tonal", "timbre", "dynamics", "mood", "genre"],
                        "comprehensive_analysis": True,
                        "deduplication_source": "server_side"
                    },
                    "audit": {
                        "reused_at": datetime.utcnow().isoformat(),
                        "original_created_at": existing_analysis.get("created_at"),
                        "original_completed_at": existing_analysis.get("completed_at"),
                        "workflow_id": workflow_id,
                        "correlation_id": correlation_id,
                        "requested_by": user_session,
                        "deduplication_method": "filename_based"
                    }
                }
                
        except Exception as e:
            # If deduplication check fails, log but continue with analysis
            logger.warning(f"Deduplication check failed for {original_filename}: {e}")
    
    # STEP 2: Perform new analysis (either no existing analysis found or force_reanalysis=True)
    logger.info(f"🎵 Performing {'forced ' if force_reanalysis else ''}new analysis for {original_filename}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(original_filename).suffix) as temp_file:
        try:
            # Save uploaded file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Perform comprehensive analysis using the existing infrastructure
            result = await persistent_analyzer.analyze_with_persistence(
                file_path=temp_file.name,
                analysis_type="comprehensive",  # Always use comprehensive analysis
                idempotency_key=f"{original_filename}-comprehensive-{user_session}",
                workflow_id=workflow_id or f"workflow-{int(time.time())}",
                correlation_id=correlation_id,
                requested_by=user_session or "anonymous",
                file_id=None,
                target_service="frontend",
                original_filename=original_filename,
                force_reanalysis=force_reanalysis  # Pass through the force flag
            )
            
            # Return in the format expected by the frontend
            return {
                "status": result.get("status", "success"),
                "analysis_id": result.get("analysis_id"),
                "database_id": result.get("database_id"),
                "cached": result.get("cached", False),
                "analysis_type": "comprehensive",
                "processing_time_ms": result.get("processing_time_ms", 0),
                "filename": original_filename,
                "results": {
                    "features": result.get("results", {}).get("features", {}),
                    "extractor_types": result.get("results", {}).get("extractor_types", []),
                    "metadata": result.get("results", {}).get("metadata", {}),
                    "extractors_used": result.get("results", {}).get("extractors_used", []),
                    "comprehensive_analysis": True,
                    "deduplication_source": "new_analysis"
                },
                "audit": result.get("audit", {})
            }
            
        except Exception as e:
            logger.error(f"Comprehensive audio analysis failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file.name)
            except:
                pass

@app.get("/analyze/features/audio")
async def get_audio_features(
    file_id: str,
    user_session: Optional[str] = Header(None, alias="X-Session-ID")
):
    """
    Extract comprehensive audio features for ML models
    Returns rich feature set from comprehensive analysis including all extractors
    """
    try:
        # Get analysis result from database
        result = await persistent_analyzer.get_analysis_for_prediction(file_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Extract comprehensive features for ML
        features = result.get("features", {})
        
        # Basic audio properties
        basic_features = {
            "duration_seconds": features.get("duration", 0),
            "sample_rate": features.get("sample_rate", 0),
            "channels": features.get("channels", 0),
        }
        
        # Rhythm features (from rhythm extractor)
        rhythm_features = {
            "tempo": features.get("tempo", 0),
            "beat_strength": features.get("beat_strength", 0),
            "rhythm_regularity": features.get("rhythm_regularity", 0),
            "onset_rate": features.get("onset_rate", 0),
        }
        
        # Spectral features (from timbre extractor)
        spectral_features = {
            "spectral_centroid": features.get("spectral_centroid", 0),
            "spectral_rolloff": features.get("spectral_rolloff", 0),
            "spectral_bandwidth": features.get("spectral_bandwidth", 0),
            "spectral_contrast": features.get("spectral_contrast", 0),
            "zero_crossing_rate": features.get("zero_crossing_rate", 0),
        }
        
        # Tonal features (from tonal extractor)
        tonal_features = {
            "key": features.get("key", "unknown"),
            "mode": features.get("mode", "unknown"),
            "key_strength": features.get("key_strength", 0),
            "chroma_energy": features.get("chroma_energy", 0),
            "harmonic_change_detection": features.get("harmonic_change_detection", 0),
        }
        
        # High-level features (calculated from comprehensive analysis)
        high_level_features = {
            "acousticness": features.get("acousticness", 0),
            "instrumentalness": features.get("instrumentalness", 0),
            "energy": features.get("energy", 0),
            "valence": features.get("valence", 0),
            "danceability": features.get("danceability", 0),
            "speechiness": features.get("speechiness", 0),
            "liveness": features.get("liveness", 0),
        }
        
        # Dynamics features (from dynamics extractor)
        dynamics_features = {
            "loudness": features.get("loudness", 0),
            "dynamic_range": features.get("dynamic_range", 0),
            "loudness_range": features.get("loudness_range", 0),
        }
        
        # Mood features (from mood extractor)
        mood_features = {
            "mood_happy": features.get("mood_happy", 0),
            "mood_sad": features.get("mood_sad", 0),
            "mood_aggressive": features.get("mood_aggressive", 0),
            "mood_relaxed": features.get("mood_relaxed", 0),
        }
        
        # Genre features (from genre extractor)
        genre_features = {
            "genre_electronic": features.get("genre_electronic", 0),
            "genre_rock": features.get("genre_rock", 0),
            "genre_pop": features.get("genre_pop", 0),
            "genre_classical": features.get("genre_classical", 0),
            "genre_confidence": features.get("genre_confidence", 0.5),
        }
        
        # Combine all features into comprehensive feature set
        comprehensive_features = {
            **basic_features,
            **rhythm_features,
            **spectral_features,
            **tonal_features,
            **high_level_features,
            **dynamics_features,
            **mood_features,
            **genre_features
        }
        
        return {
            "status": "success",
            "results": comprehensive_features,
            "analysis_id": result.get("analysis_id"),
            "feature_count": len(comprehensive_features),
            "extractors_used": result.get("extractors_used", []),
            "comprehensive_analysis": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting comprehensive audio features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# NEW PERSISTENT ANALYSIS ENDPOINTS - Fixes critical data loss issue
# =============================================================================

@app.post("/analyze/persistent/comprehensive")
async def analyze_audio_persistent_comprehensive(
    file: UploadFile = File(...),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    workflow_id: Optional[str] = Header(None, alias="Workflow-ID"),
    correlation_id: Optional[str] = Header(None, alias="Correlation-ID"),
    requested_by: Optional[str] = Header(None, alias="Requested-By"),
    file_id: Optional[str] = Header(None, alias="File-ID"),
    force_reanalysis: bool = False  # Add force reanalysis parameter
):
    """
    Comprehensive audio analysis with persistent storage and idempotency
    This is now the primary analysis endpoint for microservice integration
    """
    if not file.content_type or not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
        try:
            # Save uploaded file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Perform comprehensive analysis
            result = await persistent_analyzer.analyze_with_persistence(
                file_path=temp_file.name,
                analysis_type="comprehensive",
                idempotency_key=idempotency_key,
                workflow_id=workflow_id,
                correlation_id=correlation_id,
                requested_by=requested_by,
                file_id=file_id,
                target_service="ml-prediction",
                force_reanalysis=force_reanalysis  # Pass through force flag
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Persistent comprehensive analysis failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file.name)
            except:
                pass

# Rename the basic endpoint to be the main persistent endpoint
@app.post("/analyze/persistent")
async def analyze_audio_persistent(
    file: UploadFile = File(...),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    workflow_id: Optional[str] = Header(None, alias="Workflow-ID"),
    correlation_id: Optional[str] = Header(None, alias="Correlation-ID"),
    requested_by: Optional[str] = Header(None, alias="Requested-By"),
    file_id: Optional[str] = Header(None, alias="File-ID"),
    force_reanalysis: bool = False  # Add force reanalysis parameter
):
    """
    Main persistent audio analysis endpoint - always performs comprehensive analysis
    Use this endpoint for all microservice integrations
    """
    return await analyze_audio_persistent_comprehensive(
        file, idempotency_key, workflow_id, correlation_id, requested_by, file_id, force_reanalysis
    )

@app.post("/analyze/file")
async def analyze_file_persistent(
    request: AudioAnalysisRequest,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    workflow_id: Optional[str] = Header(None, alias="Workflow-ID"),
    correlation_id: Optional[str] = Header(None, alias="Correlation-ID"),
    requested_by: Optional[str] = Header(None, alias="Requested-By")
):
    """
    Analyze audio file by file path with comprehensive analysis and persistence
    This is the main endpoint for orchestrator and batch processing use
    """
    file_path = request.file_path
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        result = await persistent_analyzer.analyze_with_persistence(
            file_path=file_path,
            analysis_type="comprehensive",  # Always use comprehensive analysis
            idempotency_key=idempotency_key,
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            requested_by=requested_by or "workflow-orchestrator",
            file_id=getattr(request, 'file_id', None),
            target_service="ml-prediction",
            force_reanalysis=request.force_reanalysis or False  # Use force flag from request
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Persistent file analysis failed for {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Keep the old endpoint for backward compatibility but redirect to comprehensive
@app.post("/analyze/persistent/file/basic")
async def analyze_file_persistent_basic(
    request: AudioAnalysisRequest,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    workflow_id: Optional[str] = Header(None, alias="Workflow-ID"),
    correlation_id: Optional[str] = Header(None, alias="Correlation-ID"),
    requested_by: Optional[str] = Header(None, alias="Requested-By")
):
    """
    DEPRECATED: Use /analyze/file instead
    Analyze audio file by file path with persistence - redirects to comprehensive analysis
    """
    logger.warning("DEPRECATED endpoint /analyze/persistent/file/basic used. Use /analyze/file instead.")
    return await analyze_file_persistent(request, idempotency_key, workflow_id, correlation_id, requested_by)

@app.post("/analyze/persistent/file/comprehensive")
async def analyze_file_persistent_comprehensive(
    request: AudioAnalysisRequest,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    workflow_id: Optional[str] = Header(None, alias="Workflow-ID"),
    correlation_id: Optional[str] = Header(None, alias="Correlation-ID"),
    requested_by: Optional[str] = Header(None, alias="Requested-By")
):
    """
    DEPRECATED: Use /analyze/file instead
    Analyze audio file by file path with persistence - redirects to main endpoint
    """
    logger.warning("DEPRECATED endpoint /analyze/persistent/file/comprehensive used. Use /analyze/file instead.")
    return await analyze_file_persistent(request, idempotency_key, workflow_id, correlation_id, requested_by)

# =============================================================================
# DATA ACCESS ENDPOINTS - For other services to retrieve persisted data
# =============================================================================

@app.get("/analysis/{analysis_id}")
async def get_analysis_result(analysis_id: str):
    """
    Get persisted analysis result by ID
    This enables other services to access previously computed results
    """
    result = await persistent_analyzer.get_persisted_analysis(analysis_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return result

@app.get("/analysis/file/{file_id}")
async def get_analysis_by_file_id(file_id: str, analysis_type: str = "basic"):
    """
    Get analysis result by file ID
    This is used by ML prediction and AI insights services
    """
    result = await persistent_analyzer.get_analysis_for_prediction(file_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found for file")
    
    return result

@app.get("/analysis/filename/{filename}")
async def get_analysis_by_filename(filename: str):
    """
    Get analysis result by filename
    This is used by ML training service to check for existing analysis
    """
    try:
        logger.info(f"🎯 Getting analysis for filename: {filename}")
        
        # Search for existing analysis by filename using the database service
        existing_analysis = await db_service.get_analysis_by_filename(filename)
        
        if existing_analysis:
            logger.info(f"✅ Found existing analysis for {filename}")
            return {
                "status": "success",
                "filename": filename,
                "analysis_found": True,
                "analysis": existing_analysis
            }
        else:
            logger.info(f"❌ No existing analysis found for {filename}")
            raise HTTPException(status_code=404, detail="No existing analysis found for filename")
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error looking up analysis by filename {filename}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/analysis/workflow/{workflow_id}")
async def get_analyses_by_workflow(workflow_id: str):
    """
    Get all analysis results for a workflow
    Enables workflow-level data access and debugging
    """
    results = await persistent_analyzer.list_analyses_by_workflow(workflow_id)
    
    return {
        "workflow_id": workflow_id,
        "analysis_count": len(results),
        "analyses": results
    }

# =============================================================================
# BATCH ANALYSIS ENDPOINTS - For data quality and validation
# =============================================================================

@app.post("/analyze/batch/quality-check")
async def batch_quality_check(
    request: Dict[str, Any],
    correlation_id: Optional[str] = Header(None, alias="Correlation-ID")
):
    """
    Batch quality check for music datasets - MIR-focused data quality analysis
    
    This endpoint analyzes a directory of audio files and returns quality issues
    for data quality dashboards and validation systems.
    
    Expected request format:
    {
        "check_directory": "/path/to/audio/files",
        "include_metadata": true,
        "quality_thresholds": {
            "MIN_DURATION_SECONDS": 30,
            "MAX_DURATION_SECONDS": 900,
            "MIN_FILE_SIZE_MB": 0.5,
            "MAX_FILE_SIZE_MB": 50
        }
    }
    """
    try:
        import os
        import librosa
        import soundfile as sf
        from pathlib import Path
        import hashlib
        
        check_directory = request.get("check_directory", "/Users/manojveluchuri/saas/workflow/songs")
        include_metadata = request.get("include_metadata", True)
        quality_thresholds = request.get("quality_thresholds", {
            "MIN_DURATION_SECONDS": 30,
            "MAX_DURATION_SECONDS": 900,
            "MIN_FILE_SIZE_MB": 0.5,
            "MAX_FILE_SIZE_MB": 50,
            "PREFERRED_SAMPLE_RATES": [22050, 44100, 48000],
            "MIN_DYNAMIC_RANGE_DB": 6,
            "MAX_SILENCE_PERCENTAGE": 20,
            "MAX_CLIPPING_PERCENTAGE": 1
        })
        
        logger.info(f"🔍 Starting batch quality check for directory: {check_directory}")
        
        # Find all audio files
        audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg']
        audio_files = []
        issues = []
        
        if os.path.exists(check_directory):
            for root, dirs, files in os.walk(check_directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in audio_extensions):
                        audio_files.append(os.path.join(root, file))
        
        logger.info(f"📁 Found {len(audio_files)} audio files to analyze")
        
        analyzed_files = []
        
        for file_path in audio_files[:50]:  # Limit to 50 files for performance
            try:
                file_info = {
                    "path": file_path,
                    "name": os.path.basename(file_path),
                    "size_mb": os.path.getsize(file_path) / (1024 * 1024),
                    "format": Path(file_path).suffix.lower()
                }
                
                # Basic file checks
                if file_info["size_mb"] < quality_thresholds.get("MIN_FILE_SIZE_MB", 0.5):
                    issues.append({
                        "file_path": file_path,
                        "file_name": file_info["name"],
                        "issue_type": "too_small",
                        "severity": "critical",
                        "details": f"File size {file_info['size_mb']:.2f}MB below minimum {quality_thresholds.get('MIN_FILE_SIZE_MB', 0.5)}MB",
                        "file_size_mb": file_info["size_mb"],
                        "format": file_info["format"]
                    })
                
                if file_info["size_mb"] > quality_thresholds.get("MAX_FILE_SIZE_MB", 50):
                    issues.append({
                        "file_path": file_path,
                        "file_name": file_info["name"],
                        "issue_type": "too_large", 
                        "severity": "warning",
                        "details": f"File size {file_info['size_mb']:.2f}MB above recommended {quality_thresholds.get('MAX_FILE_SIZE_MB', 50)}MB",
                        "file_size_mb": file_info["size_mb"],
                        "format": file_info["format"]
                    })
                
                # Audio metadata analysis
                if include_metadata:
                    try:
                        # Use librosa for audio analysis
                        y, sr = librosa.load(file_path, sr=None, duration=30)  # Load first 30 seconds
                        duration = librosa.get_duration(filename=file_path)
                        
                        file_info.update({
                            "duration_seconds": duration,
                            "sample_rate": sr,
                            "quality_score": 100  # Start with perfect score
                        })
                        
                        # Duration checks
                        if duration < quality_thresholds.get("MIN_DURATION_SECONDS", 30):
                            issues.append({
                                "file_path": file_path,
                                "file_name": file_info["name"],
                                "issue_type": "too_short",
                                "severity": "critical",
                                "details": f"Duration {duration:.1f}s below minimum {quality_thresholds.get('MIN_DURATION_SECONDS', 30)}s",
                                "duration_seconds": duration,
                                "sample_rate": sr,
                                "format": file_info["format"]
                            })
                            file_info["quality_score"] -= 30
                        
                        if duration > quality_thresholds.get("MAX_DURATION_SECONDS", 900):
                            issues.append({
                                "file_path": file_path,
                                "file_name": file_info["name"],
                                "issue_type": "too_long",
                                "severity": "warning",
                                "details": f"Duration {duration:.1f}s above recommended {quality_thresholds.get('MAX_DURATION_SECONDS', 900)}s",
                                "duration_seconds": duration,
                                "sample_rate": sr,
                                "format": file_info["format"]
                            })
                            file_info["quality_score"] -= 10
                        
                        # Sample rate check
                        preferred_rates = quality_thresholds.get("PREFERRED_SAMPLE_RATES", [22050, 44100, 48000])
                        if sr not in preferred_rates:
                            issues.append({
                                "file_path": file_path,
                                "file_name": file_info["name"],
                                "issue_type": "unusual_sample_rate",
                                "severity": "info",
                                "details": f"Sample rate {sr}Hz not in preferred rates {preferred_rates}",
                                "sample_rate": sr,
                                "format": file_info["format"]
                            })
                            file_info["quality_score"] -= 5
                        
                        # Silence detection
                        silence_threshold = 0.01
                        silence_frames = np.sum(np.abs(y) < silence_threshold)
                        silence_percentage = (silence_frames / len(y)) * 100
                        
                        if silence_percentage > quality_thresholds.get("MAX_SILENCE_PERCENTAGE", 20):
                            issues.append({
                                "file_path": file_path,
                                "file_name": file_info["name"],
                                "issue_type": "excessive_silence",
                                "severity": "warning",
                                "details": f"Silence percentage {silence_percentage:.1f}% above threshold {quality_thresholds.get('MAX_SILENCE_PERCENTAGE', 20)}%",
                                "silence_percentage": silence_percentage,
                                "sample_rate": sr,
                                "format": file_info["format"]
                            })
                            file_info["quality_score"] -= 15
                        
                    except Exception as audio_error:
                        issues.append({
                            "file_path": file_path,
                            "file_name": file_info["name"],
                            "issue_type": "corrupt",
                            "severity": "critical",
                            "details": f"Cannot read audio file: {str(audio_error)}",
                            "format": file_info["format"]
                        })
                        file_info["quality_score"] = 0
                
                # Ensure quality score is non-negative
                file_info["quality_score"] = max(0, file_info.get("quality_score", 100))
                analyzed_files.append(file_info)
                
            except Exception as file_error:
                logger.warning(f"⚠️ Failed to analyze {file_path}: {file_error}")
                issues.append({
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "issue_type": "analysis_error",
                    "severity": "critical", 
                    "details": f"Analysis failed: {str(file_error)}",
                    "format": Path(file_path).suffix.lower()
                })
        
        logger.info(f"✅ Batch quality check completed. Found {len(issues)} issues across {len(analyzed_files)} files")
        
        return {
            "status": "success",
            "check_directory": check_directory,
            "total_files_found": len(audio_files),
            "files_analyzed": len(analyzed_files),
            "total_issues": len(issues),
            "critical_issues": len([i for i in issues if i["severity"] == "critical"]),
            "warning_issues": len([i for i in issues if i["severity"] == "warning"]),
            "info_issues": len([i for i in issues if i["severity"] == "info"]),
            "audio_files": analyzed_files,
            "issues": issues,
            "quality_thresholds_used": quality_thresholds,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "correlation_id": correlation_id
        }
        
    except Exception as e:
        logger.error(f"❌ Batch quality check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch quality check failed: {str(e)}")

# =============================================================================
# LEGACY ENDPOINTS - Maintained for backward compatibility
# =============================================================================

@app.post("/analyze/basic", response_model=AudioAnalysisResponse)
async def analyze_audio_basic(file: UploadFile = File(...)):
    """
    Basic audio analysis - LEGACY (no persistence)
    Use /analyze/persistent/basic for new implementations
    """
    if not file.content_type or not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
        try:
            # Save uploaded file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Analyze audio
            results = audio_analyzer.analyze_audio(temp_file.name)
            
            return AudioAnalysisResponse(
                status="success",
                filename=file.filename,
                analysis_type="basic",
                results=results
            )
            
        except Exception as e:
            logger.error(f"Basic analysis failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file.name)
            except:
                pass

@app.post("/analyze/comprehensive")
async def analyze_audio_comprehensive(file: UploadFile = File(...)):
    """
    Comprehensive audio analysis - LEGACY (no persistence)
    Use /analyze/persistent/comprehensive for new implementations
    """
    if not file.content_type or not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
        try:
            # Save uploaded file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Comprehensive analysis
            results = comprehensive_analyzer.analyze(temp_file.name)
            
            return JSONResponse(content={
                "status": "success",
                "filename": file.filename,
                "analysis_type": "comprehensive",
                "results": results
            })
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file.name)
            except:
                pass

@app.post("/analyze/file/basic")
async def analyze_file_basic(request: AudioAnalysisRequest):
    """
    Analyze audio file by file path - LEGACY (no persistence)
    Use /analyze/persistent/file/basic for new implementations
    """
    file_path = request.file_path
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        results = audio_analyzer.analyze_audio(file_path)
        
        return AudioAnalysisResponse(
            status="success",
            filename=Path(file_path).name,
            analysis_type="basic",
            results=results
        )
        
    except Exception as e:
        logger.error(f"Basic file analysis failed for {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/file/comprehensive")
async def analyze_file_comprehensive(request: AudioAnalysisRequest):
    """
    Analyze audio file by file path - LEGACY (no persistence)
    Use /analyze/persistent/file/comprehensive for new implementations
    """
    file_path = request.file_path
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        results = comprehensive_analyzer.analyze(file_path)
        
        return JSONResponse(content={
            "status": "success",
            "filename": Path(file_path).name,
            "analysis_type": "comprehensive",
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Comprehensive file analysis failed for {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/extractors")
async def list_extractors():
    """Get available audio extractors and their capabilities - all are used in comprehensive analysis"""
    return {
        "analysis_mode": "comprehensive_only",
        "available_extractors": [
            "audio",      # Basic audio properties and Essentia features
            "rhythm",     # Tempo, beats, rhythm patterns  
            "tonal",      # Key, scale, harmonic analysis
            "timbre",     # Spectral characteristics, MFCC
            "dynamics",   # Loudness, dynamic range
            "mood",       # Emotional characteristics
            "genre"       # Genre classification
        ],
        "extractor_info": {
            "audio": "Basic audio properties and Essentia features - foundation for all analysis",
            "rhythm": "Advanced tempo, beat tracking, and rhythm pattern analysis",
            "tonal": "Musical key detection, scale analysis, and harmonic content",
            "timbre": "Spectral characteristics, MFCC, and timbral texture analysis",
            "dynamics": "Loudness analysis, dynamic range, and volume characteristics",
            "mood": "Emotional content analysis and mood classification",
            "genre": "Automatic genre classification and style detection"
        },
        "features_extracted": {
            "total_features": "50+",
            "rhythm_features": ["tempo", "beat_strength", "rhythm_regularity", "onset_rate"],
            "spectral_features": ["spectral_centroid", "spectral_rolloff", "spectral_bandwidth", "spectral_contrast"],
            "tonal_features": ["key", "mode", "key_strength", "chroma_energy"],
            "high_level_features": ["danceability", "energy", "valence", "acousticness", "instrumentalness"],
            "mood_features": ["mood_happy", "mood_sad", "mood_aggressive", "mood_relaxed"],
            "genre_features": ["genre_electronic", "genre_rock", "genre_pop", "genre_classical"]
        },
        "comprehensive_analysis": True,
        "persistent_storage": True,
        "caching_enabled": settings.enable_feature_caching,
        "idempotency_enabled": settings.enable_idempotency,
        "processing_time": "5-15 seconds per song (comprehensive analysis)"
    }

@app.get("/health")
async def health_check():
    """Enhanced health check - comprehensive analysis service"""
    try:
        service_health = await persistent_analyzer.get_service_health()
        
        return {
            "status": "healthy",
            "service": "workflow-audio-analysis",
            "version": "2.0.0",
            "analysis_mode": "comprehensive_only",
            "capabilities": {
                "essentia_available": audio_analyzer.has_essentia_capability(),
                "extractors_available": True,
                "comprehensive_analysis": True,
                "persistent_storage": True,
                "idempotency_support": True,
                "data_lineage_tracking": True,
                "all_extractors_enabled": True
            },
            "supported_formats": [".wav", ".mp3", ".flac", ".aac", ".m4a"],
            "features": {
                "comprehensive_analysis": True,
                "basic_analysis_deprecated": True,
                "high_level_features": True,
                "spectral_analysis": True,
                "rhythm_analysis": True,
                "tonal_analysis": True,
                "mood_analysis": True,
                "genre_classification": True,
                "persistent_storage": True,
                "result_caching": True,
                "idempotency": True,
                "audit_trail": True
            },
            "extractors": {
                "total_extractors": 7,
                "always_enabled": ["audio", "rhythm", "tonal", "timbre", "dynamics", "mood", "genre"],
                "feature_count": "50+",
                "processing_time": "5-15 seconds per song"
            },
            "persistence_config": {
                "enable_result_persistence": settings.enable_result_persistence,
                "enable_feature_caching": settings.enable_feature_caching,
                "enable_idempotency": settings.enable_idempotency,
                "analysis_result_ttl_days": settings.analysis_result_ttl_days
            },
            "persistence": service_health.get("persistence", {}),
            "database_connected": service_health.get("persistence", {}).get("database", {}).get("status") == "healthy",
            "cache_connected": service_health.get("persistence", {}).get("redis", {}).get("status") == "healthy"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "service": "workflow-audio-analysis"
        }

# Legacy endpoint for backward compatibility
@app.get("/status")
async def get_status_legacy():
    """Legacy status endpoint - redirects to /health for consistency"""
    return await health_check()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.port) 