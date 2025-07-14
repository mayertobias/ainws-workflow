"""
Prediction API endpoints for workflow-ml-prediction service
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
import time
import uuid
import logging
from typing import Dict, Any, List, Optional
import asyncio

from ..models.prediction import (
    SinglePredictionRequest,
    SinglePredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeatureValidationRequest,
    FeatureValidationResponse,
    ModelListResponse,
    ModelLoadRequest,
    ModelLoadResponse,
    ExplainabilityRequest,
    ExplainabilityResponse,
    ABTestRequest,
    ABTestResponse,
    PredictionStatus,
    PredictionMetrics
)
from ..models.responses import SuccessResponse, ErrorResponse
from ..services.predictor import MLPredictorService, SmartSongPredictor

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize smart predictor
smart_predictor = SmartSongPredictor()

# Dependency for ML predictor
def get_ml_predictor() -> MLPredictorService:
    """Dependency to get ML predictor instance"""
    return MLPredictorService()

@router.post("/single", response_model=SinglePredictionResponse)
async def predict_single(
    request: SinglePredictionRequest,
    predictor: MLPredictorService = Depends(get_ml_predictor)
):
    """
    Make a single prediction
    """
    try:
        logger.info(f"Making single prediction with model: {request.model_id}")
        
        # Validate features first
        validation = await predictor.validate_features(request.features, request.model_id)
        if not validation['is_valid']:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid features: {validation['missing_features']} missing, {validation['invalid_features']} invalid"
            )
        
        # Make prediction
        result, was_cached = await predictor.predict_single(
            model_id=request.model_id,
            features=request.features,
            include_confidence=request.include_confidence,
            include_feature_importance=request.include_feature_importance,
            use_cache=request.use_cache
        )
        
        prediction_id = str(uuid.uuid4())
        
        return SinglePredictionResponse(
            status=PredictionStatus.CACHED if was_cached else PredictionStatus.COMPLETED,
            prediction=result,
            model_id=request.model_id,
            prediction_id=prediction_id,
            cached=was_cached
        )
        
    except ValueError as e:
        logger.error(f"Validation error in single prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
    except Exception as e:
        logger.error(f"Error in single prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    predictor: MLPredictorService = Depends(get_ml_predictor)
):
    """
    Make batch predictions
    """
    try:
        logger.info(f"Making batch prediction with model: {request.model_id} for {len(request.features_list)} items")
        
        # Validate batch size
        if len(request.features_list) > 1000:
            raise HTTPException(status_code=400, detail="Batch size exceeds maximum of 1000")
        
        # Validate features (sample validation on first item)
        if request.features_list:
            validation = await predictor.validate_features(request.features_list[0], request.model_id)
            if not validation['is_valid']:
                logger.warning(f"Feature validation warnings: {validation['warnings']}")
        
        start_time = time.time()
        
        # Make batch predictions
        results = await predictor.predict_batch(
            model_id=request.model_id,
            features_list=request.features_list,
            include_confidence=request.include_confidence,
            include_feature_importance=request.include_feature_importance
        )
        
        processing_time = (time.time() - start_time) * 1000
        batch_id = request.batch_id or str(uuid.uuid4())
        
        # Count successful predictions (all should be successful if we reach here)
        successful_predictions = len(results)
        failed_predictions = len(request.features_list) - successful_predictions
        
        return BatchPredictionResponse(
            status=PredictionStatus.COMPLETED,
            batch_id=batch_id,
            total_predictions=len(request.features_list),
            successful_predictions=successful_predictions,
            failed_predictions=failed_predictions,
            predictions=results,
            model_id=request.model_id,
            processing_time_ms=processing_time
        )
        
    except ValueError as e:
        logger.error(f"Validation error in batch prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/validate-features", response_model=FeatureValidationResponse)
async def validate_features(
    request: FeatureValidationRequest,
    predictor: MLPredictorService = Depends(get_ml_predictor)
):
    """
    Validate feature input
    """
    try:
        logger.info("Validating features")
        
        validation_result = await predictor.validate_features(
            features=request.features,
            model_id=request.model_id
        )
        
        return FeatureValidationResponse(**validation_result)
        
    except Exception as e:
        logger.error(f"Error validating features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models", response_model=ModelListResponse)
async def list_models(
    predictor: MLPredictorService = Depends(get_ml_predictor)
):
    """
    List available models
    """
    try:
        logger.info("Listing available models")
        
        models = await predictor.get_available_models()
        cached_count = sum(1 for model in models if model.is_cached)
        
        return ModelListResponse(
            models=models,
            total_count=len(models),
            cached_count=cached_count
        )
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/load", response_model=ModelLoadResponse)
async def load_model(
    request: ModelLoadRequest,
    predictor: MLPredictorService = Depends(get_ml_predictor)
):
    """
    Load a model into memory
    """
    try:
        logger.info(f"Loading model: {request.model_id}")
        
        start_time = time.time()
        
        # Load model
        model, metadata = await predictor.load_model(
            model_id=request.model_id,
            force_reload=request.force_reload
        )
        
        load_time = (time.time() - start_time) * 1000
        
        # Get model size (rough estimate)
        import sys
        model_size_mb = sys.getsizeof(model) / (1024 * 1024)
        
        return ModelLoadResponse(
            model_id=request.model_id,
            status="loaded",
            load_time_ms=load_time,
            model_size_mb=model_size_mb,
            feature_columns=metadata.get('feature_columns', []),
            is_cached=True
        )
        
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_id}/info")
async def get_model_info(
    model_id: str,
    predictor: MLPredictorService = Depends(get_ml_predictor)
):
    """
    Get detailed information about a model
    """
    try:
        logger.info(f"Getting info for model: {model_id}")
        
        # Get metadata
        metadata = await predictor._get_model_metadata(model_id)
        is_cached = predictor.model_cache.get(model_id) is not None
        
        return {
            "status": "success",
            "model_id": model_id,
            "metadata": metadata,
            "is_cached": is_cached,
            "cache_stats": predictor.model_cache.get_stats()
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/models/{model_id}/cache")
async def evict_model_from_cache(
    model_id: str,
    predictor: MLPredictorService = Depends(get_ml_predictor)
):
    """
    Evict a model from cache
    """
    try:
        logger.info(f"Evicting model from cache: {model_id}")
        
        predictor.model_cache.evict(model_id)
        
        return {
            "status": "success",
            "message": f"Model {model_id} evicted from cache",
            "cache_stats": predictor.model_cache.get_stats()
        }
        
    except Exception as e:
        logger.error(f"Error evicting model from cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/explain", response_model=ExplainabilityResponse)
async def explain_prediction(
    request: ExplainabilityRequest,
    predictor: MLPredictorService = Depends(get_ml_predictor)
):
    """
    Get prediction explainability
    """
    try:
        logger.info(f"Explaining prediction for model: {request.model_id}")
        
        # Load model
        model, metadata = await predictor.load_model(request.model_id)
        
        # Make prediction
        feature_array, feature_names = await predictor._prepare_features(
            request.features, metadata['feature_columns']
        )
        prediction = model.predict(feature_array)[0]
        
        # Get feature importance
        feature_importance = await predictor._get_feature_importance(model, feature_names)
        
        # Create feature contributions (simplified)
        feature_contributions = {}
        if feature_importance:
            feature_dict = request.features.dict(exclude_none=True)
            for feature, importance in feature_importance.items():
                if feature in feature_dict:
                    # Simplified contribution calculation
                    contribution = importance * feature_dict[feature] * prediction
                    feature_contributions[feature] = float(contribution)
        
        return ExplainabilityResponse(
            model_id=request.model_id,
            explanation={
                "method": request.explanation_type,
                "features_used": feature_names,
                "feature_values": request.features.dict(exclude_none=True)
            },
            feature_contributions=feature_contributions,
            base_value=0.5,  # Simplified base value
            prediction_value=float(prediction)
        )
        
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
    except Exception as e:
        logger.error(f"Error explaining prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ab-test", response_model=ABTestResponse)
async def ab_test_models(
    request: ABTestRequest,
    predictor: MLPredictorService = Depends(get_ml_predictor)
):
    """
    A/B test between two models
    """
    try:
        logger.info(f"A/B testing models: {request.model_a_id} vs {request.model_b_id}")
        
        # Make predictions with both models
        result_a, _ = await predictor.predict_single(
            model_id=request.model_a_id,
            features=request.features,
            include_confidence=True,
            include_feature_importance=False,
            use_cache=True
        )
        
        result_b, _ = await predictor.predict_single(
            model_id=request.model_b_id,
            features=request.features,
            include_confidence=True,
            include_feature_importance=False,
            use_cache=True
        )
        
        # Calculate difference
        difference = abs(result_a.hit_score - result_b.hit_score)
        
        # Simple recommendation based on confidence
        recommendation = request.model_a_id
        if result_b.confidence_interval and result_a.confidence_interval:
            if result_b.confidence_interval['margin'] < result_a.confidence_interval['margin']:
                recommendation = request.model_b_id
        
        test_id = request.test_id or str(uuid.uuid4())
        
        return ABTestResponse(
            test_id=test_id,
            model_a_result=result_a,
            model_b_result=result_b,
            difference=difference,
            confidence_comparison={
                "model_a_margin": result_a.confidence_interval['margin'] if result_a.confidence_interval else None,
                "model_b_margin": result_b.confidence_interval['margin'] if result_b.confidence_interval else None
            },
            recommendation=recommendation
        )
        
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", response_model=PredictionMetrics)
async def get_prediction_metrics(
    predictor: MLPredictorService = Depends(get_ml_predictor)
):
    """
    Get prediction service metrics
    """
    try:
        metrics = await predictor.get_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache/stats")
async def get_cache_stats(
    predictor: MLPredictorService = Depends(get_ml_predictor)
):
    """
    Get cache statistics
    """
    try:
        cache_stats = predictor.model_cache.get_stats()
        
        # Add Redis cache stats if available
        redis_stats = {}
        if predictor.redis_client:
            try:
                redis_info = predictor.redis_client.info()
                redis_stats = {
                    "redis_memory_used": redis_info.get('used_memory_human', 'unknown'),
                    "redis_keys": predictor.redis_client.dbsize(),
                    "redis_connected": True
                }
            except Exception:
                redis_stats = {"redis_connected": False}
        
        return {
            "status": "success",
            "model_cache": cache_stats,
            "redis_cache": redis_stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cache/clear")
async def clear_cache(
    predictor: MLPredictorService = Depends(get_ml_predictor)
):
    """
    Clear all caches
    """
    try:
        logger.info("Clearing all caches")
        
        # Clear model cache
        models_cleared = len(predictor.model_cache.cache)
        predictor.model_cache.clear()
        
        # Clear Redis cache
        redis_keys_cleared = 0
        if predictor.redis_client:
            try:
                redis_keys_cleared = predictor.redis_client.dbsize()
                predictor.redis_client.flushdb()
            except Exception as e:
                logger.warning(f"Could not clear Redis cache: {e}")
        
        return {
            "status": "success",
            "message": "All caches cleared",
            "models_cleared": models_cleared,
            "redis_keys_cleared": redis_keys_cleared
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health/detailed")
async def detailed_health_check(
    predictor: MLPredictorService = Depends(get_ml_predictor)
):
    """
    Perform detailed health check
    """
    try:
        logger.info("Performing detailed health check")
        
        # Basic service health
        health_data = {
            "service": "workflow-ml-prediction",
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0"
        }
        
        # Model health
        try:
            models = await predictor.get_available_models()
            health_data["models"] = {
                "total_count": len(models),
                "cached_count": sum(1 for m in models if m.is_cached),
                "available": [m.model_id for m in models]
            }
        except Exception as e:
            health_data["models"] = {"error": str(e)}
        
        # Cache health
        try:
            cache_stats = predictor.get_cache_stats()
            health_data["cache"] = cache_stats
        except Exception as e:
            health_data["cache"] = {"error": str(e)}
        
        # Disk space check
        try:
            import shutil
            models_dir = predictor.models_dir
            disk_usage = shutil.disk_usage(models_dir)
            health_data["disk"] = {
                "models_directory": str(models_dir),
                "free_space_gb": disk_usage.free / (1024**3),
                "total_space_gb": disk_usage.total / (1024**3)
            }
        except Exception as e:
            health_data["disk"] = {"error": str(e)}
        
        # Test prediction capability
        try:
            # Test with dummy features if any models available
            if models:
                test_model = models[0]
                # This is a basic test - in production you'd use known good features
                health_data["prediction_test"] = {
                    "status": "available",
                    "test_model": test_model.model_id
                }
            else:
                health_data["prediction_test"] = {
                    "status": "no_models_available"
                }
        except Exception as e:
            health_data["prediction_test"] = {"error": str(e)}
        
        return health_data
        
    except Exception as e:
        logger.error(f"Error in detailed health check: {e}")
        return {
            "service": "workflow-ml-prediction",
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": time.time()
        }

# ===============================================
# SMART PREDICTION ENDPOINTS
# ===============================================

# Pydantic models for smart prediction API
from pydantic import BaseModel, Field

class SmartPredictionRequest(BaseModel):
    """Request for smart prediction that auto-selects best model."""
    song_features: Dict[str, Any] = Field(..., description="Song features for prediction")
    model_type: Optional[str] = Field(None, description="Specific model to use (auto-select if None)")
    explain_prediction: bool = Field(default=True, description="Include prediction explanation")

class SmartPredictionResponse(BaseModel):
    """Response from smart prediction."""
    prediction: float = Field(..., description="Predicted value")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    model_used: str = Field(..., description="Model type used for prediction")
    model_performance: Dict[str, float] = Field(..., description="Model performance metrics")
    explanation: Optional[Dict[str, Any]] = Field(None, description="Prediction explanation")
    top_influencing_features: Optional[List[Dict[str, Any]]] = Field(None, description="Top influencing features")

class SmartBatchRequest(BaseModel):
    """Request for smart batch predictions."""
    songs_data: List[Dict[str, Any]] = Field(..., description="List of songs with features")
    explain_predictions: bool = Field(default=False, description="Include explanations for all predictions")

class SmartBatchResponse(BaseModel):
    """Response from smart batch predictions."""
    predictions: List[Dict[str, Any]] = Field(..., description="List of prediction results")
    summary: Dict[str, Any] = Field(..., description="Batch prediction summary")

class SmartModelInfoResponse(BaseModel):
    """Response with smart model information."""
    available_models: List[str] = Field(..., description="List of available model types")
    model_details: Dict[str, Any] = Field(..., description="Detailed model information")
    prediction_strategy: str = Field(..., description="Current prediction strategy")
    last_updated: str = Field(..., description="Last model update timestamp")

class SmartHealthResponse(BaseModel):
    """Smart prediction service health status."""
    status: str = Field(..., description="Service status")
    models_loaded: int = Field(..., description="Number of models loaded")
    available_models: List[str] = Field(..., description="List of available models")
    prediction_capability: str = Field(..., description="Prediction capability status")

# Dependency to get smart predictor
async def get_smart_predictor() -> SmartSongPredictor:
    """Dependency to get smart predictor instance."""
    return smart_predictor

@router.post("/smart/single", response_model=SmartPredictionResponse)
async def smart_predict_single(
    request: SmartPredictionRequest,
    predictor: SmartSongPredictor = Depends(get_smart_predictor)
):
    """
    Make a smart prediction for a single song.
    
    This endpoint automatically selects the best model based on available features
    and provides detailed explanations for the prediction.
    """
    try:
        logger.info(f"Making smart single prediction with {len(request.song_features)} features")
        
        prediction = await predictor.predict_single_song(
            song_features=request.song_features,
            model_type=request.model_type,
            explain_prediction=request.explain_prediction
        )
        
        return SmartPredictionResponse(**prediction)
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in smart single prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making smart prediction: {str(e)}"
        )

@router.post("/smart/batch", response_model=SmartBatchResponse)
async def smart_predict_batch(
    request: SmartBatchRequest,
    predictor: SmartSongPredictor = Depends(get_smart_predictor)
):
    """
    Make smart predictions for multiple songs.
    
    This endpoint processes multiple songs efficiently and provides
    a summary of the batch prediction results.
    """
    try:
        logger.info(f"Making smart batch predictions for {len(request.songs_data)} songs")
        
        predictions = await predictor.predict_batch(request.songs_data)
        
        # Generate summary
        successful_predictions = [p for p in predictions if 'error' not in p]
        failed_predictions = [p for p in predictions if 'error' in p]
        
        models_used = {}
        for pred in successful_predictions:
            model = pred.get('model_used', 'unknown')
            models_used[model] = models_used.get(model, 0) + 1
        
        avg_prediction = sum(p['prediction'] for p in successful_predictions) / len(successful_predictions) if successful_predictions else 0
        avg_confidence = sum(p['confidence'] for p in successful_predictions) / len(successful_predictions) if successful_predictions else 0
        
        summary = {
            'total_songs': len(request.songs_data),
            'successful_predictions': len(successful_predictions),
            'failed_predictions': len(failed_predictions),
            'models_used': models_used,
            'average_prediction': round(avg_prediction, 3),
            'average_confidence': round(avg_confidence, 3),
            'processing_timestamp': time.time()
        }
        
        return SmartBatchResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error in smart batch prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making smart batch predictions: {str(e)}"
        )

@router.get("/smart/models", response_model=SmartModelInfoResponse)
async def get_smart_model_info(
    model_type: Optional[str] = None,
    predictor: SmartSongPredictor = Depends(get_smart_predictor)
):
    """
    Get information about available smart models.
    
    Returns details about model performance, features, and capabilities.
    """
    try:
        logger.info(f"Getting smart model information for: {model_type or 'all models'}")
        
        model_info = await predictor.get_model_info(model_type)
        
        return SmartModelInfoResponse(**model_info)
        
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting smart model info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting smart model information: {str(e)}"
        )

@router.get("/smart/health", response_model=SmartHealthResponse)
async def smart_health_check(
    predictor: SmartSongPredictor = Depends(get_smart_predictor)
):
    """
    Perform health check on the smart prediction service.
    
    Returns status information about smart models, capabilities, and readiness.
    """
    try:
        health_status = await predictor.health_check()
        
        return SmartHealthResponse(
            status=health_status['status'],
            models_loaded=health_status['models_loaded'],
            available_models=health_status['available_models'],
            prediction_capability=health_status['prediction_capability']
        )
        
    except Exception as e:
        logger.error(f"Error in smart health check: {e}")
        return SmartHealthResponse(
            status="unhealthy",
            models_loaded=0,
            available_models=[],
            prediction_capability="error"
        )

@router.post("/smart/update-models")
async def update_smart_models(
    predictor: SmartSongPredictor = Depends(get_smart_predictor)
):
    """
    Force update of smart models from the registry.
    
    This endpoint reloads models from the model registry,
    useful when new models have been trained.
    """
    try:
        logger.info("Updating smart models from registry")
        
        result = await predictor.update_models()
        
        return {
            "message": "Smart models updated successfully",
            "models_loaded": result['models_loaded'],
            "available_models": result['available_models'],
            "updated_at": result['timestamp']
        }
        
    except Exception as e:
        logger.error(f"Error updating smart models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating smart models: {str(e)}"
        )

@router.get("/smart/scheduler/status")
async def get_scheduler_status(
    predictor: SmartSongPredictor = Depends(get_smart_predictor)
):
    """
    Get status of the model registry scheduler.
    
    Returns information about the automatic model registry update schedule.
    """
    try:
        from ..config.settings import settings
        
        return {
            "scheduler_active": True,
            "update_interval_hours": settings.MODEL_REGISTRY_UPDATE_INTERVAL_HOURS,
            "startup_delay_minutes": settings.MODEL_REGISTRY_STARTUP_DELAY_MINUTES,
            "models_currently_loaded": len(predictor.models),
            "model_registry_path": predictor.model_registry_path,
            "last_registry_check": predictor.last_registry_check.isoformat() if predictor.last_registry_check else None,
            "available_models": list(predictor.models.keys()),
            "scheduler_info": {
                "description": "Automatic model registry updates every 24 hours",
                "next_check": "Calculated from last_registry_check + interval",
                "manual_trigger": "Use POST /predict/smart/update-models to trigger immediate update"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting scheduler status: {str(e)}"
        )

@router.get("/smart/features/{model_type}")
async def get_smart_model_features(
    model_type: str,
    predictor: SmartSongPredictor = Depends(get_smart_predictor)
):
    """
    Get required features for a specific smart model.
    
    Returns information about what features are needed
    to make predictions with the specified model.
    """
    try:
        model_info = await predictor.get_model_info(model_type)
        
        features_info = model_info['features']
        
        return {
            "model_type": model_type,
            "required_features": features_info['all_features'],
            "audio_features": features_info['audio_features'],
            "lyrics_features": features_info['lyrics_features'],
            "total_features": len(features_info['all_features']),
            "feature_descriptions": {
                feature: predictor._get_feature_description(feature)
                for feature in features_info['all_features'][:10]  # Limit for response size
            }
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting smart model features: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting smart model features: {str(e)}"
        )

@router.post("/smart/validate-features")
async def validate_smart_features(
    song_features: Dict[str, Any],
    model_type: Optional[str] = None,
    predictor: SmartSongPredictor = Depends(get_smart_predictor)
):
    """
    Validate song features for smart prediction.
    
    Checks if the provided features are sufficient for making
    predictions with the specified model (or any available model).
    This includes calculating derived features as needed.
    """
    try:
        await predictor.load_models()
        
        if not predictor.models:
            raise HTTPException(
                status_code=503,
                detail="No smart models available for validation"
            )
        
        validation_results = {}
        
        if model_type:
            # Calculate derived features for this specific model
            enhanced_features = predictor._calculate_model_specific_derived_features(song_features, model_type)
            
            # Validate enhanced features (with derived features calculated)
            is_valid, missing = predictor.validate_features(enhanced_features, model_type)
            validation_results[model_type] = {
                "valid": is_valid,
                "missing_features": missing
            }
        else:
            # Validate for all models (calculate derived features for each)
            for available_model in predictor.get_available_models():
                # Calculate derived features for this specific model
                enhanced_features = predictor._calculate_model_specific_derived_features(song_features, available_model)
                
                # Validate enhanced features
                is_valid, missing = predictor.validate_features(enhanced_features, available_model)
                validation_results[available_model] = {
                    "valid": is_valid,
                    "missing_features": missing
                }
        
        # Find best model that can be used
        usable_models = [model for model, result in validation_results.items() if result['valid']]
        recommended_model = None
        
        if usable_models:
            try:
                # Use enhanced features with derived features for model selection
                sample_enhanced = predictor._calculate_model_specific_derived_features(song_features, usable_models[0])
                recommended_model = predictor.select_best_model(sample_enhanced)
            except:
                recommended_model = usable_models[0]
        
        return {
            "provided_features": list(song_features.keys()),
            "feature_count": len(song_features),
            "validation_results": validation_results,
            "usable_models": usable_models,
            "recommended_model": recommended_model,
            "can_predict": len(usable_models) > 0
        }
        
    except Exception as e:
        logger.error(f"Error validating smart features: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error validating smart features: {str(e)}"
        )

@router.get("/smart/registry")
async def get_smart_model_registry(
    predictor: SmartSongPredictor = Depends(get_smart_predictor)
):
    """
    Get the current smart model registry information.
    
    Returns raw registry data with all model metadata
    and training information.
    """
    try:
        await predictor.load_models()
        
        return {
            "registry_path": predictor.model_registry_path,
            "registry_data": predictor.registry,
            "models_loaded": len(predictor.models),
            "last_check": predictor.last_registry_check,
            "available_models": predictor.get_available_models()
        }
        
    except Exception as e:
        logger.error(f"Error getting smart registry: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting smart model registry: {str(e)}"
        )

@router.post("/smart/demo/sample-prediction")
async def demo_smart_prediction(
    predictor: SmartSongPredictor = Depends(get_smart_predictor)
):
    """
    Demo endpoint that makes a smart prediction with sample data.
    
    Useful for testing the smart prediction service with realistic sample features.
    """
    try:
        # Generate sample song features
        sample_features = {
            'tempo': 128.0,
            'energy': 0.8,
            'danceability': 0.9,
            'valence': 0.7,
            'loudness': -5.0,
            'speechiness': 0.1,
            'acousticness': 0.2,
            'instrumentalness': 0.05,
            'liveness': 0.3,
            'genre_confidence': 0.85,
            'lyrics_word_count': 120,
            'lyrics_sentiment_score': 0.6,
            'lyrics_complexity_score': 0.5,
            'lyrics_theme_diversity': 0.7
        }
        
        prediction = await predictor.predict_single_song(
            song_features=sample_features,
            explain_prediction=True
        )
        
        return {
            "message": "Demo smart prediction with sample data",
            "sample_features": sample_features,
            "prediction_result": prediction
        }
        
    except Exception as e:
        logger.error(f"Error in demo smart prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in demo smart prediction: {str(e)}"
        ) 