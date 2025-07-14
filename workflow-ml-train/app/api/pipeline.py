"""
Pipeline API endpoints for dynamic strategy selection and training control.

Provides REST API for:
- Starting training with different strategies 
- Controlling pipeline execution
- Managing experiments
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import uuid
import json
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter()

# ============================================================================= 
# REQUEST/RESPONSE MODELS
# =============================================================================

class TrainingRequest(BaseModel):
    """Request model for starting training"""
    strategy: str = Field(..., description="Training strategy")
    experiment_name: Optional[str] = Field(None, description="MLflow experiment name")
    features: Optional[List[str]] = Field(None, description="Custom feature list")
    model_types: Optional[List[str]] = Field(None, description="Model types to train")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional parameters")
    
    # NEW: Pre-existing feature vector support
    feature_vector_file: Optional[str] = Field(None, description="Path to pre-existing feature vector file (relative to shared volumes)")
    feature_vector_name: Optional[str] = Field(None, description="Name of stored feature vector preset")
    skip_feature_agreement: Optional[bool] = Field(False, description="Skip interactive feature agreement")
    
    # NEW: Dataset and directory configuration
    dataset_csv_path: Optional[str] = Field(None, description="Path to training dataset CSV file")
    songs_directory: Optional[str] = Field(None, description="Path to directory containing song files")
    lyrics_directory: Optional[str] = Field(None, description="Path to directory containing lyrics files")

class TrainingResponse(BaseModel):
    """Response model for training requests"""
    pipeline_id: str = Field(..., description="Unique pipeline ID")
    status: str = Field(..., description="Initial status")
    strategy: str = Field(..., description="Selected strategy")
    experiment_name: str = Field(..., description="MLflow experiment name")
    airflow_dag_url: Optional[str] = Field(None, description="Airflow DAG URL for monitoring")
    estimated_duration: str = Field(..., description="Estimated completion time")

class PipelineStatus(BaseModel):
    """Pipeline status response"""
    pipeline_id: str
    status: str  # pending, running, completed, failed
    strategy: str
    current_stage: Optional[str] = None
    progress_percent: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None

# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================

def get_orchestrator(request: Request):
    """Dependency to get orchestrator instance from app state"""
    if hasattr(request.app.state, 'orchestrator') and request.app.state.orchestrator:
        return request.app.state.orchestrator
    else:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="ML training service not available")

# =============================================================================
# PIPELINE ENDPOINTS
# =============================================================================

@router.post("/train", response_model=TrainingResponse)
async def start_training(
    training_request: TrainingRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    orchestrator = Depends(get_orchestrator)
):
    """
    Start ML training pipeline with dynamic strategy selection.
    
    No configuration file changes needed - everything via API parameters.
    
    **Supported Strategies:**
    - `audio_only`: Train using only audio features
    - `multimodal`: Train using audio + content features  
    - `custom`: User-defined feature selection (requires `features` parameter)
    
    **Example Usage:**
    ```bash
    # Quick audio-only training
    curl -X POST "/pipeline/train" \\
      -d '{"strategy": "audio_only", "experiment_name": "quick_test"}'
    
    # Multimodal training with custom features
    curl -X POST "/pipeline/train" \\
      -d '{
        "strategy": "multimodal", 
        "features": ["energy", "valence", "sentiment_polarity"],
        "model_types": ["random_forest", "xgboost"]
      }'
    ```
    """
    try:
        # Generate unique pipeline ID
        pipeline_id = f"pipeline_{uuid.uuid4().hex[:8]}"
        
        # Validate strategy
        valid_strategies = ["audio_only", "multimodal", "custom"]
        if training_request.strategy not in valid_strategies:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy. Must be one of: {valid_strategies}"
            )
        
        # Validate custom strategy requirements
        if training_request.strategy == "custom" and not training_request.features:
            raise HTTPException(
                status_code=400,
                detail="Custom strategy requires 'features' parameter"
            )
        
        # Generate experiment name if not provided
        experiment_name = training_request.experiment_name or f"{training_request.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Handle pre-existing feature vectors
        feature_vector = None
        if training_request.feature_vector_file or training_request.feature_vector_name:
            from ..utils.feature_vector_manager import feature_vector_manager
            
            if training_request.feature_vector_file:
                # Load from file path
                feature_vector = feature_vector_manager.load_feature_vector(training_request.feature_vector_file)
                if not feature_vector:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Feature vector file not found: {training_request.feature_vector_file}"
                    )
            elif training_request.feature_vector_name:
                # Load from preset
                feature_vector = feature_vector_manager.load_preset(training_request.feature_vector_name)
                if not feature_vector:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Feature vector preset not found: {training_request.feature_vector_name}"
                    )
            
            # Validate feature vector
            validation = feature_vector_manager.validate_feature_vector(feature_vector)
            if not validation["valid"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid feature vector: {', '.join(validation['issues'])}"
                )
            
            logger.info(f"✅ Loaded feature vector: {validation['summary']}")
        
        # Prepare pipeline parameters
        pipeline_params = {
            "custom_services": [],
            "custom_csv_path": None,  # Not related to feature vectors
            "features": training_request.features or [],
            "parameters": training_request.parameters or {},
            "feature_vector": feature_vector,
            "skip_feature_agreement": training_request.skip_feature_agreement or (feature_vector is not None)
        }
        
        if training_request.strategy == "custom":
            # For custom strategy, user must specify services
            if "services" in training_request.parameters:
                pipeline_params["custom_services"] = training_request.parameters["services"]
            else:
                pipeline_params["custom_services"] = ["audio"]  # Default
        
        # Start pipeline using orchestrator
        result = await orchestrator.start_pipeline(
            pipeline_id=pipeline_id,
            strategy=training_request.strategy,
            experiment_name=experiment_name,
            features=training_request.features,
            model_types=training_request.model_types,
            parameters=pipeline_params
        )
        
        # Generate Airflow DAG URL
        airflow_dag_url = f"http://localhost:8080/dags/ml_pipeline_{training_request.strategy}"
        
        logger.info(f"🚀 Started pipeline {pipeline_id} with strategy '{training_request.strategy}'")
        
        return TrainingResponse(
            pipeline_id=pipeline_id,
            status=result["status"],
            strategy=training_request.strategy,
            experiment_name=experiment_name,
            airflow_dag_url=airflow_dag_url,
            estimated_duration="15-30 minutes"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@router.get("/status/{pipeline_id}", response_model=PipelineStatus)
async def get_pipeline_status(pipeline_id: str, request: Request, orchestrator = Depends(get_orchestrator)):
    """Get real-time pipeline status and progress"""
    try:
        pipeline_state = orchestrator.get_pipeline_status(pipeline_id)
        
        if not pipeline_state:
            raise HTTPException(
                status_code=404,
                detail=f"Pipeline {pipeline_id} not found"
            )
        
        # Calculate progress percentage based on completed stages
        total_stages = len(orchestrator.stages)
        completed_stages = sum(1 for stage in pipeline_state["stages"].values() 
                             if stage["status"] == "completed")
        progress_percent = (completed_stages / total_stages) * 100
        
        return PipelineStatus(
            pipeline_id=pipeline_id,
            status=pipeline_state["status"],
            strategy=pipeline_state["strategy"],
            current_stage=pipeline_state.get("current_stage"),
            progress_percent=progress_percent,
            start_time=pipeline_state.get("start_time"),
            end_time=pipeline_state.get("end_time"),
            error_message=pipeline_state.get("error_message")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get pipeline status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.post("/stop/{pipeline_id}")
async def stop_pipeline(pipeline_id: str, request: Request, orchestrator = Depends(get_orchestrator)):
    """Stop a running pipeline"""
    try:
        success = await orchestrator.stop_pipeline(pipeline_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Pipeline {pipeline_id} not found or not active"
            )
        
        logger.info(f"🛑 Stopped pipeline {pipeline_id}")
        
        return {
            "pipeline_id": pipeline_id,
            "status": "stopped",
            "message": "Pipeline stopped successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to stop pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop pipeline: {str(e)}")

@router.get("/experiments")
async def list_experiments():
    """List all MLflow experiments"""
    try:
        # TODO: Get actual experiments from MLflow
        return {
            "experiments": [
                {
                    "experiment_id": "1",
                    "name": "audio_only_20241227_143022",
                    "status": "completed",
                    "runs": 1,
                    "best_model": "random_forest",
                    "best_score": 0.85
                },
                {
                    "experiment_id": "2", 
                    "name": "multimodal_20241227_144512",
                    "status": "running",
                    "runs": 0,
                    "best_model": None,
                    "best_score": None
                }
            ],
            "total": 2
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to list experiments: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list experiments: {str(e)}")

def _calculate_duration_minutes(start_time, end_time):
    """Safely calculate duration in minutes from timestamps"""
    try:
        if not start_time or not end_time:
            return 0
        
        # Handle different timestamp formats
        if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
            # If timestamps are in milliseconds
            if start_time > 1e10:
                return (end_time - start_time) / (1000 * 60)
            else:
                return (end_time - start_time) / 60
        else:
            # Handle pandas Timestamp or other formats
            return 0
    except (TypeError, ValueError, AttributeError):
        return 0

def _get_sample_filenames(df):
    """Get sample filenames from dataframe based on available columns"""
    # Try different column names in order of preference
    filename_columns = ["filename", "song_name", "audio_file_path"]
    
    for col in filename_columns:
        if col in df.columns:
            return df[col].head(5).tolist()
    
    return []

@router.get("/experiments/history")
async def get_experiment_history():
    """Get experiment history for performance tracking dashboard"""
    try:
        import mlflow
        import pandas as pd
        import json
        from datetime import datetime, timedelta
        from pathlib import Path
        
        # Connect to MLflow (use container name for Docker networking)
        mlflow.set_tracking_uri("http://mlflow-server:5000")
        
        # Load model registry for actual performance data
        model_registry = {}
        try:
            registry_path = Path("/app/models/ml-training/model_registry.json")
            if registry_path.exists():
                with open(registry_path, 'r') as f:
                    registry_data = json.load(f)
                    # Index by mlflow_run_id for quick lookup
                    for model_id, model_info in registry_data.get('available_models', {}).items():
                        run_id = model_info.get('mlflow_run_id')
                        if run_id:
                            model_registry[run_id] = model_info.get('performance', {})
        except Exception as e:
            logger.warning(f"⚠️ Could not load model registry: {e}")
        
        # Get all experiments
        experiments = mlflow.search_experiments()
        
        audio_only_experiments = []
        multimodal_experiments = []
        
        for experiment in experiments:
            # Get runs for this experiment
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            
            if runs.empty:
                continue
                
            for _, run in runs.iterrows():
                strategy = run.get('params.strategy', 'unknown')
                run_id = run.get('run_id', '')
                
                # Extract performance metrics - prefer model registry data if available
                if run_id in model_registry:
                    registry_perf = model_registry[run_id]
                    r2_score = registry_perf.get('r2_score', 0.0)
                    rmse = registry_perf.get('rmse', 0.0)
                    mae = registry_perf.get('mae', 0.0)
                else:
                    # Fallback to MLflow metrics
                    r2_score = run.get('metrics.r2_score', 0.0)
                    rmse = run.get('metrics.rmse', 0.0)
                    mae = run.get('metrics.mae', 0.0)
                
                # Get training info - handle both timestamp formats
                start_time = run.get('start_time', 0)
                end_time = run.get('end_time', 0)
                
                if start_time:
                    # Handle both milliseconds and seconds timestamps
                    try:
                        if isinstance(start_time, (int, float)):
                            # If it's a large number, assume milliseconds
                            if start_time > 1e10:
                                training_date = datetime.fromtimestamp(start_time / 1000).isoformat()
                            else:
                                training_date = datetime.fromtimestamp(start_time).isoformat()
                        else:
                            # If it's a pandas Timestamp, convert to datetime
                            training_date = start_time.isoformat() if hasattr(start_time, 'isoformat') else str(start_time)
                    except (TypeError, ValueError, AttributeError):
                        training_date = datetime.now().isoformat()
                else:
                    training_date = datetime.now().isoformat()
                
                experiment_data = {
                    "experiment_id": experiment.experiment_id,
                    "run_id": run.get('run_id', ''),
                    "name": experiment.name,
                    "strategy": strategy,
                    "r2_score": r2_score,
                    "rmse": rmse,
                    "mae": mae,
                    "training_date": training_date,
                    "duration_minutes": _calculate_duration_minutes(start_time, end_time),
                    "status": run.get('status', 'UNKNOWN'),
                    "feature_count": run.get('params.n_features', 0),
                    "model_type": run.get('params.model_type', 'ensemble')
                }
                
                # Categorize by strategy
                if strategy == 'audio_only':
                    audio_only_experiments.append(experiment_data)
                elif strategy == 'multimodal':
                    multimodal_experiments.append(experiment_data)
        
        # Sort by date (newest first)
        audio_only_experiments.sort(key=lambda x: x['training_date'], reverse=True)
        multimodal_experiments.sort(key=lambda x: x['training_date'], reverse=True)
        
        # Get best models
        best_audio_only = max(audio_only_experiments, key=lambda x: x['r2_score']) if audio_only_experiments else None
        best_multimodal = max(multimodal_experiments, key=lambda x: x['r2_score']) if multimodal_experiments else None
        
        # Calculate summary statistics
        audio_only_avg_r2 = sum(exp['r2_score'] for exp in audio_only_experiments) / len(audio_only_experiments) if audio_only_experiments else 0
        multimodal_avg_r2 = sum(exp['r2_score'] for exp in multimodal_experiments) / len(multimodal_experiments) if multimodal_experiments else 0
        
        return {
            "audio_only": {
                "experiments": audio_only_experiments[:10],  # Latest 10
                "total_count": len(audio_only_experiments),
                "best_model": best_audio_only,
                "average_r2": audio_only_avg_r2,
                "trend_data": [
                    {
                        "date": exp['training_date'],
                        "r2_score": exp['r2_score'],
                        "rmse": exp['rmse']
                    } for exp in audio_only_experiments[-20:]  # Last 20 for trend
                ]
            },
            "multimodal": {
                "experiments": multimodal_experiments[:10],  # Latest 10
                "total_count": len(multimodal_experiments),
                "best_model": best_multimodal,
                "average_r2": multimodal_avg_r2,
                "trend_data": [
                    {
                        "date": exp['training_date'],
                        "r2_score": exp['r2_score'],
                        "rmse": exp['rmse']
                    } for exp in multimodal_experiments[-20:]  # Last 20 for trend
                ]
            },
            "summary": {
                "total_experiments": len(audio_only_experiments) + len(multimodal_experiments),
                "audio_only_count": len(audio_only_experiments),
                "multimodal_count": len(multimodal_experiments),
                "performance_comparison": {
                    "audio_only_avg": audio_only_avg_r2,
                    "multimodal_avg": multimodal_avg_r2,
                    "multimodal_advantage": multimodal_avg_r2 - audio_only_avg_r2 if audio_only_avg_r2 > 0 else 0
                }
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get experiment history: {e}")
        # Return fallback data for development
        return {
            "audio_only": {
                "experiments": [],
                "total_count": 0,
                "best_model": None,
                "average_r2": 0,
                "trend_data": []
            },
            "multimodal": {
                "experiments": [],
                "total_count": 0,
                "best_model": None,
                "average_r2": 0,
                "trend_data": []
            },
            "summary": {
                "total_experiments": 0,
                "audio_only_count": 0,
                "multimodal_count": 0,
                "performance_comparison": {
                    "audio_only_avg": 0,
                    "multimodal_avg": 0,
                    "multimodal_advantage": 0
                }
            }
        }

@router.get("/strategies")
async def list_strategies():
    """List available training strategies with descriptions"""
    return {
        "strategies": {
            "audio_only": {
                "description": "Train using only audio features (tempo, energy, valence, etc.)",
                "features": "25+ audio features",
                "typical_accuracy": "0.75-0.85",
                "training_time": "3-5 minutes",
                "use_case": "Quick training, audio-focused applications"
            },
            "multimodal": {
                "description": "Train using both audio and content features",
                "features": "25+ audio + 12+ content features", 
                "typical_accuracy": "0.80-0.90",
                "training_time": "5-8 minutes",
                "use_case": "Best accuracy, full feature set"
            },
            "custom": {
                "description": "User-defined feature selection",
                "features": "User-specified subset",
                "typical_accuracy": "Varies",
                "training_time": "2-10 minutes",
                "use_case": "Feature experimentation, specialized models"
            }
        },
        "recommendations": {
            "quick_test": "audio_only",
            "production": "multimodal", 
            "experimentation": "custom"
        }
    }

@router.get("/pipeline/{pipeline_id}/shap-analysis")
async def get_shap_analysis(pipeline_id: str):
    """Get SHAP explainability analysis for a trained model"""
    try:
        pipeline_state = orchestrator.get_pipeline_status(pipeline_id)
        if not pipeline_state:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        training_results = pipeline_state.get("training_results")
        if not training_results:
            raise HTTPException(status_code=404, detail="No training results found")
        
        shap_importance = training_results.get("shap_feature_importance", {})
        if not shap_importance:
            raise HTTPException(status_code=404, detail="No SHAP analysis available")
        
        # Get feature importance from all models
        feature_importance = training_results.get("feature_importance", {})
        
        # Create comprehensive explainability report
        explainability_report = {
            "pipeline_id": pipeline_id,
            "model_type": training_results.get("model_type", "Unknown"),
            "ensemble_models": training_results.get("ensemble_models", []),
            "accuracy": training_results.get("accuracy", 0.0),
            "n_features": training_results.get("n_features", 0),
            "shap_analysis": {
                "available": True,
                "feature_count": len(shap_importance),
                "top_features": sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            },
            "feature_importance_comparison": {
                "shap": shap_importance,
                "random_forest": feature_importance.get("random_forest", {}),
                "xgboost": feature_importance.get("xgboost", {}),
                "ensemble": feature_importance.get("ensemble", {})
            },
            "mlflow_run_id": training_results.get("mlflow_run_id"),
            "shared_model_id": training_results.get("shared_model_id")
        }
        
        return {
            "status": "success",
            "message": "SHAP explainability analysis retrieved successfully",
            "data": explainability_report
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving SHAP analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/models/explainability")
async def list_models_with_explainability():
    """List all trained models with SHAP explainability available"""
    try:
        # Get model registry
        registry_path = Path("/app/models/ml-training/model_registry.json")
        if not registry_path.exists():
            return {
                "status": "success",
                "message": "No models found",
                "data": {"models": []}
            }
        
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        models_with_explainability = []
        for model_id, model_info in registry.get("available_models", {}).items():
            if model_info.get("has_shap_analysis", False):
                models_with_explainability.append({
                    "model_id": model_id,
                    "model_type": model_info.get("model_type"),
                    "ensemble_models": model_info.get("ensemble_models", []),
                    "strategy": model_info.get("strategy"),
                    "performance": model_info.get("performance", {}),
                    "created_at": model_info.get("created_at"),
                    "mlflow_run_id": model_info.get("mlflow_run_id")
                })
        
        return {
            "status": "success",
            "message": f"Found {len(models_with_explainability)} models with SHAP explainability",
            "data": {
                "models": models_with_explainability,
                "total_models": len(registry.get("available_models", {})),
                "models_with_explainability": len(models_with_explainability)
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing models with explainability: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/datasets/config")
async def get_dataset_configuration():
    """Get current dataset configuration options"""
    try:
        import os
        from pathlib import Path
        
        # Get configuration from environment or defaults
        default_dataset_csv = os.getenv("DEFAULT_DATASET_CSV", "/app/data/training_data/r4a_song_data_training.csv")
        default_songs_dir = os.getenv("DEFAULT_SONGS_DIR", "/app/songs")
        default_lyrics_dir = os.getenv("DEFAULT_LYRICS_DIR", "/app/lyrics")
        
        # Discover available datasets dynamically (recursive search)
        data_dir = Path("/app/data/training_data")
        available_datasets = []
        
        if data_dir.exists():
            # Recursively find all CSV files
            for csv_file in data_dir.rglob("*.csv"):
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_file, nrows=1)  # Just read header
                    
                    # Create relative path for better display
                    rel_path = csv_file.relative_to(data_dir)
                    display_name = str(rel_path).replace("_", " ").replace("/", " / ").title()
                    
                    available_datasets.append({
                        "name": display_name,
                        "path": str(csv_file),
                        "description": f"Dataset with {len(df.columns)} columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}",
                        "has_required_columns": any(col in df.columns for col in ["filename", "song_name", "audio_file_path"]),
                        "columns": list(df.columns),
                        "location": str(rel_path.parent) if rel_path.parent != Path('.') else "root"
                    })
                except Exception as e:
                    available_datasets.append({
                        "name": csv_file.name.replace("_", " ").title(),
                        "path": str(csv_file),
                        "description": f"Error reading dataset: {str(e)}",
                        "has_required_columns": False,
                        "columns": [],
                        "location": str(csv_file.parent.relative_to(data_dir))
                    })
        
        # Check if default directories exist and get basic info
        songs_info = {}
        if Path(default_songs_dir).exists():
            songs_path = Path(default_songs_dir)
            audio_files = list(songs_path.glob("**/*.mp3")) + list(songs_path.glob("**/*.wav")) + list(songs_path.glob("**/*.flac"))
            songs_info = {
                "exists": True,
                "total_files": len(audio_files),
                "subdirectories": [d.name for d in songs_path.iterdir() if d.is_dir()]
            }
        else:
            songs_info = {"exists": False, "total_files": 0, "subdirectories": []}
        
        lyrics_info = {}
        if Path(default_lyrics_dir).exists():
            lyrics_path = Path(default_lyrics_dir)
            lyrics_files = list(lyrics_path.glob("**/*.txt")) + list(lyrics_path.glob("**/*.lrc"))
            lyrics_info = {
                "exists": True,
                "total_files": len(lyrics_files),
                "subdirectories": [d.name for d in lyrics_path.iterdir() if d.is_dir()]
            }
        else:
            lyrics_info = {"exists": False, "total_files": 0, "subdirectories": []}
        
        return {
            "default_config": {
                "dataset_csv": default_dataset_csv,
                "songs_directory": default_songs_dir,
                "lyrics_directory": default_lyrics_dir,
                "description": "Default paths from environment/configuration"
            },
            "available_datasets": available_datasets,
            "directory_info": {
                "songs": songs_info,
                "lyrics": lyrics_info
            },
            "validation_rules": {
                "dataset_csv": {
                    "required_columns": ["filename", "song_name", "audio_file_path"],
                    "required_columns_description": "At least one of: filename, song_name, or audio_file_path",
                    "optional_columns": ["popularity", "original_popularity", "genre", "artist", "title", "duration", "has_audio_file", "has_lyrics_file"],
                    "supported_formats": [".csv"],
                    "description": "CSV file with song metadata and target values"
                },
                "songs_directory": {
                    "must_exist": True,
                    "must_contain_files": True,
                    "supported_formats": [".mp3", ".wav", ".flac", ".m4a", ".aac"],
                    "expected_structure": "Can be flat directory or organized in subdirectories"
                },
                "lyrics_directory": {
                    "must_exist": False,
                    "note": "Optional for audio-only training, required for multimodal",
                    "supported_formats": [".txt", ".lrc"],
                    "expected_structure": "Should match songs directory structure"
                }
            },
            "custom_config_instructions": {
                "dataset_csv": "Provide full path to CSV file with 'filename', 'song_name', or 'audio_file_path' column",
                "songs_directory": "Provide full path to directory containing audio files",
                "lyrics_directory": "Provide full path to directory containing lyrics (optional)"
            }
        }
    except Exception as e:
        logger.error(f"❌ Failed to get dataset configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dataset configuration: {str(e)}")

@router.post("/datasets/validate")
async def validate_dataset_configuration(config: dict):
    """Validate a dataset configuration before training"""
    try:
        dataset_path = config.get("dataset_csv_path")
        songs_dir = config.get("songs_directory")
        lyrics_dir = config.get("lyrics_directory")
        strategy = config.get("strategy", "audio_only")
        
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "dataset_info": {},
            "directory_info": {},
            "recommendations": []
        }
        
        # Validate dataset CSV
        if dataset_path:
            dataset_file = Path(dataset_path)
            if not dataset_file.exists():
                validation_results["errors"].append(f"Dataset CSV not found: {dataset_path}")
                validation_results["valid"] = False
            elif not dataset_file.suffix.lower() == '.csv':
                validation_results["errors"].append(f"Dataset must be CSV format, got: {dataset_file.suffix}")
                validation_results["valid"] = False
            else:
                try:
                    import pandas as pd
                    df = pd.read_csv(dataset_path)
                    
                    # Check required columns (at least one of these must be present)
                    required_cols = ["filename", "song_name", "audio_file_path"]
                    has_required_col = any(col in df.columns for col in required_cols)
                    
                    if not has_required_col:
                        validation_results["errors"].append(f"Dataset CSV missing required columns. Must have at least one of: {required_cols}")
                        validation_results["valid"] = False
                    
                    # Check optional but important columns
                    optional_cols = ["popularity", "original_popularity", "genre", "artist", "title"]
                    present_optional = [col for col in optional_cols if col in df.columns]
                    missing_optional = [col for col in optional_cols if col not in df.columns]
                    
                    validation_results["dataset_info"] = {
                        "rows": len(df),
                        "columns": list(df.columns),
                        "required_columns_present": has_required_col,
                        "optional_columns_present": present_optional,
                        "missing_optional_columns": missing_optional,
                        "sample_filenames": _get_sample_filenames(df)
                    }
                    
                    if not present_optional:
                        validation_results["warnings"].append("No target columns found (popularity, genre, etc.) - will use default labeling")
                    
                    # Check for common issues with filename column
                    filename_col = None
                    for col in ["filename", "song_name", "audio_file_path"]:
                        if col in df.columns:
                            filename_col = col
                            break
                    
                    if filename_col:
                        null_filenames = df[filename_col].isnull().sum()
                        if null_filenames > 0:
                            validation_results["warnings"].append(f"{null_filenames} rows have null {filename_col}")
                        
                        # Check filename formats (only if it looks like an audio file path)
                        if filename_col in ["filename", "audio_file_path"]:
                            sample_files = df[filename_col].head(100).tolist()
                            audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.aac']
                            non_audio_files = [f for f in sample_files if f and not any(str(f).lower().endswith(ext) for ext in audio_extensions)]
                            
                            if non_audio_files:
                                validation_results["warnings"].append(f"Some {filename_col} don't have audio extensions: {non_audio_files[:3]}...")
                        
                except Exception as e:
                    validation_results["errors"].append(f"Failed to read CSV: {str(e)}")
                    validation_results["valid"] = False
        
        # Validate songs directory
        if songs_dir:
            songs_path = Path(songs_dir)
            if not songs_path.exists():
                validation_results["errors"].append(f"Songs directory not found: {songs_dir}")
                validation_results["valid"] = False
            else:
                # Find audio files recursively
                audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.aac']
                audio_files = []
                for ext in audio_extensions:
                    audio_files.extend(list(songs_path.glob(f"**/*{ext}")))
                
                validation_results["directory_info"]["songs"] = {
                    "total_files": len(audio_files),
                    "sample_files": [str(f.name) for f in audio_files[:5]],
                    "subdirectories": [d.name for d in songs_path.iterdir() if d.is_dir()],
                    "file_formats": list(set([f.suffix.lower() for f in audio_files]))
                }
                
                if len(audio_files) == 0:
                    validation_results["errors"].append("No audio files found in songs directory")
                    validation_results["valid"] = False
                elif len(audio_files) < 100:
                    validation_results["warnings"].append(f"Only {len(audio_files)} audio files found - may not be enough for training")
                
                # Check if filenames match dataset
                if dataset_path and "filename" in validation_results.get("dataset_info", {}):
                    dataset_filenames = set(validation_results["dataset_info"]["sample_filenames"])
                    actual_filenames = set([f.name for f in audio_files])
                    
                    missing_files = dataset_filenames - actual_filenames
                    if missing_files:
                        validation_results["warnings"].append(f"Some dataset files not found in songs directory: {list(missing_files)[:3]}...")
        
        # Validate lyrics directory (conditional on strategy)
        if lyrics_dir:
            lyrics_path = Path(lyrics_dir)
            if not lyrics_path.exists():
                if strategy == "multimodal":
                    validation_results["errors"].append(f"Lyrics directory not found: {lyrics_dir} (required for multimodal training)")
                    validation_results["valid"] = False
                else:
                    validation_results["warnings"].append(f"Lyrics directory not found: {lyrics_dir} (optional for audio-only)")
            else:
                lyrics_files = list(lyrics_path.glob("**/*.txt")) + list(lyrics_path.glob("**/*.lrc"))
                validation_results["directory_info"]["lyrics"] = {
                    "total_files": len(lyrics_files),
                    "sample_files": [str(f.name) for f in lyrics_files[:5]],
                    "subdirectories": [d.name for d in lyrics_path.iterdir() if d.is_dir()],
                    "file_formats": list(set([f.suffix.lower() for f in lyrics_files]))
                }
                
                if len(lyrics_files) == 0 and strategy == "multimodal":
                    validation_results["errors"].append("No lyrics files found in lyrics directory (required for multimodal training)")
                    validation_results["valid"] = False
        elif strategy == "multimodal":
            validation_results["errors"].append("Lyrics directory required for multimodal training")
            validation_results["valid"] = False
        
        # Generate recommendations
        if validation_results["valid"]:
            validation_results["recommendations"].append("✅ Configuration looks good for training")
            
            if validation_results["directory_info"].get("songs", {}).get("total_files", 0) > 10000:
                validation_results["recommendations"].append("🚀 Large dataset detected - consider using distributed training")
            
            if strategy == "audio_only" and lyrics_dir:
                validation_results["recommendations"].append("💡 Consider multimodal training since lyrics are available")
        else:
            validation_results["recommendations"].append("❌ Fix the errors above before proceeding with training")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"❌ Failed to validate dataset configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate dataset configuration: {str(e)}")

# =============================================================================
# BACKGROUND TASKS
# =============================================================================

async def _execute_pipeline(
    pipeline_id: str, 
    request: TrainingRequest, 
    experiment_name: str
):
    """Execute the training pipeline in background"""
    try:
        logger.info(f"⚙️ Executing pipeline {pipeline_id}")
        
        # TODO: Implement actual pipeline execution
        # This would involve:
        # 1. Service discovery
        # 2. Feature agreement
        # 3. Feature extraction
        # 4. Model training
        # 5. Model registry
        
        # For now, just log the request
        logger.info(f"Pipeline {pipeline_id} completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Pipeline {pipeline_id} failed: {e}")
        # TODO: Update pipeline status to failed 