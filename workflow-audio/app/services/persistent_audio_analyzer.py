"""
Persistent Audio Analyzer Service
Wraps existing audio analysis with database persistence, idempotency, and audit trails
This solves the critical data loss issue identified in the system analysis
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib
import json

from .audio_analyzer import AudioAnalyzer
from .comprehensive_analyzer import ComprehensiveAudioAnalyzer
from .database_service import db_service
from ..models.database_models import AnalysisStatus, AnalysisType
from ..config.settings import get_settings

logger = logging.getLogger(__name__)

class PersistentAudioAnalyzer:
    """
    Audio analyzer with full persistence and data lineage tracking
    Fixes the critical data loss issue where analysis results were lost on service restart
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.basic_analyzer = AudioAnalyzer()
        self.comprehensive_analyzer = ComprehensiveAudioAnalyzer()
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connections"""
        if not self._initialized:
            await db_service.initialize()
            self._initialized = True
            logger.info("✅ Persistent Audio Analyzer initialized")
    
    async def analyze_with_persistence(
        self,
        file_path: str,
        analysis_type: str = "basic",
        idempotency_key: Optional[str] = None,
        workflow_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        requested_by: Optional[str] = None,
        file_id: Optional[str] = None,
        original_filename: Optional[str] = None,
        force_reanalysis: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform audio analysis with full persistence and idempotency
        
        This is the main method that fixes the data loss issue by:
        1. Checking for existing analysis (idempotency) - SERVER-SIDE DEDUPLICATION
        2. Generating deterministic analysis IDs
        3. Tracking analysis state throughout the process
        4. Persisting results to database with complete metadata
        5. Enabling data reuse and preventing duplicate processing
        
        Args:
            file_path: Path to the audio file
            analysis_type: Type of analysis ("basic" or "comprehensive")
            idempotency_key: Optional key for request deduplication
            workflow_id: ID of the workflow this analysis belongs to
            correlation_id: ID for tracking related operations
            requested_by: Identifier of the requesting service/user
            file_id: Optional file identifier
            original_filename: Original filename (important for deduplication)
            force_reanalysis: Whether to force reanalysis even if existing analysis found
            **kwargs: Additional analysis parameters
            
        Returns:
            Dictionary containing analysis results and metadata
        """
        try:
            # Extract filename for deduplication
            filename_for_dedup = original_filename or Path(file_path).name
            
            # STEP 1: ENHANCED SERVER-SIDE DEDUPLICATION (unless forcing reanalysis)
            if not force_reanalysis:
                try:
                    # First try content-based deduplication (most reliable)
                    file_hash = self._calculate_file_hash(file_path)
                    existing_analysis = await db_service.get_analysis_by_file_hash(file_hash)
                    
                    # If content-based lookup fails, fall back to filename-based lookup
                    if not existing_analysis:
                        logger.info(f"🔍 Content-based lookup failed, trying filename-based lookup for {filename_for_dedup}")
                        existing_analysis = await db_service.get_analysis_by_filename(filename_for_dedup)
                    
                    if existing_analysis and existing_analysis.get("status") == "completed":
                        # CRITICAL FIX: Validate that cached analysis has required basic features
                        features = existing_analysis.get("features", {})
                        analysis_section = features.get("analysis", {}) if isinstance(features, dict) else {}
                        basic_features = analysis_section.get("basic", {}) if isinstance(analysis_section, dict) else {}
                        
                        # Check for essential basic features that ML training requires
                        required_basic_features = ["energy", "valence", "tempo", "danceability"]
                        missing_features = [f for f in required_basic_features if f not in basic_features]
                        
                        if missing_features:
                            logger.warning(f"⚠️ Cached analysis for {filename_for_dedup} is incomplete - missing basic features: {missing_features}")
                            logger.info(f"🔄 Forcing fresh analysis to ensure complete feature extraction")
                            # Continue to fresh analysis instead of returning incomplete cache
                        else:
                            dedup_method = "content_hash" if file_hash else "filename"
                            logger.info(f"✅ Server-side deduplication ({dedup_method}): Found existing analysis for {filename_for_dedup}")
                            
                            # Track reuse for audit purposes
                            await db_service._log_event(
                                aggregate_id=existing_analysis["analysis_id"],
                                event_type="analysis_reused",
                                event_data={
                                    "original_filename": filename_for_dedup,
                                    "reused_by_session": requested_by,
                                    "workflow_id": workflow_id,
                                    "correlation_id": correlation_id,
                                    "endpoint": "persistent_analyzer",
                                    "force_reanalysis": False
                                },
                                correlation_id=correlation_id
                            )
                            
                            # Return existing analysis in the expected format
                            return {
                                "status": "success",
                                "analysis_id": existing_analysis["analysis_id"],
                                "database_id": existing_analysis.get("database_id"),
                                "cached": True,  # Indicate this was served from cache/database
                                "analysis_type": existing_analysis.get("analysis_type", analysis_type),
                                "processing_time_ms": 0,  # No processing time since it was cached
                                "filename": filename_for_dedup,
                                "results": {
                                    "features": existing_analysis.get("features", {}),
                                    "extractor_types": ["audio", "rhythm", "tonal", "timbre", "dynamics", "mood", "genre"],
                                    "metadata": {
                                        "original_analysis_date": existing_analysis.get("completed_at"),
                                        "original_processing_time_ms": existing_analysis.get("processing_time_ms", 0)
                                    },
                                    "extractors_used": ["audio", "rhythm", "tonal", "timbre", "dynamics", "mood", "genre"],
                                    "comprehensive_analysis": True,
                                    "deduplication_source": "server_side_persistent"
                                },
                                "audit": {
                                    "reused_at": datetime.utcnow().isoformat(),
                                    "original_created_at": existing_analysis.get("created_at"),
                                    "original_completed_at": existing_analysis.get("completed_at"),
                                    "workflow_id": workflow_id,
                                    "correlation_id": correlation_id,
                                    "requested_by": requested_by,
                                    "deduplication_method": "filename_based_persistent",
                                    "force_reanalysis": False
                                }
                            }
                        
                except Exception as e:
                    # If deduplication check fails, log but continue with analysis
                    logger.warning(f"Server-side deduplication check failed for {filename_for_dedup}: {e}")
            else:
                logger.info(f"🔄 Force reanalysis requested - bypassing server-side deduplication for {filename_for_dedup}")
            
            # STEP 2: Perform new analysis (either no existing analysis found or force_reanalysis=True)
            logger.info(f"🎵 Performing {'forced ' if force_reanalysis else ''}new analysis for {filename_for_dedup}")
            
            # Generate analysis ID
            analysis_id = self._generate_analysis_id(file_path, analysis_type, kwargs)
            
            # Debug logging
            logger.info(f"🔍 Debug: requested_by={requested_by}, analysis_kwargs={kwargs}")
            logger.info(f"🔍 Generated analysis_id: {analysis_id} for session: {requested_by}")
            
            # Step 1: Check idempotency if key provided
            if idempotency_key and not force_reanalysis:
                existing_result = await db_service.check_idempotency(
                    idempotency_key, 
                    {"file_path": file_path, "analysis_type": analysis_type, "kwargs": kwargs},
                    "/analyze/persistent"
                )
                if existing_result:
                    logger.info(f"🔄 Idempotency hit: returning existing result for {idempotency_key}")
                    return existing_result
            
            # Step 2: Check if analysis already exists (unless forcing reanalysis)
            if not force_reanalysis:
                existing_analysis = await db_service.get_analysis_result(analysis_id)
                if existing_analysis and existing_analysis.get("status") == "completed":
                    logger.info(f"🎯 Analysis already exists and completed: {analysis_id}")
                    
                    # Track the reuse
                    await self._track_analysis_reuse(analysis_id, workflow_id, correlation_id)
                    
                    # Clean and return existing analysis
                    cleaned_analysis = self._clean_analysis_response(existing_analysis)
                    return {
                        "status": "in_progress",
                        "analysis_id": analysis_id,
                        "cached": True,
                        "results": cleaned_analysis
                    }
            else:
                logger.info(f"🔄 Force reanalysis requested - bypassing existing analysis checks for {analysis_id}")
            
            # Step 3: Create pending analysis record (or update existing if forcing)
            await self._create_pending_analysis_record(
                analysis_id, file_path, analysis_type, workflow_id, requested_by, file_id, original_filename, force_reanalysis
            )
            
            # Step 4: Perform the actual analysis
            start_time = time.time()
            analysis_results = await self._perform_analysis(file_path, analysis_type, **kwargs)
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Step 5: Extract file metadata and calculate file hash for persistence
            file_metadata = await self._extract_file_metadata(file_path)
            file_hash = self._calculate_file_hash(file_path)  # Calculate hash for content-based storage
            
            # Step 6: Save complete analysis results
            database_id = await db_service.save_analysis_result(
                analysis_id=analysis_id,
                file_path=file_path,
                analysis_type=analysis_type,
                features=analysis_results.get("features", {}),
                # Additional metadata
                file_id=file_id,
                filename=original_filename or Path(file_path).name,  # Use original filename if provided
                file_hash=file_hash,  # Store file hash for content-based deduplication
                file_size=file_metadata.get("file_size"),
                duration=file_metadata.get("duration"),
                sample_rate=file_metadata.get("sample_rate"),
                channels=file_metadata.get("channels"),
                format=file_metadata.get("format"),
                raw_features=analysis_results.get("raw_features"),
                confidence_scores=analysis_results.get("confidence_scores"),
                metadata=analysis_results.get("metadata"),
                processing_time_ms=processing_time_ms,
                memory_usage_mb=kwargs.get("memory_usage_mb"),
                cpu_usage_percent=kwargs.get("cpu_usage_percent"),
                workflow_id=workflow_id,
                requested_by=requested_by,
                extractor_types=analysis_results.get("extractor_types"),
                correlation_id=correlation_id,
                target_service=kwargs.get("target_service", "unknown"),
                force_reanalysis=force_reanalysis  # Track if this was a forced reanalysis
            )
            
            # Update status to completed
            await db_service.update_analysis_status(analysis_id, AnalysisStatus.COMPLETED)
            
            # Step 7: Prepare response
            response = {
                "status": "success",
                "analysis_id": analysis_id,
                "database_id": database_id,
                "cached": False,
                "analysis_type": analysis_type,
                "processing_time_ms": processing_time_ms,
                "filename": original_filename or Path(file_path).name,  # Use original filename if provided
                "results": analysis_results,
                "audit": {
                    "created_at": datetime.utcnow().isoformat(),
                    "workflow_id": workflow_id,
                    "correlation_id": correlation_id,
                    "requested_by": requested_by,
                    "force_reanalysis": force_reanalysis
                }
            }
            
            # Clean the response to prevent JSON serialization errors
            cleaned_response = self._clean_analysis_response(response)
            
            # Step 8: Store idempotency result if key provided
            if idempotency_key:
                await db_service.store_idempotency_result(
                    idempotency_key=idempotency_key,
                    request_data={
                        "file_path": file_path,
                        "analysis_type": analysis_type,
                        "kwargs": kwargs
                    },
                    response_data=cleaned_response,
                    endpoint="/analyze/persistent",
                    analysis_id=analysis_id
                )
            
            logger.info(f"✅ Analysis completed and persisted: {analysis_id}")
            return cleaned_response
            
        except Exception as e:
            # Update analysis status to failed
            await db_service.update_analysis_status(
                analysis_id, 
                AnalysisStatus.FAILED, 
                error_message=str(e)
            )
            
            logger.error(f"❌ Analysis failed for {analysis_id}: {e}")
            raise
    
    async def get_persisted_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve persisted analysis result by ID
        Provides access to previously computed results that would otherwise be lost
        """
        result = await db_service.get_analysis_result(analysis_id)
        if result:
            logger.info(f"📖 Retrieved persisted analysis: {analysis_id}")
        return result
    
    async def list_analyses_by_workflow(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get all analysis results for a specific workflow"""
        # This would be implemented to support workflow-level data access
        # Currently returning placeholder - would need to add this query to database service
        logger.info(f"📋 Listing analyses for workflow: {workflow_id}")
        return []
    
    async def get_analysis_for_file(
        self, 
        file_path: str, 
        analysis_type: str = "basic"
    ) -> Optional[Dict[str, Any]]:
        """
        Get existing analysis for a file
        Enables data reuse and prevents duplicate processing
        """
        analysis_id = self._generate_analysis_id(file_path, analysis_type, {})
        return await self.get_persisted_analysis(analysis_id)
    
    # =============================================================================
    # CORE ANALYSIS METHODS
    # =============================================================================
    
    async def _perform_analysis(
        self, 
        file_path: str, 
        analysis_type: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """Perform the actual audio analysis"""
        if analysis_type == "basic":
            results = self.basic_analyzer.analyze_audio(file_path)
            return {
                "features": results,
                "extractor_types": ["basic_essentia"],
                "metadata": {"analysis_method": "basic"}
            }
        
        elif analysis_type == "comprehensive":
            results = self.comprehensive_analyzer.analyze(file_path)
            return {
                "features": results,
                "extractor_types": ["comprehensive_multi"],
                "metadata": {"analysis_method": "comprehensive"}
            }
        
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
    
    async def _extract_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from audio file"""
        try:
            file_path_obj = Path(file_path)
            metadata = {
                "file_size": file_path_obj.stat().st_size if file_path_obj.exists() else None,
                "format": file_path_obj.suffix.lower().lstrip('.'),
            }
            
            # Try to get audio-specific metadata using basic analyzer
            try:
                basic_info = self.basic_analyzer.get_audio_info(file_path)
                metadata.update({
                    "duration": basic_info.get("duration"),
                    "sample_rate": basic_info.get("sample_rate"),
                    "channels": basic_info.get("channels")
                })
            except:
                pass  # Fall back to file-level metadata only
                
            return metadata
            
        except Exception as e:
            logger.warning(f"Could not extract file metadata for {file_path}: {e}")
            return {}
    
    async def _create_pending_analysis_record(
        self,
        analysis_id: str,
        file_path: str,
        analysis_type: str,
        workflow_id: Optional[str],
        requested_by: Optional[str],
        file_id: Optional[str],
        original_filename: Optional[str],
        force_reanalysis: bool
    ):
        """Create a pending analysis record to track processing"""
        await db_service.save_analysis_result(
            analysis_id=analysis_id,
            file_path=file_path,
            analysis_type=analysis_type,
            features={},  # Empty until analysis completes
            file_id=file_id,
            filename=original_filename or Path(file_path).name,
            workflow_id=workflow_id,
            requested_by=requested_by,
            force_reanalysis=force_reanalysis
        )
        
        # Update status to processing
        await db_service.update_analysis_status(analysis_id, AnalysisStatus.PROCESSING)
    
    async def _track_analysis_reuse(
        self, 
        analysis_id: str, 
        workflow_id: Optional[str], 
        correlation_id: Optional[str]
    ):
        """Track when existing analysis is reused"""
        if workflow_id or correlation_id:
            # This would call db_service._track_data_lineage
            # to record that existing analysis was reused
            pass
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def _generate_analysis_id(
        self, 
        file_path: str, 
        analysis_type: str, 
        kwargs: Dict[str, Any]
    ) -> str:
        """
        Generate deterministic analysis ID based on original file info and parameters
        Uses original filename + file hash for content-based deduplication
        """
        try:
            # Get original filename from kwargs if provided (for uploaded files)
            original_filename = kwargs.get("original_filename") or Path(file_path).name
            
            # Calculate file content hash for content-based deduplication
            file_hash = self._calculate_file_hash(file_path)
            
            # Include original filename, file hash, and analysis type for deterministic ID
            input_data = {
                "original_filename": original_filename,  # Use original filename for consistency
                "file_hash": file_hash,
                "analysis_type": analysis_type,
                # Remove timestamp to enable proper deduplication
                "parameters": {k: v for k, v in kwargs.items() if k in [
                    "extractors", "sample_rate", "frame_size", "hop_size"
                ]}
            }
            
            input_str = json.dumps(input_data, sort_keys=True)
            hash_obj = hashlib.sha256(input_str.encode())
            return f"audio_analysis_{hash_obj.hexdigest()[:16]}"
            
        except Exception as e:
            logger.warning(f"Could not calculate file hash for {file_path}, falling back to filename-based ID: {e}")
            # Fallback to filename-based ID if file hash fails
            original_filename = kwargs.get("original_filename") or Path(file_path).name
            input_data = {
                "original_filename": original_filename,
                "file_path": str(file_path),  # Include file path as backup
                "analysis_type": analysis_type,
                # Remove timestamp from fallback too
                "parameters": {k: v for k, v in kwargs.items() if k in [
                    "extractors", "sample_rate", "frame_size", "hop_size"
                ]}
            }
            input_str = json.dumps(input_data, sort_keys=True)
            hash_obj = hashlib.sha256(input_str.encode())
            return f"audio_analysis_{hash_obj.hexdigest()[:16]}"
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file content"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash for {file_path}: {e}")
            # Fallback to file path + size + modification time
            file_path_obj = Path(file_path)
            if file_path_obj.exists():
                stat = file_path_obj.stat()
                fallback_data = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
                return hashlib.sha256(fallback_data.encode()).hexdigest()
            else:
                return hashlib.sha256(str(file_path).encode()).hexdigest()
    
    # =============================================================================
    # SERVICE INTEGRATION METHODS - For other services to access data
    # =============================================================================
    
    async def get_analysis_for_prediction(
        self, 
        file_id: str, 
        required_features: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get analysis data formatted for ML prediction service
        This enables the ML prediction service to access persisted audio features
        """
        # Would query by file_id and return formatted features
        # This is the method that ML prediction service would call
        logger.info(f"🎯 Getting analysis for prediction: {file_id}")
        
        # Placeholder implementation - would need to add file_id query to database service
        return None
    
    async def get_analysis_for_insights(
        self, 
        analysis_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get analysis data formatted for AI insights service
        This enables the AI insights service to access persisted audio features
        """
        results = []
        for analysis_id in analysis_ids:
            result = await self.get_persisted_analysis(analysis_id)
            if result:
                results.append(result)
        
        logger.info(f"💡 Retrieved {len(results)} analyses for insights")
        return results
    
    # =============================================================================
    # HEALTH AND METRICS
    # =============================================================================
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get health status including database connectivity"""
        health = {
            "analyzer": "healthy",
            "persistence": await db_service.get_health_status()
        }
        
        # Check if analyzers are working
        try:
            self.basic_analyzer.get_analysis_status()
            health["basic_analyzer"] = "healthy"
        except:
            health["basic_analyzer"] = "unhealthy"
        
        try:
            self.comprehensive_analyzer.get_status()
            health["comprehensive_analyzer"] = "healthy"  
        except:
            health["comprehensive_analyzer"] = "unhealthy"
        
        return health
    
    async def cleanup(self):
        """Cleanup resources"""
        await db_service.cleanup()

    def _clean_analysis_response(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Clean the analysis response to prevent JSON serialization errors"""
        def clean_value(value):
            """Recursively clean values to handle -Infinity and other JSON-incompatible values"""
            if isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [clean_value(item) for item in value]
            elif isinstance(value, float):
                if value == float('-inf'):
                    return -999999.0  # Replace -Infinity with a large negative number
                elif value == float('inf'):
                    return 999999.0   # Replace Infinity with a large positive number
                elif value != value:  # NaN check
                    return 0.0        # Replace NaN with 0
                else:
                    return value
            else:
                return value
        
        return clean_value(analysis_result)

# Global persistent analyzer instance
persistent_analyzer = PersistentAudioAnalyzer() 