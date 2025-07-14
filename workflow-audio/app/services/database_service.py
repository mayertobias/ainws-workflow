"""
Database service for workflow-audio microservice
Handles all database operations, caching, and persistence
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from contextlib import asynccontextmanager

import asyncpg
import redis.asyncio as redis
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from ..config.settings import get_settings
from ..models.database_models import (
    Base, AudioAnalysisResult, FeatureCache, IdempotencyKey, 
    EventStore, AnalysisMetrics, DataLineage,
    AnalysisStatus, AnalysisType
)

logger = logging.getLogger(__name__)

class DatabaseService:
    """
    Comprehensive database service for audio analysis persistence
    Handles the critical data loss issue identified in the analysis
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.engine = None
        self.SessionLocal = None
        self.redis_client = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize database connections and create tables"""
        if self._initialized:
            return
            
        try:
            # Initialize PostgreSQL
            self.engine = create_engine(
                self.settings.database_url,
                poolclass=QueuePool,
                pool_size=self.settings.database_pool_size,
                max_overflow=self.settings.database_max_overflow,
                pool_timeout=self.settings.database_pool_timeout,
                echo=self.settings.debug
            )
            
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            
            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Initialize Redis
            self.redis_client = redis.from_url(
                self.settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connections
            await self._test_connections()
            
            self._initialized = True
            logger.info("✅ Database service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    async def _test_connections(self):
        """Test database and Redis connections"""
        # Test PostgreSQL
        with self.SessionLocal() as session:
            session.execute(text("SELECT 1"))
            
        # Test Redis
        await self.redis_client.ping()
        
    @asynccontextmanager
    async def get_session(self):
        """Get database session with proper cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    # =============================================================================
    # CORE ANALYSIS PERSISTENCE - Fixing the critical data loss issue
    # =============================================================================
    
    def _clean_features_for_db(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean features data for database storage
        PostgreSQL JSON doesn't support -Infinity, Infinity, or NaN values
        """
        import math
        import json
        
        def clean_value(value):
            if isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [clean_value(v) for v in value]
            elif isinstance(value, float):
                if math.isinf(value) or math.isnan(value):
                    return None
                return value
            else:
                return value
        
        return clean_value(features)

    async def save_analysis_result(
        self, 
        analysis_id: str,
        file_path: str,
        analysis_type: str,
        features: Dict[str, Any],
        **kwargs
    ) -> str:
        """
        Save audio analysis result to persistent storage
        This fixes the critical issue where analysis results were lost
        """
        async with self.get_session() as session:
            try:
                # Clean features to handle -Infinity values
                cleaned_features = self._clean_features_for_db(features)
                
                # Check if analysis already exists
                existing_analysis = session.query(AudioAnalysisResult).filter(
                    AudioAnalysisResult.analysis_id == analysis_id
                ).first()
                
                if existing_analysis:
                    # Update existing analysis with new features if they're not empty
                    if cleaned_features:
                        logger.info(f"🔄 Updating existing analysis {analysis_id} with features")
                        existing_analysis.features = cleaned_features
                        existing_analysis.status = AnalysisStatus.COMPLETED
                        existing_analysis.raw_features = kwargs.get('raw_features')
                        existing_analysis.confidence_scores = kwargs.get('confidence_scores')
                        existing_analysis.feature_metadata = kwargs.get('metadata')
                        existing_analysis.processing_time_ms = kwargs.get('processing_time_ms')
                        existing_analysis.memory_usage_mb = kwargs.get('memory_usage_mb')
                        existing_analysis.cpu_usage_percent = kwargs.get('cpu_usage_percent')
                        existing_analysis.extractor_types = kwargs.get('extractor_types')
                        existing_analysis.completed_at = datetime.utcnow()
                        existing_analysis.updated_at = datetime.utcnow()
                        existing_analysis.checksum = self._calculate_feature_checksum(cleaned_features)
                        
                        # Update file metadata if provided
                        if kwargs.get('file_hash'):
                            existing_analysis.file_hash = kwargs.get('file_hash')
                        if kwargs.get('file_size'):
                            existing_analysis.file_size_bytes = kwargs.get('file_size')
                        if kwargs.get('duration'):
                            existing_analysis.duration_seconds = kwargs.get('duration')
                        if kwargs.get('sample_rate'):
                            existing_analysis.sample_rate = kwargs.get('sample_rate')
                        if kwargs.get('channels'):
                            existing_analysis.channels = kwargs.get('channels')
                        if kwargs.get('format'):
                            existing_analysis.format = kwargs.get('format')
                    else:
                        logger.info(f"📝 Analysis {analysis_id} exists with empty features - keeping existing record")
                    
                    analysis_record = existing_analysis
                    
                else:
                    # Create new analysis record
                    analysis_record = AudioAnalysisResult(
                        analysis_id=analysis_id,
                        file_path=file_path,
                        analysis_type=analysis_type,
                        features=cleaned_features,  # Use cleaned features
                        status=AnalysisStatus.COMPLETED if cleaned_features else AnalysisStatus.PENDING,
                        file_id=kwargs.get('file_id'),
                        original_filename=kwargs.get('filename'),
                        file_hash=kwargs.get('file_hash'),  # Store file hash for content-based deduplication
                        file_size_bytes=kwargs.get('file_size'),
                        duration_seconds=kwargs.get('duration'),
                        sample_rate=kwargs.get('sample_rate'),
                        channels=kwargs.get('channels'),
                        format=kwargs.get('format'),
                        raw_features=kwargs.get('raw_features'),
                        confidence_scores=kwargs.get('confidence_scores'),
                        feature_metadata=kwargs.get('metadata'),
                        processing_time_ms=kwargs.get('processing_time_ms'),
                        memory_usage_mb=kwargs.get('memory_usage_mb'),
                        cpu_usage_percent=kwargs.get('cpu_usage_percent'),
                        workflow_id=kwargs.get('workflow_id'),
                        requested_by=kwargs.get('requested_by'),
                        extractor_types=kwargs.get('extractor_types'),
                        completed_at=datetime.utcnow() if cleaned_features else None,
                        checksum=self._calculate_feature_checksum(cleaned_features) if cleaned_features else None
                    )
                    
                    session.add(analysis_record)
                
                session.flush()  # Get the ID
                
                # Cache the result for fast access (only if features exist)
                if self.settings.enable_feature_caching and cleaned_features:
                    await self._cache_analysis_result(analysis_id, cleaned_features, kwargs.get('metadata', {}))
                
                # Log event for audit trail (only if features exist)
                if cleaned_features:
                    await self._log_event(
                        analysis_id, 
                        "analysis_completed",
                        {
                            "analysis_type": analysis_type,
                            "feature_count": len(cleaned_features),
                            "file_path": file_path,
                            "processing_time_ms": kwargs.get('processing_time_ms')
                        },
                        kwargs.get('correlation_id')
                    )
                    
                    # Track data lineage
                    await self._track_data_lineage(
                        source_service="workflow-audio",
                        target_service=kwargs.get('target_service', "unknown"),
                        data_id=analysis_id,
                        operation="analysis_result_created",
                        workflow_id=kwargs.get('workflow_id'),
                        data_type="audio_features",
                        data_size_bytes=len(json.dumps(cleaned_features).encode())
                    )
                
                logger.info(f"✅ Analysis result saved: {analysis_id}")
                return str(analysis_record.id)
                
            except Exception as e:
                session.rollback()  # Rollback on any error
                logger.error(f"Failed to save analysis result {analysis_id}: {e}")
                raise
    
    async def get_analysis_result(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve analysis result by ID
        Provides access to persisted data that was previously lost
        """
        # Try cache first
        if self.settings.enable_feature_caching:
            cached = await self._get_cached_analysis(analysis_id)
            if cached:
                logger.debug(f"🚀 Cache hit for analysis {analysis_id}")
                await self._increment_cache_hit_count(analysis_id)
                return cached
        
        # Query database
        async with self.get_session() as session:
            analysis = session.query(AudioAnalysisResult).filter(
                AudioAnalysisResult.analysis_id == analysis_id
            ).first()
            
            if not analysis:
                return None
            
            result = {
                "analysis_id": analysis.analysis_id,
                "file_id": analysis.file_id,
                "file_path": analysis.file_path,
                "original_filename": analysis.original_filename,
                "analysis_type": analysis.analysis_type,
                "status": analysis.status,
                "features": analysis.features,
                "raw_features": analysis.raw_features,
                "confidence_scores": analysis.confidence_scores,
                "feature_metadata": analysis.feature_metadata,
                "file_properties": {
                    "file_size_bytes": analysis.file_size_bytes,
                    "duration_seconds": analysis.duration_seconds,
                    "sample_rate": analysis.sample_rate,
                    "channels": analysis.channels,
                    "format": analysis.format
                },
                "performance": {
                    "processing_time_ms": analysis.processing_time_ms,
                    "memory_usage_mb": analysis.memory_usage_mb,
                    "cpu_usage_percent": analysis.cpu_usage_percent
                },
                "audit": {
                    "created_at": analysis.created_at.isoformat(),
                    "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None,
                    "requested_by": analysis.requested_by,
                    "workflow_id": analysis.workflow_id,
                    "version": analysis.version,
                    "checksum": analysis.checksum
                },
                "error_info": {
                    "error_message": analysis.error_message,
                    "retry_count": analysis.retry_count
                } if analysis.error_message else None
            }
            
            # Update cache
            if self.settings.enable_feature_caching:
                await self._cache_analysis_result(analysis_id, analysis.features, analysis.feature_metadata)
            
            return result
    
    async def update_analysis_status(
        self, 
        analysis_id: str, 
        status: AnalysisStatus,
        error_message: Optional[str] = None,
        retry_count: Optional[int] = None
    ):
        """Update analysis status and error information"""
        async with self.get_session() as session:
            analysis = session.query(AudioAnalysisResult).filter(
                AudioAnalysisResult.analysis_id == analysis_id
            ).first()
            
            if analysis:
                analysis.status = status
                analysis.updated_at = datetime.utcnow()
                
                if error_message:
                    analysis.error_message = error_message
                if retry_count is not None:
                    analysis.retry_count = retry_count
                    
                # Log status change event
                await self._log_event(
                    analysis_id,
                    "status_changed",
                    {
                        "old_status": analysis.status,
                        "new_status": status,
                        "error_message": error_message,
                        "retry_count": retry_count
                    }
                )
    
    # =============================================================================
    # IDEMPOTENCY SUPPORT - Critical for preventing duplicate processing
    # =============================================================================
    
    async def check_idempotency(
        self, 
        idempotency_key: str, 
        request_data: Dict[str, Any],
        endpoint: str,
        method: str = "POST"
    ) -> Optional[Dict[str, Any]]:
        """
        Check if request was already processed (idempotency)
        Returns existing response if found, None if new request
        """
        if not self.settings.enable_idempotency:
            return None
            
        async with self.get_session() as session:
            existing = session.query(IdempotencyKey).filter(
                IdempotencyKey.idempotency_key == idempotency_key,
                IdempotencyKey.expires_at > datetime.utcnow()
            ).first()
            
            if existing:
                # Verify request matches
                request_hash = self._calculate_request_hash(request_data)
                if existing.request_hash == request_hash:
                    # Update usage count
                    existing.used_count += 1
                    logger.info(f"🔄 Idempotent request detected: {idempotency_key}")
                    return existing.response_data
                else:
                    logger.warning(f"⚠️ Idempotency key collision: {idempotency_key}")
                    
        return None
    
    async def store_idempotency_result(
        self,
        idempotency_key: str,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any],
        endpoint: str,
        method: str = "POST",
        status_code: int = 200,
        analysis_id: Optional[str] = None
    ):
        """Store result for idempotency checking"""
        if not self.settings.enable_idempotency:
            return
            
        async with self.get_session() as session:
            # Clean response_data to handle -Infinity values
            cleaned_response_data = self._clean_features_for_db(response_data)
            
            idempotency_record = IdempotencyKey(
                idempotency_key=idempotency_key,
                request_hash=self._calculate_request_hash(request_data),
                endpoint=endpoint,
                method=method,
                response_data=cleaned_response_data,  # Use cleaned data
                status_code=status_code,
                analysis_id=analysis_id,
                expires_at=datetime.utcnow() + timedelta(seconds=self.settings.redis_idempotency_ttl)
            )
            
            session.add(idempotency_record)
            logger.debug(f"💾 Stored idempotency result: {idempotency_key}")
    
    # =============================================================================
    # CACHING LAYER - Performance optimization
    # =============================================================================
    
    async def _cache_analysis_result(self, analysis_id: str, features: Dict[str, Any], metadata: Dict[str, Any]):
        """Cache analysis result in Redis"""
        # Clean data before caching to handle -Infinity values
        cleaned_features = self._clean_features_for_db(features)
        cleaned_metadata = self._clean_features_for_db(metadata)
        
        cache_data = {
            "features": cleaned_features,
            "feature_metadata": cleaned_metadata,
            "cached_at": datetime.utcnow().isoformat()
        }
        
        await self.redis_client.setex(
            f"analysis:{analysis_id}",
            self.settings.redis_cache_ttl,
            json.dumps(cache_data)
        )
    
    async def _get_cached_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result"""
        cached = await self.redis_client.get(f"analysis:{analysis_id}")
        if cached:
            return json.loads(cached)
        return None
    
    async def _increment_cache_hit_count(self, analysis_id: str):
        """Track cache hit for metrics"""
        await self.redis_client.incr(f"cache_hits:{analysis_id}")
    
    # =============================================================================
    # EVENT SOURCING - Audit trail and data lineage
    # =============================================================================
    
    async def _log_event(
        self,
        aggregate_id: str,
        event_type: str,
        event_data: Dict[str, Any],
        correlation_id: Optional[str] = None
    ):
        """Log event for audit trail"""
        async with self.get_session() as session:
            # Get next version number
            last_event = session.query(EventStore).filter(
                EventStore.aggregate_id == aggregate_id
            ).order_by(EventStore.event_version.desc()).first()
            
            next_version = (last_event.event_version + 1) if last_event else 1
            
            # Clean event_data to handle -Infinity values
            cleaned_event_data = self._clean_features_for_db(event_data)
            
            event = EventStore(
                aggregate_id=aggregate_id,
                event_type=event_type,
                event_data=cleaned_event_data,  # Use cleaned data
                event_version=next_version,
                correlation_id=correlation_id,
                service_name="workflow-audio"
            )
            
            session.add(event)
    
    async def _track_data_lineage(
        self,
        source_service: str,
        target_service: str,
        data_id: str,
        operation: str,
        **kwargs
    ):
        """Track data lineage between services"""
        async with self.get_session() as session:
            lineage = DataLineage(
                source_service=source_service,
                target_service=target_service,
                data_id=data_id,
                operation=operation,
                workflow_id=kwargs.get('workflow_id'),
                correlation_id=kwargs.get('correlation_id'),
                data_type=kwargs.get('data_type', 'unknown'),
                data_size_bytes=kwargs.get('data_size_bytes'),
                data_hash=kwargs.get('data_hash'),
                transformation_applied=kwargs.get('transformation_applied'),
                transformation_params=kwargs.get('transformation_params'),
                data_quality_score=kwargs.get('data_quality_score'),
                validation_status=kwargs.get('validation_status')
            )
            
            session.add(lineage)
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def _calculate_feature_checksum(self, features: Dict[str, Any]) -> str:
        """Calculate checksum for feature data integrity"""
        # Clean features before calculating checksum to handle -Infinity values
        cleaned_features = self._clean_features_for_db(features)
        feature_str = json.dumps(cleaned_features, sort_keys=True)
        return hashlib.sha256(feature_str.encode()).hexdigest()
    
    def _calculate_request_hash(self, request_data: Dict[str, Any]) -> str:
        """Calculate hash for request data"""
        # Clean request data before calculating hash to handle -Infinity values
        cleaned_request_data = self._clean_features_for_db(request_data)
        request_str = json.dumps(cleaned_request_data, sort_keys=True)
        return hashlib.sha256(request_str.encode()).hexdigest()
    
    async def get_analysis_result_id(self, analysis_id: str) -> Optional[str]:
        """Get database ID for analysis_id"""
        async with self.get_session() as session:
            analysis = session.query(AudioAnalysisResult).filter(
                AudioAnalysisResult.analysis_id == analysis_id
            ).first()
            return str(analysis.id) if analysis else None
    
    # =============================================================================
    # HEALTH AND METRICS
    # =============================================================================
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get database and cache health status"""
        health = {
            "database": {"status": "unknown", "response_time_ms": None},
            "redis": {"status": "unknown", "response_time_ms": None}
        }
        
        try:
            # Test database
            start_time = time.time()
            async with self.get_session() as session:
                session.execute(text("SELECT 1"))
            db_time = (time.time() - start_time) * 1000
            health["database"] = {"status": "healthy", "response_time_ms": db_time}
        except Exception as e:
            health["database"] = {"status": "unhealthy", "error": str(e)}
        
        try:
            # Test Redis
            start_time = time.time()
            await self.redis_client.ping()
            redis_time = (time.time() - start_time) * 1000
            health["redis"] = {"status": "healthy", "response_time_ms": redis_time}
        except Exception as e:
            health["redis"] = {"status": "unhealthy", "error": str(e)}
        
        return health
    
    # =============================================================================
    # HISTORY AND USER SESSION METHODS - For frontend integration
    # =============================================================================
    
    async def get_audio_analysis_history(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
        search: Optional[str] = None,
        analysis_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get audio analysis history for a user session
        Returns (items, total_count)
        """
        async with self.get_session() as session_db:
            # Build base query
            query = session_db.query(AudioAnalysisResult).filter(
                AudioAnalysisResult.requested_by == session_id
            )
            
            # Apply filters
            if search:
                query = query.filter(
                    AudioAnalysisResult.original_filename.ilike(f"%{search}%")
                )
            
            if analysis_type:
                query = query.filter(
                    AudioAnalysisResult.analysis_type == analysis_type
                )
            
            if status:
                query = query.filter(
                    AudioAnalysisResult.status == status
                )
            
            # Get total count
            total_count = query.count()
            
            # Apply pagination and ordering
            results = query.order_by(
                AudioAnalysisResult.created_at.desc()
            ).offset(offset).limit(limit).all()
            
            # Convert to dict format for API response
            items = []
            for result in results:
                # Calculate hit score if available
                hit_score = None
                if result.features and isinstance(result.features, dict):
                    # Try to extract hit score from features
                    hit_score = result.features.get('hit_score') or result.features.get('commercial_potential')
                
                item = {
                    "analysis_id": result.analysis_id,
                    "original_filename": result.original_filename,
                    "analysis_type": result.analysis_type,
                    "status": result.status.value if hasattr(result.status, 'value') else str(result.status),
                    "created_at": result.created_at.isoformat(),
                    "completed_at": result.completed_at.isoformat() if result.completed_at else None,
                    "processing_time_ms": result.processing_time_ms,
                    "workflow_id": result.workflow_id,
                    "file_id": result.file_id,
                    "hit_score": hit_score,
                    "duration_seconds": result.duration_seconds,
                    "file_size_bytes": result.file_size_bytes,
                    "features": {
                        "duration_ms": int(result.duration_seconds * 1000) if result.duration_seconds else 0,
                        "acousticness": result.features.get('acousticness', 0) if result.features else 0,
                        "instrumentalness": result.features.get('instrumentalness', 0) if result.features else 0
                    }
                }
                items.append(item)
            
            return items, total_count
    
    async def delete_analysis_result(self, analysis_id: str, session_id: str) -> bool:
        """
        Delete an analysis result (with session verification)
        """
        async with self.get_session() as session_db:
            analysis = session_db.query(AudioAnalysisResult).filter(
                AudioAnalysisResult.analysis_id == analysis_id,
                AudioAnalysisResult.requested_by == session_id
            ).first()
            
            if not analysis:
                return False
            
            session_db.delete(analysis)
            
            # Also clean up related records
            session_db.query(FeatureCache).filter(
                FeatureCache.analysis_id == analysis_id
            ).delete()
            
            session_db.query(EventStore).filter(
                EventStore.aggregate_id == analysis_id
            ).delete()
            
            return True
    
    async def get_user_analysis_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get analysis statistics for a user session
        """
        async with self.get_session() as session_db:
            # Get basic counts
            total_query = session_db.query(AudioAnalysisResult).filter(
                AudioAnalysisResult.requested_by == session_id
            )
            
            total_analyses = total_query.count()
            completed_analyses = total_query.filter(
                AudioAnalysisResult.status == AnalysisStatus.COMPLETED
            ).count()
            failed_analyses = total_query.filter(
                AudioAnalysisResult.status == AnalysisStatus.FAILED
            ).count()
            processing_analyses = total_query.filter(
                AudioAnalysisResult.status == AnalysisStatus.PROCESSING
            ).count()
            
            # Get analysis type breakdown
            basic_count = total_query.filter(
                AudioAnalysisResult.analysis_type == "basic"
            ).count()
            comprehensive_count = total_query.filter(
                AudioAnalysisResult.analysis_type == "comprehensive"
            ).count()
            
            # Calculate average processing time
            completed_results = total_query.filter(
                AudioAnalysisResult.status == AnalysisStatus.COMPLETED,
                AudioAnalysisResult.processing_time_ms.isnot(None)
            ).all()
            
            avg_processing_time = 0
            if completed_results:
                total_time = sum(r.processing_time_ms for r in completed_results if r.processing_time_ms)
                avg_processing_time = total_time / len(completed_results)
            
            return {
                "total_analyses": total_analyses,
                "completed": completed_analyses,
                "failed": failed_analyses,
                "processing": processing_analyses,
                "analysis_types": {
                    "basic": basic_count,
                    "comprehensive": comprehensive_count
                },
                "avg_processing_time_ms": avg_processing_time,
                "success_rate": (completed_analyses / total_analyses * 100) if total_analyses > 0 else 0
            }
    
    async def get_user_workflows(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get workflows for a user session
        """
        async with self.get_session() as session_db:
            # Get distinct workflows with counts
            workflows = session_db.query(
                AudioAnalysisResult.workflow_id,
                AudioAnalysisResult.created_at
            ).filter(
                AudioAnalysisResult.requested_by == session_id,
                AudioAnalysisResult.workflow_id.isnot(None)
            ).distinct().order_by(
                AudioAnalysisResult.created_at.desc()
            ).limit(limit).all()
            
            workflow_list = []
            for workflow_id, created_at in workflows:
                # Get analysis count for this workflow
                analysis_count = session_db.query(AudioAnalysisResult).filter(
                    AudioAnalysisResult.workflow_id == workflow_id,
                    AudioAnalysisResult.requested_by == session_id
                ).count()
                
                workflow_list.append({
                    "workflow_id": workflow_id,
                    "created_at": created_at.isoformat(),
                    "analysis_count": analysis_count
                })
            
            return workflow_list
    
    async def cleanup(self):
        """Cleanup database connections"""
        if self.redis_client:
            await self.redis_client.close()
        if self.engine:
            self.engine.dispose()

    async def get_analysis_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent completed analysis result by original filename
        This enables deduplication by checking if a file has already been analyzed
        """
        async with self.get_session() as session:
            try:
                # Query for the most recent completed analysis with this filename
                result = session.query(AudioAnalysisResult).filter(
                    AudioAnalysisResult.original_filename == filename,
                    AudioAnalysisResult.status == AnalysisStatus.COMPLETED
                ).order_by(AudioAnalysisResult.completed_at.desc()).first()
                
                if result:
                    logger.info(f"Found existing analysis for filename: {filename}")
                    return {
                        "analysis_id": result.analysis_id,
                        "original_filename": result.original_filename,
                        "features": result.features,
                        "status": result.status.value if hasattr(result.status, 'value') else str(result.status),
                        "created_at": result.created_at.isoformat(),
                        "completed_at": result.completed_at.isoformat() if result.completed_at else None,
                        "processing_time_ms": result.processing_time_ms,
                        "analysis_type": result.analysis_type,
                        "file_path": result.file_path,
                        "database_id": str(result.id)
                    }
                else:
                    logger.debug(f"No existing analysis found for filename: {filename}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error getting analysis by filename {filename}: {e}")
                return None

    async def get_analysis_by_file_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get analysis result by file content hash for content-based deduplication
        
        Args:
            file_hash: SHA256 hash of file content
            
        Returns:
            Dictionary with analysis data or None if not found
        """
        async with self.get_session() as session:
            try:
                # Query for the most recent completed analysis with this file hash
                result = session.query(AudioAnalysisResult).filter(
                    AudioAnalysisResult.file_hash == file_hash,
                    AudioAnalysisResult.status == AnalysisStatus.COMPLETED
                ).order_by(AudioAnalysisResult.completed_at.desc()).first()
                
                if result:
                    logger.info(f"✅ Found existing analysis by content hash: {file_hash[:16]}...")
                    return {
                        "analysis_id": result.analysis_id,
                        "original_filename": result.original_filename,
                        "features": result.features,
                        "status": result.status.value if hasattr(result.status, 'value') else str(result.status),
                        "created_at": result.created_at.isoformat(),
                        "completed_at": result.completed_at.isoformat() if result.completed_at else None,
                        "processing_time_ms": result.processing_time_ms,
                        "analysis_type": result.analysis_type,
                        "file_path": result.file_path,
                        "file_hash": result.file_hash,
                        "database_id": str(result.id)
                    }
                else:
                    logger.debug(f"No existing analysis found for file hash: {file_hash[:16]}...")
                    return None
                    
            except Exception as e:
                logger.error(f"Error getting analysis by file hash {file_hash[:16]}...: {e}")
                return None

# Global database service instance
db_service = DatabaseService() 