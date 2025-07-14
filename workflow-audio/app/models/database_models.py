"""
Database models for workflow-audio service
Handles persistent storage of audio analysis results and audit trails
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from sqlalchemy import Column, String, Integer, DateTime, Boolean, Text, Float, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

Base = declarative_base()

class AnalysisStatus(str, Enum):
    """Audio analysis status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"

class AnalysisType(str, Enum):
    """Audio analysis type enumeration"""
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    CUSTOM = "custom"

class AudioAnalysisResult(Base):
    """Database model for audio analysis results - Critical for data persistence"""
    
    __tablename__ = "audio_analysis_results"
    
    # Primary key and identifiers
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id = Column(String(255), unique=True, nullable=False, index=True)
    file_id = Column(String(255), nullable=True, index=True)  # From storage service
    file_path = Column(String(512), nullable=False)
    original_filename = Column(String(255), nullable=True)
    file_hash = Column(String(64), nullable=True, index=True)  # SHA256 hash for content-based deduplication
    
    # Analysis metadata
    analysis_type = Column(String(50), nullable=False, default=AnalysisType.BASIC)
    status = Column(String(50), nullable=False, default=AnalysisStatus.PENDING, index=True)
    extractor_types = Column(JSONB, nullable=True)  # List of extractors used
    
    # Audio file properties
    file_size_bytes = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    sample_rate = Column(Integer, nullable=True)
    channels = Column(Integer, nullable=True)
    format = Column(String(20), nullable=True)
    
    # Analysis results - Core data that was being lost
    features = Column(JSONB, nullable=True)  # Extracted audio features
    raw_features = Column(JSONB, nullable=True)  # Raw extractor outputs
    confidence_scores = Column(JSONB, nullable=True)  # Feature confidence
    feature_metadata = Column(JSONB, nullable=True)  # Extraction metadata
    
    # Performance metrics
    processing_time_ms = Column(Integer, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    cpu_usage_percent = Column(Float, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    error_traceback = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Audit trail
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    requested_by = Column(String(255), nullable=True)  # Service/user who requested
    workflow_id = Column(String(255), nullable=True, index=True)  # Parent workflow
    
    # Data lineage
    derived_from = Column(UUID(as_uuid=True), nullable=True)  # Parent analysis ID
    version = Column(Integer, default=1)
    checksum = Column(String(64), nullable=True)  # Feature data checksum
    
    # Retention and lifecycle
    expires_at = Column(DateTime, nullable=True)
    is_archived = Column(Boolean, default=False)
    archived_at = Column(DateTime, nullable=True)
    
    __table_args__ = (
        Index('idx_analysis_file_type', 'file_id', 'analysis_type'),
        Index('idx_analysis_status_created', 'status', 'created_at'),
        Index('idx_analysis_workflow', 'workflow_id', 'created_at'),
    )

class FeatureCache(Base):
    """Cache table for expensive feature computations"""
    
    __tablename__ = "feature_cache"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cache_key = Column(String(255), unique=True, nullable=False, index=True)
    file_hash = Column(String(64), nullable=False, index=True)
    extractor_config_hash = Column(String(64), nullable=False)
    
    # Cached data
    features = Column(JSONB, nullable=False)
    feature_metadata = Column(JSONB, nullable=True)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    
    # Cache management
    access_count = Column(Integer, default=1)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    
    # Performance tracking
    computation_time_ms = Column(Integer, nullable=True)
    cache_hit_count = Column(Integer, default=0)

class IdempotencyKey(Base):
    """Idempotency tracking for preventing duplicate analyses"""
    
    __tablename__ = "idempotency_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    idempotency_key = Column(String(255), unique=True, nullable=False, index=True)
    
    # Request details
    request_hash = Column(String(64), nullable=False)
    endpoint = Column(String(100), nullable=False)
    method = Column(String(10), nullable=False)
    
    # Response data
    response_data = Column(JSONB, nullable=True)
    status_code = Column(Integer, nullable=False)
    
    # Analysis reference
    analysis_id = Column(String(255), nullable=True)
    
    # Lifecycle
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    used_count = Column(Integer, default=1)

class EventStore(Base):
    """Event sourcing for complete audit trail - Critical for data lineage"""
    
    __tablename__ = "event_store"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    aggregate_id = Column(String(255), nullable=False, index=True)  # analysis_id
    event_type = Column(String(100), nullable=False, index=True)
    event_data = Column(JSONB, nullable=False)
    
    # Event metadata
    event_version = Column(Integer, nullable=False)
    caused_by_event = Column(UUID(as_uuid=True), nullable=True)
    correlation_id = Column(String(255), nullable=True, index=True)
    
    # Context
    service_name = Column(String(100), default="workflow-audio")
    user_id = Column(String(255), nullable=True)
    session_id = Column(String(255), nullable=True)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_event_aggregate_version', 'aggregate_id', 'event_version'),
        Index('idx_event_type_created', 'event_type', 'created_at'),
        Index('idx_event_correlation', 'correlation_id', 'created_at'),
    )

class AnalysisMetrics(Base):
    """Service performance and usage metrics"""
    
    __tablename__ = "analysis_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Time window
    metric_date = Column(DateTime, nullable=False, index=True)
    time_bucket = Column(String(20), nullable=False)  # hour, day, week
    
    # Counters
    total_analyses = Column(Integer, default=0)
    successful_analyses = Column(Integer, default=0)
    failed_analyses = Column(Integer, default=0)
    cached_analyses = Column(Integer, default=0)
    
    # Performance
    avg_processing_time_ms = Column(Float, nullable=True)
    min_processing_time_ms = Column(Float, nullable=True)
    max_processing_time_ms = Column(Float, nullable=True)
    total_processing_time_ms = Column(Float, default=0)
    
    # Resource usage
    avg_memory_usage_mb = Column(Float, nullable=True)
    max_memory_usage_mb = Column(Float, nullable=True)
    avg_cpu_usage_percent = Column(Float, nullable=True)
    
    # File stats
    total_files_processed = Column(Integer, default=0)
    total_bytes_processed = Column(Integer, default=0)
    avg_file_size_mb = Column(Float, nullable=True)
    
    # Analysis types
    basic_analyses = Column(Integer, default=0)
    comprehensive_analyses = Column(Integer, default=0)
    custom_analyses = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_metrics_date_bucket', 'metric_date', 'time_bucket'),
    )

class DataLineage(Base):
    """Track data flow between services - Critical for debugging and compliance"""
    
    __tablename__ = "data_lineage"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Data flow
    source_service = Column(String(100), nullable=False)
    target_service = Column(String(100), nullable=False)
    data_id = Column(String(255), nullable=False, index=True)
    operation = Column(String(100), nullable=False)
    
    # Context
    workflow_id = Column(String(255), nullable=True, index=True)
    correlation_id = Column(String(255), nullable=True, index=True)
    
    # Data details
    data_type = Column(String(100), nullable=False)
    data_size_bytes = Column(Integer, nullable=True)
    data_hash = Column(String(64), nullable=True)
    
    # Transformation info
    transformation_applied = Column(String(200), nullable=True)
    transformation_params = Column(JSONB, nullable=True)
    
    # Quality metrics
    data_quality_score = Column(Float, nullable=True)
    validation_status = Column(String(50), nullable=True)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_lineage_source_target', 'source_service', 'target_service'),
        Index('idx_lineage_workflow', 'workflow_id', 'created_at'),
        Index('idx_lineage_data_id', 'data_id', 'created_at'),
    ) 