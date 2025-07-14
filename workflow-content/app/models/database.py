"""
Database models for workflow-content service
Handles persistent storage of lyrics analysis results and user sessions
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, Integer, DateTime, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

Base = declarative_base()

class LyricsAnalysisResult(Base):
    """Database model for lyrics analysis results with history"""
    
    __tablename__ = "lyrics_analysis_results"
    
    # Primary key and identifiers
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # File and title information
    filename = Column(String(500), nullable=True)  # Original filename if uploaded
    title = Column(String(500), nullable=True)     # User-provided title or name
    
    # Original text and analysis data
    original_text = Column(Text, nullable=False)
    analysis_results = Column(JSONB, nullable=False)
    hss_features = Column(JSONB, nullable=True)
    
    # Processing metadata
    processing_time_ms = Column(Integer, nullable=True)
    
    # User session tracking for history
    user_session = Column(String(255), nullable=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_user_session_created', 'user_session', 'created_at'),
        Index('idx_analysis_id', 'analysis_id'),
        Index('idx_created_at', 'created_at'),
        Index('idx_filename', 'filename'),
        Index('idx_title', 'title'),
    )

class UserSession(Base):
    """User session tracking for personalized history"""
    
    __tablename__ = "user_sessions"
    
    # Session identifier
    session_id = Column(String(255), primary_key=True)
    
    # Session metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    analysis_count = Column(Integer, default=0)
    
    # Optional user information (for future authentication)
    user_agent = Column(Text, nullable=True)
    
    # Index for cleanup queries
    __table_args__ = (
        Index('idx_last_accessed', 'last_accessed'),
    ) 