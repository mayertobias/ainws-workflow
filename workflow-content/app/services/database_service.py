"""
Database service for workflow-content microservice
Handles all database operations for lyrics analysis history and persistence
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from contextlib import asynccontextmanager
import uuid

from sqlalchemy import create_engine, desc, and_, func, text, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import redis.asyncio as redis

from ..config.settings import settings
from ..models.database import Base, LyricsAnalysisResult, UserSession

logger = logging.getLogger(__name__)

class DatabaseService:
    """
    Database service for lyrics analysis history and user session management
    Provides persistent storage and retrieval of analysis results
    """
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self.redis_client = None
        self._initialized = False

    async def initialize(self):
        """Initialize database connection and create tables"""
        try:
            logger.info("🔄 Initializing database connection...")
            
            # Create SQLAlchemy engine with connection pooling
            self.engine = create_engine(
                settings.DATABASE_URL,
                pool_pre_ping=True,
                pool_recycle=300,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                echo=settings.DEBUG  # Log SQL queries in debug mode
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Test database connection
            with self.SessionLocal() as session:
                session.execute(text("SELECT 1"))
                session.commit()
            
            # Create all tables
            logger.info("🔄 Creating database tables...")
            Base.metadata.create_all(bind=self.engine)
            
            # Initialize Redis client for caching
            try:
                self.redis_client = redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # Test Redis connection
                await self.redis_client.ping()
                logger.info("✅ Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ Redis cache unavailable: {e}")
                self.redis_client = None
            
            self._initialized = True
            logger.info("✅ Database service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
            
    async def close(self):
        """Close database connections"""
        if self.redis_client:
            await self.redis_client.aclose()
        if self.engine:
            self.engine.dispose()
        self._initialized = False
        logger.info("🔄 Database connections closed")

    @asynccontextmanager
    async def get_session(self):
        """Get database session with context management"""
        if not self._initialized:
            raise RuntimeError("Database service not initialized")
            
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    async def save_analysis_result(
        self,
        session_id: str,
        original_text: str,
        analysis_results: Dict[str, Any],
        processing_time_ms: float,
        hss_features: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
        title: Optional[str] = None
    ) -> str:
        """Save lyrics analysis result to database"""
        try:
            async with self.get_session() as session:
                # Create or get user session
                user_session = session.query(UserSession).filter_by(session_id=session_id).first()
                if not user_session:
                    user_session = UserSession(
                        session_id=session_id,
                        first_seen=datetime.utcnow(),
                        last_accessed=datetime.utcnow()
                    )
                    session.add(user_session)
                    session.flush()  # Get the ID
                else:
                    user_session.last_accessed = datetime.utcnow()
                
                # Generate analysis ID
                analysis_id = f"lyrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
                
                # Generate a display name if none provided
                display_name = title or filename or f"Analysis {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
                
                # Create analysis result record
                analysis_record = LyricsAnalysisResult(
                    analysis_id=analysis_id,
                    user_session=session_id,
                    filename=filename,
                    title=display_name,  # Use the generated display name
                    original_text=original_text,
                    analysis_results=analysis_results,
                    hss_features=hss_features or {},
                    processing_time_ms=int(processing_time_ms),
                    created_at=datetime.utcnow()
                )
                
                session.add(analysis_record)
                session.commit()
                
                logger.info(f"✅ Saved analysis result: {analysis_id} ({display_name})")
                return analysis_id
                
        except Exception as e:
            logger.error(f"❌ Failed to save analysis result: {e}")
            raise

    async def get_analysis_history(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
        search: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Get analysis history for a session with pagination and search"""
        try:
            async with self.get_session() as session:
                # Base query
                query = session.query(LyricsAnalysisResult).filter_by(user_session=session_id)
                
                # Apply search filter - search in title, filename, and original text
                if search:
                    search_term = f"%{search}%"
                    query = query.filter(
                        or_(
                            LyricsAnalysisResult.title.ilike(search_term),
                            LyricsAnalysisResult.filename.ilike(search_term),
                            LyricsAnalysisResult.original_text.ilike(search_term)
                        )
                    )
                
                # Get total count
                total_count = query.count()
                
                # Apply pagination and ordering
                results = query.order_by(desc(LyricsAnalysisResult.created_at)).offset(offset).limit(limit).all()
                
                # Convert to response format
                history_items = []
                for result in results:
                    # Use title if available, otherwise create preview from text
                    display_name = result.title or f"Analysis {result.created_at.strftime('%Y-%m-%d %H:%M')}"
                    preview = result.original_text[:100] + "..." if len(result.original_text) > 100 else result.original_text
                    
                    history_items.append({
                        "id": result.analysis_id,
                        "title": display_name,
                        "filename": result.filename,
                        "analysisDate": result.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "timestamp": result.created_at.isoformat(),
                        "preview": preview,
                        "originalText": result.original_text,
                        "results": result.analysis_results,
                        "hssFeatures": result.hss_features,
                        "processing_time_ms": result.processing_time_ms
                    })
                
                # Get session statistics
                user_session_obj = session.query(UserSession).filter_by(session_id=session_id).first()
                stats = {
                    "total_analyses": total_count,
                    "session_created": user_session_obj.first_seen.isoformat() if user_session_obj else None,
                    "last_accessed": user_session_obj.last_accessed.isoformat() if user_session_obj else None
                }
                
                # Pagination info
                pagination = {
                    "limit": limit,
                    "offset": offset,
                    "total": total_count,
                    "has_more": offset + limit < total_count
                }
                
                return history_items, {"pagination": pagination, "stats": stats}
                
        except Exception as e:
            logger.error(f"❌ Failed to get analysis history: {e}")
            raise

    async def get_analysis_by_id(self, analysis_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """Get specific analysis by ID"""
        try:
            async with self.get_session() as session:
                result = session.query(LyricsAnalysisResult).filter_by(
                    analysis_id=analysis_id,
                    user_session=session_id
                ).first()
                
                if not result:
                    return None
                
                return {
                    "id": result.analysis_id,
                    "analysisDate": result.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp": result.created_at.isoformat(),
                    "originalText": result.original_text,
                    "results": result.analysis_results,
                    "hssFeatures": result.hss_features,
                    "processing_time_ms": result.processing_time_ms
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get analysis by ID: {e}")
            raise

    async def delete_analysis(self, analysis_id: str, session_id: str) -> bool:
        """Delete specific analysis"""
        try:
            async with self.get_session() as session:
                result = session.query(LyricsAnalysisResult).filter_by(
                    analysis_id=analysis_id,
                    user_session=session_id
                ).first()
                
                if not result:
                    return False
                
                session.delete(result)
                session.commit()
                
                logger.info(f"✅ Deleted analysis: {analysis_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to delete analysis: {e}")
            raise

    async def cleanup_old_sessions(self, days_old: int = 30):
        """Clean up old user sessions and associated data"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            async with self.get_session() as session:
                # Delete old analysis results
                old_analyses = session.query(LyricsAnalysisResult).filter(
                    LyricsAnalysisResult.created_at < cutoff_date
                ).delete()
                
                # Delete old user sessions
                old_sessions = session.query(UserSession).filter(
                    UserSession.last_accessed < cutoff_date
                ).delete()
                
                session.commit()
                
                logger.info(f"✅ Cleaned up {old_analyses} old analyses and {old_sessions} old sessions")
                
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old sessions: {e}")
            raise

# Global database service instance
db_service = DatabaseService() 