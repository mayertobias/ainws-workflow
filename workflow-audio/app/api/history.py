"""
History API endpoints for workflow-audio service
Handles audio analysis history retrieval and management
"""

from fastapi import APIRouter, HTTPException, Query, Header
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ..services.database_service import db_service
from ..models.database_models import AnalysisStatus

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/audio/history")
async def get_audio_history(
    limit: int = Query(50, ge=1, le=100, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    search: Optional[str] = Query(None, description="Search query for filename"),
    analysis_type: Optional[str] = Query(None, description="Filter by analysis type (basic/comprehensive)"),
    status: Optional[str] = Query(None, description="Filter by status (completed/failed/processing)"),
    user_session: Optional[str] = Header(None, alias="X-Session-ID", description="User session ID")
):
    """
    Get audio analysis history for a user session
    """
    try:
        if not user_session:
            raise HTTPException(status_code=400, detail="X-Session-ID header is required")
        
        # Use the database service directly
        if not db_service._initialized:
            raise HTTPException(status_code=503, detail="Database service not available")
        
        # Get history from database
        history_items, total_count = await db_service.get_audio_analysis_history(
            session_id=user_session,
            limit=limit,
            offset=offset,
            search=search,
            analysis_type=analysis_type,
            status=status
        )
        
        # Calculate pagination info
        has_next = (offset + limit) < total_count
        has_prev = offset > 0
        
        return {
            "status": "success",
            "data": {
                "items": history_items,
                "pagination": {
                    "total": total_count,
                    "limit": limit,
                    "offset": offset,
                    "has_next": has_next,
                    "has_prev": has_prev,
                    "page": (offset // limit) + 1,
                    "total_pages": ((total_count - 1) // limit) + 1 if total_count > 0 else 0
                },
                "stats": {
                    "total_analyses": total_count,
                    "completed": len([item for item in history_items if item.get("status") == "completed"]),
                    "failed": len([item for item in history_items if item.get("status") == "failed"]),
                    "processing": len([item for item in history_items if item.get("status") == "processing"])
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting audio history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history")

@router.get("/audio/history/{analysis_id}")
async def get_analysis_by_id(
    analysis_id: str,
    user_session: Optional[str] = Header(None, alias="X-Session-ID", description="User session ID")
):
    """
    Get a specific audio analysis by ID
    """
    try:
        if not user_session:
            raise HTTPException(status_code=400, detail="X-Session-ID header is required")
        
        # Use the database service directly
        if not db_service._initialized:
            raise HTTPException(status_code=503, detail="Database service not available")
        
        # Get analysis from database
        analysis = await db_service.get_analysis_result(analysis_id)
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Verify session ownership (basic security check)
        # Note: This would need to be implemented in the database service
        # For now, we'll return the analysis if it exists
        
        return {
            "status": "success",
            "data": analysis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis by ID: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis")

@router.delete("/audio/history/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    user_session: Optional[str] = Header(None, alias="X-Session-ID", description="User session ID")
):
    """
    Delete a specific audio analysis
    """
    try:
        if not user_session:
            raise HTTPException(status_code=400, detail="X-Session-ID header is required")
        
        # Use the database service directly
        if not db_service._initialized:
            raise HTTPException(status_code=503, detail="Database service not available")
        
        # Delete analysis from database
        success = await db_service.delete_analysis_result(analysis_id, user_session)
        
        if not success:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return {
            "status": "success",
            "message": "Analysis deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete analysis")

@router.get("/audio/stats")
async def get_user_stats(
    user_session: Optional[str] = Header(None, alias="X-Session-ID", description="User session ID")
):
    """
    Get analysis statistics for a user session
    """
    try:
        if not user_session:
            raise HTTPException(status_code=400, detail="X-Session-ID header is required")
        
        # Use the database service directly
        if not db_service._initialized:
            raise HTTPException(status_code=503, detail="Database service not available")
        
        # Get stats from database
        stats = await db_service.get_user_analysis_stats(user_session)
        
        return {
            "status": "success",
            "data": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

@router.get("/audio/workflows")
async def get_user_workflows(
    user_session: Optional[str] = Header(None, alias="X-Session-ID", description="User session ID"),
    limit: int = Query(20, ge=1, le=100, description="Number of workflows to return")
):
    """
    Get workflows for a user session
    """
    try:
        if not user_session:
            raise HTTPException(status_code=400, detail="X-Session-ID header is required")
        
        # Use the database service directly
        if not db_service._initialized:
            raise HTTPException(status_code=503, detail="Database service not available")
        
        # Get workflows from database
        workflows = await db_service.get_user_workflows(user_session, limit)
        
        return {
            "status": "success",
            "data": {
                "workflows": workflows,
                "total": len(workflows)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting user workflows: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve workflows") 