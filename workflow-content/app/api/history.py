"""
History API endpoints for workflow-content service
Handles lyrics analysis history retrieval and management
"""

from fastapi import APIRouter, HTTPException, Query, Header
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ..services.database_service import db_service
from ..models.responses import ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/lyrics/history")
async def get_lyrics_history(
    limit: int = Query(50, ge=1, le=100, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    search: Optional[str] = Query(None, description="Search query for lyrics text"),
    user_session: Optional[str] = Header(None, alias="X-Session-ID", description="User session ID")
):
    """
    Get lyrics analysis history for a user session
    """
    try:
        if not user_session:
            raise HTTPException(status_code=400, detail="X-Session-ID header is required")
        
        # Get history from database using the correct method
        history_items, response_data = await db_service.get_analysis_history(
            session_id=user_session,
            limit=limit,
            offset=offset,
            search=search
        )
        
        return {
            "status": "success",
            "data": {
                "items": history_items,
                "pagination": response_data["pagination"],
                "stats": response_data["stats"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting lyrics history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history")

@router.get("/lyrics/history/{analysis_id}")
async def get_analysis_by_id(
    analysis_id: str,
    user_session: Optional[str] = Header(None, alias="X-Session-ID", description="User session ID")
):
    """
    Get a specific lyrics analysis by ID
    """
    try:
        if not user_session:
            raise HTTPException(status_code=400, detail="X-Session-ID header is required")
        
        # Get analysis from database
        analysis = await db_service.get_analysis_by_id(analysis_id, user_session)
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return {
            "status": "success",
            "data": analysis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis by ID: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis")

@router.delete("/lyrics/history/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    user_session: Optional[str] = Header(None, alias="X-Session-ID", description="User session ID")
):
    """
    Delete a specific lyrics analysis
    """
    try:
        if not user_session:
            raise HTTPException(status_code=400, detail="X-Session-ID header is required")
        
        # Delete analysis from database
        success = await db_service.delete_analysis(analysis_id, user_session)
        
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

@router.get("/lyrics/stats")
async def get_user_stats(
    user_session: Optional[str] = Header(None, alias="X-Session-ID", description="User session ID")
):
    """
    Get analysis statistics for a user session
    """
    try:
        if not user_session:
            raise HTTPException(status_code=400, detail="X-Session-ID header is required")
        
        # Get stats via the history method
        _, response_data = await db_service.get_analysis_history(
            session_id=user_session,
            limit=1,
            offset=0
        )
        
        return {
            "status": "success",
            "data": response_data["stats"]
        }
        
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics") 