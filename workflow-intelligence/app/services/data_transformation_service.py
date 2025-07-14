"""
Data Transformation Service for Layer 2 Backend Validation

This service provides robust data transformation and validation to ensure
UI-friendly data structures are consistently provided to the frontend.
"""

import logging
from typing import Dict, Any, Optional, List
from pydantic import ValidationError

from ..models.intelligence import (
    AIInsightResult, InsightGenerationResponse, InsightStatus,
    MusicalMeaningInsight, HitComparisonInsight, NoveltyAssessmentInsight,
    ProductionFeedback, StrategicInsights, AnalysisType
)

logger = logging.getLogger(__name__)

class DataTransformationService:
    """
    Handles data transformation and validation for Layer 2 defense.
    
    This service:
    1. Validates all data structures before API responses
    2. Transforms data to be UI-friendly
    3. Provides fallback values for missing data
    4. Ensures consistent data types and structures
    """
    
    def __init__(self):
        """Initialize the data transformation service."""
        self.transformation_rules = {
            "string_fields": self._ensure_string_not_empty,
            "list_fields": self._ensure_list_not_empty,
            "numeric_fields": self._ensure_numeric_valid,
            "dict_fields": self._ensure_dict_not_empty
        }
        
        logger.info("Data transformation service initialized")
    
    def transform_insight_result_for_ui(self, insight_result: AIInsightResult) -> Dict[str, Any]:
        """
        Transform AIInsightResult for UI consumption.
        
        Args:
            insight_result: The AI insight result to transform
            
        Returns:
            UI-friendly dictionary with guaranteed structure
        """
        try:
            # Start with the base model conversion
            ui_data = insight_result.dict()
            
            # Apply transformations for UI compatibility
            ui_data = self._apply_ui_transformations(ui_data)
            
            # Add computed fields for UI
            ui_data = self._add_computed_ui_fields(ui_data)
            
            # Validate final structure
            self._validate_ui_structure(ui_data)
            
            logger.info("Successfully transformed insight result for UI")
            return ui_data
            
        except Exception as e:
            logger.error(f"Error transforming insight result: {e}")
            return self._create_fallback_ui_data(insight_result)
    
    def transform_insight_response_for_ui(self, response: InsightGenerationResponse) -> Dict[str, Any]:
        """
        Transform InsightGenerationResponse for UI consumption.
        
        Args:
            response: The insight generation response to transform
            
        Returns:
            UI-friendly dictionary with guaranteed structure
        """
        try:
            # Convert to dict
            ui_data = response.dict()
            
            # Transform the nested insights if present
            if response.insights:
                ui_data["insights"] = self.transform_insight_result_for_ui(response.insights)
            
            # Add UI-specific metadata
            ui_data = self._add_response_metadata(ui_data)
            
            logger.info("Successfully transformed insight response for UI")
            return ui_data
            
        except Exception as e:
            logger.error(f"Error transforming insight response: {e}")
            return self._create_fallback_response_data(response)
    
    def _apply_ui_transformations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply UI-specific transformations to data.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Transformed data dictionary
        """
        # Ensure all text fields are non-empty strings
        text_fields = [
            "executive_summary", "agent_used", "provider_used"
        ]
        
        for field in text_fields:
            if field in data:
                data[field] = self._ensure_string_not_empty(data[field])
        
        # Ensure all list fields are non-empty lists
        list_fields = [
            "key_recommendations", "analysis_types_completed"
        ]
        
        for field in list_fields:
            if field in data:
                data[field] = self._ensure_list_not_empty(data[field])
        
        # Ensure numeric fields are valid
        numeric_fields = [
            "hit_potential_score", "hit_confidence", "confidence_score", "processing_time_ms"
        ]
        
        for field in numeric_fields:
            if field in data:
                data[field] = self._ensure_numeric_valid(data[field])
        
        # Transform nested insight objects
        insight_fields = [
            "musical_meaning", "hit_comparison", "novelty_assessment",
            "production_feedback", "strategic_insights"
        ]
        
        for field in insight_fields:
            if field in data and data[field]:
                data[field] = self._transform_insight_object(data[field])
        
        return data
    
    def _add_computed_ui_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add computed fields for UI enhancement.
        
        Args:
            data: Base data dictionary
            
        Returns:
            Data with computed UI fields
        """
        # Add hit potential percentage
        if "hit_potential_score" in data and data["hit_potential_score"] is not None:
            score = data["hit_potential_score"]
            data["hit_potential_percentage"] = f"{score * 100:.1f}%"
            data["hit_potential_color"] = self._get_score_color(score)
        
        # Add confidence percentage
        if "confidence_score" in data and data["confidence_score"] is not None:
            score = data["confidence_score"]
            data["confidence_percentage"] = f"{score * 100:.0f}%"
            data["confidence_color"] = self._get_score_color(score)
        
        # Add processing time display
        if "processing_time_ms" in data and data["processing_time_ms"] is not None:
            time_ms = data["processing_time_ms"]
            if time_ms < 1000:
                data["processing_time_display"] = f"{time_ms:.0f}ms"
            else:
                data["processing_time_display"] = f"{time_ms / 1000:.1f}s"
        
        # Add analysis completeness
        completed_types = data.get("analysis_types_completed", [])
        data["analysis_completeness"] = len(completed_types)
        data["has_comprehensive_analysis"] = len(completed_types) >= 3
        
        return data
    
    def _add_response_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add response-level metadata for UI.
        
        Args:
            data: Response data dictionary
            
        Returns:
            Data with response metadata
        """
        # Add status display
        status = data.get("status", "unknown")
        data["status_display"] = status.replace("_", " ").title()
        data["status_color"] = self._get_status_color(status)
        
        # Add success indicator
        data["is_successful"] = status == InsightStatus.COMPLETED.value
        data["is_cached"] = data.get("cached", False)
        
        # Add error handling
        if "error_message" in data and data["error_message"]:
            data["has_error"] = True
            data["error_display"] = data["error_message"]
        else:
            data["has_error"] = False
            data["error_display"] = None
        
        return data
    
    def _transform_insight_object(self, insight_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform individual insight objects for UI.
        
        Args:
            insight_obj: Insight object dictionary
            
        Returns:
            Transformed insight object
        """
        if not insight_obj:
            return {}
        
        # Ensure all string fields are non-empty
        for key, value in insight_obj.items():
            if isinstance(value, str):
                insight_obj[key] = self._ensure_string_not_empty(value)
            elif isinstance(value, list):
                insight_obj[key] = self._ensure_list_not_empty(value)
            elif isinstance(value, (int, float)):
                insight_obj[key] = self._ensure_numeric_valid(value)
        
        return insight_obj
    
    def _ensure_string_not_empty(self, value: Any) -> str:
        """Ensure string value is not empty."""
        if value is None:
            return "Not available"
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else "Not available"
        return str(value) if value else "Not available"
    
    def _ensure_list_not_empty(self, value: Any) -> List[Any]:
        """Ensure list value is not empty."""
        if value is None:
            return []
        if isinstance(value, list):
            return value if value else []
        return [value] if value else []
    
    def _ensure_numeric_valid(self, value: Any) -> float:
        """Ensure numeric value is valid."""
        if value is None:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _ensure_dict_not_empty(self, value: Any) -> Dict[str, Any]:
        """Ensure dict value is not empty."""
        if value is None:
            return {}
        if isinstance(value, dict):
            return value if value else {}
        return {}
    
    def _get_score_color(self, score: float) -> str:
        """Get color class for score display."""
        if score >= 0.8:
            return "text-green-600"
        elif score >= 0.6:
            return "text-yellow-600"
        else:
            return "text-red-600"
    
    def _get_status_color(self, status: str) -> str:
        """Get color class for status display."""
        if status == "completed":
            return "text-green-600"
        elif status == "processing":
            return "text-blue-600"
        elif status == "failed":
            return "text-red-600"
        else:
            return "text-gray-600"
    
    def _validate_ui_structure(self, data: Dict[str, Any]) -> None:
        """
        Validate the final UI data structure.
        
        Args:
            data: Data to validate
            
        Raises:
            ValidationError: If data structure is invalid
        """
        required_fields = [
            "hit_potential_score", "confidence_score", "executive_summary",
            "key_recommendations", "analysis_types_completed"
        ]
        
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field: {field}")
                # Add default values
                if field == "hit_potential_score":
                    data[field] = 0.0
                elif field == "confidence_score":
                    data[field] = 0.0
                elif field == "executive_summary":
                    data[field] = "Analysis completed"
                elif field in ["key_recommendations", "analysis_types_completed"]:
                    data[field] = []
    
    def _create_fallback_ui_data(self, original_result: AIInsightResult) -> Dict[str, Any]:
        """
        Create fallback UI data when transformation fails.
        
        Args:
            original_result: Original insight result
            
        Returns:
            Fallback UI data
        """
        return {
            "hit_potential_score": 0.0,
            "hit_potential_percentage": "0.0%",
            "hit_potential_color": "text-gray-600",
            "confidence_score": 0.0,
            "confidence_percentage": "0%",
            "confidence_color": "text-gray-600",
            "executive_summary": "Analysis completed with limited results",
            "key_recommendations": ["Complete analysis recommended"],
            "analysis_types_completed": [],
            "analysis_completeness": 0,
            "has_comprehensive_analysis": False,
            "processing_time_ms": 0.0,
            "processing_time_display": "0ms",
            "agent_used": "fallback",
            "provider_used": "fallback",
            "model_used": "fallback",
            "musical_meaning": None,
            "hit_comparison": None,
            "novelty_assessment": None,
            "production_feedback": None,
            "strategic_insights": None,
            "raw_outputs": None,
            "token_usage": None
        }
    
    def _create_fallback_response_data(self, original_response: InsightGenerationResponse) -> Dict[str, Any]:
        """
        Create fallback response data when transformation fails.
        
        Args:
            original_response: Original response
            
        Returns:
            Fallback response data
        """
        return {
            "status": "completed",
            "status_display": "Completed",
            "status_color": "text-green-600",
            "insight_id": original_response.insight_id,
            "insights": self._create_fallback_ui_data(original_response.insights) if original_response.insights else None,
            "error_message": None,
            "error_display": None,
            "has_error": False,
            "timestamp": original_response.timestamp.isoformat(),
            "cached": False,
            "is_successful": True,
            "is_cached": False,
            "hit_potential_score": 0.0,
            "hit_confidence": 0.0,
            "model_used": "fallback"
        }