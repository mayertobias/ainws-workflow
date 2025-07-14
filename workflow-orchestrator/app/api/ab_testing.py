"""
Enhanced A/B Testing API endpoints for workflow orchestrator
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse

from ..models.ab_testing import (
    ABTestConfiguration, ABTestExecution, ABTestRequest, ABTestResponse,
    ABTestListResponse, ABTestPredictionRequest, ABTestPredictionResponse,
    ABTestStatus, ABTestSample, ABTestStatistics
)
from ..services.ab_testing_service import EnhancedABTestingService

logger = logging.getLogger(__name__)

router = APIRouter()

# Global A/B testing service instance
ab_testing_service = None

async def get_ab_testing_service() -> EnhancedABTestingService:
    """Dependency to get A/B testing service instance."""
    global ab_testing_service
    if ab_testing_service is None:
        ab_testing_service = EnhancedABTestingService()
        await ab_testing_service.start_background_tasks()
    return ab_testing_service

@router.post("/tests", response_model=ABTestResponse)
async def create_ab_test(
    request: ABTestRequest,
    service: EnhancedABTestingService = Depends(get_ab_testing_service)
):
    """Create a new A/B test."""
    try:
        logger.info(f"Creating A/B test: {request.configuration.name}")
        
        test_execution = await service.create_test(request.configuration)
        
        # Start immediately if requested
        if request.start_immediately:
            test_execution = await service.start_test(test_execution.test_id)
        
        return ABTestResponse(
            test_id=test_execution.test_id,
            status=test_execution.status,
            message=f"A/B test created successfully"
        )
        
    except ValueError as e:
        logger.error(f"Invalid A/B test configuration: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tests", response_model=ABTestListResponse)
async def list_ab_tests(
    status: Optional[ABTestStatus] = Query(None, description="Filter by status"),
    service: EnhancedABTestingService = Depends(get_ab_testing_service)
):
    """List all A/B tests."""
    try:
        tests = await service.list_tests()
        
        # Filter by status if specified
        if status:
            tests = [test for test in tests if test.status == status]
        
        active_count = len([test for test in tests if test.status == ABTestStatus.RUNNING])
        
        return ABTestListResponse(
            tests=tests,
            total_count=len(tests),
            active_count=active_count
        )
        
    except Exception as e:
        logger.error(f"Error listing A/B tests: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tests/{test_id}", response_model=ABTestExecution)
async def get_ab_test(
    test_id: str,
    service: EnhancedABTestingService = Depends(get_ab_testing_service)
):
    """Get A/B test details."""
    try:
        test = await service.get_test_status(test_id)
        if not test:
            raise HTTPException(status_code=404, detail=f"Test not found: {test_id}")
        
        return test
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting A/B test {test_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tests/{test_id}/start", response_model=ABTestResponse)
async def start_ab_test(
    test_id: str,
    service: EnhancedABTestingService = Depends(get_ab_testing_service)
):
    """Start an A/B test."""
    try:
        test_execution = await service.start_test(test_id)
        
        return ABTestResponse(
            test_id=test_execution.test_id,
            status=test_execution.status,
            message="A/B test started successfully"
        )
        
    except ValueError as e:
        logger.error(f"Cannot start A/B test {test_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting A/B test {test_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tests/{test_id}/stop", response_model=ABTestResponse)
async def stop_ab_test(
    test_id: str,
    reason: str = Query("Manual stop", description="Reason for stopping test"),
    service: EnhancedABTestingService = Depends(get_ab_testing_service)
):
    """Stop an A/B test."""
    try:
        test_execution = await service.stop_test(test_id, reason)
        
        return ABTestResponse(
            test_id=test_execution.test_id,
            status=test_execution.status,
            message=f"A/B test stopped: {reason}"
        )
        
    except ValueError as e:
        logger.error(f"Cannot stop A/B test {test_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error stopping A/B test {test_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tests/{test_id}/predict", response_model=ABTestPredictionResponse)
async def ab_test_prediction(
    test_id: str,
    request: ABTestPredictionRequest,
    service: EnhancedABTestingService = Depends(get_ab_testing_service)
):
    """Make a prediction through A/B test routing."""
    try:
        # Override test_id from URL
        request.test_id = test_id
        
        response = await service.route_prediction_request(request)
        return response
        
    except ValueError as e:
        logger.error(f"Invalid A/B test prediction request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error making A/B test prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tests/{test_id}/statistics", response_model=List[ABTestStatistics])
async def get_ab_test_statistics(
    test_id: str,
    service: EnhancedABTestingService = Depends(get_ab_testing_service)
):
    """Get statistical analysis results for A/B test."""
    try:
        test = await service.get_test_status(test_id)
        if not test:
            raise HTTPException(status_code=404, detail=f"Test not found: {test_id}")
        
        return test.current_statistics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting A/B test statistics {test_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tests/{test_id}/samples", response_model=List[ABTestSample])
async def get_ab_test_samples(
    test_id: str,
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of samples to return"),
    offset: int = Query(0, ge=0, description="Number of samples to skip"),
    service: EnhancedABTestingService = Depends(get_ab_testing_service)
):
    """Get samples from A/B test."""
    try:
        samples = await service.get_test_samples(test_id)
        
        # Apply pagination
        paginated_samples = samples[offset:offset + limit]
        
        return paginated_samples
        
    except Exception as e:
        logger.error(f"Error getting A/B test samples {test_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tests/{test_id}/export")
async def export_ab_test_results(
    test_id: str,
    format: str = Query("json", regex="^(json|csv)$", description="Export format"),
    service: EnhancedABTestingService = Depends(get_ab_testing_service)
):
    """Export A/B test results."""
    try:
        test = await service.get_test_status(test_id)
        if not test:
            raise HTTPException(status_code=404, detail=f"Test not found: {test_id}")
        
        samples = await service.get_test_samples(test_id)
        
        if format == "json":
            export_data = {
                "test_configuration": test.configuration.dict(),
                "test_execution": test.dict(),
                "samples": [sample.dict() for sample in samples],
                "statistics": [stat.dict() for stat in test.current_statistics]
            }
            
            return JSONResponse(
                content=export_data,
                headers={"Content-Disposition": f"attachment; filename=ab_test_{test_id}.json"}
            )
        
        elif format == "csv":
            # Convert to CSV format
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write headers
            if samples:
                headers = ["sample_id", "variant_id", "timestamp"] + list(samples[0].metrics.keys())
                writer.writerow(headers)
                
                # Write data
                for sample in samples:
                    row = [
                        sample.sample_id,
                        sample.variant_id,
                        sample.timestamp.isoformat()
                    ] + list(sample.metrics.values())
                    writer.writerow(row)
            
            csv_content = output.getvalue()
            output.close()
            
            return JSONResponse(
                content={"csv_data": csv_content},
                headers={"Content-Disposition": f"attachment; filename=ab_test_{test_id}.csv"}
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting A/B test results {test_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def ab_testing_health():
    """Health check for A/B testing service."""
    try:
        return {
            "status": "healthy",
            "service": "ab-testing",
            "version": "2.0.0",
            "features": ["statistical_analysis", "real_time_monitoring", "automated_winner_detection"],
            "timestamp": "2025-01-15T10:30:00Z"
        }
    except Exception as e:
        logger.error(f"A/B testing health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced endpoints for better analytics
@router.get("/tests/{test_id}/analytics")
async def get_test_analytics(
    test_id: str,
    service: EnhancedABTestingService = Depends(get_ab_testing_service)
):
    """Get comprehensive analytics for A/B test."""
    try:
        test = await service.get_test_status(test_id)
        if not test:
            raise HTTPException(status_code=404, detail=f"Test not found: {test_id}")
        
        samples = await service.get_test_samples(test_id)
        
        # Calculate enhanced analytics
        analytics = {
            "test_overview": {
                "test_id": test_id,
                "name": test.configuration.name,
                "status": test.status,
                "duration_hours": (
                    (test.completed_at or test.started_at or test.created_at) - 
                    (test.started_at or test.created_at)
                ).total_seconds() / 3600 if test.started_at else 0,
                "total_samples": test.total_samples,
                "conversion_rate": test.total_samples / max(1, test.total_samples) * 100
            },
            "variant_performance": {},
            "statistical_results": [stat.dict() for stat in test.current_statistics],
            "timeline": []
        }
        
        # Per-variant analytics
        for variant_id, sample_count in test.variant_samples.items():
            variant = next((v for v in test.configuration.variants if v.variant_id == variant_id), None)
            if variant:
                variant_samples = [s for s in samples if s.variant_id == variant_id]
                
                analytics["variant_performance"][variant_id] = {
                    "name": variant.name,
                    "model_id": variant.model_id,
                    "sample_count": sample_count,
                    "traffic_percentage": variant.traffic_percentage,
                    "avg_metrics": {}
                }
                
                # Calculate average metrics
                if variant_samples:
                    for metric_name in variant_samples[0].metrics.keys():
                        metric_values = [s.metrics[metric_name] for s in variant_samples if metric_name in s.metrics]
                        if metric_values:
                            analytics["variant_performance"][variant_id]["avg_metrics"][metric_name] = {
                                "mean": sum(metric_values) / len(metric_values),
                                "count": len(metric_values)
                            }
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analytics for test {test_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tests/{test_id}/simulate")
async def simulate_test_traffic(
    test_id: str,
    num_samples: int = Query(100, ge=1, le=10000, description="Number of samples to simulate"),
    service: EnhancedABTestingService = Depends(get_ab_testing_service)
):
    """Simulate traffic for A/B test (for testing purposes)."""
    try:
        test = await service.get_test_status(test_id)
        if not test:
            raise HTTPException(status_code=404, detail=f"Test not found: {test_id}")
        
        if test.status != ABTestStatus.RUNNING:
            raise HTTPException(status_code=400, detail="Test must be running to simulate traffic")
        
        import random
        
        # Simulate predictions
        results = []
        for i in range(num_samples):
            # Create mock prediction request
            request = ABTestPredictionRequest(
                test_id=test_id,
                input_features={
                    "audio_energy": random.uniform(0.3, 0.9),
                    "audio_valence": random.uniform(0.2, 0.8),
                    "audio_tempo": random.uniform(80, 160)
                },
                user_id=f"sim_user_{i}",
                session_id=f"sim_session_{i // 10}",
                metadata={"simulated": True}
            )
            
            # Route through A/B test
            response = await service.route_prediction_request(request)
            results.append(response)
        
        return {
            "message": f"Simulated {num_samples} samples for test {test_id}",
            "samples_generated": len(results),
            "variant_distribution": {
                variant_id: len([r for r in results if r.variant_id == variant_id])
                for variant_id in test.variant_samples.keys()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error simulating traffic for test {test_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))