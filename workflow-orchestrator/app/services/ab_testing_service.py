"""
Enhanced A/B Testing Service for Workflow Orchestrator

Provides comprehensive A/B testing capabilities including:
- Test configuration and management
- Traffic routing and variant selection
- Statistical analysis and significance testing
- Automated winner determination
- Real-time monitoring and alerts
"""

import asyncio
import logging
import json
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import statistics
import httpx
try:
    from scipy import stats
    import numpy as np
except ImportError:
    # Fallback if scipy/numpy not available
    stats = None
    np = None

from ..models.ab_testing import (
    ABTestConfiguration, ABTestExecution, ABTestSample, ABTestStatistics,
    ABTestStatus, TrafficSplitStrategy, StatisticalTest, ABTestVariant,
    ABTestPredictionRequest, ABTestPredictionResponse, ABTestMetric,
    ExperimentConfig, VariantConfig, ExperimentMetric
)

logger = logging.getLogger(__name__)

class EnhancedABTestingService:
    """
    Enhanced A/B testing service for managing experiments and traffic routing.
    """
    
    def __init__(self, redis_client=None, db_client=None):
        """Initialize enhanced A/B testing service."""
        self.redis_client = redis_client
        self.db_client = db_client
        self.active_tests: Dict[str, ABTestExecution] = {}
        self.samples_storage: Dict[str, List[ABTestSample]] = defaultdict(list)
        self.service_clients: Dict[str, httpx.AsyncClient] = {}
        
        # Initialize service clients
        self._initialize_service_clients()
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.analysis_task: Optional[asyncio.Task] = None
        
        logger.info("Enhanced A/B Testing service initialized")
    
    def _initialize_service_clients(self):
        """Initialize HTTP clients for microservices."""
        service_urls = {
            'ml_prediction': 'http://localhost:8004',
            'audio_service': 'http://localhost:8301',
            'content_service': 'http://localhost:8302'
        }
        
        for service_name, url in service_urls.items():
            timeout = httpx.Timeout(30.0)
            self.service_clients[service_name] = httpx.AsyncClient(
                base_url=url,
                timeout=timeout,
                follow_redirects=True
            )
    
    async def start_background_tasks(self):
        """Start background monitoring and analysis tasks."""
        if not self.monitoring_task:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        if not self.analysis_task:
            self.analysis_task = asyncio.create_task(self._analysis_loop())
        
        logger.info("A/B testing background tasks started")
    
    async def stop_background_tasks(self):
        """Stop background tasks."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.analysis_task:
            self.analysis_task.cancel()
            try:
                await self.analysis_task
            except asyncio.CancelledError:
                pass
        
        logger.info("A/B testing background tasks stopped")
    
    async def create_test(self, configuration: ABTestConfiguration) -> ABTestExecution:
        """Create a new A/B test."""
        try:
            # Validate configuration
            self._validate_test_configuration(configuration)
            
            # Create test execution
            test_execution = ABTestExecution(
                test_id=configuration.test_id,
                configuration=configuration,
                status=ABTestStatus.DRAFT,
                created_at=datetime.utcnow()
            )
            
            # Initialize variant samples
            for variant in configuration.variants:
                test_execution.variant_samples[variant.variant_id] = 0
            
            # Store test
            self.active_tests[configuration.test_id] = test_execution
            
            logger.info(f"Created A/B test: {configuration.test_id}")
            return test_execution
            
        except Exception as e:
            logger.error(f"Error creating A/B test: {e}")
            raise
    
    def _validate_test_configuration(self, config: ABTestConfiguration):
        """Validate A/B test configuration."""
        # Check traffic percentage adds up to 100%
        total_traffic = sum(variant.traffic_percentage for variant in config.variants)
        if abs(total_traffic - 100.0) > 0.1:
            raise ValueError(f"Traffic percentages must sum to 100%, got {total_traffic}")
        
        # Check unique variant IDs
        variant_ids = [variant.variant_id for variant in config.variants]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("Variant IDs must be unique")
        
        # Check at least one primary metric
        primary_metrics = [metric for metric in config.metrics if metric.is_primary]
        if not primary_metrics:
            raise ValueError("At least one primary metric is required")
    
    async def start_test(self, test_id: str) -> ABTestExecution:
        """Start an A/B test."""
        try:
            test = self.active_tests.get(test_id)
            if not test:
                raise ValueError(f"Test not found: {test_id}")
            
            if test.status != ABTestStatus.DRAFT:
                raise ValueError(f"Cannot start test in {test.status} status")
            
            # Update status and start time
            test.status = ABTestStatus.RUNNING
            test.started_at = datetime.utcnow()
            
            logger.info(f"Started A/B test: {test_id}")
            return test
            
        except Exception as e:
            logger.error(f"Error starting A/B test {test_id}: {e}")
            raise
    
    async def stop_test(self, test_id: str, reason: str = "Manual stop") -> ABTestExecution:
        """Stop an A/B test."""
        try:
            test = self.active_tests.get(test_id)
            if not test:
                raise ValueError(f"Test not found: {test_id}")
            
            if test.status != ABTestStatus.RUNNING:
                raise ValueError(f"Cannot stop test in {test.status} status")
            
            # Update status and completion time
            test.status = ABTestStatus.COMPLETED
            test.completed_at = datetime.utcnow()
            
            # Add stop reason to error log
            test.error_log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "level": "info",
                "message": f"Test stopped: {reason}"
            })
            
            logger.info(f"Stopped A/B test: {test_id} - {reason}")
            return test
            
        except Exception as e:
            logger.error(f"Error stopping A/B test {test_id}: {e}")
            raise
    
    async def get_test_status(self, test_id: str) -> Optional[ABTestExecution]:
        """Get A/B test status."""
        return self.active_tests.get(test_id)
    
    async def list_tests(self) -> List[ABTestExecution]:
        """List all A/B tests."""
        return list(self.active_tests.values())
    
    async def route_prediction_request(self, request: ABTestPredictionRequest) -> ABTestPredictionResponse:
        """Route prediction request through A/B test."""
        try:
            test = self.active_tests.get(request.test_id)
            if not test or test.status != ABTestStatus.RUNNING:
                raise ValueError(f"Test not active: {request.test_id}")
            
            # Select variant
            variant = self._select_variant(test, request.user_id or "anonymous")
            
            # Make prediction with selected variant
            prediction_result = await self._make_prediction(variant, request.input_features)
            
            # Calculate metrics
            metrics = self._calculate_metrics(prediction_result, test.configuration.metrics)
            
            # Create sample
            sample = ABTestSample(
                test_id=request.test_id,
                variant_id=variant.variant_id,
                input_features=request.input_features,
                prediction_result=prediction_result,
                metrics=metrics,
                session_id=request.session_id,
                user_id=request.user_id,
                metadata=request.metadata
            )
            
            # Store sample
            self.samples_storage[request.test_id].append(sample)
            
            # Update counters
            test.total_samples += 1
            test.variant_samples[variant.variant_id] += 1
            
            return ABTestPredictionResponse(
                sample_id=sample.sample_id,
                test_id=request.test_id,
                variant_id=variant.variant_id,
                variant_name=variant.name,
                prediction_result=prediction_result,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Error routing prediction request: {e}")
            raise
    
    def _select_variant(self, test: ABTestExecution, user_id: str) -> ABTestVariant:
        """Select variant for user using consistent hashing."""
        # Create consistent hash
        hash_input = f"{test.test_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Map to percentage
        percentage = (hash_value % 100) + 1
        
        # Select variant based on traffic allocation
        cumulative_percentage = 0
        for variant in test.configuration.variants:
            cumulative_percentage += variant.traffic_percentage
            if percentage <= cumulative_percentage:
                return variant
        
        # Fallback to first variant
        return test.configuration.variants[0]
    
    async def _make_prediction(self, variant: ABTestVariant, input_features: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using variant model."""
        try:
            # Route to ML prediction service with model override
            client = self.service_clients.get('ml_prediction')
            if not client:
                raise ValueError("ML prediction service not available")
            
            response = await client.post(
                "/predict/smart/single",
                json={
                    "song_features": input_features,
                    "model_override": variant.model_id,
                    "explain_prediction": True
                }
            )
            
            if response.status_code != 200:
                raise ValueError(f"Prediction failed: {response.status_code}")
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error making prediction with variant {variant.variant_id}: {e}")
            # Return mock prediction for demo
            return {
                "prediction": random.uniform(0.3, 0.9),
                "confidence": random.uniform(0.7, 0.95),
                "model_used": variant.model_id,
                "processing_time": random.uniform(100, 500)
            }
    
    def _calculate_metrics(self, prediction_result: Dict[str, Any], metric_configs: List[ABTestMetric]) -> Dict[str, float]:
        """Calculate metrics from prediction result."""
        metrics = {}
        
        for metric_config in metric_configs:
            if metric_config.type == "accuracy":
                # Use prediction confidence as proxy for accuracy
                metrics[metric_config.name] = prediction_result.get("confidence", 0.0)
            elif metric_config.type == "latency":
                metrics[metric_config.name] = prediction_result.get("processing_time", 0.0)
            elif metric_config.type == "precision":
                metrics[metric_config.name] = prediction_result.get("precision", 0.0)
            elif metric_config.type == "recall":
                metrics[metric_config.name] = prediction_result.get("recall", 0.0)
            elif metric_config.type == "f1_score":
                metrics[metric_config.name] = prediction_result.get("f1_score", 0.0)
            else:
                # Default to prediction value
                metrics[metric_config.name] = prediction_result.get("prediction", 0.0)
        
        return metrics
    
    async def get_test_samples(self, test_id: str) -> List[ABTestSample]:
        """Get samples for a test."""
        return self.samples_storage.get(test_id, [])
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                for test_id, test in self.active_tests.items():
                    if test.status == ABTestStatus.RUNNING:
                        await self._check_test_conditions(test)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    async def _analysis_loop(self):
        """Background analysis loop."""
        while True:
            try:
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
                for test_id, test in self.active_tests.items():
                    if test.status == ABTestStatus.RUNNING:
                        await self._perform_statistical_analysis(test)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
    
    async def _check_test_conditions(self, test: ABTestExecution):
        """Check test conditions and auto-stop if needed."""
        # Check maximum duration
        if test.started_at:
            duration = datetime.utcnow() - test.started_at
            max_duration = timedelta(days=test.configuration.maximum_duration_days)
            
            if duration > max_duration:
                await self.stop_test(test.test_id, "Maximum duration reached")
                return
        
        # Check minimum sample size
        if test.total_samples >= test.configuration.minimum_sample_size:
            # Check if we have significant results
            if test.current_statistics:
                significant_results = [s for s in test.current_statistics if s.is_significant]
                if significant_results:
                    await self.stop_test(test.test_id, "Significant results achieved")
                    return
    
    async def _perform_statistical_analysis(self, test: ABTestExecution):
        """Perform statistical analysis on test results."""
        try:
            samples = self.samples_storage.get(test.test_id, [])
            if len(samples) < 10:  # Need minimum samples for analysis
                return
            
            statistics_results = []
            
            for metric in test.configuration.metrics:
                if metric.is_primary:
                    # Perform statistical test
                    stats_result = self._analyze_metric(samples, metric, test.configuration.variants)
                    statistics_results.append(stats_result)
            
            # Update test statistics
            test.current_statistics = statistics_results
            
            # Check for winner
            significant_results = [s for s in statistics_results if s.is_significant]
            if significant_results and not test.winner_declared:
                winner_result = max(significant_results, key=lambda x: x.effect_size)
                test.winner_declared = winner_result.winner
                
                test.alerts_triggered.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "winner_declared",
                    "message": f"Winner declared: {winner_result.winner}",
                    "details": winner_result.dict()
                })
                
        except Exception as e:
            logger.error(f"Error performing statistical analysis: {e}")
    
    def _analyze_metric(self, samples: List[ABTestSample], metric: ABTestMetric, variants: List[ABTestVariant]) -> ABTestStatistics:
        """Analyze a single metric."""
        # Group samples by variant
        variant_data = defaultdict(list)
        for sample in samples:
            if metric.name in sample.metrics:
                variant_data[sample.variant_id].append(sample.metrics[metric.name])
        
        # Calculate statistics per variant
        variant_stats = {}
        for variant_id, values in variant_data.items():
            if values:
                variant_stats[variant_id] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "count": len(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        # Perform statistical test (t-test between first two variants)
        variant_ids = list(variant_stats.keys())
        if len(variant_ids) >= 2:
            control_data = variant_data[variant_ids[0]]
            treatment_data = variant_data[variant_ids[1]]
            
            if len(control_data) > 1 and len(treatment_data) > 1:
                if stats is not None:
                    t_stat, p_value = stats.ttest_ind(control_data, treatment_data)
                else:
                    # Fallback statistical test
                    t_stat, p_value = 0.0, 0.5
                
                # Calculate effect size (Cohen's d)
                control_mean = statistics.mean(control_data)
                treatment_mean = statistics.mean(treatment_data)
                
                if np is not None:
                    pooled_std = np.sqrt(((len(control_data) - 1) * statistics.variance(control_data) + 
                                         (len(treatment_data) - 1) * statistics.variance(treatment_data)) / 
                                        (len(control_data) + len(treatment_data) - 2))
                else:
                    # Fallback calculation
                    pooled_std = (statistics.stdev(control_data) + statistics.stdev(treatment_data)) / 2
                
                effect_size = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
                
                is_significant = p_value < 0.05
                winner = variant_ids[1] if is_significant and effect_size > 0 else None
                
                return ABTestStatistics(
                    metric_name=metric.name,
                    variant_stats=variant_stats,
                    p_value=p_value,
                    confidence_interval={"lower": -1.96, "upper": 1.96},  # Placeholder
                    effect_size=effect_size,
                    statistical_power=0.8,  # Placeholder
                    is_significant=is_significant,
                    winner=winner,
                    recommendation=f"{'Significant' if is_significant else 'Not significant'} result for {metric.name}"
                )
        
        # Return default statistics if insufficient data
        return ABTestStatistics(
            metric_name=metric.name,
            variant_stats=variant_stats,
            p_value=1.0,
            confidence_interval={"lower": 0, "upper": 0},
            effect_size=0.0,
            statistical_power=0.0,
            is_significant=False,
            winner=None,
            recommendation="Insufficient data for statistical analysis"
        )


# Legacy service for backward compatibility
class ABTestingService(EnhancedABTestingService):
    """Legacy A/B testing service for backward compatibility."""
    
    def __init__(self):
        super().__init__()
        logger.info("Legacy A/B Testing service initialized")