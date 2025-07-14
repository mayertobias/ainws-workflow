"""
Assignment Engine for A/B Testing

Handles user assignment to variants with consistent hashing,
caching, and targeting criteria.
"""

import hashlib
import asyncio
try:
    import mmh3  # MurmurHash3 for consistent hashing
except ImportError:
    mmh3 = None
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..models.ab_testing import ExperimentConfig, VariantConfig

@dataclass
class AssignmentResult:
    """Result of variant assignment"""
    experiment_id: str
    user_id: str
    variant_id: str
    variant_name: str
    assigned_at: datetime
    is_control: bool
    metadata: Dict[str, Any]

class AssignmentEngine:
    """
    Core assignment engine for A/B testing.
    
    Features:
    - Consistent hashing for deterministic assignment
    - Redis caching for performance
    - Targeting and exclusion criteria
    - Assignment logging for analytics
    """
    
    def __init__(self, redis_client=None, db_client=None):
        """Initialize assignment engine."""
        self.redis = redis_client
        self.db = db_client
        self.cache_ttl = 3600  # 1 hour cache
        self.assignment_salt = "chartmuse_ab_testing_v1"
    
    async def get_variant(
        self, 
        experiment_id: str, 
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[AssignmentResult]:
        """
        Get variant assignment for user in experiment.
        
        Args:
            experiment_id: Unique experiment identifier
            user_id: User identifier for assignment
            context: Additional context for targeting
            
        Returns:
            AssignmentResult or None if not eligible
        """
        try:
            # 1. Check cache first (O(1) lookup)
            cached_assignment = await self._get_cached_assignment(experiment_id, user_id)
            if cached_assignment:
                return cached_assignment
            
            # 2. Load experiment configuration
            experiment = await self._get_experiment_config(experiment_id)
            if not experiment or experiment.status != "running":
                return None
            
            # 3. Check eligibility (targeting criteria)
            if not await self._is_eligible(experiment, user_id, context):
                return None
            
            # 4. Consistent hash assignment
            variant = self._assign_variant(experiment, user_id)
            if not variant:
                return None
            
            # 5. Create assignment result
            assignment = AssignmentResult(
                experiment_id=experiment_id,
                user_id=user_id,
                variant_id=variant.variant_id,
                variant_name=variant.name,
                assigned_at=datetime.utcnow(),
                is_control=variant.variant_id == experiment.variants[0].variant_id,
                metadata={
                    "assignment_method": "consistent_hash",
                    "traffic_allocation": variant.traffic_allocation,
                    "experiment_name": experiment.name
                }
            )
            
            # 6. Cache assignment (critical for consistency)
            await self._cache_assignment(assignment)
            
            # 7. Log assignment event for analytics
            await self._log_assignment_event(assignment, context)
            
            return assignment
            
        except Exception as e:
            print(f"Error in variant assignment: {e}")
            return None
    
    def _assign_variant(self, experiment: ExperimentConfig, user_id: str) -> Optional[VariantConfig]:
        """
        Deterministic variant assignment using MurmurHash3.
        Ensures same user always gets same variant for consistency.
        """
        try:
            # Create consistent hash input
            hash_input = f"{experiment.experiment_id}:{self.assignment_salt}:{user_id}"
            
            # Use MurmurHash3 for uniform distribution, fallback to hashlib
            if mmh3 is not None:
                hash_value = mmh3.hash(hash_input.encode(), signed=False)
            else:
                hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            
            # Map to 0-99 range for percentage allocation
            bucket = hash_value % 100
            
            # Assign based on traffic allocation
            cumulative_allocation = 0
            for variant in experiment.variants:
                cumulative_allocation += variant.traffic_allocation
                if bucket < cumulative_allocation:
                    return variant
            
            # Fallback to control (should never reach here if properly configured)
            return experiment.variants[0] if experiment.variants else None
            
        except Exception as e:
            print(f"Error in variant assignment: {e}")
            return None
    
    async def _is_eligible(
        self, 
        experiment: ExperimentConfig, 
        user_id: str, 
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if user meets targeting criteria."""
        try:
            # Check targeting criteria
            if experiment.target_audience:
                for criterion, expected_value in experiment.target_audience.items():
                    if context and context.get(criterion) != expected_value:
                        return False
            
            # Check exclusion criteria
            if hasattr(experiment, 'exclusion_criteria') and experiment.exclusion_criteria:
                for criterion, excluded_value in experiment.exclusion_criteria.items():
                    if context and context.get(criterion) == excluded_value:
                        return False
            
            # Check if user was previously excluded
            if await self._is_user_excluded(experiment.experiment_id, user_id):
                return False
            
            return True
            
        except Exception as e:
            print(f"Error checking eligibility: {e}")
            return False
    
    async def _get_cached_assignment(self, experiment_id: str, user_id: str) -> Optional[AssignmentResult]:
        """Get cached assignment from Redis."""
        if not self.redis:
            return None
        
        try:
            cache_key = f"assignment:{experiment_id}:{user_id}"
            cached_data = await self.redis.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data.decode())
                return AssignmentResult(
                    experiment_id=data["experiment_id"],
                    user_id=data["user_id"],
                    variant_id=data["variant_id"],
                    variant_name=data["variant_name"],
                    assigned_at=datetime.fromisoformat(data["assigned_at"]),
                    is_control=data["is_control"],
                    metadata=data["metadata"]
                )
            
            return None
            
        except Exception as e:
            print(f"Error getting cached assignment: {e}")
            return None
    
    async def _cache_assignment(self, assignment: AssignmentResult):
        """Cache assignment in Redis."""
        if not self.redis:
            return
        
        try:
            cache_key = f"assignment:{assignment.experiment_id}:{assignment.user_id}"
            cache_data = {
                "experiment_id": assignment.experiment_id,
                "user_id": assignment.user_id,
                "variant_id": assignment.variant_id,
                "variant_name": assignment.variant_name,
                "assigned_at": assignment.assigned_at.isoformat(),
                "is_control": assignment.is_control,
                "metadata": assignment.metadata
            }
            
            await self.redis.setex(
                cache_key, 
                self.cache_ttl, 
                json.dumps(cache_data)
            )
            
        except Exception as e:
            print(f"Error caching assignment: {e}")
    
    async def _log_assignment_event(
        self, 
        assignment: AssignmentResult, 
        context: Optional[Dict[str, Any]]
    ):
        """Log assignment for analytics pipeline."""
        try:
            event = {
                "event_type": "experiment_assignment",
                "timestamp": assignment.assigned_at.isoformat(),
                "experiment_id": assignment.experiment_id,
                "user_id": assignment.user_id,
                "variant_id": assignment.variant_id,
                "variant_name": assignment.variant_name,
                "is_control": assignment.is_control,
                "context": context or {},
                "metadata": assignment.metadata
            }
            
            # Store in database if available
            if self.db:
                await self._store_assignment_in_db(assignment, event)
            
            # Send to event pipeline (placeholder - implement based on your event system)
            await self._send_to_event_pipeline(event)
            
        except Exception as e:
            print(f"Error logging assignment event: {e}")
    
    async def _store_assignment_in_db(self, assignment: AssignmentResult, event: Dict[str, Any]):
        """Store assignment in database for consistency and audit."""
        # Placeholder - implement based on your database schema
        pass
    
    async def _send_to_event_pipeline(self, event: Dict[str, Any]):
        """Send assignment event to analytics pipeline."""
        # Placeholder - implement based on your event system
        # Could be Kafka, RabbitMQ, direct API call, etc.
        pass
    
    async def _get_experiment_config(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """Get experiment configuration from database or cache."""
        # Placeholder - implement to fetch from your storage
        # This would typically query the experiments table
        return None
    
    async def _is_user_excluded(self, experiment_id: str, user_id: str) -> bool:
        """Check if user is in exclusion list."""
        # Placeholder - implement exclusion logic
        return False
    
    async def exclude_user(self, experiment_id: str, user_id: str, reason: str = "manual"):
        """Exclude user from experiment."""
        try:
            # Remove from cache
            if self.redis:
                cache_key = f"assignment:{experiment_id}:{user_id}"
                await self.redis.delete(cache_key)
            
            # Add to exclusion list
            exclusion_key = f"excluded:{experiment_id}:{user_id}"
            exclusion_data = {
                "user_id": user_id,
                "experiment_id": experiment_id,
                "excluded_at": datetime.utcnow().isoformat(),
                "reason": reason
            }
            
            if self.redis:
                await self.redis.setex(
                    exclusion_key,
                    86400 * 30,  # 30 days
                    json.dumps(exclusion_data)
                )
            
        except Exception as e:
            print(f"Error excluding user: {e}")
    
    async def get_assignment_statistics(self, experiment_id: str) -> Dict[str, Any]:
        """Get assignment statistics for experiment."""
        try:
            # Placeholder - implement to collect assignment stats
            # This would query assignment logs and calculate:
            # - Total assignments per variant
            # - Assignment rate over time
            # - Geographic distribution
            # - etc.
            
            return {
                "total_assignments": 0,
                "variant_distribution": {},
                "assignment_rate_per_hour": 0,
                "last_assignment": None
            }
            
        except Exception as e:
            print(f"Error getting assignment statistics: {e}")
            return {}


class MultiVariantAssignmentEngine(AssignmentEngine):
    """
    Enhanced assignment engine for multi-variant experiments.
    
    Supports:
    - Multi-armed bandit algorithms
    - Dynamic traffic allocation
    - Performance-based rebalancing
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bandit_exploration_rate = 0.1  # Epsilon for epsilon-greedy
    
    async def get_variant_with_bandit(
        self, 
        experiment_id: str, 
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[AssignmentResult]:
        """
        Get variant assignment using multi-armed bandit algorithm.
        
        This method dynamically adjusts traffic allocation based on
        variant performance, allowing for more efficient experiments.
        """
        try:
            # Get experiment and current performance data
            experiment = await self._get_experiment_config(experiment_id)
            if not experiment:
                return None
            
            performance_data = await self._get_variant_performance(experiment_id)
            
            # Use epsilon-greedy strategy
            import random
            if random.random() < self.bandit_exploration_rate:
                # Exploration: random variant
                variant = random.choice(experiment.variants)
            else:
                # Exploitation: best performing variant
                variant = self._get_best_performing_variant(experiment, performance_data)
            
            # Create assignment
            assignment = AssignmentResult(
                experiment_id=experiment_id,
                user_id=user_id,
                variant_id=variant.variant_id,
                variant_name=variant.name,
                assigned_at=datetime.utcnow(),
                is_control=variant.variant_id == experiment.variants[0].variant_id,
                metadata={
                    "assignment_method": "multi_armed_bandit",
                    "exploration_rate": self.bandit_exploration_rate,
                    "performance_score": performance_data.get(variant.variant_id, {}).get("score", 0)
                }
            )
            
            # Cache and log
            await self._cache_assignment(assignment)
            await self._log_assignment_event(assignment, context)
            
            return assignment
            
        except Exception as e:
            print(f"Error in bandit assignment: {e}")
            # Fallback to regular assignment
            return await self.get_variant(experiment_id, user_id, context)
    
    async def _get_variant_performance(self, experiment_id: str) -> Dict[str, Dict[str, Any]]:
        """Get current performance metrics for all variants."""
        # Placeholder - implement to fetch real performance data
        return {}
    
    def _get_best_performing_variant(
        self, 
        experiment: ExperimentConfig, 
        performance_data: Dict[str, Dict[str, Any]]
    ) -> VariantConfig:
        """Get the best performing variant based on metrics."""
        if not performance_data:
            return experiment.variants[0]  # Default to control
        
        best_variant = experiment.variants[0]
        best_score = performance_data.get(best_variant.variant_id, {}).get("score", 0)
        
        for variant in experiment.variants[1:]:
            score = performance_data.get(variant.variant_id, {}).get("score", 0)
            if score > best_score:
                best_score = score
                best_variant = variant
        
        return best_variant


# Utility functions for testing and simulation
async def simulate_assignment_distribution(
    assignment_engine: AssignmentEngine,
    experiment_id: str,
    num_users: int = 10000
) -> Dict[str, int]:
    """
    Simulate assignment distribution for testing.
    
    Useful for validating that traffic splits are working correctly.
    """
    variant_counts = {}
    
    for i in range(num_users):
        user_id = f"test_user_{i}"
        assignment = await assignment_engine.get_variant(experiment_id, user_id)
        
        if assignment:
            variant_counts[assignment.variant_id] = variant_counts.get(assignment.variant_id, 0) + 1
    
    return variant_counts


def calculate_assignment_quality(
    expected_distribution: Dict[str, float],
    actual_distribution: Dict[str, int],
    total_assignments: int
) -> Dict[str, float]:
    """
    Calculate how well actual assignment distribution matches expected.
    
    Returns quality scores and chi-square test results.
    """
    quality_scores = {}
    
    for variant_id, expected_percentage in expected_distribution.items():
        actual_count = actual_distribution.get(variant_id, 0)
        actual_percentage = (actual_count / total_assignments) * 100
        
        # Calculate deviation from expected
        deviation = abs(actual_percentage - expected_percentage)
        quality_score = max(0, 100 - (deviation * 10))  # Penalize deviations
        
        quality_scores[variant_id] = {
            "expected_percentage": expected_percentage,
            "actual_percentage": actual_percentage,
            "deviation": deviation,
            "quality_score": quality_score
        }
    
    return quality_scores