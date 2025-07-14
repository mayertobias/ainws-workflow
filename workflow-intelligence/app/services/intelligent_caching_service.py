"""
Intelligent Caching Service

Extracted from intelligence_service.py to preserve sophisticated caching,
metrics tracking, and performance optimization within the new agent architecture.
"""

import logging
import json
import hashlib
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import redis
from ..config.settings import settings

logger = logging.getLogger(__name__)

class IntelligentCachingService:
    """
    Professional caching and metrics service for AI analysis results.
    
    This service preserves the sophisticated caching and performance tracking
    capabilities from the original intelligence service.
    """
    
    def __init__(self):
        # Initialize Redis for caching
        try:
            self.redis_client = redis.from_url(settings.REDIS_URL)
            self.redis_client.ping()
            logger.info("Redis connection established for intelligent caching")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Analysis results will not be cached.")
            self.redis_client = None
        
        # Service metrics tracking
        self.metrics = {
            'total_analyses_cached': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0.0,
            'provider_usage': {},
            'analysis_type_counts': {},
            'agent_performance': {},
            'errors': 0,
            'successful_analyses': 0
        }
        
        # Cache settings
        self.default_cache_ttl = 7200  # 2 hours
        self.cache_prefix = "hss_ai_analysis:"
        
        logger.info("IntelligentCachingService initialized with metrics tracking")
    
    def generate_cache_key(self, request_data: Dict[str, Any]) -> str:
        """
        Generate intelligent cache key based on request content.
        
        This preserves the sophisticated cache key generation from intelligence_service.py
        """
        try:
            # Create a deterministic hash of the request
            cache_components = []
            
            # Add song metadata
            if "song_metadata" in request_data:
                metadata = request_data["song_metadata"]
                cache_components.extend([
                    metadata.get("title", ""),
                    metadata.get("artist", ""),
                    metadata.get("genre", "")
                ])
            
            # Add audio features (sorted for consistency)
            if "audio_analysis" in request_data or "audio_features" in request_data:
                audio_data = request_data.get("audio_analysis", request_data.get("audio_features", {}))
                if isinstance(audio_data, dict):
                    # Sort keys for consistent hashing
                    sorted_audio = sorted(audio_data.items())
                    cache_components.append(str(sorted_audio))
            
            # Add content features
            if "content_analysis" in request_data or "lyrics_analysis" in request_data:
                content_data = request_data.get("content_analysis", request_data.get("lyrics_analysis", {}))
                if isinstance(content_data, dict):
                    sorted_content = sorted(content_data.items())
                    cache_components.append(str(sorted_content))
            
            # Add analysis parameters
            if "analysis_types" in request_data:
                cache_components.append(str(sorted(request_data["analysis_types"])))
            
            # Create hash
            cache_string = "|".join(str(comp) for comp in cache_components)
            cache_hash = hashlib.md5(cache_string.encode()).hexdigest()
            
            return f"{self.cache_prefix}{cache_hash}"
            
        except Exception as e:
            logger.warning(f"Failed to generate cache key: {e}")
            # Fallback to timestamp-based key (no caching benefit)
            return f"{self.cache_prefix}fallback_{int(time.time())}"
    
    async def get_cached_analysis(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached analysis result if available"""
        if not self.redis_client:
            return None
        
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                self.metrics['cache_hits'] += 1
                logger.info(f"Cache hit for key: {cache_key[:20]}...")
                
                # Parse and return cached result
                cached_result = json.loads(cached_data.decode())
                
                # Add cache metadata
                cached_result["cached"] = True
                cached_result["cache_timestamp"] = cached_result.get("timestamp", datetime.utcnow().isoformat())
                
                return cached_result
            else:
                self.metrics['cache_misses'] += 1
                return None
                
        except Exception as e:
            logger.warning(f"Error retrieving cached analysis: {e}")
            self.metrics['cache_misses'] += 1
            return None
    
    async def cache_analysis_result(self, cache_key: str, analysis_result: Dict[str, Any], 
                                  ttl_seconds: Optional[int] = None) -> bool:
        """Cache analysis result with intelligent TTL"""
        if not self.redis_client:
            return False
        
        try:
            ttl = ttl_seconds or self.default_cache_ttl
            
            # Add cache metadata
            cache_data = analysis_result.copy()
            cache_data.update({
                "cached_at": datetime.utcnow().isoformat(),
                "cache_ttl": ttl,
                "cache_key": cache_key[:20] + "..."  # Truncated for logging
            })
            
            # Store in Redis
            success = self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(cache_data, default=str)
            )
            
            if success:
                self.metrics['total_analyses_cached'] += 1
                logger.info(f"Cached analysis result for {ttl}s: {cache_key[:20]}...")
                return True
            else:
                logger.warning(f"Failed to cache analysis result: {cache_key[:20]}...")
                return False
                
        except Exception as e:
            logger.error(f"Error caching analysis result: {e}")
            return False
    
    def update_performance_metrics(self, analysis_data: Dict[str, Any]):
        """Update comprehensive performance metrics"""
        try:
            # Processing time tracking
            if "processing_time_ms" in analysis_data:
                self.metrics['total_processing_time'] += analysis_data["processing_time_ms"]
            
            # Agent performance tracking
            if "agents_used" in analysis_data:
                for agent in analysis_data["agents_used"]:
                    if agent not in self.metrics['agent_performance']:
                        self.metrics['agent_performance'][agent] = {
                            'total_calls': 0,
                            'total_time': 0.0,
                            'success_rate': 0.0,
                            'errors': 0
                        }
                    self.metrics['agent_performance'][agent]['total_calls'] += 1
            
            # Provider usage tracking
            if "provider_used" in analysis_data:
                provider = analysis_data["provider_used"]
                self.metrics['provider_usage'][provider] = \
                    self.metrics['provider_usage'].get(provider, 0) + 1
            
            # Success/error tracking
            if "error" in analysis_data:
                self.metrics['errors'] += 1
            else:
                self.metrics['successful_analyses'] += 1
            
            # Analysis type tracking
            if "analysis_types_completed" in analysis_data:
                for analysis_type in analysis_data["analysis_types_completed"]:
                    type_key = str(analysis_type)
                    self.metrics['analysis_type_counts'][type_key] = \
                        self.metrics['analysis_type_counts'].get(type_key, 0) + 1
            
        except Exception as e:
            logger.warning(f"Error updating performance metrics: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            total_analyses = self.metrics['successful_analyses'] + self.metrics['errors']
            
            report = {
                "total_analyses": total_analyses,
                "successful_analyses": self.metrics['successful_analyses'],
                "error_rate": self.metrics['errors'] / total_analyses if total_analyses > 0 else 0.0,
                "cache_performance": {
                    "cache_hits": self.metrics['cache_hits'],
                    "cache_misses": self.metrics['cache_misses'],
                    "hit_rate": self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses']) 
                               if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0.0,
                    "total_cached": self.metrics['total_analyses_cached']
                },
                "performance_metrics": {
                    "total_processing_time_ms": self.metrics['total_processing_time'],
                    "average_processing_time_ms": self.metrics['total_processing_time'] / total_analyses 
                                                if total_analyses > 0 else 0.0
                },
                "provider_usage": self.metrics['provider_usage'],
                "analysis_type_distribution": self.metrics['analysis_type_counts'],
                "agent_performance": self.metrics['agent_performance'],
                "report_timestamp": datetime.utcnow().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
    
    async def cleanup_expired_cache(self) -> Dict[str, Any]:
        """Clean up expired cache entries"""
        if not self.redis_client:
            return {"status": "no_redis", "cleaned": 0}
        
        try:
            # Get all cache keys
            pattern = f"{self.cache_prefix}*"
            keys = self.redis_client.keys(pattern)
            
            cleaned_count = 0
            for key in keys:
                ttl = self.redis_client.ttl(key)
                if ttl == -2:  # Key has expired
                    self.redis_client.delete(key)
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} expired cache entries")
            return {
                "status": "success",
                "total_keys_checked": len(keys),
                "expired_keys_cleaned": cleaned_count,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            return {"status": "error", "error": str(e)}
    
    def reset_metrics(self):
        """Reset performance metrics (useful for testing or periodic resets)"""
        self.metrics = {
            'total_analyses_cached': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0.0,
            'provider_usage': {},
            'analysis_type_counts': {},
            'agent_performance': {},
            'errors': 0,
            'successful_analyses': 0
        }
        logger.info("Performance metrics reset")