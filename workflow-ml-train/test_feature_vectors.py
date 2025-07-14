#!/usr/bin/env python3
"""
Test Pre-existing Feature Vector Support

Tests feature vector files, presets, and pipeline integration.
"""

import asyncio
import logging
import sys
import json
from pathlib import Path

# Add app to path
sys.path.append('./app')

from app.utils.feature_vector_manager import feature_vector_manager
from app.pipeline.orchestrator import PipelineOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_feature_vectors():
    """Test pre-existing feature vector functionality"""
    logger.info("🧪 Testing Pre-existing Feature Vector Support...")
    
    try:
        # Test 1: Create default presets
        logger.info("1️⃣ Creating default presets...")
        feature_vector_manager.create_default_presets()
        presets = feature_vector_manager.list_available_presets()
        logger.info(f"✅ Created {len(presets)} presets")
        
        # Test 2: Load and validate preset
        logger.info("2️⃣ Loading preset...")
        if presets:
            preset_name = presets[0]["name"]
            preset = feature_vector_manager.load_preset(preset_name)
            validation = feature_vector_manager.validate_feature_vector(preset)
            logger.info(f"✅ Loaded preset '{preset_name}': {validation['summary']}")
        
        # Test 3: Test with pipeline
        logger.info("3️⃣ Testing with pipeline...")
        orchestrator = PipelineOrchestrator()
        
        test_vector = {
            "strategy": "audio_only",
            "selected_features": ["audio_basic_energy", "audio_basic_valence"]
        }
        
        parameters = {
            "feature_vector": test_vector,
            "skip_feature_agreement": True
        }
        
        result = await orchestrator.start_pipeline(
            pipeline_id="test_fv_123",
            strategy="audio_only", 
            experiment_name="test_feature_vector",
            parameters=parameters
        )
        
        logger.info(f"✅ Pipeline started with feature vector: {result['status']}")
        
        # Check results
        await asyncio.sleep(1)
        status = orchestrator.get_pipeline_status("test_fv_123")
        if status and status.get("feature_agreement"):
            mode = status["feature_agreement"].get("mode", "unknown")
            feature_count = status["feature_agreement"].get("feature_count", 0)
            logger.info(f"✅ Feature agreement: {mode} mode, {feature_count} features")
        
        logger.info("🎉 Pre-existing Feature Vector Support works!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_feature_vectors())
    if result:
        logger.info("✅ All tests passed!")
    else:
        logger.error("❌ Tests failed!")
        sys.exit(1) 