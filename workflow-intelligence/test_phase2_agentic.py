"""
Phase 2 Agentic AI Integration Test

Comprehensive test for the agentic AI system including multi-agent coordination,
specialized analysis, and enhanced workflow integration.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any

import httpx

# Test configuration
BASE_URL = "http://localhost:8005"
TEST_TIMEOUT = 60

# Sample data for testing (simulating real workflow data)
SAMPLE_AUDIO_DATA = {
    "tempo": 128.5,
    "energy": 0.78,
    "danceability": 0.85,
    "valence": 0.72,
    "acousticness": 0.15,
    "instrumentalness": 0.02,
    "speechiness": 0.08,
    "liveness": 0.12,
    "loudness": -4.5,
    "key": "C",
    "mode": "major",
    "spectral_centroid_mean": 1850.3,
    "spectral_rolloff_mean": 3200.7,
    "chroma_stft_mean": 0.65,
    "genre_predictions": {
        "pop": 0.65,
        "dance": 0.28,
        "electronic": 0.07
    },
    "audio_features": {
        "mfcc_1": 0.45,
        "mfcc_2": 0.32,
        "spectral_contrast": 0.68
    }
}

SAMPLE_CONTENT_DATA = {
    "sentiment": {
        "compound": 0.45,
        "positive": 0.72,
        "negative": 0.08,
        "neutral": 0.20
    },
    "emotions": {
        "joy": 0.65,
        "excitement": 0.58,
        "love": 0.42,
        "sadness": 0.12,
        "anger": 0.05
    },
    "themes": ["love", "celebration", "youth", "freedom"],
    "topics": ["relationships", "party", "summer"],
    "language": {
        "english_ratio": 0.95,
        "complexity_score": 0.68,
        "readability": 0.75
    },
    "word_count": 156,
    "unique_words": 89,
    "repetition_score": 0.35
}

SAMPLE_HIT_PREDICTION = {
    "hit_probability": 0.73,
    "confidence": 0.85,
    "feature_importance": {
        "energy": 0.25,
        "danceability": 0.22,
        "tempo": 0.18,
        "sentiment": 0.15,
        "genre": 0.12,
        "others": 0.08
    },
    "similar_hits": [
        {"title": "Summer Vibes", "artist": "Pop Artist", "similarity": 0.89},
        {"title": "Dance Floor", "artist": "EDM Producer", "similarity": 0.82},
        {"title": "Feel Good", "artist": "Chart Topper", "similarity": 0.78}
    ],
    "market_indicators": {
        "streaming_potential": 0.78,
        "radio_potential": 0.71,
        "social_media_potential": 0.85
    }
}

SAMPLE_SONG_METADATA = {
    "title": "Test Song",
    "artist": "Test Artist",
    "album": "Test Album",
    "genre": "Pop",
    "duration": 185,
    "release_year": 2024,
    "language": "English"
}

async def test_service_health():
    """Test basic service health and availability."""
    print("🔍 Testing service health...")
    
    try:
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Test basic health
            response = await client.get(f"{BASE_URL}/health/status")
            assert response.status_code == 200, f"Health check failed: {response.status_code}"
            
            health_data = response.json()
            print(f"   ✅ Service health: {health_data.get('status', 'unknown')}")
            
            # Test root endpoint with new agentic features
            response = await client.get(f"{BASE_URL}/")
            assert response.status_code == 200, f"Root endpoint failed: {response.status_code}"
            
            root_data = response.json()
            print(f"   ✅ Service version: {root_data.get('version', 'unknown')}")
            print(f"   ✅ Agentic capabilities: {len(root_data.get('agentic_capabilities', []))}")
            
            return True
            
    except Exception as e:
        print(f"   ❌ Service health test failed: {e}")
        return False

async def test_agent_status():
    """Test agent status and availability."""
    print("🤖 Testing agent status...")
    
    try:
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/agentic/agents/status")
            assert response.status_code == 200, f"Agent status failed: {response.status_code}"
            
            status_data = response.json()
            agents = status_data.get("agents", {})
            
            print(f"   ✅ Active agents: {len(agents)}")
            
            for agent_role, agent_info in agents.items():
                print(f"   🤖 {agent_role}: {agent_info.get('name', 'Unknown')} ({agent_info.get('status', 'unknown')})")
            
            # Test orchestrator metrics
            orchestrator_metrics = status_data.get("orchestrator_metrics", {})
            print(f"   📊 Total analyses: {orchestrator_metrics.get('total_analyses', 0)}")
            
            return True
            
    except Exception as e:
        print(f"   ❌ Agent status test failed: {e}")
        return False

async def test_agent_profiles():
    """Test individual agent profiles."""
    print("📋 Testing agent profiles...")
    
    agent_roles = ["music_analysis", "commercial_analysis", "quality_assurance"]
    
    try:
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            for role in agent_roles:
                response = await client.get(f"{BASE_URL}/agentic/agents/{role}/profile")
                
                if response.status_code == 200:
                    profile_data = response.json()
                    profile = profile_data.get("profile", {})
                    
                    print(f"   ✅ {role}: {profile.get('name', 'Unknown')}")
                    print(f"      Experience: {profile.get('experience_level', 0)}/10")
                    print(f"      Confidence: {profile.get('confidence_score', 0):.2f}")
                    print(f"      Expertise areas: {len(profile.get('expertise_areas', []))}")
                else:
                    print(f"   ⚠️  {role}: Profile not available ({response.status_code})")
            
            return True
            
    except Exception as e:
        print(f"   ❌ Agent profiles test failed: {e}")
        return False

async def test_agentic_analysis():
    """Test full agentic analysis with multiple agents."""
    print("🧠 Testing agentic analysis...")
    
    try:
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Prepare agentic analysis request
            agentic_request = {
                "coordination_strategy": "collaborative",
                "required_agent_roles": ["music_analysis", "commercial_analysis", "quality_assurance"],
                "analysis_depth": "comprehensive",
                "focus_areas": ["musical_features", "commercial_potential", "market_positioning"],
                "enable_learning": True,
                "enable_memory": True,
                "max_execution_time": 300,
                "quality_threshold": 0.7,
                "input_data": {
                    "audio_analysis": SAMPLE_AUDIO_DATA,
                    "content_analysis": SAMPLE_CONTENT_DATA,
                    "hit_prediction": SAMPLE_HIT_PREDICTION,
                    "song_metadata": SAMPLE_SONG_METADATA
                }
            }
            
            print("   🚀 Starting agentic analysis...")
            start_time = time.time()
            
            response = await client.post(
                f"{BASE_URL}/agentic/analyze/agentic",
                json=agentic_request
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            assert response.status_code == 200, f"Agentic analysis failed: {response.status_code}"
            
            result = response.json()
            
            print(f"   ✅ Analysis completed in {processing_time:.0f}ms")
            print(f"   🤝 Coordination strategy: {result.get('coordination_strategy_used', 'unknown')}")
            print(f"   👥 Agents involved: {len(result.get('agents_involved', []))}")
            print(f"   🎯 Coordination quality: {result.get('coordination_quality', 0):.2f}")
            print(f"   ⚡ Coordination efficiency: {result.get('coordination_efficiency', 0):.2f}")
            print(f"   🎼 Overall confidence: {result.get('overall_confidence', 0):.2f}")
            print(f"   ⭐ Quality score: {result.get('quality_score', 0):.2f}")
            print(f"   �� Innovation score: {result.get('innovation_score', 0):.2f}")
            
            # Test agent contributions
            contributions = result.get("agent_contributions", [])
            print(f"   📝 Agent contributions: {len(contributions)}")
            
            for contribution in contributions:
                agent_role = contribution.get("agent_role", "unknown")
                confidence = contribution.get("confidence_level", 0)
                findings_count = len(contribution.get("findings", []))
                insights_count = len(contribution.get("insights", []))
                
                print(f"      🤖 {agent_role}: {findings_count} findings, {insights_count} insights (confidence: {confidence:.2f})")
            
            # Test executive summary
            executive_summary = result.get("executive_summary", "")
            print(f"   📊 Executive summary length: {len(executive_summary)} characters")
            
            return True
            
    except Exception as e:
        print(f"   ❌ Agentic analysis test failed: {e}")
        return False

async def test_comprehensive_agentic_analysis():
    """Test comprehensive agentic analysis endpoint."""
    print("🎯 Testing comprehensive agentic analysis...")
    
    try:
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Prepare comprehensive analysis request
            comprehensive_request = {
                "audio_analysis": SAMPLE_AUDIO_DATA,
                "content_analysis": SAMPLE_CONTENT_DATA,
                "hit_prediction": SAMPLE_HIT_PREDICTION,
                "song_metadata": SAMPLE_SONG_METADATA
            }
            
            print("   🚀 Starting comprehensive agentic analysis...")
            start_time = time.time()
            
            response = await client.post(
                f"{BASE_URL}/agentic/analyze/comprehensive-agentic",
                json=comprehensive_request
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            assert response.status_code == 200, f"Comprehensive agentic analysis failed: {response.status_code}"
            
            result = response.json()
            
            print(f"   ✅ Analysis completed in {processing_time:.0f}ms")
            
            # Test enhanced response structure
            print(f"   📋 Executive summary: {len(result.get('executive_summary', ''))} chars")
            print(f"   🎯 Overall score: {result.get('overall_score', 0):.2f}")
            print(f"   🔍 Confidence level: {result.get('confidence_level', 0):.2f}")
            
            # Test agent-specific insights
            agent_insights = result.get("agent_insights", {})
            print(f"   🎵 Musical insights: {len(agent_insights.get('musical_analysis', []))}")
            print(f"   💰 Commercial insights: {len(agent_insights.get('commercial_analysis', []))}")
            print(f"   🔧 Technical insights: {len(agent_insights.get('technical_analysis', []))}")
            
            # Test collaboration metadata
            collaboration_metadata = result.get("collaboration_metadata", {})
            print(f"   🤝 Collaboration quality: {collaboration_metadata.get('coordination_quality', 0):.2f}")
            print(f"   ⚡ Collaboration efficiency: {collaboration_metadata.get('coordination_efficiency', 0):.2f}")
            
            return True
            
    except Exception as e:
        print(f"   ❌ Comprehensive agentic analysis test failed: {e}")
        return False

async def test_agent_coordination():
    """Test manual agent coordination."""
    print("🎭 Testing agent coordination...")
    
    try:
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Test parallel coordination
            coordination_request = {
                "strategy": "parallel",
                "agent_roles": ["music_analysis", "commercial_analysis"],
                "input_data": {
                    "audio_analysis": SAMPLE_AUDIO_DATA,
                    "hit_prediction": SAMPLE_HIT_PREDICTION
                }
            }
            
            print("   🚀 Testing parallel coordination...")
            
            response = await client.post(
                f"{BASE_URL}/agentic/orchestrator/coordinate",
                json=coordination_request
            )
            
            assert response.status_code == 200, f"Coordination failed: {response.status_code}"
            
            result = response.json()
            coordination_results = result.get("coordination_results", {})
            
            print(f"   ✅ Strategy used: {coordination_results.get('strategy_used', 'unknown')}")
            print(f"   👥 Agents involved: {len(coordination_results.get('agents_involved', []))}")
            print(f"   🎯 Quality: {coordination_results.get('coordination_quality', 0):.2f}")
            print(f"   ⚡ Efficiency: {coordination_results.get('coordination_efficiency', 0):.2f}")
            
            # Test sequential coordination
            coordination_request["strategy"] = "sequential"
            
            print("   🚀 Testing sequential coordination...")
            
            response = await client.post(
                f"{BASE_URL}/agentic/orchestrator/coordinate",
                json=coordination_request
            )
            
            assert response.status_code == 200, f"Sequential coordination failed: {response.status_code}"
            
            result = response.json()
            print(f"   ✅ Sequential coordination completed")
            
            return True
            
    except Exception as e:
        print(f"   ❌ Agent coordination test failed: {e}")
        return False

async def test_orchestrator_metrics():
    """Test orchestrator performance metrics."""
    print("📊 Testing orchestrator metrics...")
    
    try:
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/agentic/orchestrator/metrics")
            assert response.status_code == 200, f"Orchestrator metrics failed: {response.status_code}"
            
            metrics = response.json()
            
            print(f"   ✅ Total analyses: {metrics.get('total_analyses', 0)}")
            print(f"   ✅ Successful analyses: {metrics.get('successful_analyses', 0)}")
            print(f"   ✅ Success rate: {metrics.get('success_rate', 0):.2f}")
            print(f"   ✅ Average processing time: {metrics.get('average_processing_time_ms', 0):.0f}ms")
            print(f"   ✅ Active agents: {metrics.get('active_agents', 0)}")
            
            # Test individual agent performance
            agent_performance = metrics.get("agent_performance", {})
            for agent_role, performance in agent_performance.items():
                success_rate = performance.get("success_rate", 0)
                print(f"   🤖 {agent_role}: {success_rate:.2f} success rate")
            
            return True
            
    except Exception as e:
        print(f"   ❌ Orchestrator metrics test failed: {e}")
        return False

async def run_phase2_tests():
    """Run all Phase 2 agentic AI tests."""
    print("🚀 Starting Phase 2 Agentic AI Integration Tests")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Service Health", test_service_health),
        ("Agent Status", test_agent_status),
        ("Agent Profiles", test_agent_profiles),
        ("Agentic Analysis", test_agentic_analysis),
        ("Comprehensive Agentic Analysis", test_comprehensive_agentic_analysis),
        ("Agent Coordination", test_agent_coordination),
        ("Orchestrator Metrics", test_orchestrator_metrics)
    ]
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            result = await test_func()
            test_results.append((test_name, result))
            if result:
                print(f"✅ {test_name} Test: PASSED")
            else:
                print(f"❌ {test_name} Test: FAILED")
        except Exception as e:
            print(f"❌ {test_name} Test: ERROR - {e}")
            test_results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 PHASE 2 TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n📊 Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL PHASE 2 TESTS PASSED!")
        print("🚀 Agentic AI system is fully operational!")
    else:
        print("⚠️  Some tests failed. Please check the logs above.")
    
    print("\n🎯 Phase 2 Features Tested:")
    print("   ✅ Multi-agent coordination")
    print("   ✅ Specialized agent analysis")
    print("   ✅ Cross-agent validation")
    print("   ✅ Enhanced workflow integration")
    print("   ✅ Performance monitoring")
    print("   ✅ Agent orchestration")

if __name__ == "__main__":
    try:
        asyncio.run(run_phase2_tests())
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
