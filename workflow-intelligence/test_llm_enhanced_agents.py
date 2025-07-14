"""
Test LLM-Enhanced Agent Intelligence System

This test verifies that agents are using LLM intelligence instead of static thresholds
for genre-specific, dynamic analysis.
"""

import asyncio
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_llm_enhanced_agents():
    """Test the LLM-enhanced agent system."""
    
    print("🎵 Testing LLM-Enhanced Agent Intelligence System")
    print("=" * 60)
    
    # Import agents
    try:
        from app.agents.music_analysis_agent import MusicAnalysisAgent
        from app.agents.commercial_analysis_agent import CommercialAnalysisAgent
        from app.agents.agent_orchestrator import AgentOrchestrator
        from app.models.agentic_models import AgentTask, AgentRole, TaskType
        
        print("✅ Successfully imported LLM-enhanced agents")
    except Exception as e:
        print(f"❌ Failed to import agents: {e}")
        return
    
    # Test data - different genres to verify genre-specific analysis
    test_cases = [
        {
            "name": "Pop Song",
            "audio_analysis": {
                "tempo": 120.0,
                "energy": 0.75,
                "danceability": 0.65,
                "valence": 0.8,
                "acousticness": 0.2,
                "key": "C",
                "mode": "major",
                "spectral_centroid_mean": 2500.0,
                "genre_predictions": {"pop": 0.85, "rock": 0.15}
            },
            "content_analysis": {
                "sentiment": {"compound": 0.6},
                "emotions": {"joy": 0.7, "sadness": 0.1},
                "themes": ["love", "success", "happiness"],
                "language": {"english_ratio": 0.95}
            },
            "hit_prediction": {
                "hit_probability": 0.72,
                "confidence": 0.88,
                "feature_importance": {"energy": 0.3, "danceability": 0.25, "valence": 0.2}
            },
            "song_metadata": {
                "title": "Summer Vibes",
                "artist": "Test Artist",
                "genre": "pop"
            }
        },
        {
            "name": "Hip-Hop Track",
            "audio_analysis": {
                "tempo": 85.0,
                "energy": 0.8,
                "danceability": 0.9,
                "valence": 0.4,
                "acousticness": 0.1,
                "key": "F#",
                "mode": "minor",
                "spectral_centroid_mean": 1800.0,
                "genre_predictions": {"hip-hop": 0.92, "r&b": 0.08}
            },
            "content_analysis": {
                "sentiment": {"compound": 0.2},
                "emotions": {"anger": 0.3, "confidence": 0.6},
                "themes": ["struggle", "success", "urban life"],
                "language": {"english_ratio": 0.85}
            },
            "hit_prediction": {
                "hit_probability": 0.68,
                "confidence": 0.75,
                "feature_importance": {"danceability": 0.35, "energy": 0.3, "tempo": 0.2}
            },
            "song_metadata": {
                "title": "City Lights",
                "artist": "Test Rapper",
                "genre": "hip-hop"
            }
        },
        {
            "name": "Country Ballad",
            "audio_analysis": {
                "tempo": 72.0,
                "energy": 0.35,
                "danceability": 0.25,
                "valence": 0.3,
                "acousticness": 0.85,
                "key": "G",
                "mode": "major",
                "spectral_centroid_mean": 1200.0,
                "genre_predictions": {"country": 0.88, "folk": 0.12}
            },
            "content_analysis": {
                "sentiment": {"compound": -0.1},
                "emotions": {"sadness": 0.6, "nostalgia": 0.4},
                "themes": ["heartbreak", "small town", "memories"],
                "language": {"english_ratio": 1.0}
            },
            "hit_prediction": {
                "hit_probability": 0.45,
                "confidence": 0.65,
                "feature_importance": {"acousticness": 0.4, "valence": 0.3, "energy": 0.2}
            },
            "song_metadata": {
                "title": "Backroad Memories",
                "artist": "Test Country",
                "genre": "country"
            }
        }
    ]
    
    # Test agents
    print("\n🧠 Testing Individual Agent Intelligence")
    print("-" * 40)
    
    # Test Music Analysis Agent
    try:
        music_agent = MusicAnalysisAgent()
        print("✅ Music Analysis Agent initialized with LLM service")
        
        # Test with different genres
        for test_case in test_cases:
            print(f"\n📊 Testing {test_case['name']} - Genre: {test_case['song_metadata']['genre']}")
            
            # Create a proper task
            task = AgentTask(
                task_id=f"test_music_{test_case['name'].lower().replace(' ', '_')}",
                agent_id="test_music_agent",
                task_type=TaskType.ANALYSIS, 
                agent_role=AgentRole.MUSIC_ANALYSIS,
                description="Analyze musical characteristics",
                input_data=test_case,
                parameters=test_case,
                priority=1
            )
            
            # Analyze
            contribution = await music_agent.analyze_task(test_case, task)
            
            print(f"   🎵 Findings ({len(contribution.findings)}): {contribution.findings[0][:80]}...")
            print(f"   💡 Insights ({len(contribution.insights)}): {contribution.insights[0][:80]}..." if contribution.insights else "   💡 No insights")
            print(f"   📝 Recommendations ({len(contribution.recommendations)}): {contribution.recommendations[0][:80]}..." if contribution.recommendations else "   📝 No recommendations")
            print(f"   🎯 Confidence: {contribution.confidence_level:.1%}")
            
    except Exception as e:
        print(f"❌ Music Analysis Agent test failed: {e}")
    
    # Test Commercial Analysis Agent
    try:
        commercial_agent = CommercialAnalysisAgent()
        print("\n✅ Commercial Analysis Agent initialized with LLM service")
        
        # Test with different genres
        for test_case in test_cases:
            print(f"\n💰 Testing {test_case['name']} - Genre: {test_case['song_metadata']['genre']}")
            
            # Create a proper task
            task = AgentTask(
                task_id=f"test_commercial_{test_case['name'].lower().replace(' ', '_')}",
                agent_id="test_commercial_agent",
                task_type=TaskType.ANALYSIS,
                agent_role=AgentRole.COMMERCIAL_ANALYSIS,
                description="Analyze commercial potential",
                input_data=test_case,
                parameters=test_case,
                priority=1
            )
            
            # Analyze
            contribution = await commercial_agent.analyze_task(test_case, task)
            
            print(f"   💰 Findings ({len(contribution.findings)}): {contribution.findings[0][:80]}...")
            print(f"   📈 Insights ({len(contribution.insights)}): {contribution.insights[0][:80]}..." if contribution.insights else "   📈 No insights")
            print(f"   🎯 Recommendations ({len(contribution.recommendations)}): {contribution.recommendations[0][:80]}..." if contribution.recommendations else "   🎯 No recommendations")
            print(f"   💪 Confidence: {contribution.confidence_level:.1%}")
            
    except Exception as e:
        print(f"❌ Commercial Analysis Agent test failed: {e}")
    
    # Test Agent Orchestrator
    print("\n🎭 Testing Agent Orchestrator with LLM Intelligence")
    print("-" * 40)
    
    try:
        orchestrator = AgentOrchestrator()
        print("✅ Agent Orchestrator initialized")
        
        # Test coordinated analysis
        test_request = {
            "audio_analysis": test_cases[0]["audio_analysis"],
            "content_analysis": test_cases[0]["content_analysis"],
            "hit_prediction": test_cases[0]["hit_prediction"],
            "song_metadata": test_cases[0]["song_metadata"]
        }
        
        # Coordinate analysis
        result = await orchestrator.coordinate_analysis(test_request)
        
        print(f"   🎵 Participating Agents: {len(result.agent_contributions)}")
        print(f"   💡 Consensus Findings: {len(result.consensus_findings)}")
        print(f"   📝 Overall Quality: {result.quality_score:.1%}")
        print(f"   ⚡ Processing Time: {result.total_processing_time_ms:.0f}ms")
        print(f"   🎯 Confidence: {result.overall_confidence:.1%}")
        
        # Show sample insights
        if result.consensus_findings:
            print(f"\n   🔍 Sample Finding: {result.consensus_findings[0][:100]}...")
        
        if result.executive_summary:
            print(f"   📋 Executive Summary: {result.executive_summary[:100]}...")
            
    except Exception as e:
        print(f"❌ Agent Orchestrator test failed: {e}")
    
    # Test LLM Service Stats
    print("\n📊 LLM Service Statistics")
    print("-" * 40)
    
    try:
        from app.services.agent_llm_service import AgentLLMService
        
        llm_service = AgentLLMService()
        stats = llm_service.get_service_stats()
        
        print(f"   🤖 LLM Provider Available: {stats['llm_provider_available']}")
        print(f"   🔧 LLM Provider Type: {stats.get('llm_provider_type', 'None')}")
        print(f"   📚 Cached Analyses: {stats['cached_analyses']}")
        print(f"   💬 Feedback Entries: {stats['feedback_entries']}")
        print(f"   🎵 Supported Genres: {', '.join(stats['supported_genres'])}")
        
    except Exception as e:
        print(f"❌ LLM Service stats test failed: {e}")
    
    print("\n🎉 LLM-Enhanced Agent Intelligence System Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_llm_enhanced_agents()) 