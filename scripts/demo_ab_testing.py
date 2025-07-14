#!/usr/bin/env python3
"""
A/B Testing Demonstration Script

This script demonstrates how A/B testing would work in the workflow system.
It creates sample tests, generates synthetic traffic, and shows statistical analysis.
"""

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import httpx
import numpy as np
from scipy import stats

# Configuration
ORCHESTRATOR_API = "http://localhost:8006"
ML_PREDICTION_API = "http://localhost:8004"

class ABTestDemo:
    """Demonstration of A/B testing functionality."""
    
    def __init__(self):
        self.orchestrator_client = httpx.AsyncClient(base_url=ORCHESTRATOR_API)
        self.ml_client = httpx.AsyncClient(base_url=ML_PREDICTION_API)
        
    async def demo_complete_workflow(self):
        """Demonstrate the complete A/B testing workflow."""
        print("🧪 A/B Testing Workflow Demonstration")
        print("=" * 50)
        
        # Step 1: Create A/B Test
        print("\n📝 Step 1: Creating A/B Test...")
        test_config = {
            "configuration": {
                "name": "XGBoost vs Random Forest - Demo",
                "description": "Comparing model performance for hit prediction",
                "variants": [
                    {
                        "variant_id": "champion",
                        "name": "Champion (XGBoost)",
                        "model_id": "xgboost-v2.1",
                        "model_version": "v2.1",
                        "traffic_percentage": 50.0
                    },
                    {
                        "variant_id": "challenger",
                        "name": "Challenger (Random Forest)",
                        "model_id": "random-forest-v3.0",
                        "model_version": "v3.0",
                        "traffic_percentage": 50.0
                    }
                ],
                "metrics": [
                    {
                        "name": "accuracy",
                        "type": "accuracy",
                        "target_improvement": 5.0,
                        "is_primary": True
                    },
                    {
                        "name": "latency",
                        "type": "latency",
                        "target_improvement": -10.0,
                        "is_primary": False
                    }
                ],
                "confidence_level": 0.95,
                "minimum_sample_size": 100,
                "maximum_duration_days": 7
            },
            "start_immediately": True
        }
        
        test_id = await self.create_ab_test(test_config)
        if not test_id:
            print("❌ Failed to create A/B test")
            return
            
        print(f"✅ Created A/B test: {test_id}")
        
        # Step 2: Generate synthetic traffic
        print("\n🚦 Step 2: Generating synthetic traffic...")
        await self.generate_synthetic_traffic(test_id, num_requests=200)
        
        # Step 3: Analyze results
        print("\n📊 Step 3: Analyzing results...")
        await self.analyze_test_results(test_id)
        
        # Step 4: Demonstrate winner determination
        print("\n🏆 Step 4: Winner determination...")
        await self.demonstrate_winner_determination(test_id)
        
        print("\n✅ A/B Testing demonstration completed!")
        
    async def create_ab_test(self, config: Dict[str, Any]) -> str:
        """Create an A/B test."""
        try:
            response = await self.orchestrator_client.post(
                "/ab-testing/tests",
                json=config
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["test_id"]
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            print("💡 Make sure workflow-orchestrator is running on port 8006")
            
            # Simulate the test creation for demo purposes
            test_id = f"demo-test-{int(time.time())}"
            print(f"🔄 Simulating test creation: {test_id}")
            return test_id
    
    async def generate_synthetic_traffic(self, test_id: str, num_requests: int = 200):
        """Generate synthetic traffic for A/B testing."""
        print(f"Generating {num_requests} synthetic requests...")
        
        # Sample music features for testing
        sample_features = [
            {
                "audio_tempo": random.uniform(60, 180),
                "audio_energy": random.uniform(0.0, 1.0),
                "audio_danceability": random.uniform(0.0, 1.0),
                "audio_valence": random.uniform(0.0, 1.0),
                "lyrical_sentiment": random.uniform(-1.0, 1.0),
                "lyrical_complexity": random.uniform(0.0, 1.0),
                "genre": random.choice(["pop", "rock", "hip-hop", "electronic"])
            }
            for _ in range(num_requests)
        ]
        
        results = []
        
        for i, features in enumerate(sample_features):
            # Simulate A/B test routing
            variant = "champion" if random.random() < 0.5 else "challenger"
            
            # Simulate prediction results with realistic performance differences
            if variant == "champion":
                # XGBoost: slightly higher accuracy, lower latency
                accuracy = random.normalvariate(0.847, 0.05)
                latency = random.normalvariate(1.2, 0.3)
            else:
                # Random Forest: slightly lower accuracy, higher latency
                accuracy = random.normalvariate(0.862, 0.05)
                latency = random.normalvariate(2.1, 0.4)
            
            # Ensure realistic bounds
            accuracy = max(0.0, min(1.0, accuracy))
            latency = max(0.1, latency)
            
            result = {
                "request_id": f"req-{i:04d}",
                "variant": variant,
                "features": features,
                "metrics": {
                    "accuracy": accuracy,
                    "latency": latency
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            results.append(result)
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{num_requests} requests...")
        
        # Store results for analysis
        self.test_results = results
        print(f"✅ Generated {len(results)} synthetic requests")
        
        # Show traffic distribution
        champion_count = len([r for r in results if r["variant"] == "champion"])
        challenger_count = len([r for r in results if r["variant"] == "challenger"])
        print(f"  Champion: {champion_count} requests ({champion_count/len(results)*100:.1f}%)")
        print(f"  Challenger: {challenger_count} requests ({challenger_count/len(results)*100:.1f}%)")
    
    async def analyze_test_results(self, test_id: str):
        """Analyze A/B test results."""
        if not hasattr(self, 'test_results'):
            print("❌ No test results available")
            return
        
        results = self.test_results
        
        # Group by variant
        champion_results = [r for r in results if r["variant"] == "champion"]
        challenger_results = [r for r in results if r["variant"] == "challenger"]
        
        print(f"Analyzing {len(results)} total samples:")
        print(f"  Champion: {len(champion_results)} samples")
        print(f"  Challenger: {len(challenger_results)} samples")
        
        # Analyze each metric
        metrics = ["accuracy", "latency"]
        
        for metric in metrics:
            print(f"\n📈 {metric.title()} Analysis:")
            
            champion_values = [r["metrics"][metric] for r in champion_results]
            challenger_values = [r["metrics"][metric] for r in challenger_results]
            
            # Calculate statistics
            champion_mean = np.mean(champion_values)
            challenger_mean = np.mean(challenger_values)
            champion_std = np.std(champion_values)
            challenger_std = np.std(challenger_values)
            
            print(f"  Champion:   {champion_mean:.4f} ± {champion_std:.4f}")
            print(f"  Challenger: {challenger_mean:.4f} ± {challenger_std:.4f}")
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(champion_values, challenger_values)
            
            print(f"  T-statistic: {t_stat:.4f}")
            print(f"  P-value: {p_value:.4f}")
            
            # Determine significance
            alpha = 0.05
            if p_value < alpha:
                print(f"  🎯 SIGNIFICANT DIFFERENCE (p < {alpha})")
                
                # Determine winner
                if metric == "latency":
                    # Lower is better for latency
                    winner = "Champion" if champion_mean < challenger_mean else "Challenger"
                else:
                    # Higher is better for accuracy
                    winner = "Champion" if champion_mean > challenger_mean else "Challenger"
                
                improvement = abs(challenger_mean - champion_mean) / champion_mean * 100
                print(f"  🏆 Winner: {winner} ({improvement:.2f}% improvement)")
            else:
                print(f"  ⚠️  No significant difference (p >= {alpha})")
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((champion_std**2 + challenger_std**2) / 2)
            cohens_d = abs(champion_mean - challenger_mean) / pooled_std
            print(f"  Effect size (Cohen's d): {cohens_d:.4f}")
            
            if cohens_d < 0.2:
                print("    Small effect")
            elif cohens_d < 0.5:
                print("    Medium effect")
            else:
                print("    Large effect")
    
    async def demonstrate_winner_determination(self, test_id: str):
        """Demonstrate automated winner determination."""
        print("Demonstrating winner determination logic...")
        
        if not hasattr(self, 'test_results'):
            print("❌ No test results available")
            return
        
        results = self.test_results
        
        # Simulate real-time winner determination
        sample_sizes = [50, 100, 150, 200]
        
        for sample_size in sample_sizes:
            print(f"\n📊 Analysis with {sample_size} samples:")
            
            # Take subset of results
            subset = results[:sample_size]
            champion_subset = [r for r in subset if r["variant"] == "champion"]
            challenger_subset = [r for r in subset if r["variant"] == "challenger"]
            
            if len(champion_subset) < 10 or len(challenger_subset) < 10:
                print("  ⏳ Insufficient data for analysis")
                continue
            
            # Analyze primary metric (accuracy)
            champion_accuracy = [r["metrics"]["accuracy"] for r in champion_subset]
            challenger_accuracy = [r["metrics"]["accuracy"] for r in challenger_subset]
            
            t_stat, p_value = stats.ttest_ind(champion_accuracy, challenger_accuracy)
            
            champion_mean = np.mean(champion_accuracy)
            challenger_mean = np.mean(challenger_accuracy)
            
            print(f"  Champion accuracy: {champion_mean:.4f}")
            print(f"  Challenger accuracy: {challenger_mean:.4f}")
            print(f"  P-value: {p_value:.4f}")
            
            # Winner determination logic
            if p_value < 0.05:
                winner = "Challenger" if challenger_mean > champion_mean else "Champion"
                confidence = (1 - p_value) * 100
                improvement = abs(challenger_mean - champion_mean) / champion_mean * 100
                
                print(f"  🎯 WINNER DETECTED: {winner}")
                print(f"  📈 Improvement: {improvement:.2f}%")
                print(f"  🔒 Confidence: {confidence:.1f}%")
                
                if sample_size >= 150:  # Minimum sample size reached
                    print(f"  ✅ RECOMMENDATION: Deploy {winner} model")
                    break
            else:
                print(f"  ⏳ Continue testing (p = {p_value:.4f})")
        
        print("\n🎯 Winner determination completed!")
    
    async def demonstrate_traffic_routing(self):
        """Demonstrate traffic routing strategies."""
        print("\n🚦 Traffic Routing Strategies Demo")
        print("-" * 40)
        
        strategies = [
            ("percentage", "50/50 split"),
            ("sticky_session", "User-based routing"),
            ("feature_based", "Content-based routing"),
            ("gradual_rollout", "Progressive deployment")
        ]
        
        for strategy, description in strategies:
            print(f"\n📍 {strategy.upper()}: {description}")
            
            # Simulate routing decisions
            decisions = []
            for i in range(100):
                if strategy == "percentage":
                    variant = "A" if random.random() < 0.5 else "B"
                elif strategy == "sticky_session":
                    user_id = f"user-{i % 20}"  # 20 unique users
                    variant = "A" if hash(user_id) % 2 == 0 else "B"
                elif strategy == "feature_based":
                    genre = random.choice(["pop", "rock", "hip-hop"])
                    variant = "A" if genre in ["pop", "rock"] else "B"
                else:  # gradual_rollout
                    # Start with 10% traffic to B, gradually increase
                    rollout_percentage = min(90, i * 0.9)  # 0% to 90%
                    variant = "B" if random.random() * 100 < rollout_percentage else "A"
                
                decisions.append(variant)
            
            # Show distribution
            a_count = decisions.count("A")
            b_count = decisions.count("B")
            print(f"  Variant A: {a_count}% | Variant B: {b_count}%")
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.orchestrator_client.aclose()
        await self.ml_client.aclose()

async def main():
    """Main demonstration function."""
    demo = ABTestDemo()
    
    try:
        print("🎬 Starting A/B Testing Demonstration")
        print("This demo shows how A/B testing would work in production")
        print("=" * 60)
        
        # Check if services are available
        print("\n🔍 Checking service availability...")
        try:
            response = await demo.orchestrator_client.get("/health")
            if response.status_code == 200:
                print("✅ Orchestrator service: Available")
            else:
                print("⚠️  Orchestrator service: Not responding")
        except:
            print("❌ Orchestrator service: Not available")
            print("💡 This demo will use simulated data")
        
        # Run the complete demonstration
        await demo.demo_complete_workflow()
        
        # Show additional features
        await demo.demonstrate_traffic_routing()
        
        print("\n" + "=" * 60)
        print("🎉 A/B Testing Demonstration Complete!")
        print("\nKey Takeaways:")
        print("✅ Statistical significance testing")
        print("✅ Automated winner determination")
        print("✅ Multiple traffic routing strategies")
        print("✅ Real-time performance monitoring")
        print("✅ Production-ready architecture")
        
        print("\nNext Steps:")
        print("1. 🚀 Deploy the A/B testing service")
        print("2. 🔌 Integrate with ML prediction pipeline")
        print("3. 📊 Connect HSS Admin UI to real APIs")
        print("4. 🎯 Start your first A/B test!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    finally:
        await demo.cleanup()

if __name__ == "__main__":
    # Install required packages if not available
    try:
        import scipy
        import numpy
    except ImportError:
        print("📦 Installing required packages...")
        import subprocess
        subprocess.run(["pip", "install", "scipy", "numpy", "httpx"])
    
    # Run the demonstration
    asyncio.run(main()) 