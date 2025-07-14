"""
Historical Data Integrator for workflow-intelligence service

Integrates comprehensive music industry historical data for sophisticated insights generation.
Based on the original data structure from /Users/manojveluchuri/saas/r1/simpleui/backend/utils/data_loader.py
"""

import logging
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

class HistoricalDataIntegrator:
    """Integrates historical music industry data for sophisticated insights."""
    
    def __init__(self):
        """Initialize with default historical data structure."""
        self.historical_data = self._load_default_historical_data()
        self.genre_benchmarks = self._generate_genre_benchmarks()
        self.hit_patterns = self._analyze_hit_patterns()
        
    def _load_default_historical_data(self) -> Dict[str, Any]:
        """Load comprehensive historical music data for analysis."""
        return {
            "genre_analysis": {
                "pop": {
                    "average_features": {
                        "tempo": 118.5, "energy": 0.75, "danceability": 0.68,
                        "valence": 0.62, "acousticness": 0.28, "loudness": -6.2
                    },
                    "hit_characteristics": {
                        "tempo_range": [100, 140], "energy_sweet_spot": [0.6, 0.9],
                        "typical_structure": "verse-chorus-verse-chorus-bridge-chorus",
                        "common_themes": ["love", "relationships", "empowerment", "celebration"]
                    },
                    "top_hits_features": {
                        "tempo": 120.3, "energy": 0.82, "danceability": 0.74,
                        "valence": 0.68, "acousticness": 0.15, "loudness": -5.8
                    },
                    "commercial_patterns": {
                        "peak_streaming_months": ["June", "July", "December"],
                        "target_demographics": ["18-34", "mainstream"],
                        "market_positioning": "broad appeal, radio-friendly"
                    }
                },
                "rock": {
                    "average_features": {
                        "tempo": 125.2, "energy": 0.81, "danceability": 0.52,
                        "valence": 0.48, "acousticness": 0.12, "loudness": -5.5
                    },
                    "hit_characteristics": {
                        "tempo_range": [110, 160], "energy_sweet_spot": [0.7, 0.95],
                        "typical_structure": "intro-verse-chorus-verse-chorus-solo-chorus",
                        "common_themes": ["rebellion", "freedom", "identity", "social commentary"]
                    }
                },
                "hip-hop": {
                    "average_features": {
                        "tempo": 95.8, "energy": 0.73, "danceability": 0.82,
                        "valence": 0.58, "acousticness": 0.08, "loudness": -6.8
                    },
                    "hit_characteristics": {
                        "tempo_range": [80, 120], "energy_sweet_spot": [0.6, 0.85],
                        "typical_structure": "intro-verse-hook-verse-hook-bridge-hook",
                        "common_themes": ["success", "struggle", "lifestyle", "social issues"]
                    }
                },
                "electronic": {
                    "average_features": {
                        "tempo": 128.0, "energy": 0.88, "danceability": 0.79,
                        "valence": 0.71, "acousticness": 0.03, "loudness": -5.2
                    },
                    "hit_characteristics": {
                        "tempo_range": [120, 140], "energy_sweet_spot": [0.8, 0.95],
                        "typical_structure": "build-drop-breakdown-build-drop",
                        "common_themes": ["escapism", "euphoria", "technology", "freedom"]
                    }
                }
            },
            "hit_score_analysis": {
                "score_ranges": {
                    "high_potential": {"min": 0.7, "max": 1.0, "description": "Strong commercial potential"},
                    "moderate_potential": {"min": 0.4, "max": 0.69, "description": "Moderate commercial appeal"},
                    "niche_appeal": {"min": 0.15, "max": 0.39, "description": "Niche or artistic appeal"},
                    "experimental": {"min": 0.0, "max": 0.14, "description": "Experimental or avant-garde"}
                },
                "confidence_factors": {
                    "feature_completeness": 0.3,
                    "genre_alignment": 0.25,
                    "historical_similarity": 0.25,
                    "model_performance": 0.2
                }
            },
            "production_standards": {
                "modern_mastering": {
                    "loudness_lufs": {"target": -14.0, "range": [-16.0, -12.0]},
                    "dynamic_range": {"minimum": 6.0, "optimal": 8.0},
                    "frequency_balance": {
                        "low_end": "controlled sub-bass, punchy kick",
                        "mid_range": "clear vocals, present instruments",
                        "high_end": "crisp without harshness"
                    }
                },
                "streaming_optimization": {
                    "peak_levels": {"maximum": -1.0, "recommended": -2.0},
                    "intro_timing": {"hook_within_seconds": 15, "vocal_entry": 8},
                    "duration_sweet_spot": {"minimum": 150, "optimal": 180, "maximum": 240}
                }
            },
            "market_intelligence": {
                "current_trends": {
                    "genre_popularity": {"pop": 0.32, "hip-hop": 0.28, "rock": 0.18, "electronic": 0.12, "other": 0.10},
                    "emerging_subgenres": ["bedroom pop", "hyperpop", "drill", "afrobeats"],
                    "declining_trends": ["traditional country", "hard rock", "dubstep"]
                },
                "demographic_preferences": {
                    "gen_z": {"preferred_genres": ["hip-hop", "pop", "electronic"], "attention_span": 15},
                    "millennials": {"preferred_genres": ["pop", "rock", "indie"], "attention_span": 30},
                    "gen_x": {"preferred_genres": ["rock", "alternative", "classic"], "attention_span": 45}
                }
            }
        }
    
    def _generate_genre_benchmarks(self) -> Dict[str, Any]:
        """Generate sophisticated genre benchmarks for comparison."""
        benchmarks = {}
        for genre, data in self.historical_data["genre_analysis"].items():
            benchmarks[genre] = {
                "feature_weights": {
                    "tempo": 0.15,
                    "energy": 0.20,
                    "danceability": 0.18,
                    "valence": 0.12,
                    "acousticness": 0.10,
                    "loudness": 0.15,
                    "production_quality": 0.10
                },
                "commercial_factors": {
                    "radio_friendliness": data.get("commercial_patterns", {}).get("market_positioning", ""),
                    "streaming_appeal": len(data.get("hit_characteristics", {}).get("common_themes", [])),
                    "cross_genre_appeal": 0.5  # Default moderate appeal
                }
            }
        return benchmarks
    
    def _analyze_hit_patterns(self) -> Dict[str, Any]:
        """Analyze patterns that contribute to hit potential."""
        return {
            "audio_feature_patterns": {
                "high_energy_hits": {"energy": [0.7, 0.95], "danceability": [0.6, 0.9]},
                "emotional_ballads": {"valence": [0.2, 0.6], "acousticness": [0.3, 0.8]},
                "mainstream_pop": {"tempo": [110, 130], "energy": [0.6, 0.85], "valence": [0.5, 0.8]}
            },
            "structural_patterns": {
                "radio_format": {"intro": 4, "verse": 16, "chorus": 16, "total_under": 210},
                "streaming_format": {"hook_early": True, "skip_protection": 30, "engagement_curve": "ascending"}
            },
            "novelty_vs_familiarity": {
                "optimal_innovation": 0.25,  # 25% innovative, 75% familiar
                "genre_deviation_threshold": 0.3,
                "trend_alignment_weight": 0.4
            }
        }
    
    def analyze_song_against_benchmarks(self, audio_features: Dict[str, Any], genre: str = "pop") -> Dict[str, Any]:
        """Analyze a song against historical benchmarks."""
        genre_data = self.historical_data["genre_analysis"].get(genre, self.historical_data["genre_analysis"]["pop"])
        
        # Calculate feature alignment
        alignment_scores = {}
        avg_features = genre_data["average_features"]
        hit_features = genre_data.get("top_hits_features", avg_features)
        
        for feature, value in audio_features.items():
            if feature in avg_features and value is not None:
                avg_val = avg_features[feature]
                hit_val = hit_features.get(feature, avg_val)
                
                # Ensure both values are not None
                if hit_val is not None and value is not None:
                    # Calculate distance from hit patterns
                    hit_distance = abs(value - hit_val) / (max(abs(hit_val), 1.0))
                    alignment_scores[feature] = max(0, 1 - hit_distance)
        
        values = list(alignment_scores.values())
        overall_alignment = sum(values) / len(values) if values else 0.5
        
        return {
            "genre_alignment_score": overall_alignment,
            "feature_analysis": alignment_scores,
            "benchmark_comparison": {
                "vs_genre_average": self._compare_to_average(audio_features, avg_features),
                "vs_hit_patterns": self._compare_to_hits(audio_features, hit_features)
            },
            "commercial_indicators": self._assess_commercial_indicators(audio_features, genre_data)
        }
    
    def _compare_to_average(self, song_features: Dict[str, Any], avg_features: Dict[str, Any]) -> Dict[str, str]:
        """Compare song features to genre averages."""
        comparisons = {}
        for feature, avg_val in avg_features.items():
            song_val = song_features.get(feature)
            
            # Handle None/NaN values
            if song_val is None or avg_val is None:
                comparisons[feature] = "Missing data for comparison"
                continue
            
            # Handle numeric comparison safely
            try:
                song_val = float(song_val)
                avg_val = float(avg_val)
                
                if song_val > avg_val * 1.1:
                    comparisons[feature] = f"Above average ({song_val:.2f} vs {avg_val:.2f})"
                elif song_val < avg_val * 0.9:
                    comparisons[feature] = f"Below average ({song_val:.2f} vs {avg_val:.2f})"
                else:
                    comparisons[feature] = f"Average range ({song_val:.2f})"
                    
            except (ValueError, TypeError):
                comparisons[feature] = f"Invalid data types (song: {type(song_val)}, avg: {type(avg_val)})"
                
        return comparisons
    
    def _compare_to_hits(self, song_features: Dict[str, Any], hit_features: Dict[str, Any]) -> Dict[str, str]:
        """Compare song features to hit song patterns."""
        comparisons = {}
        for feature, hit_val in hit_features.items():
            song_val = song_features.get(feature)
            
            # Handle None/NaN values
            if song_val is None or hit_val is None:
                comparisons[feature] = "Missing data for comparison"
                continue
            
            # Handle numeric comparison safely
            try:
                song_val = float(song_val)
                hit_val = float(hit_val)
                
                distance = abs(song_val - hit_val) / max(hit_val, 1.0)
                if distance < 0.1:
                    comparisons[feature] = f"Very close to hit pattern ({song_val:.2f})"
                elif distance < 0.25:
                    comparisons[feature] = f"Close to hit pattern ({song_val:.2f})"
                else:
                    comparisons[feature] = f"Different from hit pattern ({song_val:.2f} vs {hit_val:.2f})"
                    
            except (ValueError, TypeError, ZeroDivisionError):
                comparisons[feature] = f"Invalid data for comparison (song: {type(song_val)}, hit: {type(hit_val)})"
                
        return comparisons
    
    def _assess_commercial_indicators(self, audio_features: Dict[str, Any], genre_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess commercial potential indicators."""
        energy = audio_features.get("energy", 0.5)
        danceability = audio_features.get("danceability", 0.5)
        valence = audio_features.get("valence", 0.5)
        tempo = audio_features.get("tempo", 120)
        
        indicators = {
            "radio_potential": self._assess_radio_potential(audio_features),
            "streaming_appeal": self._assess_streaming_appeal(audio_features),
            "crossover_potential": self._assess_crossover_potential(audio_features, genre_data),
            "demographic_appeal": self._assess_demographic_appeal(audio_features)
        }
        
        return indicators
    
    def _assess_radio_potential(self, features: Dict[str, Any]) -> str:
        """Assess radio play potential."""
        # Use defaults if values are None/NaN
        energy = features.get("energy") or 0.5
        tempo = features.get("tempo") or 120
        loudness = features.get("loudness") or -7
        
        # Ensure values are numeric
        try:
            energy = float(energy) if energy is not None else 0.5
            tempo = float(tempo) if tempo is not None else 120
            loudness = float(loudness) if loudness is not None else -7
        except (ValueError, TypeError):
            return "Cannot assess - insufficient audio data"
        
        if 110 <= tempo <= 140 and energy > 0.6 and loudness > -8:
            return "High - fits radio format standards"
        elif 100 <= tempo <= 150 and energy > 0.4:
            return "Moderate - some radio appeal"
        else:
            return "Low - may not fit commercial radio formats"
    
    def _assess_streaming_appeal(self, features: Dict[str, Any]) -> str:
        """Assess streaming platform appeal."""
        # Handle None/NaN values
        danceability = features.get("danceability") or 0.5
        energy = features.get("energy") or 0.5
        valence = features.get("valence") or 0.5
        
        try:
            danceability = float(danceability) if danceability is not None else 0.5
            energy = float(energy) if energy is not None else 0.5
            valence = float(valence) if valence is not None else 0.5
        except (ValueError, TypeError):
            return "Cannot assess - insufficient audio data"
        
        appeal_score = (danceability + energy + valence) / 3
        
        if appeal_score > 0.7:
            return "High - strong streaming engagement potential"
        elif appeal_score > 0.5:
            return "Moderate - decent streaming appeal"
        else:
            return "Specialized - may appeal to specific audiences"
    
    def _assess_crossover_potential(self, features: Dict[str, Any], genre_data: Dict[str, Any]) -> str:
        """Assess potential for cross-genre appeal."""
        # Handle None/NaN values
        energy = features.get("energy") or 0.5
        danceability = features.get("danceability") or 0.5
        
        try:
            energy = float(energy) if energy is not None else 0.5
            danceability = float(danceability) if danceability is not None else 0.5
        except (ValueError, TypeError):
            return "Cannot assess - insufficient audio data"
        
        if energy > 0.6 and danceability > 0.6:
            return "High - appeals across multiple genres"
        elif energy > 0.4 or danceability > 0.4:
            return "Moderate - some crossover appeal"
        else:
            return "Limited - likely genre-specific appeal"
    
    def _assess_demographic_appeal(self, features: Dict[str, Any]) -> Dict[str, str]:
        """Assess appeal to different demographic groups."""
        # Handle None/NaN values
        energy = features.get("energy") or 0.5
        danceability = features.get("danceability") or 0.5
        valence = features.get("valence") or 0.5
        acousticness = features.get("acousticness") or 0.5
        
        try:
            energy = float(energy) if energy is not None else 0.5
            danceability = float(danceability) if danceability is not None else 0.5
            valence = float(valence) if valence is not None else 0.5
            acousticness = float(acousticness) if acousticness is not None else 0.5
        except (ValueError, TypeError):
            return {
                "gen_z": "Cannot assess - insufficient data",
                "millennials": "Cannot assess - insufficient data", 
                "gen_x": "Cannot assess - insufficient data"
            }
        
        return {
            "gen_z": "High" if (energy > 0.6 and danceability > 0.6) else "Moderate",
            "millennials": "High" if (0.4 < energy < 0.8 and valence > 0.4) else "Moderate", 
            "gen_x": "High" if (acousticness > 0.3 or energy > 0.7) else "Moderate"
        }
    
    def get_hit_score_context(self, hit_score: float) -> Dict[str, Any]:
        """Get contextual information about a hit score."""
        score_ranges = self.historical_data["hit_score_analysis"]["score_ranges"]
        
        context = {
            "score": hit_score,
            "category": "experimental",
            "description": "Experimental or avant-garde",
            "commercial_viability": "Low",
            "recommended_strategy": "Artistic development"
        }
        
        for category, range_info in score_ranges.items():
            if range_info["min"] <= hit_score <= range_info["max"]:
                context["category"] = category
                context["description"] = range_info["description"]
                
                if category == "high_potential":
                    context["commercial_viability"] = "High"
                    context["recommended_strategy"] = "Major label pitch, radio promotion"
                elif category == "moderate_potential":
                    context["commercial_viability"] = "Moderate" 
                    context["recommended_strategy"] = "Streaming focus, playlist placement"
                elif category == "niche_appeal":
                    context["commercial_viability"] = "Niche"
                    context["recommended_strategy"] = "Target specific audiences, indie promotion"
                break
        
        return context
    
    def analyze_production_quality(self, audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze production quality against modern standards."""
        standards = self.historical_data["production_standards"]
        
        loudness = audio_features.get("loudness", -7)
        energy = audio_features.get("energy", 0.5)
        
        analysis = {
            "loudness_assessment": self._assess_loudness(loudness, standards),
            "dynamic_assessment": self._assess_dynamics(audio_features, standards),
            "modern_compliance": self._assess_modern_compliance(audio_features, standards),
            "streaming_readiness": self._assess_streaming_readiness(audio_features, standards)
        }
        
        return analysis
    
    def _assess_loudness(self, loudness: float, standards: Dict[str, Any]) -> str:
        """Assess loudness against modern mastering standards."""
        if loudness > -5:
            return "Very loud - may cause listening fatigue"
        elif loudness > -8:
            return "Appropriately loud for modern streaming"
        elif loudness > -12:
            return "Moderate loudness - good dynamic range"
        else:
            return "Quiet - may lack impact on streaming platforms"
    
    def _assess_dynamics(self, features: Dict[str, Any], standards: Dict[str, Any]) -> str:
        """Assess dynamic range and energy variation."""
        energy = features.get("energy", 0.5)
        loudness = features.get("loudness", -7)
        
        # Simplified dynamic assessment
        if energy > 0.8 and loudness > -6:
            return "High energy, compressed - typical modern production"
        elif energy > 0.6 and loudness < -8:
            return "Good energy with healthy dynamics"
        else:
            return "Moderate dynamics - room for enhancement"
    
    def _assess_modern_compliance(self, features: Dict[str, Any], standards: Dict[str, Any]) -> str:
        """Assess compliance with modern production standards."""
        compliance_score = 0
        factors = []
        
        loudness = features.get("loudness", -7)
        if -16 <= loudness <= -4:
            compliance_score += 1
            factors.append("appropriate loudness")
        
        energy = features.get("energy", 0.5)
        if energy > 0.5:
            compliance_score += 1
            factors.append("sufficient energy")
        
        if compliance_score >= 2:
            return f"High compliance - {', '.join(factors)}"
        elif compliance_score == 1:
            return f"Partial compliance - {', '.join(factors) if factors else 'some standards met'}"
        else:
            return "Low compliance - may need remastering"
    
    def _assess_streaming_readiness(self, features: Dict[str, Any], standards: Dict[str, Any]) -> str:
        """Assess readiness for streaming platforms."""
        streaming_standards = standards["streaming_optimization"]
        
        loudness = features.get("loudness", -7)
        energy = features.get("energy", 0.5)
        
        # Check against streaming targets
        if loudness > streaming_standards["peak_levels"]["recommended"] and energy > 0.6:
            return "Excellent - optimized for streaming platforms"
        elif loudness > -10 and energy > 0.4:
            return "Good - suitable for most streaming platforms"
        else:
            return "Needs optimization - may not compete effectively on streaming"
    
    async def get_relevant_context(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant historical context for the analysis request."""
        try:
            # Extract relevant information from request
            song_metadata = request_data.get("song_metadata", {})
            audio_analysis = request_data.get("audio_analysis", {})
            content_analysis = request_data.get("content_analysis", {})
            
            genre = song_metadata.get("genre", "pop")
            
            # Get historical context based on genre and features
            context = {
                "genre_context": self._get_genre_historical_context(genre, audio_analysis),
                "era_context": self._get_era_context(audio_analysis),
                "market_trends": self._get_current_market_trends(genre),
                "similar_artists": self._find_similar_historical_artists(audio_analysis, genre),
                "commercial_patterns": self._get_commercial_patterns(genre)
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            return {}
    
    def _get_genre_historical_context(self, genre: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Get historical context for a specific genre."""
        genre_data = self.historical_data.get("genre_analysis", {}).get(genre, {})
        if not genre_data:
            genre = "pop"  # Fallback to pop
            genre_data = self.historical_data.get("genre_analysis", {}).get(genre, {})
        
        return {
            "genre": genre,
            "historical_averages": genre_data.get("average_features", {}),
            "hit_characteristics": genre_data.get("hit_characteristics", {}),
            "evolution_trends": f"{genre.title()} has evolved significantly in recent years"
        }
    
    def _get_era_context(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the musical era context based on features."""
        return {
            "era": "contemporary",
            "characteristics": "Modern production techniques and digital processing",
            "context": "Current streaming era with emphasis on immediate engagement"
        }
    
    def _get_current_market_trends(self, genre: str) -> Dict[str, Any]:
        """Get current market trends for the genre."""
        return {
            "trending_elements": ["authentic storytelling", "genre blending", "social media viral potential"],
            "declining_elements": ["overproduced vocals", "formulaic structures"],
            "emerging_opportunities": ["cross-cultural collaborations", "innovative soundscapes"]
        }
    
    def _find_similar_historical_artists(self, features: Dict[str, Any], genre: str) -> List[str]:
        """Find similar historical artists based on features."""
        # This is a simplified version - in production, this would use similarity algorithms
        if genre == "pop":
            return ["Taylor Swift", "Ed Sheeran", "Billie Eilish"]
        elif genre == "rock":
            return ["Imagine Dragons", "OneRepublic", "Maroon 5"]
        else:
            return ["Various contemporary artists"]
    
    def _get_commercial_patterns(self, genre: str) -> Dict[str, Any]:
        """Get commercial success patterns for the genre."""
        genre_data = self.historical_data.get("genre_analysis", {}).get(genre, {})
        return genre_data.get("commercial_patterns", {
            "peak_streaming_months": ["summer", "holiday season"],
            "target_demographics": ["18-34"],
            "market_positioning": "mainstream appeal"
        }) 