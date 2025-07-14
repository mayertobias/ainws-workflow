"""
Historical Benchmarking Service

Extracted from intelligence_service.py to preserve sophisticated historical analysis
and genre benchmarking capabilities within the new agent architecture.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics

logger = logging.getLogger(__name__)

@dataclass
class GenreBenchmark:
    """Statistical benchmarks for genre comparison"""
    feature_name: str
    mean: float
    std_dev: float
    min_val: float
    max_val: float
    optimal_range: Tuple[float, float]
    
    def calculate_z_score(self, value: float) -> float:
        """Calculate how many standard deviations away from mean"""
        return (value - self.mean) / self.std_dev if self.std_dev > 0 else 0.0
    
    def get_alignment_assessment(self, value: float) -> str:
        """Get text assessment of value alignment with genre norms"""
        z_score = self.calculate_z_score(value)
        if abs(z_score) < 0.5:
            return f"perfectly aligned with genre norm (mean: {self.mean:.2f})"
        elif abs(z_score) < 1.0:
            return f"slightly {'above' if z_score > 0 else 'below'} genre average ({z_score:+.1f}σ)"
        elif abs(z_score) < 2.0:
            return f"notably {'higher' if z_score > 0 else 'lower'} than typical ({z_score:+.1f}σ)"
        else:
            return f"significantly {'above' if z_score > 0 else 'below'} genre norm ({z_score:+.1f}σ)"

@dataclass
class ProductionStandards:
    """Industry production standards and targets"""
    streaming_lufs_target: float = -14.0
    radio_lufs_target: float = -23.0
    dynamic_range_minimum: float = 6.0
    dynamic_range_optimal: float = 12.0
    frequency_balance_tolerance: float = 3.0

class HistoricalBenchmarkingService:
    """
    Professional benchmarking service for historical analysis and genre comparison.
    
    This service preserves the sophisticated benchmarking capabilities from the
    original intelligence service while integrating with the new agent architecture.
    """
    
    def __init__(self):
        self.production_standards = ProductionStandards()
        self.genre_benchmarks = self._load_genre_benchmarks()
        logger.info("HistoricalBenchmarkingService initialized with genre benchmarks")
    
    def _load_genre_benchmarks(self) -> Dict[str, Dict[str, GenreBenchmark]]:
        """Load statistical benchmarks for different genres"""
        # Professional industry benchmarks based on analysis of hit songs
        benchmarks = {
            "pop": {
                "tempo": GenreBenchmark("tempo", 120.0, 15.0, 80.0, 180.0, (100.0, 140.0)),
                "energy": GenreBenchmark("energy", 0.67, 0.12, 0.2, 1.0, (0.55, 0.85)),
                "danceability": GenreBenchmark("danceability", 0.65, 0.15, 0.2, 1.0, (0.50, 0.80)),
                "valence": GenreBenchmark("valence", 0.55, 0.20, 0.0, 1.0, (0.40, 0.75)),
                "loudness": GenreBenchmark("loudness", -8.5, 2.5, -20.0, -3.0, (-11.0, -6.0)),
                "acousticness": GenreBenchmark("acousticness", 0.35, 0.25, 0.0, 1.0, (0.10, 0.60)),
                "instrumentalness": GenreBenchmark("instrumentalness", 0.05, 0.12, 0.0, 1.0, (0.0, 0.15)),
            },
            "rock": {
                "tempo": GenreBenchmark("tempo", 125.0, 20.0, 90.0, 200.0, (110.0, 150.0)),
                "energy": GenreBenchmark("energy", 0.78, 0.15, 0.3, 1.0, (0.65, 0.95)),
                "danceability": GenreBenchmark("danceability", 0.55, 0.18, 0.2, 1.0, (0.40, 0.75)),
                "valence": GenreBenchmark("valence", 0.60, 0.22, 0.0, 1.0, (0.45, 0.80)),
                "loudness": GenreBenchmark("loudness", -7.5, 2.0, -15.0, -3.0, (-9.5, -5.5)),
                "acousticness": GenreBenchmark("acousticness", 0.25, 0.20, 0.0, 1.0, (0.05, 0.45)),
                "instrumentalness": GenreBenchmark("instrumentalness", 0.15, 0.25, 0.0, 1.0, (0.0, 0.40)),
            },
            "electronic": {
                "tempo": GenreBenchmark("tempo", 128.0, 25.0, 70.0, 180.0, (120.0, 140.0)),
                "energy": GenreBenchmark("energy", 0.75, 0.18, 0.2, 1.0, (0.60, 0.90)),
                "danceability": GenreBenchmark("danceability", 0.75, 0.12, 0.3, 1.0, (0.65, 0.90)),
                "valence": GenreBenchmark("valence", 0.65, 0.25, 0.0, 1.0, (0.45, 0.85)),
                "loudness": GenreBenchmark("loudness", -6.5, 3.0, -15.0, -2.0, (-9.0, -4.0)),
                "acousticness": GenreBenchmark("acousticness", 0.15, 0.15, 0.0, 1.0, (0.0, 0.30)),
                "instrumentalness": GenreBenchmark("instrumentalness", 0.35, 0.30, 0.0, 1.0, (0.05, 0.65)),
            },
            "hip_hop": {
                "tempo": GenreBenchmark("tempo", 95.0, 18.0, 60.0, 140.0, (80.0, 110.0)),
                "energy": GenreBenchmark("energy", 0.72, 0.15, 0.3, 1.0, (0.60, 0.85)),
                "danceability": GenreBenchmark("danceability", 0.78, 0.12, 0.4, 1.0, (0.70, 0.90)),
                "valence": GenreBenchmark("valence", 0.50, 0.25, 0.0, 1.0, (0.30, 0.70)),
                "loudness": GenreBenchmark("loudness", -7.0, 2.5, -15.0, -2.0, (-9.0, -5.0)),
                "acousticness": GenreBenchmark("acousticness", 0.20, 0.18, 0.0, 1.0, (0.05, 0.35)),
                "speechiness": GenreBenchmark("speechiness", 0.25, 0.15, 0.03, 0.96, (0.15, 0.40)),
            }
        }
        return benchmarks
    
    def analyze_against_benchmarks(self, audio_features: Dict[str, Any], genre: str = "pop") -> Dict[str, Any]:
        """
        Analyze song features against historical genre benchmarks.
        
        This method preserves the sophisticated benchmarking from intelligence_service.py
        """
        try:
            # Clean the genre name using the same logic as agent_llm_service
            cleaned_genre = self._clean_genre_name(genre)
            
            # Normalize genre name  
            genre_key = cleaned_genre.lower().replace(" ", "_").replace("-", "_")
            if genre_key not in self.genre_benchmarks:
                # Try common genre variations before defaulting to pop
                genre_variations = {
                    "hip_hop": ["hiphop", "rap"],
                    "r_b": ["rnb", "r&b"],
                    "electronic": ["edm", "techno", "house"],
                    "alternative": ["alt", "indie"],
                    "classical": ["orchestral"],
                    "metal": ["heavy_metal", "death_metal"]
                }
                
                found_variation = False
                for bench_genre, variations in genre_variations.items():
                    if genre_key in variations or any(var in genre_key for var in variations):
                        if bench_genre in self.genre_benchmarks:
                            genre_key = bench_genre
                            found_variation = True
                            logger.info(f"Historical benchmarking: Mapped genre variation '{genre}' -> '{bench_genre}'")
                            break
                
                if not found_variation:
                    logger.warning(f"Historical benchmarking: Genre '{genre}' not found in benchmarks, available genres: {list(self.genre_benchmarks.keys())}")
                    genre_key = "pop"  # Final fallback to pop
            
            benchmarks = self.genre_benchmarks[genre_key]
            analysis_result = {
                "genre_analyzed": genre_key,
                "feature_assessments": {},
                "overall_alignment": {},
                "recommendations": []
            }
            
            # Analyze each feature against benchmarks
            total_alignment_score = 0.0
            features_analyzed = 0
            
            for feature_name, benchmark in benchmarks.items():
                # Handle different feature name formats
                feature_value = None
                for key in audio_features:
                    if key.endswith(feature_name) or feature_name in key.lower():
                        feature_value = audio_features[key]
                        break
                
                if feature_value is not None:
                    z_score = benchmark.calculate_z_score(feature_value)
                    alignment_text = benchmark.get_alignment_assessment(feature_value)
                    
                    # Calculate alignment score (1.0 = perfect, 0.0 = very misaligned)
                    alignment_score = max(0.0, 1.0 - abs(z_score) / 3.0)
                    
                    analysis_result["feature_assessments"][feature_name] = {
                        "value": feature_value,
                        "genre_mean": benchmark.mean,
                        "z_score": z_score,
                        "alignment_score": alignment_score,
                        "assessment": alignment_text,
                        "optimal_range": benchmark.optimal_range,
                        "in_optimal_range": benchmark.optimal_range[0] <= feature_value <= benchmark.optimal_range[1]
                    }
                    
                    total_alignment_score += alignment_score
                    features_analyzed += 1
            
            # Calculate overall alignment
            if features_analyzed > 0:
                overall_score = total_alignment_score / features_analyzed
                analysis_result["overall_alignment"] = {
                    "score": overall_score,
                    "assessment": self._get_overall_assessment(overall_score),
                    "features_analyzed": features_analyzed
                }
            
            # Generate recommendations
            analysis_result["recommendations"] = self._generate_benchmarking_recommendations(
                analysis_result["feature_assessments"], genre_key
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in benchmark analysis: {e}")
            return {"error": str(e), "genre_analyzed": genre}
    
    def _get_overall_assessment(self, score: float) -> str:
        """Get text assessment of overall genre alignment"""
        if score >= 0.8:
            return "Excellent alignment with genre conventions"
        elif score >= 0.6:
            return "Good alignment with some unique characteristics"
        elif score >= 0.4:
            return "Moderate alignment with notable deviations"
        elif score >= 0.2:
            return "Significant departures from genre norms"
        else:
            return "Highly unconventional for this genre"
    
    def _generate_benchmarking_recommendations(self, assessments: Dict[str, Any], genre: str) -> List[str]:
        """Generate actionable recommendations based on benchmark analysis"""
        recommendations = []
        
        for feature_name, assessment in assessments.items():
            if assessment["alignment_score"] < 0.5:  # Significantly misaligned
                if assessment["z_score"] > 2.0:
                    recommendations.append(
                        f"Consider reducing {feature_name} ({assessment['value']:.2f}) "
                        f"to better align with {genre} standards (typical: {assessment['genre_mean']:.2f})"
                    )
                elif assessment["z_score"] < -2.0:
                    recommendations.append(
                        f"Consider increasing {feature_name} ({assessment['value']:.2f}) "
                        f"to better align with {genre} standards (typical: {assessment['genre_mean']:.2f})"
                    )
            
            if not assessment["in_optimal_range"]:
                optimal_min, optimal_max = assessment["optimal_range"]
                recommendations.append(
                    f"For optimal {genre} appeal, target {feature_name} "
                    f"between {optimal_min:.2f} and {optimal_max:.2f}"
                )
        
        return recommendations
    
    def get_production_assessment(self, audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """Assess production quality against industry standards"""
        assessment = {
            "lufs_compliance": {},
            "dynamic_range_analysis": {},
            "overall_production_score": 0.0,
            "recommendations": []
        }
        
        # Analyze loudness if available
        if "loudness" in audio_features:
            loudness = audio_features["loudness"]
            
            # Streaming platform compliance
            streaming_deviation = abs(loudness - self.production_standards.streaming_lufs_target)
            radio_deviation = abs(loudness - self.production_standards.radio_lufs_target)
            
            assessment["lufs_compliance"] = {
                "current_lufs": loudness,
                "streaming_target": self.production_standards.streaming_lufs_target,
                "radio_target": self.production_standards.radio_lufs_target,
                "streaming_compliant": streaming_deviation <= 2.0,
                "radio_compliant": radio_deviation <= 2.0
            }
        
        return assessment
    
    def _clean_genre_name(self, genre_name: str) -> str:
        """Clean and normalize genre names from various formats (shared with agent_llm_service)."""
        if not genre_name:
            return "pop"
        
        original_genre = genre_name
        
        # Handle Essentia format like "Electronic---House" or "Funk / Soul---Funk"
        if "---" in genre_name:
            # For electronic music, prefer the main category over sub-genre
            parts = genre_name.split("---")
            main_category = parts[0].strip()
            sub_genre = parts[-1].strip()
            
            # Check if main category is electronic and sub-genre is electronic sub-type
            if main_category.lower() == "electronic" and sub_genre.lower() in ["house", "techno", "trance", "edm", "dubstep", "drum", "bass"]:
                main_genre = main_category  # Use "Electronic" instead of "House"
            else:
                main_genre = sub_genre  # Use sub-genre for other cases like "Funk / Soul---Funk"
        elif " / " in genre_name:
            # Take the first part before the slash
            main_genre = genre_name.split(" / ")[0].strip()
        else:
            main_genre = genre_name.strip()
        
        # Normalize to lowercase for consistency
        main_genre = main_genre.lower()
        
        # Map common variations to standard genre names
        genre_mapping = {
            "funk": "funk",
            "soul": "soul", 
            "r&b": "rnb",
            "hip hop": "hip-hop",
            "hip-hop": "hip-hop",
            "electronic": "electronic",
            "house": "electronic",  # House is a type of electronic
            "techno": "electronic", # Techno is a type of electronic
            "edm": "electronic",    # EDM is electronic dance music
            "trance": "electronic", # Trance is electronic
            "dubstep": "electronic", # Dubstep is electronic
            "drum": "electronic",   # Drum & bass is electronic
            "bass": "electronic",   # Bass music is electronic
            "rock": "rock",
            "pop": "pop",
            "country": "country",
            "jazz": "jazz",
            "classical": "classical",
            "blues": "blues",
            "reggae": "reggae",
            "folk": "folk",
            "metal": "metal",
            "punk": "punk",
            "alternative": "alternative",
            "indie": "indie"
        }
        
        # Find best match
        for key, value in genre_mapping.items():
            if key in main_genre or main_genre in key:
                logger.debug(f"Historical benchmarking: Mapped genre '{original_genre}' -> '{value}' (via '{main_genre}')")
                return value
        
        # If no mapping found, return the cleaned main genre
        logger.debug(f"Historical benchmarking: No mapping found for genre '{original_genre}', using cleaned: '{main_genre}'")
        return main_genre