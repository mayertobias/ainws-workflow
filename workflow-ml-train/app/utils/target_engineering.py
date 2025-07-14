"""
Target Variable Engineering for Hit Song Science Regression Models

Implements the expert-recommended approach for creating continuous hit_score 
from multiple success metrics instead of binary classification.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class ContinuousHitScoreEngineer:
    """
    Engineer continuous hit_score (0.0-1.0) from multiple success metrics.
    
    Based on expert recommendation to combine:
    - Chart Performance Score (0-1): Inverse of chart position
    - Chart Longevity Score (0-1): Time spent on charts
    - Streaming Score (0-1): Logarithmic streaming numbers
    - Popularity Score (0-1): General popularity metrics
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the hit score engineer.
        
        Args:
            weights: Custom weights for score components
                    Default: {'chart': 0.4, 'longevity': 0.2, 'streams': 0.2, 'popularity': 0.2}
        """
        self.weights = weights or {
            'chart': 0.4,        # Chart position is most important indicator
            'longevity': 0.2,    # How long it stayed popular
            'streams': 0.2,      # Streaming success
            'popularity': 0.2    # General popularity score
        }
        
        # Ensure weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        logger.info(f"Initialized ContinuousHitScoreEngineer with weights: {self.weights}")
    
    def engineer_hit_score(self, df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, any]]:
        """
        Create continuous hit_score from available success metrics.
        
        Args:
            df: DataFrame with song data and success metrics
            
        Returns:
            Tuple of (hit_scores, engineering_report)
        """
        logger.info(f"Engineering continuous hit_score for {len(df)} songs")
        
        # Initialize components
        components = {}
        available_metrics = []
        
        # 1. Chart Performance Score (higher = better, 0-1 scale)
        if self._has_chart_data(df):
            components['chart'] = self._calculate_chart_score(df)
            available_metrics.append('chart')
            logger.info("✅ Chart performance score calculated")
        else:
            logger.warning("⚠️ No chart position data available")
        
        # 2. Chart Longevity Score (0-1 scale)
        if self._has_longevity_data(df):
            components['longevity'] = self._calculate_longevity_score(df)
            available_metrics.append('longevity')
            logger.info("✅ Chart longevity score calculated")
        else:
            logger.warning("⚠️ No chart longevity data available")
        
        # 3. Streaming Score (logarithmic scale, 0-1)
        if self._has_streaming_data(df):
            components['streams'] = self._calculate_streaming_score(df)
            available_metrics.append('streams')
            logger.info("✅ Streaming score calculated")
        else:
            logger.warning("⚠️ No streaming data available")
        
        # 4. Popularity Score (0-1 scale)
        if self._has_popularity_data(df):
            components['popularity'] = self._calculate_popularity_score(df)
            available_metrics.append('popularity')
            logger.info("✅ Popularity score calculated")
        else:
            logger.warning("⚠️ No popularity data available")
        
        # Combine available components
        if not components:
            raise ValueError("No success metrics available for hit_score engineering")
        
        # Adjust weights for available metrics
        adjusted_weights = self._adjust_weights_for_available_metrics(available_metrics)
        logger.info(f"Adjusted weights for available metrics: {adjusted_weights}")
        
        # Calculate weighted combination
        raw_score = np.zeros(len(df))
        for metric, weight in adjusted_weights.items():
            if metric in components:
                raw_score += weight * components[metric]
        
        # Final normalization to ensure 0-1 range
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        hit_scores = scaler.fit_transform(raw_score.reshape(-1, 1)).flatten()
        
        # Create engineering report
        report = {
            'total_songs': len(df),
            'available_metrics': available_metrics,
            'adjusted_weights': adjusted_weights,
            'score_statistics': {
                'min': float(np.min(hit_scores)),
                'max': float(np.max(hit_scores)),
                'mean': float(np.mean(hit_scores)),
                'std': float(np.std(hit_scores)),
                'median': float(np.median(hit_scores))
            },
            'component_correlations': self._calculate_component_correlations(components),
            'percentile_distribution': {
                f'p{p}': float(np.percentile(hit_scores, p)) 
                for p in [10, 25, 50, 75, 90, 95, 99]
            }
        }
        
        logger.info(f"✅ Hit score engineering completed:")
        logger.info(f"   📊 Score range: {report['score_statistics']['min']:.3f} - {report['score_statistics']['max']:.3f}")
        logger.info(f"   📊 Mean: {report['score_statistics']['mean']:.3f}, Std: {report['score_statistics']['std']:.3f}")
        logger.info(f"   📊 Metrics used: {', '.join(available_metrics)}")
        
        return pd.Series(hit_scores, index=df.index), report
    
    def _has_chart_data(self, df: pd.DataFrame) -> bool:
        """Check if chart position data is available."""
        chart_columns = ['chart_position', 'peak_position', 'chart_peak', 'billboard_position']
        return any(col in df.columns and not df[col].isna().all() for col in chart_columns)
    
    def _has_longevity_data(self, df: pd.DataFrame) -> bool:
        """Check if chart longevity data is available."""
        longevity_columns = ['weeks_on_chart', 'chart_weeks', 'chart_duration']
        return any(col in df.columns and not df[col].isna().all() for col in longevity_columns)
    
    def _has_streaming_data(self, df: pd.DataFrame) -> bool:
        """Check if streaming data is available."""
        streaming_columns = ['streams', 'spotify_streams', 'total_streams', 'play_count']
        return any(col in df.columns and not df[col].isna().all() for col in streaming_columns)
    
    def _has_popularity_data(self, df: pd.DataFrame) -> bool:
        """Check if popularity data is available."""
        popularity_columns = ['popularity', 'original_popularity', 'popularity_score']
        return any(col in df.columns and not df[col].isna().all() for col in popularity_columns)
    
    def _calculate_chart_score(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate chart performance score (0-1, higher is better)."""
        # Try different chart position columns
        chart_columns = ['chart_position', 'peak_position', 'chart_peak', 'billboard_position']
        chart_col = None
        
        for col in chart_columns:
            if col in df.columns and not df[col].isna().all():
                chart_col = col
                break
        
        if chart_col is None:
            raise ValueError("No chart position data found")
        
        # Get chart positions (lower position = better)
        positions = df[chart_col].fillna(200)  # Fill NaN with poor position
        
        # Convert to score using inverse square root (as recommended)
        # Position 1 = score 1.0, Position 100 = score 0.1
        chart_scores = 1.0 / np.sqrt(positions.astype(float))
        
        # Normalize to 0-1 range
        chart_scores = (chart_scores - chart_scores.min()) / (chart_scores.max() - chart_scores.min())
        
        return chart_scores.values
    
    def _calculate_longevity_score(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate chart longevity score (0-1, longer is better)."""
        longevity_columns = ['weeks_on_chart', 'chart_weeks', 'chart_duration']
        longevity_col = None
        
        for col in longevity_columns:
            if col in df.columns and not df[col].isna().all():
                longevity_col = col
                break
        
        if longevity_col is None:
            raise ValueError("No chart longevity data found")
        
        # Get weeks on chart
        weeks = df[longevity_col].fillna(0).astype(float)
        
        # Normalize by maximum weeks in dataset
        max_weeks = weeks.max()
        if max_weeks > 0:
            longevity_scores = weeks / max_weeks
        else:
            longevity_scores = np.zeros(len(weeks))
        
        return longevity_scores.values
    
    def _calculate_streaming_score(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate streaming score (0-1, logarithmic scale)."""
        streaming_columns = ['streams', 'spotify_streams', 'total_streams', 'play_count']
        streaming_col = None
        
        for col in streaming_columns:
            if col in df.columns and not df[col].isna().all():
                streaming_col = col
                break
        
        if streaming_col is None:
            raise ValueError("No streaming data found")
        
        # Get streaming numbers (use log scale due to wide range)
        streams = df[streaming_col].fillna(1).astype(float)  # Fill NaN with 1 to avoid log(0)
        streams = np.maximum(streams, 1)  # Ensure positive values
        
        # Logarithmic scaling
        log_streams = np.log10(streams)
        max_log_streams = log_streams.max()
        min_log_streams = log_streams.min()
        
        if max_log_streams > min_log_streams:
            streaming_scores = (log_streams - min_log_streams) / (max_log_streams - min_log_streams)
        else:
            streaming_scores = np.ones(len(streams)) * 0.5  # All equal, assign middle score
        
        return streaming_scores.values
    
    def _calculate_popularity_score(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate popularity score (0-1 scale)."""
        popularity_columns = ['popularity', 'original_popularity', 'popularity_score']
        popularity_col = None
        
        for col in popularity_columns:
            if col in df.columns and not df[col].isna().all():
                popularity_col = col
                break
        
        if popularity_col is None:
            raise ValueError("No popularity data found")
        
        # Get popularity values
        popularity = df[popularity_col].fillna(0).astype(float)
        
        # Check if already in 0-1 range
        if popularity.max() <= 1.0 and popularity.min() >= 0.0:
            return popularity.values
        
        # Normalize to 0-1 range
        min_pop = popularity.min()
        max_pop = popularity.max()
        
        if max_pop > min_pop:
            popularity_scores = (popularity - min_pop) / (max_pop - min_pop)
        else:
            popularity_scores = np.ones(len(popularity)) * 0.5
        
        return popularity_scores.values
    
    def _adjust_weights_for_available_metrics(self, available_metrics: list) -> Dict[str, float]:
        """Adjust weights based on available metrics."""
        if not available_metrics:
            raise ValueError("No metrics available for weight adjustment")
        
        # Get weights for available metrics
        available_weights = {metric: self.weights[metric] for metric in available_metrics if metric in self.weights}
        
        if not available_weights:
            # Equal weighting if no predefined weights
            return {metric: 1.0/len(available_metrics) for metric in available_metrics}
        
        # Normalize available weights to sum to 1.0
        total_weight = sum(available_weights.values())
        return {metric: weight/total_weight for metric, weight in available_weights.items()}
    
    def _calculate_component_correlations(self, components: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between different score components."""
        correlations = {}
        
        if len(components) < 2:
            return correlations
        
        import itertools
        
        for metric1, metric2 in itertools.combinations(components.keys(), 2):
            try:
                corr = np.corrcoef(components[metric1], components[metric2])[0, 1]
                if metric1 not in correlations:
                    correlations[metric1] = {}
                if metric2 not in correlations:
                    correlations[metric2] = {}
                
                correlations[metric1][metric2] = float(corr) if not np.isnan(corr) else 0.0
                correlations[metric2][metric1] = float(corr) if not np.isnan(corr) else 0.0
            except Exception as e:
                logger.warning(f"Could not calculate correlation between {metric1} and {metric2}: {e}")
        
        return correlations

# Convenience function for easy usage
def create_continuous_hit_score(df: pd.DataFrame, 
                               weights: Optional[Dict[str, float]] = None) -> Tuple[pd.Series, Dict[str, any]]:
    """
    Convenience function to create continuous hit_score from DataFrame.
    
    Args:
        df: DataFrame with song data and success metrics
        weights: Optional custom weights for score components
        
    Returns:
        Tuple of (hit_scores, engineering_report)
    """
    engineer = ContinuousHitScoreEngineer(weights=weights)
    return engineer.engineer_hit_score(df)