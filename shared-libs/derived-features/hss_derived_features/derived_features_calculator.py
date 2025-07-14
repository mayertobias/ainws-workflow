"""
Derived Features Calculator

Combines audio and content features to create higher-level derived features
used by both ML training and ML prediction services.

This ensures consistency between training and prediction pipelines.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class DerivedFeaturesCalculator:
    """
    Calculator for multimodal derived features that combine audio and content analysis.
    
    These derived features were originally computed in ML training but are now
    extracted to a shared library for consistency between training and prediction.
    """
    
    def __init__(self):
        self.derived_feature_names = [
            # Original derived features
            'rhythmic_appeal_index',
            'emotional_impact_score', 
            'commercial_viability_index',
            'sonic_sophistication_score',
            # Additional engineered features for feature balancing (from ML training orchestrator)
            'audio_energy_valence_ratio',
            'audio_rhythmic_intensity',
            'audio_timbral_complexity',
            'lyrics_word_count_normalized',
            'lyrics_unique_words_ratio',
            'lyrics_verse_count_normalized'
        ]
    
    def calculate_derived_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate all derived features from combined audio and content features.
        
        Args:
            features: Dictionary containing audio and content features
            
        Returns:
            Dictionary containing the original features plus derived features
        """
        # Convert to DataFrame for easier manipulation (preserving original logic)
        feature_df = pd.DataFrame([features])
        derived_features = {}
        
        try:
            # 1. RHYTHMIC APPEAL INDEX (tempo optimization + danceability)
            rhythmic_appeal = self._calculate_rhythmic_appeal_index(feature_df)
            if rhythmic_appeal is not None:
                derived_features['rhythmic_appeal_index'] = rhythmic_appeal
                
            # 2. EMOTIONAL IMPACT SCORE (combines valence, energy, and sentiment)
            emotional_impact = self._calculate_emotional_impact_score(feature_df)
            if emotional_impact is not None:
                derived_features['emotional_impact_score'] = emotional_impact
                
            # 3. COMMERCIAL VIABILITY INDEX (music theory-based hit prediction)
            commercial_viability = self._calculate_commercial_viability_index(feature_df)
            if commercial_viability is not None:
                derived_features['commercial_viability_index'] = commercial_viability
                
            # 4. SONIC SOPHISTICATION SCORE (complexity + production quality)
            sonic_sophistication = self._calculate_sonic_sophistication_score(feature_df)
            if sonic_sophistication is not None:
                derived_features['sonic_sophistication_score'] = sonic_sophistication
            
            # 5. ADDITIONAL ENGINEERED FEATURES FOR FEATURE BALANCING
            # (These were originally in ML training orchestrator's _apply_feature_balancing method)
            
            # Audio ratio features
            audio_energy_valence_ratio = self._calculate_audio_energy_valence_ratio(feature_df)
            if audio_energy_valence_ratio is not None:
                derived_features['audio_energy_valence_ratio'] = audio_energy_valence_ratio
                
            audio_rhythmic_intensity = self._calculate_audio_rhythmic_intensity(feature_df)
            if audio_rhythmic_intensity is not None:
                derived_features['audio_rhythmic_intensity'] = audio_rhythmic_intensity
                
            audio_timbral_complexity = self._calculate_audio_timbral_complexity(feature_df)
            if audio_timbral_complexity is not None:
                derived_features['audio_timbral_complexity'] = audio_timbral_complexity
            
            # Normalized lyrical features (prevent count-based dominance)
            lyrics_word_count_normalized = self._calculate_lyrics_word_count_normalized(feature_df)
            if lyrics_word_count_normalized is not None:
                derived_features['lyrics_word_count_normalized'] = lyrics_word_count_normalized
                
            lyrics_unique_words_ratio = self._calculate_lyrics_unique_words_ratio(feature_df)
            if lyrics_unique_words_ratio is not None:
                derived_features['lyrics_unique_words_ratio'] = lyrics_unique_words_ratio
                
            lyrics_verse_count_normalized = self._calculate_lyrics_verse_count_normalized(feature_df)
            if lyrics_verse_count_normalized is not None:
                derived_features['lyrics_verse_count_normalized'] = lyrics_verse_count_normalized
            
            logger.info(f"✅ Calculated {len(derived_features)} derived features: {list(derived_features.keys())}")
            
        except Exception as e:
            logger.error(f"❌ Error calculating derived features: {e}")
            
        return derived_features
    
    def _calculate_audio_energy_valence_ratio(self, feature_df: pd.DataFrame) -> Optional[float]:
        """
        Calculate audio energy to valence ratio for enhanced discrimination.
        
        Formula: energy / max(valence, 0.01) to prevent division by zero
        """
        energy_col = 'audio_energy'
        valence_col = 'audio_valence'
        
        if energy_col not in feature_df.columns or valence_col not in feature_df.columns:
            logger.warning(f"Missing features for audio_energy_valence_ratio: need {energy_col}, {valence_col}")
            return None
            
        try:
            energy = feature_df[energy_col].iloc[0]
            valence = feature_df[valence_col].iloc[0]
            
            # Prevent division by zero
            valence_safe = max(valence, 0.01)
            ratio = energy / valence_safe
            
            # Normalize to 0-1 range (assuming max ratio ~10)
            normalized_ratio = min(ratio / 10.0, 1.0)
            
            return float(normalized_ratio)
            
        except Exception as e:
            logger.error(f"Error calculating audio_energy_valence_ratio: {e}")
            return None
    
    def _calculate_audio_rhythmic_intensity(self, feature_df: pd.DataFrame) -> Optional[float]:
        """
        Calculate rhythmic intensity combining tempo and danceability.
        
        Formula: (tempo_normalized * danceability) for rhythmic drive
        """
        tempo_col = 'audio_tempo'
        danceability_col = 'audio_danceability'
        
        if tempo_col not in feature_df.columns or danceability_col not in feature_df.columns:
            logger.warning(f"Missing features for audio_rhythmic_intensity: need {tempo_col}, {danceability_col}")
            return None
            
        try:
            tempo = feature_df[tempo_col].iloc[0]
            danceability = feature_df[danceability_col].iloc[0]
            
            # Normalize tempo to 0-1 range (40-200 BPM typical range)
            tempo_normalized = max(0, min((tempo - 40) / 160, 1.0))
            
            # Combine with danceability
            rhythmic_intensity = tempo_normalized * danceability
            
            return float(rhythmic_intensity)
            
        except Exception as e:
            logger.error(f"Error calculating audio_rhythmic_intensity: {e}")
            return None
    
    def _calculate_audio_timbral_complexity(self, feature_df: pd.DataFrame) -> Optional[float]:
        """
        Calculate timbral complexity from brightness and harmonic strength.
        
        Formula: (brightness + harmonic_strength) / 2 for timbral richness
        """
        brightness_col = 'audio_brightness'
        harmonic_col = 'audio_harmonic_strength'
        
        available_cols = []
        if brightness_col in feature_df.columns:
            available_cols.append(brightness_col)
        if harmonic_col in feature_df.columns:
            available_cols.append(harmonic_col)
            
        if len(available_cols) == 0:
            logger.warning(f"Missing features for audio_timbral_complexity: need at least one of {brightness_col}, {harmonic_col}")
            return None
            
        try:
            if len(available_cols) == 2:
                # Use both features
                timbral_complexity = (feature_df[brightness_col].iloc[0] + feature_df[harmonic_col].iloc[0]) / 2
            else:
                # Use available feature as proxy
                timbral_complexity = feature_df[available_cols[0]].iloc[0]
            
            return float(timbral_complexity)
            
        except Exception as e:
            logger.error(f"Error calculating audio_timbral_complexity: {e}")
            return None
    
    def _calculate_lyrics_word_count_normalized(self, feature_df: pd.DataFrame) -> Optional[float]:
        """
        Calculate normalized lyrics word count to prevent scale dominance.
        
        Formula: min(word_count / 500, 1.0) to normalize to 0-1 range
        """
        word_count_col = 'lyrics_word_count'
        
        if word_count_col not in feature_df.columns:
            logger.warning(f"Missing feature for lyrics_word_count_normalized: need {word_count_col}")
            return None
            
        try:
            word_count = feature_df[word_count_col].iloc[0]
            
            # Normalize to 0-1 range (500 words as typical song length)
            normalized_count = min(word_count / 500.0, 1.0)
            
            return float(normalized_count)
            
        except Exception as e:
            logger.error(f"Error calculating lyrics_word_count_normalized: {e}")
            return None
    
    def _calculate_lyrics_unique_words_ratio(self, feature_df: pd.DataFrame) -> Optional[float]:
        """
        Calculate ratio of unique words to total words (vocabulary richness).
        
        Formula: unique_words / max(word_count, 1) to prevent division by zero
        """
        unique_words_col = 'lyrics_unique_words'
        word_count_col = 'lyrics_word_count'
        
        if unique_words_col not in feature_df.columns or word_count_col not in feature_df.columns:
            logger.warning(f"Missing features for lyrics_unique_words_ratio: need {unique_words_col}, {word_count_col}")
            return None
            
        try:
            unique_words = feature_df[unique_words_col].iloc[0]
            word_count = feature_df[word_count_col].iloc[0]
            
            # Calculate ratio (prevent division by zero)
            word_count_safe = max(word_count, 1)
            ratio = unique_words / word_count_safe
            
            # Ensure ratio is between 0 and 1
            ratio_clamped = max(0, min(ratio, 1.0))
            
            return float(ratio_clamped)
            
        except Exception as e:
            logger.error(f"Error calculating lyrics_unique_words_ratio: {e}")
            return None
    
    def _calculate_lyrics_verse_count_normalized(self, feature_df: pd.DataFrame) -> Optional[float]:
        """
        Calculate normalized lyrics verse count to prevent scale dominance.
        
        Formula: min(verse_count / 10, 1.0) to normalize to 0-1 range
        """
        verse_count_col = 'lyrics_verse_count'
        
        if verse_count_col not in feature_df.columns:
            logger.warning(f"Missing feature for lyrics_verse_count_normalized: need {verse_count_col}")
            return None
            
        try:
            verse_count = feature_df[verse_count_col].iloc[0]
            
            # Normalize to 0-1 range (10 verses as typical max)
            normalized_count = min(verse_count / 10.0, 1.0)
            
            return float(normalized_count)
            
        except Exception as e:
            logger.error(f"Error calculating lyrics_verse_count_normalized: {e}")
            return None
    
    def _calculate_rhythmic_appeal_index(self, feature_df: pd.DataFrame) -> Optional[float]:
        """
        Calculate rhythmic appeal index combining tempo optimization and danceability.
        
        Formula: tempo_normalized * 0.4 + danceability * 0.6
        """
        tempo_col = 'audio_tempo'
        danceability_col = 'audio_danceability'
        
        if tempo_col not in feature_df.columns or danceability_col not in feature_df.columns:
            logger.warning(f"Missing features for rhythmic_appeal_index: need {tempo_col}, {danceability_col}")
            return None
            
        try:
            tempo = feature_df[tempo_col].iloc[0]
            danceability = feature_df[danceability_col].iloc[0]
            
            # Tempo optimization curve (preserving original ML training logic)
            if 120 <= tempo <= 130:
                tempo_normalized = 1.0  # Optimal dance tempo
            elif 110 <= tempo <= 140:
                tempo_normalized = 0.8  # Good dance tempo
            elif 100 <= tempo <= 150:
                tempo_normalized = 0.6  # Acceptable
            else:
                tempo_normalized = 0.4  # Suboptimal
            
            rhythmic_appeal = tempo_normalized * 0.4 + danceability * 0.6
            return float(rhythmic_appeal)
            
        except Exception as e:
            logger.error(f"Error calculating rhythmic_appeal_index: {e}")
            return None
    
    def _calculate_emotional_impact_score(self, feature_df: pd.DataFrame) -> Optional[float]:
        """
        Calculate emotional impact score combining audio emotions and content sentiment.
        
        Formula: mean of [valence, energy, sentiment_positive (if available)]
        """
        valence_col = 'audio_valence'
        energy_col = 'audio_energy'
        sentiment_col = 'lyrics_sentiment_positive'  # May not always be available
        
        emotional_cols = []
        
        if valence_col in feature_df.columns:
            emotional_cols.append(valence_col)
        if energy_col in feature_df.columns:
            emotional_cols.append(energy_col)
        if sentiment_col in feature_df.columns:
            emotional_cols.append(sentiment_col)
            
        if len(emotional_cols) < 2:
            logger.warning(f"Insufficient features for emotional_impact_score: need at least 2 of {[valence_col, energy_col, sentiment_col]}")
            return None
            
        try:
            emotional_impact = feature_df[emotional_cols].mean(axis=1).iloc[0]
            return float(emotional_impact)
            
        except Exception as e:
            logger.error(f"Error calculating emotional_impact_score: {e}")
            return None
    
    def _calculate_commercial_viability_index(self, feature_df: pd.DataFrame) -> Optional[float]:
        """
        Calculate commercial viability index based on music theory hit prediction.
        
        Formula: mean of [danceability, energy, valence, joy (if available)]
        """
        commercial_features = []
        
        # Core commercial features
        potential_features = [
            'audio_danceability',
            'audio_energy', 
            'audio_valence',
            'lyrics_emotion_joy'  # May not always be available
        ]
        
        for feature in potential_features:
            if feature in feature_df.columns:
                commercial_features.append(feature)
                
        if len(commercial_features) < 2:
            logger.warning(f"Insufficient features for commercial_viability_index: need at least 2 of {potential_features}")
            return None
            
        try:
            commercial_viability = feature_df[commercial_features].mean(axis=1).iloc[0]
            return float(commercial_viability)
            
        except Exception as e:
            logger.error(f"Error calculating commercial_viability_index: {e}")
            return None
    
    def _calculate_sonic_sophistication_score(self, feature_df: pd.DataFrame) -> Optional[float]:
        """
        Calculate sonic sophistication score combining complexity and production quality.
        
        Formula: mean of [audio_complexity, audio_brightness, lyrics_complexity_score (if available)]
        """
        sophistication_cols = []
        
        # Core sophistication features
        potential_features = [
            'audio_complexity',
            'audio_brightness',
            'lyrics_complexity_score'  # May not always be available
        ]
        
        for feature in potential_features:
            if feature in feature_df.columns:
                sophistication_cols.append(feature)
                
        if len(sophistication_cols) < 2:
            logger.warning(f"Insufficient features for sonic_sophistication_score: need at least 2 of {potential_features}")
            return None
            
        try:
            sonic_sophistication = feature_df[sophistication_cols].mean(axis=1).iloc[0]
            return float(sonic_sophistication)
            
        except Exception as e:
            logger.error(f"Error calculating sonic_sophistication_score: {e}")
            return None
    
    def get_required_features_for_derived_features(self) -> Dict[str, List[str]]:
        """
        Get the mapping of derived features to their required input features.
        
        Returns:
            Dictionary mapping derived feature names to required input features
        """
        return {
            'rhythmic_appeal_index': ['audio_tempo', 'audio_danceability'],
            'emotional_impact_score': ['audio_valence', 'audio_energy', 'lyrics_sentiment_positive'],  # lyrics_sentiment_positive is optional
            'commercial_viability_index': ['audio_danceability', 'audio_energy', 'audio_valence', 'lyrics_emotion_joy'],  # lyrics_emotion_joy is optional
            'sonic_sophistication_score': ['audio_complexity', 'audio_brightness', 'lyrics_complexity_score'],  # lyrics_complexity_score is optional
            # Additional engineered features for feature balancing
            'audio_energy_valence_ratio': ['audio_energy', 'audio_valence'],
            'audio_rhythmic_intensity': ['audio_tempo', 'audio_danceability'],
            'audio_timbral_complexity': ['audio_brightness', 'audio_harmonic_strength'],  # audio_harmonic_strength is optional
            'lyrics_word_count_normalized': ['lyrics_word_count'],
            'lyrics_unique_words_ratio': ['lyrics_unique_words', 'lyrics_word_count'],
            'lyrics_verse_count_normalized': ['lyrics_verse_count']
        }
    
    def get_available_derived_features(self, features: Dict[str, Any]) -> List[str]:
        """
        Get list of derived features that can be calculated from available input features.
        
        Args:
            features: Dictionary containing available input features
            
        Returns:
            List of derived feature names that can be calculated
        """
        requirements = self.get_required_features_for_derived_features()
        available_derived = []
        
        for derived_feature, required_features in requirements.items():
            # Check if we have at least the minimum required features
            if derived_feature == 'rhythmic_appeal_index':
                # Requires both tempo and danceability
                if all(f in features for f in ['audio_tempo', 'audio_danceability']):
                    available_derived.append(derived_feature)
            elif derived_feature == 'emotional_impact_score':
                # Requires at least valence and energy (sentiment is optional)
                if all(f in features for f in ['audio_valence', 'audio_energy']):
                    available_derived.append(derived_feature)
            elif derived_feature == 'commercial_viability_index':
                # Requires at least 2 of the core features
                core_features = ['audio_danceability', 'audio_energy', 'audio_valence']
                available_core = [f for f in core_features if f in features]
                if len(available_core) >= 2:
                    available_derived.append(derived_feature)
            elif derived_feature == 'sonic_sophistication_score':
                # Requires at least 2 of the core features
                core_features = ['audio_complexity', 'audio_brightness']
                available_core = [f for f in core_features if f in features]
                if len(available_core) >= 2:
                    available_derived.append(derived_feature)
            # Additional engineered features for feature balancing
            elif derived_feature == 'audio_energy_valence_ratio':
                if all(f in features for f in ['audio_energy', 'audio_valence']):
                    available_derived.append(derived_feature)
            elif derived_feature == 'audio_rhythmic_intensity':
                if all(f in features for f in ['audio_tempo', 'audio_danceability']):
                    available_derived.append(derived_feature)
            elif derived_feature == 'audio_timbral_complexity':
                # Requires at least one of the timbral features
                timbral_features = ['audio_brightness', 'audio_harmonic_strength']
                available_timbral = [f for f in timbral_features if f in features]
                if len(available_timbral) >= 1:
                    available_derived.append(derived_feature)
            elif derived_feature == 'lyrics_word_count_normalized':
                if 'lyrics_word_count' in features:
                    available_derived.append(derived_feature)
            elif derived_feature == 'lyrics_unique_words_ratio':
                if all(f in features for f in ['lyrics_unique_words', 'lyrics_word_count']):
                    available_derived.append(derived_feature)
            elif derived_feature == 'lyrics_verse_count_normalized':
                if 'lyrics_verse_count' in features:
                    available_derived.append(derived_feature)
        
        return available_derived

# Test validation
if __name__ == "__main__":
    calculator = DerivedFeaturesCalculator()
    
    # Test with sample features
    test_features = {
        'audio_tempo': 125.0,
        'audio_danceability': 0.8,
        'audio_valence': 0.7,
        'audio_energy': 0.9,
        'audio_complexity': 0.6,
        'audio_brightness': 0.7
    }
    
    derived = calculator.calculate_derived_features(test_features)
    print("Derived features test:", derived)
    
    available = calculator.get_available_derived_features(test_features)
    print("Available derived features:", available)