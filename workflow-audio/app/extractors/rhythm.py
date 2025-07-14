from .base import BaseFeatureExtractor

class RhythmExtractor(BaseFeatureExtractor):
    """
    Main rhythm feature extractor that combines tempo and danceability analysis.
    Provides comprehensive rhythm-based features for audio analysis.
    """
    
    def __init__(self):
        """Initialize sub-extractors"""
        self.tempo_extractor = TempoExtractor()
        self.danceability_extractor = DanceabilityExtractor()
    
    def extract(self, data):
        """Extract comprehensive rhythm features from audio data."""
        features = {}
        
        try:
            # Extract tempo (BPM)
            features['tempo'] = self.tempo_extractor.extract(data)
            
            # Extract danceability
            features['danceability'] = self.danceability_extractor.extract(data)
            
            # Add raw rhythm data if available
            if 'rhythm' in data:
                rhythm_data = data['rhythm']
                features.update({
                    'bpm': rhythm_data.get('bpm', 0),
                    'onset_rate': rhythm_data.get('onset_rate', 0),
                    'beats_loudness': rhythm_data.get('beats_loudness', {}),
                    'beats_strength': rhythm_data.get('beats_strength', 0)
                })
                
        except Exception as e:
            # If data extraction fails, return default values
            features = {
                'tempo': 0,
                'danceability': 0,
                'bpm': 0,
                'onset_rate': 0,
                'beats_loudness': {},
                'beats_strength': 0,
                'error': str(e)
            }
        
        return features

class TempoExtractor(BaseFeatureExtractor):
    """
    Extracts tempo (BPM - Beats Per Minute) feature from audio data.
    Represents the speed of the track as its actual BPM.
    """
    
    def extract(self, data):
        """Extract tempo (BPM) feature."""
        bpm = data['rhythm']['bpm']
        # Original normalization (kept for reference):
        # return self.normalize(bpm, 50, 200)
        return bpm


class DanceabilityExtractor(BaseFeatureExtractor):
    """
    Extracts danceability feature from audio data.
    Combines BPM, onset rate, and beats loudness to estimate how suitable
    the track is for dancing.
    """
    
    def extract(self, data):
        """Calculate danceability based on rhythm features."""
        bpm = data['rhythm']['bpm']
        onset_rate = data['rhythm']['onset_rate']
        beats_loudness_mean = data['rhythm']['beats_loudness']['mean']
        
        # Combine rhythm factors that contribute to danceability
        dance_score = 0.4 * self.normalize(bpm, 80, 160) + \
                     0.4 * self.normalize(onset_rate, 0, 10) + \
                     0.2 * self.normalize(beats_loudness_mean, 0, 0.2)
        
        return min(max(dance_score, 0), 1) 