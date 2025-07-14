from .base import BaseFeatureExtractor

class TonalExtractor(BaseFeatureExtractor):
    """
    Main tonal feature extractor that combines all tonal analysis components.
    Provides comprehensive tonal features including valence, harmonic strength, key, and mode.
    """
    
    def __init__(self):
        """Initialize sub-extractors"""
        self.valence_extractor = ValenceExtractor()
        self.harmonic_strength_extractor = HarmonicStrengthExtractor()
        self.key_extractor = KeyExtractor()
        self.mode_extractor = ModeExtractor()
    
    def extract(self, data):
        """Extract comprehensive tonal features from audio data."""
        features = {}
        
        try:
            # Extract valence (emotional positivity)
            features['valence'] = self.valence_extractor.extract(data)
            
            # Extract harmonic strength 
            features['harmonic_strength'] = self.harmonic_strength_extractor.extract(data)
            
            # Extract musical key
            features['key'] = self.key_extractor.extract(data)
            
            # Extract mode (major/minor)
            features['mode'] = self.mode_extractor.extract(data)
            
            # Add raw tonal data if available
            if 'tonal' in data:
                tonal_data = data['tonal']
                features.update({
                    'key_krumhansl': tonal_data.get('key_krumhansl', {}),
                    'chords_strength': tonal_data.get('chords_strength', {}),
                    'chords_histogram': tonal_data.get('chords_histogram', {}),
                    'tuning_frequency': tonal_data.get('tuning_frequency', 440.0)
                })
                
        except Exception as e:
            # If data extraction fails, return default values
            features = {
                'valence': 0.5,  # neutral valence
                'harmonic_strength': 0.0,
                'key': 'C',  # default key
                'mode': 1,  # major mode
                'key_krumhansl': {},
                'chords_strength': {},
                'chords_histogram': {},
                'tuning_frequency': 440.0,
                'error': str(e)
            }
        
        return features

class ValenceExtractor(BaseFeatureExtractor):
    """
    Extracts valence feature from audio data.
    Represents the musical positiveness conveyed by the track,
    with higher values indicating more positive mood (happiness, cheerfulness).
    """
    
    def extract(self, data):
        """Calculate valence (happiness/positivity) of the track."""
        key_scale = data['tonal']['key_krumhansl']['scale']
        spectral_spread = data['lowlevel']['spectral_spread']['mean']
        
        # Major keys are generally happier
        valence_score = 0.6 if key_scale == 'major' else 0.4
        # Brighter sounds (higher spread) tend to be happier
        valence_score += 0.4 * self.normalize(spectral_spread, 0, 10000000)
        
        return min(max(valence_score, 0), 1)


class HarmonicStrengthExtractor(BaseFeatureExtractor):
    """
    Extracts harmonic strength feature from audio data.
    Represents the clarity and strength of the harmonic content,
    using key and chord strengths as indicators.
    """
    
    def extract(self, data):
        """Calculate harmonic strength/quality of the track."""
        key_strength = data['tonal']['key_krumhansl']['strength']
        chord_strength = data['tonal']['chords_strength']['mean']
        
        harmonic_score = 0.6 * self.normalize(key_strength, 0.3, 0.9) + \
                        0.4 * self.normalize(chord_strength, 0.3, 0.9)
        
        return min(max(harmonic_score, 0), 1)


class KeyExtractor(BaseFeatureExtractor):
    """
    Extracts musical key from audio data.
    Represents the key of the track (C, C#, D, etc.) using the Krumhansl key detection algorithm.
    Returns a string value.
    """
    
    # Mapping of key names to numerical values (C=0, C#=1, etc.)
    KEY_MAPPING = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
        'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11,
        'Db': 1, 'Eb': 3, 'Gb': 6, 'Ab': 8, 'Bb': 10
    }
    
    def extract(self, data):
        """Extract the musical key of the track."""
        # Use krumhansl algorithm from Essentia as it's generally reliable
        key = data['tonal']['key_krumhansl']['key']
        
        # Return the key name as is (no normalization needed as it's categorical)
        return key


class ModeExtractor(BaseFeatureExtractor):
    """
    Extracts musical mode from audio data.
    Represents whether the track is in a major (1) or minor (0) scale.
    """
    
    def extract(self, data):
        """Extract the musical mode (major/minor) of the track."""
        # Get the scale from Krumhansl algorithm
        mode = data['tonal']['key_krumhansl']['scale']
        
        # Convert to binary value: 1 for major, 0 for minor
        mode_value = 1 if mode.lower() == 'major' else 0
        
        return mode_value 