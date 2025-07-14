from .base import BaseFeatureExtractor

class TimbreExtractor(BaseFeatureExtractor):
    """
    Main timbre feature extractor that combines all timbral analysis components.
    Provides comprehensive timbre features including acousticness, instrumentalness, 
    liveness, speechiness, brightness, complexity, and warmth.
    """
    
    def __init__(self):
        """Initialize sub-extractors"""
        self.acousticness_extractor = AcousticnessExtractor()
        self.instrumentalness_extractor = InstrumentalnessExtractor()
        self.liveness_extractor = LivenessExtractor()
        self.speechiness_extractor = SpeechinessExtractor()
        self.brightness_extractor = BrightnessExtractor()
        self.complexity_extractor = ComplexityExtractor()
        self.warmth_extractor = WarmthExtractor()
    
    def extract(self, data):
        """Extract comprehensive timbre features from audio data."""
        features = {}
        
        try:
            # Extract all timbre features
            features['acousticness'] = self.acousticness_extractor.extract(data)
            features['instrumentalness'] = self.instrumentalness_extractor.extract(data)
            features['liveness'] = self.liveness_extractor.extract(data)
            features['speechiness'] = self.speechiness_extractor.extract(data)
            features['brightness'] = self.brightness_extractor.extract(data)
            features['complexity'] = self.complexity_extractor.extract(data)
            features['warmth'] = self.warmth_extractor.extract(data)
            
            # Add raw spectral data if available
            if 'lowlevel' in data:
                lowlevel_data = data['lowlevel']
                features.update({
                    'spectral_centroid': lowlevel_data.get('spectral_centroid', {}),
                    'spectral_spread': lowlevel_data.get('spectral_spread', {}),
                    'spectral_entropy': lowlevel_data.get('spectral_entropy', {}),
                    'spectral_complexity': lowlevel_data.get('spectral_complexity', {}),
                    'spectral_flux': lowlevel_data.get('spectral_flux', {}),
                    'spectral_rolloff': lowlevel_data.get('spectral_rolloff', {}),
                    'spectral_kurtosis': lowlevel_data.get('spectral_kurtosis', {})
                })
                
        except Exception as e:
            # If data extraction fails, return default values
            features = {
                'acousticness': 0.5,
                'instrumentalness': 0.5,
                'liveness': 0.1,
                'speechiness': 0.1,
                'brightness': 0.5,
                'complexity': 0.5,
                'warmth': 0.5,
                'spectral_centroid': {},
                'spectral_spread': {},
                'spectral_entropy': {},
                'spectral_complexity': {},
                'spectral_flux': {},
                'spectral_rolloff': {},
                'spectral_kurtosis': {},
                'error': str(e)
            }
        
        return features

class AcousticnessExtractor(BaseFeatureExtractor):
    """
    Extracts acousticness feature from audio data.
    Represents the degree to which the track uses acoustic instruments
    rather than electronic/synthetic sounds.
    """
    
    def extract(self, data):
        """Calculate how acoustic (vs. electronic) the track sounds."""
        spectral_centroid = data['lowlevel']['spectral_centroid']['mean']
        spectral_entropy = data['lowlevel']['spectral_entropy']['mean']
        
        # Lower centroid and entropy usually indicate more acoustic sound
        acousticness_score = 1 - self.normalize(spectral_centroid, 0, 5000) * self.normalize(spectral_entropy, 0, 10)
        
        return min(max(acousticness_score, 0), 1)


class InstrumentalnessExtractor(BaseFeatureExtractor):
    """
    Extracts instrumentalness feature from audio data.
    Predicts whether a track contains no vocals. Higher values represent
    instrumental tracks, while lower values suggest the presence of vocals.
    """
    
    def extract(self, data):
        """Calculate how instrumental (vs. vocal) the track is."""
        spectral_complexity = data['lowlevel']['spectral_complexity']['mean']
        silence_rate_30dB = data['lowlevel']['silence_rate_30dB']['mean']
        
        # High complexity and low silence often indicate instrumental music
        instrumentalness_score = self.normalize(spectral_complexity, 0, 60) * (1 - silence_rate_30dB)
        
        return min(max(instrumentalness_score, 0), 1)


class LivenessExtractor(BaseFeatureExtractor):
    """
    Extracts liveness feature from audio data.
    Detects the presence of an audience in the recording,
    with higher values representing a higher probability of live performance.
    """
    
    def extract(self, data):
        """Calculate how likely the track is a live recording."""
        spectral_spread = data['lowlevel']['spectral_spread']['mean']
        spectral_kurtosis = data['lowlevel']['spectral_kurtosis']['mean']
        
        # Live recordings often have higher spread and specific kurtosis patterns
        liveness_score = self.normalize(spectral_spread, 0, 10000000) * self.normalize(spectral_kurtosis, 0, 200)
        
        return min(max(liveness_score, 0), 1)


class SpeechinessExtractor(BaseFeatureExtractor):
    """
    Extracts speechiness feature from audio data.
    Detects the presence of spoken words in a track,
    with higher values indicating more speech-like content.
    """
    
    def extract(self, data):
        """Calculate presence of spoken words vs. music."""
        spectral_flux = data['lowlevel']['spectral_flux']['mean']
        silence_rate_60dB = data['lowlevel']['silence_rate_60dB']['mean']
        
        # Speech has characteristic flux patterns and silence rates
        speechiness_score = self.normalize(spectral_flux, 0, 0.5) * self.normalize(silence_rate_60dB, 0, 1)
        
        return min(max(speechiness_score, 0), 1)


class BrightnessExtractor(BaseFeatureExtractor):
    """
    Extracts brightness feature from audio data.
    Represents the amount of high-frequency content in the track,
    with higher values indicating a brighter, more treble-heavy sound.
    """
    
    def extract(self, data):
        """Calculate brightness of the track."""
        # Brightness correlates with higher centroid values
        spectral_centroid = data['lowlevel']['spectral_centroid']['mean']
        spectral_rolloff = data['lowlevel']['spectral_rolloff']['mean']
        
        brightness_score = 0.7 * self.normalize(spectral_centroid, 500, 8000) + \
                          0.3 * self.normalize(spectral_rolloff, 1000, 15000)
        
        return min(max(brightness_score, 0), 1)


class ComplexityExtractor(BaseFeatureExtractor):
    """
    Extracts complexity feature from audio data.
    Represents the musical complexity in terms of spectral, dynamic,
    and harmonic variation.
    """
    
    def extract(self, data):
        """Calculate musical complexity of the track."""
        # Using several factors that contribute to perceived complexity
        spectral_complexity = data['lowlevel']['spectral_complexity']['mean']
        dynamic_complexity = data['lowlevel']['dynamic_complexity']
        chord_changes = data['tonal']['chords_changes_rate']
        
        complexity_score = 0.4 * self.normalize(spectral_complexity, 5, 50) + \
                          0.3 * self.normalize(dynamic_complexity, 0, 10) + \
                          0.3 * self.normalize(chord_changes, 0, 0.2)
        
        return min(max(complexity_score, 0), 1)


class WarmthExtractor(BaseFeatureExtractor):
    """
    Extracts warmth feature from audio data.
    Represents the timbral warmth, which typically correlates with
    a good balance of low-mids and a smooth high-end.
    """
    
    def extract(self, data):
        """Calculate warmth of the track."""
        # Warm sound usually has good low-mid content and smooth high end
        spectral_centroid = data['lowlevel']['spectral_centroid']['mean']
        spectral_kurtosis = data['lowlevel']['spectral_kurtosis']['mean']
        
        # Lower centroid tends to sound warmer, higher kurtosis indicates smoother high end
        warmth_score = 0.7 * (1 - self.normalize(spectral_centroid, 500, 5000)) + \
                      0.3 * self.normalize(spectral_kurtosis, 0, 20)
        
        return min(max(warmth_score, 0), 1) 