from .base import BaseFeatureExtractor

class DynamicsExtractor(BaseFeatureExtractor):
    """
    Main dynamics feature extractor that analyzes the dynamic characteristics of audio.
    Provides comprehensive dynamics features including energy and loudness analysis.
    """
    
    def __init__(self):
        """Initialize sub-extractors"""
        self.energy_extractor = EnergyExtractor()
    
    def extract(self, data):
        """Extract comprehensive dynamics features from audio data."""
        features = {}
        
        try:
            # Extract energy feature
            features['energy'] = self.energy_extractor.extract(data)
            
            # Add raw dynamics data if available
            if 'lowlevel' in data:
                lowlevel_data = data['lowlevel']
                features.update({
                    'spectral_energy': lowlevel_data.get('spectral_energy', {}),
                    'loudness_ebu128': lowlevel_data.get('loudness_ebu128', {}),
                    'dynamic_complexity': lowlevel_data.get('dynamic_complexity', 0),
                    'loudness_lufs': lowlevel_data.get('loudness_lufs', {})
                })
                
        except Exception as e:
            # If data extraction fails, return default values
            features = {
                'energy': 0.5,
                'spectral_energy': {},
                'loudness_ebu128': {},
                'dynamic_complexity': 0,
                'loudness_lufs': {},
                'error': str(e)
            }
        
        return features

class EnergyExtractor(BaseFeatureExtractor):
    """
    Extracts energy feature from audio data.
    Represents the perceived intensity and activity in the track,
    based on spectral energy, loudness, and dynamic complexity.
    """
    
    def extract(self, data):
        """Calculate energy based on spectral and loudness features."""
        spectral_energy = data['lowlevel']['spectral_energy']['mean']
        loudness = -data['lowlevel']['loudness_ebu128']['integrated']  # Convert to positive
        dynamic_complexity = data['lowlevel']['dynamic_complexity']
        
        energy_score = 0.5 * self.normalize(spectral_energy, 0, 0.1) + \
                      0.3 * self.normalize(loudness, 0, 60) + \
                      0.2 * (1 - self.normalize(dynamic_complexity, 0, 10))
        
        return min(max(energy_score, 0), 1) 