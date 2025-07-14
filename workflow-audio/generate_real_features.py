#!/usr/bin/env python3
"""
Generate real audio features using Essentia and save to JSON
This will help us understand what actual features look like
"""

import json
import essentia.standard as es
from pathlib import Path

def extract_real_features(audio_path: str):
    """Extract real features and show structure"""
    print(f"🎵 Extracting real features from: {audio_path}")
    
    try:
        # Initialize MusicExtractor
        extractor = es.MusicExtractor()
        
        # Extract features
        features_pool, features_frames_pool = extractor(str(audio_path))
        
        # Create a proper structured data for basic features
        def get_feature_value(key: str, default=0.0):
            try:
                value = features_pool[key]
                if hasattr(value, '__len__') and len(value) > 0:
                    return float(value[0]) if isinstance(value[0], (int, float)) else value[0]
                elif isinstance(value, (int, float)):
                    return float(value)
                return value
            except:
                return default
        
        # Build comprehensive real features
        real_features = {
            'lowlevel': {
                'spectral_centroid': {'mean': get_feature_value('lowlevel.spectral_centroid.mean')},
                'spectral_entropy': {'mean': get_feature_value('lowlevel.spectral_entropy.mean')},
                'spectral_spread': {'mean': get_feature_value('lowlevel.spectral_spread.mean')},
                'spectral_flux': {'mean': get_feature_value('lowlevel.spectral_flux.mean')},
                'spectral_kurtosis': {'mean': get_feature_value('lowlevel.spectral_kurtosis.mean')},
                'spectral_complexity': {'mean': get_feature_value('lowlevel.spectral_complexity.mean')},
                'spectral_rolloff': {'mean': get_feature_value('lowlevel.spectral_rolloff.mean')},
                'silence_rate_30dB': {'mean': get_feature_value('lowlevel.silence_rate_30dB.mean')},
                'silence_rate_60dB': {'mean': get_feature_value('lowlevel.silence_rate_60dB.mean')},
                'spectral_energy': {'mean': get_feature_value('lowlevel.spectral_energy.mean')},
                'dynamic_complexity': get_feature_value('lowlevel.dynamic_complexity'),
                'loudness_ebu128': {
                    'integrated': get_feature_value('lowlevel.loudness_ebu128.integrated'),
                    'loudness_range': get_feature_value('lowlevel.loudness_ebu128.loudness_range')
                }
            },
            'tonal': {
                'chords_changes_rate': get_feature_value('tonal.chords_changes_rate'),
                'chords_strength': {'mean': get_feature_value('tonal.chords_strength.mean')},
                'key_krumhansl': {
                    'key': get_feature_value('tonal.key_krumhansl.key', 'C'),
                    'scale': get_feature_value('tonal.key_krumhansl.scale', 'major'),
                    'strength': get_feature_value('tonal.key_krumhansl.strength')
                }
            },
            'rhythm': {
                'bpm': get_feature_value('rhythm.bpm'),
                'danceability': get_feature_value('rhythm.danceability'),
                'onset_rate': get_feature_value('rhythm.onset_rate'),
                'beats_loudness': {'mean': get_feature_value('rhythm.beats_loudness.mean')}
            },
            'metadata': {
                'audio_properties': {
                    'length': get_feature_value('metadata.audio_properties.length'),
                    'sample_rate': get_feature_value('metadata.audio_properties.sample_rate')
                }
            }
        }
        
        # Calculate derived features (like the working reference file shows)
        def normalize(value, min_val, max_val):
            if min_val == max_val:
                return 0.0
            norm = (value - min_val) / (max_val - min_val)
            return max(0.0, min(1.0, norm))
        
        # Calculate high-level features using real data
        spectral_centroid = real_features['lowlevel']['spectral_centroid']['mean']
        spectral_entropy = real_features['lowlevel']['spectral_entropy']['mean']
        spectral_rolloff = real_features['lowlevel']['spectral_rolloff']['mean']
        loudness = real_features['lowlevel']['loudness_ebu128']['integrated']
        bpm = real_features['rhythm']['bpm']
        
        derived_features = {
            'energy': normalize(-loudness, 0, 60),  # Convert negative loudness to positive energy
            'valence': 0.4 + 0.6 * normalize(spectral_centroid, 500, 5000),  # Brightness affects valence
            'tempo': bpm,
            'danceability': real_features['rhythm']['danceability'],
            'acousticness': 1 - normalize(spectral_centroid, 0, 5000) * normalize(spectral_entropy, 0, 10),
            'instrumentalness': normalize(real_features['lowlevel']['spectral_complexity']['mean'], 0, 60),
            'liveness': normalize(real_features['lowlevel']['spectral_spread']['mean'], 0, 10000000) * 0.1,
            'speechiness': normalize(real_features['lowlevel']['spectral_flux']['mean'], 0, 0.5),
            'loudness': loudness,
            'brightness': 0.7 * normalize(spectral_centroid, 500, 8000) + 0.3 * normalize(spectral_rolloff, 1000, 15000),
            'complexity': normalize(real_features['lowlevel']['spectral_complexity']['mean'], 5, 50),
            'warmth': 1 - normalize(spectral_centroid, 500, 5000),
            'harmonic_strength': real_features['tonal']['key_krumhansl']['strength'],
            'key': real_features['tonal']['key_krumhansl']['key'],
            'mode': 1 if real_features['tonal']['key_krumhansl']['scale'] == 'major' else 0,
            'time_signature': 4,  # Default for most music
            'duration_ms': int(real_features['metadata']['audio_properties']['length'] * 1000)
        }
        
        # Combine all features
        complete_analysis = {
            'structured_data': real_features,
            'derived_features': derived_features,
            'raw_essentia_sample': {
                'lowlevel.spectral_centroid.mean': get_feature_value('lowlevel.spectral_centroid.mean'),
                'lowlevel.spectral_rolloff.mean': get_feature_value('lowlevel.spectral_rolloff.mean'),
                'rhythm.bpm': get_feature_value('rhythm.bpm'),
                'rhythm.danceability': get_feature_value('rhythm.danceability'),
                'tonal.key_krumhansl.key': get_feature_value('tonal.key_krumhansl.key'),
                'tonal.key_krumhansl.scale': get_feature_value('tonal.key_krumhansl.scale')
            }
        }
        
        # Save to file
        output_file = 'real_features_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(complete_analysis, f, indent=2)
        
        print(f"✅ Real features saved to: {output_file}")
        print(f"🎹 Key: {derived_features['key']} {real_features['tonal']['key_krumhansl']['scale']}")
        print(f"🎵 Tempo: {derived_features['tempo']:.1f} BPM")
        print(f"⚡ Energy: {derived_features['energy']:.3f}")
        print(f"💃 Danceability: {derived_features['danceability']:.3f}")
        print(f"🔆 Valence: {derived_features['valence']:.3f}")
        
        return complete_analysis
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    audio_file = "/Users/manojveluchuri/saas/workflow/workflow-audio/uploads/billie_jean.mp3"
    if Path(audio_file).exists():
        features = extract_real_features(audio_file)
    else:
        print(f"❌ Audio file not found: {audio_file}")