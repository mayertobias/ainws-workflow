"""
Complete Audio Analysis Service using Essentia

This module provides audio feature extraction using Essentia's MusicExtractor
with modular feature extractors for comprehensive audio analysis.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json 
import os

# Initialize logger first
logger = logging.getLogger(__name__)

# Import essentia
try:
    import essentia
    import essentia.standard as es
    logger.info(f"Successfully imported Essentia version: {essentia.__version__}")
except ImportError as e:
    logger.error(f"Failed to import Essentia: {e}")
    raise

def normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value between 0 and 1 based on min and max ranges."""
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0

class AudioAnalyzer:
    """
    Service for extracting audio features using Essentia.

    This class provides a clean interface to Essentia's MusicExtractor for extracting
    audio features from music files. It handles the initialization of Essentia and
    provides error handling for file operations and feature extraction.

    Returns raw feature values. Scaling/Normalization happens in the Celery task.
    """

    def __init__(self):
        """Initialize the audio analyzer with Essentia's MusicExtractor."""
        self._initialize_analyzer()

    def _initialize_analyzer(self) -> None:
        """Initialize the Essentia MusicExtractor with appropriate configuration."""
        try:
            # Use default 'music' profile instead of custom profile
            logger.info("Initializing Essentia MusicExtractor with default 'music' profile")
            self.extractor = es.MusicExtractor()
            logger.info("Successfully initialized Essentia MusicExtractor with default profile")
        except Exception as e:
            logger.error(f"Failed to initialize Essentia analyzer: {e}")
            raise

    def calculate_high_level_features(self, raw_features: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate high-level features from raw audio features.
        
        Args:
            raw_features: Dictionary containing raw audio features
            
        Returns:
            Dictionary containing high-level features normalized between 0 and 1
        """
        hlf = {}
        
        # Danceability
        bpm = raw_features['tempo']
        onset_rate = raw_features['onset_rate']
        beats_loudness_mean = raw_features['beats_loudness_mean']
        danceability = (0.4 * normalize(bpm, 60, 120) +
                       0.4 * normalize(onset_rate, 0, 10) +
                       0.2 * normalize(beats_loudness_mean, 0, 0.5))
        hlf['danceability'] = min(max(danceability, 0), 1)

        # Energy
        spectral_energy = raw_features['spectral_energy_mean']
        loudness = -raw_features['loudness_ebu128_integrated']  # Convert to positive
        dynamic_complexity = raw_features['dynamic_complexity']
        energy = (0.4 * normalize(spectral_energy, 0, 0.1) +
                 0.4 * normalize(loudness, 0, 60) +
                 0.2 * (1 - normalize(dynamic_complexity, 0, 10)))
        hlf['energy'] = min(max(energy, 0), 1)

        # Valence
        key_scale = raw_features['mode']
        spectral_spread = raw_features['spectral_spread_mean']
        valence = 0.4 if key_scale == 'major' else 0.3
        valence += 0.6 * normalize(spectral_spread, 0, 10000000)
        hlf['valence'] = min(max(valence, 0), 1)

        # Acousticness
        spectral_centroid = raw_features['spectral_centroid_mean']
        spectral_entropy = raw_features['spectral_entropy_mean']
        acousticness = 1 - normalize(spectral_centroid, 0, 5000) * normalize(spectral_entropy, 0, 10)
        hlf['acousticness'] = min(max(acousticness, 0), 1)

        # Instrumentalness
        spectral_complexity = raw_features['spectral_complexity_mean']
        silence_rate_60dB = raw_features['silence_rate_60dB_mean']
        instrumentalness = (0.7 * normalize(spectral_complexity, 0, 60) *
                          (1 - silence_rate_60dB))
        hlf['instrumentalness'] = min(max(instrumentalness, 0), 1)

        # Liveness
        spectral_spread = raw_features['spectral_spread_mean']
        spectral_kurtosis = raw_features['spectral_kurtosis_mean']
        liveness = normalize(spectral_spread, 0, 10000000) * normalize(spectral_kurtosis, 0, 200)
        hlf['liveness'] = min(max(liveness, 0), 1)

        # Speechiness
        spectral_flux = raw_features['spectral_flux_mean']
        speechiness = (0.6 * normalize(spectral_flux, 0, 0.5) +
                      0.4 * normalize(silence_rate_60dB, 0, 1))
        hlf['speechiness'] = min(max(speechiness, 0), 1)

        # Grooviness
        grooviness = (0.4 * normalize(onset_rate, 0, 10) +
                     0.4 * normalize(bpm, 60, 120) +
                     0.2 * normalize(beats_loudness_mean, 0, 0.5))
        hlf['grooviness'] = min(max(grooviness, 0), 1)

        return hlf

    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Extract audio features from a file using Essentia's MusicExtractor.

        Args:
            audio_path: Path to the audio file

        Returns:
            Dictionary containing raw audio features extracted by Essentia

        Raises:
            FileNotFoundError: If the audio file doesn't exist
            Exception: If feature extraction fails
        """
        audio_file_path = Path(audio_path)
        if not audio_file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            logger.info(f"Analyzing audio file: {audio_path}")

            # Extract features using MusicExtractor
            features_pool, features_frames_pool = self.extractor(str(audio_file_path))
            # logger.info(f"Available Essentia descriptor names: {features_pool.descriptorNames()}")
            raw_features_dict = {}
            available_keys = features_pool.descriptorNames()
            logger.debug(f"Available keys in features_pool: {available_keys}")

            # --- Helper function for safe access ---
            def get_feature_value(key_list: list, default: Any = 0.0) -> Any:
                """Safely gets a value trying multiple keys from the pool."""
                for key in key_list:
                    if key in available_keys:
                        try:
                            value = features_pool[key]
                            if isinstance(value, (np.ndarray, list)) and len(value) >= 1:
                                if isinstance(value[0], (int, float, np.number)):
                                     return float(value[0])
                                elif isinstance(value[0], str):
                                     return str(value[0])
                            elif isinstance(value, (int, float, np.number)):
                                return float(value)
                            elif isinstance(value, str):
                                return str(value)
                            else:
                                logger.warning(f"Descriptor '{key}' has non-scalar type {type(value)}. Returning as is or default.")
                                return value
                        except KeyError:
                            logger.warning(f"KeyError accessing '{key}'. Trying next key or using default.")
                            continue
                        except Exception as e:
                            logger.error(f"Error processing value for key '{key}': {e}")
                            continue

                logger.warning(f"None of the keys {key_list} found or valid in Essentia output. Using default value {default}.")
                return default

            def get_raw_features(features_pool) -> Dict[str, Any]:
                """Extract raw features from Essentia features pool using the new extraction method."""
                raw_features = {}
                
                # Rhythm features
                raw_features['tempo'] = get_feature_value(['rhythm.bpm'])
                raw_features['onset_rate'] = get_feature_value(['rhythm.onset_rate'])
                raw_features['beats_loudness_mean'] = get_feature_value(['rhythm.beats_loudness.mean'])
                
                # Low-level features
                raw_features['spectral_energy_mean'] = get_feature_value(['lowlevel.spectral_energy.mean'])
                raw_features['loudness_ebu128_integrated'] = get_feature_value(['lowlevel.loudness_ebu128.integrated'])
                raw_features['dynamic_complexity'] = get_feature_value(['lowlevel.dynamic_complexity'])
                raw_features['spectral_spread_mean'] = get_feature_value(['lowlevel.spectral_spread.mean'])
                raw_features['spectral_centroid_mean'] = get_feature_value(['lowlevel.spectral_centroid.mean'])
                raw_features['spectral_entropy_mean'] = get_feature_value(['lowlevel.spectral_entropy.mean'])
                raw_features['spectral_complexity_mean'] = get_feature_value(['lowlevel.spectral_complexity.mean'])
                raw_features['silence_rate_60dB_mean'] = get_feature_value(['lowlevel.silence_rate_60dB.mean'])
                raw_features['spectral_flux_mean'] = get_feature_value(['lowlevel.spectral_flux.mean'])
                raw_features['spectral_kurtosis_mean'] = get_feature_value(['lowlevel.spectral_kurtosis.mean'])
                
                # Tonal features
                raw_features['key'] = get_feature_value(['tonal.key_krumhansl.key'])
                raw_features['mode'] = get_feature_value(['tonal.key_krumhansl.scale'])

                raw_features['danceability'] = get_feature_value(['rhythm.danceability'])
                raw_features['energy'] = get_feature_value(['lowlevel.energy', 'lowlevel.average_loudness'])
                
                return raw_features

            # Extract high-level features from extractor
            high_level_features = self.extract_high_level_features_from_extractor(features_pool, audio_path)

            # Extract comprehensive raw features for production quality assessment
            raw_features = self.extract_comprehensive_raw_features(features_pool)

            # Return both high-level and raw features for comprehensive analysis
            # This maintains backward compatibility while adding production quality features
            result = {
                **high_level_features,  # All existing high-level features (acousticness, energy, etc.)
                'raw_features': raw_features  # NEW: Comprehensive raw features for production assessment
            }

            logger.info("Successfully extracted high-level and raw audio features")
            return result

        except Exception as e:
            logger.error(f"Error analyzing audio file {audio_path}: {type(e).__name__} - {str(e)}", exc_info=True)
            raise

    def extract_comprehensive_raw_features(self, features_pool) -> Dict[str, Any]:
        """
        Extract comprehensive raw features from Essentia features pool for production quality assessment.
        
        This method extracts all available production-quality features including:
        - Spectral analysis features for mix quality assessment
        - Rhythm and temporal features for production timing analysis  
        - Harmonic content features for music theory analysis
        - Dynamic range and loudness features for mastering assessment
        
        Args:
            features_pool: Essentia FeaturePool containing all extracted features
            
        Returns:
            Dictionary containing comprehensive raw features organized by category
        """
        available_keys = features_pool.descriptorNames()
        logger.debug(f"Extracting comprehensive raw features from {len(available_keys)} available descriptors")
        
        # Helper function for safe feature extraction
        def safe_extract(key: str, default: Any = None) -> Any:
            """Safely extract a feature value with proper type handling."""
            if key not in available_keys:
                return default
            try:
                value = features_pool[key]
                if isinstance(value, (np.ndarray, list)):
                    if len(value) == 0:
                        return default
                    elif len(value) == 1:
                        return float(value[0]) if isinstance(value[0], (int, float, np.number)) else value[0]
                    else:
                        # For arrays/lists with multiple values, return first value or stats
                        if isinstance(value[0], (int, float, np.number)):
                            return float(value[0])  # Take first value for now
                        else:
                            return value.tolist() if hasattr(value, 'tolist') else list(value)
                elif isinstance(value, (int, float, np.number)):
                    return float(value)
                elif isinstance(value, str):
                    return str(value)
                else:
                    return value
            except Exception as e:
                logger.warning(f"Error extracting feature '{key}': {e}")
                return default
        
        # Extract comprehensive raw features organized by category
        raw_features = {
            # SPECTRAL FEATURES - Core for production quality assessment
            'spectral': {
                'centroid': {
                    'mean': safe_extract('lowlevel.spectral_centroid.mean'),
                    'var': safe_extract('lowlevel.spectral_centroid.var'),
                    'min': safe_extract('lowlevel.spectral_centroid.min'),
                    'max': safe_extract('lowlevel.spectral_centroid.max')
                },
                'rolloff': {
                    'mean': safe_extract('lowlevel.spectral_rolloff.mean'),
                    'var': safe_extract('lowlevel.spectral_rolloff.var')
                },
                'spread': {
                    'mean': safe_extract('lowlevel.spectral_spread.mean'),
                    'var': safe_extract('lowlevel.spectral_spread.var')
                },
                'entropy': {
                    'mean': safe_extract('lowlevel.spectral_entropy.mean'),
                    'var': safe_extract('lowlevel.spectral_entropy.var')
                },
                'complexity': {
                    'mean': safe_extract('lowlevel.spectral_complexity.mean'),
                    'var': safe_extract('lowlevel.spectral_complexity.var')
                },
                'flux': {
                    'mean': safe_extract('lowlevel.spectral_flux.mean'),
                    'var': safe_extract('lowlevel.spectral_flux.var')
                },
                'kurtosis': {
                    'mean': safe_extract('lowlevel.spectral_kurtosis.mean'),
                    'var': safe_extract('lowlevel.spectral_kurtosis.var')
                },
                'skewness': {
                    'mean': safe_extract('lowlevel.spectral_skewness.mean'),
                    'var': safe_extract('lowlevel.spectral_skewness.var')
                },
                'decrease': {
                    'mean': safe_extract('lowlevel.spectral_decrease.mean'),
                    'var': safe_extract('lowlevel.spectral_decrease.var')
                },
                'energy': {
                    'mean': safe_extract('lowlevel.spectral_energy.mean'),
                    'var': safe_extract('lowlevel.spectral_energy.var')
                },
                'energyband': {
                    'low': {
                        'mean': safe_extract('lowlevel.spectral_energyband_low.mean'),
                        'var': safe_extract('lowlevel.spectral_energyband_low.var')
                    },
                    'middle_low': {
                        'mean': safe_extract('lowlevel.spectral_energyband_middle_low.mean'),
                        'var': safe_extract('lowlevel.spectral_energyband_middle_low.var')
                    },
                    'middle_high': {
                        'mean': safe_extract('lowlevel.spectral_energyband_middle_high.mean'),
                        'var': safe_extract('lowlevel.spectral_energyband_middle_high.var')
                    },
                    'high': {
                        'mean': safe_extract('lowlevel.spectral_energyband_high.mean'),
                        'var': safe_extract('lowlevel.spectral_energyband_high.var')
                    }
                }
            },
            
            # RHYTHM FEATURES - For production timing and groove analysis
            'rhythm': {
                'bpm': safe_extract('rhythm.bpm'),
                'bpm_histogram_first_peak_bpm': safe_extract('rhythm.bpm_histogram_first_peak_bpm'),
                'bpm_histogram_second_peak_bpm': safe_extract('rhythm.bpm_histogram_second_peak_bpm'),
                'danceability': safe_extract('rhythm.danceability'),
                'onset_rate': safe_extract('rhythm.onset_rate'),
                'beats_loudness': {
                    'mean': safe_extract('rhythm.beats_loudness.mean'),
                    'var': safe_extract('rhythm.beats_loudness.var')
                },
                'beats_position': safe_extract('rhythm.beats_position')
            },
            
            # TONAL FEATURES - For harmonic content analysis
            'tonal': {
                'key_krumhansl': {
                    'key': safe_extract('tonal.key_krumhansl.key'),
                    'scale': safe_extract('tonal.key_krumhansl.scale'),
                    'strength': safe_extract('tonal.key_krumhansl.strength')
                },
                'key_temperley': {
                    'key': safe_extract('tonal.key_temperley.key'),
                    'scale': safe_extract('tonal.key_temperley.scale'),
                    'strength': safe_extract('tonal.key_temperley.strength')
                },
                'chords_changes_rate': safe_extract('tonal.chords_changes_rate'),
                'chords_strength': {
                    'mean': safe_extract('tonal.chords_strength.mean'),
                    'var': safe_extract('tonal.chords_strength.var')
                },
                'hpcp': {
                    'mean': safe_extract('tonal.hpcp.mean'),
                    'var': safe_extract('tonal.hpcp.var')
                },
                'thpcp': {
                    'mean': safe_extract('tonal.thpcp.mean'),
                    'var': safe_extract('tonal.thpcp.var')
                }
            },
            
            # LOWLEVEL DYNAMIC FEATURES - For mastering and production assessment
            'dynamics': {
                'dynamic_complexity': safe_extract('lowlevel.dynamic_complexity'),
                'loudness_ebu128': {
                    'integrated': safe_extract('lowlevel.loudness_ebu128.integrated'),
                    'loudness_range': safe_extract('lowlevel.loudness_ebu128.loudness_range'),
                    'momentary': safe_extract('lowlevel.loudness_ebu128.momentary'),
                    'short_term': safe_extract('lowlevel.loudness_ebu128.short_term')
                },
                'silence_rate_20dB': {
                    'mean': safe_extract('lowlevel.silence_rate_20dB.mean'),
                    'var': safe_extract('lowlevel.silence_rate_20dB.var')
                },
                'silence_rate_30dB': {
                    'mean': safe_extract('lowlevel.silence_rate_30dB.mean'),
                    'var': safe_extract('lowlevel.silence_rate_30dB.var')
                },
                'silence_rate_60dB': {
                    'mean': safe_extract('lowlevel.silence_rate_60dB.mean'),
                    'var': safe_extract('lowlevel.silence_rate_60dB.var')
                }
            },
            
            # MFCC FEATURES - For timbre analysis
            'mfcc': {
                'mean': safe_extract('lowlevel.mfcc.mean'),
                'var': safe_extract('lowlevel.mfcc.var'),
                'cov': safe_extract('lowlevel.mfcc.cov')
            },
            
            # METADATA - File and processing information
            'metadata': {
                'length': safe_extract('metadata.audio_properties.length'),
                'sample_rate': safe_extract('metadata.audio_properties.sample_rate'),
                'bit_rate': safe_extract('metadata.audio_properties.bit_rate'),
                'codec': safe_extract('metadata.audio_properties.codec')
            }
        }
        
        logger.info(f"Extracted comprehensive raw features with {len([k for category in raw_features.values() for k in category.keys() if isinstance(category, dict)])} feature categories")
        return raw_features

    def get_analysis_status(self) -> Dict[str, str]:
        """Get the current status of the analyzer."""
        return {
            "status": "ready",
            "analyzer": "MusicExtractor",
            "essentia_version": essentia.__version__
        }

    def extract_high_level_features_from_extractor(self, features_pool, audio_path: Optional[str] = None) -> Dict[str, float]:
        """
        Extract high-level features using existing extractor classes from services.
        
        Args:
            features_pool: The features pool from the extractor
            audio_path: Optional path to the audio file (required for mood extraction)
            
        Returns:
            Dictionary containing high-level features normalized between 0 and 1
        """
        hlf = {}
        
        try:
            # Restructure the features_pool into the expected nested format
            structured_data = {
                'lowlevel': {},
                'rhythm': {},
                'tonal': {},
                'metadata': {'audio_properties': {}}
            }
            
            # Helper function to get feature value
            def get_feature_value(key: str, default: Any = 0.0) -> Any:
                try:
                    value = features_pool[key]
                    if isinstance(value, (np.ndarray, list)) and len(value) >= 1:
                        return float(value[0]) if isinstance(value[0], (int, float, np.number)) else value[0]
                    elif isinstance(value, (int, float, np.number)):
                        return float(value)
                    return value
                except (KeyError, IndexError):
                    return default
                except Exception as e:
                    logger.error(f"Error processing value for key '{key}': {e}")
                    return default

            # Structure lowlevel features
            structured_data['lowlevel']['spectral_centroid'] = {'mean': get_feature_value('lowlevel.spectral_centroid.mean')}
            structured_data['lowlevel']['spectral_entropy'] = {'mean': get_feature_value('lowlevel.spectral_entropy.mean')}
            structured_data['lowlevel']['spectral_spread'] = {'mean': get_feature_value('lowlevel.spectral_spread.mean')}
            structured_data['lowlevel']['spectral_flux'] = {'mean': get_feature_value('lowlevel.spectral_flux.mean')}
            structured_data['lowlevel']['spectral_kurtosis'] = {'mean': get_feature_value('lowlevel.spectral_kurtosis.mean')}
            structured_data['lowlevel']['spectral_complexity'] = {'mean': get_feature_value('lowlevel.spectral_complexity.mean')}
            structured_data['lowlevel']['spectral_rolloff'] = {'mean': get_feature_value('lowlevel.spectral_rolloff.mean')}  # MISSING FEATURE ADDED
            structured_data['lowlevel']['silence_rate_30dB'] = {'mean': get_feature_value('lowlevel.silence_rate_30dB.mean')}
            structured_data['lowlevel']['silence_rate_60dB'] = {'mean': get_feature_value('lowlevel.silence_rate_60dB.mean')}
            structured_data['lowlevel']['spectral_energy'] = {'mean': get_feature_value('lowlevel.spectral_energy.mean')}
            structured_data['lowlevel']['dynamic_complexity'] = get_feature_value('lowlevel.dynamic_complexity')  # MISSING FEATURE ADDED

            # --- Logging for True Peak raw data ---
            true_peak_descriptor_found = 'lowlevel.true_peak' in features_pool.descriptorNames()
            raw_true_peak_value = None
            if true_peak_descriptor_found:
                raw_true_peak_value = get_feature_value('lowlevel.true_peak')
                logger.info(f"Raw True Peak: Found 'lowlevel.true_peak' in Essentia output with value: {raw_true_peak_value}")
                structured_data['lowlevel']['true_peak_linear'] = raw_true_peak_value
            else:
                logger.info("Raw True Peak: Descriptor 'lowlevel.true_peak' NOT FOUND in Essentia output.")
            # --- End Logging for True Peak ---

            # Directly fetch momentary and short-term streams
            momentary_stream = []
            available_descriptor_names = features_pool.descriptorNames()
            if 'lowlevel.loudness_ebu128.momentary' in available_descriptor_names:
                try:
                    raw_momentary = features_pool['lowlevel.loudness_ebu128.momentary']
                    logger.info(f"Raw Momentary Loudness: Found 'lowlevel.loudness_ebu128.momentary'. Type: {type(raw_momentary)}, Value (first 5 if list/array else full): {str(raw_momentary[:5]) if isinstance(raw_momentary, (list, np.ndarray)) and len(raw_momentary) > 5 else str(raw_momentary)}")
                    if isinstance(raw_momentary, (list, np.ndarray)):
                        # Filter out -inf values and replace with None
                        momentary_stream = [None if x == float('-inf') or x == float('inf') or np.isinf(x) else x for x in raw_momentary]
                    else:
                        logger.warning(f"Feature 'lowlevel.loudness_ebu128.momentary' is type {type(raw_momentary)}, expected list/array. Defaulting to [].")
                except Exception as e:
                    logger.error(f"Error retrieving 'lowlevel.loudness_ebu128.momentary': {e}. Defaulting to [].")
            else:
                logger.info("Raw Momentary Loudness: Descriptor 'lowlevel.loudness_ebu128.momentary' NOT FOUND in Essentia output.")
            
            short_term_stream = []
            if 'lowlevel.loudness_ebu128.short_term' in available_descriptor_names:
                try:
                    raw_short_term = features_pool['lowlevel.loudness_ebu128.short_term']
                    logger.info(f"Raw Short-Term Loudness: Found 'lowlevel.loudness_ebu128.short_term'. Type: {type(raw_short_term)}, Value (first 5 if list/array else full): {str(raw_short_term[:5]) if isinstance(raw_short_term, (list, np.ndarray)) and len(raw_short_term) > 5 else str(raw_short_term)}")
                    if isinstance(raw_short_term, (list, np.ndarray)):
                        # Filter out -inf values and replace with None
                        short_term_stream = [None if x == float('-inf') or x == float('inf') or np.isinf(x) else x for x in raw_short_term]
                    else:
                        logger.warning(f"Feature 'lowlevel.loudness_ebu128.short_term' is type {type(raw_short_term)}, expected list/array. Defaulting to [].")
                except Exception as e:
                    logger.error(f"Error retrieving 'lowlevel.loudness_ebu128.short_term': {e}. Defaulting to [].")
            else:
                logger.info("Raw Short-Term Loudness: Descriptor 'lowlevel.loudness_ebu128.short_term' NOT FOUND in Essentia output.")

            structured_data['lowlevel']['loudness_ebu128'] = {
                'integrated': get_feature_value('lowlevel.loudness_ebu128.integrated'),
                'loudness_range': get_feature_value('lowlevel.loudness_ebu128.loudness_range', 0.0),
                'momentary': momentary_stream, # Use the directly fetched stream
                'short_term': short_term_stream  # Use the directly fetched stream
            }

            # Calculate max values from the streams, handling None values
            max_momentary_lufs = None
            if momentary_stream:
                valid_momentary = [x for x in momentary_stream if x is not None and not np.isinf(x)]
                if valid_momentary:
                    max_momentary_lufs = max(valid_momentary)
            
            max_short_term_lufs = None
            if short_term_stream:
                valid_short_term = [x for x in short_term_stream if x is not None and not np.isinf(x)]
                if valid_short_term:
                    max_short_term_lufs = max(valid_short_term)

            # Calculate true peak in dBTP
            true_peak_dbtp = None
            if raw_true_peak_value is not None and not np.isinf(raw_true_peak_value):
                try:
                    # Convert linear true peak to dBTP
                    if raw_true_peak_value > 0:
                        true_peak_dbtp = 20 * np.log10(raw_true_peak_value)
                    else:
                        true_peak_dbtp = None  # Cannot convert 0 or negative values
                except Exception as e:
                    logger.error(f"Error converting true peak to dBTP: {e}")
                    true_peak_dbtp = None

            # Create loudness features with safe values
            hlf['loudness_features'] = {
                'integrated_lufs': get_feature_value('lowlevel.loudness_ebu128.integrated'),
                'loudness_range_lu': get_feature_value('lowlevel.loudness_ebu128.loudness_range', 0.0),
                'max_momentary_lufs': max_momentary_lufs,  # Safe value, no -Infinity
                'max_short_term_lufs': max_short_term_lufs,  # Safe value, no -Infinity  
                'true_peak_dbtp': true_peak_dbtp  # Safe value, no -Infinity
            }

            # Structure tonal features (COMPLETE MAPPING)
            structured_data['tonal']['chords_changes_rate'] = get_feature_value('tonal.chords_changes_rate')
            structured_data['tonal']['chords_strength'] = {'mean': get_feature_value('tonal.chords_strength.mean')}
            structured_data['tonal']['key_krumhansl'] = {
                'key': get_feature_value('tonal.key_krumhansl.key'),
                'scale': get_feature_value('tonal.key_krumhansl.scale'),
                'strength': get_feature_value('tonal.key_krumhansl.strength')
            }

            # Structure rhythm features (COMPLETE MAPPING)  
            structured_data['rhythm']['bpm'] = get_feature_value('rhythm.bpm')
            structured_data['rhythm']['danceability'] = get_feature_value('rhythm.danceability')
            structured_data['rhythm']['onset_rate'] = get_feature_value('rhythm.onset_rate')
            structured_data['rhythm']['beats_loudness'] = {'mean': get_feature_value('rhythm.beats_loudness.mean')}
            
            # Structure metadata (COMPLETE MAPPING)
            structured_data['metadata']['audio_properties']['length'] = get_feature_value('metadata.audio_properties.length')
            structured_data['metadata']['audio_properties']['sample_rate'] = get_feature_value('metadata.audio_properties.sample_rate')

            # Now use the structured data with the extractors
            from ..extractors.timbre import (
                AcousticnessExtractor, InstrumentalnessExtractor,
                LivenessExtractor, SpeechinessExtractor,
                BrightnessExtractor, ComplexityExtractor,
                WarmthExtractor
            )
            
            hlf['acousticness'] = AcousticnessExtractor().extract(structured_data)
            hlf['instrumentalness'] = InstrumentalnessExtractor().extract(structured_data)
            hlf['liveness'] = LivenessExtractor().extract(structured_data)
            hlf['speechiness'] = SpeechinessExtractor().extract(structured_data)
            hlf['brightness'] = BrightnessExtractor().extract(structured_data)
            hlf['complexity'] = ComplexityExtractor().extract(structured_data)
            hlf['warmth'] = WarmthExtractor().extract(structured_data)

            from ..extractors.tonal import (
                ValenceExtractor, HarmonicStrengthExtractor,
                KeyExtractor, ModeExtractor
            )
            
            hlf['valence'] = ValenceExtractor().extract(structured_data)
            hlf['harmonic_strength'] = HarmonicStrengthExtractor().extract(structured_data)
            hlf['key'] = KeyExtractor().extract(structured_data)
            hlf['mode'] = ModeExtractor().extract(structured_data)

            from ..extractors.rhythm import TempoExtractor, DanceabilityExtractor
            hlf['tempo'] = TempoExtractor().extract(structured_data)
            hlf['danceability'] = DanceabilityExtractor().extract(structured_data)

            from ..extractors.dynamics import EnergyExtractor
            hlf['energy'] = EnergyExtractor().extract(structured_data)

            from ..extractors.audio import LoudnessExtractor, LoudnessFeaturesExtractor, DurationExtractor, TimeSignatureExtractor
            hlf['loudness'] = LoudnessExtractor().extract(structured_data)
            hlf['loudness_features'] = LoudnessFeaturesExtractor().extract(structured_data)
            hlf['duration_ms'] = DurationExtractor().extract(structured_data)
            hlf['time_signature'] = TimeSignatureExtractor().extract(structured_data)
            
            return hlf
            
        except Exception as e:
            logger.error(f"Error extracting high-level features from extractor: {e}", exc_info=True)
            # Don't hide errors with fallbacks - let the user know extraction failed
            raise RuntimeError(f"Audio feature extraction failed: {e}") from e

    def extract_genre(self, audio_path: str) -> Dict[str, Any]:
        """
        Extract genre information from an audio file using GenreExtractor.

        Args:
            audio_path: Path to the audio file

        Returns:
            Dictionary containing genre information including:
            - top_genres: List of tuples (genre_name, probability) for top 5 genres
            - genre_probabilities: Dictionary of all genres and their probabilities
            - primary_genre: The most likely genre

        Raises:
            FileNotFoundError: If the audio file doesn't exist
            Exception: If genre extraction fails
        """
        audio_file_path = Path(audio_path)
        if not audio_file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            logger.info(f"Extracting genre information from: {audio_path}")
            
            # Import GenreExtractor here to avoid circular imports
            from ..extractors.genre import GenreExtractor
            
            # Initialize the genre extractor
            genre_extractor = GenreExtractor()
            
            # Extract genre information
            genre_info = genre_extractor.predict_from_audio(str(audio_file_path))
            
            logger.info(f"Successfully extracted genre information: {genre_info['primary_genre']}")
            return genre_info
            
        except Exception as e:
            logger.error(f"Error extracting genre from {audio_path}: {type(e).__name__} - {str(e)}", exc_info=True)
            raise

    def has_essentia_capability(self) -> bool:
        """Check if any Essentia capability is available"""
        try:
            import essentia
            return True
        except ImportError:
            return False
