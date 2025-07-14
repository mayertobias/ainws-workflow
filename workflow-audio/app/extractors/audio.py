from .base import BaseFeatureExtractor
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AudioExtractor(BaseFeatureExtractor):
    """
    Main audio feature extractor that combines basic audio features.
    Provides core audio characteristics like loudness, duration, and time signature.
    """
    
    def __init__(self):
        """Initialize sub-extractors"""
        self.loudness_extractor = LoudnessExtractor()
        self.loudness_features_extractor = LoudnessFeaturesExtractor()
        self.duration_extractor = DurationExtractor()
        self.time_signature_extractor = TimeSignatureExtractor()
    
    def extract(self, data):
        """Extract comprehensive audio features from audio data."""
        features = {}
        
        try:
            # Extract basic loudness
            features['loudness'] = self.loudness_extractor.extract(data)
        except Exception as e:
            logger.warning(f"Loudness extraction failed: {e}")
            features['loudness'] = -60.0  # Default quiet value
        
        try:
            # Extract detailed loudness features
            loudness_features = self.loudness_features_extractor.extract(data)
            features.update(loudness_features)
        except Exception as e:
            logger.warning(f"Loudness features extraction failed: {e}")
        
        try:
            # Extract duration
            features['duration_ms'] = self.duration_extractor.extract(data)
        except Exception as e:
            logger.warning(f"Duration extraction failed: {e}")
            features['duration_ms'] = 0
        
        try:
            # Extract time signature
            features['time_signature'] = self.time_signature_extractor.extract(data)
        except Exception as e:
            logger.warning(f"Time signature extraction failed: {e}")
            features['time_signature'] = 4
        
        return features

class LoudnessExtractor(BaseFeatureExtractor):
    """
    Extracts loudness feature from audio data.
    Represents the overall loudness of the track in decibels (dB)
    using the EBU R128 integrated loudness.
    """
    
    def extract(self, data):
        """Extract the loudness of the track in dB."""
        # Get the integrated loudness from EBU R128 measurement (already in dB)
        loudness = data['lowlevel']['loudness_ebu128']['integrated']
        
        # Return the actual loudness value in dB (not normalized)
        # This matches the Spotify API which returns the actual dB value
        return loudness


class LoudnessFeaturesExtractor(BaseFeatureExtractor):
    """
    Extracts a comprehensive set of loudness features from pre-computed audio analysis results.
    Uses EBU R128 metrics (Integrated, Range, Max Momentary, Max Short-Term) 
    and True Peak (if available in the input data).
    """
    
    def extract(self, data):
        """
        Extracts loudness features from a dictionary of pre-computed Essentia analysis results.

        Args:
            data (dict): A dictionary containing pre-computed Essentia analysis results.
                         Expected structure for EBU R128 metrics (typically from Essentia's LoudnessEBU128):
                         data['lowlevel']['loudness_ebu128'] = {
                             'integrated': float,      // Integrated Loudness in LUFS
                             'loudness_range': float,  // Loudness Range (LRA) in LU
                             'momentary': np.array,    // Stream of momentary loudness values in LUFS
                             'short_term': np.array     // Stream of short-term loudness values in LUFS
                         }
                         Expected structure for True Peak (typically from Essentia's TruePeak, one of these under 'lowlevel' or a similar key):
                         data['lowlevel']['true_peak_dbtp']: float // True Peak already in dBTP
                         OR
                         data['lowlevel']['true_peak_linear']: float // True Peak as linear amplitude (will be converted to dBTP)

        Returns:
            dict: A dictionary containing:
                'integrated_lufs': Integrated loudness (LUFS). 
                                   Defaults to -np.inf if not available or track is silent.
                'loudness_range_lu': Loudness Range (LRA in LU). 
                                     Defaults to 0.0 if not available or track has very low/no dynamic range.
                'max_momentary_lufs': Maximum Momentary Loudness (LUFS) from the 400ms window. 
                                      Defaults to -np.inf if not available or track is silent.
                'max_short_term_lufs': Maximum Short-Term Loudness (LUFS) from the 3s window. 
                                       Defaults to -np.inf if not available or track is silent.
                'true_peak_dbtp': True Peak (dBTP). 
                                  Defaults to None if not found in input 'data'. 
                                  Can be -np.inf if linear peak input is zero.
        """
        features = {}
        
        # Gracefully access nested dictionaries for EBU R128 data
        # data.get('lowlevel', {}) ensures that if 'lowlevel' key doesn't exist, we get an empty dict
        # then .get('loudness_ebu128', {}) on that ensures if 'loudness_ebu128' doesn't exist, we get an empty dict
        lowlevel_data = data.get('lowlevel', {})
        ebu_data = lowlevel_data.get('loudness_ebu128', {})

        logger.info(f"LoudnessFeaturesExtractor received ebu_data: {ebu_data}")
        logger.info(f"LoudnessFeaturesExtractor received lowlevel_data for true peak: true_peak_dbtp={lowlevel_data.get('true_peak_dbtp')}, true_peak_linear={lowlevel_data.get('true_peak_linear')}")

        # 1. Integrated Loudness (LUFS)
        # EBU R128 Integrated Loudness can be -np.inf for very quiet/silent signals.
        integrated_lufs = ebu_data.get('integrated', None) 
        features['integrated_lufs'] = float(integrated_lufs) if integrated_lufs is not None and not np.isnan(integrated_lufs) else None
        logger.info(f"LoudnessFeaturesExtractor: integrated_lufs = {features['integrated_lufs']}")

        # 2. Loudness Range (LRA in LU)
        # LRA is typically positive; 0.0 can mean very compressed or not enough data.
        loudness_range_lu = ebu_data.get('loudness_range', 0.0)
        features['loudness_range_lu'] = float(loudness_range_lu) if not np.isnan(loudness_range_lu) else 0.0
        logger.info(f"LoudnessFeaturesExtractor: loudness_range_lu = {features['loudness_range_lu']}")

        # 3. Max Momentary Loudness (from 400ms windows)
        momentary_stream_raw = ebu_data.get('momentary') # This could be None, a list, or a numpy array
        logger.info(f"LoudnessFeaturesExtractor: momentary_stream_raw type: {type(momentary_stream_raw)}, value (first 5 if list/array else full): {str(momentary_stream_raw[:5]) if isinstance(momentary_stream_raw, (list, np.ndarray)) and len(momentary_stream_raw) > 5 else str(momentary_stream_raw)}")
        if momentary_stream_raw is not None:
            # Ensure it's a numpy array for consistent processing
            momentary_stream = np.asarray(momentary_stream_raw)
            if momentary_stream.size > 0:
                # Filter out potential NaNs or Infs from the stream before taking max
                valid_momentary_values = momentary_stream[np.isfinite(momentary_stream)]
                logger.info(f"LoudnessFeaturesExtractor: momentary_stream size: {momentary_stream.size}, valid_momentary_values size: {valid_momentary_values.size}, first 5 valid: {valid_momentary_values[:5] if valid_momentary_values.size > 0 else 'N/A'}")
                if valid_momentary_values.size > 0:
                    features['max_momentary_lufs'] = float(np.max(valid_momentary_values))
                else: # Stream was all non-finite values or became empty
                    features['max_momentary_lufs'] = None  # Use None instead of -inf 
            else: # Stream was empty
                 logger.info("LoudnessFeaturesExtractor: momentary_stream was empty.")
                 features['max_momentary_lufs'] = None  # Use None instead of -inf
        else: # 'momentary' key was not in ebu_data or its value was None
            logger.info("LoudnessFeaturesExtractor: momentary_stream_raw was None.")
            features['max_momentary_lufs'] = None  # Use None instead of -inf
        logger.info(f"LoudnessFeaturesExtractor: max_momentary_lufs = {features['max_momentary_lufs']}")

        # 4. Max Short-Term Loudness (from 3s windows)
        short_term_stream_raw = ebu_data.get('short_term')
        logger.info(f"LoudnessFeaturesExtractor: short_term_stream_raw type: {type(short_term_stream_raw)}, value (first 5 if list/array else full): {str(short_term_stream_raw[:5]) if isinstance(short_term_stream_raw, (list, np.ndarray)) and len(short_term_stream_raw) > 5 else str(short_term_stream_raw)}")
        if short_term_stream_raw is not None:
            short_term_stream = np.asarray(short_term_stream_raw)
            if short_term_stream.size > 0:
                valid_short_term_values = short_term_stream[np.isfinite(short_term_stream)]
                logger.info(f"LoudnessFeaturesExtractor: short_term_stream size: {short_term_stream.size}, valid_short_term_values size: {valid_short_term_values.size}, first 5 valid: {valid_short_term_values[:5] if valid_short_term_values.size > 0 else 'N/A'}")
                if valid_short_term_values.size > 0:
                    features['max_short_term_lufs'] = float(np.max(valid_short_term_values))
                else:
                    features['max_short_term_lufs'] = None  # Use None instead of -inf
            else:
                logger.info("LoudnessFeaturesExtractor: short_term_stream was empty.")
                features['max_short_term_lufs'] = None  # Use None instead of -inf
        else:
            logger.info("LoudnessFeaturesExtractor: short_term_stream_raw was None.")
            features['max_short_term_lufs'] = None  # Use None instead of -inf
        logger.info(f"LoudnessFeaturesExtractor: max_short_term_lufs = {features['max_short_term_lufs']}")

        # 5. True Peak (dBTP)
        # This value is NOT part of the LoudnessEBU128 algorithm's direct output bundle.
        # It must be calculated by a separate TruePeak algorithm (e.g., es.TruePeak)
        # and its result added to the 'lowlevel_data' dictionary by your processing pipeline.
        true_peak_dbtp = None # Default to None, indicating data not provided
        true_peak_source = "None"

        # Check if dBTP is already provided
        if 'true_peak_dbtp' in lowlevel_data:
            tp_val = lowlevel_data['true_peak_dbtp']
            true_peak_source = f"Found true_peak_dbtp in lowlevel_data: {tp_val}"
            # Ensure it's a valid float, not None or NaN before assigning
            if tp_val is not None and not (isinstance(tp_val, float) and np.isnan(tp_val)):
                true_peak_dbtp = float(tp_val)
        # Else, check if linear True Peak is provided (to be converted)
        elif 'true_peak_linear' in lowlevel_data:
            linear_peak = lowlevel_data['true_peak_linear']
            true_peak_source = f"Found true_peak_linear in lowlevel_data: {linear_peak}"
            if linear_peak is not None and not (isinstance(linear_peak, float) and np.isnan(linear_peak)):
                linear_peak = float(linear_peak) # Ensure it's a float
                if linear_peak > 1e-9: # Check if positive and non-negligible (approx -180 dBFS)
                    true_peak_dbtp = 20 * np.log10(linear_peak)
                elif linear_peak >= 0: # Handles linear_peak == 0 or very small positive values
                    true_peak_dbtp = None  # Use None instead of -inf 
                # If linear_peak is negative (should not happen for amplitude), it remains None
        
        features['true_peak_dbtp'] = true_peak_dbtp
        logger.info(f"LoudnessFeaturesExtractor: True Peak Source: {true_peak_source}, Resulting true_peak_dbtp = {features['true_peak_dbtp']}")

        return features

class DurationExtractor(BaseFeatureExtractor):
    """
    Extracts duration feature from audio data.
    Represents the duration of the track in milliseconds.
    """
    
    def extract(self, data):
        """Extract the duration of the track in milliseconds."""
        # Get the length from audio properties (in seconds)
        duration_sec = data['metadata']['audio_properties']['length']
        
        # Convert to milliseconds
        duration_ms = int(duration_sec * 1000)
        
        return duration_ms


class TimeSignatureExtractor(BaseFeatureExtractor):
    """
    Extracts time signature from audio data.
    Represents the time signature of the track (e.g., 3/4, 4/4).
    
    Note: Essentia doesn't directly provide time signature estimation,
    so this is an approximation based on beat patterns.
    """
    
    def extract(self, data):
        """
        Estimate the time signature of the track.
        As Essentia doesn't directly provide this, we'll use a heuristic approach.
        """
        # Default to 4/4 time signature as it's most common in popular music
        # More sophisticated algorithms would analyze beat patterns in detail
        # For more accuracy, you could implement a custom algorithm here
        
        # You could potentially use beats_loudness to analyze patterns
        # and detect 3/4 vs 4/4, but that would require more complex analysis
        
        # For now, we'll return 4 as the most common time signature
        return 4 