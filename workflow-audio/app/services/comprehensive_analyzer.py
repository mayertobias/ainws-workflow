"""
Comprehensive Audio Analyzer

This module orchestrates all available audio extractors to provide
comprehensive analysis of audio files.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import time

from .audio_analyzer import AudioAnalyzer
from ..extractors.audio import AudioExtractor
from ..extractors.rhythm import RhythmExtractor
from ..extractors.tonal import TonalExtractor
from ..extractors.timbre import TimbreExtractor
from ..extractors.dynamics import DynamicsExtractor
from ..extractors.mood import MoodExtractor
from ..extractors.genre import GenreExtractor

logger = logging.getLogger(__name__)

class ComprehensiveAudioAnalyzer:
    """
    Comprehensive audio analyzer that combines AudioAnalyzer with specialized extractors
    
    Architecture:
    - AudioAnalyzer: Handles all basic audio features (energy, valence, tempo, etc.)
    - Genre & Mood extractors: Add specialized classification capabilities
    """
    
    def __init__(self):
        """Initialize analyzer and specialized extractors"""
        self.audio_analyzer = AudioAnalyzer()
        
        # Only initialize extractors that add value beyond basic analysis
        self.specialized_extractors = {
            'genre': GenreExtractor(),
            'mood': MoodExtractor()
        }
        
        logger.info(f"Initialized comprehensive analyzer with basic analysis + {len(self.specialized_extractors)} specialized extractors")
    
    def analyze(self, audio_path: str, extractors: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive audio analysis
        
        Args:
            audio_path: Path to the audio file
            extractors: List of specific extractors to use (if None, uses all)
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        start_time = time.time()
        
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Starting comprehensive analysis of: {audio_path}")
        
        results = {
            'file_info': {
                'path': str(audio_path),
                'filename': Path(audio_path).name,
                'analysis_timestamp': time.time()
            },
            'analysis': {}
        }
        
        # Get basic audio analysis first (provides foundation for other extractors)
        try:
            logger.info("Running basic audio analysis...")
            basic_analysis = self.audio_analyzer.analyze_audio(audio_path)
            results['analysis']['basic'] = basic_analysis
        except Exception as e:
            logger.error(f"Basic audio analysis failed: {e}")
            results['analysis']['basic'] = {'error': str(e)}
        
        # Run specialized extractors (only genre and mood - basic features already extracted)
        extractors_to_run = extractors if extractors else list(self.specialized_extractors.keys())
        
        logger.info(f"Running specialized extractors: {extractors_to_run}")
        
        # Run each specialized extractor
        for extractor_name in extractors_to_run:
            if extractor_name in self.specialized_extractors:
                try:
                    logger.info(f"Running {extractor_name} extractor...")
                    
                    # Different extractors expect different input types
                    if extractor_name == 'mood':
                        # Mood extractor expects dictionary with 'audio_file' key
                        extractor_input = {'audio_file': audio_path}
                    elif extractor_name == 'genre':
                        # Genre extractor can handle direct file path
                        extractor_input = audio_path
                    else:
                        # Fallback to audio path
                        extractor_input = audio_path
                    
                    extractor_results = self.specialized_extractors[extractor_name].extract(extractor_input)
                    results['analysis'][extractor_name] = extractor_results
                    logger.info(f"✅ {extractor_name} extractor completed successfully")
                except Exception as e:
                    logger.error(f"❌ {extractor_name} extractor failed: {e}")
                    results['analysis'][extractor_name] = {'error': str(e)}
            else:
                logger.warning(f"Unknown specialized extractor: {extractor_name}")
        
        # Add processing metadata
        processing_time = time.time() - start_time
        results['metadata'] = {
            'processing_time_seconds': processing_time,
            'extractors_used': extractors_to_run,
            'extractors_successful': [
                name for name, result in results['analysis'].items() 
                if 'error' not in result
            ],
            'extractors_failed': [
                name for name, result in results['analysis'].items() 
                if 'error' in result
            ]
        }
        
        logger.info(f"Comprehensive analysis completed in {processing_time:.2f} seconds")
        return results
    
    def get_available_extractors(self) -> List[str]:
        """Get list of available specialized extractors"""
        return ['basic'] + list(self.specialized_extractors.keys())
    
    def get_extractor_info(self, extractor_name: str) -> Dict[str, Any]:
        """Get information about a specific extractor"""
        if extractor_name == 'basic':
            return {
                'name': 'basic',
                'description': 'Comprehensive basic audio features (energy, valence, tempo, etc.)',
                'features': ['energy', 'valence', 'danceability', 'tempo', 'acousticness', 'instrumentalness', 'liveness', 'speechiness', 'loudness', 'key', 'mode', 'time_signature']
            }
        elif extractor_name in self.specialized_extractors:
            extractor = self.specialized_extractors[extractor_name]
            return {
                'name': extractor_name,
                'description': getattr(extractor, '__doc__', 'No description available'),
                'features': getattr(extractor, 'get_feature_names', lambda: [])()
            }
        else:
            raise ValueError(f"Unknown extractor: {extractor_name}")
    
    def validate_audio_file(self, audio_path: str) -> bool:
        """Validate if the audio file can be processed"""
        try:
            path = Path(audio_path)
            if not path.exists():
                return False
            
            # Check file extension
            supported_extensions = ['.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg']
            if path.suffix.lower() not in supported_extensions:
                return False
            
            # Check file size (max 100MB)
            max_size = 100 * 1024 * 1024
            if path.stat().st_size > max_size:
                return False
            
            return True
            
        except Exception:
            return False 