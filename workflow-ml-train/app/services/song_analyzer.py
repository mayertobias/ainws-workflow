"""
Song Analyzer for ML Training Pipeline

Handles intelligent extraction of features from songs based on:
- Dynamic service selection (audio, content, or both)
- Feature discovery and agreement
- Caching with force extract option
- Error handling and logging
"""

import pandas as pd
import logging
import asyncio
import httpx
import json
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import os

# Import feature translator as proper package
from hss_feature_translator import FeatureTranslator

logger = logging.getLogger(__name__)

class SongAnalyzer:
    """
    Intelligent song analyzer that extracts features from audio and content services
    based on dynamic strategy selection and feature agreements.
    """
    
    # CRITICAL: Consistent column naming throughout the pipeline
    CSV_COLUMNS = {
        'song_name': 'song_name',
        'popularity': 'original_popularity',  # Use original_popularity consistently
        'has_audio_file': 'has_audio_file',
        'audio_file_path': 'audio_file_path',
        'has_lyrics': 'has_lyrics_file',  # Fix: CSV uses has_lyrics_file
        'lyrics_file_path': 'lyrics_file_path'
    }
    
    def __init__(self, cache_dir: str = "./cache", base_data_dir: str = "/app"):
        """Initialize the song analyzer"""
        self.cache_dir = Path(cache_dir)
        self.base_data_dir = Path(base_data_dir)
        
        # Create cache directories
        self.raw_features_cache = self.cache_dir / "raw_features"
        self.training_matrices_cache = self.cache_dir / "training_matrices"
        self.raw_features_cache.mkdir(parents=True, exist_ok=True)
        self.training_matrices_cache.mkdir(parents=True, exist_ok=True)
        
        # Service URLs from environment variables
        self.service_urls = {
            'audio': os.getenv('WORKFLOW_AUDIO_URL', 'http://localhost:8001'),
            'content': os.getenv('WORKFLOW_CONTENT_URL', 'http://localhost:8002')
        }
        
        # Feature discovery cache
        self.feature_schemas = {}
        
        # Initialize feature translator
        try:
            self.feature_translator = FeatureTranslator()
            logger.info("✅ SongAnalyzer initialized with FeatureTranslator")
        except Exception as e:
            logger.error(f"❌ Failed to initialize FeatureTranslator in SongAnalyzer: {e}")
            raise ValueError(f"Cannot proceed with analysis - feature translator issues: {e}")
        
        logger.info("🎵 SongAnalyzer initialized")
        logger.info(f"🔗 Service URLs: {self.service_urls}")
    
    async def analyze_songs_from_csv(
        self,
        csv_path: str,
        services_to_use: List[str],
        agreement_id: Optional[str] = None,
        force_extract: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Analyze songs from CSV with intelligent service-based extraction using PARALLEL PROCESSING.
        
        Args:
            csv_path: Path to CSV with song metadata
            services_to_use: List of services to use ['audio'] or ['content'] or ['audio', 'content']
            agreement_id: Feature agreement ID for filtering
            force_extract: Force re-extraction even if cached
            
        Returns:
            Tuple of (raw_features_df, extraction_report)
        """
        try:
            logger.info(f"🎵 Starting PARALLEL song analysis from CSV: {csv_path}")
            logger.info(f"📋 Services to use: {services_to_use}")
            
            # Load and validate CSV
            df = self._load_and_validate_csv(csv_path)
            total_songs = len(df)
            
            # Get parallel processing configuration - reduced to prevent overload
            max_concurrent = int(os.getenv('MAX_CONCURRENT_SONGS', '3'))  # Reduced from 8 to 3
            batch_size = int(os.getenv('SONG_BATCH_SIZE', '1'))  # Reduced from 5 to 1
            enable_parallel = os.getenv('ENABLE_PARALLEL_PROCESSING', 'true').lower() == 'true'
            
            logger.info(f"⚡ Parallel processing config: max_concurrent={max_concurrent}, batch_size={batch_size}, enabled={enable_parallel}")
            
            # Initialize tracking
            extraction_report = {
                'total_songs': total_songs,
                'successful_extractions': 0,
                'failed_extractions': 0,
                'skipped_extractions': 0,
                'services_used': services_to_use,
                'extraction_errors': [],
                'start_time': datetime.now(),
                'agreement_id': agreement_id,
                'parallel_processing': {
                    'enabled': enable_parallel,
                    'max_concurrent': max_concurrent,
                    'batch_size': batch_size
                }
            }
            
            # Discover service features if not cached
            await self._discover_service_features(services_to_use)
            
            # Extract features using parallel processing
            if enable_parallel:
                all_features = await self._extract_features_parallel(
                    df, services_to_use, force_extract, max_concurrent, batch_size, extraction_report
                )
            else:
                all_features = await self._extract_features_sequential(
                    df, services_to_use, force_extract, extraction_report
                )
            
            # Create DataFrame from extracted features
            if all_features:
                # 🔍 DEBUG: Check what features we extracted
                logger.info(f"🔍 DEBUG: all_features length: {len(all_features)}")
                if all_features:
                    sample_features = all_features[0]
                    logger.info(f"🔍 DEBUG: Sample song features keys: {list(sample_features.keys())}")
                    audio_features = [k for k in sample_features.keys() if k.startswith('audio_')]
                    logger.info(f"🔍 DEBUG: Audio features in sample: {len(audio_features)} - {audio_features[:5]}")
                
                features_df = pd.DataFrame(all_features)
                logger.info(f"🔍 DEBUG: DataFrame created - shape: {features_df.shape}")
                logger.info(f"🔍 DEBUG: DataFrame columns sample: {list(features_df.columns)[:10]}")
                audio_cols = [col for col in features_df.columns if col.startswith('audio_')]
                logger.info(f"🔍 DEBUG: Audio columns in DataFrame: {len(audio_cols)} - {audio_cols[:5]}")
                
                extraction_report['end_time'] = datetime.now()
                extraction_report['total_duration'] = (extraction_report['end_time'] - extraction_report['start_time']).total_seconds()
                
                # Cache the raw features
                await self._cache_raw_features(features_df, services_to_use)
                
                logger.info(f"🎉 Feature extraction completed: {extraction_report['successful_extractions']}/{total_songs} songs")
                logger.info(f"⏱️ Total duration: {extraction_report['total_duration']:.2f} seconds")
                if enable_parallel:
                    logger.info(f"🚀 Parallel processing achieved {total_songs/extraction_report['total_duration']:.2f} songs/second")
                
                return features_df, extraction_report
            else:
                raise ValueError("No features extracted from any songs")
                
        except Exception as e:
            logger.error(f"❌ Song analysis failed: {e}")
            raise
    
    def _load_and_validate_csv(self, csv_path: str) -> pd.DataFrame:
        """Load and validate the CSV file structure"""
        try:
            df = pd.read_csv(csv_path)
            
            # Ensure consistent column naming BEFORE validation
            if self.CSV_COLUMNS['popularity'] not in df.columns and 'popularity_score' in df.columns:
                logger.warning("⚠️ Found 'popularity_score' column, renaming to 'original_popularity' for consistency")
                df = df.rename(columns={'popularity_score': self.CSV_COLUMNS['popularity']})
            
            # Validate required columns
            required_columns = [
                self.CSV_COLUMNS['song_name'],
                self.CSV_COLUMNS['popularity']
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Log CSV structure
            logger.info(f"📊 Loaded CSV with {len(df)} songs and columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load CSV {csv_path}: {e}")
            raise
    
    async def _discover_service_features(self, services_to_use: List[str]):
        """Discover available features from selected services"""
        try:
            logger.info(f"🔍 Discovering features from services: {services_to_use}")
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                for service in services_to_use:
                    if service not in self.service_urls:
                        logger.warning(f"⚠️ Unknown service: {service}")
                        continue
                    
                    url = f"{self.service_urls[service]}/features"
                    try:
                        response = await client.get(url)
                        if response.status_code == 200:
                            schema = response.json()
                            self.feature_schemas[service] = schema
                            logger.info(f"✅ Discovered {service} service: {schema.get('capabilities', {}).get('total_features', 'unknown')} features")
                        else:
                            logger.warning(f"⚠️ {service} service returned {response.status_code}")
                    except Exception as e:
                        logger.error(f"❌ Failed to discover {service} service features: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Feature discovery failed: {e}")
            raise
    
    async def _extract_features_parallel(
        self,
        df: pd.DataFrame,
        services_to_use: List[str],
        force_extract: bool,
        max_concurrent: int,
        batch_size: int,
        extraction_report: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract features using parallel processing with batching and rate limiting"""
        
        all_features = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Process songs in batches
        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]
            
            logger.info(f"🔄 Processing batch {batch_start//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}: songs {batch_start+1}-{batch_end}")
            
            # Create tasks for this batch
            tasks = []
            for idx, row in batch_df.iterrows():
                task = self._extract_song_features_with_semaphore(
                    row, services_to_use, force_extract, semaphore, extraction_report
                )
                tasks.append(task)
            
            # Execute batch in parallel
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    song_name = batch_df.iloc[i][self.CSV_COLUMNS['song_name']]
                    extraction_report['failed_extractions'] += 1
                    extraction_report['extraction_errors'].append({
                        'song_name': song_name,
                        'error': str(result)
                    })
                    logger.error(f"❌ Exception in parallel extraction for {song_name}: {result}")
                elif result:
                    all_features.append(result)
                    extraction_report['successful_extractions'] += 1
                else:
                    song_name = batch_df.iloc[i][self.CSV_COLUMNS['song_name']]
                    extraction_report['failed_extractions'] += 1
                    extraction_report['extraction_errors'].append({
                        'song_name': song_name,
                        'error': 'Feature extraction returned empty'
                    })
                    logger.warning(f"⚠️ No features extracted for: {song_name}")
            
            # Small delay between batches to prevent overwhelming services
            if batch_end < len(df):
                await asyncio.sleep(0.1)
        
        return all_features
    
    async def _extract_song_features_with_semaphore(
        self,
        song_row: pd.Series,
        services_to_use: List[str],
        force_extract: bool,
        semaphore: asyncio.Semaphore,
        extraction_report: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract features for a single song with semaphore for rate limiting"""
        
        async with semaphore:
            try:
                song_features = await self._extract_features_for_song(
                    song_row, services_to_use, force_extract
                )
                
                if song_features:
                    # Add metadata to features
                    song_features.update({
                        'song_name': song_row[self.CSV_COLUMNS['song_name']],
                        'original_popularity': song_row[self.CSV_COLUMNS['popularity']],
                        'audio_file_path': song_row.get(self.CSV_COLUMNS['audio_file_path'], ''),
                        'lyrics_file_path': song_row.get(self.CSV_COLUMNS['lyrics_file_path'], '')
                    })
                    
                    logger.debug(f"✅ Extracted features for: {song_row[self.CSV_COLUMNS['song_name']]}")
                    return song_features
                else:
                    logger.warning(f"⚠️ No features extracted for: {song_row[self.CSV_COLUMNS['song_name']]}")
                    return None
                    
            except Exception as e:
                logger.error(f"❌ Failed to extract features for {song_row[self.CSV_COLUMNS['song_name']]}: {e}")
                raise
    
    async def _extract_features_sequential(
        self,
        df: pd.DataFrame,
        services_to_use: List[str],
        force_extract: bool,
        extraction_report: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract features using sequential processing (fallback)"""
        
        logger.info("🐌 Using sequential processing (parallel disabled)")
        all_features = []
        total_songs = len(df)
        
        for idx, row in df.iterrows():
            try:
                song_features = await self._extract_features_for_song(
                    row, services_to_use, force_extract
                )
                
                if song_features:
                    # Add metadata to features
                    song_features.update({
                        'song_name': row[self.CSV_COLUMNS['song_name']],
                        'original_popularity': row[self.CSV_COLUMNS['popularity']],
                        'audio_file_path': row.get(self.CSV_COLUMNS['audio_file_path'], ''),
                        'lyrics_file_path': row.get(self.CSV_COLUMNS['lyrics_file_path'], '')
                    })
                    
                    all_features.append(song_features)
                    extraction_report['successful_extractions'] += 1
                    
                    logger.info(f"✅ Extracted features for: {row[self.CSV_COLUMNS['song_name']]} ({idx+1}/{total_songs})")
                else:
                    extraction_report['failed_extractions'] += 1
                    extraction_report['extraction_errors'].append({
                        'song_name': row[self.CSV_COLUMNS['song_name']],
                        'error': 'Feature extraction returned empty'
                    })
                    logger.warning(f"⚠️ No features extracted for: {row[self.CSV_COLUMNS['song_name']]}")
                    
            except Exception as e:
                extraction_report['failed_extractions'] += 1
                extraction_report['extraction_errors'].append({
                    'song_name': row[self.CSV_COLUMNS['song_name']],
                    'error': str(e)
                })
                logger.error(f"❌ Failed to extract features for {row[self.CSV_COLUMNS['song_name']]}: {e}")
        
        return all_features
    
    async def _extract_features_for_song(
        self,
        song_row: pd.Series,
        services_to_use: List[str],
        force_extract: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Extract features for a single song from specified services"""
        song_name = song_row[self.CSV_COLUMNS['song_name']]
        
        # Check cache first (unless force extract)
        if not force_extract:
            cached_features = await self._get_cached_features(song_name, services_to_use)
            if cached_features:
                logger.debug(f"📋 Using cached features for: {song_name}")
                return cached_features
        
        # Extract fresh features
        extracted_features = {}
        
        try:
            # Extract audio features if requested
            if 'audio' in services_to_use:
                audio_features = await self._call_audio_service(song_row)
                if audio_features:
                    extracted_features.update(audio_features)
            
            # Extract content features if requested
            if 'content' in services_to_use:
                content_features = await self._call_content_service(song_row)
                if content_features:
                    extracted_features.update(content_features)
            
            # Cache the extracted features
            if extracted_features:
                await self._cache_song_features(song_name, extracted_features, services_to_use)
            
            return extracted_features if extracted_features else None
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed for {song_name}: {e}")
            return None
    
    async def _call_audio_service(self, song_row: pd.Series) -> Optional[Dict[str, Any]]:
        """Call audio service to extract audio features"""
        song_name = song_row[self.CSV_COLUMNS['song_name']]
        
        # Check if song has audio file
        has_audio = song_row.get(self.CSV_COLUMNS['has_audio_file'], False)
        if not has_audio:
            logger.debug(f"⏭️ Skipping audio extraction for {song_name}: no audio file")
            return None
        
        audio_path = song_row.get(self.CSV_COLUMNS['audio_file_path'], '')
        if not audio_path:
            logger.warning(f"⚠️ No audio file path for {song_name}")
            return None
        
        # Resolve full path - handle both absolute and relative paths
        if audio_path.startswith('/'):
            # Already absolute path
            full_audio_path = Path(audio_path)
        else:
            # Relative path, prepend base_data_dir
            full_audio_path = self.base_data_dir / audio_path.strip('/')
        if not full_audio_path.exists():
            logger.warning(f"⚠️ Audio file not found: {full_audio_path}")
            return None
        
        try:
            # Call audio service PERSISTENT ANALYZER (has JSON cleaning!)
            # Increased timeout for TensorFlow model inference (genre classification)
            async with httpx.AsyncClient(timeout=120.0) as client:
                with open(full_audio_path, 'rb') as audio_file:
                    # Detect proper MIME type based on file extension
                    file_extension = full_audio_path.suffix.lower()
                    if file_extension == '.mp3':
                        mime_type = 'audio/mpeg'
                    elif file_extension == '.wav':
                        mime_type = 'audio/wav'
                    elif file_extension == '.flac':
                        mime_type = 'audio/flac'
                    elif file_extension in ['.m4a', '.aac']:
                        mime_type = 'audio/aac'
                    else:
                        mime_type = 'audio/mpeg'  # Default to MP3
                    
                    # Create proper multipart form with explicit content-type headers
                    files = {
                        'file': (
                            full_audio_path.name, 
                            audio_file, 
                            mime_type
                        )
                    }
                    
                    response = await client.post(
                        f"{self.service_urls['audio']}/analyze/persistent",
                        files=files
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    # Parse using discovered schema
                    return self._parse_audio_response(result)
                else:
                    logger.error(f"❌ Audio service error for {song_name}: {response.status_code}")
                    logger.error(f"❌ Response content: {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Audio service call failed for {song_name}: {e}")
            return None
    
    async def _call_content_service(self, song_row: pd.Series) -> Optional[Dict[str, Any]]:
        """Call content service to extract content features"""
        song_name = song_row[self.CSV_COLUMNS['song_name']]
        
        # Check if song has lyrics
        has_lyrics = song_row.get(self.CSV_COLUMNS['has_lyrics'], False)
        if not has_lyrics:
            logger.debug(f"⏭️ Skipping content extraction for {song_name}: no lyrics")
            return None
        
        lyrics_path = song_row.get(self.CSV_COLUMNS['lyrics_file_path'], '')
        if not lyrics_path:
            logger.warning(f"⚠️ No lyrics file path for {song_name}")
            return None
        
        # Resolve full path - handle both absolute and relative paths
        if lyrics_path.startswith('/'):
            # Already absolute path
            full_lyrics_path = Path(lyrics_path)
        else:
            # Relative path, prepend base_data_dir
            full_lyrics_path = self.base_data_dir / lyrics_path.strip('/')
        if not full_lyrics_path.exists():
            logger.warning(f"⚠️ Lyrics file not found: {full_lyrics_path}")
            return None
        
        try:
            # Read lyrics content
            with open(full_lyrics_path, 'r', encoding='utf-8') as f:
                lyrics_content = f.read()
            
            # Call content service - use correct endpoint and request format
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.service_urls['content']}/analyze/lyrics",
                    json={
                        'text': lyrics_content,
                        'filename': f"{song_name}.txt",
                        'title': song_name
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Parse using actual response structure
                    return self._parse_content_response(result)
                else:
                    logger.error(f"❌ Content service error for {song_name}: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Content service call failed for {song_name}: {e}")
            return None
    
    def _parse_audio_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse audio service response using schema-based FeatureTranslator"""
        
        # 🔍 DEBUG: Log raw audio service response
        logger.info(f"🔍 DEBUG: Raw audio service response keys: {list(response.keys()) if response else 'None'}")
        if response and 'results' in response:
            logger.info(f"🔍 DEBUG: Response.results keys: {list(response['results'].keys())}")
            if 'features' in response['results']:
                logger.info(f"🔍 DEBUG: Response.results.features keys: {list(response['results']['features'].keys())}")
                if 'analysis' in response['results']['features']:
                    analysis = response['results']['features']['analysis']
                    logger.info(f"🔍 DEBUG: Analysis keys: {list(analysis.keys())}")
                    if 'basic' in analysis:
                        logger.info(f"🔍 DEBUG: Basic features count: {len(analysis['basic'])}")
                        logger.info(f"🔍 DEBUG: Basic features sample: {list(analysis['basic'].keys())[:5]}")
        
        try:
            # Use FeatureTranslator for schema-based parsing
            translated_features = self.feature_translator.audio_producer_to_consumer(response)
            logger.info(f"🎵 ✅ Schema-based translation SUCCESS: {len(translated_features)} features from audio service")
            logger.info(f"🔍 DEBUG: Translated feature sample: {list(translated_features.keys())[:5]}")
            return translated_features
            
        except Exception as e:
            logger.error(f"❌ Schema-based audio translation failed: {e}")
            logger.error(f"❌ Response structure: {json.dumps(response, indent=2)[:500]}...")
            
            # Fallback to manual parsing only for debugging
            logger.warning("⚠️ Attempting manual fallback parsing for debugging...")
            features = {}
            
            # PERSISTENT ENDPOINT: response.results.features.analysis
            if ('results' in response and 
                'features' in response['results'] and 
                'analysis' in response['results']['features']):
                
                analysis = response['results']['features']['analysis']
                
                # Extract basic audio features
                if 'basic' in analysis:
                    basic_features = analysis['basic']
                    for feature_name, feature_value in basic_features.items():
                        if not isinstance(feature_value, dict) and feature_value is not None:
                            # Strip 'basic_' prefix if present to avoid double prefixing
                            clean_name = feature_name.replace('basic_', '') if feature_name.startswith('basic_') else feature_name
                            features[f"audio_{clean_name}"] = feature_value
                
                # Extract genre features 
                if 'genre' in analysis:
                    genre_data = analysis['genre']
                    if isinstance(genre_data, dict):
                        features['audio_primary_genre'] = genre_data.get('primary_genre', 'unknown')
                        features['audio_top_genre_1_prob'] = genre_data.get('top_genre_1_prob', 0.0)
                        features['audio_top_genre_2_prob'] = genre_data.get('top_genre_2_prob', 0.0)
                
                # Extract mood features
                if 'mood' in analysis:
                    mood_data = analysis['mood']
                    if isinstance(mood_data, dict):
                        for mood_name, mood_value in mood_data.items():
                            features[f"audio_mood_{mood_name}"] = 1 if mood_value in ['happy', 'sad', 'aggressive', 'relaxed', 'party', 'electronic', 'acoustic'] else 0
                
                logger.warning(f"⚠️ Manual fallback extracted {len(features)} features")
                return features
            
            else:
                logger.error(f"❌ Unknown audio response structure: {list(response.keys()) if response else 'None'}")
                logger.error(f"❌ Cannot parse audio features. Expected 'results.features.analysis.basic' structure")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Failed to parse audio response: {e}")
            return {}
    
    def _parse_content_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse content service response using schema-based FeatureTranslator"""
        try:
            # Use FeatureTranslator for schema-based parsing
            translated_features = self.feature_translator.content_producer_to_consumer(response)
            logger.info(f"📝 Schema-based translation: {len(translated_features)} content features")
            return translated_features
            
        except Exception as e:
            logger.error(f"❌ Schema-based content translation failed: {e}")
            logger.error(f"❌ Response structure: {json.dumps(response, indent=2)[:500]}...")
            
            # Fallback to manual parsing for debugging
            logger.warning("⚠️ Attempting manual content fallback parsing...")
            features = {}
            
            if 'results' in response:
                results = response['results']
                
                # Extract sentiment features
                if 'sentiment' in results:
                    sentiment = results['sentiment']
                    features['content_sentiment_polarity'] = sentiment.get('polarity', 0.0)
                    features['content_sentiment_subjectivity'] = sentiment.get('subjectivity', 0.0)
                    
                    # Extract emotional scores
                    if 'emotional_scores' in sentiment:
                        emotional_scores = sentiment['emotional_scores']
                        for emotion, score in emotional_scores.items():
                            features[f'content_emotion_{emotion}'] = score
                
                # Extract complexity features
                if 'complexity' in results:
                    complexity = results['complexity']
                    features['content_avg_sentence_length'] = complexity.get('avg_sentence_length', 0.0)
                    features['content_avg_word_length'] = complexity.get('avg_word_length', 0.0)
                    features['content_lexical_diversity'] = complexity.get('lexical_diversity', 0.0)
                
                # Extract theme features (use first few words/nouns/verbs for ML)
                if 'themes' in results:
                    themes = results['themes']
                    
                    # Convert top words to features (use top 3)
                    top_words = themes.get('top_words', [])
                    for i, word in enumerate(top_words[:3]):
                        features[f'content_top_word_{i+1}'] = word
                    
                    # Count features for themes
                    features['content_themes_word_count'] = len(themes.get('top_words', []))
                    features['content_themes_noun_count'] = len(themes.get('main_nouns', []))
                    features['content_themes_verb_count'] = len(themes.get('main_verbs', []))
                    features['content_themes_entity_count'] = len(themes.get('entities', []))
                
                # Extract readability
                if 'readability' in results:
                    features['content_readability'] = results['readability']
                
                # Extract statistics
                if 'statistics' in results:
                    stats = results['statistics']
                    features['content_word_count'] = stats.get('word_count', 0)
                    features['content_unique_words'] = stats.get('unique_words', 0)
                    features['content_vocabulary_density'] = stats.get('vocabulary_density', 0.0)
                    features['content_sentence_count'] = stats.get('sentence_count', 0)
                    features['content_avg_words_per_sentence'] = stats.get('avg_words_per_sentence', 0.0)
                
                # Extract narrative structure (FIXED: Map to feature translator schema)
                if 'narrative_structure' in results:
                    narrative = results['narrative_structure']
                    features['lyrics_verse_count'] = narrative.get('verse_count', 0)
                    features['lyrics_repetition_score'] = narrative.get('repetition_score', 0.0)
                    features['lyrics_avg_verse_length'] = narrative.get('avg_verse_length', 0.0)
                
                # NOTE: Removed content_* prefixed features that don't exist in feature translator
                # These are now handled by schema-based translation using lyrics_* prefix
            
            logger.debug(f"📝 Parsed {len(features)} content features from lyrics API")
            return features
            
        except Exception as e:
            logger.error(f"❌ Failed to parse content response: {e}")
            logger.error(f"Response structure: {list(response.keys()) if response else 'None'}")
            return {}
    
    async def _cache_song_features(self, song_name: str, features: Dict[str, Any], services_used: List[str]):
        """Cache extracted features for a song"""
        try:
            cache_key = self._generate_cache_key(song_name, services_used)
            cache_file = self.raw_features_cache / f"{cache_key}.json"
            
            cache_data = {
                'song_name': song_name,
                'services_used': services_used,
                'features': features,
                'extracted_at': datetime.now().isoformat(),
                'feature_count': len(features)
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
                
            logger.debug(f"💾 Cached features for {song_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to cache features for {song_name}: {e}")
    
    async def _get_cached_features(self, song_name: str, services_used: List[str]) -> Optional[Dict[str, Any]]:
        """Get cached features for a song"""
        try:
            cache_key = self._generate_cache_key(song_name, services_used)
            cache_file = self.raw_features_cache / f"{cache_key}.json"
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                return cache_data.get('features', {})
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get cached features for {song_name}: {e}")
            return None
    
    def _generate_cache_key(self, song_name: str, services_used: List[str]) -> str:
        """Generate a unique cache key for song + services combination"""
        services_str = "_".join(sorted(services_used))
        combined = f"{song_name}_{services_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def _cache_raw_features(self, features_df: pd.DataFrame, services_used: List[str]):
        """Cache the complete raw features DataFrame"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            services_str = "_".join(sorted(services_used))
            filename = f"raw_features_{services_str}_{timestamp}.csv"
            cache_file = self.raw_features_cache / filename
            
            features_df.to_csv(cache_file, index=False)
            logger.info(f"💾 Cached raw features DataFrame: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to cache raw features DataFrame: {e}")
    
    def create_training_matrix(
        self,
        raw_features_df: pd.DataFrame,
        selected_features: List[str],
        agreement_id: str
    ) -> pd.DataFrame:
        """Create filtered training matrix based on feature agreement"""
        try:
            logger.info(f"🔧 Creating training matrix with {len(selected_features)} selected features")
            
            # Always include metadata columns
            metadata_columns = ['song_name', 'original_popularity']
            
            # 🔍 DEBUG: Log the raw_features_df state
            logger.info(f"🔍 DEBUG: raw_features_df shape: {raw_features_df.shape}")
            logger.info(f"🔍 DEBUG: raw_features_df columns: {list(raw_features_df.columns)}")
            audio_cols_in_df = [col for col in raw_features_df.columns if col.startswith('audio_')]
            logger.info(f"🔍 DEBUG: Audio columns in raw_features_df: {len(audio_cols_in_df)} - {audio_cols_in_df[:5]}")
            
            # Filter to selected features + metadata
            available_features = [col for col in selected_features if col in raw_features_df.columns]
            missing_features = [col for col in selected_features if col not in raw_features_df.columns]
            
            logger.info(f"🔍 DEBUG: Selected features: {len(selected_features)} - {selected_features[:5]}")
            logger.info(f"🔍 DEBUG: Available features: {len(available_features)} - {available_features[:5]}")
            
            if missing_features:
                logger.warning(f"⚠️ Missing features: {missing_features}")
            
            columns_to_keep = metadata_columns + available_features
            training_matrix = raw_features_df[columns_to_keep].copy()
            
            # Cache the training matrix
            self._cache_training_matrix(training_matrix, agreement_id)
            
            logger.info(f"✅ Created training matrix: {len(training_matrix)} rows × {len(columns_to_keep)} columns")
            return training_matrix
            
        except Exception as e:
            logger.error(f"❌ Failed to create training matrix: {e}")
            raise
    
    def _cache_training_matrix(self, training_matrix: pd.DataFrame, agreement_id: str):
        """Cache the training matrix"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_matrix_{agreement_id}_{timestamp}.csv"
            cache_file = self.training_matrices_cache / filename
            
            training_matrix.to_csv(cache_file, index=False)
            logger.info(f"💾 Cached training matrix: {filename}")
            
            # Store the latest cache filename for easy retrieval
            self._latest_training_matrix_cache = str(cache_file)
            
        except Exception as e:
            logger.error(f"❌ Failed to cache training matrix: {e}")
    
    def load_cached_training_matrix(self, agreement_id: str) -> Optional[pd.DataFrame]:
        """Load the most recent cached training matrix for an agreement"""
        try:
            # Look for the most recent training matrix file for this agreement
            pattern = f"training_matrix_{agreement_id}_*.csv"
            matching_files = list(self.training_matrices_cache.glob(pattern))
            
            if not matching_files:
                logger.warning(f"⚠️ No cached training matrix found for agreement {agreement_id}")
                return None
            
            # Get the most recent file (sorted by filename which includes timestamp)
            latest_file = sorted(matching_files)[-1]
            
            logger.info(f"📂 Loading cached training matrix: {latest_file.name}")
            training_matrix = pd.read_csv(latest_file)
            
            logger.info(f"✅ Loaded cached training matrix: {training_matrix.shape}")
            return training_matrix
            
        except Exception as e:
            logger.error(f"❌ Failed to load cached training matrix: {e}")
            return None 