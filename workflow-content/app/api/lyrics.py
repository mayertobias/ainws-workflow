"""
Lyrics analysis API endpoints for workflow-content service
"""

from fastapi import APIRouter, HTTPException, Depends, Header
import time
import logging
from typing import Dict, Any, Optional
import uuid
from datetime import datetime

from ..models.lyrics import (
    LyricsAnalysisRequest, 
    SentimentRequest,
    LyricsAnalysisResponse,
    SentimentResponse,
    AnalysisResult
)
from ..models.responses import ErrorResponse
from ..services.lyrics_analyzer import LyricsAnalyzer
from ..services.database_service import db_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency for lyrics analyzer
def get_lyrics_analyzer() -> LyricsAnalyzer:
    """Dependency to get lyrics analyzer instance"""
    return LyricsAnalyzer()

@router.post("/lyrics", response_model=AnalysisResult)
async def analyze_lyrics(
    request: LyricsAnalysisRequest,
    analyzer: LyricsAnalyzer = Depends(get_lyrics_analyzer),
    user_session: Optional[str] = Header(None, alias="X-Session-ID", description="User session ID"),
    user_agent: Optional[str] = Header(None, alias="User-Agent", description="User agent string")
):
    """
    Analyze lyrics for comprehensive features and sentiment
    Results are automatically saved to the database for history tracking
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting lyrics analysis for text of length {len(request.text)}")
        
        # Perform analysis
        analysis = await analyzer.analyze(request.text)
        
        # Convert to response model
        results = LyricsAnalysisResponse(
            sentiment=SentimentResponse(**analysis["sentiment"]),
            complexity=analysis["complexity"],
            themes=analysis["themes"],
            readability=analysis["readability"],
            emotional_progression=analysis["emotional_progression"],
            narrative_structure=analysis["narrative_structure"],
            key_motifs=analysis["key_motifs"],
            theme_clusters=analysis["theme_clusters"],
            statistics=analysis["statistics"]
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Create the result object
        result = AnalysisResult(
            status="success",
            results=results,
            processing_time_ms=processing_time
        )
        
        # Save to database if user session is provided and get the actual analysis_id
        if user_session:
            try:
                db_analysis_id = await db_service.save_analysis_result(
                    session_id=user_session,
                    original_text=request.text,
                    analysis_results=analysis,
                    processing_time_ms=int(processing_time),
                    filename=request.filename,
                    title=request.title
                )
                logger.info(f"Analysis saved to database: {db_analysis_id}")
                
                # Add the actual database analysis_id to response for frontend reference
                result.analysis_id = db_analysis_id
                
            except Exception as e:
                logger.error(f"Failed to save analysis to database: {e}")
                # Continue without failing the request
        
        logger.info(f"Lyrics analysis completed in {processing_time:.2f}ms")
        
        return result
        
    except ValueError as e:
        logger.error(f"Validation error in lyrics analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in lyrics analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/sentiment", response_model=Dict[str, Any])
async def analyze_sentiment(
    request: SentimentRequest,
    analyzer: LyricsAnalyzer = Depends(get_lyrics_analyzer)
):
    """
    Analyze sentiment of text
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting sentiment analysis for text of length {len(request.text)}")
        
        # Perform sentiment analysis
        sentiment = await analyzer.analyze_sentiment(request.text)
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Sentiment analysis completed in {processing_time:.2f}ms")
        
        return {
            "status": "success",
            "results": sentiment,
            "processing_time_ms": processing_time
        }
        
    except ValueError as e:
        logger.error(f"Validation error in sentiment analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/features/hss")
async def get_hss_features(
    text: str,
    analyzer: LyricsAnalyzer = Depends(get_lyrics_analyzer),
    user_session: Optional[str] = Header(None, alias="X-Session-ID", description="User session ID")
):
    """
    Extract HSS (Hit Song Science) features from lyrics text.
    This endpoint provides features for machine learning models.
    Does not save to database - this is a feature extraction only endpoint.
    """
    try:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Perform full analysis to extract HSS features
        analysis = await analyzer.analyze(text.strip())
        
        # Extract relevant features for HSS
        hss_features = {
            # Sentiment features
            "sentiment_polarity": analysis["sentiment"]["polarity"],
            "sentiment_subjectivity": analysis["sentiment"]["subjectivity"],
            
            # Complexity features
            "avg_sentence_length": analysis["complexity"]["avg_sentence_length"],
            "avg_word_length": analysis["complexity"]["avg_word_length"],
            "lexical_diversity": analysis["complexity"]["lexical_diversity"],
            
            # Statistical features
            "word_count": analysis["statistics"]["word_count"],
            "unique_words": analysis["statistics"]["unique_words"],
            "vocabulary_density": analysis["statistics"]["vocabulary_density"],
            
            # Readability
            "readability": analysis["readability"],
            
            # Theme diversity (number of theme clusters)
            "theme_diversity": len(analysis["theme_clusters"]),
            
            # Motif repetition
            "motif_count": len(analysis["key_motifs"]),
            
            # Narrative structure complexity
            "narrative_complexity": 1 if analysis["narrative_structure"]["structure"] == "complex" else 0,
        }
        
        # Generate analysis ID for reference but don't save to database
        analysis_id = str(uuid.uuid4())
        
        return {
            "status": "success",
            "features": hss_features,
            "analysis_id": analysis_id  # Reference ID only, not saved
        }
        
    except Exception as e:
        logger.error(f"Error extracting HSS features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/lyrics-quality")
async def batch_lyrics_quality_check(
    request: Dict[str, Any],
    analyzer: LyricsAnalyzer = Depends(get_lyrics_analyzer),
    correlation_id: Optional[str] = Header(None, alias="Correlation-ID")
):
    """
    Batch quality check for lyrics datasets - comprehensive lyrics quality analysis
    
    This endpoint analyzes a directory of lyrics files and returns quality issues
    for data quality dashboards and validation systems.
    
    Expected request format:
    {
        "check_directory": "/path/to/lyrics/files",
        "language_detection": true,
        "structure_analysis": true,
        "quality_thresholds": {
            "MIN_WORD_COUNT": 10,
            "MAX_WORD_COUNT": 2000,
            "MIN_LINE_COUNT": 4,
            "SUPPORTED_LANGUAGES": ["en", "es", "fr"],
            "REQUIRED_ENCODING": "UTF-8"
        }
    }
    """
    try:
        import os
        import chardet
        from pathlib import Path
        from langdetect import detect, DetectorFactory
        
        # Ensure consistent language detection
        DetectorFactory.seed = 0
        
        check_directory = request.get("check_directory", "/Users/manojveluchuri/saas/workflow/lyrics")
        language_detection = request.get("language_detection", True)
        structure_analysis = request.get("structure_analysis", True)
        quality_thresholds = request.get("quality_thresholds", {
            "MIN_WORD_COUNT": 10,
            "MAX_WORD_COUNT": 2000,
            "MIN_LINE_COUNT": 4,
            "SUPPORTED_LANGUAGES": ["en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh"],
            "REQUIRED_ENCODING": "UTF-8"
        })
        
        logger.info(f"🔍 Starting batch lyrics quality check for directory: {check_directory}")
        
        # Find all lyrics files
        lyrics_extensions = ['.txt', '.lrc', '.srt']
        lyrics_files = []
        issues = []
        
        if os.path.exists(check_directory):
            for root, dirs, files in os.walk(check_directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in lyrics_extensions):
                        lyrics_files.append(os.path.join(root, file))
        
        logger.info(f"📁 Found {len(lyrics_files)} lyrics files to analyze")
        
        analyzed_files = []
        
        for file_path in lyrics_files[:100]:  # Limit to 100 files for performance
            try:
                file_info = {
                    "path": file_path,
                    "name": os.path.basename(file_path),
                    "format": Path(file_path).suffix.lower()
                }
                
                # Read file content and detect encoding
                try:
                    with open(file_path, 'rb') as f:
                        raw_data = f.read()
                    
                    # Detect encoding
                    encoding_info = chardet.detect(raw_data)
                    detected_encoding = encoding_info.get('encoding', 'unknown')
                    encoding_confidence = encoding_info.get('confidence', 0.0)
                    
                    # Try to decode with detected encoding
                    try:
                        if detected_encoding:
                            content = raw_data.decode(detected_encoding)
                        else:
                            content = raw_data.decode('utf-8', errors='replace')
                    except UnicodeDecodeError:
                        content = raw_data.decode('utf-8', errors='replace')
                        issues.append({
                            "file_path": file_path,
                            "file_name": file_info["name"],
                            "issue_type": "wrong_encoding",
                            "severity": "warning",
                            "details": f"File encoding issues detected. Expected UTF-8, detected {detected_encoding}",
                            "encoding": detected_encoding,
                            "encoding_confidence": encoding_confidence
                        })
                    
                    file_info.update({
                        "encoding": detected_encoding,
                        "encoding_confidence": encoding_confidence,
                        "content_length": len(content)
                    })
                    
                except Exception as read_error:
                    issues.append({
                        "file_path": file_path,
                        "file_name": file_info["name"],
                        "issue_type": "corrupt",
                        "severity": "critical",
                        "details": f"Cannot read lyrics file: {str(read_error)}"
                    })
                    file_info["quality_score"] = 0
                    analyzed_files.append(file_info)
                    continue
                
                # Check if file is empty
                if not content.strip():
                    issues.append({
                        "file_path": file_path,
                        "file_name": file_info["name"],
                        "issue_type": "empty",
                        "severity": "critical",
                        "details": "Lyrics file is empty or contains no readable text",
                        "word_count": 0,
                        "line_count": 0
                    })
                    file_info.update({
                        "word_count": 0,
                        "line_count": 0,
                        "quality_score": 0
                    })
                    analyzed_files.append(file_info)
                    continue
                
                # Basic text analysis
                lines = content.split('\n')
                words = content.split()
                word_count = len(words)
                line_count = len([line for line in lines if line.strip()])
                
                file_info.update({
                    "word_count": word_count,
                    "line_count": line_count,
                    "quality_score": 100  # Start with perfect score
                })
                
                # Word count checks
                if word_count < quality_thresholds.get("MIN_WORD_COUNT", 10):
                    issues.append({
                        "file_path": file_path,
                        "file_name": file_info["name"],
                        "issue_type": "too_short",
                        "severity": "warning",
                        "details": f"Word count {word_count} below minimum {quality_thresholds.get('MIN_WORD_COUNT', 10)}",
                        "word_count": word_count,
                        "line_count": line_count
                    })
                    file_info["quality_score"] -= 20
                
                if word_count > quality_thresholds.get("MAX_WORD_COUNT", 2000):
                    issues.append({
                        "file_path": file_path,
                        "file_name": file_info["name"],
                        "issue_type": "too_long",
                        "severity": "warning",
                        "details": f"Word count {word_count} above maximum {quality_thresholds.get('MAX_WORD_COUNT', 2000)}",
                        "word_count": word_count,
                        "line_count": line_count
                    })
                    file_info["quality_score"] -= 10
                
                # Line count check
                if line_count < quality_thresholds.get("MIN_LINE_COUNT", 4):
                    issues.append({
                        "file_path": file_path,
                        "file_name": file_info["name"],
                        "issue_type": "poor_structure",
                        "severity": "info",
                        "details": f"Line count {line_count} below minimum {quality_thresholds.get('MIN_LINE_COUNT', 4)}",
                        "word_count": word_count,
                        "line_count": line_count
                    })
                    file_info["quality_score"] -= 5
                
                # Language detection
                if language_detection and word_count > 5:
                    try:
                        detected_language = detect(content)
                        file_info["detected_language"] = detected_language
                        
                        supported_languages = quality_thresholds.get("SUPPORTED_LANGUAGES", ["en"])
                        if detected_language not in supported_languages:
                            issues.append({
                                "file_path": file_path,
                                "file_name": file_info["name"],
                                "issue_type": "wrong_language",
                                "severity": "info",
                                "details": f"Detected language '{detected_language}' not in supported list {supported_languages}",
                                "detected_language": detected_language,
                                "word_count": word_count,
                                "line_count": line_count
                            })
                            file_info["quality_score"] -= 5
                        
                    except Exception as lang_error:
                        file_info["detected_language"] = "unknown"
                        issues.append({
                            "file_path": file_path,
                            "file_name": file_info["name"],
                            "issue_type": "language_detection_failed",
                            "severity": "info",
                            "details": f"Language detection failed: {str(lang_error)}",
                            "word_count": word_count,
                            "line_count": line_count
                        })
                
                # Structure analysis
                if structure_analysis:
                    try:
                        # Check for common lyrics patterns
                        content_lower = content.lower()
                        
                        # Check for verse/chorus markers
                        verse_markers = ['verse', 'chorus', 'bridge', 'refrain', '[verse', '[chorus', '[bridge']
                        has_structure_markers = any(marker in content_lower for marker in verse_markers)
                        
                        # Check for repetitive patterns (common in songs)
                        unique_lines = set(line.strip().lower() for line in lines if line.strip())
                        repetition_ratio = (line_count - len(unique_lines)) / line_count if line_count > 0 else 0
                        
                        file_info.update({
                            "has_structure_markers": has_structure_markers,
                            "repetition_ratio": repetition_ratio,
                            "unique_lines": len(unique_lines)
                        })
                        
                        # Flag if no repetition (unusual for song lyrics)
                        if repetition_ratio < 0.1 and line_count > 10:
                            issues.append({
                                "file_path": file_path,
                                "file_name": file_info["name"],
                                "issue_type": "unusual_structure",
                                "severity": "info",
                                "details": f"Very low repetition ratio ({repetition_ratio:.2f}) - unusual for song lyrics",
                                "repetition_ratio": repetition_ratio,
                                "word_count": word_count,
                                "line_count": line_count
                            })
                        
                    except Exception as structure_error:
                        logger.warning(f"Structure analysis failed for {file_path}: {structure_error}")
                
                # Check if lyrics have audio file pairing
                audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.aac']
                base_name = Path(file_path).stem
                audio_file_paired = None
                
                # Look for matching audio file in parent or sibling directories
                parent_dir = Path(file_path).parent
                possible_audio_paths = []
                
                # Check same directory
                for ext in audio_extensions:
                    audio_path = parent_dir / f"{base_name}{ext}"
                    if audio_path.exists():
                        audio_file_paired = str(audio_path)
                        break
                
                # Check sibling songs directory if not found
                if not audio_file_paired:
                    songs_dir = parent_dir.parent / 'songs'
                    if songs_dir.exists():
                        for ext in audio_extensions:
                            audio_path = songs_dir / f"{base_name}{ext}"
                            if audio_path.exists():
                                audio_file_paired = str(audio_path)
                                break
                
                file_info["audio_file_paired"] = audio_file_paired
                
                # Ensure quality score is non-negative
                file_info["quality_score"] = max(0, file_info.get("quality_score", 100))
                analyzed_files.append(file_info)
                
            except Exception as file_error:
                logger.warning(f"⚠️ Failed to analyze {file_path}: {file_error}")
                issues.append({
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "issue_type": "analysis_error",
                    "severity": "critical",
                    "details": f"Analysis failed: {str(file_error)}"
                })
        
        logger.info(f"✅ Batch lyrics quality check completed. Found {len(issues)} issues across {len(analyzed_files)} files")
        
        return {
            "status": "success",
            "check_directory": check_directory,
            "total_files_found": len(lyrics_files),
            "files_analyzed": len(analyzed_files),
            "total_issues": len(issues),
            "critical_issues": len([i for i in issues if i["severity"] == "critical"]),
            "warning_issues": len([i for i in issues if i["severity"] == "warning"]),
            "info_issues": len([i for i in issues if i["severity"] == "info"]),
            "lyrics_files": analyzed_files,
            "issues": issues,
            "quality_thresholds_used": quality_thresholds,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "correlation_id": correlation_id
        }
        
    except Exception as e:
        logger.error(f"❌ Batch lyrics quality check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch lyrics quality check failed: {str(e)}") 