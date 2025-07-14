"""
workflow-content: Content Analysis Microservice

This service handles lyrics analysis, text processing, and sentiment analysis
for the workflow system.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Dict, Any, Optional
import asyncio
import uvloop
from contextlib import asynccontextmanager

from .api.lyrics import router as lyrics_router
from .api.history import router as history_router
from .api.health import router as health_router
from .config.settings import settings
from .services.database_service import db_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set event loop policy for better performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events"""
    # Startup
    logger.info("Starting Workflow Content Service with Database Persistence")
    
    # Initialize database service
    try:
        await db_service.initialize()
        logger.info("✅ Database service initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Workflow Content Service")

# Create FastAPI application
app = FastAPI(
    title="Workflow Content Service",
    description="Content analysis microservice for lyrics and text processing with persistent history",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, prefix="/health", tags=["health"])
app.include_router(lyrics_router, prefix="/analyze", tags=["analysis"])
app.include_router(history_router, prefix="/history", tags=["history"])

# =============================================================================
# SERVICE DISCOVERY ENDPOINT
# =============================================================================

@app.get("/features")
async def list_features():
    """
    Returns the actual feature names extracted by the content analysis service
    
    This endpoint provides a flat list of feature names that ML training services
    can use for feature discovery and model training.
    """
    return {
        "service": "content",
        "version": "2.0.0",
        "features": [
            # Sentiment features
            "sentiment_polarity",
            "sentiment_subjectivity",
            "emotional_score_love",
            "emotional_score_heart",
            
            # Complexity features
            "avg_sentence_length",
            "avg_word_length", 
            "lexical_diversity",
            
            # Readability score
            "readability",
            
            # Narrative structure features
            "structure_type",
            "verse_count",
            "repetition_score",
            "avg_verse_length",
            
            # Statistical features
            "word_count",
            "unique_words",
            "vocabulary_density",
            "sentence_count",
            "avg_words_per_sentence",
            
            # Theme analysis features
            "top_words_count",
            "main_nouns_count",
            "main_verbs_count",
            "entities_count",
            "key_motifs_count",
            "theme_clusters_count",
            
            # Emotional progression features
            "emotional_density",
            "emotional_progression_variance",
            
            # Advanced features (if available)
            "flesch_reading_ease",
            "flesch_kincaid_grade",
            "smog_index",
            "coleman_liau_index",
            "automated_readability_index",
            "dale_chall_readability_score",
            "difficult_words",
            "linsear_write_formula",
            "gunning_fog",
            "avg_letter_per_word",
            
            # Additional sentiment features
            "sentiment_positive",
            "sentiment_negative", 
            "sentiment_neutral",
            "sentiment_compound",
            
            # Emotion scores
            "emotion_joy",
            "emotion_sadness",
            "emotion_anger",
            "emotion_fear",
            "emotion_surprise",
            "emotion_disgust",
            
            # Stylistic elements
            "type_token_ratio",
            "punctuation_ratio",
            
            # Punctuation frequency
            "punctuation_exclamation",
            "punctuation_question", 
            "punctuation_period",
            "punctuation_comma",
            "punctuation_semicolon",
            "punctuation_colon"
        ],
        "feature_count": 52,
        "analyzers": ["sentiment", "complexity", "themes", "readability", "structure", "statistics"],
        "description": "Content features extracted using spaCy + TextBlob with comprehensive NLP analysis"
    }

@app.get("/features/legacy")
async def list_features_legacy():
    """
    [LEGACY] Original features documentation endpoint
    
    This is kept for backward compatibility. Use /features for the new format.
    """
    return {
        "service": "content", 
        "version": "1.0.0",
        "endpoint": "/analyze/lyrics",
        "description": "Content analysis service providing lyrics and text analysis features",
        "capabilities": {
            "analyzers": ["sentiment", "structure", "themes", "language"],
            "total_features": 12,
            "analysis_time_avg": "1-2 seconds per song"
        },
        "features": {
            "sentiment": {
                "polarity": {"type": "float", "range": "-1-1", "description": "Negative (-1) to positive (+1) sentiment"},
                "subjectivity": {"type": "float", "range": "0-1", "description": "Objective (0) to subjective (1)"},
                "compound": {"type": "float", "range": "-1-1", "description": "Overall sentiment score"},
                "positive": {"type": "float", "range": "0-1", "description": "Positive sentiment probability"},
                "negative": {"type": "float", "range": "0-1", "description": "Negative sentiment probability"},
                "neutral": {"type": "float", "range": "0-1", "description": "Neutral sentiment probability"}
            },
            "structure": {
                "word_count": {"type": "integer", "description": "Total number of words"},
                "verse_count": {"type": "integer", "description": "Number of verses/sections"},
                "unique_words": {"type": "integer", "description": "Number of unique words"},
                "avg_word_length": {"type": "float", "description": "Average word length in characters"},
                "readability_score": {"type": "float", "range": "0-100", "description": "Reading ease score (0=difficult, 100=easy)"},
                "complexity_score": {"type": "float", "range": "0-1", "description": "Language complexity measure"}
            }
        },
        "response_structure": {
            "description": "Structure returned by /analyze/lyrics endpoint",
            "path_to_features": "results",
            "structure": {
                "results": {
                    "sentiment": {
                        "polarity": "float",
                        "subjectivity": "float",
                        "compound": "float",
                        "positive": "float",
                        "negative": "float",
                        "neutral": "float"
                    },
                    "structure": {
                        "word_count": "int",
                        "verse_count": "int", 
                        "unique_words": "int",
                        "avg_word_length": "float",
                        "readability_score": "float",
                        "complexity_score": "float"
                    },
                    "themes": {
                        "love": "float",
                        "party": "float",
                        "sadness": "float",
                        "empowerment": "float"
                    },
                    "language": {
                        "language": "string",
                        "confidence": "float"
                    }
                }
            }
        },
        "usage": {
            "example_call": "POST /analyze/lyrics with lyrics text or file",
            "parsing_example": "sentiment = response['results']['sentiment']",
            "key_features_for_ml": ["polarity", "subjectivity", "word_count", "readability_score", "complexity_score"]
        }
    }

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "workflow-content",
        "version": "1.0.0",
        "status": "running",
        "description": "Content analysis microservice with persistent history",
        "features": [
            "lyrics_analysis",
            "sentiment_analysis",
            "hss_features",
            "analysis_history",
            "user_sessions"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    ) 