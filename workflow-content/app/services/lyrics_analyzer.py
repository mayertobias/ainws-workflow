"""
Lyrics Analysis Service for workflow-content microservice

Extracted and adapted from the monolithic workflow-server application.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import spacy
from collections import defaultdict, Counter
import logging
import json
from textblob import TextBlob

from ..config.settings import settings

logger = logging.getLogger(__name__)

class LyricsAnalyzer:
    def __init__(self):
        """Initialize the LyricsAnalyzer for microservice environment."""
        # Ensure directories exist
        self.lyrics_dir = Path(settings.LYRICS_DIR)
        self.output_dir = Path(settings.OUTPUT_DIR)
        
        self.lyrics_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize NLP models
        try:
            self.nlp = spacy.load(settings.SPACY_MODEL)
        except OSError:
            logger.warning(f"spaCy model {settings.SPACY_MODEL} not found. Attempting to download...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", settings.SPACY_MODEL])
            self.nlp = spacy.load(settings.SPACY_MODEL)
        
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self._initialize_service()
    
    def _initialize_service(self) -> None:
        """Initialize sentiment lexicon and theme keywords."""
        try:
            self.sentiment_lexicon = self._load_sentiment_lexicon()
            self.theme_keywords = self._load_theme_keywords()
            logger.info("Successfully initialized lyrics analyzer")
        except Exception as e:
            logger.error(f"Failed to initialize lyrics analyzer: {e}")
            raise
    
    def _load_sentiment_lexicon(self) -> Dict[str, float]:
        """Load sentiment lexicon for emotional analysis."""
        return {
            "happy": 1.0, "joy": 0.9, "love": 0.9, "excited": 0.8, "amazing": 0.8,
            "wonderful": 0.8, "great": 0.7, "good": 0.6, "nice": 0.5, "beautiful": 0.8,
            "sad": -1.0, "angry": -0.8, "hate": -0.9, "terrible": -0.8, "awful": -0.8,
            "bad": -0.6, "horrible": -0.8, "disgusting": -0.7, "ugly": -0.6, "pain": -0.7,
            "hurt": -0.6, "cry": -0.5, "tears": -0.4, "lonely": -0.6, "empty": -0.5,
            "heart": 0.3, "soul": 0.2, "dream": 0.4, "hope": 0.6, "light": 0.3,
            "dark": -0.3, "shadow": -0.2, "fear": -0.7, "scared": -0.6, "worried": -0.4
        }
    
    def _load_theme_keywords(self) -> Dict[str, List[str]]:
        """Load theme keywords for clustering."""
        return {
            "love": ["heart", "love", "kiss", "romance", "darling", "baby", "honey", "soul"],
            "nature": ["sun", "moon", "stars", "ocean", "sky", "earth", "wind", "fire"],
            "society": ["world", "people", "life", "time", "society", "human", "together"],
            "emotions": ["feel", "emotion", "happy", "sad", "angry", "joy", "pain", "hurt"],
            "relationships": ["friend", "family", "together", "alone", "trust", "betray"],
            "spirituality": ["god", "heaven", "soul", "spirit", "prayer", "faith", "believe"],
            "party": ["dance", "party", "music", "fun", "celebration", "drink", "night"],
            "struggle": ["fight", "battle", "war", "struggle", "overcome", "survive", "strong"]
        }
    
    async def analyze(self, lyrics_text: str) -> Dict[str, Any]:
        """
        Analyze the given lyrics text and return comprehensive metrics.
        
        Args:
            lyrics_text (str): The lyrics text to analyze
            
        Returns:
            Dict[str, Any]: Dictionary containing various lyrics analysis metrics
            
        Raises:
            ValueError: If lyrics is empty or invalid
        """
        if not lyrics_text or not isinstance(lyrics_text, str):
            raise ValueError("Invalid lyrics input")
            
        if len(lyrics_text) > settings.MAX_TEXT_LENGTH:
            raise ValueError(f"Text too long. Maximum {settings.MAX_TEXT_LENGTH} characters allowed")
            
        try:
            # Basic text preprocessing
            lyrics_text = lyrics_text.lower()
            doc = self.nlp(lyrics_text)
            
            # Split into lines and verses
            lines = [line.strip() for line in lyrics_text.split('\n') if line.strip()]
            verses = self._split_into_verses(lines)
            
            # Calculate basic metrics
            words = [token.text for token in doc if token.is_alpha]
            sentences = list(doc.sents)
            words_no_stop = [token.text for token in doc if token.is_alpha and not token.is_stop]
            
            # Generate comprehensive analysis
            analysis = {
                "sentiment": self._analyze_sentiment(doc),
                "complexity": self._calculate_complexity(sentences, words),
                "themes": self._analyze_themes(doc, words_no_stop),
                "readability": self._calculate_readability(sentences, words),
                "emotional_progression": self._analyze_emotional_progression(verses),
                "narrative_structure": self._analyze_narrative_structure(verses),
                "key_motifs": self._detect_key_motifs(lines),
                "theme_clusters": self._analyze_theme_clusters(lines),
                "statistics": {
                    "word_count": len(words),
                    "unique_words": len(set(words_no_stop)),
                    "vocabulary_density": len(set(words_no_stop)) / len(words) if words else 0,
                    "sentence_count": len(sentences),
                    "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing lyrics: {e}")
            raise ValueError(f"Error analyzing lyrics: {str(e)}")
    
    async def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of given text."""
        if not text or not isinstance(text, str):
            raise ValueError("Invalid text input")
            
        doc = self.nlp(text.lower())
        return self._analyze_sentiment(doc)
    
    def _analyze_sentiment(self, doc: spacy.tokens.Doc) -> Dict[str, float]:
        """Analyze sentiment using spaCy and TextBlob."""
        # Use TextBlob for sentiment analysis
        blob = TextBlob(doc.text)
        
        # Calculate sentiment scores
        sentiment_scores = {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }
        
        # Add emotional analysis using custom lexicon
        emotional_scores = defaultdict(float)
        for token in doc:
            if token.text in self.sentiment_lexicon:
                emotional_scores[token.text] = self.sentiment_lexicon[token.text]
        
        sentiment_scores["emotional_scores"] = dict(emotional_scores)
        return sentiment_scores
    
    def _calculate_complexity(self, sentences: List[spacy.tokens.Span], words: List[str]) -> Dict[str, float]:
        """Calculate various complexity metrics for the lyrics."""
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # Calculate lexical diversity
        unique_words = set(words)
        lexical_diversity = len(unique_words) / len(words) if words else 0
        
        return {
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
            "lexical_diversity": lexical_diversity
        }
    
    def _analyze_themes(self, doc: spacy.tokens.Doc, words: List[str], top_n: int = 5) -> Dict[str, List[str]]:
        """Analyze main themes and topics in the lyrics."""
        # Get word frequency distribution
        word_freq = Counter(words)
        
        # Extract nouns and verbs using spaCy
        nouns = [token.text for token in doc if token.pos_ == "NOUN"]
        verbs = [token.text for token in doc if token.pos_ == "VERB"]
        
        # Extract named entities
        entities = [ent.text for ent in doc.ents]
        
        return {
            "top_words": [word for word, _ in word_freq.most_common(top_n)],
            "main_nouns": [word for word, _ in Counter(nouns).most_common(top_n)],
            "main_verbs": [word for word, _ in Counter(verbs).most_common(top_n)],
            "entities": list(set(entities))
        }
    
    def _calculate_readability(self, sentences: List[spacy.tokens.Span], words: List[str]) -> float:
        """Calculate a simplified readability score."""
        if not sentences or not words:
            return 0.0
        
        # Calculate average sentence length and syllable count
        avg_sentence_length = len(words) / len(sentences)
        
        # Estimate syllables (simplified)
        total_syllables = sum(self._count_syllables(word) for word in words)
        avg_syllables_per_word = total_syllables / len(words) if words else 0
        
        # Simplified Flesch Reading Ease formula adaptation
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Normalize to 0-1 scale
        return max(0, min(1, readability / 100))
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count in a word."""
        word = word.lower()
        vowels = "aeiouy"
        syllables = 0
        previous_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not previous_was_vowel:
                    syllables += 1
                previous_was_vowel = True
            else:
                previous_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllables > 1:
            syllables -= 1
        
        return max(1, syllables)
    
    def _split_into_verses(self, lines: List[str]) -> List[List[str]]:
        """Split lines into verses."""
        verses = []
        current_verse = []
        
        for line in lines:
            if line.strip():
                current_verse.append(line)
            else:
                if current_verse:
                    verses.append(current_verse)
                    current_verse = []
        
        if current_verse:
            verses.append(current_verse)
        
        return verses
    
    def _analyze_theme_clusters(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Analyze theme clusters using TF-IDF and K-means."""
        if len(lines) < 2:
            return []
        
        try:
            # Vectorize the lines
            vectors = self.vectorizer.fit_transform(lines)
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Determine optimal number of clusters
            n_clusters = min(3, len(lines))
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(vectors)
            
            # Extract top terms for each cluster
            cluster_results = []
            for i in range(n_clusters):
                cluster_center = kmeans.cluster_centers_[i]
                top_terms = self._get_top_terms(cluster_center, feature_names)
                cluster_results.append({
                    "cluster_id": i,
                    "top_terms": top_terms,
                    "lines_count": int(np.sum(clusters == i))
                })
            
            return cluster_results
            
        except Exception as e:
            logger.warning(f"Theme clustering failed: {e}")
            return []
    
    def _get_top_terms(self, vector, feature_names, top_n: int = 5) -> List[str]:
        """Get top terms from a cluster center vector."""
        top_indices = vector.argsort()[-top_n:][::-1]
        return [feature_names[i] for i in top_indices]
    
    def _analyze_emotional_progression(self, verses: List[List[str]]) -> List[Dict[str, float]]:
        """Analyze emotional progression through verses."""
        progression = []
        
        for i, verse in enumerate(verses):
            verse_text = " ".join(verse).lower()
            doc = self.nlp(verse_text)
            
            # Calculate sentiment for this verse
            blob = TextBlob(verse_text)
            
            # Count emotional words
            emotional_count = sum(1 for token in doc if token.text in self.sentiment_lexicon)
            
            verse_emotion = {
                "verse": i + 1,
                "polarity": blob.sentiment.polarity,
                "subjectivity": blob.sentiment.subjectivity,
                "emotional_density": emotional_count / len(verse) if verse else 0
            }
            
            progression.append(verse_emotion)
        
        return progression
    
    def _analyze_narrative_structure(self, verses: List[List[str]]) -> Dict[str, Any]:
        """Analyze narrative structure of the lyrics."""
        if not verses:
            return {"structure": "unknown", "verse_count": 0}
        
        verse_count = len(verses)
        
        # Analyze repetition patterns
        verse_similarities = []
        for i in range(len(verses)):
            for j in range(i + 1, len(verses)):
                similarity = self._calculate_verse_similarity(verses[i], verses[j])
                verse_similarities.append(similarity)
        
        avg_similarity = np.mean(verse_similarities) if verse_similarities else 0
        
        # Determine structure
        structure = self._identify_structure(verses)
        
        return {
            "structure": structure,
            "verse_count": verse_count,
            "repetition_score": avg_similarity,
            "avg_verse_length": np.mean([len(verse) for verse in verses])
        }
    
    def _calculate_verse_similarity(self, verse1: List[str], verse2: List[str]) -> float:
        """Calculate similarity between two verses."""
        text1 = " ".join(verse1).lower()
        text2 = " ".join(verse2).lower()
        
        # Simple word overlap similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _identify_structure(self, verses: List[List[str]]) -> str:
        """Identify the narrative structure pattern."""
        if len(verses) <= 2:
            return "simple"
        elif len(verses) <= 4:
            return "verse-chorus"
        else:
            return "complex"
    
    def _detect_key_motifs(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Detect recurring motifs and phrases."""
        motifs = []
        
        # Look for repeated phrases (2-4 words)
        phrases = defaultdict(int)
        
        for line in lines:
            words = line.lower().split()
            
            # Extract 2-4 word phrases
            for length in range(2, min(5, len(words) + 1)):
                for i in range(len(words) - length + 1):
                    phrase = " ".join(words[i:i + length])
                    if len(phrase) > 5:  # Ignore very short phrases
                        phrases[phrase] += 1
        
        # Filter repeated phrases
        for phrase, count in phrases.items():
            if count > 1:
                motifs.append({
                    "phrase": phrase,
                    "frequency": count,
                    "significance": count / len(lines)
                })
        
        # Sort by frequency and return top motifs
        motifs.sort(key=lambda x: x["frequency"], reverse=True)
        return motifs[:10]  # Return top 10 motifs 