#!/usr/bin/env python3
"""
Test script for LyricsAnalyzer

Tests the comprehensive lyrics analysis functionality using provided test lyrics
to validate sentiment analysis, theme detection, and other features.
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path
import json

# Fix the import path - add both app and parent directories
current_dir = Path(__file__).parent
app_dir = current_dir.parent / "app"
sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(current_dir.parent))

try:
    from app.services.lyrics_analyzer import LyricsAnalyzer
except ImportError:
    try:
        from services.lyrics_analyzer import LyricsAnalyzer
    except ImportError:
        # Last resort - direct file import
        import importlib.util
        analyzer_path = current_dir.parent / "app" / "services" / "lyrics_analyzer.py"
        spec = importlib.util.spec_from_file_location("lyrics_analyzer", analyzer_path)
        lyrics_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lyrics_module)
        LyricsAnalyzer = lyrics_module.LyricsAnalyzer

class TestLyricsAnalyzer:
    """Test suite for LyricsAnalyzer"""
    
    @pytest.fixture
    async def analyzer(self):
        """Create a LyricsAnalyzer instance for testing"""
        return LyricsAnalyzer()
    
    @pytest.fixture
    def test_lyrics(self):
        """Load test lyrics from the provided file"""
        lyrics_path = Path(__file__).parent.parent.parent / "lyrics" / "popularity_9" / "Billie Jean.txt"
        
        if lyrics_path.exists():
            with open(lyrics_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # Fallback sample for testing if file not found
            return "This is a test song with emotions. Love and happiness fill the air. Dance and music bring joy."
    
    @pytest.fixture
    def short_sample(self):
        """Short sample text for basic tests"""
        return "I love this beautiful song. It makes me happy and excited!"

    @pytest.mark.asyncio
    async def test_analyzer_initialization(self, analyzer):
        """Test that the analyzer initializes correctly"""
        assert analyzer is not None
        assert hasattr(analyzer, 'nlp')
        assert hasattr(analyzer, 'vectorizer')
        assert hasattr(analyzer, 'sentiment_lexicon')
        assert hasattr(analyzer, 'theme_keywords')
        assert len(analyzer.sentiment_lexicon) > 0
        assert len(analyzer.theme_keywords) > 0

    @pytest.mark.asyncio
    async def test_analyze_structure(self, analyzer, test_lyrics):
        """Test that analysis returns correct structure"""
        result = await analyzer.analyze(test_lyrics)
        
        # Check main structure
        assert isinstance(result, dict)
        expected_keys = [
            "sentiment", "complexity", "themes", "readability", 
            "emotional_progression", "narrative_structure", 
            "key_motifs", "theme_clusters", "statistics"
        ]
        
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_sentiment_analysis(self, analyzer, short_sample):
        """Test sentiment analysis functionality"""
        result = await analyzer.analyze_sentiment(short_sample)
        
        assert isinstance(result, dict)
        assert "polarity" in result
        assert "subjectivity" in result
        assert isinstance(result["polarity"], float)
        assert isinstance(result["subjectivity"], float)
        assert -1.0 <= result["polarity"] <= 1.0
        assert 0.0 <= result["subjectivity"] <= 1.0
        
        # Should detect positive sentiment in the sample
        assert result["polarity"] > 0  # Contains positive words

    @pytest.mark.asyncio
    async def test_statistics_calculation(self, analyzer, test_lyrics):
        """Test basic statistics calculation"""
        result = await analyzer.analyze(test_lyrics)
        stats = result["statistics"]
        
        required_stats = [
            "word_count", "unique_words", "vocabulary_density", 
            "sentence_count", "avg_words_per_sentence"
        ]
        
        for stat in required_stats:
            assert stat in stats
            assert isinstance(stats[stat], (int, float))
        
        # Basic validation
        assert stats["word_count"] > 0
        assert stats["unique_words"] > 0
        assert stats["sentence_count"] > 0
        assert 0 <= stats["vocabulary_density"] <= 1

    @pytest.mark.asyncio
    async def test_complexity_metrics(self, analyzer, test_lyrics):
        """Test complexity calculation"""
        result = await analyzer.analyze(test_lyrics)
        complexity = result["complexity"]
        
        assert "avg_sentence_length" in complexity
        assert "avg_word_length" in complexity
        assert "lexical_diversity" in complexity
        
        assert complexity["avg_sentence_length"] > 0
        assert complexity["avg_word_length"] > 0
        assert 0 <= complexity["lexical_diversity"] <= 1

    @pytest.mark.asyncio
    async def test_theme_detection(self, analyzer, test_lyrics):
        """Test theme detection functionality"""
        result = await analyzer.analyze(test_lyrics)
        themes = result["themes"]
        
        required_theme_keys = ["top_words", "main_nouns", "main_verbs", "entities"]
        
        for key in required_theme_keys:
            assert key in themes
            assert isinstance(themes[key], list)

    @pytest.mark.asyncio
    async def test_narrative_structure(self, analyzer, test_lyrics):
        """Test narrative structure analysis"""
        result = await analyzer.analyze(test_lyrics)
        structure = result["narrative_structure"]
        
        assert "structure" in structure
        assert "verse_count" in structure
        assert "repetition_score" in structure
        assert "avg_verse_length" in structure
        
        assert structure["verse_count"] >= 0
        assert 0 <= structure["repetition_score"] <= 1

    @pytest.mark.asyncio
    async def test_emotional_progression(self, analyzer, test_lyrics):
        """Test emotional progression tracking"""
        result = await analyzer.analyze(test_lyrics)
        progression = result["emotional_progression"]
        
        assert isinstance(progression, list)
        
        for verse_emotion in progression:
            assert "verse" in verse_emotion
            assert "polarity" in verse_emotion
            assert "subjectivity" in verse_emotion
            assert "emotional_density" in verse_emotion
            
            assert isinstance(verse_emotion["verse"], int)
            assert -1.0 <= verse_emotion["polarity"] <= 1.0
            assert 0.0 <= verse_emotion["subjectivity"] <= 1.0
            assert verse_emotion["emotional_density"] >= 0

    @pytest.mark.asyncio
    async def test_motif_detection(self, analyzer, test_lyrics):
        """Test key motif detection"""
        result = await analyzer.analyze(test_lyrics)
        motifs = result["key_motifs"]
        
        assert isinstance(motifs, list)
        
        for motif in motifs:
            assert "phrase" in motif
            assert "frequency" in motif
            assert "significance" in motif
            
            assert isinstance(motif["phrase"], str)
            assert isinstance(motif["frequency"], int)
            assert isinstance(motif["significance"], float)
            assert motif["frequency"] > 1

    @pytest.mark.asyncio
    async def test_readability_score(self, analyzer, test_lyrics):
        """Test readability calculation"""
        result = await analyzer.analyze(test_lyrics)
        readability = result["readability"]
        
        assert isinstance(readability, float)
        assert 0 <= readability <= 1

    @pytest.mark.asyncio
    async def test_error_handling(self, analyzer):
        """Test error handling with invalid inputs"""
        
        # Test empty input
        with pytest.raises(ValueError):
            await analyzer.analyze("")
        
        # Test None input
        with pytest.raises(ValueError):
            await analyzer.analyze(None)
        
        # Test non-string input
        with pytest.raises(ValueError):
            await analyzer.analyze(123)

    @pytest.mark.asyncio
    async def test_sentiment_lexicon(self, analyzer):
        """Test sentiment lexicon functionality"""
        lexicon = analyzer.sentiment_lexicon
        
        # Should have positive and negative words
        positive_words = [word for word, score in lexicon.items() if score > 0]
        negative_words = [word for word, score in lexicon.items() if score < 0]
        
        assert len(positive_words) > 0
        assert len(negative_words) > 0
        
        # Check specific sentiment words exist
        sentiment_words = ["love", "happy", "sad"]
        for word in sentiment_words:
            assert word in lexicon

    def test_syllable_counting(self, analyzer):
        """Test syllable counting accuracy"""
        test_cases = [
            ("hello", 2),
            ("beautiful", 3), 
            ("a", 1),
            ("the", 1)
        ]
        
        for word, expected_min in test_cases:
            syllables = analyzer._count_syllables(word)
            assert syllables >= expected_min

    @pytest.mark.asyncio
    async def test_consistency(self, analyzer, test_lyrics):
        """Test analysis consistency across multiple runs"""
        result1 = await analyzer.analyze(test_lyrics)
        result2 = await analyzer.analyze(test_lyrics)
        
        # Core metrics should be identical
        assert result1["statistics"]["word_count"] == result2["statistics"]["word_count"]
        assert result1["statistics"]["sentence_count"] == result2["statistics"]["sentence_count"]
        assert result1["sentiment"]["polarity"] == result2["sentiment"]["polarity"]

# Standalone test runner
if __name__ == "__main__":
    async def run_standalone_test():
        """Run a basic test without pytest framework"""
        print("🎵 Testing Lyrics Analyzer")
        print("=" * 50)
        
        try:
            # Initialize analyzer
            print("1. Initializing analyzer...")
            analyzer = LyricsAnalyzer()
            print("✅ Analyzer initialized")
            
            # Load test file
            print("2. Loading test lyrics...")
            lyrics_path = Path(__file__).parent.parent.parent / "lyrics" / "popularity_9" / "Billie Jean.txt"
            
            if lyrics_path.exists():
                with open(lyrics_path, 'r', encoding='utf-8') as f:
                    test_text = f.read()
                print(f"✅ Loaded test file ({len(test_text)} chars)")
            else:
                test_text = "This is a test song about emotions and love."
                print("⚠️ Using fallback test text")
            
            # Run analysis
            print("3. Running analysis...")
            result = await analyzer.analyze(test_text)
            print("✅ Analysis completed")
            
            # Display key results
            print(f"\n📊 Results Summary:")
            print(f"Words: {result['statistics']['word_count']}")
            print(f"Sentences: {result['statistics']['sentence_count']}")
            print(f"Sentiment: {result['sentiment']['polarity']:.3f}")
            print(f"Readability: {result['readability']:.3f}")
            print(f"Structure: {result['narrative_structure']['structure']}")
            
            if result['key_motifs']:
                print(f"Repeated phrases found: {len(result['key_motifs'])}")
            
            print("\n✅ All tests completed successfully!")
            return True
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            return False
    
    # Run the test
    success = asyncio.run(run_standalone_test())
    sys.exit(0 if success else 1) 