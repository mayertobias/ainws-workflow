#!/usr/bin/env python3
"""
Simple Lyrics Analyzer Test

A lightweight test that validates basic NLP functionality without requiring
the full microservice setup. Good for development and debugging.
"""

import sys
from pathlib import Path

def test_basic_nlp():
    """Test basic NLP libraries are working"""
    print("🧪 Testing Basic NLP Libraries")
    print("=" * 40)
    
    try:
        # Test TextBlob
        print("1. Testing TextBlob...")
        from textblob import TextBlob
        test_text = "I love this beautiful song. It makes me happy!"
        blob = TextBlob(test_text)
        print(f"✅ TextBlob sentiment: {blob.sentiment.polarity:.3f}")
        
        # Test spaCy
        print("2. Testing spaCy...")
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp("This is a test sentence with emotions.")
            print(f"✅ spaCy loaded with {len(doc)} tokens")
        except OSError:
            print("⚠️ spaCy model not found. Run: python -m spacy download en_core_web_sm")
        
        # Test scikit-learn
        print("3. Testing scikit-learn...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=10)
        test_docs = ["This is a test", "Another test document", "More test content"]
        vectors = vectorizer.fit_transform(test_docs)
        print(f"✅ TF-IDF vectorizer: {vectors.shape}")
        
        # Test numpy
        print("4. Testing numpy...")
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        print(f"✅ Numpy array: {arr.mean():.2f}")
        
        print("\n✅ All basic NLP libraries working!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_lyrics_analysis():
    """Test lyrics analysis with sample text"""
    print("\n🎵 Testing Lyrics Analysis")
    print("=" * 40)
    
    try:
        # Load test lyrics
        lyrics_path = Path(__file__).parent.parent.parent / "lyrics" / "popularity_9" / "Billie Jean.txt"
        
        if lyrics_path.exists():
            with open(lyrics_path, 'r', encoding='utf-8') as f:
                lyrics = f.read()
            print(f"✅ Loaded test lyrics ({len(lyrics)} chars)")
        else:
            lyrics = """This is a test song about love and emotions
It talks about dancing and feeling the music
The rhythm makes me happy and excited
Love is in the air, dancing through the night"""
            print("⚠️ Using fallback test lyrics")
        
        # Test basic text analysis
        from textblob import TextBlob
        blob = TextBlob(lyrics)
        
        print(f"\n📊 Basic Analysis:")
        print(f"Word count: {len(lyrics.split())}")
        print(f"Sentence count: {len(blob.sentences)}")
        print(f"Sentiment polarity: {blob.sentiment.polarity:.3f}")
        print(f"Sentiment subjectivity: {blob.sentiment.subjectivity:.3f}")
        
        # Test advanced features if available
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(lyrics)
            
            # Extract entities
            entities = [ent.text for ent in doc.ents]
            print(f"Named entities: {entities[:5]}")
            
            # Extract nouns and verbs
            nouns = [token.text for token in doc if token.pos_ == "NOUN"]
            verbs = [token.text for token in doc if token.pos_ == "VERB"]
            print(f"Top nouns: {list(set(nouns))[:5]}")
            print(f"Top verbs: {list(set(verbs))[:5]}")
            
        except Exception as e:
            print(f"⚠️ Advanced analysis skipped: {e}")
        
        print("\n✅ Lyrics analysis test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Lyrics analysis failed: {e}")
        return False

def test_performance():
    """Test performance with larger text"""
    print("\n⚡ Testing Performance")
    print("=" * 40)
    
    try:
        import time
        from textblob import TextBlob
        
        # Create larger test text
        test_text = """
        This is a performance test with multiple sentences.
        We want to see how fast the analysis can process text.
        The lyrics contain emotions, themes, and narrative structure.
        Music brings joy and happiness to people around the world.
        Dance and rhythm create connections between different cultures.
        """ * 10  # Repeat 10 times
        
        start_time = time.time()
        blob = TextBlob(test_text)
        sentiment = blob.sentiment
        sentences = blob.sentences
        words = blob.words
        end_time = time.time()
        
        processing_time = end_time - start_time
        words_per_second = len(words) / processing_time if processing_time > 0 else 0
        
        print(f"Text length: {len(test_text)} chars")
        print(f"Word count: {len(words)}")
        print(f"Sentence count: {len(sentences)}")
        print(f"Processing time: {processing_time:.3f}s")
        print(f"Speed: {words_per_second:.0f} words/second")
        
        print("\n✅ Performance test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

if __name__ == "__main__":
    print("🎵 Simple Lyrics Analyzer Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic NLP Libraries", test_basic_nlp),
        ("Lyrics Analysis", test_lyrics_analysis),
        ("Performance", test_performance)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your NLP environment is ready.")
        sys.exit(0)
    else:
        print("🔧 Some tests failed. Check the errors above.")
        sys.exit(1) 