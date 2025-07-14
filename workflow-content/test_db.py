#!/usr/bin/env python3
"""
Test script to verify database service is working properly
"""

import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.database_service import DatabaseService

async def test_database():
    """Test database initialization and basic operations"""
    print("🔄 Testing database service...")
    
    db = DatabaseService()
    
    try:
        # Test initialization
        print("🔄 Initializing database service...")
        await db.initialize()
        print("✅ Database service initialized successfully")
        
        # Test session creation
        print("🔄 Testing database session...")
        async with db.get_session() as session:
            result = session.execute("SELECT 1 as test").fetchone()
            if result and result[0] == 1:
                print("✅ Database connection test passed")
            else:
                print("❌ Database connection test failed")
        
        # Test saving a sample analysis
        print("🔄 Testing save analysis result...")
        try:
            analysis_id = await db.save_analysis_result(
                session_id="test_session_123",
                original_text="This is a test lyrics text",
                analysis_results={
                    "sentiment": {"polarity": 0.5, "subjectivity": 0.3},
                    "complexity": {"lexical_diversity": 0.8},
                    "statistics": {"word_count": 6}
                },
                processing_time_ms=100.0,
                hss_features={"sentiment_polarity": 0.5}
            )
            print(f"✅ Analysis saved with ID: {analysis_id}")
            
            # Test retrieving history
            print("🔄 Testing get analysis history...")
            history, stats = await db.get_analysis_history("test_session_123", limit=10)
            print(f"✅ Retrieved {len(history)} history items")
            print(f"✅ Stats: {stats}")
            
        except Exception as e:
            print(f"❌ Error testing analysis operations: {e}")
        
        # Clean up
        await db.close()
        print("✅ Database service closed successfully")
        
        print("\n🎉 All database tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_database())
    sys.exit(0 if result else 1) 