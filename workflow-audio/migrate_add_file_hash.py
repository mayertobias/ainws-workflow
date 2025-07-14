#!/usr/bin/env python3
"""
Database migration to add file_hash column to audio_analysis_results table
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent / "app"))

from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def migrate_database():
    """Add file_hash column to audio_analysis_results table"""
    
    # Use database URL with localhost (since we're running from host)
    database_url = os.getenv('AUDIO_DATABASE_URL', 'postgresql://postgres:postgres@localhost:5435/workflow_audio')
    
    # Create database engine
    engine = create_engine(database_url)
    
    try:
        with engine.connect() as connection:
            # Check if file_hash column already exists
            check_sql = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'audio_analysis_results' 
            AND column_name = 'file_hash'
            """
            
            result = connection.execute(text(check_sql))
            existing = result.fetchone()
            
            if existing:
                logger.info("✅ file_hash column already exists")
                return
            
            # Add the file_hash column
            alter_sql = """
            ALTER TABLE audio_analysis_results 
            ADD COLUMN file_hash VARCHAR(64);
            """
            
            logger.info("🔧 Adding file_hash column to audio_analysis_results table...")
            connection.execute(text(alter_sql))
            
            # Add index for performance
            index_sql = """
            CREATE INDEX IF NOT EXISTS idx_audio_analysis_file_hash 
            ON audio_analysis_results(file_hash);
            """
            
            logger.info("🔧 Adding index on file_hash column...")
            connection.execute(text(index_sql))
            
            # Commit all changes
            connection.commit()
            
            logger.info("✅ Database migration completed successfully!")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(migrate_database())