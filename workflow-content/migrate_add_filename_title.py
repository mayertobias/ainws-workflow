#!/usr/bin/env python3
"""
Database migration: Add filename and title columns to lyrics_analysis_results table
"""

import asyncio
import logging
from sqlalchemy import text
from app.services.database_service import db_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def migrate_add_filename_title():
    """Add filename and title columns to existing table"""
    try:
        await db_service.initialize()
        
        async with db_service.get_session() as session:
            # Check if columns already exist
            check_columns = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'lyrics_analysis_results' 
            AND column_name IN ('filename', 'title');
            """
            
            result = session.execute(text(check_columns))
            existing_columns = [row[0] for row in result.fetchall()]
            
            if 'filename' not in existing_columns:
                logger.info("Adding filename column...")
                session.execute(text("ALTER TABLE lyrics_analysis_results ADD COLUMN filename VARCHAR(500);"))
                session.execute(text("CREATE INDEX idx_filename ON lyrics_analysis_results(filename);"))
                logger.info("✅ Added filename column")
            else:
                logger.info("filename column already exists")
            
            if 'title' not in existing_columns:
                logger.info("Adding title column...")
                session.execute(text("ALTER TABLE lyrics_analysis_results ADD COLUMN title VARCHAR(500);"))
                session.execute(text("CREATE INDEX idx_title ON lyrics_analysis_results(title);"))
                logger.info("✅ Added title column")
            else:
                logger.info("title column already exists")
            
            # Update existing records with generated titles
            logger.info("Updating existing records with generated titles...")
            update_query = """
            UPDATE lyrics_analysis_results 
            SET title = CONCAT('Analysis ', TO_CHAR(created_at, 'YYYY-MM-DD HH24:MI'))
            WHERE title IS NULL;
            """
            result = session.execute(text(update_query))
            updated_count = result.rowcount
            logger.info(f"✅ Updated {updated_count} existing records with generated titles")
            
            session.commit()
            logger.info("🎉 Migration completed successfully!")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        await db_service.close()

if __name__ == "__main__":
    asyncio.run(migrate_add_filename_title()) 