#!/bin/bash

echo "🧹 Starting comprehensive database cleanup..."
echo "================================"

# Audio Database Cleanup
echo "📊 Cleaning AUDIO database (postgres-audio)..."
docker exec -i postgres-audio psql -U postgres -d workflow_audio << 'EOF'
-- Clean audio analysis results
TRUNCATE TABLE audio_analysis_results CASCADE;
TRUNCATE TABLE feature_cache CASCADE;
TRUNCATE TABLE training_jobs CASCADE;
TRUNCATE TABLE model_registry CASCADE;

-- Reset sequences
ALTER SEQUENCE audio_analysis_results_id_seq RESTART WITH 1;
ALTER SEQUENCE training_jobs_id_seq RESTART WITH 1;
ALTER SEQUENCE model_registry_id_seq RESTART WITH 1;

-- Show cleanup results
SELECT 'audio_analysis_results' as table_name, COUNT(*) as remaining_records FROM audio_analysis_results
UNION ALL
SELECT 'feature_cache' as table_name, COUNT(*) as remaining_records FROM feature_cache
UNION ALL  
SELECT 'training_jobs' as table_name, COUNT(*) as remaining_records FROM training_jobs
UNION ALL
SELECT 'model_registry' as table_name, COUNT(*) as remaining_records FROM model_registry;
EOF

echo "✅ Audio database cleanup completed"

# Content Database Cleanup
echo "📝 Cleaning CONTENT database (postgres-content)..."
docker exec -i postgres-content psql -U postgres -d workflow_content << 'EOF'
-- Clean content analysis results
TRUNCATE TABLE lyrics_analysis_results CASCADE;

-- Reset sequences
ALTER SEQUENCE lyrics_analysis_results_id_seq RESTART WITH 1;

-- Show cleanup results
SELECT 'lyrics_analysis_results' as table_name, COUNT(*) as remaining_records FROM lyrics_analysis_results;
EOF

echo "✅ Content database cleanup completed"

echo "================================"
echo "🎯 Database cleanup summary:"
echo "- Audio database: All training data cleared"
echo "- Content database: All lyrics analysis data cleared"
echo "- Sequences reset to start from 1"
echo "✅ Ready for fresh training!" 