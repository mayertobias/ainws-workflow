#!/bin/bash

# Start script for workflow-content microservice

echo "Starting workflow-content service..."

# Check if spaCy model is available
echo "Checking spaCy model..."
python -c "import spacy; spacy.load('en_core_web_sm')" || {
    echo "Downloading spaCy model..."
    python -m spacy download en_core_web_sm
}

# Create necessary directories
mkdir -p /tmp/lyrics /tmp/output

# Download NLTK data for TextBlob
echo "Downloading NLTK data..."
python -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('brown', quiet=True)
except:
    pass
"

echo "Starting FastAPI application..."

# Start the application
if [ "$DEBUG" = "true" ]; then
    echo "Running in debug mode..."
    uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8002} --reload --log-level debug
else
    echo "Running in production mode..."
    uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8002} --workers 4
fi 