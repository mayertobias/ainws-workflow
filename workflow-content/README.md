# workflow-content: Content Analysis Microservice

This microservice handles lyrics analysis, text processing, and sentiment analysis for the workflow system. It's part of the microservices architecture refactoring of the monolithic workflow-server application.

## Overview

The workflow-content service provides comprehensive text analysis capabilities including:

- **Sentiment Analysis**: Polarity, subjectivity, and emotional scoring
- **Complexity Metrics**: Sentence length, word length, lexical diversity
- **Theme Analysis**: Topic extraction, named entity recognition
- **Readability Scoring**: Flesch reading ease adaptation
- **Narrative Structure**: Verse analysis and progression
- **Motif Detection**: Recurring phrase identification
- **HSS Features**: Hit Song Science feature extraction for ML models

## API Endpoints

### Health Checks
- `GET /health` - Service health check
- `GET /health/ready` - Readiness probe
- `GET /health/live` - Liveness probe

### Analysis Endpoints
- `POST /analyze/lyrics` - Comprehensive lyrics analysis
- `POST /analyze/sentiment` - Text sentiment analysis
- `GET /analyze/features/hss` - Extract HSS features for ML models

## Quick Start

### Using Docker Compose

1. **Start the service**:
   ```bash
   cd workflow-content
   docker-compose up --build
   ```

2. **Test the service**:
   ```bash
   python test_service.py
   ```

### Manual Development Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Set environment variables**:
   ```bash
   export DEBUG=true
   export PORT=8002
   export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/workflow_content
   export REDIS_URL=redis://localhost:6379/1
   ```

3. **Run the service**:
   ```bash
   ./start.sh
   # or
   uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
   ```

## Configuration

The service can be configured using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8002 | Service port |
| `DEBUG` | false | Debug mode |
| `DATABASE_URL` | postgres://... | PostgreSQL connection |
| `REDIS_URL` | redis://... | Redis connection |
| `LYRICS_DIR` | /tmp/lyrics | Lyrics storage directory |
| `OUTPUT_DIR` | /tmp/output | Output directory |
| `MAX_TEXT_LENGTH` | 10000 | Maximum text length |

## API Usage Examples

### Analyze Lyrics

```bash
curl -X POST "http://localhost:8002/analyze/lyrics" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Walking down the street tonight\nFeeling like everything'\''s alright",
    "language": "en",
    "analysis_type": "comprehensive"
  }'
```

### Get Sentiment

```bash
curl -X POST "http://localhost:8002/analyze/sentiment" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I am feeling so happy today!"
  }'
```

### Extract HSS Features

```bash
curl "http://localhost:8002/analyze/features/hss?text=Walking%20down%20the%20street%20tonight"
```

## Response Format

### Lyrics Analysis Response

```json
{
  "status": "success",
  "results": {
    "sentiment": {
      "polarity": 0.25,
      "subjectivity": 0.6,
      "emotional_scores": {"happy": 0.8}
    },
    "complexity": {
      "avg_sentence_length": 8.5,
      "avg_word_length": 4.2,
      "lexical_diversity": 0.75
    },
    "themes": {
      "top_words": ["love", "heart", "time"],
      "main_nouns": ["street", "night", "feeling"],
      "main_verbs": ["walking", "feeling"],
      "entities": []
    },
    "readability": 0.65,
    "statistics": {
      "word_count": 42,
      "unique_words": 35,
      "vocabulary_density": 0.83,
      "sentence_count": 5,
      "avg_words_per_sentence": 8.4
    }
  },
  "timestamp": "2024-12-10T10:30:00Z",
  "processing_time_ms": 1250.5
}
```

## Integration with Other Services

This service is designed to be called by:

- **workflow-orchestrator**: For complete song analysis workflows
- **workflow-ml-training**: For extracting training features
- **workflow-ml-prediction**: For real-time feature extraction
- **workflow-gateway**: Direct user requests

### Service-to-Service Communication

```python
import httpx

async def get_lyrics_features(lyrics_text: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://workflow-content:8002/analyze/lyrics",
            json={"text": lyrics_text}
        )
        return response.json()
```

## Development

### Project Structure

```
workflow-content/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── api/                    # API endpoints
│   │   ├── health.py          # Health checks
│   │   └── lyrics.py          # Lyrics analysis endpoints
│   ├── models/                 # Pydantic models
│   │   ├── lyrics.py          # Request/response models
│   │   └── responses.py       # Common responses
│   ├── services/               # Business logic
│   │   └── lyrics_analyzer.py # Lyrics analysis service
│   └── config/                 # Configuration
│       └── settings.py        # Application settings
├── tests/                      # Test suites
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Development environment
├── requirements.txt            # Python dependencies
├── start.sh                   # Startup script
└── test_service.py            # Service tests
```

### Running Tests

```bash
# Unit tests
python -m pytest tests/

# Integration tests
python test_service.py

# Load testing
# TODO: Add load testing instructions
```

### Adding New Features

1. Add new analysis logic to `LyricsAnalyzer`
2. Create request/response models in `models/`
3. Add API endpoints in `api/`
4. Update tests and documentation

## Performance

- **Average Response Time**: < 1500ms for comprehensive analysis
- **Throughput**: ~10 requests/second on single instance
- **Memory Usage**: ~200MB base + ~50MB per concurrent request
- **Scalability**: Horizontally scalable (stateless)

## Dependencies

### Core Dependencies
- **FastAPI**: Web framework
- **spaCy**: NLP processing
- **TextBlob**: Sentiment analysis
- **scikit-learn**: ML algorithms for clustering
- **uvicorn**: ASGI server

### NLP Models
- **en_core_web_sm**: English language model (spaCy)
- **NLTK data**: TextBlob dependencies

## Deployment

### Docker

The service includes health checks and graceful shutdown:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1
```

### Kubernetes

Example deployment configuration:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: workflow-content
spec:
  replicas: 3
  selector:
    matchLabels:
      app: workflow-content
  template:
    metadata:
      labels:
        app: workflow-content
    spec:
      containers:
      - name: workflow-content
        image: workflow-content:latest
        ports:
        - containerPort: 8002
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8002
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8002
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Monitoring

### Metrics

The service exposes metrics for:
- Request count and latency
- Error rates
- Processing time per analysis type
- Memory and CPU usage

### Logging

Structured logging with correlation IDs:

```json
{
  "timestamp": "2024-12-10T10:30:00Z",
  "service": "workflow-content",
  "level": "INFO",
  "correlation_id": "req-123",
  "message": "Lyrics analysis completed",
  "duration_ms": 1250,
  "text_length": 842
}
```

## Troubleshooting

### Common Issues

1. **spaCy model not found**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **Memory issues with large texts**:
   - Check `MAX_TEXT_LENGTH` setting
   - Monitor memory usage
   - Consider text chunking

3. **Slow analysis**:
   - Check if NLP models are properly loaded
   - Monitor CPU usage
   - Consider caching results

### Health Checks

```bash
# Service health
curl http://localhost:8002/health

# Check if ready to receive traffic
curl http://localhost:8002/health/ready

# Check if service is alive
curl http://localhost:8002/health/live
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Submit a pull request

## License

This project is part of the workflow system and follows the same licensing terms. 