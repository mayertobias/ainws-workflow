# Microservice: workflow-audio

## 1. Executive Summary

### 1.1. Core Purpose
Processes raw audio files to extract a comprehensive set of 50+ musical features using Essentia's advanced audio analysis algorithms, providing the foundational audio intelligence for the workflow ecosystem's music analysis, ML training, and hit prediction capabilities.

### 1.2. Key Responsibilities
- **Comprehensive Feature Extraction**: Extracts 50+ audio features across 7 specialized extractors (audio, rhythm, tonal, timbre, dynamics, mood, genre)
- **Persistent Data Storage**: Stores all analysis results in PostgreSQL with complete audit trails and data lineage tracking
- **Server-Side Deduplication**: Prevents duplicate processing through intelligent filename and content-based deduplication
- **Idempotency Support**: Ensures identical requests return consistent results without reprocessing
- **ML Pipeline Integration**: Provides optimized feature vectors for machine learning model training and prediction
- **Real-time Analysis**: Processes audio files in 5-15 seconds with comprehensive feature extraction
- **Caching & Performance**: Implements Redis caching for sub-second response times on previously analyzed files

### 1.3. Role in the Ecosystem
This service acts as the foundational audio processing layer in the workflow ecosystem. It is primarily called by `workflow-orchestrator` at the start of new song analysis pipelines and serves as the data source for downstream services. Its comprehensive audio features are consumed by `workflow-intelligence` for novelty analysis and insights generation, and by `workflow-ml-prediction` for hit prediction models. The service's persistent storage architecture ensures analysis results are preserved across system restarts, solving critical data loss issues.

---

## 2. Technical Deep Dive

### 2.1. Technology Stack
- **Language:** Python 3.9.18
- **Framework:** FastAPI 0.104.1
- **Key Libraries:** 
  - essentia-tensorflow (advanced audio analysis)
  - librosa 0.10.1 (audio processing)
  - tensorflow 2.8.4 (ML models for mood/genre classification)
  - numpy 1.24.3 (numerical computing)
  - scikit-learn 1.3.2 (ML algorithms)
- **Databases:** PostgreSQL 15 (persistent storage), Redis 5.0.1 (caching)
- **Container:** Docker with optimized multi-stage builds

### 2.2. Environment Configuration
| Variable | Description | Example Value |
|----------|-------------|---------------|
| `AUDIO_DEBUG` | Enable debug logging and detailed output | `false` |
| `AUDIO_DATABASE_URL` | PostgreSQL connection string for persistent storage | `postgresql://postgres:postgres@postgres-audio:5432/workflow_audio` |
| `AUDIO_REDIS_URL` | Redis connection string for caching and idempotency | `redis://redis-audio:6379/0` |
| `AUDIO_MAX_FILE_SIZE` | Maximum audio file size for processing | `100MB` |
| `AUDIO_ENABLE_RESULT_PERSISTENCE` | Enable database persistence of analysis results | `true` |
| `AUDIO_ENABLE_FEATURE_CACHING` | Enable Redis caching for performance | `true` |
| `AUDIO_ENABLE_IDEMPOTENCY` | Enable idempotency key support | `true` |
| `ESSENTIA_MODELS_PATH` | Path to Essentia TensorFlow models | `/app/models/essentia` |
| `AUDIO_LOG_LEVEL` | Logging verbosity level | `INFO` |

---

## 3. API & Data Contracts

### 3.1. API Endpoints

#### `POST /analyze/audio`
- **Description:** Primary comprehensive analysis endpoint that processes audio files and returns 50+ extracted features across 7 specialized extractors. This endpoint always performs comprehensive analysis and includes intelligent server-side deduplication.
- **Request Body Schema (`multipart/form-data`):**
  ```
  file: binary (audio file - WAV, MP3, FLAC, AAC, M4A, OGG)
  ```
- **Headers:**
  - `X-Session-ID`: Optional session identifier for tracking
  - `Workflow-ID`: Parent workflow identifier for audit trails
  - `Correlation-ID`: Request correlation for distributed tracing
- **Success Response (200 OK):**
  ```json
  {
    "status": "success",
    "analysis_id": "audio_analysis_abc123",
    "database_id": 42,
    "cached": false,
    "analysis_type": "comprehensive",
    "processing_time_ms": 8500,
    "filename": "song.mp3",
    "results": {
      "features": {
        "tempo": 120.0,
        "energy": 0.85,
        "valence": 0.73,
        "acousticness": 0.45,
        "danceability": 0.68,
        "instrumentalness": 0.12,
        "liveness": 0.08,
        "speechiness": 0.04,
        "loudness": -8.2,
        "key": "F#",
        "mode": "major",
        "spectral_centroid": 2500.0,
        "mood_happy": 0.8,
        "genre_pop": 0.7,
        "duration_ms": 185000,
        "time_signature": 4
      },
      "extractor_types": ["audio", "rhythm", "tonal", "timbre", "dynamics", "mood", "genre"],
      "comprehensive_analysis": true
    },
    "audit": {
      "created_at": "2024-01-15T10:30:00Z",
      "workflow_id": "workflow456",
      "requested_by": "user123"
    }
  }
  ```
- **Error Responses:**
  - `400 Bad Request`: Invalid audio file format or corrupted file
  - `413 Payload Too Large`: File exceeds maximum size limit (100MB)
  - `422 Unprocessable Entity`: File format not supported or analysis failed
  - `500 Internal Server Error`: Unexpected error during feature extraction

#### `GET /analyze/features/audio`
- **Description:** ML-optimized feature extraction endpoint that returns comprehensive features formatted for machine learning pipelines. Used by ML training and prediction services.
- **Query Parameters:**
  - `file_id`: Analysis ID from previous analysis
- **Success Response (200 OK):**
  ```json
  {
    "file_id": "audio_analysis_abc123",
    "comprehensive_features": {
      "basic_features": [0.85, 0.73, 120.0, 0.68, 0.45, 0.12, 0.08, 0.04, -8.2],
      "spectral_features": [2500.0, 4200.0, 0.02, [13 MFCC coefficients]],
      "tonal_features": [6, 1, 0.87],
      "rhythm_features": [0.92, 0.78, 4],
      "mood_features": [0.8, 0.2, 0.1, 0.6],
      "genre_features": [0.7, 0.3, 0.1, 0.05]
    },
    "feature_names": ["energy", "valence", "tempo", "..."],
    "metadata": {
      "analysis_date": "2024-01-15T10:30:00Z",
      "processing_time_ms": 8500,
      "extractor_version": "2.0.0"
    }
  }
  ```

#### `POST /analyze/persistent`
- **Description:** Microservice integration endpoint designed for inter-service communication with full persistence, idempotency, and audit trail support.
- **Headers:**
  - `Idempotency-Key`: Unique key for request deduplication
  - `Workflow-ID`: Parent workflow identifier
  - `Correlation-ID`: Request correlation identifier
  - `Requested-By`: Identifier of requesting service
- **Request/Response:** Similar to `/analyze/audio` but with enhanced metadata

#### `GET /analysis/{analysis_id}`
- **Description:** Retrieve stored analysis results by analysis ID from persistent storage.
- **Path Parameters:**
  - `analysis_id`: Unique analysis identifier
- **Success Response:** Complete analysis result with audit trail

#### `GET /health`
- **Description:** Service health check endpoint including database and cache connectivity status.
- **Success Response (200 OK):**
  ```json
  {
    "status": "healthy",
    "service": "workflow-audio-analysis",
    "version": "2.0.0",
    "capabilities": {
      "comprehensive_analysis": true,
      "persistent_storage": true,
      "caching": true,
      "idempotency": true
    },
    "database_connected": true,
    "cache_connected": true,
    "extractors_available": 7
  }
  ```

### 3.2. Core Data Models
- **`AudioAnalysisRequest`**: Input model for file-based analysis requests with analysis type and extractor specifications
- **`AudioAnalysisResponse`**: Comprehensive response model containing analysis results, metadata, and audit information
- **`AudioAnalysisResult`**: Database model for persistent storage including features, file metadata, and audit trails
- **`FeatureCache`**: Redis cache model for high-performance feature retrieval
- **`IdempotencyKey`**: Model for request deduplication and idempotency support

---

## 4. Key Internal Logic & Business Rules

### 4.1. Feature Extraction Pipeline
- **7-Stage Extraction Process**: Each audio file is processed through 7 specialized extractors in sequence (audio, rhythm, tonal, timbre, dynamics, mood, genre)
- **Essentia MusicExtractor**: Uses Essentia's comprehensive MusicExtractor with TensorFlow models for mood and genre classification
- **Normalization Algorithm**: All features are normalized to 0-1 range using domain-specific normalization functions
- **Quality Validation**: Extracted features undergo validation checks for NaN/Infinity values and are cleaned before storage

### 4.2. Server-Side Deduplication Algorithm
- **Filename-Based Deduplication**: Primary deduplication uses original filename matching to prevent reprocessing identical files
- **Content-Based Verification**: Secondary deduplication uses SHA-256 file hash for content verification
- **Deterministic Analysis IDs**: Analysis IDs are generated deterministically based on file content and parameters for consistent identification
- **Force Reanalysis Override**: `force_reanalysis` parameter bypasses deduplication for intentional reprocessing

### 4.3. Idempotency Implementation
- **Request Hashing**: Idempotency keys are combined with request parameter hashes for unique identification
- **TTL-Based Expiration**: Idempotency records expire after 2 hours to balance consistency and storage efficiency
- **Response Caching**: Identical requests return cached responses without reprocessing

### 4.4. Persistent Storage Strategy
- **Complete Audit Trail**: All analysis requests, results, and metadata are stored with full audit trails
- **Event Sourcing**: Critical events are logged to EventStore for complete data lineage tracking
- **Data Integrity**: Feature checksums ensure data integrity and detect corruption
- **Retention Policy**: Analysis results are retained for 30 days with configurable archival policies

### 4.5. Performance Optimization Rules
- **Redis Caching**: Frequently accessed analysis results are cached in Redis for sub-second retrieval
- **Connection Pooling**: Database connections use connection pooling with configurable pool sizes
- **Concurrent Processing**: Service supports up to 10 concurrent analyses with queue management
- **Memory Management**: Temporary audio files are automatically cleaned up after processing

---

## 5. Inter-Service Communication

### 5.1. Upstream Dependencies (Services it Calls)
**None**: This service is completely self-contained and does not make HTTP calls to other workflow-* services.

**External Dependencies:**
- **PostgreSQL Database**: Primary data persistence for analysis results and audit trails
- **Redis Cache**: Feature caching and idempotency key storage
- **File System**: Temporary audio file storage during processing

### 5.2. Downstream Consumers (Services that Call it)
- **workflow-orchestrator**: Primary consumer that initiates audio analysis via `/analyze/audio` and `/analyze/persistent` endpoints
- **workflow-intelligence**: Consumes analysis results via `/analysis/{id}` endpoints for novelty analysis and insights generation
- **workflow-ml-prediction**: Retrieves optimized feature vectors via `/analyze/features/audio` endpoint for hit prediction models
- **workflow-ml-train**: Accesses comprehensive feature sets for ML model training data

### 5.3. Asynchronous Communication (Events)
**Publishes:**
- **Topic**: `audio.analysis.completed`
- **Payload**: Complete analysis results with metadata
- **Description**: Published to EventStore upon successful analysis completion for audit trail and downstream service notification

**Subscribes:**
- **None**: This service operates in a request-response pattern and does not subscribe to external events

---

## 6. Process Flow Diagram

```mermaid
sequenceDiagram
    participant Orchestrator as workflow-orchestrator
    participant AudioSvc as workflow-audio
    participant DB as PostgreSQL
    participant Cache as Redis
    participant Intelligence as workflow-intelligence
    participant MLPred as workflow-ml-prediction

    Orchestrator->>AudioSvc: POST /analyze/audio (file, metadata)
    activate AudioSvc
    
    AudioSvc->>DB: Check for existing analysis (deduplication)
    DB-->>AudioSvc: Analysis not found
    
    AudioSvc->>Cache: Check idempotency key
    Cache-->>AudioSvc: Key not found
    
    AudioSvc->>AudioSvc: Extract features (7 extractors)
    Note over AudioSvc: 5-15 seconds processing<br/>50+ features extracted
    
    AudioSvc->>DB: Store analysis results + audit trail
    AudioSvc->>Cache: Cache results for performance
    
    AudioSvc-->>Orchestrator: Analysis complete (analysis_id, features)
    deactivate AudioSvc
    
    Note over DB: Event logged to EventStore<br/>for audit trail
    
    Intelligence->>AudioSvc: GET /analysis/{analysis_id}
    AudioSvc->>Cache: Check cache first
    Cache-->>AudioSvc: Cache hit
    AudioSvc-->>Intelligence: Return cached results
    
    MLPred->>AudioSvc: GET /analyze/features/audio?file_id={id}
    AudioSvc->>DB: Get ML-optimized features
    AudioSvc-->>MLPred: Feature vectors for prediction
```

---

## 7. Local Development & Operations

### 7.1. Prerequisites
- **Docker & Docker Compose** (v20.10+)
- **Python 3.9+** (for local development)
- **PostgreSQL 15** (for database)
- **Redis 5.0+** (for caching)

### 7.2. Setup & Installation

**Docker Deployment (Recommended):**
```bash
# Start dependencies
docker-compose -f docker-compose.microservices.yml up -d postgres-audio redis-audio

# Build and start audio service
docker-compose -f docker-compose.microservices.yml up -d workflow-audio

# Verify health
curl http://localhost:8001/health
```

**Local Development:**
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export AUDIO_DATABASE_URL="postgresql://postgres:postgres@localhost:5439/workflow_audio"
export AUDIO_REDIS_URL="redis://localhost:6379/0"

# Create directories
mkdir -p uploads output temp

# Start service
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### 7.3. Running the Service
```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 4
```

### 7.4. Running Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=app tests/

# Run specific test categories
pytest tests/test_extractors.py  # Feature extraction tests
pytest tests/test_persistence.py  # Database persistence tests
pytest tests/test_api.py  # API endpoint tests
```

### 7.5. Environment Configuration
```bash
# Core service settings
AUDIO_DEBUG=false
AUDIO_LOG_LEVEL=INFO
AUDIO_MAX_FILE_SIZE=100MB

# Database configuration
AUDIO_DATABASE_URL=postgresql://postgres:postgres@postgres-audio:5432/workflow_audio
AUDIO_DATABASE_POOL_SIZE=10
AUDIO_DATABASE_MAX_OVERFLOW=20

# Redis configuration
AUDIO_REDIS_URL=redis://redis-audio:6379/0
AUDIO_REDIS_CACHE_TTL=3600
AUDIO_REDIS_IDEMPOTENCY_TTL=7200

# Feature extraction settings
ESSENTIA_SAMPLE_RATE=44100
ESSENTIA_FRAME_SIZE=2048
ESSENTIA_HOP_SIZE=1024
ESSENTIA_MODELS_PATH=/app/models/essentia

# Performance settings
AUDIO_MAX_CONCURRENT_ANALYSES=10
AUDIO_ANALYSIS_QUEUE_SIZE=100
AUDIO_ENABLE_BACKGROUND_PROCESSING=true
```

### 7.6. Monitoring & Troubleshooting
```bash
# Check service logs
docker-compose logs -f workflow-audio

# Monitor database performance
docker-compose exec postgres-audio psql -U postgres -d workflow_audio -c "SELECT COUNT(*) FROM audio_analysis_results;"

# Monitor Redis cache
docker-compose exec redis-audio redis-cli info stats

# Test feature extraction
curl -X POST "http://localhost:8001/analyze/audio" -F "file=@test.mp3"
```

---

## 8. Performance Characteristics

### 8.1. Processing Performance
- **Analysis Time**: 5-15 seconds per song (depends on file length and complexity)
- **Feature Count**: 50+ features extracted per analysis
- **Concurrent Analyses**: Up to 10 simultaneous analyses supported
- **Cache Performance**: <100ms response time for cached results
- **Database Storage**: ~100ms for analysis result persistence

### 8.2. Resource Requirements
- **Memory**: 200MB base + 50MB per concurrent analysis
- **CPU**: Single-threaded Essentia processing (CPU-intensive)
- **Storage**: ~5KB per analysis result in database
- **Network**: Minimal bandwidth requirements

### 8.3. Supported File Formats
- **Audio Formats**: WAV, MP3, FLAC, AAC, M4A, OGG
- **Sample Rates**: 8kHz - 192kHz (auto-resampled to 44.1kHz)
- **File Sizes**: Up to 100MB per file
- **Channels**: Mono and stereo (converted to mono for analysis)

---

## 9. Integration Examples

### 9.1. Orchestrator Integration
```python
# workflow-orchestrator calling workflow-audio
async def process_audio_file(file_path: str, workflow_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://workflow-audio:8001/analyze/file",
            json={"file_path": file_path, "analysis_type": "comprehensive"},
            headers={
                "Workflow-ID": workflow_id,
                "Requested-By": "workflow-orchestrator"
            }
        )
        return response.json()
```

### 9.2. ML Prediction Integration
```python
# workflow-ml-prediction accessing features
def get_audio_features_for_prediction(analysis_id: str):
    response = requests.get(
        f"http://workflow-audio:8001/analyze/features/audio",
        params={"file_id": analysis_id}
    )
    features = response.json()["comprehensive_features"]
    return np.array(features["basic_features"])
```

### 9.3. Intelligence Service Integration
```python
# workflow-intelligence accessing analysis results
async def get_audio_analysis_for_insights(analysis_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://workflow-audio:8001/analysis/{analysis_id}"
        )
        return response.json()
```

---

This comprehensive README provides complete documentation for the workflow-audio microservice, covering all aspects from high-level purpose to detailed implementation specifics, enabling any developer to understand and work with the service effectively. 