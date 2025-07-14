# Workflow Audio Analysis Service

**Version 2.0.0** - Comprehensive Analysis with Persistent Storage

A high-performance microservice for comprehensive audio analysis using Essentia, featuring persistent storage, data lineage tracking, and rich feature extraction for machine learning applications.

## 🚀 **Key Features**

### **Comprehensive Analysis Engine**
- **7 Advanced Extractors**: Audio, Rhythm, Tonal, Timbre, Dynamics, Mood, Genre
- **50+ Audio Features**: From basic properties to advanced spectral characteristics
- **Always Comprehensive**: No more basic/comprehensive confusion - all analysis is comprehensive
- **Rich Feature Set**: Optimized for ML model training and hit prediction

### **Persistent Storage & Data Lineage**
- **Database Persistence**: All analysis results stored in PostgreSQL
- **Redis Caching**: High-performance result caching and session management
- **Idempotency Support**: Duplicate request detection and response caching
- **Audit Trail**: Complete data lineage tracking for compliance and debugging
- **Result Reuse**: Automatic detection and reuse of existing analysis results

### **Production-Ready Architecture**
- **Microservice Design**: Containerized with Docker and health checks
- **API-First**: RESTful API with OpenAPI/Swagger documentation
- **Error Handling**: Comprehensive error handling and logging
- **Performance Optimized**: Efficient processing with result caching
- **Scalable**: Designed for high-throughput production environments

## 📊 **Analysis Capabilities**

### **Audio Extractor**
Foundation Essentia features providing core audio characteristics:
- Duration, sample rate, channels
- Basic spectral and temporal features
- Audio quality metrics

### **Rhythm Extractor**
Advanced rhythm and tempo analysis:
- **Tempo Detection**: BPM estimation with confidence scores
- **Beat Tracking**: Beat positions and strength analysis
- **Rhythm Regularity**: Consistency of rhythmic patterns
- **Onset Rate**: Note onset frequency analysis

### **Tonal Extractor**
Harmonic and key analysis:
- **Key Detection**: Musical key identification (C, D, E, F, G, A, B)
- **Mode Classification**: Major/minor mode detection
- **Key Strength**: Confidence in key detection
- **Chroma Energy**: Harmonic content analysis

### **Timbre Extractor**
Spectral characteristics and texture:
- **Spectral Centroid**: Brightness measure
- **Spectral Rolloff**: High-frequency content
- **Spectral Bandwidth**: Frequency spread
- **Spectral Contrast**: Timbral texture
- **Zero Crossing Rate**: Signal complexity

### **Dynamics Extractor**
Loudness and dynamic range analysis:
- **Dynamic Range**: Difference between loud and quiet sections
- **Loudness Range**: Perceptual loudness variation
- **RMS Energy**: Root mean square energy levels

### **Mood Extractor**
Emotional content classification:
- **Mood Happy**: Happiness/positivity detection
- **Mood Sad**: Sadness/melancholy detection
- **Mood Aggressive**: Aggressiveness/intensity detection
- **Mood Relaxed**: Calmness/relaxation detection

### **Genre Extractor**
Musical style classification:
- **Genre Electronic**: Electronic music detection
- **Genre Rock**: Rock music characteristics
- **Genre Pop**: Pop music features
- **Genre Classical**: Classical music elements
- **Genre Confidence**: Overall classification confidence

## 🏗️ **Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    Workflow Audio Service                   │
├─────────────────────────────────────────────────────────────┤
│  FastAPI Application (Port 8001)                          │
│  ├── Main Endpoints (/analyze/audio, /analyze/features)    │
│  ├── Persistent Endpoints (/analyze/persistent/*)          │
│  ├── Legacy Endpoints (/analyze/basic, /analyze/comprehensive) │
│  └── Data Access (/analysis/{id}, /history/*)              │
├─────────────────────────────────────────────────────────────┤
│  Core Services                                             │
│  ├── PersistentAudioAnalyzer (Main orchestrator)          │
│  ├── ComprehensiveAudioAnalyzer (7 extractors)            │
│  ├── DatabaseService (PostgreSQL operations)              │
│  └── AudioAnalyzer (Basic Essentia features)              │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                │
│  ├── PostgreSQL (Analysis results, metadata, audit)       │
│  ├── Redis (Caching, sessions, idempotency)               │
│  └── File System (Temporary audio processing)             │
└─────────────────────────────────────────────────────────────┘
```

## 🔌 **API Endpoints**

### **Primary Endpoints (Recommended)**

#### `POST /analyze/audio`
**Main comprehensive analysis endpoint** - Always performs comprehensive analysis with all 7 extractors.

```bash
curl -X POST "http://localhost:8001/analyze/audio" \
  -H "X-Session-ID: user123" \
  -H "Workflow-ID: workflow456" \
  -F "file=@song.mp3"
```

**Response:**
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
      "duration": 180.5,
      "tempo": 120.0,
      "key": "C",
      "mode": "major",
      "spectral_centroid": 2500.0,
      "mood_happy": 0.8,
      "genre_pop": 0.7,
      // ... 40+ more features
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

#### `GET /analyze/features/audio?file_id={id}`
**ML feature extraction endpoint** - Returns comprehensive features optimized for machine learning models.

```bash
curl "http://localhost:8001/analyze/features/audio?file_id=audio_analysis_abc123"
```

### **Persistent Storage Endpoints**

#### `POST /analyze/persistent`
**Microservice integration endpoint** - For use by other services with full persistence and idempotency.

#### `POST /analyze/file`
**File path analysis** - For orchestrator and batch processing.

### **Data Access Endpoints**

#### `GET /analysis/{analysis_id}`
Retrieve stored analysis results by ID.

#### `GET /analysis/file/{file_id}`
Get analysis results by file identifier.

#### `GET /history/analyses`
List analysis history with filtering options.

### **Legacy Endpoints (Deprecated)**

- `POST /analyze/basic` - ⚠️ **Deprecated**: Use `/analyze/audio` instead
- `POST /analyze/comprehensive` - ⚠️ **Deprecated**: Use `/analyze/audio` instead

## 🛠️ **Installation & Setup**

### **Docker Deployment (Recommended)**

1. **Start Dependencies:**
```bash
# Start PostgreSQL and Redis
docker-compose -f docker-compose.microservices.yml up -d postgres-audio redis-audio
```

2. **Build and Start Service:**
```bash
# Build and start the audio service
docker-compose -f docker-compose.microservices.yml up -d workflow-audio
```

3. **Verify Health:**
```bash
curl http://localhost:8001/health
```

### **Environment Variables**

```bash
# Service Configuration
DEBUG=true
AUDIO_DEBUG=false
AUDIO_MAX_FILE_SIZE=100MB

# Database Configuration
AUDIO_DATABASE_URL=postgresql://postgres:postgres@postgres-audio:5432/workflow_audio

# Redis Configuration
REDIS_URL=redis://redis-audio:6379/0

# Feature Extraction
ESSENTIA_SAMPLE_RATE=44100
ESSENTIA_FRAME_SIZE=2048
ESSENTIA_HOP_SIZE=1024
```

### **Local Development**

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export AUDIO_DATABASE_URL="postgresql://postgres:postgres@localhost:5435/workflow_audio"
export REDIS_URL="redis://localhost:6386/0"

# Run the service
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

## 📈 **Performance Characteristics**

### **Processing Times**
- **Comprehensive Analysis**: 5-15 seconds per song
- **Feature Extraction**: 50+ features extracted
- **Caching**: Sub-second response for cached results
- **Database Storage**: ~100ms for result persistence

### **Supported Formats**
- **Audio Formats**: WAV, MP3, FLAC, AAC, M4A
- **Sample Rates**: 8kHz - 192kHz (auto-resampled to 44.1kHz)
- **File Sizes**: Up to 100MB per file
- **Channels**: Mono and stereo (converted to mono for analysis)

### **Resource Usage**
- **Memory**: ~200MB base + ~50MB per concurrent analysis
- **CPU**: Single-threaded Essentia processing
- **Storage**: ~5KB per analysis result in database
- **Network**: Minimal - only for API communication

## 🔧 **Integration Examples**

### **Frontend Integration**
```typescript
// TypeScript/React example
const analyzeAudio = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:8001/analyze/audio', {
    method: 'POST',
    headers: {
      'X-Session-ID': sessionId,
      'Workflow-ID': workflowId
    },
    body: formData
  });
  
  return response.json();
};
```

### **ML Pipeline Integration**
```python
# Python ML pipeline example
import requests

def get_audio_features(file_id: str) -> dict:
    response = requests.get(
        f"http://localhost:8001/analyze/features/audio",
        params={"file_id": file_id}
    )
    return response.json()

# Use in ML model
features = get_audio_features("audio_analysis_abc123")
prediction = model.predict([features["comprehensive_features"]])
```

### **Workflow Orchestrator Integration**
```python
# Orchestrator service integration
async def analyze_audio_task(file_path: str, workflow_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://workflow-audio:8001/analyze/file",
            json={"file_path": file_path},
            headers={
                "Workflow-ID": workflow_id,
                "Requested-By": "workflow-orchestrator"
            }
        )
        return response.json()
```

## 📊 **Monitoring & Health**

### **Health Check**
```bash
curl http://localhost:8001/health
```

**Response includes:**
- Service status and version
- Database connectivity
- Redis connectivity
- Feature extraction capabilities
- Supported formats and extractors

### **Metrics & Logging**
- **Structured Logging**: JSON format with correlation IDs
- **Performance Metrics**: Processing times, cache hit rates
- **Error Tracking**: Detailed error messages and stack traces
- **Audit Trail**: Complete request/response logging

### **Database Monitoring**
```sql
-- Check analysis volume
SELECT COUNT(*) as total_analyses, 
       AVG(processing_time_ms) as avg_processing_time
FROM audio_analyses 
WHERE created_at > NOW() - INTERVAL '24 hours';

-- Check feature extraction performance
SELECT analysis_type, 
       COUNT(*) as count,
       AVG(processing_time_ms) as avg_time
FROM audio_analyses 
GROUP BY analysis_type;
```

## 🔄 **Migration from v1.x**

### **Breaking Changes**
1. **Comprehensive Only**: All analysis now uses comprehensive mode
2. **New Response Format**: Enhanced with metadata and audit information
3. **Database Required**: Persistent storage is now mandatory
4. **Deprecated Endpoints**: `/analyze/basic` and `/analyze/comprehensive` are legacy

### **Migration Steps**
1. **Update API Calls**: Change to `/analyze/audio` endpoint
2. **Handle New Response**: Adapt to new response structure
3. **Database Setup**: Ensure PostgreSQL and Redis are available
4. **Environment Variables**: Update configuration for new dependencies

### **Backward Compatibility**
- Legacy endpoints still work but return deprecation warnings
- Response format is enhanced but maintains core structure
- Existing integrations continue to work with warnings

## 🤝 **Contributing**

### **Development Setup**
```bash
# Clone and setup
git clone <repository>
cd workflow-audio

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 app/
black app/
```

### **Testing**
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Performance tests
pytest tests/performance/

# Test with real audio files
python test_service.py
```

## 📝 **Changelog**

### **v2.0.0 (Current)**
- ✅ **Comprehensive Analysis Only**: Eliminated basic/comprehensive confusion
- ✅ **Persistent Storage**: PostgreSQL + Redis for data persistence
- ✅ **7 Advanced Extractors**: Rich feature extraction with 50+ features
- ✅ **Idempotency Support**: Duplicate request handling
- ✅ **Data Lineage**: Complete audit trail and tracking
- ✅ **Performance Optimization**: Caching and efficient processing
- ✅ **Production Ready**: Health checks, monitoring, error handling

### **v1.x (Legacy)**
- Basic Essentia feature extraction
- In-memory processing only
- Basic/comprehensive mode confusion
- No persistence or caching

## 📞 **Support**

### **API Documentation**
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc
- **OpenAPI Spec**: http://localhost:8001/openapi.json

### **Troubleshooting**

**Service Won't Start:**
```bash
# Check dependencies
docker ps | grep -E "(postgres-audio|redis-audio)"

# Check logs
docker logs workflow-workflow-audio-1

# Verify network connectivity
docker network ls | grep workflow
```

**Analysis Fails:**
```bash
# Check file format
file audio.mp3

# Verify file size
ls -lh audio.mp3

# Test with curl
curl -X POST http://localhost:8001/analyze/audio -F "file=@audio.mp3"
```

**Performance Issues:**
```bash
# Check resource usage
docker stats workflow-workflow-audio-1

# Monitor database
docker logs postgres-audio

# Check Redis
docker exec redis-audio redis-cli ping
```

---

**Workflow Audio Analysis Service v2.0.0** - Comprehensive, Persistent, Production-Ready 🎵