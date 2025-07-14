# HSS Workflow Airflow Integration

This directory contains the Apache Airflow replacement for the current HSS Workflow orchestrator system. Airflow provides superior workflow management, monitoring, and scheduling capabilities compared to the custom orchestrator implementation.

## Why Migrate to Airflow?

### Current Orchestrator Limitations
- **Custom Implementation**: Requires maintenance and has limited features
- **Basic Monitoring**: Minimal workflow visualization and debugging tools
- **No Scheduling**: Manual workflow triggering only
- **Limited Scalability**: Custom async workers with basic load balancing
- **Error Handling**: Basic retry logic without advanced failure recovery

### Airflow Advantages
- **Mature Platform**: Battle-tested with extensive community support
- **Rich Web UI**: Beautiful interface for monitoring, debugging, and managing workflows
- **Advanced Scheduling**: Cron-based scheduling, sensors, and complex triggers
- **Scalability**: Kubernetes executor, distributed workers, and auto-scaling
- **Extensive Operators**: Hundreds of pre-built operators and integrations
- **Monitoring & Alerting**: Comprehensive logging, metrics, and notification systems
- **DAG Versioning**: Git-based DAG management with rollback capabilities

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Airflow UI    │    │  Airflow API    │    │   Flower UI     │
│   Port: 8080    │    │   (REST API)    │    │   Port: 5555    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────────────────────┼─────────────────────────────────┐
│                    Airflow Scheduler                               │
│                  (DAG Processing)                                  │
└─────────────────────────────────┼─────────────────────────────────┘
                                 │
┌─────────────────────────────────┼─────────────────────────────────┐
│                   Celery Workers                                   │
│              (Task Execution)                                      │
└─────────────────────────────────┼─────────────────────────────────┘
                                 │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  PostgreSQL     │    │     Redis       │    │  Workflow       │
│  (Metadata)     │    │   (Broker)      │    │  Services       │
│  Port: 5437     │    │   Port: 6384    │    │  (Load Balanced)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Components

### Core Airflow Services
- **Webserver**: Web UI for monitoring and managing workflows (Port 8080)
- **Scheduler**: DAG processing and task scheduling
- **Worker**: Celery workers for distributed task execution
- **Flower**: Celery monitoring UI (Port 5555)
- **PostgreSQL**: Metadata database (Port 5437)
- **Redis**: Message broker for Celery (Port 6384)

### Custom Components
- **Custom Operators**: Workflow-specific operators in `plugins/workflow_operators.py`
- **DAGs**: Workflow definitions in `dags/` directory
- **Shared Storage**: Access to models and data via mounted volumes

## Setup Instructions

### 1. Prerequisites
Ensure your workflow services are running:
```bash
cd workflow-ml-training
docker-compose up -d
```

### 2. Start Airflow
```bash
cd workflow-airflow
docker-compose up -d
```

### 3. Initialize Airflow (First Time Setup)
```bash
# Create admin user
docker exec -it airflow-webserver airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Initialize database
docker exec -it airflow-webserver airflow db init
```

### 4. Access Airflow UI
- **Airflow UI**: http://localhost:8080 (admin/admin)
- **Flower UI**: http://localhost:5555 (Celery monitoring)

## Available DAGs

### 1. Comprehensive Analysis (`comprehensive_analysis`)
Replaces the comprehensive workflow template from the current orchestrator.

**Tasks:**
- Audio Analysis (parallel)
- Content Analysis (parallel)
- Hit Prediction (depends on audio + content)
- AI Insights (depends on all previous)
- Results Aggregation

**Usage:**
```bash
# Trigger via UI or API
curl -X POST "http://localhost:8080/api/v1/dags/comprehensive_analysis/dagRuns" \
  -H "Content-Type: application/json" \
  -d '{
    "conf": {
      "song_id": "example_song_123",
      "song_path": "/app/songs/example.mp3"
    }
  }'
```

### 2. ML Training (`ml_training_smart_ensemble`)
Replaces the ML training orchestration with smart ensemble training.

**Tasks:**
- Validate Training Config
- Process Training Dataset
- Smart Ensemble Training
- Results Aggregation

**Usage:**
```bash
# Trigger smart ensemble training
curl -X POST "http://localhost:8080/api/v1/dags/ml_training_smart_ensemble/dagRuns" \
  -H "Content-Type: application/json" \
  -d '{
    "conf": {
      "dataset_path": "/app/data/training_data/filtered/filtered_audio_only_corrected_20250621_180350.csv",
      "training_id": "airflow_smart_training_1",
      "batch_size": 400,
      "max_concurrent_requests": 50
    }
  }'
```

## Custom Operators

### WorkflowServiceOperator
Base operator for communicating with workflow microservices.
- Handles HTTP requests to services
- Automatic retry logic
- Service discovery via environment variables

### AudioAnalysisOperator
- Service: `workflow-audio`
- Endpoint: `/analyze/comprehensive`
- Features: Essentia audio analysis with 150+ features

### ContentAnalysisOperator  
- Service: `workflow-content`
- Endpoint: `/analyze/lyrics`
- Features: NLP analysis with sentiment and features

### MLTrainingOperator
- Service: `workflow-ml-training`
- Endpoint: `/train/smart_ensemble`
- Features: Smart ensemble training with 5 models

### MLPredictionOperator
- Service: `workflow-ml-prediction`
- Endpoint: `/predict/single`
- Features: Hit prediction with explanations

### IntelligenceOperator
- Service: `workflow-intelligence`
- Endpoint: `/insights/generate`
- Features: AI-powered insights and recommendations

## Migration Guide

### 1. Update Gateway Service
Replace orchestrator URLs in your gateway service:

```python
# OLD
WORKFLOW_ORCHESTRATOR_URL = "http://workflow-orchestrator:8006"

# NEW  
AIRFLOW_API_URL = "http://airflow-webserver:8080/api/v1"
```

### 2. Update Frontend Integration
Replace orchestrator API calls with Airflow API calls:

```javascript
// OLD
const response = await fetch('/api/orchestrator/execute', {
  method: 'POST',
  body: JSON.stringify({ template: 'comprehensive' })
});

// NEW
const response = await fetch('/api/airflow/dags/comprehensive_analysis/dagRuns', {
  method: 'POST',
  body: JSON.stringify({ conf: { song_id: 'example' } })
});
```

### 3. Stop Current Orchestrator
```bash
# Remove orchestrator from docker-compose.yml
docker-compose stop workflow-orchestrator
docker-compose rm workflow-orchestrator
```

### 4. Update Architecture Documentation
Update your architecture diagrams to replace the orchestrator with Airflow components.

## Monitoring & Debugging

### Airflow Web UI Features
- **DAG View**: Visual representation of workflow dependencies
- **Graph View**: Real-time task status and execution flow
- **Gantt Chart**: Task duration and overlap analysis
- **Task Logs**: Detailed logs for each task execution
- **Code View**: DAG source code inspection
- **Admin Panel**: User management and system configuration

### Flower Monitoring
- **Worker Status**: Real-time worker health and task distribution
- **Task Monitoring**: Active, completed, and failed task statistics
- **Broker Monitoring**: Redis queue status and message flow

### Logging
All task logs are stored in `/opt/airflow/logs` and accessible via the web UI.

## Scheduling Examples

### Daily Training
```python
# Schedule daily training at 2 AM
dag = DAG(
    'daily_model_training',
    schedule_interval='0 2 * * *',  # Cron expression
    ...
)
```

### Sensor-Based Triggering
```python
from airflow.sensors.filesystem import FileSensor

# Wait for new dataset file
wait_for_dataset = FileSensor(
    task_id='wait_for_new_dataset',
    filepath='/app/data/training_data/new_dataset.csv',
    dag=dag,
)
```

## Performance Optimization

### Scaling Workers
```bash
# Scale up workers
docker-compose up -d --scale airflow-worker=5
```

### Resource Limits
```yaml
# In docker-compose.yml
airflow-worker:
  deploy:
    resources:
      limits:
        cpus: '2.0'
        memory: 4G
```

## Troubleshooting

### Common Issues

1. **DAG Import Errors**
   - Check logs: `docker logs airflow-scheduler`
   - Validate Python syntax in DAG files
   - Ensure all imports are available

2. **Task Failures**
   - Check task logs in Airflow UI
   - Verify service connectivity
   - Check environment variables

3. **Worker Issues**
   - Monitor Flower UI for worker status
   - Check Redis connectivity
   - Verify resource limits

### Health Checks
```bash
# Check all services
docker-compose ps

# Check Airflow components
curl http://localhost:8080/health
curl http://localhost:5555/api/workers
```

## Security Considerations

### Authentication
- Default admin user: admin/admin (change in production)
- RBAC enabled for role-based access control
- API authentication via basic auth or JWT

### Network Security
- Services communicate within Docker network
- External access only through defined ports
- Environment variables for sensitive configuration

## Next Steps

1. **Deploy Airflow**: Follow setup instructions
2. **Test DAGs**: Run sample workflows to verify integration
3. **Update Services**: Modify gateway and frontend to use Airflow APIs
4. **Monitor Performance**: Use Airflow UI and Flower for monitoring
5. **Add Scheduling**: Configure cron-based scheduling for regular workflows
6. **Extend DAGs**: Add more complex workflows as needed

## Support

For issues or questions:
1. Check Airflow documentation: https://airflow.apache.org/docs/
2. Review DAG logs in Airflow UI
3. Monitor Flower for worker issues
4. Check service connectivity and environment variables 