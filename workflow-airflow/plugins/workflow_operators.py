"""
Custom Airflow operators for HSS Workflow system.
These operators replace the current orchestrator's task execution logic.
"""

import json
import logging
import os
from datetime import timedelta
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
from airflow.models import BaseOperator
from airflow.utils.context import Context
from airflow.utils.decorators import apply_defaults
from airflow.exceptions import AirflowException

logger = logging.getLogger(__name__)


class WorkflowServiceOperator(BaseOperator):
    """
    Base operator for communicating with workflow microservices.
    Replaces the HTTP client logic from the current orchestrator.
    """
    
    template_fields = ['endpoint', 'data']
    
    @apply_defaults
    def __init__(
        self,
        service_name: str,
        endpoint: str,
        method: str = 'POST',
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 300,
        retries: int = 3,
        retry_delay: timedelta = timedelta(seconds=5),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.service_name = service_name
        self.endpoint = endpoint
        self.method = method.upper()
        self.data = data or {}
        self.params = params or {}
        self.timeout = timeout
        self.service_url = self._get_service_url()
        
    def _get_service_url(self) -> str:
        """Get service URL from environment variables."""
        service_env_map = {
            'workflow-audio': 'WORKFLOW_AUDIO_URL',
            'workflow-content': 'WORKFLOW_CONTENT_URL',
            'workflow-ml-training': 'WORKFLOW_ML_TRAINING_URL',
            'workflow-ml-prediction': 'WORKFLOW_ML_PREDICTION_URL',
            'workflow-intelligence': 'WORKFLOW_INTELLIGENCE_URL',
            'workflow-storage': 'WORKFLOW_STORAGE_URL',
        }
        
        env_var = service_env_map.get(self.service_name)
        if not env_var:
            raise ValueError(f"Unknown service: {self.service_name}")
            
        url = os.getenv(env_var)
        if not url:
            raise ValueError(f"Environment variable {env_var} not set")
            
        return url
    
    def execute(self, context: Context) -> Dict[str, Any]:
        """Execute the service call."""
        full_url = f"{self.service_url}{self.endpoint}"
        
        logger.info(f"Calling {self.method} {full_url}")
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                if self.method == 'GET':
                    response = client.get(full_url, params=self.params)
                elif self.method == 'POST':
                    response = client.post(full_url, json=self.data, params=self.params)
                elif self.method == 'PUT':
                    response = client.put(full_url, json=self.data, params=self.params)
                else:
                    raise ValueError(f"Unsupported HTTP method: {self.method}")
                
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"Service call successful: {self.service_name}{self.endpoint}")
                return result
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise AirflowException(f"Service call failed: {e}")
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise AirflowException(f"Service call failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise AirflowException(f"Service call failed: {e}")


class AudioAnalysisOperator(WorkflowServiceOperator):
    """
    Operator for audio analysis tasks.
    Replaces the audio analysis logic from the current orchestrator.
    """
    
    @apply_defaults
    def __init__(
        self,
        song_id: Optional[str] = None,
        song_path: Optional[str] = None,
        include_advanced_features: bool = True,
        include_spectral: bool = True,
        **kwargs
    ):
        # Set default service configuration
        kwargs.setdefault('service_name', 'workflow-audio')
        kwargs.setdefault('endpoint', '/analyze/comprehensive')
        kwargs.setdefault('timeout', 180)  # 3 minutes for audio processing
        
        super().__init__(**kwargs)
        
        self.song_id = song_id
        self.song_path = song_path
        self.include_advanced_features = include_advanced_features
        self.include_spectral = include_spectral
    
    def execute(self, context: Context) -> Dict[str, Any]:
        """Execute audio analysis."""
        # Prepare data payload
        self.data = {
            'include_advanced_features': self.include_advanced_features,
            'include_spectral': self.include_spectral
        }
        
        if self.song_id:
            self.data['song_id'] = self.song_id
        if self.song_path:
            self.data['song_path'] = self.song_path
            
        # Add context data if available
        if 'song_info' in context['params']:
            self.data.update(context['params']['song_info'])
        
        result = super().execute(context)
        
        # Store audio features in XCom for downstream tasks
        if 'audio_features' in result:
            context['task_instance'].xcom_push(
                key='audio_features',
                value=result['audio_features']
            )
        
        return result


class ContentAnalysisOperator(WorkflowServiceOperator):
    """
    Operator for content/lyrics analysis tasks.
    Replaces the content analysis logic from the current orchestrator.
    """
    
    @apply_defaults
    def __init__(
        self,
        lyrics_text: Optional[str] = None,
        lyrics_path: Optional[str] = None,
        include_sentiment: bool = True,
        include_features: bool = True,
        **kwargs
    ):
        # Set default service configuration
        kwargs.setdefault('service_name', 'workflow-content')
        kwargs.setdefault('endpoint', '/analyze/lyrics')
        kwargs.setdefault('timeout', 120)  # 2 minutes for content processing
        
        super().__init__(**kwargs)
        
        self.lyrics_text = lyrics_text
        self.lyrics_path = lyrics_path
        self.include_sentiment = include_sentiment
        self.include_features = include_features
    
    def execute(self, context: Context) -> Dict[str, Any]:
        """Execute content analysis."""
        # Prepare data payload
        self.data = {
            'include_sentiment': self.include_sentiment,
            'include_features': self.include_features
        }
        
        if self.lyrics_text:
            self.data['lyrics'] = self.lyrics_text
        if self.lyrics_path:
            self.data['lyrics_path'] = self.lyrics_path
            
        # Add context data if available
        if 'song_info' in context['params']:
            self.data.update(context['params']['song_info'])
        
        result = super().execute(context)
        
        # Store content features in XCom for downstream tasks
        if 'content_features' in result:
            context['task_instance'].xcom_push(
                key='content_features',
                value=result['content_features']
            )
        
        return result


class MLTrainingOperator(WorkflowServiceOperator):
    """
    Operator for ML training tasks.
    Replaces the ML training orchestration from the current system.
    """
    
    @apply_defaults
    def __init__(
        self,
        training_strategy: str = 'smart_ensemble',
        dataset_path: Optional[str] = None,
        training_id: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Set default service configuration
        kwargs.setdefault('service_name', 'workflow-ml-training')
        kwargs.setdefault('endpoint', '/train/smart_ensemble')
        kwargs.setdefault('timeout', 7200)  # 2 hours for training
        
        super().__init__(**kwargs)
        
        self.training_strategy = training_strategy
        self.dataset_path = dataset_path
        self.training_id = training_id
        self.model_config = model_config or {}
    
    def execute(self, context: Context) -> Dict[str, Any]:
        """Execute ML training."""
        # Get all parameters from context (for simplified approach)
        params = context.get('params', {})
        
        # Prepare data payload - start with basic required fields
        self.data = {
            'strategy': self.training_strategy,
            'training_id': self.training_id or f"airflow_training_{context['ts_nodash']}",
        }
        
        # Add dataset path if provided
        if self.dataset_path:
            self.data['dataset_path'] = self.dataset_path
            
        # Add model_config if it's a dictionary
        if self.model_config and isinstance(self.model_config, dict):
            self.data.update(self.model_config)
        
        # Add all parameters from context (this is the key for simplified approach)
        if params:
            self.data.update(params)
            
        # Legacy support: Add context data if available
        if 'training_config' in context['params']:
            self.data.update(context['params']['training_config'])
        
        result = super().execute(context)
        
        # Store training results in XCom
        if 'training_id' in result:
            context['task_instance'].xcom_push(
                key='training_id',
                value=result['training_id']
            )
        if 'model_paths' in result:
            context['task_instance'].xcom_push(
                key='model_paths',
                value=result['model_paths']
            )
        
        return result


class MLPredictionOperator(WorkflowServiceOperator):
    """
    Operator for ML prediction tasks.
    """
    
    @apply_defaults
    def __init__(
        self,
        model_type: str = 'hit_prediction',
        include_explanation: bool = True,
        **kwargs
    ):
        # Set default service configuration
        kwargs.setdefault('service_name', 'workflow-ml-prediction')
        kwargs.setdefault('endpoint', '/predict/single')
        kwargs.setdefault('timeout', 60)  # 1 minute for prediction
        
        super().__init__(**kwargs)
        
        self.model_type = model_type
        self.include_explanation = include_explanation
    
    def execute(self, context: Context) -> Dict[str, Any]:
        """Execute ML prediction."""
        # Get features from upstream tasks
        audio_features = context['task_instance'].xcom_pull(key='audio_features')
        content_features = context['task_instance'].xcom_pull(key='content_features')
        
        # Prepare data payload
        self.data = {
            'model_type': self.model_type,
            'include_explanation': self.include_explanation
        }
        
        if audio_features:
            self.data['audio_features'] = audio_features
        if content_features:
            self.data['content_features'] = content_features
            
        # Add context data if available
        if 'prediction_config' in context['params']:
            self.data.update(context['params']['prediction_config'])
        
        result = super().execute(context)
        
        # Store prediction results in XCom
        if 'prediction' in result:
            context['task_instance'].xcom_push(
                key='prediction',
                value=result['prediction']
            )
        
        return result


class IntelligenceOperator(WorkflowServiceOperator):
    """
    Operator for AI insights generation.
    """
    
    @apply_defaults
    def __init__(
        self,
        analysis_types: List[str] = None,
        agent_type: str = 'comprehensive',
        **kwargs
    ):
        # Set default service configuration
        kwargs.setdefault('service_name', 'workflow-intelligence')
        kwargs.setdefault('endpoint', '/insights/generate')
        kwargs.setdefault('timeout', 120)  # 2 minutes for AI insights
        
        super().__init__(**kwargs)
        
        self.analysis_types = analysis_types or ['musical_meaning', 'hit_comparison', 'strategic_insights']
        self.agent_type = agent_type
    
    def execute(self, context: Context) -> Dict[str, Any]:
        """Execute AI insights generation."""
        # Get data from upstream tasks
        audio_features = context['task_instance'].xcom_pull(key='audio_features')
        content_features = context['task_instance'].xcom_pull(key='content_features')
        prediction = context['task_instance'].xcom_pull(key='prediction')
        
        # Prepare data payload
        self.data = {
            'analysis_types': self.analysis_types,
            'agent_type': self.agent_type
        }
        
        if audio_features:
            self.data['audio_features'] = audio_features
        if content_features:
            self.data['content_features'] = content_features
        if prediction:
            self.data['prediction'] = prediction
            
        # Add context data if available
        if 'intelligence_config' in context['params']:
            self.data.update(context['params']['intelligence_config'])
        
        result = super().execute(context)
        
        # Store insights in XCom
        if 'insights' in result:
            context['task_instance'].xcom_push(
                key='insights',
                value=result['insights']
            )
        
        return result


class DatasetProcessingOperator(BaseOperator):
    """
    Operator for processing training datasets.
    Handles CSV processing and song preparation for batch training.
    """
    
    template_fields = ['dataset_path']
    
    @apply_defaults
    def __init__(
        self,
        dataset_path: str,
        output_key: str = 'song_list',
        batch_size: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset_path = dataset_path
        self.output_key = output_key
        self.batch_size = batch_size
    
    def execute(self, context: Context) -> List[Dict[str, Any]]:
        """Process dataset and return song list."""
        try:
            # Load dataset
            df = pd.read_csv(self.dataset_path)
            logger.info(f"Loaded dataset with {len(df)} songs from {self.dataset_path}")
            
            # Convert to list of dictionaries
            song_list = df.to_dict('records')
            
            # Apply batch size if specified - ensure it's an integer
            if self.batch_size:
                # Convert to int if it's a string (from DAG parameters)
                batch_size = int(self.batch_size) if isinstance(self.batch_size, str) else self.batch_size
                song_list = song_list[:batch_size]
                logger.info(f"Limited to {batch_size} songs for processing")
            
            # Store in XCom
            context['task_instance'].xcom_push(
                key=self.output_key,
                value=song_list
            )
            
            logger.info(f"Processed {len(song_list)} songs for workflow")
            return song_list
            
        except Exception as e:
            logger.error(f"Error processing dataset {self.dataset_path}: {e}")
            raise AirflowException(f"Dataset processing failed: {e}")


class WorkflowResultsAggregatorOperator(BaseOperator):
    """
    Operator for aggregating results from multiple workflow tasks.
    Replaces the output compilation logic from the current orchestrator.
    """
    
    @apply_defaults
    def __init__(
        self,
        result_keys: List[str],
        output_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.result_keys = result_keys
        self.output_path = output_path
    
    def execute(self, context: Context) -> Dict[str, Any]:
        """Aggregate workflow results."""
        aggregated_results = {
            'workflow_id': context['dag_run'].run_id,
            'execution_date': context['execution_date'].isoformat(),
            'results': {}
        }
        
        # Collect results from XCom
        for key in self.result_keys:
            result = context['task_instance'].xcom_pull(key=key)
            if result:
                aggregated_results['results'][key] = result
        
        # Save to file if output path specified
        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, 'w') as f:
                json.dump(aggregated_results, f, indent=2)
            logger.info(f"Results saved to {self.output_path}")
        
        # Store in XCom
        context['task_instance'].xcom_push(
            key='final_results',
            value=aggregated_results
        )
        
        logger.info(f"Aggregated {len(self.result_keys)} workflow results")
        return aggregated_results 