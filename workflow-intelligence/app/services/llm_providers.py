"""
LLM Provider Abstraction Layer for workflow-intelligence

This module provides a flexible abstraction for different LLM providers,
allowing easy switching between OpenAI, Gemini, Anthropic, Ollama, and other providers.
"""

import logging
import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from ..config.settings import settings

logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, **config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cost_tracking = {"total_tokens": 0, "total_cost": 0.0}
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return {
            "provider": self.__class__.__name__,
            "available": self.is_available(),
            "config_keys": list(self.config.keys()),
            "cost_tracking": self.cost_tracking
        }
    
    def track_usage(self, tokens: int, cost: float = 0.0):
        """Track token usage and cost."""
        self.cost_tracking["total_tokens"] += tokens
        self.cost_tracking["total_cost"] += cost

class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider using OpenAI SDK."""
    
    def __init__(self, **config):
        super().__init__(**config)
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize OpenAI client."""
        try:
            from openai import AsyncOpenAI
            
            api_key = self.config.get('api_key') or self._get_api_key()
            if not api_key:
                self.logger.warning("OpenAI API key not found.")
                return
            
            self.client = AsyncOpenAI(api_key=api_key)
            self.logger.info("OpenAI provider initialized successfully.")
            
        except ImportError as e:
            self.logger.error(f"Failed to import openai: {e}")
            self.logger.info("Install with: pip install openai")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI provider: {e}")
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from various sources."""
        # Try environment variable first
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            return api_key
        
        # Try settings
        if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
            return settings.OPENAI_API_KEY
        
        return None
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI."""
        if not self.client:
            raise RuntimeError("OpenAI provider not initialized")
        
        try:
            # Prepare parameters
            model = kwargs.get('model_name', self.config.get('model_name', 'gpt-4'))
            temperature = kwargs.get('temperature', self.config.get('temperature', 0.7))
            max_tokens = kwargs.get('max_tokens', self.config.get('max_tokens', 2000))
            
            # Make API call
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Track usage
            if hasattr(response, 'usage'):
                total_tokens = response.usage.total_tokens
                # Estimate cost (rough GPT-4 pricing)
                cost = (total_tokens / 1000) * 0.03
                self.track_usage(total_tokens, cost)
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error generating with OpenAI: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return self.client is not None

class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider."""
    
    def __init__(self, **config):
        super().__init__(**config)
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Gemini LLM."""
        try:
            import google.generativeai as genai
            
            api_key = self.config.get('api_key') or self._get_api_key()
            if not api_key:
                self.logger.warning("Gemini API key not found.")
                return
            
            # Configure the API key
            genai.configure(api_key=api_key)
            
            # Default configuration
            default_config = {
                "model_name": "gemini-1.5-flash",
                "temperature": 0.7,
                "max_output_tokens": 2000,
                "top_p": 0.9,
                "top_k": 40
            }
            
            # Merge with user config
            model_config = {**default_config, **self.config}
            model_name = model_config.pop('model_name')
            
            # Create generation config
            generation_config = genai.types.GenerationConfig(
                temperature=model_config.get('temperature', 0.7),
                max_output_tokens=model_config.get('max_output_tokens', 2000),
                top_p=model_config.get('top_p', 0.9),
                top_k=model_config.get('top_k', 40)
            )
            
            # Initialize the model
            self.model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config
            )
            
            self.logger.info(f"Gemini provider initialized with model: {model_name}")
            
        except ImportError as e:
            self.logger.error(f"Failed to import google.generativeai: {e}")
            self.logger.info("Install with: pip install google-generativeai")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini provider: {e}")
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from various sources."""
        # Try environment variable first
        api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        if api_key:
            return api_key
        
        # Try settings
        if hasattr(settings, 'GEMINI_API_KEY') and settings.GEMINI_API_KEY:
            return settings.GEMINI_API_KEY
        
        return None
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Gemini."""
        if not self.model:
            raise RuntimeError("Gemini provider not initialized")
        
        try:
            # Generate content
            response = await self.model.generate_content_async(prompt)
            
            # Track usage (rough estimation)
            prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate
            response_tokens = len(response.text.split()) * 1.3 if hasattr(response, 'text') else 0
            total_tokens = int(prompt_tokens + response_tokens)
            # Gemini is relatively cheap
            cost = (total_tokens / 1000) * 0.001
            self.track_usage(total_tokens, cost)
            
            # Extract text from response
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    if len(parts) > 0 and hasattr(parts[0], 'text'):
                        return parts[0].text
            
            # Fallback
            return str(response)
            
        except Exception as e:
            self.logger.error(f"Error generating with Gemini: {e}")
            # Try sync version as fallback
            try:
                response = self.model.generate_content(prompt)
                return response.text if hasattr(response, 'text') else str(response)
            except Exception as sync_e:
                self.logger.error(f"Sync fallback also failed: {sync_e}")
                raise e
    
    def is_available(self) -> bool:
        """Check if Gemini is available."""
        if not self.model:
            return False
        
        # Try a simple test call
        try:
            test_response = self.model.generate_content("Hello")
            return hasattr(test_response, 'text') or hasattr(test_response, 'candidates')
        except Exception as e:
            self.logger.debug(f"Gemini availability test failed: {e}")
            return False

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider."""
    
    def __init__(self, **config):
        super().__init__(**config)
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Anthropic client."""
        try:
            from anthropic import AsyncAnthropic
            
            api_key = self.config.get('api_key') or self._get_api_key()
            if not api_key:
                self.logger.warning("Anthropic API key not found.")
                return
            
            self.client = AsyncAnthropic(api_key=api_key)
            self.logger.info("Anthropic provider initialized successfully.")
            
        except ImportError as e:
            self.logger.error(f"Failed to import anthropic: {e}")
            self.logger.info("Install with: pip install anthropic")
        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic provider: {e}")
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from various sources."""
        # Try environment variable first
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if api_key:
            return api_key
        
        # Try settings
        if hasattr(settings, 'ANTHROPIC_API_KEY') and settings.ANTHROPIC_API_KEY:
            return settings.ANTHROPIC_API_KEY
        
        return None
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic."""
        if not self.client:
            raise RuntimeError("Anthropic provider not initialized")
        
        try:
            # Prepare parameters
            model = kwargs.get('model_name', self.config.get('model_name', 'claude-3-sonnet-20240229'))
            temperature = kwargs.get('temperature', self.config.get('temperature', 0.7))
            max_tokens = kwargs.get('max_tokens', self.config.get('max_tokens', 2000))
            
            # Make API call
            response = await self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Track usage
            if hasattr(response, 'usage'):
                total_tokens = response.usage.input_tokens + response.usage.output_tokens
                # Estimate cost (rough Claude pricing)
                cost = (total_tokens / 1000) * 0.015
                self.track_usage(total_tokens, cost)
            
            return response.content[0].text
            
        except Exception as e:
            self.logger.error(f"Error generating with Anthropic: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Anthropic is available."""
        return self.client is not None

class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""
    
    def __init__(self, **config):
        super().__init__(**config)
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.model_name = config.get('model', 'llama2')
        self._initialize()
    
    def _initialize(self):
        """Initialize Ollama client."""
        try:
            import httpx
            self.client = httpx.AsyncClient()
            self.logger.info(f"Ollama provider initialized with model: {self.model_name}")
        except ImportError as e:
            self.logger.error(f"Failed to import httpx: {e}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama provider: {e}")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama."""
        if not hasattr(self, 'client'):
            raise RuntimeError("Ollama provider not initialized")
        
        try:
            temperature = kwargs.get('temperature', self.config.get('temperature', 0.7))
            
            # Prepare request
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False
            }
            
            # Make API call
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=120.0
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Track usage (rough estimation)
            prompt_tokens = len(prompt.split()) * 1.3
            response_tokens = len(result.get('response', '').split()) * 1.3
            total_tokens = int(prompt_tokens + response_tokens)
            # Ollama is free
            self.track_usage(total_tokens, 0.0)
            
            return result.get('response', '')
            
        except Exception as e:
            self.logger.error(f"Error generating with Ollama: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            import httpx
            import asyncio
            
            async def test_connection():
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.base_url}/api/tags", timeout=5.0)
                    return response.status_code == 200
            
            return asyncio.run(test_connection())
        except Exception:
            return False

class HuggingFaceProvider(BaseLLMProvider):
    """Hugging Face transformers provider for local models."""
    
    def __init__(self, **config):
        super().__init__(**config)
        self.pipeline = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Hugging Face model."""
        try:
            from transformers import pipeline
            
            model_name = self.config.get('model_name', 'microsoft/DialoGPT-medium')
            task = self.config.get('task', 'text-generation')
            
            self.pipeline = pipeline(
                task,
                model=model_name,
                **{k: v for k, v in self.config.items() if k not in ['model_name', 'task']}
            )
            self.logger.info(f"HuggingFace provider initialized with model: {model_name}")
            
        except ImportError as e:
            self.logger.error(f"Failed to import transformers: {e}")
            self.logger.info("Install with: pip install transformers torch")
        except Exception as e:
            self.logger.error(f"Failed to initialize HuggingFace provider: {e}")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Hugging Face."""
        if not self.pipeline:
            raise RuntimeError("HuggingFace provider not initialized")
        
        try:
            max_length = kwargs.get('max_length', 200)
            
            # Run in executor to avoid blocking
            import asyncio
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None, 
                lambda: self.pipeline(prompt, max_length=max_length, **kwargs)
            )
            
            # Track usage (rough estimation)
            prompt_tokens = len(prompt.split()) * 1.3
            if isinstance(response, list) and len(response) > 0:
                generated_text = response[0].get('generated_text', str(response[0]))
                response_tokens = len(generated_text.split()) * 1.3
                self.track_usage(int(prompt_tokens + response_tokens), 0.0)  # HF is free
                return generated_text
            
            return str(response)
            
        except Exception as e:
            self.logger.error(f"Error generating with HuggingFace: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if HuggingFace is available."""
        return self.pipeline is not None

class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    _providers = {
        'openai': OpenAIProvider,
        'gemini': GeminiProvider,
        'anthropic': AnthropicProvider,
        'ollama': OllamaProvider,
        'huggingface': HuggingFaceProvider,
    }
    
    @classmethod
    def register_provider(cls, name: str, provider_class):
        """Register a new provider."""
        cls._providers[name] = provider_class
    
    @classmethod
    def create_provider(cls, provider_type: str, **config) -> BaseLLMProvider:
        """Create a provider instance."""
        if provider_type not in cls._providers:
            raise ValueError(f"Unknown provider: {provider_type}. Available: {list(cls._providers.keys())}")
        
        # Use settings configuration if not provided
        if not config:
            config = settings.get_llm_config(provider_type)
        
        return cls._providers[provider_type](**config)
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List available providers."""
        return list(cls._providers.keys())
    
    @classmethod
    def auto_detect_provider(cls, **config) -> Optional[BaseLLMProvider]:
        """Auto-detect and return the first available provider."""
        # Priority order for auto-detection based on quality and availability
        priority_order = ['openai', 'gemini', 'anthropic', 'ollama', 'huggingface']
        
        for provider_type in priority_order:
            try:
                provider_config = config or settings.get_llm_config(provider_type)
                provider = cls.create_provider(provider_type, **provider_config)
                if provider.is_available():
                    logger.info(f"Auto-detected provider: {provider_type}")
                    return provider
            except Exception as e:
                logger.debug(f"Provider {provider_type} not available: {e}")
        
        logger.warning("No LLM providers available")
        return None
    
    @classmethod
    def get_provider_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all providers."""
        info = {}
        for provider_name in cls._providers:
            try:
                config = settings.get_llm_config(provider_name)
                provider = cls.create_provider(provider_name, **config)
                info[provider_name] = provider.get_info()
            except Exception as e:
                info[provider_name] = {
                    "provider": provider_name,
                    "available": False,
                    "error": str(e)
                }
        return info

def create_llm_from_config(config: Dict[str, Any]) -> Optional[BaseLLMProvider]:
    """Create LLM provider from configuration dictionary."""
    provider_type = config.get('provider', 'auto')
    provider_config = config.get('config', {})
    
    try:
        if provider_type == 'auto':
            return LLMProviderFactory.auto_detect_provider(**provider_config)
        else:
            return LLMProviderFactory.create_provider(provider_type, **provider_config)
    except Exception as e:
        logger.error(f"Failed to create LLM provider: {e}")
        return None 