"""
Local model provider implementation for the AI Agent System.

This module provides the LocalModelProvider class that integrates with local
language models (like Ollama, Hugging Face Transformers, etc.) to provide
language model capabilities through the ModelInterface.
"""

import os
import logging
import asyncio
import subprocess
import json
import time
from typing import Any, Dict, Optional, AsyncGenerator
from pathlib import Path

from ..models.model_provider import ModelProvider
from ..models.data_models import ModelInfo, ModelType
from ..exceptions import ModelError, ConfigurationError


logger = logging.getLogger(__name__)


class LocalModelProvider(ModelProvider):
    """
    Local model provider implementation.
    
    This class provides integration with local language models, supporting
    various backends like Ollama, Hugging Face Transformers, or custom implementations.
    """
    
    def __init__(
        self, 
        model_name: str = "llama2", 
        backend: str = "ollama",
        model_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the local model provider.
        
        Args:
            model_name: Name of the local model to use
            backend: Backend to use ('ollama', 'transformers', 'custom')
            model_path: Path to the model files (for custom backend)
            **kwargs: Additional configuration parameters
        """
        # Set up model info
        model_info = ModelInfo(
            name=model_name,
            model_type=ModelType.LOCAL,
            description=f"Local {model_name} model via {backend}",
            max_tokens=kwargs.get("max_tokens", 4096),
            supports_streaming=backend == "ollama",  # Ollama supports streaming
            api_endpoint=kwargs.get("api_endpoint", "http://localhost:11434"),
            version="1.0",
            enabled=True
        )
        
        # Store local model-specific configuration
        self.model_name = model_name
        self.backend = backend.lower()
        self.model_path = model_path
        self.api_endpoint = kwargs.get("api_endpoint", "http://localhost:11434")
        self.timeout = kwargs.get("timeout", 60.0)  # Local models may be slower
        
        # Backend-specific initialization
        self.client = None
        self.tokenizer = None
        self.model = None
        
        # Initialize the base provider
        super().__init__(model_info, **kwargs)
    
    def _initialize(self) -> None:
        """
        Initialize the local model provider.
        
        Sets up the appropriate backend and validates model availability.
        
        Raises:
            ConfigurationError: If backend configuration is invalid
            ModelError: If initialization fails
        """
        try:
            if self.backend == "ollama":
                self._initialize_ollama()
            elif self.backend == "transformers":
                self._initialize_transformers()
            elif self.backend == "custom":
                self._initialize_custom()
            else:
                raise ConfigurationError(f"Unsupported backend: {self.backend}")
            
            self._is_initialized = True
            logger.info(f"Local model provider initialized successfully with {self.backend} backend")
            
        except Exception as e:
            logger.error(f"Failed to initialize local model provider: {e}")
            raise ModelError(f"Local model provider initialization failed: {e}")
    
    def _initialize_ollama(self) -> None:
        """
        Initialize Ollama backend.
        
        Raises:
            ModelError: If Ollama is not available or model is not found
        """
        try:
            # Check if Ollama is running
            import requests
            
            # Test connection to Ollama
            response = requests.get(f"{self.api_endpoint}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ModelError("Ollama server is not responding")
            
            # Check if the model is available
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            if self.model_name not in model_names:
                logger.warning(f"Model {self.model_name} not found in Ollama. Available models: {model_names}")
                # Try to pull the model
                self._pull_ollama_model()
            
            logger.info(f"Ollama backend initialized with model {self.model_name}")
            
        except ImportError:
            raise ModelError("requests library not installed. Install with: pip install requests")
        except Exception as e:
            raise ModelError(f"Ollama initialization failed: {e}")
    
    def _pull_ollama_model(self) -> None:
        """
        Pull a model in Ollama.
        
        Raises:
            ModelError: If model pull fails
        """
        try:
            import requests
            
            logger.info(f"Pulling model {self.model_name} in Ollama...")
            
            response = requests.post(
                f"{self.api_endpoint}/api/pull",
                json={"name": self.model_name},
                timeout=300  # Model pulling can take a while
            )
            
            if response.status_code != 200:
                raise ModelError(f"Failed to pull model {self.model_name}")
            
            logger.info(f"Successfully pulled model {self.model_name}")
            
        except Exception as e:
            raise ModelError(f"Failed to pull model {self.model_name}: {e}")
    
    def _initialize_transformers(self) -> None:
        """
        Initialize Hugging Face Transformers backend.
        
        Raises:
            ModelError: If transformers is not available or model loading fails
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # Load tokenizer and model
            logger.info(f"Loading model {self.model_name} with transformers...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Transformers backend initialized with model {self.model_name}")
            
        except ImportError:
            raise ModelError(
                "transformers library not installed. Install with: pip install transformers torch"
            )
        except Exception as e:
            raise ModelError(f"Transformers initialization failed: {e}")
    
    def _initialize_custom(self) -> None:
        """
        Initialize custom backend.
        
        Raises:
            ConfigurationError: If model path is not provided
            ModelError: If custom model loading fails
        """
        if not self.model_path:
            raise ConfigurationError("model_path is required for custom backend")
        
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise ConfigurationError(f"Model path does not exist: {self.model_path}")
        
        # Custom initialization logic would go here
        # This is a placeholder for custom model implementations
        logger.info(f"Custom backend initialized with model at {self.model_path}")
    
    def _make_request(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Make a request to the local model.
        
        Args:
            prompt: The input prompt
            context: Optional context information
            
        Returns:
            The model's response
            
        Raises:
            ModelError: If the request fails
        """
        try:
            if self.backend == "ollama":
                return self._make_ollama_request(prompt, context)
            elif self.backend == "transformers":
                return self._make_transformers_request(prompt, context)
            elif self.backend == "custom":
                return self._make_custom_request(prompt, context)
            else:
                raise ModelError(f"Unsupported backend: {self.backend}")
                
        except Exception as e:
            logger.error(f"Local model request failed: {e}")
            raise ModelError(f"Local model request failed: {e}")
    
    def _make_ollama_request(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Make a request to Ollama.
        
        Args:
            prompt: The input prompt
            context: Optional context information
            
        Returns:
            The model's response
        """
        import requests
        
        # Prepare the request
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.parameters.get("temperature", 0.7),
                "top_p": self.parameters.get("top_p", 1.0),
                "num_predict": self.parameters.get("max_tokens", 1000),
            }
        }
        
        # Add context if provided
        if context and "system_message" in context:
            data["system"] = context["system_message"]
        
        # Make the request
        response = requests.post(
            f"{self.api_endpoint}/api/generate",
            json=data,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise ModelError(f"Ollama request failed with status {response.status_code}")
        
        result = response.json()
        return result.get("response", "").strip()
    
    def _make_transformers_request(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Make a request using Hugging Face Transformers.
        
        Args:
            prompt: The input prompt
            context: Optional context information
            
        Returns:
            The model's response
        """
        import torch
        
        # Prepare the input
        full_prompt = prompt
        if context and "system_message" in context:
            full_prompt = f"{context['system_message']}\n\n{prompt}"
        
        # Tokenize input
        inputs = self.tokenizer.encode(full_prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.parameters.get("max_tokens", 1000),
                temperature=self.parameters.get("temperature", 0.7),
                top_p=self.parameters.get("top_p", 1.0),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the response
        if response.startswith(full_prompt):
            response = response[len(full_prompt):].strip()
        
        return response
    
    def _make_custom_request(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Make a request using custom backend.
        
        Args:
            prompt: The input prompt
            context: Optional context information
            
        Returns:
            The model's response
        """
        # Placeholder for custom implementation
        # This would be implemented based on the specific custom model
        return f"Custom model response to: {prompt}"
    
    async def generate_response_stream(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the local model.
        
        Args:
            prompt: The input prompt for the model
            context: Optional context information to include
            
        Yields:
            Chunks of the response as they are generated
            
        Raises:
            ModelError: If streaming response generation fails
        """
        if not self.supports_streaming():
            # Fallback to non-streaming
            response = self.generate_response(prompt, context)
            yield response
            return
        
        if self.backend == "ollama":
            async for chunk in self._stream_ollama_request(prompt, context):
                yield chunk
        else:
            # For other backends, fall back to non-streaming
            response = self.generate_response(prompt, context)
            yield response
    
    async def _stream_ollama_request(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response from Ollama.
        
        Args:
            prompt: The input prompt
            context: Optional context information
            
        Yields:
            Chunks of the response
        """
        import aiohttp
        
        # Prepare the request
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.parameters.get("temperature", 0.7),
                "top_p": self.parameters.get("top_p", 1.0),
                "num_predict": self.parameters.get("max_tokens", 1000),
            }
        }
        
        # Add context if provided
        if context and "system_message" in context:
            data["system"] = context["system_message"]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_endpoint}/api/generate",
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status != 200:
                        raise ModelError(f"Ollama streaming request failed with status {response.status}")
                    
                    async for line in response.content:
                        if line:
                            try:
                                chunk_data = json.loads(line.decode('utf-8'))
                                if "response" in chunk_data:
                                    yield chunk_data["response"]
                                if chunk_data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            logger.error(f"Ollama streaming request failed: {e}")
            raise ModelError(f"Ollama streaming request failed: {e}")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the given text.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        if not text:
            return 0
        
        if self.backend == "transformers" and self.tokenizer:
            # Use the actual tokenizer for accurate counting
            try:
                tokens = self.tokenizer.encode(text)
                return len(tokens)
            except Exception as e:
                logger.warning(f"Token estimation with tokenizer failed: {e}")
        
        # Fallback to simple estimation
        return super().estimate_tokens(text)
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """
        Get default parameters for local models.
        
        Returns:
            Dictionary of default parameters
        """
        defaults = super()._get_default_parameters()
        defaults.update({
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0,
            "timeout": 60.0,  # Local models may be slower
            "retry_attempts": 2,  # Fewer retries for local models
            "retry_delay": 2.0
        })
        return defaults
    
    def is_available(self) -> bool:
        """
        Check if the local model is currently available for use.
        
        Returns:
            True if the model is available, False otherwise
        """
        if not self._is_initialized:
            return False
        
        try:
            if self.backend == "ollama":
                import requests
                response = requests.get(f"{self.api_endpoint}/api/tags", timeout=5)
                return response.status_code == 200
            elif self.backend == "transformers":
                return self.model is not None and self.tokenizer is not None
            elif self.backend == "custom":
                return True  # Assume available if initialized
            else:
                return False
        except Exception:
            return False
    
    def get_model_info(self) -> ModelInfo:
        """
        Get information about the local model.
        
        Returns:
            ModelInfo object with current model details
        """
        # Update model info with current status
        self.model_info.enabled = self.is_available()
        return self.model_info
    
    def __str__(self) -> str:
        """String representation of the local model provider."""
        return f"LocalModelProvider({self.model_name}, {self.backend})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the local model provider."""
        return f"LocalModelProvider(model_name='{self.model_name}', backend='{self.backend}')"