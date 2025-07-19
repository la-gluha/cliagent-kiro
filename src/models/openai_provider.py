"""
OpenAI model provider implementation for the AI Agent System.

This module provides the OpenAIProvider class that integrates with OpenAI's API
to provide language model capabilities through the ModelInterface.
"""

import os
import logging
import asyncio
from typing import Any, Dict, Optional, AsyncGenerator
import json
import time

from ..models.model_provider import ModelProvider
from ..models.data_models import ModelInfo, ModelType
from ..exceptions import ModelError, ConfigurationError


logger = logging.getLogger(__name__)


class OpenAIProvider(ModelProvider):
    """
    OpenAI API provider implementation.
    
    This class provides integration with OpenAI's API for language model
    interactions, supporting both synchronous and streaming responses.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None, **kwargs):
        """
        Initialize the OpenAI provider.
        
        Args:
            model_name: Name of the OpenAI model to use
            api_key: OpenAI API key (if not provided, will look for OPENAI_API_KEY env var)
            **kwargs: Additional configuration parameters
        """
        # Set up model info
        model_info = ModelInfo(
            name=model_name,
            model_type=ModelType.OPENAI,
            description=f"OpenAI {model_name} model",
            max_tokens=self._get_model_max_tokens(model_name),
            supports_streaming=True,
            api_endpoint="https://api.openai.com/v1/chat/completions",
            version="1.0",
            enabled=True
        )
        
        # Store OpenAI-specific configuration
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.base_url = kwargs.get("base_url", "https://api.openai.com/v1")
        self.organization = kwargs.get("organization")
        
        # Initialize the base provider
        super().__init__(model_info, **kwargs)
    
    def _get_model_max_tokens(self, model_name: str) -> int:
        """
        Get the maximum token limit for a given OpenAI model.
        
        Args:
            model_name: Name of the OpenAI model
            
        Returns:
            Maximum token limit for the model
        """
        model_limits = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4-turbo-preview": 128000,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "text-davinci-003": 4097,
            "text-davinci-002": 4097,
            "code-davinci-002": 8001
        }
        
        return model_limits.get(model_name, 4096)
    
    def _initialize(self) -> None:
        """
        Initialize the OpenAI provider.
        
        Validates API key and tests connectivity.
        
        Raises:
            ConfigurationError: If API key is missing or invalid
            ModelError: If initialization fails
        """
        if not self.api_key:
            raise ConfigurationError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or provide api_key parameter."
            )
        
        # Test API connectivity
        try:
            self._test_connection()
            self._is_initialized = True
            logger.info(f"OpenAI provider initialized successfully with model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
            raise ModelError(f"OpenAI provider initialization failed: {e}")
    
    def _test_connection(self) -> None:
        """
        Test the connection to OpenAI API.
        
        Raises:
            ModelError: If connection test fails
        """
        try:
            # Try to import openai
            try:
                import openai
            except ImportError:
                raise ModelError(
                    "OpenAI library not installed. Install with: pip install openai"
                )
            
            # Set up the client
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization
            )
            
            # Test with a minimal request
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
                timeout=10
            )
            
            if not response or not response.choices:
                raise ModelError("Invalid response from OpenAI API")
                
        except Exception as e:
            if "openai" in str(e).lower():
                raise ModelError(f"OpenAI API error: {e}")
            else:
                raise ModelError(f"Connection test failed: {e}")
    
    def _make_request(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Make a request to the OpenAI API.
        
        Args:
            prompt: The input prompt
            context: Optional context information
            
        Returns:
            The model's response
            
        Raises:
            ModelError: If the request fails
        """
        try:
            # Prepare messages
            messages = self._prepare_messages(prompt, context)
            
            # Make the API call with retry logic
            response = self._make_api_call_with_retry(messages)
            
            # Extract and return the response
            if not response.choices or not response.choices[0].message:
                raise ModelError("Empty response from OpenAI API")
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            raise ModelError(f"OpenAI request failed: {e}")
    
    def _prepare_messages(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> list:
        """
        Prepare messages for the OpenAI API call.
        
        Args:
            prompt: The user prompt
            context: Optional context information
            
        Returns:
            List of messages formatted for OpenAI API
        """
        messages = []
        
        # Add system message if context contains system instructions
        if context and "system_message" in context:
            messages.append({
                "role": "system",
                "content": context["system_message"]
            })
        
        # Add conversation history if provided
        if context and "conversation_history" in context:
            for msg in context["conversation_history"]:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    # Handle Message objects
                    role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
                    messages.append({
                        "role": role,
                        "content": msg.content
                    })
                elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                    # Handle dictionary messages
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        
        # Add the current user prompt
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return messages
    
    def _make_api_call_with_retry(self, messages: list) -> Any:
        """
        Make an API call with retry logic.
        
        Args:
            messages: List of messages for the API call
            
        Returns:
            OpenAI API response
            
        Raises:
            ModelError: If all retry attempts fail
        """
        retry_attempts = self.parameters.get("retry_attempts", 3)
        retry_delay = self.parameters.get("retry_delay", 1.0)
        
        last_error = None
        
        for attempt in range(retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.parameters.get("temperature", 0.7),
                    max_tokens=self.parameters.get("max_tokens", 1000),
                    top_p=self.parameters.get("top_p", 1.0),
                    frequency_penalty=self.parameters.get("frequency_penalty", 0.0),
                    presence_penalty=self.parameters.get("presence_penalty", 0.0),
                    timeout=self.parameters.get("timeout", 30.0)
                )
                return response
                
            except Exception as e:
                last_error = e
                logger.warning(f"OpenAI API call attempt {attempt + 1} failed: {e}")
                
                if attempt < retry_attempts - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    break
        
        raise ModelError(f"All retry attempts failed. Last error: {last_error}")
    
    async def generate_response_stream(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from OpenAI.
        
        Args:
            prompt: The input prompt for the model
            context: Optional context information to include
            
        Yields:
            Chunks of the response as they are generated
            
        Raises:
            ModelError: If streaming response generation fails
        """
        if not self.supports_streaming():
            raise ModelError("Streaming is not supported by this model provider")
        
        if not self.is_available():
            raise ModelError(f"Model {self.model_info.name} is not available")
        
        if not self.validate_input(prompt, context):
            raise ModelError("Invalid input provided to model")
        
        try:
            # Prepare messages
            messages = self._prepare_messages(prompt, context)
            
            # Make streaming API call
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.parameters.get("temperature", 0.7),
                max_tokens=self.parameters.get("max_tokens", 1000),
                top_p=self.parameters.get("top_p", 1.0),
                frequency_penalty=self.parameters.get("frequency_penalty", 0.0),
                presence_penalty=self.parameters.get("presence_penalty", 0.0),
                stream=True,
                timeout=self.parameters.get("timeout", 30.0)
            )
            
            # Yield chunks as they arrive
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming request failed: {e}")
            raise ModelError(f"OpenAI streaming request failed: {e}")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the given text for OpenAI models.
        
        This uses a more accurate estimation based on OpenAI's tokenization.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        if not text:
            return 0
        
        try:
            # Try to use tiktoken for accurate token counting
            import tiktoken
            
            # Get the encoding for the model
            if "gpt-4" in self.model_name:
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in self.model_name:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                # Fallback to cl100k_base encoding
                encoding = tiktoken.get_encoding("cl100k_base")
            
            return len(encoding.encode(text))
            
        except ImportError:
            # Fallback to simple estimation if tiktoken is not available
            logger.warning("tiktoken not available, using simple token estimation")
            return super().estimate_tokens(text)
        except Exception as e:
            logger.warning(f"Token estimation failed: {e}, using simple estimation")
            return super().estimate_tokens(text)
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """
        Get default parameters for OpenAI models.
        
        Returns:
            Dictionary of default parameters
        """
        defaults = super()._get_default_parameters()
        defaults.update({
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "timeout": 30.0,
            "retry_attempts": 3,
            "retry_delay": 1.0
        })
        return defaults
    
    def get_model_info(self) -> ModelInfo:
        """
        Get information about the OpenAI model.
        
        Returns:
            ModelInfo object with current model details
        """
        # Update model info with current status
        self.model_info.enabled = self._is_initialized
        return self.model_info
    
    def __str__(self) -> str:
        """String representation of the OpenAI provider."""
        return f"OpenAIProvider({self.model_name})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the OpenAI provider."""
        return f"OpenAIProvider(model_name='{self.model_name}', api_key={'***' if self.api_key else None})"