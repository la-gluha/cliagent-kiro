"""
Base model provider implementation for the AI Agent System.

This module provides the ModelProvider base class with common functionality
that concrete model implementations can inherit from.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncGenerator
from ..interfaces.model_interface import ModelInterface
from ..models.data_models import ModelInfo, ModelType
from ..exceptions import ModelError, ValidationError


logger = logging.getLogger(__name__)


class ModelProvider(ModelInterface, ABC):
    """
    Base class for language model providers.
    
    This class provides common functionality for model providers including
    parameter management, input validation, and error handling.
    """
    
    def __init__(self, model_info: ModelInfo, **kwargs):
        """
        Initialize the model provider.
        
        Args:
            model_info: Information about the model
            **kwargs: Additional configuration parameters
        """
        self.model_info = model_info
        self.parameters = self._get_default_parameters()
        self.parameters.update(kwargs)
        self._last_request_time = 0.0
        self._request_count = 0
        self._is_initialized = False
        
        # Initialize the provider
        self._initialize()
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """
        Get default parameters for the model.
        
        Returns:
            Dictionary of default parameters
        """
        return {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "timeout": 30.0,
            "retry_attempts": 3,
            "retry_delay": 1.0
        }
    
    @abstractmethod
    def _initialize(self) -> None:
        """
        Initialize the model provider.
        
        This method should be implemented by concrete providers to perform
        any necessary initialization (API key validation, model loading, etc.).
        
        Raises:
            ModelError: If initialization fails
        """
        pass
    
    @abstractmethod
    def _make_request(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Make a request to the model.
        
        This method should be implemented by concrete providers to handle
        the actual model request.
        
        Args:
            prompt: The input prompt
            context: Optional context information
            
        Returns:
            The model's response
            
        Raises:
            ModelError: If the request fails
        """
        pass
    
    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a response from the language model.
        
        Args:
            prompt: The input prompt for the model
            context: Optional context information to include
            
        Returns:
            The generated response as a string
            
        Raises:
            ModelError: If response generation fails
        """
        if not self.is_available():
            raise ModelError(f"Model {self.model_info.name} is not available")
        
        if not self.validate_input(prompt, context):
            raise ValidationError("Invalid input provided to model")
        
        try:
            self._update_request_stats()
            response = self._make_request(prompt, context)
            logger.info(f"Generated response using {self.model_info.name}")
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise ModelError(f"Failed to generate response: {e}")
    
    def generate_reasoning(self, problem: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate reasoning steps for a given problem.
        
        Args:
            problem: The problem to reason about
            context: Optional context information to include
            
        Returns:
            The reasoning steps as a string
            
        Raises:
            ModelError: If reasoning generation fails
        """
        reasoning_prompt = self._create_reasoning_prompt(problem, context)
        return self.generate_response(reasoning_prompt, context)
    
    def _create_reasoning_prompt(self, problem: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a prompt for reasoning about a problem.
        
        Args:
            problem: The problem to reason about
            context: Optional context information
            
        Returns:
            A formatted prompt for reasoning
        """
        base_prompt = f"""Think step by step about the following problem:

Problem: {problem}

Please provide your reasoning in a clear, structured way, breaking down the problem into smaller parts and explaining your thought process."""
        
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            base_prompt += f"\n\nContext:\n{context_str}"
        
        return base_prompt
    
    def is_available(self) -> bool:
        """
        Check if the model is currently available for use.
        
        Returns:
            True if the model is available, False otherwise
        """
        return self._is_initialized and self.model_info.enabled
    
    def get_model_info(self) -> ModelInfo:
        """
        Get information about the model.
        
        Returns:
            ModelInfo object containing model details
        """
        return self.model_info
    
    def validate_input(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate input before sending to the model.
        
        Args:
            prompt: The input prompt to validate
            context: Optional context to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        if not prompt or not isinstance(prompt, str):
            logger.warning("Invalid prompt: must be a non-empty string")
            return False
        
        if len(prompt.strip()) == 0:
            logger.warning("Invalid prompt: cannot be empty or whitespace only")
            return False
        
        # Check token limit
        estimated_tokens = self.estimate_tokens(prompt)
        if context:
            context_text = str(context)
            estimated_tokens += self.estimate_tokens(context_text)
        
        if estimated_tokens > self.model_info.max_tokens:
            logger.warning(f"Input too long: {estimated_tokens} tokens exceeds limit of {self.model_info.max_tokens}")
            return False
        
        if context is not None and not isinstance(context, dict):
            logger.warning("Invalid context: must be a dictionary or None")
            return False
        
        return True
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the given text.
        
        This is a simple estimation based on word count.
        Concrete implementations should override this for more accurate estimates.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        if not text:
            return 0
        
        # Simple estimation: roughly 1.3 tokens per word
        words = len(text.split())
        return int(words * 1.3)
    
    def supports_streaming(self) -> bool:
        """
        Check if the model supports streaming responses.
        
        Returns:
            True if streaming is supported, False otherwise
        """
        return self.model_info.supports_streaming
    
    async def generate_response_stream(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the language model.
        
        Base implementation raises NotImplementedError.
        Concrete providers should override this if they support streaming.
        
        Args:
            prompt: The input prompt for the model
            context: Optional context information to include
            
        Yields:
            Chunks of the response as they are generated
            
        Raises:
            NotImplementedError: If streaming is not supported
        """
        if not self.supports_streaming():
            raise NotImplementedError("Streaming is not supported by this model provider")
        
        # Default implementation: yield the full response at once
        response = self.generate_response(prompt, context)
        yield response
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set model parameters (temperature, max_tokens, etc.).
        
        Args:
            parameters: Dictionary of parameter names and values
            
        Raises:
            ModelError: If parameters cannot be set
            ValueError: If parameters are invalid
        """
        if not isinstance(parameters, dict):
            raise ValueError("Parameters must be a dictionary")
        
        # Validate parameters
        for key, value in parameters.items():
            if not self._validate_parameter(key, value):
                raise ValueError(f"Invalid parameter: {key}={value}")
        
        # Update parameters
        self.parameters.update(parameters)
        logger.info(f"Updated parameters for {self.model_info.name}: {parameters}")
    
    def _validate_parameter(self, key: str, value: Any) -> bool:
        """
        Validate a single parameter.
        
        Args:
            key: Parameter name
            value: Parameter value
            
        Returns:
            True if parameter is valid, False otherwise
        """
        if key == "temperature":
            return isinstance(value, (int, float)) and 0.0 <= value <= 2.0
        elif key == "max_tokens":
            return isinstance(value, int) and value > 0
        elif key == "top_p":
            return isinstance(value, (int, float)) and 0.0 <= value <= 1.0
        elif key in ["frequency_penalty", "presence_penalty"]:
            return isinstance(value, (int, float)) and -2.0 <= value <= 2.0
        elif key == "timeout":
            return isinstance(value, (int, float)) and value > 0
        elif key == "retry_attempts":
            return isinstance(value, int) and value >= 0
        elif key == "retry_delay":
            return isinstance(value, (int, float)) and value >= 0
        else:
            # Allow unknown parameters (provider-specific)
            return True
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current model parameters.
        
        Returns:
            Dictionary of current parameter names and values
        """
        return self.parameters.copy()
    
    def reset_parameters(self) -> None:
        """
        Reset model parameters to default values.
        
        Raises:
            ModelError: If parameters cannot be reset
        """
        try:
            self.parameters = self._get_default_parameters()
            logger.info(f"Reset parameters for {self.model_info.name} to defaults")
        except Exception as e:
            raise ModelError(f"Failed to reset parameters: {e}")
    
    def _update_request_stats(self) -> None:
        """Update request statistics."""
        self._last_request_time = time.time()
        self._request_count += 1
    
    def get_request_stats(self) -> Dict[str, Any]:
        """
        Get request statistics.
        
        Returns:
            Dictionary containing request statistics
        """
        return {
            "request_count": self._request_count,
            "last_request_time": self._last_request_time,
            "is_initialized": self._is_initialized
        }
    
    def __str__(self) -> str:
        """String representation of the model provider."""
        return f"{self.__class__.__name__}({self.model_info.name})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model provider."""
        return f"{self.__class__.__name__}(model_info={self.model_info}, parameters={self.parameters})"