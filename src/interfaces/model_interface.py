"""
Model interface definition for the AI Agent System.

This module defines the abstract interface that all language model implementations
must follow to ensure consistent behavior across different model providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncGenerator
from ..models.data_models import ModelInfo


class ModelInterface(ABC):
    """
    Abstract interface for language model interactions in the AI Agent System.
    
    This interface defines the contract for generating responses, reasoning,
    and managing model availability. Implementations can use different model
    providers (OpenAI, local models, etc.).
    """
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the model is currently available for use.
        
        Returns:
            True if the model is available, False otherwise
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """
        Get information about the model.
        
        Returns:
            ModelInfo object containing model details
            
        Raises:
            ModelError: If model information cannot be retrieved
        """
        pass
    
    @abstractmethod
    def validate_input(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate input before sending to the model.
        
        Args:
            prompt: The input prompt to validate
            context: Optional context to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the given text.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        pass
    
    @abstractmethod
    def supports_streaming(self) -> bool:
        """
        Check if the model supports streaming responses.
        
        Returns:
            True if streaming is supported, False otherwise
        """
        pass
    
    @abstractmethod
    async def generate_response_stream(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the language model.
        
        Args:
            prompt: The input prompt for the model
            context: Optional context information to include
            
        Yields:
            Chunks of the response as they are generated
            
        Raises:
            ModelError: If streaming response generation fails
            NotImplementedError: If streaming is not supported
        """
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set model parameters (temperature, max_tokens, etc.).
        
        Args:
            parameters: Dictionary of parameter names and values
            
        Raises:
            ModelError: If parameters cannot be set
            ValueError: If parameters are invalid
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current model parameters.
        
        Returns:
            Dictionary of current parameter names and values
        """
        pass
    
    @abstractmethod
    def reset_parameters(self) -> None:
        """
        Reset model parameters to default values.
        
        Raises:
            ModelError: If parameters cannot be reset
        """
        pass