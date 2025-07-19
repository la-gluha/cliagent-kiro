"""
Unit tests for the model interface and base provider.

This module contains tests for the ModelInterface compliance and
ModelProvider base class functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from typing import Dict, Any, Optional, AsyncGenerator

from src.interfaces.model_interface import ModelInterface
from src.models.model_provider import ModelProvider
from src.models.data_models import ModelInfo, ModelType
from src.exceptions import ModelError, ValidationError


class MockModelProvider(ModelProvider):
    """Mock model provider for testing."""
    
    def _initialize(self) -> None:
        """Initialize the mock provider."""
        self._is_initialized = True
    
    def _make_request(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Mock request implementation."""
        if "error" in prompt.lower():
            raise Exception("Mock error")
        return f"Mock response to: {prompt[:50]}..."


class TestModelInterface:
    """Test cases for ModelInterface compliance."""
    
    def test_interface_methods_exist(self):
        """Test that all required interface methods exist."""
        required_methods = [
            'generate_response',
            'generate_reasoning',
            'is_available',
            'get_model_info',
            'validate_input',
            'estimate_tokens',
            'supports_streaming',
            'generate_response_stream',
            'set_parameters',
            'get_parameters',
            'reset_parameters'
        ]
        
        for method in required_methods:
            assert hasattr(ModelInterface, method), f"ModelInterface missing method: {method}"
    
    def test_interface_is_abstract(self):
        """Test that ModelInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ModelInterface()


class TestModelProvider:
    """Test cases for ModelProvider base class."""
    
    @pytest.fixture
    def model_info(self):
        """Create a test ModelInfo instance."""
        return ModelInfo(
            name="test-model",
            model_type=ModelType.CUSTOM,
            description="Test model for unit tests",
            max_tokens=2048,
            supports_streaming=False
        )
    
    @pytest.fixture
    def provider(self, model_info):
        """Create a test ModelProvider instance."""
        return MockModelProvider(model_info)
    
    def test_initialization(self, model_info):
        """Test provider initialization."""
        provider = MockModelProvider(model_info)
        
        assert provider.model_info == model_info
        assert provider._is_initialized is True
        assert isinstance(provider.parameters, dict)
        assert "temperature" in provider.parameters
        assert "max_tokens" in provider.parameters
    
    def test_default_parameters(self, provider):
        """Test default parameter values."""
        params = provider.get_parameters()
        
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 1000
        assert params["top_p"] == 1.0
        assert params["frequency_penalty"] == 0.0
        assert params["presence_penalty"] == 0.0
        assert params["timeout"] == 30.0
        assert params["retry_attempts"] == 3
        assert params["retry_delay"] == 1.0
    
    def test_is_available(self, provider):
        """Test availability checking."""
        assert provider.is_available() is True
        
        # Test when model is disabled
        provider.model_info.enabled = False
        assert provider.is_available() is False
        
        # Test when not initialized
        provider.model_info.enabled = True
        provider._is_initialized = False
        assert provider.is_available() is False
    
    def test_get_model_info(self, provider):
        """Test getting model information."""
        info = provider.get_model_info()
        assert info == provider.model_info
        assert info.name == "test-model"
        assert info.model_type == ModelType.CUSTOM
    
    def test_validate_input_valid(self, provider):
        """Test input validation with valid inputs."""
        assert provider.validate_input("Hello, world!") is True
        assert provider.validate_input("Test prompt", {"key": "value"}) is True
    
    def test_validate_input_invalid(self, provider):
        """Test input validation with invalid inputs."""
        # Empty or None prompt
        assert provider.validate_input("") is False
        assert provider.validate_input(None) is False
        assert provider.validate_input("   ") is False
        
        # Non-string prompt
        assert provider.validate_input(123) is False
        assert provider.validate_input([]) is False
        
        # Invalid context
        assert provider.validate_input("Valid prompt", "invalid context") is False
    
    def test_validate_input_too_long(self, provider):
        """Test input validation with overly long input."""
        # Create a very long prompt that exceeds token limit
        long_prompt = "word " * 3000  # Should exceed 2048 token limit
        assert provider.validate_input(long_prompt) is False
    
    def test_estimate_tokens(self, provider):
        """Test token estimation."""
        assert provider.estimate_tokens("") == 0
        assert provider.estimate_tokens("hello world") == 2  # 2 words * 1.3 ≈ 2
        assert provider.estimate_tokens("one two three four five") == 6  # 5 words * 1.3 ≈ 6
    
    def test_supports_streaming(self, provider):
        """Test streaming support check."""
        assert provider.supports_streaming() is False
        
        # Test with streaming enabled
        provider.model_info.supports_streaming = True
        assert provider.supports_streaming() is True
    
    def test_generate_response_success(self, provider):
        """Test successful response generation."""
        response = provider.generate_response("Hello, how are you?")
        assert response.startswith("Mock response to: Hello, how are you?")
    
    def test_generate_response_with_context(self, provider):
        """Test response generation with context."""
        context = {"user": "test_user", "session": "123"}
        response = provider.generate_response("Hello", context)
        assert response.startswith("Mock response to: Hello")
    
    def test_generate_response_unavailable(self, provider):
        """Test response generation when model is unavailable."""
        provider._is_initialized = False
        
        with pytest.raises(ModelError, match="is not available"):
            provider.generate_response("Hello")
    
    def test_generate_response_invalid_input(self, provider):
        """Test response generation with invalid input."""
        with pytest.raises(ValidationError, match="Invalid input"):
            provider.generate_response("")
    
    def test_generate_response_request_error(self, provider):
        """Test response generation with request error."""
        with pytest.raises(ModelError, match="Failed to generate response"):
            provider.generate_response("This should cause an error")
    
    def test_generate_reasoning(self, provider):
        """Test reasoning generation."""
        reasoning = provider.generate_reasoning("What is 2+2?")
        assert "Mock response to: Think step by step" in reasoning
    
    def test_generate_reasoning_with_context(self, provider):
        """Test reasoning generation with context."""
        context = {"domain": "mathematics"}
        reasoning = provider.generate_reasoning("What is 2+2?", context)
        assert "Mock response to: Think step by step" in reasoning
    
    def test_set_parameters_valid(self, provider):
        """Test setting valid parameters."""
        new_params = {
            "temperature": 0.5,
            "max_tokens": 500,
            "custom_param": "value"
        }
        
        provider.set_parameters(new_params)
        params = provider.get_parameters()
        
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 500
        assert params["custom_param"] == "value"
    
    def test_set_parameters_invalid(self, provider):
        """Test setting invalid parameters."""
        # Invalid parameter type
        with pytest.raises(ValueError, match="Parameters must be a dictionary"):
            provider.set_parameters("invalid")
        
        # Invalid parameter values
        with pytest.raises(ValueError, match="Invalid parameter"):
            provider.set_parameters({"temperature": -1.0})
        
        with pytest.raises(ValueError, match="Invalid parameter"):
            provider.set_parameters({"max_tokens": -100})
    
    def test_parameter_validation(self, provider):
        """Test individual parameter validation."""
        # Valid parameters
        assert provider._validate_parameter("temperature", 0.7) is True
        assert provider._validate_parameter("max_tokens", 1000) is True
        assert provider._validate_parameter("top_p", 0.9) is True
        assert provider._validate_parameter("frequency_penalty", 0.5) is True
        assert provider._validate_parameter("custom_param", "any_value") is True
        
        # Invalid parameters
        assert provider._validate_parameter("temperature", -0.1) is False
        assert provider._validate_parameter("temperature", 2.1) is False
        assert provider._validate_parameter("max_tokens", 0) is False
        assert provider._validate_parameter("max_tokens", -1) is False
        assert provider._validate_parameter("top_p", 1.1) is False
        assert provider._validate_parameter("frequency_penalty", -2.1) is False
        assert provider._validate_parameter("frequency_penalty", 2.1) is False
    
    def test_reset_parameters(self, provider):
        """Test parameter reset."""
        # Change parameters
        provider.set_parameters({"temperature": 0.5, "max_tokens": 500})
        
        # Reset parameters
        provider.reset_parameters()
        params = provider.get_parameters()
        
        assert params["temperature"] == 0.7  # Default value
        assert params["max_tokens"] == 1000  # Default value
    
    def test_get_request_stats(self, provider):
        """Test request statistics."""
        stats = provider.get_request_stats()
        
        assert "request_count" in stats
        assert "last_request_time" in stats
        assert "is_initialized" in stats
        assert stats["request_count"] == 0
        assert stats["is_initialized"] is True
        
        # Make a request and check stats update
        provider.generate_response("Test")
        stats = provider.get_request_stats()
        assert stats["request_count"] == 1
        assert stats["last_request_time"] > 0
    
    @pytest.mark.asyncio
    async def test_generate_response_stream_not_supported(self, provider):
        """Test streaming when not supported."""
        with pytest.raises(NotImplementedError, match="Streaming is not supported"):
            async for chunk in provider.generate_response_stream("Hello"):
                pass
    
    @pytest.mark.asyncio
    async def test_generate_response_stream_supported(self, provider):
        """Test streaming when supported."""
        provider.model_info.supports_streaming = True
        
        chunks = []
        async for chunk in provider.generate_response_stream("Hello"):
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert chunks[0].startswith("Mock response to: Hello")
    
    def test_string_representations(self, provider):
        """Test string representations."""
        str_repr = str(provider)
        assert "MockModelProvider" in str_repr
        assert "test-model" in str_repr
        
        repr_str = repr(provider)
        assert "MockModelProvider" in repr_str
        assert "model_info=" in repr_str
        assert "parameters=" in repr_str


class TestModelInterfaceCompliance:
    """Test that concrete implementations comply with the interface."""
    
    def test_mock_provider_implements_interface(self):
        """Test that MockModelProvider implements all interface methods."""
        model_info = ModelInfo(
            name="test-model",
            model_type=ModelType.CUSTOM,
            description="Test model"
        )
        provider = MockModelProvider(model_info)
        
        # Test that provider is an instance of ModelInterface
        assert isinstance(provider, ModelInterface)
        
        # Test that all interface methods are callable
        interface_methods = [
            'generate_response',
            'generate_reasoning',
            'is_available',
            'get_model_info',
            'validate_input',
            'estimate_tokens',
            'supports_streaming',
            'generate_response_stream',
            'set_parameters',
            'get_parameters',
            'reset_parameters'
        ]
        
        for method_name in interface_methods:
            method = getattr(provider, method_name)
            assert callable(method), f"Method {method_name} is not callable"