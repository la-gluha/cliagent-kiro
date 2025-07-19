"""
Integration tests for concrete model provider implementations.

This module contains tests for OpenAIProvider and LocalModelProvider
functionality and integration.
"""

import pytest
import asyncio
import os
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.models.openai_provider import OpenAIProvider
from src.models.local_model_provider import LocalModelProvider
from src.models.data_models import ModelInfo, ModelType
from src.exceptions import ModelError, ConfigurationError, ValidationError


class TestOpenAIProvider:
    """Test cases for OpenAIProvider."""
    
    @pytest.fixture
    def mock_api_key(self):
        """Provide a mock API key for testing."""
        return "sk-test-key-12345"
    
    @pytest.fixture
    def provider_with_mock_key(self, mock_api_key):
        """Create an OpenAI provider with a mock API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": mock_api_key}):
            with patch.object(OpenAIProvider, '_test_api_connection'):
                return OpenAIProvider(model_name="gpt-3.5-turbo")
    
    def test_initialization_with_api_key(self, mock_api_key):
        """Test provider initialization with API key."""
        with patch.object(OpenAIProvider, '_test_api_connection'):
            provider = OpenAIProvider(model_name="gpt-3.5-turbo", api_key=mock_api_key)
            
            assert provider.api_key == mock_api_key
            assert provider.model_info.name == "gpt-3.5-turbo"
            assert provider.model_info.model_type == ModelType.OPENAI
            assert provider.is_available() is True
    
    def test_initialization_from_environment(self, mock_api_key):
        """Test provider initialization from environment variable."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": mock_api_key}):
            with patch.object(OpenAIProvider, '_test_api_connection'):
                provider = OpenAIProvider(model_name="gpt-4")
                
                assert provider.api_key == mock_api_key
                assert provider.model_info.name == "gpt-4"
    
    def test_initialization_without_api_key(self):
        """Test provider initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError, match="OpenAI API key is required"):
                OpenAIProvider(model_name="gpt-3.5-turbo")
    
    def test_model_info_creation(self, provider_with_mock_key):
        """Test model info creation for different models."""
        provider = provider_with_mock_key
        
        assert provider.model_info.name == "gpt-3.5-turbo"
        assert provider.model_info.max_tokens == 4096
        assert provider.model_info.supports_streaming is True
        assert "GPT-3.5 Turbo" in provider.model_info.description
    
    def test_model_info_gpt4(self, mock_api_key):
        """Test model info for GPT-4."""
        with patch.object(OpenAIProvider, '_test_api_connection'):
            provider = OpenAIProvider(model_name="gpt-4", api_key=mock_api_key)
            
            assert provider.model_info.max_tokens == 8192
            assert provider.model_info.supports_streaming is True
            assert "GPT-4" in provider.model_info.description
    
    def test_model_info_unknown_model(self, mock_api_key):
        """Test model info for unknown model."""
        with patch.object(OpenAIProvider, '_test_api_connection'):
            provider = OpenAIProvider(model_name="unknown-model", api_key=mock_api_key)
            
            assert provider.model_info.name == "unknown-model"
            assert provider.model_info.max_tokens == 4096  # Default
            assert provider.model_info.supports_streaming is False  # Default
    
    @patch('src.models.openai_provider.urlopen')
    def test_api_connection_test_success(self, mock_urlopen, mock_api_key):
        """Test successful API connection test."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        provider = OpenAIProvider(model_name="gpt-3.5-turbo", api_key=mock_api_key)
        assert provider.is_available() is True
    
    @patch('src.models.openai_provider.urlopen')
    def test_api_connection_test_failure(self, mock_urlopen, mock_api_key):
        """Test failed API connection test."""
        # Mock failed response
        mock_urlopen.side_effect = Exception("Connection failed")
        
        with pytest.raises(ModelError, match="OpenAI provider initialization failed"):
            OpenAIProvider(model_name="gpt-3.5-turbo", api_key=mock_api_key)
    
    def test_prepare_messages(self, provider_with_mock_key):
        """Test message preparation for API."""
        provider = provider_with_mock_key
        
        # Test simple prompt
        messages = provider._prepare_messages("Hello, world!")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, world!"
        
        # Test with system message in context
        context = {"system_message": "You are a helpful assistant."}
        messages = provider._prepare_messages("Hello", context)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"
    
    def test_prepare_request_data(self, provider_with_mock_key):
        """Test request data preparation."""
        provider = provider_with_mock_key
        messages = [{"role": "user", "content": "Hello"}]
        
        request_data = provider._prepare_request_data(messages)
        
        assert request_data["model"] == "gpt-3.5-turbo"
        assert request_data["messages"] == messages
        assert "temperature" in request_data
        assert "max_tokens" in request_data
        assert request_data["max_tokens"] <= provider.model_info.max_tokens
    
    @patch('src.models.openai_provider.urlopen')
    def test_successful_api_call(self, mock_urlopen, provider_with_mock_key):
        """Test successful API call."""
        # Mock successful API response
        mock_response_data = {
            "choices": [
                {
                    "message": {
                        "content": "Hello! How can I help you today?"
                    }
                }
            ]
        }
        
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_response_data).encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        provider = provider_with_mock_key
        response = provider.generate_response("Hello")
        
        assert response == "Hello! How can I help you today?"
    
    @patch('src.models.openai_provider.urlopen')
    def test_api_error_response(self, mock_urlopen, provider_with_mock_key):
        """Test API error response handling."""
        # Mock error response
        mock_response_data = {
            "error": {
                "message": "Invalid API key"
            }
        }
        
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_response_data).encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        provider = provider_with_mock_key
        
        with pytest.raises(ModelError, match="OpenAI API error: Invalid API key"):
            provider.generate_response("Hello")
    
    def test_token_estimation(self, provider_with_mock_key):
        """Test token estimation for OpenAI models."""
        provider = provider_with_mock_key
        
        assert provider.estimate_tokens("") == 0
        assert provider.estimate_tokens("hello world") == 3  # 8 chars / 4 + 1
        assert provider.estimate_tokens("a" * 100) == 26  # 100 chars / 4 + 1
    
    def test_set_api_key(self, provider_with_mock_key):
        """Test setting a new API key."""
        provider = provider_with_mock_key
        old_key = provider.api_key
        new_key = "sk-new-test-key"
        
        with patch.object(provider, '_test_api_connection'):
            provider.set_api_key(new_key)
            assert provider.api_key == new_key
    
    def test_set_invalid_api_key(self, provider_with_mock_key):
        """Test setting an invalid API key."""
        provider = provider_with_mock_key
        old_key = provider.api_key
        
        with patch.object(provider, '_test_api_connection', side_effect=Exception("Invalid key")):
            with pytest.raises(ModelError, match="Invalid API key"):
                provider.set_api_key("invalid-key")
            
            # Should restore old key
            assert provider.api_key == old_key
    
    def test_usage_stats(self, provider_with_mock_key):
        """Test usage statistics."""
        provider = provider_with_mock_key
        stats = provider.get_usage_stats()
        
        assert "model_name" in stats
        assert "api_endpoint" in stats
        assert "request_count" in stats
        assert stats["model_name"] == "gpt-3.5-turbo"


class TestLocalModelProvider:
    """Test cases for LocalModelProvider."""
    
    @pytest.fixture
    def provider(self):
        """Create a local model provider for testing."""
        return LocalModelProvider(model_name="test-local-model", mock_mode=True)
    
    def test_initialization_mock_mode(self):
        """Test provider initialization in mock mode."""
        provider = LocalModelProvider(model_name="test-model", mock_mode=True)
        
        assert provider.model_info.name == "test-model"
        assert provider.model_info.model_type == ModelType.LOCAL
        assert provider.mock_mode is True
        assert provider.is_available() is True
    
    def test_initialization_non_mock_mode(self):
        """Test provider initialization in non-mock mode."""
        provider = LocalModelProvider(
            model_name="test-model", 
            mock_mode=False, 
            model_path="/path/to/model"
        )
        
        assert provider.mock_mode is False
        assert provider.model_path == "/path/to/model"
        assert provider.is_available() is True
    
    def test_initialization_non_mock_without_path(self):
        """Test provider initialization in non-mock mode without path."""
        with pytest.raises(ModelError, match="Local model provider initialization failed"):
            LocalModelProvider(model_name="test-model", mock_mode=False)
    
    def test_mock_response_generation(self, provider):
        """Test mock response generation."""
        # Test greeting
        response = provider.generate_response("Hello there!")
        assert any(word in response.lower() for word in ["hello", "hi", "greetings", "help"])
        
        # Test question
        response = provider.generate_response("What is the weather like?")
        assert any(word in response.lower() for word in ["question", "think", "interesting", "perspective"])
        
        # Test reasoning
        response = provider.generate_response("Think step by step about this problem")
        # Just check that we get a non-empty response for reasoning prompts
        assert isinstance(response, str) and len(response) > 0
    
    def test_mock_response_with_context(self, provider):
        """Test mock response generation with context."""
        context = {"user": "test_user", "domain": "science"}
        response = provider.generate_response("What is gravity?", context)
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_mock_error_simulation(self, provider):
        """Test mock error simulation."""
        with pytest.raises(ModelError, match="Local model request failed"):
            provider.generate_response("This should cause an error")
    
    def test_reasoning_response(self, provider):
        """Test reasoning response generation."""
        response = provider._generate_reasoning_response("Solve this problem", None)
        
        assert "break down" in response.lower()
        assert "reasoning" in response.lower()
        assert "conclusion" in response.lower()
    
    def test_token_estimation(self, provider):
        """Test token estimation for local models."""
        assert provider.estimate_tokens("") == 0
        assert provider.estimate_tokens("hello world") == 2  # 2 words * 1.2 ≈ 2
        assert provider.estimate_tokens("one two three four five") == 6  # 5 words * 1.2 = 6
    
    @pytest.mark.asyncio
    async def test_streaming_response(self, provider):
        """Test streaming response generation."""
        chunks = []
        async for chunk in provider.generate_response_stream("Tell me a story"):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert len(full_response) > 0
    
    @pytest.mark.asyncio
    async def test_streaming_not_supported(self):
        """Test streaming when not supported."""
        provider = LocalModelProvider(
            model_name="no-stream-model", 
            mock_mode=True, 
            supports_streaming=False
        )
        
        with pytest.raises(NotImplementedError, match="Streaming is not supported"):
            async for chunk in provider.generate_response_stream("Hello"):
                pass
    
    def test_set_mock_mode(self, provider):
        """Test setting mock mode."""
        assert provider.mock_mode is True
        
        provider.set_mock_mode(False)
        assert provider.mock_mode is False
        
        provider.set_mock_mode(True)
        assert provider.mock_mode is True
    
    def test_set_response_delay(self, provider):
        """Test setting response delay."""
        assert provider.response_delay == 0.5  # Default
        
        provider.set_response_delay(1.0)
        assert provider.response_delay == 1.0
        
        with pytest.raises(ValueError, match="Response delay must be non-negative"):
            provider.set_response_delay(-1.0)
    
    def test_model_status(self, provider):
        """Test getting model status."""
        status = provider.get_model_status()
        
        assert "model_name" in status
        assert "mock_mode" in status
        assert "response_delay" in status
        assert "is_initialized" in status
        assert "is_available" in status
        
        assert status["model_name"] == "test-local-model"
        assert status["mock_mode"] is True
        assert status["is_available"] is True
    
    def test_reload_model(self, provider):
        """Test model reloading."""
        assert provider.is_available() is True
        
        provider.reload_model()
        assert provider.is_available() is True
    
    def test_memory_usage(self, provider):
        """Test memory usage information."""
        usage = provider.get_memory_usage()
        
        assert "model_memory_mb" in usage
        assert "cache_memory_mb" in usage
        assert "total_memory_mb" in usage
        assert "mock_mode" in usage
        
        # In mock mode, memory should be 0
        assert usage["model_memory_mb"] == 0
        assert usage["mock_mode"] is True
    
    def test_clear_cache(self, provider):
        """Test cache clearing."""
        # Should not raise any exceptions
        provider.clear_cache()
    
    def test_string_representation(self, provider):
        """Test string representations."""
        str_repr = str(provider)
        assert "LocalModelProvider" in str_repr
        assert "test-local-model" in str_repr
        assert "mock" in str_repr


class TestModelProviderIntegration:
    """Integration tests for model providers."""
    
    def test_provider_interface_compliance(self):
        """Test that both providers implement the interface correctly."""
        from src.interfaces.model_interface import ModelInterface
        
        # Test LocalModelProvider
        local_provider = LocalModelProvider("test-local", mock_mode=True)
        assert isinstance(local_provider, ModelInterface)
        
        # Test OpenAIProvider (with mocked initialization)
        with patch.object(OpenAIProvider, '_test_api_connection'):
            openai_provider = OpenAIProvider("gpt-3.5-turbo", api_key="test-key")
            assert isinstance(openai_provider, ModelInterface)
    
    def test_parameter_management(self):
        """Test parameter management across providers."""
        local_provider = LocalModelProvider("test-local", mock_mode=True)
        
        # Test setting parameters
        new_params = {"temperature": 0.5, "max_tokens": 500}
        local_provider.set_parameters(new_params)
        
        params = local_provider.get_parameters()
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 500
        
        # Test parameter reset
        local_provider.reset_parameters()
        params = local_provider.get_parameters()
        assert params["temperature"] == 0.7  # Default
    
    def test_error_handling_consistency(self):
        """Test consistent error handling across providers."""
        local_provider = LocalModelProvider("test-local", mock_mode=True)
        
        # Test invalid input handling
        with pytest.raises(ValidationError):
            local_provider.generate_response("")
        
        # Test unavailable model
        local_provider._is_initialized = False
        with pytest.raises(ModelError, match="is not available"):
            local_provider.generate_response("Hello")
    
    def test_model_info_consistency(self):
        """Test model info consistency across providers."""
        local_provider = LocalModelProvider("test-local", mock_mode=True)
        
        info = local_provider.get_model_info()
        assert info.name == "test-local"
        assert info.model_type == ModelType.LOCAL
        assert isinstance(info.max_tokens, int)
        assert isinstance(info.supports_streaming, bool)
        assert isinstance(info.enabled, bool)