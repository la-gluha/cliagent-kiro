"""
AI API Manager for the AI Agent System.

This module provides centralized management of AI model providers,
including configuration, authentication, and provider selection.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Type
from enum import Enum

from ..models.model_provider import ModelProvider
from ..models.openai_provider import OpenAIProvider
from ..models.local_model_provider import LocalModelProvider
from ..models.data_models import ModelInfo, ModelType
from ..exceptions import ModelError, ConfigurationError


logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    """Status of a model provider."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    NOT_CONFIGURED = "not_configured"


class AIAPIManager:
    """
    Centralized manager for AI model providers.
    
    This class handles provider registration, configuration management,
    authentication, and provider selection for the AI Agent System.
    """
    
    def __init__(self):
        """Initialize the AI API Manager."""
        self.providers: Dict[str, ModelProvider] = {}
        self.provider_classes: Dict[ModelType, Type[ModelProvider]] = {
            ModelType.OPENAI: OpenAIProvider,
            ModelType.LOCAL: LocalModelProvider
        }
        self.default_provider: Optional[str] = None
        self.configurations: Dict[str, Dict[str, Any]] = {}
        
        # Load configurations from environment
        self._load_configurations()
    
    def _load_configurations(self) -> None:
        """
        Load provider configurations from environment variables and config files.
        """
        # OpenAI configuration
        openai_config = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "organization": os.getenv("OPENAI_ORGANIZATION"),
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        }
        self.configurations["openai"] = openai_config
        
        # Local model configuration
        local_config = {
            "backend": os.getenv("LOCAL_MODEL_BACKEND", "ollama"),
            "api_endpoint": os.getenv("LOCAL_MODEL_ENDPOINT", "http://localhost:11434"),
            "model_path": os.getenv("LOCAL_MODEL_PATH"),
            "timeout": float(os.getenv("LOCAL_MODEL_TIMEOUT", "60.0"))
        }
        self.configurations["local"] = local_config
        
        logger.info("Loaded provider configurations from environment")
    
    def register_provider(
        self, 
        provider_id: str, 
        model_type: ModelType, 
        model_name: str,
        **kwargs
    ) -> None:
        """
        Register a new model provider.
        
        Args:
            provider_id: Unique identifier for the provider
            model_type: Type of the model provider
            model_name: Name of the model to use
            **kwargs: Additional configuration parameters
            
        Raises:
            ConfigurationError: If provider type is not supported
            ModelError: If provider registration fails
        """
        try:
            if model_type not in self.provider_classes:
                raise ConfigurationError(f"Unsupported model type: {model_type}")
            
            provider_class = self.provider_classes[model_type]
            
            # Merge with stored configuration
            config = self.configurations.get(model_type.value, {}).copy()
            config.update(kwargs)
            
            # Create and register the provider
            if model_type == ModelType.OPENAI:
                provider = provider_class(model_name=model_name, **config)
            elif model_type == ModelType.LOCAL:
                provider = provider_class(model_name=model_name, **config)
            else:
                provider = provider_class(model_name=model_name, **config)
            
            self.providers[provider_id] = provider
            
            # Set as default if it's the first provider
            if self.default_provider is None:
                self.default_provider = provider_id
            
            logger.info(f"Registered provider '{provider_id}' with model '{model_name}'")
            
        except Exception as e:
            logger.error(f"Failed to register provider '{provider_id}': {e}")
            raise ModelError(f"Provider registration failed: {e}")
    
    def get_provider(self, provider_id: Optional[str] = None) -> ModelProvider:
        """
        Get a model provider by ID.
        
        Args:
            provider_id: ID of the provider to get (uses default if None)
            
        Returns:
            The requested model provider
            
        Raises:
            ModelError: If provider is not found or not available
        """
        if provider_id is None:
            provider_id = self.default_provider
        
        if provider_id is None:
            raise ModelError("No providers registered and no default provider set")
        
        if provider_id not in self.providers:
            raise ModelError(f"Provider '{provider_id}' not found")
        
        provider = self.providers[provider_id]
        
        if not provider.is_available():
            raise ModelError(f"Provider '{provider_id}' is not available")
        
        return provider
    
    def list_providers(self) -> List[Dict[str, Any]]:
        """
        List all registered providers with their status.
        
        Returns:
            List of provider information dictionaries
        """
        providers_info = []
        
        for provider_id, provider in self.providers.items():
            try:
                model_info = provider.get_model_info()
                status = ProviderStatus.AVAILABLE if provider.is_available() else ProviderStatus.UNAVAILABLE
            except Exception as e:
                model_info = None
                status = ProviderStatus.ERROR
                logger.warning(f"Error getting info for provider '{provider_id}': {e}")
            
            providers_info.append({
                "id": provider_id,
                "model_name": model_info.name if model_info else "unknown",
                "model_type": model_info.model_type.value if model_info else "unknown",
                "status": status.value,
                "is_default": provider_id == self.default_provider,
                "supports_streaming": model_info.supports_streaming if model_info else False
            })
        
        return providers_info
    
    def set_default_provider(self, provider_id: str) -> None:
        """
        Set the default provider.
        
        Args:
            provider_id: ID of the provider to set as default
            
        Raises:
            ModelError: If provider is not found
        """
        if provider_id not in self.providers:
            raise ModelError(f"Provider '{provider_id}' not found")
        
        self.default_provider = provider_id
        logger.info(f"Set default provider to '{provider_id}'")
    
    def remove_provider(self, provider_id: str) -> None:
        """
        Remove a provider.
        
        Args:
            provider_id: ID of the provider to remove
            
        Raises:
            ModelError: If provider is not found
        """
        if provider_id not in self.providers:
            raise ModelError(f"Provider '{provider_id}' not found")
        
        del self.providers[provider_id]
        
        # Update default provider if necessary
        if self.default_provider == provider_id:
            self.default_provider = next(iter(self.providers.keys())) if self.providers else None
        
        logger.info(f"Removed provider '{provider_id}'")
    
    def test_provider(self, provider_id: str) -> Dict[str, Any]:
        """
        Test a provider's connectivity and functionality.
        
        Args:
            provider_id: ID of the provider to test
            
        Returns:
            Dictionary with test results
            
        Raises:
            ModelError: If provider is not found
        """
        if provider_id not in self.providers:
            raise ModelError(f"Provider '{provider_id}' not found")
        
        provider = self.providers[provider_id]
        test_results = {
            "provider_id": provider_id,
            "is_available": False,
            "response_test": False,
            "error": None,
            "response_time": None
        }
        
        try:
            # Test availability
            test_results["is_available"] = provider.is_available()
            
            if test_results["is_available"]:
                # Test response generation
                import time
                start_time = time.time()
                
                response = provider.generate_response("Hello, this is a test.")
                
                test_results["response_time"] = time.time() - start_time
                test_results["response_test"] = bool(response and len(response.strip()) > 0)
            
        except Exception as e:
            test_results["error"] = str(e)
            logger.warning(f"Provider test failed for '{provider_id}': {e}")
        
        return test_results
    
    def get_available_providers(self) -> List[str]:
        """
        Get a list of available provider IDs.
        
        Returns:
            List of provider IDs that are currently available
        """
        available = []
        for provider_id, provider in self.providers.items():
            try:
                if provider.is_available():
                    available.append(provider_id)
            except Exception:
                continue
        
        return available
    
    def update_provider_config(self, provider_id: str, config: Dict[str, Any]) -> None:
        """
        Update configuration for a provider.
        
        Args:
            provider_id: ID of the provider to update
            config: New configuration parameters
            
        Raises:
            ModelError: If provider is not found
        """
        if provider_id not in self.providers:
            raise ModelError(f"Provider '{provider_id}' not found")
        
        provider = self.providers[provider_id]
        
        try:
            # Update parameters that can be changed at runtime
            if "parameters" in config:
                provider.set_parameters(config["parameters"])
            
            logger.info(f"Updated configuration for provider '{provider_id}'")
            
        except Exception as e:
            logger.error(f"Failed to update configuration for provider '{provider_id}': {e}")
            raise ModelError(f"Configuration update failed: {e}")
    
    def auto_configure(self) -> None:
        """
        Automatically configure providers based on available credentials and services.
        """
        logger.info("Starting auto-configuration of providers...")
        
        # Try to configure OpenAI provider
        if self.configurations["openai"]["api_key"]:
            try:
                self.register_provider(
                    "openai-gpt-3.5",
                    ModelType.OPENAI,
                    "gpt-3.5-turbo"
                )
                logger.info("Auto-configured OpenAI GPT-3.5 provider")
            except Exception as e:
                logger.warning(f"Failed to auto-configure OpenAI provider: {e}")
        
        # Try to configure local Ollama provider
        try:
            self.register_provider(
                "local-llama2",
                ModelType.LOCAL,
                "llama2",
                backend="ollama"
            )
            logger.info("Auto-configured local Llama2 provider")
        except Exception as e:
            logger.warning(f"Failed to auto-configure local provider: {e}")
        
        if not self.providers:
            logger.warning("No providers could be auto-configured")
        else:
            logger.info(f"Auto-configured {len(self.providers)} provider(s)")
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all providers.
        
        Returns:
            Dictionary with provider statistics
        """
        stats = {
            "total_providers": len(self.providers),
            "available_providers": len(self.get_available_providers()),
            "default_provider": self.default_provider,
            "provider_types": {}
        }
        
        # Count providers by type
        for provider in self.providers.values():
            try:
                model_type = provider.get_model_info().model_type.value
                stats["provider_types"][model_type] = stats["provider_types"].get(model_type, 0) + 1
            except Exception:
                continue
        
        return stats
    
    def __str__(self) -> str:
        """String representation of the AI API Manager."""
        return f"AIAPIManager({len(self.providers)} providers)"
    
    def __repr__(self) -> str:
        """Detailed string representation of the AI API Manager."""
        return f"AIAPIManager(providers={list(self.providers.keys())}, default='{self.default_provider}')"