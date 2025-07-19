"""
Unit tests for the configuration models.

Tests validation, serialization, and configuration loading for AgentConfig,
DisplayConfig, and ConfigurationManager classes.
"""

import pytest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from src.models.config_models import AgentConfig, DisplayConfig, ConfigurationManager


class TestAgentConfig:
    """Test cases for the AgentConfig data model."""
    
    def test_agent_config_creation_with_defaults(self):
        """Test creating agent config with default values."""
        config = AgentConfig()
        
        assert config.model_provider == "openai"
        assert config.model_name == "gpt-3.5-turbo"
        assert config.max_reasoning_steps == 10
        assert config.tool_timeout == 30.0
        assert config.memory_limit == 1000
    
    def test_agent_config_creation_with_custom_values(self):
        """Test creating agent config with custom values."""
        config = AgentConfig(
            model_provider="local",
            model_name="llama-2",
            max_reasoning_steps=5,
            tool_timeout=60.0,
            memory_limit=500
        )
        
        assert config.model_provider == "local"
        assert config.model_name == "llama-2"
        assert config.max_reasoning_steps == 5
        assert config.tool_timeout == 60.0
        assert config.memory_limit == 500
    
    def test_agent_config_validation_empty_provider(self):
        """Test that empty model_provider raises ValueError."""
        with pytest.raises(ValueError, match="AgentConfig model_provider must be a non-empty string"):
            AgentConfig(model_provider="")
    
    def test_agent_config_validation_empty_model_name(self):
        """Test that empty model_name raises ValueError."""
        with pytest.raises(ValueError, match="AgentConfig model_name must be a non-empty string"):
            AgentConfig(model_name="")
    
    def test_agent_config_validation_invalid_reasoning_steps(self):
        """Test that invalid max_reasoning_steps raises ValueError."""
        with pytest.raises(ValueError, match="AgentConfig max_reasoning_steps must be a positive integer"):
            AgentConfig(max_reasoning_steps=0)
        
        with pytest.raises(ValueError, match="AgentConfig max_reasoning_steps must be a positive integer"):
            AgentConfig(max_reasoning_steps=-1)
    
    def test_agent_config_validation_invalid_tool_timeout(self):
        """Test that invalid tool_timeout raises ValueError."""
        with pytest.raises(ValueError, match="AgentConfig tool_timeout must be a positive number"):
            AgentConfig(tool_timeout=0)
        
        with pytest.raises(ValueError, match="AgentConfig tool_timeout must be a positive number"):
            AgentConfig(tool_timeout=-1.0)
    
    def test_agent_config_validation_invalid_memory_limit(self):
        """Test that invalid memory_limit raises ValueError."""
        with pytest.raises(ValueError, match="AgentConfig memory_limit must be a positive integer"):
            AgentConfig(memory_limit=0)
        
        with pytest.raises(ValueError, match="AgentConfig memory_limit must be a positive integer"):
            AgentConfig(memory_limit=-1)
    
    def test_agent_config_to_dict(self):
        """Test agent config serialization to dictionary."""
        config = AgentConfig(
            model_provider="local",
            model_name="test-model",
            max_reasoning_steps=15,
            tool_timeout=45.0,
            memory_limit=2000
        )
        
        expected = {
            "model_provider": "local",
            "model_name": "test-model",
            "max_reasoning_steps": 15,
            "tool_timeout": 45.0,
            "memory_limit": 2000
        }
        
        assert config.to_dict() == expected
    
    def test_agent_config_from_dict(self):
        """Test agent config deserialization from dictionary."""
        data = {
            "model_provider": "anthropic",
            "model_name": "claude-2",
            "max_reasoning_steps": 8,
            "tool_timeout": 25.0,
            "memory_limit": 1500
        }
        
        config = AgentConfig.from_dict(data)
        
        assert config.model_provider == "anthropic"
        assert config.model_name == "claude-2"
        assert config.max_reasoning_steps == 8
        assert config.tool_timeout == 25.0
        assert config.memory_limit == 1500
    
    def test_agent_config_from_dict_with_defaults(self):
        """Test agent config deserialization with missing values uses defaults."""
        data = {"model_provider": "custom"}
        
        config = AgentConfig.from_dict(data)
        
        assert config.model_provider == "custom"
        assert config.model_name == "gpt-3.5-turbo"  # default
        assert config.max_reasoning_steps == 10  # default


class TestDisplayConfig:
    """Test cases for the DisplayConfig data model."""
    
    def test_display_config_creation_with_defaults(self):
        """Test creating display config with default values."""
        config = DisplayConfig()
        
        assert config.animation_speed == 1.0
        assert config.color_scheme == "default"
        assert config.progress_style == "spinner"
        assert config.verbose_mode is False
    
    def test_display_config_creation_with_custom_values(self):
        """Test creating display config with custom values."""
        config = DisplayConfig(
            animation_speed=2.5,
            color_scheme="dark",
            progress_style="bar",
            verbose_mode=True
        )
        
        assert config.animation_speed == 2.5
        assert config.color_scheme == "dark"
        assert config.progress_style == "bar"
        assert config.verbose_mode is True
    
    def test_display_config_validation_invalid_animation_speed(self):
        """Test that invalid animation_speed raises ValueError."""
        with pytest.raises(ValueError, match="DisplayConfig animation_speed must be a positive number"):
            DisplayConfig(animation_speed=0)
        
        with pytest.raises(ValueError, match="DisplayConfig animation_speed must be a positive number"):
            DisplayConfig(animation_speed=-1.0)
    
    def test_display_config_validation_empty_color_scheme(self):
        """Test that empty color_scheme raises ValueError."""
        with pytest.raises(ValueError, match="DisplayConfig color_scheme must be a non-empty string"):
            DisplayConfig(color_scheme="")
    
    def test_display_config_validation_invalid_progress_style(self):
        """Test that invalid progress_style raises ValueError."""
        with pytest.raises(ValueError, match="DisplayConfig progress_style must be one of"):
            DisplayConfig(progress_style="invalid")
    
    def test_display_config_validation_non_bool_verbose(self):
        """Test that non-boolean verbose_mode raises ValueError."""
        with pytest.raises(ValueError, match="DisplayConfig verbose_mode must be a boolean"):
            DisplayConfig(verbose_mode="true")
    
    def test_display_config_to_dict(self):
        """Test display config serialization to dictionary."""
        config = DisplayConfig(
            animation_speed=0.5,
            color_scheme="light",
            progress_style="dots",
            verbose_mode=True
        )
        
        expected = {
            "animation_speed": 0.5,
            "color_scheme": "light",
            "progress_style": "dots",
            "verbose_mode": True
        }
        
        assert config.to_dict() == expected
    
    def test_display_config_from_dict(self):
        """Test display config deserialization from dictionary."""
        data = {
            "animation_speed": 3.0,
            "color_scheme": "custom",
            "progress_style": "minimal",
            "verbose_mode": False
        }
        
        config = DisplayConfig.from_dict(data)
        
        assert config.animation_speed == 3.0
        assert config.color_scheme == "custom"
        assert config.progress_style == "minimal"
        assert config.verbose_mode is False


class TestConfigurationManager:
    """Test cases for the ConfigurationManager class."""
    
    def test_configuration_manager_creation(self):
        """Test creating configuration manager."""
        manager = ConfigurationManager()
        assert manager.config_file is None
        
        manager_with_file = ConfigurationManager("config.json")
        assert manager_with_file.config_file == Path("config.json")
    
    def test_load_agent_config_defaults(self):
        """Test loading agent config with defaults when no file exists."""
        manager = ConfigurationManager()
        config = manager.load_agent_config()
        
        assert isinstance(config, AgentConfig)
        assert config.model_provider == "openai"
        assert config.model_name == "gpt-3.5-turbo"
    
    def test_load_display_config_defaults(self):
        """Test loading display config with defaults when no file exists."""
        manager = ConfigurationManager()
        config = manager.load_display_config()
        
        assert isinstance(config, DisplayConfig)
        assert config.animation_speed == 1.0
        assert config.color_scheme == "default"
    
    def test_load_config_from_file(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "agent": {
                "model_provider": "local",
                "model_name": "test-model",
                "max_reasoning_steps": 5
            },
            "display": {
                "animation_speed": 2.0,
                "color_scheme": "dark"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            manager = ConfigurationManager(config_file)
            
            agent_config = manager.load_agent_config()
            assert agent_config.model_provider == "local"
            assert agent_config.model_name == "test-model"
            assert agent_config.max_reasoning_steps == 5
            assert agent_config.tool_timeout == 30.0  # default value
            
            display_config = manager.load_display_config()
            assert display_config.animation_speed == 2.0
            assert display_config.color_scheme == "dark"
            assert display_config.progress_style == "spinner"  # default value
        finally:
            os.unlink(config_file)
    
    def test_load_config_invalid_json(self):
        """Test that invalid JSON raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json")
            config_file = f.name
        
        try:
            manager = ConfigurationManager(config_file)
            with pytest.raises(ValueError, match="Failed to load agent config from file"):
                manager.load_agent_config()
        finally:
            os.unlink(config_file)
    
    @patch.dict(os.environ, {
        'AGENT_MODEL_PROVIDER': 'custom',
        'AGENT_MAX_REASONING_STEPS': '15',
        'AGENT_TOOL_TIMEOUT': '45.5',
        'DISPLAY_ANIMATION_SPEED': '2.5',
        'DISPLAY_VERBOSE_MODE': 'true'
    })
    def test_load_config_from_environment(self):
        """Test loading configuration from environment variables."""
        manager = ConfigurationManager()
        
        agent_config = manager.load_agent_config()
        assert agent_config.model_provider == "custom"
        assert agent_config.max_reasoning_steps == 15
        assert agent_config.tool_timeout == 45.5
        
        display_config = manager.load_display_config()
        assert display_config.animation_speed == 2.5
        assert display_config.verbose_mode is True
    
    @patch.dict(os.environ, {'AGENT_MAX_REASONING_STEPS': 'invalid'})
    def test_load_config_invalid_env_value(self):
        """Test that invalid environment variable values raise ValueError."""
        manager = ConfigurationManager()
        with pytest.raises(ValueError, match="Invalid integer value"):
            manager.load_agent_config()
    
    def test_save_config(self):
        """Test saving configuration to file."""
        agent_config = AgentConfig(model_provider="test", model_name="test-model")
        display_config = DisplayConfig(animation_speed=2.0, color_scheme="test")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name
        
        try:
            manager = ConfigurationManager(config_file)
            manager.save_config(agent_config, display_config)
            
            # Verify the file was written correctly
            with open(config_file, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["agent"]["model_provider"] == "test"
            assert saved_data["agent"]["model_name"] == "test-model"
            assert saved_data["display"]["animation_speed"] == 2.0
            assert saved_data["display"]["color_scheme"] == "test"
        finally:
            os.unlink(config_file)
    
    def test_save_config_no_file(self):
        """Test that saving without config file raises ValueError."""
        manager = ConfigurationManager()
        agent_config = AgentConfig()
        display_config = DisplayConfig()
        
        with pytest.raises(ValueError, match="No config file specified for saving"):
            manager.save_config(agent_config, display_config)
    
    def test_reload_config(self):
        """Test reloading configuration clears cached values."""
        manager = ConfigurationManager()
        
        # Load configs to cache them
        agent_config1 = manager.load_agent_config()
        display_config1 = manager.load_display_config()
        
        # Reload should clear cache
        manager.reload_config()
        
        # Loading again should create new instances
        agent_config2 = manager.load_agent_config()
        display_config2 = manager.load_display_config()
        
        # Should be equal but not the same object
        assert agent_config1.to_dict() == agent_config2.to_dict()
        assert display_config1.to_dict() == display_config2.to_dict()
    
    def test_config_caching(self):
        """Test that configurations are cached after first load."""
        manager = ConfigurationManager()
        
        # First load
        agent_config1 = manager.load_agent_config()
        display_config1 = manager.load_display_config()
        
        # Second load should return same objects
        agent_config2 = manager.load_agent_config()
        display_config2 = manager.load_display_config()
        
        assert agent_config1 is agent_config2
        assert display_config1 is display_config2