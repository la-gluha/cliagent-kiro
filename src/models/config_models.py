"""
Configuration models for the AI Agent System.

This module contains configuration classes for agent and display settings,
including loading from files and environment variables.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
from pathlib import Path


@dataclass
class AgentConfig:
    """
    Configuration for AI Agent behavior and settings.
    
    Attributes:
        model_provider: The AI model provider to use (e.g., 'openai', 'local')
        model_name: The specific model name to use
        max_reasoning_steps: Maximum number of reasoning steps in ReAct cycle
        tool_timeout: Timeout for tool execution in seconds
        memory_limit: Maximum number of messages to keep in memory
    """
    model_provider: str = "openai"
    model_name: str = "gpt-3.5-turbo"
    max_reasoning_steps: int = 10
    tool_timeout: float = 30.0
    memory_limit: int = 1000
    
    def __post_init__(self):
        """Validate the agent configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate the agent configuration.
        
        Raises:
            ValueError: If validation fails
        """
        if not self.model_provider or not isinstance(self.model_provider, str):
            raise ValueError("AgentConfig model_provider must be a non-empty string")
        
        if not self.model_name or not isinstance(self.model_name, str):
            raise ValueError("AgentConfig model_name must be a non-empty string")
        
        if not isinstance(self.max_reasoning_steps, int) or self.max_reasoning_steps <= 0:
            raise ValueError("AgentConfig max_reasoning_steps must be a positive integer")
        
        if not isinstance(self.tool_timeout, (int, float)) or self.tool_timeout <= 0:
            raise ValueError("AgentConfig tool_timeout must be a positive number")
        
        if not isinstance(self.memory_limit, int) or self.memory_limit <= 0:
            raise ValueError("AgentConfig memory_limit must be a positive integer")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the agent config to a dictionary for serialization."""
        return {
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "max_reasoning_steps": self.max_reasoning_steps,
            "tool_timeout": self.tool_timeout,
            "memory_limit": self.memory_limit
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create an AgentConfig from a dictionary."""
        return cls(
            model_provider=data.get("model_provider", "openai"),
            model_name=data.get("model_name", "gpt-3.5-turbo"),
            max_reasoning_steps=data.get("max_reasoning_steps", 10),
            tool_timeout=data.get("tool_timeout", 30.0),
            memory_limit=data.get("memory_limit", 1000)
        )


@dataclass
class DisplayConfig:
    """
    Configuration for display and animation settings.
    
    Attributes:
        animation_speed: Speed of animations (higher = faster)
        color_scheme: Color scheme for terminal output
        progress_style: Style of progress indicators
        verbose_mode: Whether to show verbose output
    """
    animation_speed: float = 1.0
    color_scheme: str = "default"
    progress_style: str = "spinner"
    verbose_mode: bool = False
    
    def __post_init__(self):
        """Validate the display configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate the display configuration.
        
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(self.animation_speed, (int, float)) or self.animation_speed <= 0:
            raise ValueError("DisplayConfig animation_speed must be a positive number")
        
        if not self.color_scheme or not isinstance(self.color_scheme, str):
            raise ValueError("DisplayConfig color_scheme must be a non-empty string")
        
        valid_progress_styles = ["spinner", "bar", "dots", "minimal"]
        if self.progress_style not in valid_progress_styles:
            raise ValueError(f"DisplayConfig progress_style must be one of: {valid_progress_styles}")
        
        if not isinstance(self.verbose_mode, bool):
            raise ValueError("DisplayConfig verbose_mode must be a boolean")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the display config to a dictionary for serialization."""
        return {
            "animation_speed": self.animation_speed,
            "color_scheme": self.color_scheme,
            "progress_style": self.progress_style,
            "verbose_mode": self.verbose_mode
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DisplayConfig":
        """Create a DisplayConfig from a dictionary."""
        return cls(
            animation_speed=data.get("animation_speed", 1.0),
            color_scheme=data.get("color_scheme", "default"),
            progress_style=data.get("progress_style", "spinner"),
            verbose_mode=data.get("verbose_mode", False)
        )


class ConfigurationManager:
    """
    Manages loading and saving of configuration from files and environment variables.
    
    Supports loading configuration from:
    - JSON files
    - Environment variables
    - Default values
    """
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to the configuration file (optional)
        """
        self.config_file = Path(config_file) if config_file else None
        self._agent_config: Optional[AgentConfig] = None
        self._display_config: Optional[DisplayConfig] = None
    
    def load_agent_config(self) -> AgentConfig:
        """
        Load agent configuration from file and environment variables.
        
        Returns:
            AgentConfig: The loaded configuration
        """
        if self._agent_config is not None:
            return self._agent_config
        
        # Start with default values
        config_data = {}
        
        # Load from file if it exists
        if self.config_file and self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    config_data.update(file_data.get('agent', {}))
            except (json.JSONDecodeError, KeyError, IOError) as e:
                raise ValueError(f"Failed to load agent config from file: {e}")
        
        # Override with environment variables
        env_overrides = self._load_agent_config_from_env()
        config_data.update(env_overrides)
        
        # Create and cache the configuration
        self._agent_config = AgentConfig.from_dict(config_data)
        return self._agent_config
    
    def load_display_config(self) -> DisplayConfig:
        """
        Load display configuration from file and environment variables.
        
        Returns:
            DisplayConfig: The loaded configuration
        """
        if self._display_config is not None:
            return self._display_config
        
        # Start with default values
        config_data = {}
        
        # Load from file if it exists
        if self.config_file and self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    config_data.update(file_data.get('display', {}))
            except (json.JSONDecodeError, KeyError, IOError) as e:
                raise ValueError(f"Failed to load display config from file: {e}")
        
        # Override with environment variables
        env_overrides = self._load_display_config_from_env()
        config_data.update(env_overrides)
        
        # Create and cache the configuration
        self._display_config = DisplayConfig.from_dict(config_data)
        return self._display_config
    
    def save_config(self, agent_config: AgentConfig, display_config: DisplayConfig) -> None:
        """
        Save configuration to file.
        
        Args:
            agent_config: The agent configuration to save
            display_config: The display configuration to save
        
        Raises:
            ValueError: If no config file is specified or saving fails
        """
        if not self.config_file:
            raise ValueError("No config file specified for saving")
        
        config_data = {
            "agent": agent_config.to_dict(),
            "display": display_config.to_dict()
        }
        
        try:
            # Ensure the directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)
        except IOError as e:
            raise ValueError(f"Failed to save config to file: {e}")
    
    def _load_agent_config_from_env(self) -> Dict[str, Any]:
        """Load agent configuration from environment variables."""
        env_config = {}
        
        # Map environment variable names to config keys
        env_mappings = {
            'AGENT_MODEL_PROVIDER': 'model_provider',
            'AGENT_MODEL_NAME': 'model_name',
            'AGENT_MAX_REASONING_STEPS': 'max_reasoning_steps',
            'AGENT_TOOL_TIMEOUT': 'tool_timeout',
            'AGENT_MEMORY_LIMIT': 'memory_limit'
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert to appropriate type
                if config_key in ['max_reasoning_steps', 'memory_limit']:
                    try:
                        env_config[config_key] = int(value)
                    except ValueError:
                        raise ValueError(f"Invalid integer value for {env_var}: {value}")
                elif config_key == 'tool_timeout':
                    try:
                        env_config[config_key] = float(value)
                    except ValueError:
                        raise ValueError(f"Invalid float value for {env_var}: {value}")
                else:
                    env_config[config_key] = value
        
        return env_config
    
    def _load_display_config_from_env(self) -> Dict[str, Any]:
        """Load display configuration from environment variables."""
        env_config = {}
        
        # Map environment variable names to config keys
        env_mappings = {
            'DISPLAY_ANIMATION_SPEED': 'animation_speed',
            'DISPLAY_COLOR_SCHEME': 'color_scheme',
            'DISPLAY_PROGRESS_STYLE': 'progress_style',
            'DISPLAY_VERBOSE_MODE': 'verbose_mode'
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert to appropriate type
                if config_key == 'animation_speed':
                    try:
                        env_config[config_key] = float(value)
                    except ValueError:
                        raise ValueError(f"Invalid float value for {env_var}: {value}")
                elif config_key == 'verbose_mode':
                    env_config[config_key] = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    env_config[config_key] = value
        
        return env_config
    
    def reload_config(self) -> None:
        """Reload configuration from file and environment variables."""
        self._agent_config = None
        self._display_config = None