"""
Interface definitions for the AI Agent System.

This module contains abstract base classes that define the contracts
for all major system components.
"""

from .agent_interface import AgentInterface
from .cli_interface import CLIInterface
from .display_interface import DisplayInterface, OutputStyle, AnimationType
from .memory_interface import MemoryInterface
from .model_interface import ModelInterface
from .tool_interface import ToolInterface

__all__ = [
    'AgentInterface',
    'CLIInterface', 
    'DisplayInterface',
    'OutputStyle',
    'AnimationType',
    'MemoryInterface',
    'ModelInterface',
    'ToolInterface'
]