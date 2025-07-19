"""
Custom exceptions for the AI Agent System.

This module defines custom exception classes used throughout the system
to provide clear error handling and debugging information.
"""


class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass


class MemoryError(AgentError):
    """Raised when memory operations fail."""
    pass


class ToolExecutionError(AgentError):
    """Raised when tool execution fails."""
    pass


class ModelError(AgentError):
    """Raised when model interactions fail."""
    pass


class ConfigurationError(AgentError):
    """Raised when configuration is invalid or missing."""
    pass


class ValidationError(AgentError):
    """Raised when data validation fails."""
    pass