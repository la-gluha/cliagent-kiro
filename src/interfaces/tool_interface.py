"""
Tool interface definition for the AI Agent System.

This module defines the abstract interface that all tool implementations
must follow to ensure consistent behavior across different tool types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from ..models.data_models import ToolInfo


class ToolResult:
    """
    Represents the result of a tool execution.
    
    Attributes:
        success: Whether the tool execution was successful
        result: The result data from the tool execution
        error_message: Error message if execution failed
        execution_time: Time taken to execute in seconds
    """
    
    def __init__(self, success: bool, result: Any = None, error_message: str = None, execution_time: float = 0.0):
        self.success = success
        self.result = result
        self.error_message = error_message
        self.execution_time = execution_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the tool result to a dictionary."""
        return {
            "success": self.success,
            "result": self.result,
            "error_message": self.error_message,
            "execution_time": self.execution_time
        }


class Tool(ABC):
    """
    Abstract base class for all tools.
    
    Each tool must implement the execute method and provide tool information.
    """
    
    @abstractmethod
    def get_info(self) -> ToolInfo:
        """
        Get information about this tool.
        
        Returns:
            ToolInfo object describing the tool
        """
        pass
    
    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> ToolResult:
        """
        Execute the tool with the given parameters.
        
        Args:
            params: Dictionary of parameters for the tool
            
        Returns:
            ToolResult containing the execution result
        """
        pass
    
    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate the parameters for this tool.
        
        Args:
            params: Dictionary of parameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        pass


class ToolInterface(ABC):
    """
    Abstract interface for tool management in the AI Agent System.
    
    This interface defines the contract for registering, discovering, and
    executing tools within the system.
    """
    
    @abstractmethod
    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool with the system.
        
        Args:
            tool: The tool instance to register
            
        Raises:
            ValueError: If the tool is invalid or already registered
        """
        pass
    
    @abstractmethod
    def unregister_tool(self, tool_name: str) -> None:
        """
        Unregister a tool from the system.
        
        Args:
            tool_name: Name of the tool to unregister
            
        Raises:
            ValueError: If the tool is not found
        """
        pass
    
    @abstractmethod
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters to pass to the tool
            
        Returns:
            ToolResult containing the execution result
            
        Raises:
            ValueError: If the tool is not found or parameters are invalid
        """
        pass
    
    @abstractmethod
    def get_available_tools(self) -> List[ToolInfo]:
        """
        Get information about all available tools.
        
        Returns:
            List of ToolInfo objects for all registered tools
        """
        pass
    
    @abstractmethod
    def get_tool_info(self, tool_name: str) -> ToolInfo:
        """
        Get information about a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            ToolInfo object for the specified tool
            
        Raises:
            ValueError: If the tool is not found
        """
        pass
    
    @abstractmethod
    def validate_tool_input(self, tool_name: str, params: Dict[str, Any]) -> bool:
        """
        Validate input parameters for a tool.
        
        Args:
            tool_name: Name of the tool
            params: Parameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
            
        Raises:
            ValueError: If the tool is not found
        """
        pass
    
    @abstractmethod
    def is_tool_available(self, tool_name: str) -> bool:
        """
        Check if a tool is available and enabled.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if the tool is available and enabled, False otherwise
        """
        pass
    
    @abstractmethod
    def get_tools_by_category(self, category: str) -> List[ToolInfo]:
        """
        Get tools filtered by category.
        
        Args:
            category: The category to filter by
            
        Returns:
            List of ToolInfo objects for tools in the specified category
        """
        pass