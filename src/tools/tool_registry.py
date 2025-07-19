"""
Tool registry implementation for the AI Agent System.

This module provides the ToolRegistry class that manages tool registration,
discovery, and execution within the system.
"""

import time
from typing import Any, Dict, List
from ..interfaces.tool_interface import ToolInterface, Tool, ToolResult
from ..models.data_models import ToolInfo, ToolCategory


class ToolRegistry(ToolInterface):
    """
    Implementation of the ToolInterface for managing tools.
    
    This class provides a centralized registry for tools, handling registration,
    validation, and execution of tools within the AI Agent System.
    """
    
    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, Tool] = {}
        self._tool_info_cache: Dict[str, ToolInfo] = {}
    
    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool with the system.
        
        Args:
            tool: The tool instance to register
            
        Raises:
            ValueError: If the tool is invalid or already registered
        """
        if not isinstance(tool, Tool):
            raise ValueError("Tool must be an instance of Tool class")
        
        tool_info = tool.get_info()
        tool_name = tool_info.name
        
        if tool_name in self._tools:
            raise ValueError(f"Tool '{tool_name}' is already registered")
        
        # Validate the tool info
        tool_info.validate()
        
        # Register the tool
        self._tools[tool_name] = tool
        self._tool_info_cache[tool_name] = tool_info
    
    def unregister_tool(self, tool_name: str) -> None:
        """
        Unregister a tool from the system.
        
        Args:
            tool_name: Name of the tool to unregister
            
        Raises:
            ValueError: If the tool is not found
        """
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' is not registered")
        
        del self._tools[tool_name]
        del self._tool_info_cache[tool_name]
    
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
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' is not registered")
        
        tool = self._tools[tool_name]
        tool_info = self._tool_info_cache[tool_name]
        
        # Check if tool is enabled
        if not tool_info.enabled:
            return ToolResult(
                success=False,
                error_message=f"Tool '{tool_name}' is disabled"
            )
        
        # Validate parameters
        if not tool.validate_params(params):
            return ToolResult(
                success=False,
                error_message=f"Invalid parameters for tool '{tool_name}'"
            )
        
        # Execute the tool
        start_time = time.time()
        try:
            result = tool.execute(params)
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                success=False,
                error_message=f"Tool execution failed: {str(e)}",
                execution_time=execution_time
            )
    
    def get_available_tools(self) -> List[ToolInfo]:
        """
        Get information about all available tools.
        
        Returns:
            List of ToolInfo objects for all registered tools
        """
        return list(self._tool_info_cache.values())
    
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
        if tool_name not in self._tool_info_cache:
            raise ValueError(f"Tool '{tool_name}' is not registered")
        
        return self._tool_info_cache[tool_name]
    
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
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' is not registered")
        
        tool = self._tools[tool_name]
        return tool.validate_params(params)
    
    def is_tool_available(self, tool_name: str) -> bool:
        """
        Check if a tool is available and enabled.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if the tool is available and enabled, False otherwise
        """
        if tool_name not in self._tool_info_cache:
            return False
        
        tool_info = self._tool_info_cache[tool_name]
        return tool_info.enabled
    
    def get_tools_by_category(self, category: str) -> List[ToolInfo]:
        """
        Get tools filtered by category.
        
        Args:
            category: The category to filter by
            
        Returns:
            List of ToolInfo objects for tools in the specified category
        """
        # Convert string to ToolCategory if needed
        if isinstance(category, str):
            try:
                category_enum = ToolCategory(category)
            except ValueError:
                return []
        else:
            category_enum = category
        
        return [
            tool_info for tool_info in self._tool_info_cache.values()
            if tool_info.category == category_enum
        ]
    
    def get_tool_count(self) -> int:
        """
        Get the total number of registered tools.
        
        Returns:
            Number of registered tools
        """
        return len(self._tools)
    
    def get_enabled_tool_count(self) -> int:
        """
        Get the number of enabled tools.
        
        Returns:
            Number of enabled tools
        """
        return sum(1 for tool_info in self._tool_info_cache.values() if tool_info.enabled)
    
    def enable_tool(self, tool_name: str) -> None:
        """
        Enable a tool.
        
        Args:
            tool_name: Name of the tool to enable
            
        Raises:
            ValueError: If the tool is not found
        """
        if tool_name not in self._tool_info_cache:
            raise ValueError(f"Tool '{tool_name}' is not registered")
        
        self._tool_info_cache[tool_name].enabled = True
    
    def disable_tool(self, tool_name: str) -> None:
        """
        Disable a tool.
        
        Args:
            tool_name: Name of the tool to disable
            
        Raises:
            ValueError: If the tool is not found
        """
        if tool_name not in self._tool_info_cache:
            raise ValueError(f"Tool '{tool_name}' is not registered")
        
        self._tool_info_cache[tool_name].enabled = False
    
    def clear_registry(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        self._tool_info_cache.clear()