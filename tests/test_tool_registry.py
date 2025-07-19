"""
Unit tests for the ToolRegistry class.

This module contains comprehensive tests for tool registration, validation,
and execution functionality.
"""

import pytest
from unittest.mock import Mock, patch
from src.tools.tool_registry import ToolRegistry
from src.interfaces.tool_interface import Tool, ToolResult
from src.models.data_models import ToolInfo, ToolCategory


class MockTool(Tool):
    """Mock tool for testing purposes."""
    
    def __init__(self, name="mock_tool", enabled=True, should_fail=False):
        self.name = name
        self.enabled = enabled
        self.should_fail = should_fail
        self.execute_called = False
        self.validate_called = False
    
    def get_info(self) -> ToolInfo:
        return ToolInfo(
            name=self.name,
            description="A mock tool for testing",
            parameters={"param1": {"type": "string"}},
            category=ToolCategory.CUSTOM,
            enabled=self.enabled
        )
    
    def execute(self, params):
        self.execute_called = True
        if self.should_fail:
            raise Exception("Mock tool execution failed")
        return ToolResult(success=True, result="mock result")
    
    def validate_params(self, params):
        self.validate_called = True
        return isinstance(params, dict) and "param1" in params


class TestToolRegistry:
    """Test cases for the ToolRegistry class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.registry = ToolRegistry()
        self.mock_tool = MockTool()
    
    def test_init(self):
        """Test ToolRegistry initialization."""
        registry = ToolRegistry()
        assert registry.get_tool_count() == 0
        assert registry.get_enabled_tool_count() == 0
        assert registry.get_available_tools() == []
    
    def test_register_tool_success(self):
        """Test successful tool registration."""
        self.registry.register_tool(self.mock_tool)
        
        assert self.registry.get_tool_count() == 1
        assert self.registry.get_enabled_tool_count() == 1
        assert self.registry.is_tool_available("mock_tool")
        
        tools = self.registry.get_available_tools()
        assert len(tools) == 1
        assert tools[0].name == "mock_tool"
    
    def test_register_tool_invalid_type(self):
        """Test registering invalid tool type."""
        with pytest.raises(ValueError, match="Tool must be an instance of Tool class"):
            self.registry.register_tool("not a tool")
    
    def test_register_duplicate_tool(self):
        """Test registering duplicate tool."""
        self.registry.register_tool(self.mock_tool)
        
        duplicate_tool = MockTool(name="mock_tool")
        with pytest.raises(ValueError, match="Tool 'mock_tool' is already registered"):
            self.registry.register_tool(duplicate_tool)
    
    def test_unregister_tool_success(self):
        """Test successful tool unregistration."""
        self.registry.register_tool(self.mock_tool)
        assert self.registry.get_tool_count() == 1
        
        self.registry.unregister_tool("mock_tool")
        assert self.registry.get_tool_count() == 0
        assert not self.registry.is_tool_available("mock_tool")
    
    def test_unregister_nonexistent_tool(self):
        """Test unregistering non-existent tool."""
        with pytest.raises(ValueError, match="Tool 'nonexistent' is not registered"):
            self.registry.unregister_tool("nonexistent")
    
    def test_execute_tool_success(self):
        """Test successful tool execution."""
        self.registry.register_tool(self.mock_tool)
        
        params = {"param1": "test_value"}
        result = self.registry.execute_tool("mock_tool", params)
        
        assert result.success
        assert result.result == "mock result"
        assert self.mock_tool.execute_called
        assert self.mock_tool.validate_called
    
    def test_execute_nonexistent_tool(self):
        """Test executing non-existent tool."""
        with pytest.raises(ValueError, match="Tool 'nonexistent' is not registered"):
            self.registry.execute_tool("nonexistent", {})
    
    def test_execute_disabled_tool(self):
        """Test executing disabled tool."""
        disabled_tool = MockTool(name="disabled_tool", enabled=False)
        self.registry.register_tool(disabled_tool)
        
        result = self.registry.execute_tool("disabled_tool", {"param1": "test"})
        
        assert not result.success
        assert "disabled" in result.error_message
    
    def test_execute_tool_invalid_params(self):
        """Test executing tool with invalid parameters."""
        self.registry.register_tool(self.mock_tool)
        
        # Mock validate_params to return False
        self.mock_tool.validate_params = Mock(return_value=False)
        
        result = self.registry.execute_tool("mock_tool", {"invalid": "params"})
        
        assert not result.success
        assert "Invalid parameters" in result.error_message
    
    def test_execute_tool_exception(self):
        """Test tool execution with exception."""
        failing_tool = MockTool(name="failing_tool", should_fail=True)
        self.registry.register_tool(failing_tool)
        
        result = self.registry.execute_tool("failing_tool", {"param1": "test"})
        
        assert not result.success
        assert "Tool execution failed" in result.error_message
        assert result.execution_time >= 0  # Should be non-negative
    
    def test_get_tool_info_success(self):
        """Test getting tool info for existing tool."""
        self.registry.register_tool(self.mock_tool)
        
        info = self.registry.get_tool_info("mock_tool")
        
        assert info.name == "mock_tool"
        assert info.description == "A mock tool for testing"
        assert info.category == ToolCategory.CUSTOM
    
    def test_get_tool_info_nonexistent(self):
        """Test getting tool info for non-existent tool."""
        with pytest.raises(ValueError, match="Tool 'nonexistent' is not registered"):
            self.registry.get_tool_info("nonexistent")
    
    def test_validate_tool_input_success(self):
        """Test successful tool input validation."""
        self.registry.register_tool(self.mock_tool)
        
        is_valid = self.registry.validate_tool_input("mock_tool", {"param1": "test"})
        
        assert is_valid
        assert self.mock_tool.validate_called
    
    def test_validate_tool_input_nonexistent(self):
        """Test validating input for non-existent tool."""
        with pytest.raises(ValueError, match="Tool 'nonexistent' is not registered"):
            self.registry.validate_tool_input("nonexistent", {})
    
    def test_get_tools_by_category(self):
        """Test getting tools by category."""
        tool1 = MockTool(name="tool1")
        tool2 = MockTool(name="tool2")
        
        # Mock different categories
        tool1.get_info = Mock(return_value=ToolInfo(
            name="tool1", description="Test", category=ToolCategory.FILE_OPERATIONS
        ))
        tool2.get_info = Mock(return_value=ToolInfo(
            name="tool2", description="Test", category=ToolCategory.CALCULATION
        ))
        
        self.registry.register_tool(tool1)
        self.registry.register_tool(tool2)
        
        file_tools = self.registry.get_tools_by_category("file_operations")
        calc_tools = self.registry.get_tools_by_category("calculation")
        
        assert len(file_tools) == 1
        assert file_tools[0].name == "tool1"
        assert len(calc_tools) == 1
        assert calc_tools[0].name == "tool2"
    
    def test_get_tools_by_invalid_category(self):
        """Test getting tools by invalid category."""
        self.registry.register_tool(self.mock_tool)
        
        tools = self.registry.get_tools_by_category("invalid_category")
        
        assert tools == []
    
    def test_enable_disable_tool(self):
        """Test enabling and disabling tools."""
        self.registry.register_tool(self.mock_tool)
        
        # Initially enabled
        assert self.registry.is_tool_available("mock_tool")
        
        # Disable tool
        self.registry.disable_tool("mock_tool")
        assert not self.registry.is_tool_available("mock_tool")
        assert self.registry.get_enabled_tool_count() == 0
        
        # Enable tool
        self.registry.enable_tool("mock_tool")
        assert self.registry.is_tool_available("mock_tool")
        assert self.registry.get_enabled_tool_count() == 1
    
    def test_enable_disable_nonexistent_tool(self):
        """Test enabling/disabling non-existent tool."""
        with pytest.raises(ValueError, match="Tool 'nonexistent' is not registered"):
            self.registry.enable_tool("nonexistent")
        
        with pytest.raises(ValueError, match="Tool 'nonexistent' is not registered"):
            self.registry.disable_tool("nonexistent")
    
    def test_clear_registry(self):
        """Test clearing the registry."""
        self.registry.register_tool(self.mock_tool)
        self.registry.register_tool(MockTool(name="tool2"))
        
        assert self.registry.get_tool_count() == 2
        
        self.registry.clear_registry()
        
        assert self.registry.get_tool_count() == 0
        assert self.registry.get_enabled_tool_count() == 0
        assert self.registry.get_available_tools() == []
    
    def test_execution_time_tracking(self):
        """Test that execution time is properly tracked."""
        self.registry.register_tool(self.mock_tool)
        
        with patch('time.time', side_effect=[0.0, 0.5]):  # Mock time progression
            result = self.registry.execute_tool("mock_tool", {"param1": "test"})
        
        assert result.execution_time == 0.5