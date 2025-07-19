"""
Unit tests for the ActionExecutor class.

This module contains comprehensive tests for the action executor component
of the ReAct agent system.
"""

import pytest
from unittest.mock import Mock, MagicMock
import time
from src.agents.action_executor import ActionExecutor
from src.interfaces.tool_interface import ToolResult
from src.models.data_models import AgentState, ToolInfo, ToolCategory
from src.exceptions import AgentError, ToolExecutionError


class TestActionExecutor:
    """Test cases for the ActionExecutor class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_tool_interface = Mock()
        self.action_executor = ActionExecutor(self.mock_tool_interface)
        self.agent_state = AgentState()
    
    def test_init(self):
        """Test ActionExecutor initialization."""
        assert self.action_executor.tool_interface == self.mock_tool_interface
        assert self.action_executor._execution_timeout == 30.0
    
    def test_execute_action_success(self):
        """Test successful action execution."""
        # Arrange
        action_name = "search"
        parameters = {"query": "test query"}
        expected_result = ToolResult(success=True, result="search results")
        
        self.mock_tool_interface.is_tool_available.return_value = True
        self.mock_tool_interface.validate_tool_input.return_value = True
        self.mock_tool_interface.execute_tool.return_value = expected_result
        
        # Act
        result = self.action_executor.execute_action(
            action_name=action_name,
            parameters=parameters,
            agent_state=self.agent_state
        )
        
        # Assert
        assert result == expected_result
        self.mock_tool_interface.is_tool_available.assert_called_once_with(action_name)
        self.mock_tool_interface.validate_tool_input.assert_called_once_with(action_name, parameters)
        self.mock_tool_interface.execute_tool.assert_called_once_with(action_name, parameters)
        
        # Check that action was recorded in agent state
        assert len(self.agent_state.action_history) == 1
        action_record = self.agent_state.action_history[0]
        assert action_record["action"] == action_name
        assert action_record["parameters"] == parameters
        assert action_record["status"] == "completed"
    
    def test_execute_action_no_parameters(self):
        """Test action execution without parameters."""
        # Arrange
        action_name = "list_files"
        expected_result = ToolResult(success=True, result=["file1.txt", "file2.txt"])
        
        self.mock_tool_interface.is_tool_available.return_value = True
        self.mock_tool_interface.validate_tool_input.return_value = True
        self.mock_tool_interface.execute_tool.return_value = expected_result
        
        # Act
        result = self.action_executor.execute_action(action_name)
        
        # Assert
        assert result == expected_result
        self.mock_tool_interface.validate_tool_input.assert_called_once_with(action_name, {})
        self.mock_tool_interface.execute_tool.assert_called_once_with(action_name, {})
    
    def test_execute_action_tool_not_available(self):
        """Test action execution when tool is not available."""
        # Arrange
        action_name = "unavailable_tool"
        self.mock_tool_interface.is_tool_available.return_value = False
        
        # Act & Assert
        with pytest.raises(AgentError, match="Action 'unavailable_tool' is not available"):
            self.action_executor.execute_action(action_name)
    
    def test_execute_action_invalid_parameters(self):
        """Test action execution with invalid parameters."""
        # Arrange
        action_name = "search"
        parameters = {"invalid": "params"}
        
        self.mock_tool_interface.is_tool_available.return_value = True
        self.mock_tool_interface.validate_tool_input.return_value = False
        
        # Act & Assert
        with pytest.raises(AgentError, match="Invalid parameters for action 'search'"):
            self.action_executor.execute_action(action_name, parameters)
    
    def test_execute_action_tool_execution_error(self):
        """Test action execution when tool execution fails."""
        # Arrange
        action_name = "failing_tool"
        parameters = {"query": "test"}
        
        self.mock_tool_interface.is_tool_available.return_value = True
        self.mock_tool_interface.validate_tool_input.return_value = True
        self.mock_tool_interface.execute_tool.side_effect = ToolExecutionError("Tool failed")
        
        # Act & Assert
        with pytest.raises(AgentError, match="Tool execution failed for 'failing_tool'"):
            self.action_executor.execute_action(
                action_name=action_name,
                parameters=parameters,
                agent_state=self.agent_state
            )
        
        # Check that failure was recorded in agent state
        assert len(self.agent_state.action_history) == 1
        action_record = self.agent_state.action_history[0]
        assert action_record["status"] == "failed"
        assert "Tool failed" in action_record["error"]
    
    def test_execute_action_failed_result(self):
        """Test action execution when tool returns failed result."""
        # Arrange
        action_name = "search"
        parameters = {"query": "test"}
        failed_result = ToolResult(success=False, error_message="Search service unavailable")
        
        self.mock_tool_interface.is_tool_available.return_value = True
        self.mock_tool_interface.validate_tool_input.return_value = True
        self.mock_tool_interface.execute_tool.return_value = failed_result
        
        # Act
        result = self.action_executor.execute_action(
            action_name=action_name,
            parameters=parameters,
            agent_state=self.agent_state
        )
        
        # Assert
        assert result == failed_result
        
        # Check that failure was recorded in agent state
        action_record = self.agent_state.action_history[0]
        assert action_record["status"] == "failed"
        assert action_record["error"] == "Search service unavailable"
    
    def test_get_available_actions(self):
        """Test getting available actions."""
        # Arrange
        tool_infos = [
            ToolInfo(name="search", description="Search tool", enabled=True),
            ToolInfo(name="calculate", description="Calculator", enabled=True),
            ToolInfo(name="disabled_tool", description="Disabled", enabled=False)
        ]
        self.mock_tool_interface.get_available_tools.return_value = tool_infos
        
        # Act
        actions = self.action_executor.get_available_actions()
        
        # Assert
        assert actions == ["search", "calculate"]  # Only enabled tools
    
    def test_get_available_actions_error(self):
        """Test getting available actions when interface raises error."""
        # Arrange
        self.mock_tool_interface.get_available_tools.side_effect = Exception("Interface error")
        
        # Act & Assert
        with pytest.raises(AgentError, match="Failed to get available actions"):
            self.action_executor.get_available_actions()
    
    def test_get_action_info(self):
        """Test getting action information."""
        # Arrange
        action_name = "search"
        tool_info = ToolInfo(
            name="search",
            description="Web search tool",
            parameters={"query": {"type": "string", "required": True}},
            category=ToolCategory.WEB_SEARCH,
            enabled=True
        )
        self.mock_tool_interface.get_tool_info.return_value = tool_info
        
        # Act
        info = self.action_executor.get_action_info(action_name)
        
        # Assert
        assert info["name"] == "search"
        assert info["description"] == "Web search tool"
        assert info["parameters"] == {"query": {"type": "string", "required": True}}
        assert info["category"] == "web_search"
        assert info["enabled"] is True
    
    def test_validate_action_parameters_valid(self):
        """Test parameter validation for valid parameters."""
        # Arrange
        action_name = "search"
        parameters = {"query": "test"}
        self.mock_tool_interface.validate_tool_input.return_value = True
        
        # Act
        result = self.action_executor.validate_action_parameters(action_name, parameters)
        
        # Assert
        assert result is True
    
    def test_validate_action_parameters_invalid(self):
        """Test parameter validation for invalid parameters."""
        # Arrange
        action_name = "search"
        parameters = {"invalid": "params"}
        self.mock_tool_interface.validate_tool_input.return_value = False
        
        # Act
        result = self.action_executor.validate_action_parameters(action_name, parameters)
        
        # Assert
        assert result is False
    
    def test_validate_action_parameters_exception(self):
        """Test parameter validation when interface raises exception."""
        # Arrange
        action_name = "search"
        parameters = {"query": "test"}
        self.mock_tool_interface.validate_tool_input.side_effect = Exception("Validation error")
        
        # Act
        result = self.action_executor.validate_action_parameters(action_name, parameters)
        
        # Assert
        assert result is False
    
    def test_prepare_action_parameters_success(self):
        """Test successful parameter preparation."""
        # Arrange
        action_name = "search"
        raw_parameters = {"query": "test search", "limit": "10"}
        action_info = {
            "parameters": {
                "query": {"type": "string", "required": True},
                "limit": {"type": "integer", "required": False}
            }
        }
        
        # Mock get_action_info to return the action info
        self.action_executor.get_action_info = Mock(return_value=action_info)
        
        # Act
        prepared = self.action_executor.prepare_action_parameters(action_name, raw_parameters)
        
        # Assert
        assert prepared["query"] == "test search"
        assert prepared["limit"] == 10  # Should be converted to integer
    
    def test_prepare_action_parameters_type_conversion(self):
        """Test parameter preparation with type conversions."""
        # Arrange
        action_name = "calculate"
        raw_parameters = {
            "value": "42.5",
            "precision": "2",
            "enabled": "true"
        }
        action_info = {
            "parameters": {
                "value": {"type": "number"},
                "precision": {"type": "integer"},
                "enabled": {"type": "boolean"}
            }
        }
        
        self.action_executor.get_action_info = Mock(return_value=action_info)
        
        # Act
        prepared = self.action_executor.prepare_action_parameters(action_name, raw_parameters)
        
        # Assert
        assert prepared["value"] == 42.5
        assert prepared["precision"] == 2
        assert prepared["enabled"] is True
    
    def test_prepare_action_parameters_missing_required(self):
        """Test parameter preparation with missing required parameter."""
        # Arrange
        action_name = "search"
        raw_parameters = {"limit": "10"}
        action_info = {
            "parameters": {
                "query": {"type": "string", "required": True},
                "limit": {"type": "integer", "required": False}
            }
        }
        
        self.action_executor.get_action_info = Mock(return_value=action_info)
        
        # Act & Assert
        with pytest.raises(AgentError, match="Required parameter 'query' missing"):
            self.action_executor.prepare_action_parameters(action_name, raw_parameters)
    
    def test_set_execution_timeout(self):
        """Test setting execution timeout."""
        # Act
        self.action_executor.set_execution_timeout(60.0)
        
        # Assert
        assert self.action_executor.get_execution_timeout() == 60.0
    
    def test_set_execution_timeout_invalid(self):
        """Test setting invalid execution timeout."""
        # Act & Assert
        with pytest.raises(ValueError, match="Timeout must be positive"):
            self.action_executor.set_execution_timeout(-1.0)
    
    def test_get_action_history(self):
        """Test getting action history."""
        # Arrange
        self.agent_state.add_action({"action": "search", "status": "completed"})
        self.agent_state.add_action({"action": "calculate", "status": "failed"})
        
        # Act
        history = self.action_executor.get_action_history(self.agent_state)
        
        # Assert
        assert len(history) == 2
        assert history[0]["action"] == "search"
        assert history[1]["action"] == "calculate"
    
    def test_get_action_history_with_limit(self):
        """Test getting action history with limit."""
        # Arrange
        for i in range(5):
            self.agent_state.add_action({"action": f"action_{i}", "status": "completed"})
        
        # Act
        history = self.action_executor.get_action_history(self.agent_state, limit=3)
        
        # Assert
        assert len(history) == 3
        assert history[0]["action"] == "action_2"  # Last 3 actions
        assert history[1]["action"] == "action_3"
        assert history[2]["action"] == "action_4"
    
    def test_clear_action_history(self):
        """Test clearing action history."""
        # Arrange
        self.agent_state.add_action({"action": "search", "status": "completed"})
        assert len(self.agent_state.action_history) == 1
        
        # Act
        self.action_executor.clear_action_history(self.agent_state)
        
        # Assert
        assert len(self.agent_state.action_history) == 0