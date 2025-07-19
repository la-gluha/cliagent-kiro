"""
Unit tests for the core data models.

Tests validation, serialization, and deserialization of all data model classes.
"""

import pytest
from datetime import datetime
from src.models.data_models import (
    Message, MessageRole, TaskResult, ToolInfo, ToolCategory, AgentState
)


class TestMessage:
    """Test cases for the Message data model."""
    
    def test_message_creation_with_defaults(self):
        """Test creating a message with default values."""
        message = Message(content="Hello", role=MessageRole.USER)
        
        assert message.content == "Hello"
        assert message.role == MessageRole.USER
        assert isinstance(message.id, str)
        assert isinstance(message.timestamp, datetime)
        assert message.metadata == {}
    
    def test_message_creation_with_all_fields(self):
        """Test creating a message with all fields specified."""
        timestamp = datetime.now()
        metadata = {"key": "value"}
        
        message = Message(
            id="test-id",
            content="Test message",
            role=MessageRole.AGENT,
            timestamp=timestamp,
            metadata=metadata
        )
        
        assert message.id == "test-id"
        assert message.content == "Test message"
        assert message.role == MessageRole.AGENT
        assert message.timestamp == timestamp
        assert message.metadata == metadata
    
    def test_message_role_string_conversion(self):
        """Test that string roles are converted to MessageRole enum."""
        message = Message(content="Hello", role="user")
        assert message.role == MessageRole.USER
    
    def test_message_validation_empty_content(self):
        """Test that empty content raises ValueError."""
        with pytest.raises(ValueError, match="Message content must be a non-empty string"):
            Message(content="", role=MessageRole.USER)
    
    def test_message_validation_invalid_role(self):
        """Test that invalid role raises ValueError."""
        with pytest.raises(ValueError, match="Invalid message role"):
            Message(content="Hello", role="invalid_role")
    
    def test_message_validation_non_string_content(self):
        """Test that non-string content raises ValueError."""
        with pytest.raises(ValueError, match="Message content must be a non-empty string"):
            Message(content=123, role=MessageRole.USER)
    
    def test_message_validation_non_dict_metadata(self):
        """Test that non-dict metadata raises ValueError."""
        with pytest.raises(ValueError, match="Message metadata must be a dictionary"):
            Message(content="Hello", role=MessageRole.USER, metadata="not_a_dict")
    
    def test_message_to_dict(self):
        """Test message serialization to dictionary."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        message = Message(
            id="test-id",
            content="Test",
            role=MessageRole.USER,
            timestamp=timestamp,
            metadata={"key": "value"}
        )
        
        result = message.to_dict()
        expected = {
            "id": "test-id",
            "content": "Test",
            "role": "user",
            "timestamp": "2023-01-01T12:00:00",
            "metadata": {"key": "value"}
        }
        
        assert result == expected
    
    def test_message_from_dict(self):
        """Test message deserialization from dictionary."""
        data = {
            "id": "test-id",
            "content": "Test",
            "role": "user",
            "timestamp": "2023-01-01T12:00:00",
            "metadata": {"key": "value"}
        }
        
        message = Message.from_dict(data)
        
        assert message.id == "test-id"
        assert message.content == "Test"
        assert message.role == MessageRole.USER
        assert message.timestamp == datetime(2023, 1, 1, 12, 0, 0)
        assert message.metadata == {"key": "value"}


class TestTaskResult:
    """Test cases for the TaskResult data model."""
    
    def test_task_result_creation_success(self):
        """Test creating a successful task result."""
        result = TaskResult(success=True, result="completed", execution_time=1.5)
        
        assert result.success is True
        assert result.result == "completed"
        assert result.error_message is None
        assert result.execution_time == 1.5
        assert result.steps_taken == []
    
    def test_task_result_creation_failure(self):
        """Test creating a failed task result."""
        steps = ["step1", "step2"]
        result = TaskResult(
            success=False,
            error_message="Task failed",
            execution_time=0.5,
            steps_taken=steps
        )
        
        assert result.success is False
        assert result.result is None
        assert result.error_message == "Task failed"
        assert result.execution_time == 0.5
        assert result.steps_taken == steps
    
    def test_task_result_validation_non_bool_success(self):
        """Test that non-boolean success raises ValueError."""
        with pytest.raises(ValueError, match="TaskResult success must be a boolean"):
            TaskResult(success="true")
    
    def test_task_result_validation_non_string_error(self):
        """Test that non-string error message raises ValueError."""
        with pytest.raises(ValueError, match="TaskResult error_message must be a string or None"):
            TaskResult(success=False, error_message=123)
    
    def test_task_result_validation_negative_execution_time(self):
        """Test that negative execution time raises ValueError."""
        with pytest.raises(ValueError, match="TaskResult execution_time must be a non-negative number"):
            TaskResult(success=True, execution_time=-1.0)
    
    def test_task_result_validation_non_list_steps(self):
        """Test that non-list steps_taken raises ValueError."""
        with pytest.raises(ValueError, match="TaskResult steps_taken must be a list"):
            TaskResult(success=True, steps_taken="not_a_list")
    
    def test_task_result_validation_non_string_steps(self):
        """Test that non-string steps in steps_taken raise ValueError."""
        with pytest.raises(ValueError, match="All steps in steps_taken must be strings"):
            TaskResult(success=True, steps_taken=["step1", 123])
    
    def test_task_result_to_dict(self):
        """Test task result serialization to dictionary."""
        result = TaskResult(
            success=True,
            result="data",
            error_message=None,
            execution_time=2.5,
            steps_taken=["step1", "step2"]
        )
        
        expected = {
            "success": True,
            "result": "data",
            "error_message": None,
            "execution_time": 2.5,
            "steps_taken": ["step1", "step2"]
        }
        
        assert result.to_dict() == expected
    
    def test_task_result_from_dict(self):
        """Test task result deserialization from dictionary."""
        data = {
            "success": False,
            "result": None,
            "error_message": "Failed",
            "execution_time": 1.0,
            "steps_taken": ["step1"]
        }
        
        result = TaskResult.from_dict(data)
        
        assert result.success is False
        assert result.result is None
        assert result.error_message == "Failed"
        assert result.execution_time == 1.0
        assert result.steps_taken == ["step1"]


class TestToolInfo:
    """Test cases for the ToolInfo data model."""
    
    def test_tool_info_creation_with_defaults(self):
        """Test creating tool info with default values."""
        tool = ToolInfo(name="test_tool", description="A test tool")
        
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.parameters == {}
        assert tool.category == ToolCategory.CUSTOM
        assert tool.enabled is True
    
    def test_tool_info_creation_with_all_fields(self):
        """Test creating tool info with all fields specified."""
        parameters = {"param1": "string", "param2": "int"}
        tool = ToolInfo(
            name="file_reader",
            description="Reads files",
            parameters=parameters,
            category=ToolCategory.FILE_OPERATIONS,
            enabled=False
        )
        
        assert tool.name == "file_reader"
        assert tool.description == "Reads files"
        assert tool.parameters == parameters
        assert tool.category == ToolCategory.FILE_OPERATIONS
        assert tool.enabled is False
    
    def test_tool_info_category_string_conversion(self):
        """Test that string categories are converted to ToolCategory enum."""
        tool = ToolInfo(name="calc", description="Calculator", category="calculation")
        assert tool.category == ToolCategory.CALCULATION
    
    def test_tool_info_validation_empty_name(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="ToolInfo name must be a non-empty string"):
            ToolInfo(name="", description="Test")
    
    def test_tool_info_validation_empty_description(self):
        """Test that empty description raises ValueError."""
        with pytest.raises(ValueError, match="ToolInfo description must be a non-empty string"):
            ToolInfo(name="test", description="")
    
    def test_tool_info_validation_non_dict_parameters(self):
        """Test that non-dict parameters raise ValueError."""
        with pytest.raises(ValueError, match="ToolInfo parameters must be a dictionary"):
            ToolInfo(name="test", description="Test", parameters="not_a_dict")
    
    def test_tool_info_validation_invalid_category(self):
        """Test that invalid category raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tool category"):
            ToolInfo(name="test", description="Test", category="invalid_category")
    
    def test_tool_info_validation_non_bool_enabled(self):
        """Test that non-boolean enabled raises ValueError."""
        with pytest.raises(ValueError, match="ToolInfo enabled must be a boolean"):
            ToolInfo(name="test", description="Test", enabled="true")
    
    def test_tool_info_to_dict(self):
        """Test tool info serialization to dictionary."""
        tool = ToolInfo(
            name="web_search",
            description="Search the web",
            parameters={"query": "string"},
            category=ToolCategory.WEB_SEARCH,
            enabled=True
        )
        
        expected = {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {"query": "string"},
            "category": "web_search",
            "enabled": True
        }
        
        assert tool.to_dict() == expected
    
    def test_tool_info_from_dict(self):
        """Test tool info deserialization from dictionary."""
        data = {
            "name": "calculator",
            "description": "Performs calculations",
            "parameters": {"expression": "string"},
            "category": "calculation",
            "enabled": False
        }
        
        tool = ToolInfo.from_dict(data)
        
        assert tool.name == "calculator"
        assert tool.description == "Performs calculations"
        assert tool.parameters == {"expression": "string"}
        assert tool.category == ToolCategory.CALCULATION
        assert tool.enabled is False


class TestAgentState:
    """Test cases for the AgentState data model."""
    
    def test_agent_state_creation_with_defaults(self):
        """Test creating agent state with default values."""
        state = AgentState()
        
        assert state.current_task is None
        assert state.reasoning_history == []
        assert state.action_history == []
        assert state.context == {}
        assert state.is_active is False
    
    def test_agent_state_creation_with_all_fields(self):
        """Test creating agent state with all fields specified."""
        reasoning = ["step1", "step2"]
        actions = [{"action": "search", "params": {}}]
        context = {"key": "value"}
        
        state = AgentState(
            current_task="test task",
            reasoning_history=reasoning,
            action_history=actions,
            context=context,
            is_active=True
        )
        
        assert state.current_task == "test task"
        assert state.reasoning_history == reasoning
        assert state.action_history == actions
        assert state.context == context
        assert state.is_active is True
    
    def test_agent_state_validation_non_string_task(self):
        """Test that non-string current_task raises ValueError."""
        with pytest.raises(ValueError, match="AgentState current_task must be a string or None"):
            AgentState(current_task=123)
    
    def test_agent_state_validation_non_list_reasoning(self):
        """Test that non-list reasoning_history raises ValueError."""
        with pytest.raises(ValueError, match="AgentState reasoning_history must be a list"):
            AgentState(reasoning_history="not_a_list")
    
    def test_agent_state_validation_non_string_reasoning_steps(self):
        """Test that non-string reasoning steps raise ValueError."""
        with pytest.raises(ValueError, match="All items in reasoning_history must be strings"):
            AgentState(reasoning_history=["step1", 123])
    
    def test_agent_state_validation_non_list_actions(self):
        """Test that non-list action_history raises ValueError."""
        with pytest.raises(ValueError, match="AgentState action_history must be a list"):
            AgentState(action_history="not_a_list")
    
    def test_agent_state_validation_non_dict_actions(self):
        """Test that non-dict actions raise ValueError."""
        with pytest.raises(ValueError, match="All items in action_history must be dictionaries"):
            AgentState(action_history=[{"action": "test"}, "not_a_dict"])
    
    def test_agent_state_validation_non_dict_context(self):
        """Test that non-dict context raises ValueError."""
        with pytest.raises(ValueError, match="AgentState context must be a dictionary"):
            AgentState(context="not_a_dict")
    
    def test_agent_state_validation_non_bool_active(self):
        """Test that non-boolean is_active raises ValueError."""
        with pytest.raises(ValueError, match="AgentState is_active must be a boolean"):
            AgentState(is_active="true")
    
    def test_agent_state_add_reasoning_step(self):
        """Test adding a reasoning step."""
        state = AgentState()
        state.add_reasoning_step("New reasoning step")
        
        assert state.reasoning_history == ["New reasoning step"]
    
    def test_agent_state_add_reasoning_step_invalid(self):
        """Test that adding non-string reasoning step raises ValueError."""
        state = AgentState()
        with pytest.raises(ValueError, match="Reasoning step must be a string"):
            state.add_reasoning_step(123)
    
    def test_agent_state_add_action(self):
        """Test adding an action."""
        state = AgentState()
        action = {"action": "search", "params": {"query": "test"}}
        state.add_action(action)
        
        assert state.action_history == [action]
    
    def test_agent_state_add_action_invalid(self):
        """Test that adding non-dict action raises ValueError."""
        state = AgentState()
        with pytest.raises(ValueError, match="Action must be a dictionary"):
            state.add_action("not_a_dict")
    
    def test_agent_state_clear_history(self):
        """Test clearing history."""
        state = AgentState(
            reasoning_history=["step1", "step2"],
            action_history=[{"action": "test"}]
        )
        
        state.clear_history()
        
        assert state.reasoning_history == []
        assert state.action_history == []
    
    def test_agent_state_to_dict(self):
        """Test agent state serialization to dictionary."""
        state = AgentState(
            current_task="test",
            reasoning_history=["step1"],
            action_history=[{"action": "search"}],
            context={"key": "value"},
            is_active=True
        )
        
        expected = {
            "current_task": "test",
            "reasoning_history": ["step1"],
            "action_history": [{"action": "search"}],
            "context": {"key": "value"},
            "is_active": True
        }
        
        assert state.to_dict() == expected
    
    def test_agent_state_from_dict(self):
        """Test agent state deserialization from dictionary."""
        data = {
            "current_task": "test task",
            "reasoning_history": ["step1", "step2"],
            "action_history": [{"action": "test"}],
            "context": {"key": "value"},
            "is_active": False
        }
        
        state = AgentState.from_dict(data)
        
        assert state.current_task == "test task"
        assert state.reasoning_history == ["step1", "step2"]
        assert state.action_history == [{"action": "test"}]
        assert state.context == {"key": "value"}
        assert state.is_active is False