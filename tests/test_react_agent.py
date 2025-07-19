"""
Unit tests for the ReActAgent class.

This module contains comprehensive tests for the main ReAct agent
implementation.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.agents.react_agent import ReActAgent
from src.models.data_models import TaskResult, AgentState, Message, MessageRole
from src.interfaces.tool_interface import ToolResult
from src.exceptions import AgentError


class TestReActAgent:
    """Test cases for the ReActAgent class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_model = Mock()
        self.mock_tool_interface = Mock()
        self.mock_memory = Mock()
        
        self.agent = ReActAgent(
            model=self.mock_model,
            tool_interface=self.mock_tool_interface,
            memory=self.mock_memory,
            max_iterations=3,
            max_reasoning_steps=2
        )
    
    def test_init(self):
        """Test ReActAgent initialization."""
        assert self.agent.model == self.mock_model
        assert self.agent.tool_interface == self.mock_tool_interface
        assert self.agent.memory == self.mock_memory
        assert self.agent.max_iterations == 3
        assert self.agent.max_reasoning_steps == 2
        assert isinstance(self.agent.state, AgentState)
        assert not self.agent._stop_requested
    
    def test_process_input_success(self):
        """Test successful input processing."""
        # Arrange
        user_input = "What is the weather today?"
        expected_response = "The weather is sunny today."
        
        # Mock the ReAct cycle to return a response
        with patch.object(self.agent, '_execute_react_cycle', return_value=expected_response):
            # Act
            result = self.agent.process_input(user_input)
        
        # Assert
        assert result == expected_response
        
        # Verify memory interactions
        assert self.mock_memory.store_message.call_count == 2  # User message + agent response
        
        # Check user message was stored
        user_message_call = self.mock_memory.store_message.call_args_list[0]
        user_message = user_message_call[0][0]
        assert user_message.content == user_input
        assert user_message.role == MessageRole.USER
        
        # Check agent message was stored
        agent_message_call = self.mock_memory.store_message.call_args_list[1]
        agent_message = agent_message_call[0][0]
        assert agent_message.content == expected_response
        assert agent_message.role == MessageRole.AGENT
        
        # Check state was managed properly
        assert not self.agent.state.is_active
        assert self.agent.state.current_task is None
    
    def test_process_input_error(self):
        """Test input processing when an error occurs."""
        # Arrange
        user_input = "Test input"
        
        # Mock the ReAct cycle to raise an error
        with patch.object(self.agent, '_execute_react_cycle', side_effect=Exception("Test error")):
            # Act & Assert
            with pytest.raises(AgentError, match="Failed to process input"):
                self.agent.process_input(user_input)
        
        # Check state was cleaned up
        assert not self.agent.state.is_active
        assert self.agent.state.current_task is None
    
    def test_execute_task_success(self):
        """Test successful task execution."""
        # Arrange
        task = "Calculate the sum of 2 and 3"
        expected_result = "The sum is 5"
        
        # Mock the ReAct cycle
        with patch.object(self.agent, '_execute_react_cycle', return_value=expected_result):
            # Act
            result = self.agent.execute_task(task)
        
        # Assert
        assert isinstance(result, TaskResult)
        assert result.success is True
        assert result.result == expected_result
        assert result.execution_time >= 0
        assert len(result.steps_taken) >= 0
        
        # Check state was managed properly
        assert not self.agent.state.is_active
        assert self.agent.state.current_task is None
    
    def test_execute_task_failure(self):
        """Test task execution when an error occurs."""
        # Arrange
        task = "Failing task"
        
        # Mock the ReAct cycle to raise an error
        with patch.object(self.agent, '_execute_react_cycle', side_effect=Exception("Task failed")):
            # Act
            result = self.agent.execute_task(task)
        
        # Assert
        assert isinstance(result, TaskResult)
        assert result.success is False
        assert "Task failed" in result.error_message
        assert result.execution_time >= 0
    
    def test_get_available_actions(self):
        """Test getting available actions."""
        # Arrange
        expected_actions = ["search", "calculate", "file_read"]
        
        with patch.object(self.agent.action_executor, 'get_available_actions', return_value=expected_actions):
            # Act
            actions = self.agent.get_available_actions()
        
        # Assert
        assert actions == expected_actions
    
    def test_reset_context(self):
        """Test context reset."""
        # Arrange
        self.agent.state.current_task = "Some task"
        self.agent.state.is_active = True
        self.agent.state.add_reasoning_step("Some reasoning")
        
        # Act
        self.agent.reset_context()
        
        # Assert
        assert self.agent.state.current_task is None
        assert not self.agent.state.is_active
        assert len(self.agent.state.reasoning_history) == 0
        assert not self.agent._stop_requested
    
    def test_get_state(self):
        """Test getting agent state."""
        # Act
        state = self.agent.get_state()
        
        # Assert
        assert state == self.agent.state
    
    def test_set_state(self):
        """Test setting agent state."""
        # Arrange
        new_state = AgentState()
        new_state.current_task = "New task"
        new_state.is_active = True
        
        # Act
        self.agent.set_state(new_state)
        
        # Assert
        assert self.agent.state == new_state
        assert self.agent.state.current_task == "New task"
        assert self.agent.state.is_active is True
    
    def test_set_state_invalid(self):
        """Test setting invalid agent state."""
        # Arrange
        invalid_state = Mock()
        invalid_state.validate.side_effect = ValueError("Invalid state")
        
        # Act & Assert
        with pytest.raises(AgentError, match="Failed to set state"):
            self.agent.set_state(invalid_state)
    
    def test_is_active(self):
        """Test checking if agent is active."""
        # Initially not active
        assert not self.agent.is_active()
        
        # Set active
        self.agent.state.is_active = True
        assert self.agent.is_active()
    
    def test_stop(self):
        """Test stopping the agent."""
        # Arrange
        self.agent.state.is_active = True
        
        # Act
        self.agent.stop()
        
        # Assert
        assert self.agent._stop_requested is True
        assert not self.agent.state.is_active
    
    def test_execute_react_cycle_simple_response(self):
        """Test ReAct cycle that provides direct response without actions."""
        # Arrange
        input_text = "What is 2 + 2?"
        expected_reasoning = "This is a simple math question. The answer is 4."
        expected_response = "The answer is 4."
        
        # Mock reasoning engine methods and available actions
        with patch.object(self.agent, 'get_available_actions', return_value=["search", "calculate"]):
            with patch.object(self.agent.reasoning_engine, 'generate_reasoning', return_value=expected_reasoning):
                with patch.object(self.agent.reasoning_engine, 'validate_reasoning', return_value=True):
                    with patch.object(self.agent, '_should_take_action', return_value=False):
                        with patch.object(self.agent, '_generate_final_response', return_value=expected_response):
                            # Act
                            result = self.agent._execute_react_cycle(input_text)
        
        # Assert
        assert result == expected_response
        assert len(self.agent.state.reasoning_history) == 1
        assert self.agent.state.reasoning_history[0] == expected_reasoning
    
    def test_execute_react_cycle_with_action(self):
        """Test ReAct cycle that takes an action."""
        # Arrange
        input_text = "Search for Python tutorials"
        reasoning = "I need to search for Python tutorials online."
        action_name = "search"
        action_result = ToolResult(success=True, result="Found 10 Python tutorials")
        final_response = "I found 10 Python tutorials for you."
        
        # Mock components
        with patch.object(self.agent, 'get_available_actions', return_value=["search", "calculate"]):
            with patch.object(self.agent.reasoning_engine, 'generate_reasoning', return_value=reasoning):
                with patch.object(self.agent.reasoning_engine, 'validate_reasoning', return_value=True):
                    with patch.object(self.agent.reasoning_engine, 'determine_next_action', return_value=action_name):
                        with patch.object(self.agent.action_executor, 'execute_action', return_value=action_result):
                            with patch.object(self.agent, '_should_take_action', side_effect=[True, False]):
                                with patch.object(self.agent, '_extract_action_parameters', return_value={"query": "Python tutorials"}):
                                    with patch.object(self.agent, '_process_action_result', return_value="Found tutorials"):
                                        with patch.object(self.agent, '_generate_final_response', return_value=final_response):
                                            # Act
                                            result = self.agent._execute_react_cycle(input_text)
        
        # Assert
        assert result == final_response
        assert len(self.agent.state.reasoning_history) >= 1
    
    def test_execute_react_cycle_max_iterations(self):
        """Test ReAct cycle reaching maximum iterations."""
        # Arrange
        input_text = "Complex task"
        reasoning = "I need to think about this more."
        
        # Mock to always require actions (never provide final response)
        with patch.object(self.agent, 'get_available_actions', return_value=["search", "calculate"]):
            with patch.object(self.agent.reasoning_engine, 'generate_reasoning', return_value=reasoning):
                with patch.object(self.agent.reasoning_engine, 'validate_reasoning', return_value=True):
                    with patch.object(self.agent.reasoning_engine, 'determine_next_action', return_value="search"):
                        with patch.object(self.agent.action_executor, 'execute_action', return_value=ToolResult(success=True, result="result")):
                            with patch.object(self.agent, '_should_take_action', return_value=True):
                                with patch.object(self.agent, '_extract_action_parameters', return_value={}):
                                    with patch.object(self.agent, '_process_action_result', return_value="observation"):
                                        with patch.object(self.agent, '_generate_fallback_response', return_value="fallback"):
                                            # Act
                                            result = self.agent._execute_react_cycle(input_text)
        
        # Assert
        assert "maximum number of reasoning iterations" in result
        assert "fallback" in result
    
    def test_should_take_action_with_indicators(self):
        """Test action decision with action indicators in reasoning."""
        # Arrange
        reasoning_with_action = "I need to search for more information about this topic."
        
        # Act
        result = self.agent._should_take_action(reasoning_with_action, 0)
        
        # Assert
        assert result is True
    
    def test_should_take_action_without_indicators(self):
        """Test action decision without action indicators."""
        # Arrange
        reasoning_without_action = "Based on my knowledge, the answer is clear."
        
        # Act
        result = self.agent._should_take_action(reasoning_without_action, 0)
        
        # Assert
        assert result is False
    
    def test_should_take_action_near_max_iterations(self):
        """Test action decision near maximum iterations."""
        # Arrange
        reasoning_with_action = "I need to search for information."
        
        # Act - near max iterations (should not take action to ensure response)
        result = self.agent._should_take_action(reasoning_with_action, self.agent.max_iterations - 1)
        
        # Assert
        assert result is False
    
    def test_extract_action_parameters_with_quotes(self):
        """Test parameter extraction with quoted strings."""
        # Arrange
        reasoning = 'I need to search for "Python programming tutorial" to find resources.'
        action_name = "search"
        
        # Act
        params = self.agent._extract_action_parameters(reasoning, action_name)
        
        # Assert
        assert "query" in params
        assert params["query"] == "Python programming tutorial"
    
    def test_extract_action_parameters_with_numbers(self):
        """Test parameter extraction with numbers for calculation."""
        # Arrange
        reasoning = "I need to calculate the result of 42.5 using the math tool."
        action_name = "calculate"
        
        # Act
        params = self.agent._extract_action_parameters(reasoning, action_name)
        
        # Assert
        assert "value" in params
        assert params["value"] == 42.5
    
    def test_process_action_result_success(self):
        """Test processing successful action result."""
        # Arrange
        action_result = ToolResult(success=True, result="Operation completed successfully")
        action_name = "test_action"
        
        # Act
        observation = self.agent._process_action_result(action_result, action_name)
        
        # Assert
        assert "SUCCESS" in observation
        assert "test_action" in observation
    
    def test_process_action_result_failure(self):
        """Test processing failed action result."""
        # Arrange
        action_result = ToolResult(success=False, error_message="Network timeout")
        action_name = "test_action"
        
        # Act
        observation = self.agent._process_action_result(action_result, action_name)
        
        # Assert
        assert "FAILED" in observation
        assert "test_action" in observation
    
    def test_generate_final_response(self):
        """Test generating final response."""
        # Arrange
        reasoning = "Based on my analysis, the answer is clear."
        context = "User asked about Python"
        expected_response = "Here's what I found about Python."
        
        self.mock_model.generate_response.return_value = expected_response
        
        with patch.object(self.agent, '_get_recent_conversation_summary', return_value="Recent conversation"):
            # Act
            response = self.agent._generate_final_response(reasoning, context)
        
        # Assert
        assert response == expected_response
    
    def test_generate_final_response_model_error(self):
        """Test generating final response when model fails."""
        # Arrange
        reasoning = "Some reasoning"
        context = "Some context"
        
        self.mock_model.generate_response.side_effect = Exception("Model error")
        
        # Act
        response = self.agent._generate_final_response(reasoning, context)
        
        # Assert
        assert "completed my analysis" in response
    
    def test_generate_fallback_response_with_history(self):
        """Test generating fallback response with reasoning history."""
        # Arrange
        self.agent.state.add_reasoning_step("This is my reasoning about the problem and potential solutions.")
        
        # Act
        response = self.agent._generate_fallback_response()
        
        # Assert
        assert "Based on my analysis:" in response
        assert "This is my reasoning" in response
    
    def test_generate_fallback_response_without_history(self):
        """Test generating fallback response without reasoning history."""
        # Act
        response = self.agent._generate_fallback_response()
        
        # Assert
        assert "attempted to analyze" in response
    
    def test_get_recent_conversation_summary(self):
        """Test getting recent conversation summary."""
        # Arrange
        messages = [
            Message(content="Hello", role=MessageRole.USER),
            Message(content="Hi there!", role=MessageRole.AGENT),
            Message(content="How are you?", role=MessageRole.USER)
        ]
        self.mock_memory.retrieve_history.return_value = messages
        
        # Act
        summary = self.agent._get_recent_conversation_summary()
        
        # Assert
        assert "User: Hello" in summary
        assert "Agent: Hi there!" in summary
        assert "User: How are you?" in summary
    
    def test_get_recent_conversation_summary_no_history(self):
        """Test getting conversation summary with no history."""
        # Arrange
        self.mock_memory.retrieve_history.return_value = []
        
        # Act
        summary = self.agent._get_recent_conversation_summary()
        
        # Assert
        assert summary == "No previous conversation"
    
    def test_get_recent_conversation_summary_error(self):
        """Test getting conversation summary when memory fails."""
        # Arrange
        self.mock_memory.retrieve_history.side_effect = Exception("Memory error")
        
        # Act
        summary = self.agent._get_recent_conversation_summary()
        
        # Assert
        assert "Unable to retrieve" in summary