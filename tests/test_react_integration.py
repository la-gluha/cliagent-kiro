"""
Integration tests for the complete ReAct agent system.

This module contains integration tests that verify the complete ReAct cycle
works correctly with all components working together.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.agents.react_agent import ReActAgent
from src.agents.observation_processor import ObservationProcessor
from src.models.data_models import TaskResult, AgentState, Message, MessageRole, ToolInfo, ToolCategory
from src.interfaces.tool_interface import ToolResult
from src.exceptions import AgentError


class TestReActIntegration:
    """Integration test cases for the complete ReAct agent system."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_model = Mock()
        self.mock_tool_interface = Mock()
        self.mock_memory = Mock()
        
        # Set up mock tool interface with realistic responses
        self.mock_tool_interface.get_available_tools.return_value = [
            ToolInfo(name="search", description="Web search", category=ToolCategory.WEB_SEARCH),
            ToolInfo(name="calculate", description="Calculator", category=ToolCategory.CALCULATION),
            ToolInfo(name="file_read", description="Read files", category=ToolCategory.FILE_OPERATIONS)
        ]
        
        self.mock_tool_interface.is_tool_available.return_value = True
        self.mock_tool_interface.validate_tool_input.return_value = True
        
        # Set up mock memory
        self.mock_memory.get_context.return_value = {}
        self.mock_memory.retrieve_history.return_value = []
        
        self.agent = ReActAgent(
            model=self.mock_model,
            tool_interface=self.mock_tool_interface,
            memory=self.mock_memory,
            max_iterations=3,
            max_reasoning_steps=2
        )
    
    def test_complete_react_cycle_with_search_action(self):
        """Test a complete ReAct cycle that performs a search action."""
        # Arrange
        user_input = "Find information about Python programming"
        
        # Mock model responses for reasoning and final response
        reasoning_responses = [
            "I need to search for information about Python programming to help the user.",
            "Based on the search results, I can now provide a comprehensive answer."
        ]
        self.mock_model.generate_reasoning.side_effect = reasoning_responses
        self.mock_model.generate_response.return_value = "Python is a versatile programming language..."
        
        # Mock tool execution
        search_result = ToolResult(
            success=True,
            result=["Python tutorial", "Python documentation", "Python examples"],
            execution_time=0.5
        )
        self.mock_tool_interface.execute_tool.return_value = search_result
        
        # Mock the agent to take action on first iteration, then provide final response
        with patch.object(self.agent, '_should_take_action', side_effect=[True, False]):
            with patch.object(self.agent.reasoning_engine, 'determine_next_action', return_value='search'):
                # Act
                response = self.agent.process_input(user_input)
        
        # Assert
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Verify the ReAct cycle components were used
        assert self.mock_model.generate_reasoning.call_count >= 1
        assert self.mock_tool_interface.execute_tool.called
        assert self.mock_memory.store_message.call_count == 2  # User input + agent response
        
        # Verify agent state was updated
        assert len(self.agent.state.reasoning_history) >= 1
        assert len(self.agent.state.action_history) >= 1
        assert len(self.agent.get_observation_history()) >= 1
        
        # Verify agent is no longer active
        assert not self.agent.is_active()
        assert self.agent.state.current_task is None
    
    def test_complete_react_cycle_with_calculation_action(self):
        """Test a complete ReAct cycle that performs a calculation."""
        # Arrange
        user_input = "What is 15 * 23?"
        
        # Mock model responses
        reasoning_responses = [
            "I need to calculate 15 * 23 to answer the user's question.",
            "Now I have the calculation result and can provide the answer."
        ]
        self.mock_model.generate_reasoning.side_effect = reasoning_responses
        self.mock_model.generate_response.return_value = "15 * 23 = 345"
        
        # Mock tool execution
        calc_result = ToolResult(
            success=True,
            result=345,
            execution_time=0.1
        )
        self.mock_tool_interface.execute_tool.return_value = calc_result
        
        # Mock the agent to take action on first iteration, then provide final response
        with patch.object(self.agent, '_should_take_action', side_effect=[True, False]):
            with patch.object(self.agent.reasoning_engine, 'determine_next_action', return_value='calculate'):
                # Act
                response = self.agent.process_input(user_input)
        
        # Assert
        assert "345" in response or "15 * 23" in response
        
        # Verify observation processing
        observations = self.agent.get_observation_history()
        assert len(observations) >= 1
        
        latest_observation = observations[-1]
        assert latest_observation['action'] == 'calculate'
        assert latest_observation['success'] is True
        assert 'numeric_result' in latest_observation.get('details', {})
    
    def test_react_cycle_with_failed_action_recovery(self):
        """Test ReAct cycle handling of failed actions and recovery."""
        # Arrange
        user_input = "Search for information about quantum computing"
        
        # Mock model responses
        reasoning_responses = [
            "I need to search for quantum computing information.",
            "The search failed, but I can provide general information from my knowledge."
        ]
        self.mock_model.generate_reasoning.side_effect = reasoning_responses
        self.mock_model.generate_response.return_value = "Quantum computing is a field that uses quantum mechanics..."
        
        # Mock failed tool execution
        failed_result = ToolResult(
            success=False,
            error_message="Network timeout",
            execution_time=5.0
        )
        self.mock_tool_interface.execute_tool.return_value = failed_result
        
        # Mock the agent to take action on first iteration, then provide final response
        with patch.object(self.agent, '_should_take_action', side_effect=[True, False]):
            with patch.object(self.agent.reasoning_engine, 'determine_next_action', return_value='search'):
                # Act
                response = self.agent.process_input(user_input)
        
        # Assert
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Verify failure was handled gracefully
        observations = self.agent.get_observation_history()
        assert len(observations) >= 1
        
        failed_observation = observations[-1]
        assert failed_observation['success'] is False
        assert 'error_message' in failed_observation.get('details', {})
        assert failed_observation['details']['error_message'] == "Network timeout"
    
    def test_task_execution_with_multiple_actions(self):
        """Test autonomous task execution with multiple actions."""
        # Arrange
        task = "Research Python web frameworks and calculate their popularity scores"
        
        # Mock model responses for multi-step reasoning
        reasoning_responses = [
            "I need to search for Python web frameworks first.",
            "Now I need to calculate popularity scores based on the search results.",
            "I have all the information needed to provide a comprehensive answer."
        ]
        self.mock_model.generate_reasoning.side_effect = reasoning_responses
        self.mock_model.generate_response.return_value = "Based on my research, Django and Flask are the most popular..."
        
        # Mock multiple tool executions
        search_result = ToolResult(
            success=True,
            result=["Django", "Flask", "FastAPI", "Pyramid"],
            execution_time=1.0
        )
        calc_result = ToolResult(
            success=True,
            result={"Django": 85, "Flask": 78, "FastAPI": 72, "Pyramid": 45},
            execution_time=0.3
        )
        self.mock_tool_interface.execute_tool.side_effect = [search_result, calc_result]
        
        # Mock the agent to take actions on first two iterations, then provide final response
        with patch.object(self.agent, '_should_take_action', side_effect=[True, True, False]):
            with patch.object(self.agent.reasoning_engine, 'determine_next_action', side_effect=['search', 'calculate']):
                # Act
                result = self.agent.execute_task(task)
        
        # Assert
        assert isinstance(result, TaskResult)
        assert result.success is True
        assert result.execution_time >= 0
        assert len(result.steps_taken) > 0
        
        # Verify multiple actions were taken
        assert len(self.agent.state.action_history) >= 2
        assert len(self.agent.get_observation_history()) >= 2
        
        # Verify observation patterns analysis
        patterns = self.agent.analyze_observation_patterns()
        assert patterns['total_observations'] >= 2
        assert patterns['success_rate'] > 0
    
    def test_context_management_across_iterations(self):
        """Test that context is properly managed across ReAct iterations."""
        # Arrange
        user_input = "Find the latest news and then summarize the top 3 stories"
        
        # Mock model responses
        reasoning_responses = [
            "I need to search for the latest news first.",
            "Now I need to analyze and summarize the top 3 stories from the results."
        ]
        self.mock_model.generate_reasoning.side_effect = reasoning_responses
        self.mock_model.generate_response.return_value = "Here are the top 3 news stories..."
        
        # Mock tool execution
        news_result = ToolResult(
            success=True,
            result=[
                "Breaking: New AI breakthrough announced",
                "Economy shows signs of recovery",
                "Climate summit reaches agreement"
            ],
            execution_time=1.2
        )
        self.mock_tool_interface.execute_tool.return_value = news_result
        
        # Mock the agent to take action on first iteration, then provide final response
        with patch.object(self.agent, '_should_take_action', side_effect=[True, False]):
            with patch.object(self.agent.reasoning_engine, 'determine_next_action', return_value='search'):
                # Act
                response = self.agent.process_input(user_input)
        
        # Assert
        context_summary = self.agent.get_context_summary()
        
        assert context_summary['reasoning_steps'] >= 1
        assert context_summary['actions_taken'] >= 1
        assert context_summary['observations_recorded'] >= 1
        assert len(context_summary['recent_reasoning']) > 0
        assert len(context_summary['recent_actions']) > 0
        
        # Verify observation history is maintained
        observations = self.agent.get_observation_history()
        assert len(observations) >= 1
        
        # Verify context window management
        original_window_size = self.agent.get_context_window_size()
        self.agent.set_context_window_size(5)
        assert self.agent.get_context_window_size() == 5
    
    def test_observation_processor_integration(self):
        """Test that the observation processor is properly integrated."""
        # Arrange
        user_input = "Calculate the square root of 144"
        
        # Mock model responses
        self.mock_model.generate_reasoning.return_value = "I need to calculate the square root of 144."
        self.mock_model.generate_response.return_value = "The square root of 144 is 12."
        
        # Mock tool execution
        calc_result = ToolResult(
            success=True,
            result=12.0,
            execution_time=0.1
        )
        self.mock_tool_interface.execute_tool.return_value = calc_result
        
        # Mock the agent to take action on first iteration, then provide final response
        with patch.object(self.agent, '_should_take_action', side_effect=[True, False]):
            with patch.object(self.agent.reasoning_engine, 'determine_next_action', return_value='calculate'):
                # Act
                response = self.agent.process_input(user_input)
        
        # Assert
        observations = self.agent.get_observation_history()
        assert len(observations) >= 1
        
        observation = observations[-1]
        
        # Verify observation structure
        required_keys = ['action', 'success', 'execution_time', 'timestamp', 'summary', 'details', 'extracted_info']
        for key in required_keys:
            assert key in observation
        
        # Verify observation content
        assert observation['action'] == 'calculate'
        assert observation['success'] is True
        assert observation['execution_time'] == 0.1
        assert 'numeric_result' in observation['details']
        assert observation['details']['numeric_result'] == 12.0
    
    def test_state_management_and_history_tracking(self):
        """Test comprehensive state management and history tracking."""
        # Arrange
        task = "Search for Python tutorials and count the results"
        
        # Mock responses
        reasoning_responses = [
            "I'll search for Python tutorials first.",
            "Now I'll count the search results."
        ]
        self.mock_model.generate_reasoning.side_effect = reasoning_responses
        self.mock_model.generate_response.return_value = "I found 25 Python tutorials."
        
        # Mock tool executions
        search_result = ToolResult(success=True, result=["tutorial1", "tutorial2"], execution_time=0.8)
        count_result = ToolResult(success=True, result=25, execution_time=0.1)
        self.mock_tool_interface.execute_tool.side_effect = [search_result, count_result]
        
        # Mock the agent to take actions on first two iterations, then provide final response
        with patch.object(self.agent, '_should_take_action', side_effect=[True, True, False]):
            with patch.object(self.agent.reasoning_engine, 'determine_next_action', side_effect=['search', 'calculate']):
                # Act
                result = self.agent.execute_task(task)
        
        # Assert
        assert result.success is True
        
        # Test reasoning history summary
        reasoning_summary = self.agent.get_reasoning_history_summary()
        assert "Step 1:" in reasoning_summary
        assert "search" in reasoning_summary.lower()
        
        # Test action history summary
        action_summary = self.agent.get_action_history_summary()
        assert "Action 1:" in action_summary
        assert "search" in action_summary.lower()
        
        # Test observation patterns
        patterns = self.agent.analyze_observation_patterns()
        assert patterns['total_observations'] >= 1
        assert patterns['success_rate'] > 0
        assert 'common_actions' in patterns
    
    def test_context_updates_and_configuration(self):
        """Test context updates and agent configuration."""
        # Arrange
        context_updates = {
            'agent_context': {'user_preference': 'detailed_responses'},
            'observation_settings': {
                'max_length': 1000,
                'context_window_size': 15
            }
        }
        
        # Act
        self.agent.update_context(context_updates)
        
        # Assert
        assert self.agent.state.context['user_preference'] == 'detailed_responses'
        assert self.agent.get_context_window_size() == 15
        assert self.agent.observation_processor.get_max_observation_length() == 1000
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Arrange
        user_input = "Perform an impossible task"
        
        # Mock model to raise an error
        self.mock_model.generate_reasoning.side_effect = Exception("Model error")
        
        # Act & Assert
        with pytest.raises(AgentError):
            self.agent.process_input(user_input)
        
        # Verify agent state was cleaned up
        assert not self.agent.is_active()
        assert self.agent.state.current_task is None
    
    def test_stop_functionality(self):
        """Test agent stop functionality during execution."""
        # Arrange
        user_input = "Long running task"
        
        # Mock long-running model response
        def slow_reasoning(*args, **kwargs):
            self.agent.stop()  # Stop the agent during execution
            return "This is taking too long..."
        
        self.mock_model.generate_reasoning.side_effect = slow_reasoning
        self.mock_model.generate_response.return_value = "Task was stopped"
        
        # Mock the agent to provide a direct response (no actions)
        with patch.object(self.agent, '_should_take_action', return_value=False):
            # Act
            response = self.agent.process_input(user_input)
        
        # Assert
        assert not self.agent.is_active()
        assert self.agent._stop_requested is True
    
    def test_memory_integration(self):
        """Test integration with memory system."""
        # Arrange
        user_input = "Remember this: my favorite color is blue"
        
        # Mock memory responses
        self.mock_memory.get_context.return_value = {'previous_preferences': {}}
        
        # Mock model responses
        self.mock_model.generate_reasoning.return_value = "I should remember the user's preference."
        self.mock_model.generate_response.return_value = "I'll remember that your favorite color is blue."
        
        # Act
        response = self.agent.process_input(user_input)
        
        # Assert
        # Verify messages were stored in memory
        assert self.mock_memory.store_message.call_count == 2
        
        # Verify user message was stored
        user_message_call = self.mock_memory.store_message.call_args_list[0]
        user_message = user_message_call[0][0]
        assert user_message.content == user_input
        assert user_message.role == MessageRole.USER
        
        # Verify agent message was stored
        agent_message_call = self.mock_memory.store_message.call_args_list[1]
        agent_message = agent_message_call[0][0]
        assert agent_message.role == MessageRole.AGENT
    
    def test_max_iterations_handling(self):
        """Test handling of maximum iterations limit."""
        # Arrange
        user_input = "Complex multi-step task"
        
        # Mock model to always suggest taking actions (never finish)
        self.mock_model.generate_reasoning.return_value = "I need to search for more information."
        self.mock_model.generate_response.return_value = "I've reached the maximum iterations."
        
        # Mock tool execution
        search_result = ToolResult(success=True, result=["result"], execution_time=0.1)
        self.mock_tool_interface.execute_tool.return_value = search_result
        
        # Mock the agent to always take actions (never provide final response until max iterations)
        with patch.object(self.agent, '_should_take_action', return_value=True):
            with patch.object(self.agent.reasoning_engine, 'determine_next_action', return_value='search'):
                with patch.object(self.agent, '_generate_fallback_response', return_value="fallback response"):
                    # Act
                    response = self.agent.process_input(user_input)
        
        # Assert
        assert "maximum number of reasoning iterations" in response
        assert len(self.agent.state.reasoning_history) == self.agent.max_iterations
        assert len(self.agent.get_observation_history()) >= 1
    
    def test_enhanced_state_management(self):
        """Test enhanced state management features."""
        # Arrange
        user_input = "Test enhanced state management"
        
        # Mock model responses
        self.mock_model.generate_reasoning.return_value = "I need to search for information."
        self.mock_model.generate_response.return_value = "Here's the information you requested."
        
        # Mock tool execution
        search_result = ToolResult(success=True, result=["data1", "data2"], execution_time=1.5)
        self.mock_tool_interface.execute_tool.return_value = search_result
        
        # Set initial context metadata
        initial_metadata = {"user_preference": "detailed", "session_id": "test123"}
        self.agent.set_context_metadata(initial_metadata)
        
        # Mock the agent to take action on first iteration, then provide final response
        with patch.object(self.agent, '_should_take_action', side_effect=[True, False]):
            with patch.object(self.agent.reasoning_engine, 'determine_next_action', return_value='search'):
                # Act
                response = self.agent.process_input(user_input)
        
        # Assert enhanced state management features
        
        # Test context metadata
        metadata = self.agent.get_context_metadata()
        assert metadata["user_preference"] == "detailed"
        assert metadata["session_id"] == "test123"
        
        # Test performance metrics
        metrics = self.agent.get_performance_metrics()
        assert metrics['total_actions'] == 1
        assert metrics['successful_actions'] == 1
        assert metrics['failed_actions'] == 0
        assert metrics['average_action_time'] == 1.5
        
        # Test detailed state info
        state_info = self.agent.get_detailed_state_info()
        assert 'basic_state' in state_info
        assert 'history_counts' in state_info
        assert 'context_info' in state_info
        assert 'performance_metrics' in state_info
        assert 'recent_activity' in state_info
        assert 'configuration' in state_info
        
        assert state_info['history_counts']['reasoning_steps'] >= 1
        assert state_info['history_counts']['actions_taken'] >= 1
        assert state_info['history_counts']['observations_recorded'] >= 1
    
    def test_context_metadata_management(self):
        """Test context metadata management functionality."""
        # Arrange
        initial_metadata = {"theme": "dark", "language": "en"}
        updates = {"theme": "light", "timezone": "UTC"}
        
        # Act
        self.agent.set_context_metadata(initial_metadata)
        self.agent.update_context_metadata(updates)
        
        # Assert
        final_metadata = self.agent.get_context_metadata()
        assert final_metadata["theme"] == "light"  # Updated
        assert final_metadata["language"] == "en"  # Preserved
        assert final_metadata["timezone"] == "UTC"  # Added
    
    def test_performance_metrics_tracking(self):
        """Test performance metrics tracking across multiple actions."""
        # Arrange
        user_input = "Perform multiple actions for testing"
        
        # Mock model responses
        reasoning_responses = [
            "I need to search first.",
            "Now I need to calculate something.",
            "Finally, I can provide the answer."
        ]
        self.mock_model.generate_reasoning.side_effect = reasoning_responses
        self.mock_model.generate_response.return_value = "Task completed successfully."
        
        # Mock tool executions with different success rates and times
        search_result = ToolResult(success=True, result=["data"], execution_time=2.0)
        calc_result = ToolResult(success=False, error_message="Division by zero", execution_time=0.5)
        self.mock_tool_interface.execute_tool.side_effect = [search_result, calc_result]
        
        # Mock the agent to take actions on first two iterations, then provide final response
        with patch.object(self.agent, '_should_take_action', side_effect=[True, True, False]):
            with patch.object(self.agent.reasoning_engine, 'determine_next_action', side_effect=['search', 'calculate']):
                # Act
                response = self.agent.process_input(user_input)
        
        # Assert performance metrics
        metrics = self.agent.get_performance_metrics()
        assert metrics['total_actions'] == 2
        assert metrics['successful_actions'] == 1
        assert metrics['failed_actions'] == 1
        assert metrics['average_action_time'] == 1.25  # (2.0 + 0.5) / 2
        assert metrics['reasoning_steps_per_task'] >= 2
    
    def test_state_snapshot_export_import(self):
        """Test state snapshot export and import functionality."""
        # Arrange
        user_input = "Test state snapshot functionality"
        
        # Mock model responses
        self.mock_model.generate_reasoning.return_value = "I need to search for information."
        self.mock_model.generate_response.return_value = "Information found."
        
        # Mock tool execution
        search_result = ToolResult(success=True, result=["result"], execution_time=1.0)
        self.mock_tool_interface.execute_tool.return_value = search_result
        
        # Set up initial state
        self.agent.set_context_metadata({"test": "value"})
        
        # Mock the agent to take action on first iteration, then provide final response
        with patch.object(self.agent, '_should_take_action', side_effect=[True, False]):
            with patch.object(self.agent.reasoning_engine, 'determine_next_action', return_value='search'):
                # Execute to create state
                response = self.agent.process_input(user_input)
        
        # Act - Export state snapshot
        snapshot = self.agent.export_state_snapshot()
        
        # Assert snapshot structure
        assert 'timestamp' in snapshot
        assert 'agent_state' in snapshot
        assert 'observation_history' in snapshot
        assert 'context_metadata' in snapshot
        assert 'performance_metrics' in snapshot
        assert 'configuration' in snapshot
        
        # Verify snapshot content
        assert snapshot['context_metadata']['test'] == 'value'
        assert len(snapshot['observation_history']) >= 1
        assert snapshot['performance_metrics']['total_actions'] >= 1
        
        # Test import functionality
        # Create a new agent and import the snapshot
        new_agent = ReActAgent(
            model=self.mock_model,
            tool_interface=self.mock_tool_interface,
            memory=self.mock_memory
        )
        
        new_agent.import_state_snapshot(snapshot)
        
        # Assert imported state
        imported_metadata = new_agent.get_context_metadata()
        assert imported_metadata['test'] == 'value'
        
        imported_metrics = new_agent.get_performance_metrics()
        assert imported_metrics['total_actions'] >= 1
    
    def test_state_consistency_validation(self):
        """Test state consistency validation functionality."""
        # Arrange - Create agent with some state
        user_input = "Test state validation"
        
        # Mock model responses
        self.mock_model.generate_reasoning.return_value = "I need to search."
        self.mock_model.generate_response.return_value = "Search completed."
        
        # Mock tool execution
        search_result = ToolResult(success=True, result=["data"], execution_time=1.0)
        self.mock_tool_interface.execute_tool.return_value = search_result
        
        # Execute to create state
        with patch.object(self.agent, '_should_take_action', side_effect=[True, False]):
            with patch.object(self.agent.reasoning_engine, 'determine_next_action', return_value='search'):
                response = self.agent.process_input(user_input)
        
        # Act - Validate state consistency
        validation_results = self.agent.validate_state_consistency()
        
        # Assert validation results
        assert 'is_valid' in validation_results
        assert 'issues' in validation_results
        assert 'warnings' in validation_results
        
        # Should be valid with normal operation
        assert validation_results['is_valid'] is True
        
        # Test with inconsistent state
        # Manually corrupt performance metrics
        self.agent._performance_metrics['successful_actions'] = 999
        
        validation_results = self.agent.validate_state_consistency()
        assert validation_results['is_valid'] is False
        assert len(validation_results['issues']) > 0
    
    def test_context_window_management(self):
        """Test context window size management for observations."""
        # Arrange
        original_window_size = 3
        self.agent.set_context_window_size(original_window_size)
        
        # Mock multiple tool executions to generate observations
        search_results = [
            ToolResult(success=True, result=[f"result{i}"], execution_time=0.1)
            for i in range(5)  # More than window size
        ]
        
        # Simulate multiple observations
        for i, result in enumerate(search_results):
            with patch.object(self.agent.observation_processor, 'process_observation') as mock_process:
                mock_process.return_value = {
                    'action': 'search',
                    'success': True,
                    'summary': f'Search {i} completed',
                    'details': {},
                    'extracted_info': {},
                    'next_steps_suggested': []
                }
                
                # Process each action result
                self.agent._process_action_result(result, 'search')
        
        # Assert context window management
        observations = self.agent.get_observation_history()
        assert len(observations) == original_window_size  # Should be limited to window size
        
        # Test window size change
        new_window_size = 2
        self.agent.set_context_window_size(new_window_size)
        
        observations_after_resize = self.agent.get_observation_history()
        assert len(observations_after_resize) == new_window_size
        assert self.agent.get_context_window_size() == new_window_size
    
    def test_comprehensive_context_updates(self):
        """Test comprehensive context update functionality."""
        # Arrange
        context_updates = {
            'agent_context': {
                'user_id': 'user123',
                'session_type': 'interactive'
            },
            'observation_settings': {
                'max_length': 800,
                'context_window_size': 15
            }
        }
        
        # Act
        self.agent.update_context(context_updates)
        
        # Assert agent context updates
        agent_context = self.agent.state.context
        assert agent_context['user_id'] == 'user123'
        assert agent_context['session_type'] == 'interactive'
        
        # Assert observation settings updates
        assert self.agent.get_context_window_size() == 15
        assert self.agent.observation_processor.get_max_observation_length() == 800
    
    def test_reasoning_and_action_history_summaries(self):
        """Test reasoning and action history summary generation."""
        # Arrange
        user_input = "Test history summaries"
        
        # Mock model responses
        reasoning_responses = [
            "First, I need to understand the problem by searching for relevant information.",
            "Based on the search results, I should calculate the optimal solution.",
            "Now I can provide a comprehensive answer to the user."
        ]
        self.mock_model.generate_reasoning.side_effect = reasoning_responses
        self.mock_model.generate_response.return_value = "Here's your comprehensive answer."
        
        # Mock tool executions
        search_result = ToolResult(success=True, result=["info"], execution_time=1.2)
        calc_result = ToolResult(success=True, result=42, execution_time=0.3)
        self.mock_tool_interface.execute_tool.side_effect = [search_result, calc_result]
        
        # Mock the agent to take actions on first two iterations, then provide final response
        with patch.object(self.agent, '_should_take_action', side_effect=[True, True, False]):
            with patch.object(self.agent.reasoning_engine, 'determine_next_action', side_effect=['search', 'calculate']):
                # Act
                response = self.agent.process_input(user_input)
        
        # Assert reasoning history summary
        reasoning_summary = self.agent.get_reasoning_history_summary()
        assert "Step 1:" in reasoning_summary
        assert "Step 2:" in reasoning_summary
        assert "Step 3:" in reasoning_summary
        # The reasoning contains the observations from actions
        assert "search" in reasoning_summary.lower()
        assert "action:" in reasoning_summary.lower()
        
        # Assert action history summary
        action_summary = self.agent.get_action_history_summary()
        assert "Action 1:" in action_summary
        assert "Action 2:" in action_summary
        assert "search" in action_summary.lower()
        assert "calculate" in action_summary.lower()
        # Check that execution times are recorded (format: (X.XXs))
        assert "(" in action_summary and "s)" in action_summary  # execution time format
    
    def test_performance_metrics_reset(self):
        """Test performance metrics reset functionality."""
        # Arrange - Execute some actions to generate metrics
        user_input = "Generate some metrics"
        
        # Mock model and tool responses
        self.mock_model.generate_reasoning.return_value = "I need to search."
        self.mock_model.generate_response.return_value = "Search completed."
        search_result = ToolResult(success=True, result=["data"], execution_time=1.0)
        self.mock_tool_interface.execute_tool.return_value = search_result
        
        # Execute to generate metrics
        with patch.object(self.agent, '_should_take_action', side_effect=[True, False]):
            with patch.object(self.agent.reasoning_engine, 'determine_next_action', return_value='search'):
                response = self.agent.process_input(user_input)
        
        # Verify metrics were generated
        metrics_before = self.agent.get_performance_metrics()
        assert metrics_before['total_actions'] > 0
        
        # Act - Reset metrics
        self.agent.clear_performance_metrics()
        
        # Assert metrics were reset
        metrics_after = self.agent.get_performance_metrics()
        assert metrics_after['total_actions'] == 0
        assert metrics_after['successful_actions'] == 0
        assert metrics_after['failed_actions'] == 0
        assert metrics_after['average_action_time'] == 0.0
        assert metrics_after['reasoning_steps_per_task'] == 0.0