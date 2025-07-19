"""
Unit tests for the ObservationProcessor class.

This module contains comprehensive tests for the observation processor component
of the ReAct agent system.
"""

import pytest
from unittest.mock import Mock, patch
from src.agents.observation_processor import ObservationProcessor
from src.interfaces.tool_interface import ToolResult
from src.models.data_models import AgentState
from src.exceptions import AgentError


class TestObservationProcessor:
    """Test cases for the ObservationProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.processor = ObservationProcessor()
        self.agent_state = AgentState()
    
    def test_init(self):
        """Test ObservationProcessor initialization."""
        assert self.processor._max_observation_length == 500
        assert isinstance(self.processor._key_patterns, dict)
        assert 'error' in self.processor._key_patterns
        assert 'success' in self.processor._key_patterns
    
    def test_process_successful_observation(self):
        """Test processing a successful action result."""
        # Arrange
        action_result = ToolResult(
            success=True,
            result=["Python tutorial", "Django guide", "Flask documentation"],
            execution_time=1.2
        )
        action_name = "search"
        
        # Act
        observation = self.processor.process_observation(
            action_result=action_result,
            action_name=action_name,
            agent_state=self.agent_state
        )
        
        # Assert
        assert observation['action'] == action_name
        assert observation['success'] is True
        assert observation['execution_time'] == 1.2
        assert 'timestamp' in observation
        assert 'summary' in observation
        assert 'details' in observation
        assert 'extracted_info' in observation
        assert 'next_steps_suggested' in observation
        
        # Check details for search action
        details = observation['details']
        assert details['result_type'] == 'list'
        assert details['result_size'] == 3
        assert details['item_count'] == 3
        assert details['has_results'] is True
    
    def test_process_failed_observation(self):
        """Test processing a failed action result."""
        # Arrange
        action_result = ToolResult(
            success=False,
            error_message="Network timeout occurred",
            execution_time=5.0
        )
        action_name = "search"
        
        # Act
        observation = self.processor.process_observation(
            action_result=action_result,
            action_name=action_name,
            agent_state=self.agent_state
        )
        
        # Assert
        assert observation['success'] is False
        assert observation['execution_time'] == 5.0
        
        details = observation['details']
        assert details['error_message'] == "Network timeout occurred"
        assert details['error_type'] == 'network'
        assert details['is_recoverable'] is True
    
    def test_process_calculation_observation(self):
        """Test processing a calculation action result."""
        # Arrange
        action_result = ToolResult(
            success=True,
            result=42.5,
            execution_time=0.1
        )
        action_name = "calculate"
        
        # Act
        observation = self.processor.process_observation(
            action_result=action_result,
            action_name=action_name,
            agent_state=self.agent_state
        )
        
        # Assert
        details = observation['details']
        # The calculation-specific processing overwrites result_type to 'number'
        assert details['numeric_result'] == 42.5
        assert details['result_type'] == 'number'
    
    def test_extract_key_information(self):
        """Test key information extraction from text."""
        # Arrange
        text = "Found 25 results with 3 errors. Visit https://example.com for more info. File: data.json"
        
        # Act
        extracted = self.processor._extract_key_information(text)
        
        # Assert
        assert 'error' in extracted
        assert 'data' in extracted
        assert 'count' in extracted
        assert 'url' in extracted
        assert 'file' in extracted
        assert 'number' in extracted
        
        # Check specific extractions
        assert any('error' in str(item).lower() for item in extracted['error'])
        assert 'https://example.com' in extracted['url']
        assert 'data.json' in extracted['file']
    
    def test_format_observation_for_reasoning(self):
        """Test formatting observation for reasoning prompts."""
        # Arrange
        observation = {
            'action': 'search',
            'success': True,
            'summary': 'Search completed successfully with 5 items',
            'execution_time': 1.5,
            'extracted_info': {
                'data': ['results', 'information'],
                'count': ['5 results']
            },
            'next_steps_suggested': ['analyze_results', 'filter_data']
        }
        
        # Act
        formatted = self.processor.format_observation_for_reasoning(observation)
        
        # Assert
        assert 'Action: search - SUCCESS' in formatted
        assert 'Summary: Search completed successfully with 5 items' in formatted
        assert 'Key Info:' in formatted
        assert 'data: ' in formatted
        assert 'count: ' in formatted
        assert 'Execution Time: 1.5s' in formatted
        assert 'Suggested Next Steps:' in formatted
    
    def test_format_observation_for_reasoning_failed(self):
        """Test formatting failed observation for reasoning."""
        # Arrange
        observation = {
            'action': 'search',
            'success': False,
            'summary': 'Search failed due to network error',
            'execution_time': 0.5,
            'extracted_info': {},
            'next_steps_suggested': ['retry_action']
        }
        
        # Act
        formatted = self.processor.format_observation_for_reasoning(observation)
        
        # Assert
        assert 'Action: search - FAILED' in formatted
        assert 'Summary: Search failed due to network error' in formatted
        assert 'Suggested Next Steps: retry_action' in formatted
    
    def test_analyze_observation_patterns(self):
        """Test analysis of observation patterns."""
        # Arrange
        observations = [
            {
                'action': 'search',
                'success': True,
                'execution_time': 1.0
            },
            {
                'action': 'search',
                'success': False,
                'execution_time': 2.0,
                'details': {'error_type': 'network'}
            },
            {
                'action': 'calculate',
                'success': True,
                'execution_time': 0.5
            }
        ]
        
        # Act
        analysis = self.processor.analyze_observation_patterns(observations)
        
        # Assert
        assert analysis['total_observations'] == 3
        assert analysis['success_rate'] == 2/3  # 2 out of 3 successful
        assert analysis['common_actions']['search'] == 2
        assert analysis['common_actions']['calculate'] == 1
        assert abs(analysis['average_execution_time'] - 1.1667) < 0.001  # (1.0 + 2.0 + 0.5) / 3
        assert len(analysis['error_patterns']) >= 0
        assert len(analysis['success_patterns']) >= 0
    
    def test_analyze_empty_observations(self):
        """Test analysis with empty observation list."""
        # Act
        analysis = self.processor.analyze_observation_patterns([])
        
        # Assert
        assert analysis['total_observations'] == 0
        assert analysis['success_rate'] == 0.0
        assert analysis['common_actions'] == {}
        assert analysis['average_execution_time'] == 0.0
    
    def test_classify_error_types(self):
        """Test error type classification."""
        # Test cases: (error_message, expected_type)
        test_cases = [
            ("Network connection timeout", "network"),
            ("Permission denied", "permission"),
            ("File not found", "not_found"),
            ("Invalid input format", "validation"),
            ("API quota exceeded", "quota"),
            ("Unknown system error", "unknown")
        ]
        
        for error_msg, expected_type in test_cases:
            # Act
            error_type = self.processor._classify_error(error_msg)
            
            # Assert
            assert error_type == expected_type, f"Failed for: {error_msg}"
    
    def test_is_error_recoverable(self):
        """Test error recoverability assessment."""
        # Test cases: (error_message, expected_recoverable)
        test_cases = [
            ("Network timeout", True),
            ("API quota exceeded", True),
            ("Permission denied", False),
            ("File not found", False),
            ("Invalid syntax", False)
        ]
        
        for error_msg, expected_recoverable in test_cases:
            # Act
            is_recoverable = self.processor._is_error_recoverable(error_msg)
            
            # Assert
            assert is_recoverable == expected_recoverable, f"Failed for: {error_msg}"
    
    def test_suggest_next_steps_successful_search(self):
        """Test next step suggestions for successful search."""
        # Arrange
        observation = {'success': True}
        action_name = "search"
        
        # Act
        suggestions = self.processor._suggest_next_steps(observation, action_name, self.agent_state)
        
        # Assert
        assert len(suggestions) <= 5
        assert 'analyze_results' in suggestions
        assert 'filter_data' in suggestions
        assert 'extract_details' in suggestions
    
    def test_suggest_next_steps_failed_recoverable(self):
        """Test next step suggestions for failed but recoverable action."""
        # Arrange
        observation = {
            'success': False,
            'details': {
                'error_type': 'network',
                'is_recoverable': True
            }
        }
        action_name = "search"
        
        # Act
        suggestions = self.processor._suggest_next_steps(observation, action_name, self.agent_state)
        
        # Assert
        assert len(suggestions) <= 5
        assert 'retry_action' in suggestions
        assert 'adjust_parameters' in suggestions
        assert 'try_alternative' in suggestions
    
    def test_suggest_next_steps_failed_non_recoverable(self):
        """Test next step suggestions for failed non-recoverable action."""
        # Arrange
        observation = {
            'success': False,
            'details': {
                'error_type': 'permission',
                'is_recoverable': False
            }
        }
        action_name = "file_read"
        
        # Act
        suggestions = self.processor._suggest_next_steps(observation, action_name, self.agent_state)
        
        # Assert
        assert len(suggestions) <= 5
        assert 'report_error' in suggestions
        assert 'try_different_approach' in suggestions
        assert 'seek_help' in suggestions
    
    def test_get_result_size(self):
        """Test result size calculation."""
        # Test cases: (result, expected_size)
        test_cases = [
            (None, 0),
            ("hello", 5),
            ([1, 2, 3], 3),
            ({"a": 1, "b": 2}, 2),
            (42, 1),
            ([], 0)
        ]
        
        for result, expected_size in test_cases:
            # Act
            size = self.processor._get_result_size(result)
            
            # Assert
            assert size == expected_size, f"Failed for: {result}"
    
    def test_process_search_result(self):
        """Test search-specific result processing."""
        # Test list result
        list_result = ["item1", "item2", "item3"]
        details = self.processor._process_search_result(list_result)
        assert details['item_count'] == 3
        assert details['has_results'] is True
        
        # Test string result
        string_result = "Found some information"
        details = self.processor._process_search_result(string_result)
        assert details['content_length'] == len(string_result)
        assert details['has_content'] is True
        
        # Test empty result
        empty_result = []
        details = self.processor._process_search_result(empty_result)
        assert details['item_count'] == 0
        assert details['has_results'] is False
    
    def test_process_calculation_result(self):
        """Test calculation-specific result processing."""
        # Test numeric result
        numeric_result = 42.5
        details = self.processor._process_calculation_result(numeric_result)
        assert details['numeric_result'] == 42.5
        assert details['result_type'] == 'number'
        
        # Test string with numbers
        string_result = "The answer is 42.5 and also 10"
        details = self.processor._process_calculation_result(string_result)
        assert details['extracted_numbers'] == [42.5, 10.0]
    
    def test_process_data_result(self):
        """Test data loading/reading result processing."""
        # Test dictionary result
        dict_result = {"name": "John", "age": 30, "city": "NYC"}
        details = self.processor._process_data_result(dict_result)
        assert details['data_structure'] == 'dictionary'
        assert set(details['data_keys']) == {"name", "age", "city"}
        
        # Test list result
        list_result = [1, 2, 3, 4, 5]
        details = self.processor._process_data_result(list_result)
        assert details['data_structure'] == 'list'
        assert details['data_length'] == 5
        
        # Test string result
        string_result = "Some text content"
        details = self.processor._process_data_result(string_result)
        assert details['data_structure'] == 'text'
        assert details['content_length'] == len(string_result)
    
    def test_identify_error_patterns(self):
        """Test error pattern identification."""
        # Arrange
        failed_observations = [
            {'details': {'error_type': 'network'}},
            {'details': {'error_type': 'network'}},
            {'details': {'error_type': 'permission'}},
            {'details': {'error_type': 'quota'}}
        ]
        
        # Act
        patterns = self.processor._identify_error_patterns(failed_observations)
        
        # Assert
        assert len(patterns) >= 1
        assert any('network' in pattern for pattern in patterns)
        assert any('2 occurrences' in pattern for pattern in patterns)
    
    def test_identify_success_patterns(self):
        """Test success pattern identification."""
        # Arrange
        successful_observations = [
            {'action': 'search'},
            {'action': 'search'},
            {'action': 'calculate'},
            {'action': 'search'}
        ]
        
        # Act
        patterns = self.processor._identify_success_patterns(successful_observations)
        
        # Assert
        assert len(patterns) >= 1
        assert any('search' in pattern for pattern in patterns)
        assert any('3 successes' in pattern for pattern in patterns)
    
    def test_set_max_observation_length(self):
        """Test setting maximum observation length."""
        # Act
        self.processor.set_max_observation_length(1000)
        
        # Assert
        assert self.processor.get_max_observation_length() == 1000
    
    def test_set_invalid_max_observation_length(self):
        """Test setting invalid maximum observation length."""
        # Act & Assert
        with pytest.raises(ValueError, match="Maximum observation length must be positive"):
            self.processor.set_max_observation_length(-1)
    
    def test_process_observation_error_handling(self):
        """Test error handling in observation processing."""
        # Arrange
        action_result = ToolResult(success=True, result="test")
        action_name = "test_action"
        
        # Mock timestamp method to raise an error
        with patch.object(self.processor, '_get_current_timestamp', side_effect=Exception("Time error")):
            # Act & Assert
            with pytest.raises(AgentError, match="Failed to process observation"):
                self.processor.process_observation(action_result, action_name)
    
    def test_format_observation_error_handling(self):
        """Test error handling in observation formatting."""
        # Arrange
        invalid_observation = {"invalid": "structure"}
        
        # Act
        formatted = self.processor.format_observation_for_reasoning(invalid_observation)
        
        # Assert
        assert "Error formatting observation" in formatted