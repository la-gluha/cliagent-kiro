"""
Unit tests for the ReasoningEngine class.

This module contains comprehensive tests for the reasoning engine component
of the ReAct agent system.
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.agents.reasoning_engine import ReasoningEngine
from src.models.data_models import AgentState
from src.exceptions import AgentError, ModelError


class TestReasoningEngine:
    """Test cases for the ReasoningEngine class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_model = Mock()
        self.reasoning_engine = ReasoningEngine(self.mock_model)
        self.agent_state = AgentState()
    
    def test_init(self):
        """Test ReasoningEngine initialization."""
        assert self.reasoning_engine.model == self.mock_model
        assert hasattr(self.reasoning_engine, '_reasoning_prompt_template')
    
    def test_generate_reasoning_success(self):
        """Test successful reasoning generation."""
        # Arrange
        situation = "I need to find information about Python"
        available_actions = ["search", "calculate"]
        expected_reasoning = "I should search for Python information first."
        
        self.mock_model.generate_reasoning.return_value = expected_reasoning
        
        # Act
        result = self.reasoning_engine.generate_reasoning(
            situation=situation,
            available_actions=available_actions,
            agent_state=self.agent_state
        )
        
        # Assert
        assert result == expected_reasoning
        self.mock_model.generate_reasoning.assert_called_once()
        
        # Check that the prompt was properly formatted
        call_args = self.mock_model.generate_reasoning.call_args
        prompt = call_args[0][0]
        assert situation in prompt
        assert "search, calculate" in prompt
    
    def test_generate_reasoning_with_context(self):
        """Test reasoning generation with additional context."""
        # Arrange
        situation = "Calculate the area of a circle"
        available_actions = ["calculate"]
        context = {"radius": 5}
        expected_reasoning = "I need to use the formula π * r²"
        
        self.mock_model.generate_reasoning.return_value = expected_reasoning
        
        # Act
        result = self.reasoning_engine.generate_reasoning(
            situation=situation,
            available_actions=available_actions,
            agent_state=self.agent_state,
            context=context
        )
        
        # Assert
        assert result == expected_reasoning
        call_args = self.mock_model.generate_reasoning.call_args
        assert call_args[0][1] == context  # Context passed as second positional argument
    
    def test_generate_reasoning_with_previous_steps(self):
        """Test reasoning generation with previous reasoning history."""
        # Arrange
        situation = "Continue the analysis"
        available_actions = ["search"]
        self.agent_state.add_reasoning_step("Step 1: Analyzed the problem")
        self.agent_state.add_reasoning_step("Step 2: Identified requirements")
        expected_reasoning = "Step 3: Now I should search for more information"
        
        self.mock_model.generate_reasoning.return_value = expected_reasoning
        
        # Act
        result = self.reasoning_engine.generate_reasoning(
            situation=situation,
            available_actions=available_actions,
            agent_state=self.agent_state
        )
        
        # Assert
        assert result == expected_reasoning
        call_args = self.mock_model.generate_reasoning.call_args
        prompt = call_args[0][0]
        assert "Step 1: Analyzed the problem" in prompt
        assert "Step 2: Identified requirements" in prompt
    
    def test_generate_reasoning_model_error(self):
        """Test reasoning generation when model raises an error."""
        # Arrange
        self.mock_model.generate_reasoning.side_effect = ModelError("Model unavailable")
        
        # Act & Assert
        with pytest.raises(AgentError, match="Failed to generate reasoning"):
            self.reasoning_engine.generate_reasoning(
                situation="test",
                available_actions=["test"],
                agent_state=self.agent_state
            )
    
    def test_generate_reasoning_empty_response(self):
        """Test reasoning generation when model returns empty response."""
        # Arrange
        self.mock_model.generate_reasoning.return_value = ""
        
        # Act & Assert
        with pytest.raises(AgentError, match="Generated reasoning is empty"):
            self.reasoning_engine.generate_reasoning(
                situation="test",
                available_actions=["test"],
                agent_state=self.agent_state
            )
    
    def test_analyze_problem_success(self):
        """Test successful problem analysis."""
        # Arrange
        problem = "How to optimize database queries"
        expected_analysis = """
Goal: Optimize database performance
Requirements: Identify slow queries
Constraints: Limited downtime
Approach: Use query analysis tools
Next Steps: Profile current queries
"""
        self.mock_model.generate_reasoning.return_value = expected_analysis
        
        # Act
        result = self.reasoning_engine.analyze_problem(problem)
        
        # Assert
        assert isinstance(result, dict)
        assert "raw_analysis" in result
        assert "goal" in result
        assert "requirements" in result
        assert "constraints" in result
        assert "approach" in result
        assert "next_steps" in result
        assert result["raw_analysis"] == expected_analysis
    
    def test_analyze_problem_with_context(self):
        """Test problem analysis with context."""
        # Arrange
        problem = "Improve system performance"
        context = {"current_load": "high", "resources": "limited"}
        expected_analysis = "Analysis with context"
        
        self.mock_model.generate_reasoning.return_value = expected_analysis
        
        # Act
        result = self.reasoning_engine.analyze_problem(problem, context)
        
        # Assert
        call_args = self.mock_model.generate_reasoning.call_args
        prompt = call_args[0][0]
        assert str(context) in prompt
    
    def test_determine_next_action_success(self):
        """Test successful action determination."""
        # Arrange
        reasoning = "I need to search for information about the topic"
        available_actions = ["search", "calculate", "file_read"]
        expected_response = "search"
        
        self.mock_model.generate_response.return_value = expected_response
        
        # Act
        result = self.reasoning_engine.determine_next_action(
            reasoning=reasoning,
            available_actions=available_actions
        )
        
        # Assert
        assert result == "search"
    
    def test_determine_next_action_partial_match(self):
        """Test action determination with partial string matching."""
        # Arrange
        reasoning = "I should use the search functionality"
        available_actions = ["web_search", "calculate", "file_read"]
        expected_response = "I should use the search functionality"
        
        self.mock_model.generate_response.return_value = expected_response
        
        # Act
        result = self.reasoning_engine.determine_next_action(
            reasoning=reasoning,
            available_actions=available_actions
        )
        
        # Assert
        assert result == "web_search"  # Should match "search" in "web_search"
    
    def test_determine_next_action_fallback(self):
        """Test action determination fallback to first available action."""
        # Arrange
        reasoning = "I need to do something"
        available_actions = ["action1", "action2", "action3"]
        expected_response = "unknown_action"
        
        self.mock_model.generate_response.return_value = expected_response
        
        # Act
        result = self.reasoning_engine.determine_next_action(
            reasoning=reasoning,
            available_actions=available_actions
        )
        
        # Assert
        assert result == "action1"  # Should fallback to first action
    
    def test_determine_next_action_no_actions(self):
        """Test action determination with no available actions."""
        # Arrange
        reasoning = "I need to do something"
        available_actions = []
        
        self.mock_model.generate_response.return_value = "any response"
        
        # Act & Assert
        with pytest.raises(AgentError, match="No suitable action could be determined"):
            self.reasoning_engine.determine_next_action(
                reasoning=reasoning,
                available_actions=available_actions
            )
    
    def test_extract_section_success(self):
        """Test successful section extraction from text."""
        # Arrange
        text = """
Some intro text
Goal: Achieve the objective
This is the goal content
Requirements: List of requirements
This is requirements content
Constraints: System limitations
"""
        
        # Act
        goal = self.reasoning_engine._extract_section(text, "Goal:")
        requirements = self.reasoning_engine._extract_section(text, "Requirements:")
        
        # Assert
        assert "Achieve the objective" in goal
        assert "This is the goal content" in goal
        assert "List of requirements" in requirements
        assert "This is requirements content" in requirements
    
    def test_extract_section_not_found(self):
        """Test section extraction when section is not found."""
        # Arrange
        text = "Some text without the section"
        
        # Act
        result = self.reasoning_engine._extract_section(text, "Goal:")
        
        # Assert
        assert result == ""
    
    def test_validate_reasoning_valid(self):
        """Test validation of valid reasoning."""
        # Arrange
        valid_reasoning = "I think we should consider this approach because it makes sense"
        
        # Act
        result = self.reasoning_engine.validate_reasoning(valid_reasoning)
        
        # Assert
        assert result is True
    
    def test_validate_reasoning_invalid_empty(self):
        """Test validation of empty reasoning."""
        # Act & Assert
        assert self.reasoning_engine.validate_reasoning("") is False
        assert self.reasoning_engine.validate_reasoning("   ") is False
        assert self.reasoning_engine.validate_reasoning(None) is False
    
    def test_validate_reasoning_invalid_too_short(self):
        """Test validation of too short reasoning."""
        # Arrange
        short_reasoning = "Yes"
        
        # Act
        result = self.reasoning_engine.validate_reasoning(short_reasoning)
        
        # Assert
        assert result is False
    
    def test_validate_reasoning_invalid_no_thinking_words(self):
        """Test validation of reasoning without thinking words."""
        # Arrange
        non_thinking_reasoning = "This is just a statement without reasoning words"
        
        # Act
        result = self.reasoning_engine.validate_reasoning(non_thinking_reasoning)
        
        # Assert
        assert result is False
    
    def test_validate_reasoning_valid_with_thinking_words(self):
        """Test validation of reasoning with various thinking words."""
        thinking_words = ["think", "consider", "analyze", "because", "therefore", "since"]
        
        for word in thinking_words:
            # Arrange
            reasoning = f"I {word} this is a good approach for solving the problem"
            
            # Act
            result = self.reasoning_engine.validate_reasoning(reasoning)
            
            # Assert
            assert result is True, f"Failed for thinking word: {word}"