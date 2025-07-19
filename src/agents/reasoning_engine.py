"""
Reasoning engine for the ReAct agent system.

This module implements the reasoning component of the ReAct cycle,
responsible for generating step-by-step thinking and problem analysis.
"""

from typing import Any, Dict, List, Optional
from ..interfaces.model_interface import ModelInterface
from ..models.data_models import AgentState
from ..exceptions import AgentError, ModelError


class ReasoningEngine:
    """
    Engine for generating reasoning steps in the ReAct cycle.
    
    The ReasoningEngine uses a language model to generate step-by-step
    thinking about problems and determine appropriate actions to take.
    """
    
    def __init__(self, model: ModelInterface):
        """
        Initialize the reasoning engine.
        
        Args:
            model: The language model interface to use for reasoning
        """
        self.model = model
        self._reasoning_prompt_template = """
You are an AI agent using the ReAct (Reasoning and Acting) framework. 
Your task is to think step by step about the given problem and determine what action to take next.

Current situation: {situation}
Available actions: {available_actions}
Previous reasoning steps: {previous_steps}
Context: {context}

Think through this step by step:
1. What is the current problem or goal?
2. What information do I have?
3. What information do I need?
4. What action should I take next?
5. Why is this the best action to take?

Provide your reasoning in a clear, step-by-step format.
"""
    
    def generate_reasoning(
        self, 
        situation: str, 
        available_actions: List[str],
        agent_state: AgentState,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate reasoning steps for the current situation.
        
        Args:
            situation: Description of the current situation or problem
            available_actions: List of actions the agent can take
            agent_state: Current state of the agent
            context: Additional context information
            
        Returns:
            Generated reasoning steps as a string
            
        Raises:
            AgentError: If reasoning generation fails
        """
        try:
            # Prepare the prompt
            previous_steps = "\n".join(agent_state.reasoning_history[-3:])  # Last 3 steps
            actions_str = ", ".join(available_actions)
            context_str = str(context) if context else "No additional context"
            
            prompt = self._reasoning_prompt_template.format(
                situation=situation,
                available_actions=actions_str,
                previous_steps=previous_steps or "No previous steps",
                context=context_str
            )
            
            # Generate reasoning using the model
            reasoning = self.model.generate_reasoning(prompt, context)
            
            if not reasoning or not reasoning.strip():
                raise AgentError("Generated reasoning is empty")
            
            return reasoning.strip()
            
        except ModelError as e:
            raise AgentError(f"Failed to generate reasoning: {e}")
        except Exception as e:
            raise AgentError(f"Unexpected error during reasoning generation: {e}")
    
    def analyze_problem(
        self, 
        problem: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a problem and break it down into components.
        
        Args:
            problem: The problem to analyze
            context: Additional context information
            
        Returns:
            Dictionary containing problem analysis
            
        Raises:
            AgentError: If problem analysis fails
        """
        try:
            analysis_prompt = f"""
Analyze the following problem and break it down:

Problem: {problem}
Context: {context or 'No additional context'}

Provide analysis in the following format:
- Goal: What needs to be accomplished?
- Requirements: What are the key requirements?
- Constraints: What limitations exist?
- Approach: What general approach should be taken?
- Next Steps: What are the immediate next steps?
"""
            
            analysis = self.model.generate_reasoning(analysis_prompt, context)
            
            # Parse the analysis into structured format
            # This is a simplified parser - could be enhanced with more sophisticated parsing
            analysis_dict = {
                "raw_analysis": analysis,
                "goal": self._extract_section(analysis, "Goal:"),
                "requirements": self._extract_section(analysis, "Requirements:"),
                "constraints": self._extract_section(analysis, "Constraints:"),
                "approach": self._extract_section(analysis, "Approach:"),
                "next_steps": self._extract_section(analysis, "Next Steps:")
            }
            
            return analysis_dict
            
        except ModelError as e:
            raise AgentError(f"Failed to analyze problem: {e}")
        except Exception as e:
            raise AgentError(f"Unexpected error during problem analysis: {e}")
    
    def determine_next_action(
        self, 
        reasoning: str, 
        available_actions: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Determine the next action to take based on reasoning.
        
        Args:
            reasoning: The reasoning steps generated
            available_actions: List of available actions
            context: Additional context information
            
        Returns:
            The name of the next action to take
            
        Raises:
            AgentError: If action determination fails
        """
        try:
            action_prompt = f"""
Based on the following reasoning, determine the best next action to take:

Reasoning: {reasoning}
Available actions: {', '.join(available_actions)}
Context: {context or 'No additional context'}

Choose ONE action from the available actions list and explain why it's the best choice.
Respond with just the action name.
"""
            
            response = self.model.generate_response(action_prompt, context)
            
            # Extract action name from response
            action = response.strip().lower()
            
            # Find matching action (case-insensitive)
            for available_action in available_actions:
                if available_action.lower() in action or action in available_action.lower():
                    return available_action
            
            # If no exact match, return the first available action as fallback
            if available_actions:
                return available_actions[0]
            
            raise AgentError("No suitable action could be determined")
            
        except ModelError as e:
            raise AgentError(f"Failed to determine next action: {e}")
        except Exception as e:
            raise AgentError(f"Unexpected error during action determination: {e}")
    
    def _extract_section(self, text: str, section_header: str) -> str:
        """
        Extract a section from structured text.
        
        Args:
            text: The text to extract from
            section_header: The header to look for
            
        Returns:
            The extracted section content
        """
        lines = text.split('\n')
        section_content = []
        in_section = False
        
        for line in lines:
            if section_header in line:
                in_section = True
                # Add content after the header on the same line
                content_after_header = line.split(section_header, 1)[1].strip()
                if content_after_header:
                    section_content.append(content_after_header)
            elif in_section:
                # Stop if we hit another section header
                if ':' in line and line.strip().endswith(':'):
                    break
                section_content.append(line.strip())
        
        return '\n'.join(section_content).strip()
    
    def validate_reasoning(self, reasoning: str) -> bool:
        """
        Validate that reasoning meets quality standards.
        
        Args:
            reasoning: The reasoning to validate
            
        Returns:
            True if reasoning is valid, False otherwise
        """
        if not reasoning or not reasoning.strip():
            return False
        
        # Check for minimum length
        if len(reasoning.strip()) < 10:
            return False
        
        # Check for some structure (basic heuristics)
        reasoning_lower = reasoning.lower()
        has_thinking_words = any(word in reasoning_lower for word in 
                               ['think', 'consider', 'analyze', 'because', 'therefore', 'since'])
        
        return has_thinking_words