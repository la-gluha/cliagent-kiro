"""
Agent interface definition for the AI Agent System.

This module defines the abstract interface that all agent implementations
must follow to ensure consistent behavior across different agent types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ..models.data_models import TaskResult, AgentState


class AgentInterface(ABC):
    """
    Abstract interface for agent implementations in the AI Agent System.
    
    This interface defines the contract for processing user input, executing tasks,
    and managing agent state. Implementations can use different reasoning approaches
    (ReAct, Chain-of-Thought, etc.).
    """
    
    @abstractmethod
    def process_input(self, user_input: str) -> str:
        """
        Process user input and generate a response.
        
        Args:
            user_input: The input from the user
            
        Returns:
            The agent's response as a string
            
        Raises:
            AgentError: If input processing fails
        """
        pass
    
    @abstractmethod
    def execute_task(self, task: str) -> TaskResult:
        """
        Execute a task autonomously.
        
        Args:
            task: Description of the task to execute
            
        Returns:
            TaskResult containing the execution outcome
            
        Raises:
            AgentError: If task execution fails
        """
        pass
    
    @abstractmethod
    def get_available_actions(self) -> List[str]:
        """
        Get a list of actions the agent can perform.
        
        Returns:
            List of available action names
        """
        pass
    
    @abstractmethod
    def reset_context(self) -> None:
        """
        Reset the agent's context and state.
        
        Raises:
            AgentError: If context reset fails
        """
        pass
    
    @abstractmethod
    def get_state(self) -> AgentState:
        """
        Get the current state of the agent.
        
        Returns:
            AgentState object representing current state
        """
        pass
    
    @abstractmethod
    def set_state(self, state: AgentState) -> None:
        """
        Set the agent's state.
        
        Args:
            state: The AgentState to set
            
        Raises:
            AgentError: If state cannot be set
        """
        pass
    
    @abstractmethod
    def is_active(self) -> bool:
        """
        Check if the agent is currently active/processing.
        
        Returns:
            True if the agent is active, False otherwise
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """
        Stop the agent's current operation.
        
        Raises:
            AgentError: If the agent cannot be stopped
        """
        pass