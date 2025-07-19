"""
Action executor for the ReAct agent system.

This module implements the action execution component of the ReAct cycle,
responsible for invoking tools and processing their results.
"""

import time
from typing import Any, Dict, List, Optional
from ..interfaces.tool_interface import ToolInterface, ToolResult
from ..models.data_models import AgentState
from ..exceptions import AgentError, ToolExecutionError


class ActionExecutor:
    """
    Executor for actions in the ReAct cycle.
    
    The ActionExecutor manages tool invocation, parameter validation,
    and result processing for agent actions.
    """
    
    def __init__(self, tool_interface: ToolInterface):
        """
        Initialize the action executor.
        
        Args:
            tool_interface: The tool interface for executing actions
        """
        self.tool_interface = tool_interface
        self._execution_timeout = 30.0  # Default timeout in seconds
    
    def execute_action(
        self, 
        action_name: str, 
        parameters: Optional[Dict[str, Any]] = None,
        agent_state: Optional[AgentState] = None
    ) -> ToolResult:
        """
        Execute an action with the given parameters.
        
        Args:
            action_name: Name of the action/tool to execute
            parameters: Parameters to pass to the action
            agent_state: Current agent state for context
            
        Returns:
            ToolResult containing the execution outcome
            
        Raises:
            AgentError: If action execution fails
        """
        if parameters is None:
            parameters = {}
        
        try:
            # Validate that the tool exists and is available
            if not self.tool_interface.is_tool_available(action_name):
                raise AgentError(f"Action '{action_name}' is not available")
            
            # Validate parameters
            if not self.tool_interface.validate_tool_input(action_name, parameters):
                raise AgentError(f"Invalid parameters for action '{action_name}': {parameters}")
            
            # Record the action attempt in agent state
            if agent_state:
                action_record = {
                    "action": action_name,
                    "parameters": parameters,
                    "timestamp": time.time(),
                    "status": "executing"
                }
                agent_state.add_action(action_record)
            
            # Execute the tool
            start_time = time.time()
            result = self.tool_interface.execute_tool(action_name, parameters)
            execution_time = time.time() - start_time
            
            # Update the action record with results
            if agent_state and agent_state.action_history:
                agent_state.action_history[-1].update({
                    "status": "completed" if result.success else "failed",
                    "execution_time": execution_time,
                    "result_summary": str(result.result)[:100] if result.result else None,
                    "error": result.error_message
                })
            
            return result
            
        except ToolExecutionError as e:
            error_msg = f"Tool execution failed for '{action_name}': {e}"
            if agent_state and agent_state.action_history:
                agent_state.action_history[-1].update({
                    "status": "failed",
                    "error": str(e)
                })
            raise AgentError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error executing action '{action_name}': {e}"
            if agent_state and agent_state.action_history:
                agent_state.action_history[-1].update({
                    "status": "error",
                    "error": str(e)
                })
            raise AgentError(error_msg)
    
    def get_available_actions(self) -> List[str]:
        """
        Get a list of available actions.
        
        Returns:
            List of available action names
        """
        try:
            tool_infos = self.tool_interface.get_available_tools()
            return [tool.name for tool in tool_infos if tool.enabled]
        except Exception as e:
            raise AgentError(f"Failed to get available actions: {e}")
    
    def get_action_info(self, action_name: str) -> Dict[str, Any]:
        """
        Get information about a specific action.
        
        Args:
            action_name: Name of the action to get info for
            
        Returns:
            Dictionary containing action information
            
        Raises:
            AgentError: If action info cannot be retrieved
        """
        try:
            tool_info = self.tool_interface.get_tool_info(action_name)
            return {
                "name": tool_info.name,
                "description": tool_info.description,
                "parameters": tool_info.parameters,
                "category": tool_info.category.value,
                "enabled": tool_info.enabled
            }
        except Exception as e:
            raise AgentError(f"Failed to get action info for '{action_name}': {e}")
    
    def validate_action_parameters(
        self, 
        action_name: str, 
        parameters: Dict[str, Any]
    ) -> bool:
        """
        Validate parameters for an action.
        
        Args:
            action_name: Name of the action
            parameters: Parameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        try:
            return self.tool_interface.validate_tool_input(action_name, parameters)
        except Exception:
            return False
    
    def prepare_action_parameters(
        self, 
        action_name: str, 
        raw_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare and clean parameters for action execution.
        
        Args:
            action_name: Name of the action
            raw_parameters: Raw parameters to prepare
            
        Returns:
            Cleaned and prepared parameters
            
        Raises:
            AgentError: If parameter preparation fails
        """
        try:
            # Get action info to understand expected parameters
            action_info = self.get_action_info(action_name)
            expected_params = action_info.get("parameters", {})
            
            prepared_params = {}
            
            # Process each expected parameter
            for param_name, param_info in expected_params.items():
                if param_name in raw_parameters:
                    value = raw_parameters[param_name]
                    
                    # Basic type conversion based on parameter info
                    if isinstance(param_info, dict) and "type" in param_info:
                        param_type = param_info["type"]
                        if param_type == "string" and not isinstance(value, str):
                            value = str(value)
                        elif param_type == "integer" and not isinstance(value, int):
                            try:
                                value = int(value)
                            except (ValueError, TypeError):
                                raise AgentError(f"Cannot convert '{value}' to integer for parameter '{param_name}'")
                        elif param_type == "number" and not isinstance(value, (int, float)):
                            try:
                                value = float(value)
                            except (ValueError, TypeError):
                                raise AgentError(f"Cannot convert '{value}' to number for parameter '{param_name}'")
                        elif param_type == "boolean" and not isinstance(value, bool):
                            if isinstance(value, str):
                                value = value.lower() in ("true", "yes", "1", "on")
                            else:
                                value = bool(value)
                    
                    prepared_params[param_name] = value
                elif isinstance(param_info, dict) and param_info.get("required", False):
                    raise AgentError(f"Required parameter '{param_name}' missing for action '{action_name}'")
            
            return prepared_params
            
        except Exception as e:
            if isinstance(e, AgentError):
                raise
            raise AgentError(f"Failed to prepare parameters for action '{action_name}': {e}")
    
    def set_execution_timeout(self, timeout: float) -> None:
        """
        Set the execution timeout for actions.
        
        Args:
            timeout: Timeout in seconds
        """
        if timeout <= 0:
            raise ValueError("Timeout must be positive")
        self._execution_timeout = timeout
    
    def get_execution_timeout(self) -> float:
        """
        Get the current execution timeout.
        
        Returns:
            Current timeout in seconds
        """
        return self._execution_timeout
    
    def get_action_history(self, agent_state: AgentState, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the action execution history from agent state.
        
        Args:
            agent_state: The agent state containing action history
            limit: Maximum number of actions to return
            
        Returns:
            List of action records
        """
        history = agent_state.action_history
        if limit is not None and limit > 0:
            history = history[-limit:]
        return history
    
    def clear_action_history(self, agent_state: AgentState) -> None:
        """
        Clear the action history in agent state.
        
        Args:
            agent_state: The agent state to clear history from
        """
        agent_state.action_history.clear()