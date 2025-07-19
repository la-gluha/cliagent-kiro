"""
ReAct agent implementation for the AI Agent System.

This module implements the main ReAct (Reasoning and Acting) agent that
coordinates the reasoning-acting cycle for autonomous task execution.
"""

import time
from typing import Any, Dict, List, Optional
from ..interfaces.agent_interface import AgentInterface
from ..interfaces.model_interface import ModelInterface
from ..interfaces.tool_interface import ToolInterface
from ..interfaces.memory_interface import MemoryInterface
from ..models.data_models import TaskResult, AgentState, Message, MessageRole
from ..exceptions import AgentError
from .reasoning_engine import ReasoningEngine
from .action_executor import ActionExecutor
from .observation_processor import ObservationProcessor


class ReActAgent(AgentInterface):
    """
    ReAct (Reasoning and Acting) agent implementation.
    
    This agent follows the ReAct framework: it reasons about problems,
    takes actions using available tools, observes the results, and
    continues this cycle until tasks are completed.
    """
    
    def __init__(
        self,
        model: ModelInterface,
        tool_interface: ToolInterface,
        memory: MemoryInterface,
        max_iterations: int = 10,
        max_reasoning_steps: int = 5
    ):
        """
        Initialize the ReAct agent.
        
        Args:
            model: Language model interface for reasoning
            tool_interface: Tool interface for actions
            memory: Memory interface for conversation history
            max_iterations: Maximum ReAct cycle iterations
            max_reasoning_steps: Maximum reasoning steps per iteration
        """
        self.model = model
        self.tool_interface = tool_interface
        self.memory = memory
        self.max_iterations = max_iterations
        self.max_reasoning_steps = max_reasoning_steps
        
        # Initialize components
        self.reasoning_engine = ReasoningEngine(model)
        self.action_executor = ActionExecutor(tool_interface)
        self.observation_processor = ObservationProcessor()
        
        # Initialize state
        self.state = AgentState()
        self._stop_requested = False
        
        # Enhanced state management
        self._observation_history = []  # Store processed observations
        self._context_window_size = 10  # Number of recent observations to keep in context
        self._context_metadata = {}  # Additional context metadata
        self._performance_metrics = {
            'total_actions': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'average_action_time': 0.0,
            'reasoning_steps_per_task': 0.0
        }
    
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
        try:
            # Store user message in memory
            user_message = Message(content=user_input, role=MessageRole.USER)
            self.memory.store_message(user_message)
            
            # Set current task and activate agent
            self.state.current_task = user_input
            self.state.is_active = True
            self._stop_requested = False
            
            # Execute ReAct cycle
            response = self._execute_react_cycle(user_input)
            
            # Update reasoning metrics
            self._update_reasoning_metrics()
            
            # Store agent response in memory
            agent_message = Message(content=response, role=MessageRole.AGENT)
            self.memory.store_message(agent_message)
            
            # Deactivate agent
            self.state.is_active = False
            self.state.current_task = None
            
            return response
            
        except Exception as e:
            self.state.is_active = False
            self.state.current_task = None
            raise AgentError(f"Failed to process input: {e}") 
   
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
        start_time = time.time()
        steps_taken = []
        
        try:
            # Set up task execution
            self.state.current_task = task
            self.state.is_active = True
            self._stop_requested = False
            
            # Clear previous reasoning and action history for new task
            self.state.clear_history()
            
            # Execute ReAct cycle for task completion
            result = self._execute_react_cycle(task)
            
            # Update reasoning metrics
            self._update_reasoning_metrics()
            
            execution_time = time.time() - start_time
            steps_taken = [f"Reasoning step {i+1}" for i in range(len(self.state.reasoning_history))]
            steps_taken.extend([f"Action: {action.get('action', 'unknown')}" for action in self.state.action_history])
            
            return TaskResult(
                success=True,
                result=result,
                execution_time=execution_time,
                steps_taken=steps_taken
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TaskResult(
                success=False,
                error_message=str(e),
                execution_time=execution_time,
                steps_taken=steps_taken
            )
        finally:
            self.state.is_active = False
            self.state.current_task = None
    
    def get_available_actions(self) -> List[str]:
        """
        Get a list of actions the agent can perform.
        
        Returns:
            List of available action names
        """
        return self.action_executor.get_available_actions()
    
    def reset_context(self) -> None:
        """
        Reset the agent's context and state.
        
        Raises:
            AgentError: If context reset fails
        """
        try:
            self.state = AgentState()
            self._stop_requested = False
            self._observation_history.clear()
            self._context_metadata.clear()
            # Note: We don't clear memory here as it should persist across resets
            # Note: We don't clear performance metrics as they track overall performance
        except Exception as e:
            raise AgentError(f"Failed to reset context: {e}")
    
    def get_state(self) -> AgentState:
        """
        Get the current state of the agent.
        
        Returns:
            AgentState object representing current state
        """
        return self.state
    
    def set_state(self, state: AgentState) -> None:
        """
        Set the agent's state.
        
        Args:
            state: The AgentState to set
            
        Raises:
            AgentError: If state cannot be set
        """
        try:
            state.validate()
            self.state = state
        except Exception as e:
            raise AgentError(f"Failed to set state: {e}")
    
    def is_active(self) -> bool:
        """
        Check if the agent is currently active/processing.
        
        Returns:
            True if the agent is active, False otherwise
        """
        return self.state.is_active
    
    def stop(self) -> None:
        """
        Stop the agent's current operation.
        
        Raises:
            AgentError: If the agent cannot be stopped
        """
        self._stop_requested = True
        if self.state.is_active:
            self.state.is_active = False    

    def _execute_react_cycle(self, input_text: str) -> str:
        """
        Execute the main ReAct reasoning-acting cycle.
        
        Args:
            input_text: The input to process
            
        Returns:
            The final response from the ReAct cycle
            
        Raises:
            AgentError: If the ReAct cycle fails
        """
        iteration = 0
        final_response = ""
        
        while iteration < self.max_iterations and not self._stop_requested:
            try:
                # REASONING PHASE
                available_actions = self.get_available_actions()
                context = self.memory.get_context()
                
                reasoning = self.reasoning_engine.generate_reasoning(
                    situation=input_text,
                    available_actions=available_actions,
                    agent_state=self.state,
                    context=context
                )
                
                # Validate and store reasoning
                if not self.reasoning_engine.validate_reasoning(reasoning):
                    reasoning = f"Iteration {iteration + 1}: Analyzing the situation: {input_text}"
                
                self.state.add_reasoning_step(reasoning)
                
                # Determine if we need to take an action or if we can provide a final answer
                if self._should_take_action(reasoning, iteration):
                    # ACTION PHASE
                    action_name = self.reasoning_engine.determine_next_action(
                        reasoning=reasoning,
                        available_actions=available_actions,
                        context=context
                    )
                    
                    # Prepare action parameters (simplified - could be enhanced)
                    action_params = self._extract_action_parameters(reasoning, action_name)
                    
                    # Execute the action
                    action_result = self.action_executor.execute_action(
                        action_name=action_name,
                        parameters=action_params,
                        agent_state=self.state
                    )
                    
                    # OBSERVATION PHASE
                    observation = self._process_action_result(action_result, action_name)
                    
                    # Update input for next iteration with observation
                    input_text = f"{input_text}\n\nObservation from {action_name}: {observation}"
                    
                else:
                    # Generate final response
                    final_response = self._generate_final_response(reasoning, input_text)
                    break
                
                iteration += 1
                
            except Exception as e:
                # If we encounter an error, try to provide a reasonable response
                error_msg = f"Error in ReAct cycle iteration {iteration + 1}: {e}"
                self.state.add_reasoning_step(error_msg)
                
                if iteration == 0:
                    # If first iteration fails, raise the error
                    raise AgentError(error_msg)
                else:
                    # If later iteration fails, provide best effort response
                    final_response = f"I encountered an issue while processing your request: {e}. Based on my analysis so far, here's what I can tell you: {self._generate_fallback_response()}"
                    break
        
        if iteration >= self.max_iterations:
            final_response = f"I've reached the maximum number of reasoning iterations ({self.max_iterations}). Based on my analysis: {self._generate_fallback_response()}"
        
        return final_response or "I was unable to process your request." 
   
    def _should_take_action(self, reasoning: str, iteration: int) -> bool:
        """
        Determine if the agent should take an action based on reasoning.
        
        Args:
            reasoning: The current reasoning step
            iteration: Current iteration number
            
        Returns:
            True if an action should be taken, False if ready to respond
        """
        # Simple heuristics - could be enhanced with more sophisticated logic
        reasoning_lower = reasoning.lower()
        
        # If reasoning mentions needing to search, calculate, or use a tool
        action_indicators = [
            "need to search", "should search", "let me search",
            "need to calculate", "should calculate", "let me calculate",
            "need to check", "should check", "let me check",
            "need to find", "should find", "let me find",
            "use tool", "execute", "run"
        ]
        
        has_action_indicator = any(indicator in reasoning_lower for indicator in action_indicators)
        
        # Don't take action if we're in the last few iterations (to ensure we provide a response)
        near_max_iterations = iteration >= (self.max_iterations - 2)
        
        return has_action_indicator and not near_max_iterations
    
    def _extract_action_parameters(self, reasoning: str, action_name: str) -> Dict[str, Any]:
        """
        Extract parameters for an action from reasoning text.
        
        Args:
            reasoning: The reasoning text
            action_name: Name of the action to extract parameters for
            
        Returns:
            Dictionary of extracted parameters
        """
        # This is a simplified implementation
        # In a more sophisticated system, this would use NLP to extract parameters
        params = {}
        
        # Basic parameter extraction based on common patterns
        reasoning_lower = reasoning.lower()
        
        # Look for quoted strings that might be search queries or file names
        import re
        quoted_strings = re.findall(r'"([^"]*)"', reasoning)
        if quoted_strings:
            # Use the first quoted string as a query parameter
            params["query"] = quoted_strings[0]
        
        # Look for numbers that might be parameters
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', reasoning)
        if numbers and action_name.lower() in ["calculate", "math"]:
            params["value"] = float(numbers[0])
        
        return params
    
    def _process_action_result(self, action_result, action_name: str) -> str:
        """
        Process the result of an action into an observation.
        
        Args:
            action_result: The ToolResult from action execution
            action_name: Name of the action that was executed
            
        Returns:
            Observation string describing the result
        """
        try:
            # Update performance metrics
            self._update_performance_metrics(action_result, action_result.execution_time)
            
            # Process observation using the observation processor
            processed_observation = self.observation_processor.process_observation(
                action_result=action_result,
                action_name=action_name,
                agent_state=self.state
            )
            
            # Store the processed observation in history
            self._observation_history.append(processed_observation)
            
            # Keep only recent observations within context window
            if len(self._observation_history) > self._context_window_size:
                self._observation_history = self._observation_history[-self._context_window_size:]
            
            # Format observation for reasoning
            formatted_observation = self.observation_processor.format_observation_for_reasoning(
                processed_observation
            )
            
            return formatted_observation
            
        except Exception as e:
            # Fallback to simple processing if observation processor fails
            if action_result.success:
                result_str = str(action_result.result) if action_result.result is not None else "Action completed successfully"
                return f"Success: {result_str}"
            else:
                return f"Failed: {action_result.error_message or 'Unknown error'}" 
   
    def _generate_final_response(self, reasoning: str, context: str) -> str:
        """
        Generate the final response based on reasoning and context.
        
        Args:
            reasoning: The final reasoning step
            context: The full context including observations
            
        Returns:
            Final response string
        """
        try:
            response_prompt = f"""
Based on my reasoning and analysis, provide a helpful response to the user.

Context: {context}
My reasoning: {reasoning}
Conversation history: {self._get_recent_conversation_summary()}

Provide a clear, helpful response that addresses the user's request.
"""
            
            response = self.model.generate_response(response_prompt)
            return response.strip() if response else "I've completed my analysis of your request."
            
        except Exception:
            return "I've completed my analysis of your request."
    
    def _generate_fallback_response(self) -> str:
        """
        Generate a fallback response when the ReAct cycle encounters issues.
        
        Returns:
            Fallback response string
        """
        if self.state.reasoning_history:
            last_reasoning = self.state.reasoning_history[-1]
            return f"Based on my analysis: {last_reasoning[:200]}..."
        return "I attempted to analyze your request but encountered some difficulties."
    
    def _get_recent_conversation_summary(self) -> str:
        """
        Get a summary of recent conversation history.
        
        Returns:
            Summary of recent messages
        """
        try:
            recent_messages = self.memory.retrieve_history(limit=5)
            if not recent_messages:
                return "No previous conversation"
            
            summary_parts = []
            for msg in recent_messages[-3:]:  # Last 3 messages
                role = msg.role.value.capitalize()
                content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                summary_parts.append(f"{role}: {content}")
            
            return "\n".join(summary_parts)
            
        except Exception:
            return "Unable to retrieve conversation history"
    
    def get_observation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the observation history.
        
        Args:
            limit: Maximum number of observations to return
            
        Returns:
            List of processed observations
        """
        if limit is None:
            return self._observation_history.copy()
        return self._observation_history[-limit:] if limit > 0 else []
    
    def analyze_observation_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in the observation history.
        
        Returns:
            Dictionary containing pattern analysis
        """
        return self.observation_processor.analyze_observation_patterns(self._observation_history)
    
    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the current context.
        
        Returns:
            Dictionary containing context summary
        """
        try:
            context_summary = {
                'current_task': self.state.current_task,
                'is_active': self.state.is_active,
                'reasoning_steps': len(self.state.reasoning_history),
                'actions_taken': len(self.state.action_history),
                'observations_recorded': len(self._observation_history),
                'recent_reasoning': self.state.reasoning_history[-3:] if self.state.reasoning_history else [],
                'recent_actions': [action.get('action', 'unknown') for action in self.state.action_history[-3:]],
                'observation_patterns': self.analyze_observation_patterns(),
                'memory_context': self.memory.get_context() if hasattr(self.memory, 'get_context') else {}
            }
            
            return context_summary
            
        except Exception as e:
            return {'error': f"Failed to generate context summary: {e}"}    

    def update_context(self, context_updates: Dict[str, Any]) -> None:
        """
        Update the agent's context with new information.
        
        Args:
            context_updates: Dictionary of context updates to apply
            
        Raises:
            AgentError: If context update fails
        """
        try:
            # Update agent state context
            if 'agent_context' in context_updates:
                self.state.context.update(context_updates['agent_context'])
            
            # Update memory context if supported
            if 'memory_context' in context_updates and hasattr(self.memory, 'set_context'):
                current_memory_context = self.memory.get_context()
                current_memory_context.update(context_updates['memory_context'])
                self.memory.set_context(current_memory_context)
            
            # Update observation processor settings if provided
            if 'observation_settings' in context_updates:
                settings = context_updates['observation_settings']
                if 'max_length' in settings:
                    self.observation_processor.set_max_observation_length(settings['max_length'])
                if 'context_window_size' in settings:
                    self._context_window_size = max(1, int(settings['context_window_size']))
            
        except Exception as e:
            raise AgentError(f"Failed to update context: {e}")
    
    def get_reasoning_history_summary(self) -> str:
        """
        Get a formatted summary of the reasoning history.
        
        Returns:
            Formatted string summarizing reasoning steps
        """
        if not self.state.reasoning_history:
            return "No reasoning history available"
        
        summary_parts = []
        for i, reasoning in enumerate(self.state.reasoning_history, 1):
            # Truncate long reasoning steps
            truncated = reasoning[:150] + "..." if len(reasoning) > 150 else reasoning
            summary_parts.append(f"Step {i}: {truncated}")
        
        return "\n".join(summary_parts)
    
    def get_action_history_summary(self) -> str:
        """
        Get a formatted summary of the action history.
        
        Returns:
            Formatted string summarizing actions taken
        """
        if not self.state.action_history:
            return "No actions taken"
        
        summary_parts = []
        for i, action in enumerate(self.state.action_history, 1):
            action_name = action.get('action', 'unknown')
            status = action.get('status', 'unknown')
            exec_time = action.get('execution_time', 0)
            
            summary_parts.append(f"Action {i}: {action_name} - {status} ({exec_time:.2f}s)")
        
        return "\n".join(summary_parts)
    
    def set_context_window_size(self, size: int) -> None:
        """
        Set the context window size for observations.
        
        Args:
            size: Number of recent observations to keep in context
            
        Raises:
            ValueError: If size is not positive
        """
        if size <= 0:
            raise ValueError("Context window size must be positive")
        
        self._context_window_size = size
        
        # Trim observation history if necessary
        if len(self._observation_history) > size:
            self._observation_history = self._observation_history[-size:]
    
    def get_context_window_size(self) -> int:
        """
        Get the current context window size.
        
        Returns:
            Current context window size
        """
        return self._context_window_size    

    def get_context_metadata(self) -> Dict[str, Any]:
        """
        Get the current context metadata.
        
        Returns:
            Dictionary containing context metadata
        """
        return self._context_metadata.copy()
    
    def set_context_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Set context metadata.
        
        Args:
            metadata: Dictionary of metadata to set
            
        Raises:
            AgentError: If metadata cannot be set
        """
        try:
            if not isinstance(metadata, dict):
                raise ValueError("Metadata must be a dictionary")
            self._context_metadata = metadata.copy()
        except Exception as e:
            raise AgentError(f"Failed to set context metadata: {e}")
    
    def update_context_metadata(self, metadata_updates: Dict[str, Any]) -> None:
        """
        Update context metadata with new information.
        
        Args:
            metadata_updates: Dictionary of metadata updates to apply
        """
        self._context_metadata.update(metadata_updates)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        return self._performance_metrics.copy()
    
    def _update_performance_metrics(self, action_result, execution_time: float) -> None:
        """
        Update performance metrics based on action results.
        
        Args:
            action_result: The result of the action
            execution_time: Time taken to execute the action
        """
        self._performance_metrics['total_actions'] += 1
        
        if action_result.success:
            self._performance_metrics['successful_actions'] += 1
        else:
            self._performance_metrics['failed_actions'] += 1
        
        # Update average action time
        total_actions = self._performance_metrics['total_actions']
        current_avg = self._performance_metrics['average_action_time']
        self._performance_metrics['average_action_time'] = (
            (current_avg * (total_actions - 1) + execution_time) / total_actions
        )
    
    def _update_reasoning_metrics(self) -> None:
        """Update reasoning-related performance metrics."""
        if self.state.reasoning_history:
            self._performance_metrics['reasoning_steps_per_task'] = len(self.state.reasoning_history)
    
    def get_detailed_state_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the agent's current state.
        
        Returns:
            Dictionary containing comprehensive state information
        """
        try:
            state_info = {
                'basic_state': {
                    'current_task': self.state.current_task,
                    'is_active': self.state.is_active,
                    'stop_requested': self._stop_requested
                },
                'history_counts': {
                    'reasoning_steps': len(self.state.reasoning_history),
                    'actions_taken': len(self.state.action_history),
                    'observations_recorded': len(self._observation_history)
                },
                'context_info': {
                    'context_window_size': self._context_window_size,
                    'context_metadata': self._context_metadata,
                    'agent_context_keys': list(self.state.context.keys())
                },
                'performance_metrics': self._performance_metrics,
                'recent_activity': {
                    'last_reasoning': self.state.reasoning_history[-1] if self.state.reasoning_history else None,
                    'last_action': self.state.action_history[-1] if self.state.action_history else None,
                    'last_observation': self._observation_history[-1] if self._observation_history else None
                },
                'configuration': {
                    'max_iterations': self.max_iterations,
                    'max_reasoning_steps': self.max_reasoning_steps,
                    'observation_processor_max_length': self.observation_processor.get_max_observation_length()
                }
            }
            
            return state_info
            
        except Exception as e:
            return {'error': f"Failed to generate detailed state info: {e}"}
    
    def clear_performance_metrics(self) -> None:
        """Reset all performance metrics to initial values."""
        self._performance_metrics = {
            'total_actions': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'average_action_time': 0.0,
            'reasoning_steps_per_task': 0.0
        }    

    def export_state_snapshot(self) -> Dict[str, Any]:
        """
        Export a complete snapshot of the agent's current state.
        
        Returns:
            Dictionary containing complete state snapshot
        """
        try:
            snapshot = {
                'timestamp': time.time(),
                'agent_state': self.state.to_dict(),
                'observation_history': self._observation_history.copy(),
                'context_metadata': self._context_metadata.copy(),
                'performance_metrics': self._performance_metrics.copy(),
                'configuration': {
                    'max_iterations': self.max_iterations,
                    'max_reasoning_steps': self.max_reasoning_steps,
                    'context_window_size': self._context_window_size
                },
                'memory_context': self.memory.get_context() if hasattr(self.memory, 'get_context') else {}
            }
            
            return snapshot
            
        except Exception as e:
            return {'error': f"Failed to export state snapshot: {e}"}
    
    def import_state_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        Import a state snapshot to restore agent state.
        
        Args:
            snapshot: Dictionary containing state snapshot
            
        Raises:
            AgentError: If snapshot import fails
        """
        try:
            if 'agent_state' in snapshot:
                self.state = AgentState.from_dict(snapshot['agent_state'])
            
            if 'observation_history' in snapshot:
                self._observation_history = snapshot['observation_history'].copy()
            
            if 'context_metadata' in snapshot:
                self._context_metadata = snapshot['context_metadata'].copy()
            
            if 'performance_metrics' in snapshot:
                self._performance_metrics.update(snapshot['performance_metrics'])
            
            if 'configuration' in snapshot:
                config = snapshot['configuration']
                if 'context_window_size' in config:
                    self.set_context_window_size(config['context_window_size'])
            
        except Exception as e:
            raise AgentError(f"Failed to import state snapshot: {e}")
    
    def validate_state_consistency(self) -> Dict[str, Any]:
        """
        Validate the consistency of the agent's state.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        try:
            # Validate basic state
            self.state.validate()
            
            # Check observation history consistency
            if len(self._observation_history) > self._context_window_size:
                validation_results['warnings'].append(
                    f"Observation history ({len(self._observation_history)}) exceeds context window size ({self._context_window_size})"
                )
            
            # Check performance metrics consistency
            total_actions = self._performance_metrics['total_actions']
            successful_actions = self._performance_metrics['successful_actions']
            failed_actions = self._performance_metrics['failed_actions']
            
            if successful_actions + failed_actions != total_actions:
                validation_results['issues'].append(
                    f"Performance metrics inconsistent: {successful_actions} + {failed_actions} != {total_actions}"
                )
                validation_results['is_valid'] = False
            
            # Check reasoning/action history alignment
            reasoning_count = len(self.state.reasoning_history)
            action_count = len(self.state.action_history)
            
            if reasoning_count > 0 and action_count > reasoning_count:
                validation_results['warnings'].append(
                    f"More actions ({action_count}) than reasoning steps ({reasoning_count})"
                )
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"State validation error: {e}")
        
        return validation_results