"""
Observation processor for the ReAct agent system.

This module implements the observation processing component of the ReAct cycle,
responsible for analyzing action results and extracting meaningful insights.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union
from ..interfaces.tool_interface import ToolResult
from ..models.data_models import AgentState
from ..exceptions import AgentError


class ObservationProcessor:
    """
    Processor for observations in the ReAct cycle.
    
    The ObservationProcessor analyzes action results, extracts key information,
    and formats observations for the reasoning engine to use in subsequent steps.
    """
    
    def __init__(self):
        """Initialize the observation processor."""
        self._max_observation_length = 500  # Maximum length for processed observations
        self._key_patterns = {
            'error': [r'error', r'failed', r'exception', r'invalid'],
            'success': [r'success', r'completed', r'done', r'finished'],
            'data': [r'found', r'result', r'data', r'information'],
            'count': [r'\d+\s+(?:items?|results?|files?|entries?)'],
            'url': [r'https?://[^\s]+'],
            'file': [r'[^\s]+\.[a-zA-Z]{2,4}'],
            'number': [r'\b\d+(?:\.\d+)?\b']
        }
    
    def process_observation(
        self, 
        action_result: ToolResult, 
        action_name: str,
        agent_state: Optional[AgentState] = None
    ) -> Dict[str, Any]:
        """
        Process an action result into a structured observation.
        
        Args:
            action_result: The result from tool execution
            action_name: Name of the action that was executed
            agent_state: Current agent state for context
            
        Returns:
            Dictionary containing processed observation data
            
        Raises:
            AgentError: If observation processing fails
        """
        try:
            observation = {
                'action': action_name,
                'success': action_result.success,
                'execution_time': action_result.execution_time,
                'timestamp': self._get_current_timestamp(),
                'summary': '',
                'details': {},
                'extracted_info': {},
                'next_steps_suggested': []
            }
            
            if action_result.success:
                observation.update(self._process_successful_result(action_result, action_name))
            else:
                observation.update(self._process_failed_result(action_result, action_name))
            
            # Extract key information from the result
            observation['extracted_info'] = self._extract_key_information(
                str(action_result.result) if action_result.result else ""
            )
            
            # Generate summary
            observation['summary'] = self._generate_observation_summary(observation)
            
            # Suggest next steps based on the observation
            observation['next_steps_suggested'] = self._suggest_next_steps(
                observation, action_name, agent_state
            )
            
            return observation
            
        except Exception as e:
            raise AgentError(f"Failed to process observation for action '{action_name}': {e}")
    
    def format_observation_for_reasoning(self, observation: Dict[str, Any]) -> str:
        """
        Format an observation for use in reasoning prompts.
        
        Args:
            observation: The processed observation dictionary
            
        Returns:
            Formatted observation string for reasoning
        """
        try:
            formatted_parts = []
            
            # Add action and success status
            status = "SUCCESS" if observation['success'] else "FAILED"
            formatted_parts.append(f"Action: {observation['action']} - {status}")
            
            # Add summary
            if observation.get('summary'):
                formatted_parts.append(f"Summary: {observation['summary']}")
            
            # Add key extracted information
            extracted_info = observation.get('extracted_info', {})
            if extracted_info:
                info_parts = []
                for key, value in extracted_info.items():
                    if value:
                        info_parts.append(f"{key}: {value}")
                if info_parts:
                    formatted_parts.append(f"Key Info: {', '.join(info_parts)}")
            
            # Add execution time if significant
            exec_time = observation.get('execution_time', 0)
            if exec_time > 1.0:  # Only mention if > 1 second
                formatted_parts.append(f"Execution Time: {exec_time:.1f}s")
            
            # Add suggested next steps
            next_steps = observation.get('next_steps_suggested', [])
            if next_steps:
                formatted_parts.append(f"Suggested Next Steps: {', '.join(next_steps[:3])}")
            
            return "\n".join(formatted_parts)
            
        except Exception as e:
            return f"Error formatting observation: {e}"
    
    def analyze_observation_patterns(
        self, 
        observations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze patterns across multiple observations.
        
        Args:
            observations: List of processed observations
            
        Returns:
            Dictionary containing pattern analysis
        """
        try:
            analysis = {
                'total_observations': len(observations),
                'success_rate': 0.0,
                'common_actions': {},
                'average_execution_time': 0.0,
                'error_patterns': [],
                'success_patterns': [],
                'trends': {}
            }
            
            if not observations:
                return analysis
            
            # Calculate success rate
            successful = sum(1 for obs in observations if obs.get('success', False))
            analysis['success_rate'] = successful / len(observations)
            
            # Count common actions
            for obs in observations:
                action = obs.get('action', 'unknown')
                analysis['common_actions'][action] = analysis['common_actions'].get(action, 0) + 1
            
            # Calculate average execution time
            exec_times = [obs.get('execution_time', 0) for obs in observations]
            analysis['average_execution_time'] = sum(exec_times) / len(exec_times)
            
            # Identify error patterns
            failed_observations = [obs for obs in observations if not obs.get('success', True)]
            analysis['error_patterns'] = self._identify_error_patterns(failed_observations)
            
            # Identify success patterns
            successful_observations = [obs for obs in observations if obs.get('success', False)]
            analysis['success_patterns'] = self._identify_success_patterns(successful_observations)
            
            return analysis
            
        except Exception as e:
            return {'error': f"Failed to analyze observation patterns: {e}"}
    
    def _process_successful_result(
        self, 
        action_result: ToolResult, 
        action_name: str
    ) -> Dict[str, Any]:
        """Process a successful action result."""
        details = {
            'result_type': type(action_result.result).__name__,
            'result_size': self._get_result_size(action_result.result)
        }
        
        # Action-specific processing
        if action_name.lower() in ['search', 'find', 'query']:
            details.update(self._process_search_result(action_result.result))
        elif action_name.lower() in ['calculate', 'compute', 'math']:
            details.update(self._process_calculation_result(action_result.result))
        elif action_name.lower() in ['read', 'load', 'get']:
            details.update(self._process_data_result(action_result.result))
        
        return {'details': details}
    
    def _process_failed_result(
        self, 
        action_result: ToolResult, 
        action_name: str
    ) -> Dict[str, Any]:
        """Process a failed action result."""
        error_msg = action_result.error_message or "Unknown error"
        
        details = {
            'error_message': error_msg,
            'error_type': self._classify_error(error_msg),
            'is_recoverable': self._is_error_recoverable(error_msg)
        }
        
        return {'details': details}
    
    def _extract_key_information(self, text: str) -> Dict[str, Any]:
        """Extract key information from text using patterns."""
        extracted = {}
        
        for category, patterns in self._key_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(found)
            
            if matches:
                # Take first few matches to avoid overwhelming
                extracted[category] = matches[:3] if len(matches) > 3 else matches
        
        return extracted
    
    def _generate_observation_summary(self, observation: Dict[str, Any]) -> str:
        """Generate a concise summary of the observation."""
        action = observation.get('action', 'unknown')
        success = observation.get('success', False)
        
        if success:
            details = observation.get('details', {})
            result_size = details.get('result_size', 0)
            
            if result_size > 0:
                return f"{action} completed successfully with {result_size} items"
            else:
                return f"{action} completed successfully"
        else:
            error_type = observation.get('details', {}).get('error_type', 'unknown')
            return f"{action} failed due to {error_type} error"
    
    def _suggest_next_steps(
        self, 
        observation: Dict[str, Any], 
        action_name: str,
        agent_state: Optional[AgentState] = None
    ) -> List[str]:
        """Suggest next steps based on the observation."""
        suggestions = []
        
        if observation['success']:
            # Successful action suggestions
            if action_name.lower() in ['search', 'find']:
                suggestions.extend(['analyze_results', 'filter_data', 'extract_details'])
            elif action_name.lower() in ['calculate', 'compute']:
                suggestions.extend(['verify_result', 'format_output', 'explain_calculation'])
            elif action_name.lower() in ['read', 'load']:
                suggestions.extend(['process_data', 'validate_content', 'extract_key_info'])
        else:
            # Failed action suggestions
            error_type = observation.get('details', {}).get('error_type', '')
            is_recoverable = observation.get('details', {}).get('is_recoverable', False)
            
            if is_recoverable:
                suggestions.extend(['retry_action', 'adjust_parameters', 'try_alternative'])
            else:
                suggestions.extend(['report_error', 'try_different_approach', 'seek_help'])
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def _get_result_size(self, result: Any) -> int:
        """Get the size/length of a result."""
        if result is None:
            return 0
        elif isinstance(result, (list, dict, str)):
            return len(result)
        elif hasattr(result, '__len__'):
            return len(result)
        else:
            return 1
    
    def _process_search_result(self, result: Any) -> Dict[str, Any]:
        """Process search-specific results."""
        details = {}
        
        if isinstance(result, list):
            details['item_count'] = len(result)
            details['has_results'] = len(result) > 0
        elif isinstance(result, str):
            details['content_length'] = len(result)
            details['has_content'] = len(result.strip()) > 0
        
        return details
    
    def _process_calculation_result(self, result: Any) -> Dict[str, Any]:
        """Process calculation-specific results."""
        details = {}
        
        if isinstance(result, (int, float)):
            details['numeric_result'] = result
            details['result_type'] = 'number'
        elif isinstance(result, str):
            # Try to extract numbers from string result
            numbers = re.findall(r'-?\d+(?:\.\d+)?', result)
            if numbers:
                details['extracted_numbers'] = [float(n) for n in numbers]
        
        return details
    
    def _process_data_result(self, result: Any) -> Dict[str, Any]:
        """Process data loading/reading results."""
        details = {}
        
        if isinstance(result, dict):
            details['data_keys'] = list(result.keys())[:10]  # First 10 keys
            details['data_structure'] = 'dictionary'
        elif isinstance(result, list):
            details['data_length'] = len(result)
            details['data_structure'] = 'list'
        elif isinstance(result, str):
            details['content_length'] = len(result)
            details['data_structure'] = 'text'
        
        return details
    
    def _classify_error(self, error_message: str) -> str:
        """Classify the type of error based on the message."""
        error_msg_lower = error_message.lower()
        
        if any(word in error_msg_lower for word in ['network', 'connection', 'timeout']):
            return 'network'
        elif any(word in error_msg_lower for word in ['permission', 'access', 'forbidden']):
            return 'permission'
        elif any(word in error_msg_lower for word in ['not found', '404', 'missing']):
            return 'not_found'
        elif any(word in error_msg_lower for word in ['invalid', 'malformed', 'syntax']):
            return 'validation'
        elif any(word in error_msg_lower for word in ['quota', 'limit', 'exceeded']):
            return 'quota'
        else:
            return 'unknown'
    
    def _is_error_recoverable(self, error_message: str) -> bool:
        """Determine if an error is recoverable."""
        error_type = self._classify_error(error_message)
        
        # Network and quota errors are often recoverable with retry
        recoverable_types = ['network', 'quota']
        return error_type in recoverable_types
    
    def _identify_error_patterns(self, failed_observations: List[Dict[str, Any]]) -> List[str]:
        """Identify common patterns in failed observations."""
        patterns = []
        
        if not failed_observations:
            return patterns
        
        # Group by error type
        error_types = {}
        for obs in failed_observations:
            error_type = obs.get('details', {}).get('error_type', 'unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Identify most common error types
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            if count > 1:
                patterns.append(f"Frequent {error_type} errors ({count} occurrences)")
        
        return patterns[:5]  # Top 5 patterns
    
    def _identify_success_patterns(self, successful_observations: List[Dict[str, Any]]) -> List[str]:
        """Identify common patterns in successful observations."""
        patterns = []
        
        if not successful_observations:
            return patterns
        
        # Group by action type
        action_types = {}
        for obs in successful_observations:
            action = obs.get('action', 'unknown')
            action_types[action] = action_types.get(action, 0) + 1
        
        # Identify most successful actions
        for action, count in sorted(action_types.items(), key=lambda x: x[1], reverse=True):
            if count > 1:
                patterns.append(f"Reliable {action} actions ({count} successes)")
        
        return patterns[:5]  # Top 5 patterns
    
    def _get_current_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()
    
    def set_max_observation_length(self, length: int) -> None:
        """Set the maximum length for processed observations."""
        if length <= 0:
            raise ValueError("Maximum observation length must be positive")
        self._max_observation_length = length
    
    def get_max_observation_length(self) -> int:
        """Get the maximum length for processed observations."""
        return self._max_observation_length