"""
Tool executor implementation with timeout and comprehensive error handling.

This module provides the ToolExecutor class that handles tool execution
with timeout management, input/output validation, and robust error handling.
"""

import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Dict, List, Optional, Callable
from ..interfaces.tool_interface import Tool, ToolResult, ToolInterface
from ..models.data_models import ToolInfo


class ToolExecutionError(Exception):
    """Custom exception for tool execution errors."""
    
    def __init__(self, message: str, tool_name: str = None, original_error: Exception = None):
        super().__init__(message)
        self.tool_name = tool_name
        self.original_error = original_error


class ToolTimeoutError(ToolExecutionError):
    """Exception raised when tool execution times out."""
    pass


class ToolValidationError(ToolExecutionError):
    """Exception raised when tool input validation fails."""
    pass


class ToolExecutor:
    """
    Advanced tool executor with timeout, validation, and error handling.
    
    This class provides robust tool execution capabilities including:
    - Configurable timeouts for tool execution
    - Input/output validation and sanitization
    - Comprehensive error handling and recovery
    - Execution monitoring and logging
    """
    
    def __init__(self, 
                 default_timeout: float = 30.0,
                 max_concurrent_executions: int = 5,
                 enable_input_sanitization: bool = True,
                 enable_output_validation: bool = True):
        """
        Initialize the tool executor.
        
        Args:
            default_timeout: Default timeout for tool execution in seconds
            max_concurrent_executions: Maximum number of concurrent tool executions
            enable_input_sanitization: Whether to sanitize tool inputs
            enable_output_validation: Whether to validate tool outputs
        """
        self.default_timeout = default_timeout
        self.max_concurrent_executions = max_concurrent_executions
        self.enable_input_sanitization = enable_input_sanitization
        self.enable_output_validation = enable_output_validation
        
        # Thread pool for concurrent execution
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_executions)
        
        # Execution tracking
        self._active_executions: Dict[str, threading.Thread] = {}
        self._execution_history: List[Dict[str, Any]] = []
        
        # Input sanitization rules
        self._sanitization_rules = {
            'max_string_length': 10000,
            'max_list_length': 1000,
            'max_dict_depth': 10,
            'forbidden_patterns': [
                r'__\w+__',  # Dunder methods
                r'eval\s*\(',
                r'exec\s*\(',
                r'import\s+',
                r'from\s+\w+\s+import',
            ]
        }
    
    def execute_tool_with_timeout(self, 
                                  tool: Tool, 
                                  params: Dict[str, Any],
                                  timeout: Optional[float] = None) -> ToolResult:
        """
        Execute a tool with timeout and comprehensive error handling.
        
        Args:
            tool: The tool to execute
            params: Parameters for the tool
            timeout: Timeout in seconds (uses default if None)
            
        Returns:
            ToolResult containing the execution result
            
        Raises:
            ToolExecutionError: If execution fails
            ToolTimeoutError: If execution times out
            ToolValidationError: If validation fails
        """
        timeout = timeout or self.default_timeout
        tool_info = tool.get_info()
        tool_name = tool_info.name
        
        # Record execution start
        execution_id = f"{tool_name}_{int(time.time() * 1000)}"
        execution_record = {
            'id': execution_id,
            'tool_name': tool_name,
            'start_time': time.time(),
            'timeout': timeout,
            'status': 'started'
        }
        
        try:
            # Validate and sanitize inputs
            if self.enable_input_sanitization:
                params = self._sanitize_input(params, tool_name)
            
            # Validate parameters
            if not tool.validate_params(params):
                raise ToolValidationError(
                    f"Invalid parameters for tool '{tool_name}'",
                    tool_name=tool_name
                )
            
            # Execute with timeout
            start_time = time.time()
            future = self._executor.submit(self._safe_execute_tool, tool, params)
            
            try:
                result = future.result(timeout=timeout)
                execution_time = time.time() - start_time
                
                # Validate output if enabled
                if self.enable_output_validation:
                    result = self._validate_output(result, tool_name)
                
                # Update execution record
                execution_record.update({
                    'status': 'completed',
                    'execution_time': execution_time,
                    'success': result.success
                })
                
                # Set execution time if not already set
                if result.execution_time == 0.0:
                    result.execution_time = execution_time
                
                return result
                
            except FutureTimeoutError:
                future.cancel()
                execution_record.update({
                    'status': 'timeout',
                    'execution_time': timeout
                })
                raise ToolTimeoutError(
                    f"Tool '{tool_name}' execution timed out after {timeout} seconds",
                    tool_name=tool_name
                )
                
        except (ToolValidationError, ToolTimeoutError):
            # Re-raise these specific exceptions
            raise
        except Exception as e:
            execution_record.update({
                'status': 'error',
                'error': str(e)
            })
            raise ToolExecutionError(
                f"Tool '{tool_name}' execution failed: {str(e)}",
                tool_name=tool_name,
                original_error=e
            )
        finally:
            # Record execution
            execution_record['end_time'] = time.time()
            self._execution_history.append(execution_record)
            
            # Clean up old history (keep last 100 executions)
            if len(self._execution_history) > 100:
                self._execution_history = self._execution_history[-100:]
    
    def execute_tool_safe(self, 
                          tool: Tool, 
                          params: Dict[str, Any],
                          timeout: Optional[float] = None) -> ToolResult:
        """
        Execute a tool safely, returning a ToolResult even on errors.
        
        This method never raises exceptions, instead returning error information
        in the ToolResult object.
        
        Args:
            tool: The tool to execute
            params: Parameters for the tool
            timeout: Timeout in seconds (uses default if None)
            
        Returns:
            ToolResult containing the execution result or error information
        """
        try:
            return self.execute_tool_with_timeout(tool, params, timeout)
        except (ToolTimeoutError, ToolValidationError, ToolExecutionError) as e:
            # These exceptions are already handled and recorded in execute_tool_with_timeout
            if isinstance(e, ToolTimeoutError):
                return ToolResult(
                    success=False,
                    error_message=f"Timeout: {str(e)}",
                    execution_time=timeout or self.default_timeout
                )
            elif isinstance(e, ToolValidationError):
                return ToolResult(
                    success=False,
                    error_message=f"Validation error: {str(e)}",
                    execution_time=0.0
                )
            else:  # ToolExecutionError
                return ToolResult(
                    success=False,
                    error_message=str(e),
                    execution_time=0.0
                )
        except Exception as e:
            # Handle unexpected exceptions that weren't caught by execute_tool_with_timeout
            tool_info = tool.get_info()
            tool_name = tool_info.name
            
            # Record this unexpected error
            execution_record = {
                'id': f"{tool_name}_{int(time.time() * 1000)}",
                'tool_name': tool_name,
                'start_time': time.time(),
                'end_time': time.time(),
                'timeout': timeout or self.default_timeout,
                'status': 'error',
                'error': str(e),
                'success': False
            }
            self._execution_history.append(execution_record)
            
            return ToolResult(
                success=False,
                error_message=f"Unexpected error: {str(e)}",
                execution_time=0.0
            )
    
    def _safe_execute_tool(self, tool: Tool, params: Dict[str, Any]) -> ToolResult:
        """
        Safely execute a tool with additional error handling.
        
        Args:
            tool: The tool to execute
            params: Parameters for the tool
            
        Returns:
            ToolResult containing the execution result
            
        Raises:
            Exception: Re-raises the original exception for proper error handling
        """
        return tool.execute(params)
    
    def _sanitize_input(self, params: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """
        Sanitize input parameters to prevent security issues.
        
        Args:
            params: Parameters to sanitize
            tool_name: Name of the tool (for error reporting)
            
        Returns:
            Sanitized parameters
            
        Raises:
            ToolValidationError: If sanitization fails or dangerous content is found
        """
        try:
            return self._sanitize_value(params, 0)
        except Exception as e:
            raise ToolValidationError(
                f"Input sanitization failed for tool '{tool_name}': {str(e)}",
                tool_name=tool_name,
                original_error=e
            )
    
    def _sanitize_value(self, value: Any, depth: int) -> Any:
        """
        Recursively sanitize a value.
        
        Args:
            value: Value to sanitize
            depth: Current recursion depth
            
        Returns:
            Sanitized value
            
        Raises:
            ValueError: If value is dangerous or exceeds limits
        """
        # Check recursion depth
        if depth > self._sanitization_rules['max_dict_depth']:
            raise ValueError(f"Maximum nesting depth exceeded: {depth}")
        
        if isinstance(value, str):
            # Check string length
            if len(value) > self._sanitization_rules['max_string_length']:
                raise ValueError(f"String too long: {len(value)} characters")
            
            # Check for forbidden patterns
            import re
            for pattern in self._sanitization_rules['forbidden_patterns']:
                if re.search(pattern, value, re.IGNORECASE):
                    raise ValueError(f"Forbidden pattern detected: {pattern}")
            
            return value
        
        elif isinstance(value, dict):
            # Check dictionary size
            if len(value) > self._sanitization_rules['max_list_length']:
                raise ValueError(f"Dictionary too large: {len(value)} items")
            
            # Recursively sanitize dictionary values
            sanitized = {}
            for k, v in value.items():
                sanitized_key = self._sanitize_value(k, depth + 1)
                sanitized_value = self._sanitize_value(v, depth + 1)
                sanitized[sanitized_key] = sanitized_value
            return sanitized
        
        elif isinstance(value, list):
            # Check list length
            if len(value) > self._sanitization_rules['max_list_length']:
                raise ValueError(f"List too long: {len(value)} items")
            
            # Recursively sanitize list items
            return [self._sanitize_value(item, depth + 1) for item in value]
        
        elif isinstance(value, (int, float, bool, type(None))):
            # These types are safe
            return value
        
        else:
            # Convert other types to string and sanitize
            return self._sanitize_value(str(value), depth)
    
    def _validate_output(self, result: ToolResult, tool_name: str) -> ToolResult:
        """
        Validate tool output for safety and consistency.
        
        Args:
            result: The tool result to validate
            tool_name: Name of the tool (for error reporting)
            
        Returns:
            Validated tool result
            
        Raises:
            ToolValidationError: If validation fails
        """
        if not isinstance(result, ToolResult):
            raise ToolValidationError(
                f"Tool '{tool_name}' returned invalid result type: {type(result)}",
                tool_name=tool_name
            )
        
        # Validate result attributes
        if not isinstance(result.success, bool):
            result.success = bool(result.success)
        
        if result.error_message is not None and not isinstance(result.error_message, str):
            result.error_message = str(result.error_message)
        
        if not isinstance(result.execution_time, (int, float)):
            result.execution_time = 0.0
        elif result.execution_time < 0:
            result.execution_time = 0.0
        
        # Sanitize result data if it's a dictionary or list
        if isinstance(result.result, (dict, list)):
            try:
                result.result = self._sanitize_value(result.result, 0)
            except ValueError as e:
                # If sanitization fails, convert to string
                result.result = f"<sanitization failed: {str(e)}>"
        
        return result
    
    def get_execution_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the execution history.
        
        Args:
            limit: Maximum number of records to return (None for all)
            
        Returns:
            List of execution records
        """
        history = self._execution_history.copy()
        if limit is not None:
            history = history[-limit:]
        return history
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Returns:
            Dictionary containing execution statistics
        """
        if not self._execution_history:
            return {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'timeout_executions': 0,
                'average_execution_time': 0.0,
                'success_rate': 0.0
            }
        
        total = len(self._execution_history)
        successful = sum(1 for record in self._execution_history if record.get('success', False))
        failed = sum(1 for record in self._execution_history if record.get('status') == 'error')
        timeouts = sum(1 for record in self._execution_history if record.get('status') == 'timeout')
        
        # Calculate average execution time for completed executions
        completed_times = [
            record.get('execution_time', 0) 
            for record in self._execution_history 
            if record.get('status') == 'completed'
        ]
        avg_time = sum(completed_times) / len(completed_times) if completed_times else 0.0
        
        return {
            'total_executions': total,
            'successful_executions': successful,
            'failed_executions': failed,
            'timeout_executions': timeouts,
            'average_execution_time': avg_time,
            'success_rate': successful / total if total > 0 else 0.0
        }
    
    def clear_history(self) -> None:
        """Clear the execution history."""
        self._execution_history.clear()
    
    def shutdown(self) -> None:
        """Shutdown the executor and clean up resources."""
        self._executor.shutdown(wait=True)
        self._active_executions.clear()
        self._execution_history.clear()