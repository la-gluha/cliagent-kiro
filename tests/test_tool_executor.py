"""
Integration tests for the ToolExecutor class.

This module contains comprehensive tests for tool execution with timeout,
validation, and error handling scenarios.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
from src.tools.tool_executor import (
    ToolExecutor, 
    ToolExecutionError, 
    ToolTimeoutError, 
    ToolValidationError
)
from src.interfaces.tool_interface import Tool, ToolResult
from src.models.data_models import ToolInfo, ToolCategory


class MockTool(Tool):
    """Mock tool for testing purposes."""
    
    def __init__(self, name="mock_tool", execution_time=0.1, should_fail=False, 
                 should_timeout=False, invalid_output=False):
        self.name = name
        self.execution_time = execution_time
        self.should_fail = should_fail
        self.should_timeout = should_timeout
        self.invalid_output = invalid_output
        self.execute_called = False
        self.validate_called = False
    
    def get_info(self) -> ToolInfo:
        return ToolInfo(
            name=self.name,
            description="A mock tool for testing",
            parameters={"param1": {"type": "string"}},
            category=ToolCategory.CUSTOM,
            enabled=True
        )
    
    def execute(self, params):
        self.execute_called = True
        
        if self.should_timeout:
            # Sleep longer than typical timeout
            time.sleep(5.0)
        
        if self.should_fail:
            raise Exception("Mock tool execution failed")
        
        # Simulate execution time
        time.sleep(self.execution_time)
        
        if self.invalid_output:
            return "invalid result type"  # Should be ToolResult
        
        return ToolResult(
            success=True, 
            result="mock result",
            execution_time=self.execution_time
        )
    
    def validate_params(self, params):
        self.validate_called = True
        return isinstance(params, dict) and "param1" in params


class SlowTool(Tool):
    """Tool that takes a long time to execute for timeout testing."""
    
    def get_info(self) -> ToolInfo:
        return ToolInfo(
            name="slow_tool",
            description="A slow tool for timeout testing",
            parameters={"delay": {"type": "number"}},
            category=ToolCategory.CUSTOM,
            enabled=True
        )
    
    def execute(self, params):
        delay = params.get("delay", 2.0)
        time.sleep(delay)
        return ToolResult(success=True, result=f"Completed after {delay}s")
    
    def validate_params(self, params):
        return isinstance(params, dict) and "delay" in params


class TestToolExecutor:
    """Test cases for the ToolExecutor class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.executor = ToolExecutor(
            default_timeout=1.0,
            max_concurrent_executions=3,
            enable_input_sanitization=True,
            enable_output_validation=True
        )
        self.mock_tool = MockTool()
    
    def teardown_method(self):
        """Clean up after each test method."""
        self.executor.shutdown()
    
    def test_init(self):
        """Test ToolExecutor initialization."""
        executor = ToolExecutor(
            default_timeout=30.0,
            max_concurrent_executions=5,
            enable_input_sanitization=False,
            enable_output_validation=False
        )
        
        assert executor.default_timeout == 30.0
        assert executor.max_concurrent_executions == 5
        assert not executor.enable_input_sanitization
        assert not executor.enable_output_validation
        
        executor.shutdown()
    
    def test_execute_tool_success(self):
        """Test successful tool execution."""
        params = {"param1": "test_value"}
        
        result = self.executor.execute_tool_with_timeout(self.mock_tool, params)
        
        assert result.success
        assert result.result == "mock result"
        assert result.execution_time > 0
        assert self.mock_tool.execute_called
        assert self.mock_tool.validate_called
    
    def test_execute_tool_safe_success(self):
        """Test successful tool execution with safe method."""
        params = {"param1": "test_value"}
        
        result = self.executor.execute_tool_safe(self.mock_tool, params)
        
        assert result.success
        assert result.result == "mock result"
        assert self.mock_tool.execute_called
    
    def test_execute_tool_timeout(self):
        """Test tool execution timeout."""
        slow_tool = SlowTool()
        params = {"delay": 2.0}
        
        with pytest.raises(ToolTimeoutError) as exc_info:
            self.executor.execute_tool_with_timeout(slow_tool, params, timeout=0.5)
        
        assert "timed out" in str(exc_info.value)
        assert exc_info.value.tool_name == "slow_tool"
    
    def test_execute_tool_safe_timeout(self):
        """Test tool execution timeout with safe method."""
        slow_tool = SlowTool()
        params = {"delay": 2.0}
        
        result = self.executor.execute_tool_safe(slow_tool, params, timeout=0.5)
        
        assert not result.success
        assert "Timeout" in result.error_message
        assert result.execution_time == 0.5
    
    def test_execute_tool_validation_error(self):
        """Test tool execution with validation error."""
        params = {"invalid": "params"}  # Missing required param1
        
        with pytest.raises(ToolValidationError) as exc_info:
            self.executor.execute_tool_with_timeout(self.mock_tool, params)
        
        assert "Invalid parameters" in str(exc_info.value)
        assert exc_info.value.tool_name == "mock_tool"
    
    def test_execute_tool_safe_validation_error(self):
        """Test tool execution validation error with safe method."""
        params = {"invalid": "params"}
        
        result = self.executor.execute_tool_safe(self.mock_tool, params)
        
        assert not result.success
        assert "Validation error" in result.error_message
    
    def test_execute_tool_execution_error(self):
        """Test tool execution with execution error."""
        failing_tool = MockTool(name="failing_tool", should_fail=True)
        params = {"param1": "test"}
        
        with pytest.raises(ToolExecutionError) as exc_info:
            self.executor.execute_tool_with_timeout(failing_tool, params)
        
        assert "execution failed" in str(exc_info.value)
        assert exc_info.value.tool_name == "failing_tool"
    
    def test_execute_tool_safe_execution_error(self):
        """Test tool execution error with safe method."""
        failing_tool = MockTool(name="failing_tool", should_fail=True)
        params = {"param1": "test"}
        
        result = self.executor.execute_tool_safe(failing_tool, params)
        
        assert not result.success
        assert "execution failed" in result.error_message
    
    def test_input_sanitization_valid(self):
        """Test input sanitization with valid inputs."""
        params = {
            "param1": "safe string",
            "param2": 123,
            "param3": [1, 2, 3],
            "param4": {"nested": "value"}
        }
        
        result = self.executor.execute_tool_safe(self.mock_tool, params)
        
        assert result.success
    
    def test_input_sanitization_dangerous_patterns(self):
        """Test input sanitization with dangerous patterns."""
        dangerous_params = [
            {"param1": "eval(malicious_code)"},
            {"param1": "import os"},
            {"param1": "__import__('os')"},
            {"param1": "exec('print(1)')"}
        ]
        
        for params in dangerous_params:
            result = self.executor.execute_tool_safe(self.mock_tool, params)
            assert not result.success
            assert "Validation error" in result.error_message
    
    def test_input_sanitization_size_limits(self):
        """Test input sanitization with size limits."""
        # Test string too long
        long_string = "a" * 20000  # Exceeds max_string_length
        params = {"param1": long_string}
        
        result = self.executor.execute_tool_safe(self.mock_tool, params)
        
        assert not result.success
        assert "Validation error" in result.error_message
    
    def test_input_sanitization_depth_limit(self):
        """Test input sanitization with depth limit."""
        # Create deeply nested dictionary
        nested_dict = {"param1": "value"}
        for i in range(15):  # Exceeds max_dict_depth
            nested_dict = {"level": nested_dict}
        
        result = self.executor.execute_tool_safe(self.mock_tool, nested_dict)
        
        assert not result.success
        assert "Validation error" in result.error_message
    
    def test_output_validation_invalid_type(self):
        """Test output validation with invalid result type."""
        invalid_tool = MockTool(name="invalid_tool", invalid_output=True)
        params = {"param1": "test"}
        
        with pytest.raises(ToolValidationError) as exc_info:
            self.executor.execute_tool_with_timeout(invalid_tool, params)
        
        assert "invalid result type" in str(exc_info.value)
    
    def test_output_validation_correction(self):
        """Test output validation with automatic correction."""
        # Create a tool that returns a result with invalid attributes
        class InvalidOutputTool(Tool):
            def get_info(self):
                return ToolInfo(name="invalid_output", description="Test")
            
            def execute(self, params):
                result = ToolResult(success="true", result="test")  # success should be bool
                result.execution_time = "invalid"  # should be number
                return result
            
            def validate_params(self, params):
                return True
        
        tool = InvalidOutputTool()
        params = {"param1": "test"}
        
        result = self.executor.execute_tool_safe(tool, params)
        
        assert result.success is True  # Should be corrected to boolean
        assert result.execution_time >= 0.0  # Should be corrected to non-negative float
    
    def test_execution_history_tracking(self):
        """Test execution history tracking."""
        params = {"param1": "test"}
        
        # Execute multiple tools
        self.executor.execute_tool_safe(self.mock_tool, params)
        self.executor.execute_tool_safe(MockTool(name="tool2"), params)
        
        history = self.executor.get_execution_history()
        
        assert len(history) == 2
        assert history[0]['tool_name'] == "mock_tool"
        assert history[1]['tool_name'] == "tool2"
        assert all('start_time' in record for record in history)
        assert all('status' in record for record in history)
    
    def test_execution_history_limit(self):
        """Test execution history with limit."""
        params = {"param1": "test"}
        
        # Execute multiple tools
        for i in range(5):
            self.executor.execute_tool_safe(MockTool(name=f"tool{i}"), params)
        
        # Get limited history
        history = self.executor.get_execution_history(limit=3)
        
        assert len(history) == 3
        # Should return the last 3 executions
        assert history[0]['tool_name'] == "tool2"
        assert history[1]['tool_name'] == "tool3"
        assert history[2]['tool_name'] == "tool4"
    
    def test_execution_stats(self):
        """Test execution statistics."""
        params = {"param1": "test"}
        
        # Execute successful tools
        self.executor.execute_tool_safe(self.mock_tool, params)
        self.executor.execute_tool_safe(MockTool(name="tool2"), params)
        
        # Execute failing tool
        self.executor.execute_tool_safe(MockTool(name="failing", should_fail=True), params)
        
        # Execute timeout tool
        self.executor.execute_tool_safe(SlowTool(), {"delay": 2.0}, timeout=0.1)
        
        stats = self.executor.get_execution_stats()
        
        assert stats['total_executions'] == 4
        assert stats['successful_executions'] == 2
        assert stats['failed_executions'] >= 1
        assert stats['timeout_executions'] >= 1
        assert 0 <= stats['success_rate'] <= 1
        assert stats['average_execution_time'] >= 0
    
    def test_execution_stats_empty(self):
        """Test execution statistics with no executions."""
        stats = self.executor.get_execution_stats()
        
        assert stats['total_executions'] == 0
        assert stats['successful_executions'] == 0
        assert stats['failed_executions'] == 0
        assert stats['timeout_executions'] == 0
        assert stats['success_rate'] == 0.0
        assert stats['average_execution_time'] == 0.0
    
    def test_clear_history(self):
        """Test clearing execution history."""
        params = {"param1": "test"}
        
        # Execute some tools
        self.executor.execute_tool_safe(self.mock_tool, params)
        self.executor.execute_tool_safe(MockTool(name="tool2"), params)
        
        assert len(self.executor.get_execution_history()) == 2
        
        # Clear history
        self.executor.clear_history()
        
        assert len(self.executor.get_execution_history()) == 0
        
        # Stats should also be reset
        stats = self.executor.get_execution_stats()
        assert stats['total_executions'] == 0
    
    def test_concurrent_execution(self):
        """Test concurrent tool execution."""
        params = {"param1": "test"}
        
        # Create multiple tools with different execution times
        tools = [
            MockTool(name=f"tool{i}", execution_time=0.1) 
            for i in range(3)
        ]
        
        # Execute concurrently using threads
        import threading
        results = []
        threads = []
        
        def execute_tool(tool):
            result = self.executor.execute_tool_safe(tool, params)
            results.append(result)
        
        start_time = time.time()
        
        for tool in tools:
            thread = threading.Thread(target=execute_tool, args=(tool,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # All executions should succeed
        assert len(results) == 3
        assert all(result.success for result in results)
        
        # Concurrent execution should be faster than sequential
        # (allowing some overhead for thread management)
        assert total_time < 0.5  # Should be much less than 3 * 0.1 = 0.3s
    
    def test_sanitization_disabled(self):
        """Test execution with sanitization disabled."""
        executor = ToolExecutor(enable_input_sanitization=False)
        
        # This would normally fail sanitization
        params = {"param1": "eval(dangerous_code)"}
        
        result = executor.execute_tool_safe(self.mock_tool, params)
        
        # Should succeed because sanitization is disabled
        assert result.success
        
        executor.shutdown()
    
    def test_output_validation_disabled(self):
        """Test execution with output validation disabled."""
        executor = ToolExecutor(enable_output_validation=False)
        
        invalid_tool = MockTool(name="invalid_tool", invalid_output=True)
        params = {"param1": "test"}
        
        # This should not raise an exception because validation is disabled
        result = executor.execute_tool_safe(invalid_tool, params)
        
        # The result should still fail because the tool returns invalid type
        assert not result.success
        
        executor.shutdown()
    
    def test_custom_timeout(self):
        """Test execution with custom timeout."""
        slow_tool = SlowTool()
        params = {"delay": 0.3}
        
        # Should succeed with longer timeout
        result = self.executor.execute_tool_safe(slow_tool, params, timeout=1.0)
        assert result.success
        
        # Should timeout with shorter timeout
        result = self.executor.execute_tool_safe(slow_tool, params, timeout=0.1)
        assert not result.success
        assert "Timeout" in result.error_message