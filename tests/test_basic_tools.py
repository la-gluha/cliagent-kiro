"""
Unit tests for basic tool implementations.

This module contains comprehensive tests for FileOperationsTool,
CalculatorTool, and WebSearchTool.
"""

import os
import tempfile
import pytest
import math
from unittest.mock import patch, mock_open
from src.tools.basic_tools import FileOperationsTool, CalculatorTool, WebSearchTool
from src.models.data_models import ToolCategory


class TestFileOperationsTool:
    """Test cases for the FileOperationsTool class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.tool = FileOperationsTool()
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        self.test_content = "Hello, World!\nThis is a test file."
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Clean up temp files
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_get_info(self):
        """Test getting tool information."""
        info = self.tool.get_info()
        
        assert info.name == "file_operations"
        assert info.category == ToolCategory.FILE_OPERATIONS
        assert info.enabled
        assert "operation" in info.parameters
        assert "path" in info.parameters
    
    def test_validate_params_valid(self):
        """Test parameter validation with valid parameters."""
        valid_params = [
            {"operation": "read", "path": "/test/file.txt"},
            {"operation": "write", "path": "/test/file.txt", "content": "test"},
            {"operation": "list", "path": "/test/dir"},
            {"operation": "exists", "path": "/test/file.txt"},
            {"operation": "delete", "path": "/test/file.txt"}
        ]
        
        for params in valid_params:
            assert self.tool.validate_params(params)
    
    def test_validate_params_invalid(self):
        """Test parameter validation with invalid parameters."""
        invalid_params = [
            {},  # Missing required params
            {"operation": "read"},  # Missing path
            {"path": "/test/file.txt"},  # Missing operation
            {"operation": "invalid", "path": "/test/file.txt"},  # Invalid operation
            {"operation": "write", "path": "/test/file.txt"},  # Missing content for write
            {"operation": "read", "path": ""},  # Empty path
            {"operation": "read", "path": 123},  # Non-string path
            "not a dict"  # Not a dictionary
        ]
        
        for params in invalid_params:
            assert not self.tool.validate_params(params)
    
    def test_write_and_read_file(self):
        """Test writing and reading a file."""
        # Write file
        write_params = {
            "operation": "write",
            "path": self.test_file,
            "content": self.test_content
        }
        
        write_result = self.tool.execute(write_params)
        
        assert write_result.success
        assert write_result.result["path"] == self.test_file
        assert write_result.result["bytes_written"] > 0
        
        # Read file
        read_params = {
            "operation": "read",
            "path": self.test_file
        }
        
        read_result = self.tool.execute(read_params)
        
        assert read_result.success
        assert read_result.result["content"] == self.test_content
        assert read_result.result["path"] == self.test_file
        assert read_result.result["size"] == len(self.test_content)
    
    def test_read_nonexistent_file(self):
        """Test reading a non-existent file."""
        params = {
            "operation": "read",
            "path": "/nonexistent/file.txt"
        }
        
        result = self.tool.execute(params)
        
        assert not result.success
        assert "File not found" in result.error_message
    
    def test_read_directory_as_file(self):
        """Test reading a directory as if it were a file."""
        params = {
            "operation": "read",
            "path": self.temp_dir
        }
        
        result = self.tool.execute(params)
        
        assert not result.success
        assert "Path is not a file" in result.error_message
    
    def test_list_directory(self):
        """Test listing directory contents."""
        # Create test files
        test_files = ["file1.txt", "file2.txt"]
        for filename in test_files:
            filepath = os.path.join(self.temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write("test content")
        
        params = {
            "operation": "list",
            "path": self.temp_dir
        }
        
        result = self.tool.execute(params)
        
        assert result.success
        assert result.result["path"] == self.temp_dir
        assert result.result["count"] == len(test_files)
        
        item_names = [item["name"] for item in result.result["items"]]
        for filename in test_files:
            assert filename in item_names
        
        # Clean up
        for filename in test_files:
            os.remove(os.path.join(self.temp_dir, filename))
    
    def test_list_nonexistent_directory(self):
        """Test listing a non-existent directory."""
        params = {
            "operation": "list",
            "path": "/nonexistent/directory"
        }
        
        result = self.tool.execute(params)
        
        assert not result.success
        assert "Directory not found" in result.error_message
    
    def test_check_exists(self):
        """Test checking file existence."""
        # Create test file
        with open(self.test_file, 'w') as f:
            f.write("test")
        
        # Test existing file
        params = {
            "operation": "exists",
            "path": self.test_file
        }
        
        result = self.tool.execute(params)
        
        assert result.success
        assert result.result["exists"]
        assert result.result["is_file"]
        assert not result.result["is_directory"]
        
        # Test non-existing file
        params["path"] = "/nonexistent/file.txt"
        result = self.tool.execute(params)
        
        assert result.success
        assert not result.result["exists"]
    
    def test_delete_file(self):
        """Test deleting a file."""
        # Create test file
        with open(self.test_file, 'w') as f:
            f.write("test")
        
        assert os.path.exists(self.test_file)
        
        params = {
            "operation": "delete",
            "path": self.test_file
        }
        
        result = self.tool.execute(params)
        
        assert result.success
        assert not os.path.exists(self.test_file)
        assert "Successfully deleted" in result.result["message"]
    
    def test_delete_nonexistent_file(self):
        """Test deleting a non-existent file."""
        params = {
            "operation": "delete",
            "path": "/nonexistent/file.txt"
        }
        
        result = self.tool.execute(params)
        
        assert not result.success
        assert "File not found" in result.error_message


class TestCalculatorTool:
    """Test cases for the CalculatorTool class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.tool = CalculatorTool()
    
    def test_get_info(self):
        """Test getting tool information."""
        info = self.tool.get_info()
        
        assert info.name == "calculator"
        assert info.category == ToolCategory.CALCULATION
        assert info.enabled
        assert "expression" in info.parameters
    
    def test_validate_params_valid(self):
        """Test parameter validation with valid parameters."""
        valid_params = [
            {"expression": "2 + 3"},
            {"expression": "sin(pi/2)"},
            {"expression": "sqrt(16)"},
            {"expression": "log(100)"}
        ]
        
        for params in valid_params:
            assert self.tool.validate_params(params)
    
    def test_validate_params_invalid(self):
        """Test parameter validation with invalid parameters."""
        invalid_params = [
            {},  # Missing expression
            {"expression": ""},  # Empty expression
            {"expression": 123},  # Non-string expression
            {"expression": None},  # None expression
            "not a dict"  # Not a dictionary
        ]
        
        for params in invalid_params:
            assert not self.tool.validate_params(params)
    
    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        test_cases = [
            ("2 + 3", 5),
            ("10 - 4", 6),
            ("3 * 4", 12),
            ("15 / 3", 5),
            ("2 ** 3", 8),
            ("17 % 5", 2)
        ]
        
        for expression, expected in test_cases:
            params = {"expression": expression}
            result = self.tool.execute(params)
            
            assert result.success
            assert result.result["result"] == expected
            assert result.result["expression"] == expression
    
    def test_mathematical_functions(self):
        """Test mathematical functions."""
        test_cases = [
            ("sqrt(16)", 4.0),
            ("abs(-5)", 5),
            ("round(3.7)", 4),
            ("max(1, 2, 3)", 3),
            ("min(1, 2, 3)", 1)
        ]
        
        for expression, expected in test_cases:
            params = {"expression": expression}
            result = self.tool.execute(params)
            
            assert result.success
            assert result.result["result"] == expected
    
    def test_trigonometric_functions(self):
        """Test trigonometric functions."""
        test_cases = [
            ("sin(0)", 0.0),
            ("cos(0)", 1.0),
            ("tan(0)", 0.0)
        ]
        
        for expression, expected in test_cases:
            params = {"expression": expression}
            result = self.tool.execute(params)
            
            assert result.success
            assert abs(result.result["result"] - expected) < 1e-10
    
    def test_constants(self):
        """Test mathematical constants."""
        params = {"expression": "pi"}
        result = self.tool.execute(params)
        
        assert result.success
        assert abs(result.result["result"] - math.pi) < 1e-10
        
        params = {"expression": "e"}
        result = self.tool.execute(params)
        
        assert result.success
        assert abs(result.result["result"] - math.e) < 1e-10
    
    def test_complex_expressions(self):
        """Test complex mathematical expressions."""
        test_cases = [
            ("2 + 3 * 4", 14),  # Order of operations
            ("(2 + 3) * 4", 20),  # Parentheses
            ("sqrt(16) + 2**3", 12),  # Mixed functions and operators
            ("sin(pi/2) + cos(0)", 2.0)  # Trigonometric with constants
        ]
        
        for expression, expected in test_cases:
            params = {"expression": expression}
            result = self.tool.execute(params)
            
            assert result.success
            assert abs(result.result["result"] - expected) < 1e-10
    
    def test_power_operator_conversion(self):
        """Test conversion of ^ to ** for power operations."""
        params = {"expression": "2^3"}
        result = self.tool.execute(params)
        
        assert result.success
        assert result.result["result"] == 8
    
    def test_dangerous_expressions(self):
        """Test that dangerous expressions are rejected."""
        dangerous_expressions = [
            "import os",
            "exec('print(1)')",
            "eval('1+1')",
            "open('file.txt')",
            "__import__('os')"
        ]
        
        for expression in dangerous_expressions:
            params = {"expression": expression}
            result = self.tool.execute(params)
            
            assert not result.success
            assert "dangerous" in result.error_message.lower() or "not allowed" in result.error_message.lower()
    
    def test_invalid_expressions(self):
        """Test invalid mathematical expressions."""
        invalid_expressions = [
            "2 +",  # Incomplete expression
            "unknown_function(1)",  # Unknown function
            "1 / 0",  # Division by zero
            "sqrt(-1)"  # Invalid operation (for real numbers)
        ]
        
        for expression in invalid_expressions:
            params = {"expression": expression}
            result = self.tool.execute(params)
            
            assert not result.success
            assert "failed" in result.error_message.lower()


class TestWebSearchTool:
    """Test cases for the WebSearchTool class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.tool = WebSearchTool()
    
    def test_get_info(self):
        """Test getting tool information."""
        info = self.tool.get_info()
        
        assert info.name == "web_search"
        assert info.category == ToolCategory.WEB_SEARCH
        assert info.enabled
        assert "query" in info.parameters
        assert "limit" in info.parameters
    
    def test_validate_params_valid(self):
        """Test parameter validation with valid parameters."""
        valid_params = [
            {"query": "python programming"},
            {"query": "machine learning", "limit": 3},
            {"query": "web development", "limit": 10}
        ]
        
        for params in valid_params:
            assert self.tool.validate_params(params)
    
    def test_validate_params_invalid(self):
        """Test parameter validation with invalid parameters."""
        invalid_params = [
            {},  # Missing query
            {"query": ""},  # Empty query
            {"query": 123},  # Non-string query
            {"query": "test", "limit": 0},  # Invalid limit (too low)
            {"query": "test", "limit": 25},  # Invalid limit (too high)
            {"query": "test", "limit": "5"},  # Non-integer limit
            "not a dict"  # Not a dictionary
        ]
        
        for params in invalid_params:
            assert not self.tool.validate_params(params)
    
    def test_execute_search_default_limit(self):
        """Test executing search with default limit."""
        params = {"query": "python programming"}
        result = self.tool.execute(params)
        
        assert result.success
        assert result.result["query"] == "python programming"
        assert len(result.result["results"]) == 5  # Default limit
        assert result.result["total_results"] == 5
        assert "mock implementation" in result.result["note"]
    
    def test_execute_search_custom_limit(self):
        """Test executing search with custom limit."""
        params = {"query": "machine learning", "limit": 3}
        result = self.tool.execute(params)
        
        assert result.success
        assert result.result["query"] == "machine learning"
        assert len(result.result["results"]) == 3
        assert result.result["total_results"] == 3
    
    def test_mock_results_structure(self):
        """Test that mock results have the expected structure."""
        params = {"query": "test query", "limit": 2}
        result = self.tool.execute(params)
        
        assert result.success
        
        for search_result in result.result["results"]:
            assert "title" in search_result
            assert "url" in search_result
            assert "snippet" in search_result
            assert "source" in search_result
            
            # Check that query is incorporated into results
            assert "test query" in search_result["title"] or "test query" in search_result["snippet"]
    
    def test_url_encoding(self):
        """Test that URLs are properly encoded."""
        params = {"query": "test query with spaces", "limit": 1}
        result = self.tool.execute(params)
        
        assert result.success
        
        # Check that URLs contain encoded query
        search_result = result.result["results"][0]
        assert "test+query+with+spaces" in search_result["url"] or "test%20query%20with%20spaces" in search_result["url"]
    
    def test_different_query_types(self):
        """Test search with different types of queries."""
        test_queries = [
            "simple query",
            "query with numbers 123",
            "query-with-hyphens",
            "query with special chars !@#"
        ]
        
        for query in test_queries:
            params = {"query": query, "limit": 1}
            result = self.tool.execute(params)
            
            assert result.success
            assert result.result["query"] == query
            assert len(result.result["results"]) == 1