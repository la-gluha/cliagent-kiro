"""
Basic tool implementations for the AI Agent System.

This module provides fundamental tools including file operations,
calculator, and web search functionality.
"""

import os
import json
import math
import operator
import re
from typing import Any, Dict
from urllib.parse import quote_plus
from ..interfaces.tool_interface import Tool, ToolResult
from ..models.data_models import ToolInfo, ToolCategory


class FileOperationsTool(Tool):
    """
    Tool for basic file operations like reading, writing, and listing files.
    """
    
    def get_info(self) -> ToolInfo:
        """Get information about this tool."""
        return ToolInfo(
            name="file_operations",
            description="Perform basic file operations including read, write, list, and check existence",
            parameters={
                "operation": {
                    "type": "string",
                    "description": "The operation to perform",
                    "enum": ["read", "write", "list", "exists", "delete"]
                },
                "path": {
                    "type": "string",
                    "description": "The file or directory path"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (required for write operation)",
                    "required_for": ["write"]
                }
            },
            category=ToolCategory.FILE_OPERATIONS,
            enabled=True
        )
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate the parameters for this tool."""
        if not isinstance(params, dict):
            return False
        
        # Check required parameters
        if "operation" not in params or "path" not in params:
            return False
        
        operation = params["operation"]
        if operation not in ["read", "write", "list", "exists", "delete"]:
            return False
        
        # Check if content is provided for write operation
        if operation == "write" and "content" not in params:
            return False
        
        # Validate path
        path = params["path"]
        if not isinstance(path, str) or not path.strip():
            return False
        
        return True
    
    def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute the file operation."""
        operation = params["operation"]
        path = params["path"]
        
        try:
            if operation == "read":
                return self._read_file(path)
            elif operation == "write":
                content = params["content"]
                return self._write_file(path, content)
            elif operation == "list":
                return self._list_directory(path)
            elif operation == "exists":
                return self._check_exists(path)
            elif operation == "delete":
                return self._delete_file(path)
            else:
                return ToolResult(
                    success=False,
                    error_message=f"Unknown operation: {operation}"
                )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"File operation failed: {str(e)}"
            )
    
    def _read_file(self, path: str) -> ToolResult:
        """Read a file and return its contents."""
        if not os.path.exists(path):
            return ToolResult(
                success=False,
                error_message=f"File not found: {path}"
            )
        
        if not os.path.isfile(path):
            return ToolResult(
                success=False,
                error_message=f"Path is not a file: {path}"
            )
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return ToolResult(
                success=True,
                result={
                    "content": content,
                    "size": len(content),
                    "path": path
                }
            )
        except UnicodeDecodeError:
            # Try binary mode for non-text files
            with open(path, 'rb') as f:
                content = f.read()
            
            return ToolResult(
                success=True,
                result={
                    "content": f"<binary file, {len(content)} bytes>",
                    "size": len(content),
                    "path": path,
                    "binary": True
                }
            )
    
    def _write_file(self, path: str, content: str) -> ToolResult:
        """Write content to a file."""
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return ToolResult(
                success=True,
                result={
                    "path": path,
                    "bytes_written": len(content.encode('utf-8')),
                    "message": f"Successfully wrote to {path}"
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Failed to write file: {str(e)}"
            )
    
    def _list_directory(self, path: str) -> ToolResult:
        """List contents of a directory."""
        if not os.path.exists(path):
            return ToolResult(
                success=False,
                error_message=f"Directory not found: {path}"
            )
        
        if not os.path.isdir(path):
            return ToolResult(
                success=False,
                error_message=f"Path is not a directory: {path}"
            )
        
        try:
            items = []
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                items.append({
                    "name": item,
                    "path": item_path,
                    "is_file": os.path.isfile(item_path),
                    "is_directory": os.path.isdir(item_path),
                    "size": os.path.getsize(item_path) if os.path.isfile(item_path) else None
                })
            
            return ToolResult(
                success=True,
                result={
                    "path": path,
                    "items": items,
                    "count": len(items)
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Failed to list directory: {str(e)}"
            )
    
    def _check_exists(self, path: str) -> ToolResult:
        """Check if a file or directory exists."""
        exists = os.path.exists(path)
        result = {
            "path": path,
            "exists": exists
        }
        
        if exists:
            result.update({
                "is_file": os.path.isfile(path),
                "is_directory": os.path.isdir(path),
                "size": os.path.getsize(path) if os.path.isfile(path) else None
            })
        
        return ToolResult(success=True, result=result)
    
    def _delete_file(self, path: str) -> ToolResult:
        """Delete a file."""
        if not os.path.exists(path):
            return ToolResult(
                success=False,
                error_message=f"File not found: {path}"
            )
        
        if not os.path.isfile(path):
            return ToolResult(
                success=False,
                error_message=f"Path is not a file: {path}"
            )
        
        try:
            os.remove(path)
            return ToolResult(
                success=True,
                result={
                    "path": path,
                    "message": f"Successfully deleted {path}"
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Failed to delete file: {str(e)}"
            )


class CalculatorTool(Tool):
    """
    Tool for performing mathematical calculations.
    """
    
    def get_info(self) -> ToolInfo:
        """Get information about this tool."""
        return ToolInfo(
            name="calculator",
            description="Perform mathematical calculations including basic arithmetic, trigonometry, and logarithms",
            parameters={
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4', 'sin(pi/2)', 'log(100)')"
                }
            },
            category=ToolCategory.CALCULATION,
            enabled=True
        )
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate the parameters for this tool."""
        if not isinstance(params, dict):
            return False
        
        if "expression" not in params:
            return False
        
        expression = params["expression"]
        if not isinstance(expression, str) or not expression.strip():
            return False
        
        return True
    
    def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute the calculation."""
        expression = params["expression"].strip()
        
        try:
            # Sanitize and evaluate the expression
            result = self._safe_eval(expression)
            
            return ToolResult(
                success=True,
                result={
                    "expression": expression,
                    "result": result,
                    "type": type(result).__name__
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Calculation failed: {str(e)}"
            )
    
    def _safe_eval(self, expression: str) -> float:
        """Safely evaluate a mathematical expression."""
        # Define allowed names for evaluation
        allowed_names = {
            # Math functions
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,
            "log": math.log,
            "log10": math.log10,
            "log2": math.log2,
            "exp": math.exp,
            "ceil": math.ceil,
            "floor": math.floor,
            "degrees": math.degrees,
            "radians": math.radians,
            # Constants
            "pi": math.pi,
            "e": math.e,
            "tau": math.tau,
            # Operators
            "__add__": operator.add,
            "__sub__": operator.sub,
            "__mul__": operator.mul,
            "__truediv__": operator.truediv,
            "__floordiv__": operator.floordiv,
            "__mod__": operator.mod,
            "__pow__": operator.pow,
        }
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'__\w+__',  # Dunder methods (except allowed ones)
            r'import',
            r'exec',
            r'eval',
            r'open',
            r'file',
            r'input',
            r'raw_input'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                if pattern not in ['__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__', '__mod__', '__pow__']:
                    raise ValueError(f"Potentially dangerous expression: contains '{pattern}'")
        
        # Replace common mathematical notation
        expression = expression.replace('^', '**')  # Power operator
        
        try:
            # Compile and evaluate the expression
            code = compile(expression, '<string>', 'eval')
            
            # Check that only allowed names are used
            for name in code.co_names:
                if name not in allowed_names:
                    raise ValueError(f"Name '{name}' is not allowed in expressions")
            
            result = eval(code, {"__builtins__": {}}, allowed_names)
            
            # Convert result to appropriate type
            if isinstance(result, (int, float, complex)):
                return result
            else:
                return float(result)
                
        except Exception as e:
            raise ValueError(f"Invalid mathematical expression: {str(e)}")


class WebSearchTool(Tool):
    """
    Tool for performing web searches (mock implementation).
    
    Note: This is a mock implementation for demonstration purposes.
    In a real system, this would integrate with actual search APIs.
    """
    
    def get_info(self) -> ToolInfo:
        """Get information about this tool."""
        return ToolInfo(
            name="web_search",
            description="Search the web for information (mock implementation)",
            parameters={
                "query": {
                    "type": "string",
                    "description": "Search query to execute"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                    "default": 5
                }
            },
            category=ToolCategory.WEB_SEARCH,
            enabled=True
        )
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate the parameters for this tool."""
        if not isinstance(params, dict):
            return False
        
        if "query" not in params:
            return False
        
        query = params["query"]
        if not isinstance(query, str) or not query.strip():
            return False
        
        # Validate limit if provided
        if "limit" in params:
            limit = params["limit"]
            if not isinstance(limit, int) or limit < 1 or limit > 20:
                return False
        
        return True
    
    def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute the web search (mock implementation)."""
        query = params["query"].strip()
        limit = params.get("limit", 5)
        
        try:
            # Mock search results
            mock_results = self._generate_mock_results(query, limit)
            
            return ToolResult(
                success=True,
                result={
                    "query": query,
                    "results": mock_results,
                    "total_results": len(mock_results),
                    "note": "This is a mock implementation. In a real system, this would use actual search APIs."
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Web search failed: {str(e)}"
            )
    
    def _generate_mock_results(self, query: str, limit: int) -> list:
        """Generate mock search results for demonstration."""
        # Create mock results based on the query
        base_results = [
            {
                "title": f"Search result for '{query}' - Article 1",
                "url": f"https://example.com/article1?q={quote_plus(query)}",
                "snippet": f"This is a mock search result for the query '{query}'. It contains relevant information about the topic.",
                "source": "example.com"
            },
            {
                "title": f"'{query}' - Wikipedia",
                "url": f"https://en.wikipedia.org/wiki/{quote_plus(query)}",
                "snippet": f"Wikipedia article about {query}. Comprehensive information and references.",
                "source": "wikipedia.org"
            },
            {
                "title": f"How to understand {query}",
                "url": f"https://tutorial.com/how-to-{query.lower().replace(' ', '-')}",
                "snippet": f"A comprehensive guide to understanding {query} with examples and explanations.",
                "source": "tutorial.com"
            },
            {
                "title": f"{query} - Latest News",
                "url": f"https://news.com/latest/{quote_plus(query)}",
                "snippet": f"Latest news and updates about {query} from reliable sources.",
                "source": "news.com"
            },
            {
                "title": f"Best practices for {query}",
                "url": f"https://bestpractices.com/{query.lower().replace(' ', '-')}",
                "snippet": f"Expert recommendations and best practices related to {query}.",
                "source": "bestpractices.com"
            }
        ]
        
        # Return only the requested number of results
        return base_results[:limit]