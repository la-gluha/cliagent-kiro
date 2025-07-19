"""
Unit tests for the display system.

Tests the DisplayManager, OutputFormatter, and related functionality
to ensure proper formatting and display operations.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from src.display.display_manager import DisplayManager
from src.display.output_formatter import OutputFormatter, ColorCodes
from src.interfaces.display_interface import OutputStyle, AnimationType
from src.models.data_models import Message, MessageRole, TaskResult, ToolInfo, ToolCategory, AgentState


class TestOutputFormatter:
    """Test cases for the OutputFormatter class."""
    
    def test_init_with_color_enabled(self):
        """Test formatter initialization with color enabled."""
        formatter = OutputFormatter(color_enabled=True)
        assert formatter.color_enabled is True
    
    def test_init_with_color_disabled(self):
        """Test formatter initialization with color disabled."""
        formatter = OutputFormatter(color_enabled=False)
        assert formatter.color_enabled is False
    
    def test_set_color_enabled(self):
        """Test enabling/disabling color output."""
        formatter = OutputFormatter(color_enabled=True)
        formatter.set_color_enabled(False)
        assert formatter.color_enabled is False
        
        formatter.set_color_enabled(True)
        assert formatter.color_enabled is True
    
    def test_apply_style_with_color_enabled(self):
        """Test applying styles with color enabled."""
        formatter = OutputFormatter(color_enabled=True)
        
        # Test success style
        result = formatter.apply_style("test", OutputStyle.SUCCESS)
        assert ColorCodes.BRIGHT_GREEN in result
        assert ColorCodes.RESET in result
        assert "test" in result
        
        # Test error style
        result = formatter.apply_style("error", OutputStyle.ERROR)
        assert ColorCodes.BRIGHT_RED in result
        assert ColorCodes.RESET in result
    
    def test_apply_style_with_color_disabled(self):
        """Test applying styles with color disabled."""
        formatter = OutputFormatter(color_enabled=False)
        
        result = formatter.apply_style("test", OutputStyle.SUCCESS)
        assert result == "test"
        assert ColorCodes.BRIGHT_GREEN not in result
    
    def test_format_message(self):
        """Test formatting a message."""
        formatter = OutputFormatter(color_enabled=False)
        
        message = Message(
            content="Hello, world!",
            role=MessageRole.USER,
            timestamp=datetime(2023, 1, 1, 12, 0, 0)
        )
        
        result = formatter.format_message(message)
        assert "[12:00:00]" in result
        assert "USER:" in result
        assert "Hello, world!" in result
    
    def test_format_message_multiline(self):
        """Test formatting a multiline message."""
        formatter = OutputFormatter(color_enabled=False)
        
        message = Message(
            content="Line 1\nLine 2\nLine 3",
            role=MessageRole.AGENT
        )
        
        result = formatter.format_message(message)
        lines = result.split('\n')
        
        # Check that content lines are properly indented
        assert "  Line 1" in result
        assert "  Line 2" in result
        assert "  Line 3" in result
    
    def test_format_task_result_success(self):
        """Test formatting a successful task result."""
        formatter = OutputFormatter(color_enabled=False)
        
        result = TaskResult(
            success=True,
            result="Task completed",
            execution_time=1.5,
            steps_taken=["Step 1", "Step 2"]
        )
        
        formatted = formatter.format_task_result(result)
        assert "SUCCESS" in formatted
        assert "1.50s" in formatted
        assert "Step 1" in formatted
        assert "Step 2" in formatted
        assert "Task completed" in formatted
    
    def test_format_task_result_failure(self):
        """Test formatting a failed task result."""
        formatter = OutputFormatter(color_enabled=False)
        
        result = TaskResult(
            success=False,
            error_message="Something went wrong",
            execution_time=0.5
        )
        
        formatted = formatter.format_task_result(result)
        assert "FAILED" in formatted
        assert "Something went wrong" in formatted
        assert "0.50s" in formatted
    
    def test_format_tool_info(self):
        """Test formatting tool information."""
        formatter = OutputFormatter(color_enabled=False)
        
        tool_info = ToolInfo(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.CALCULATION,
            enabled=True,
            parameters={
                "param1": {"type": "string", "required": True},
                "param2": {"type": "number", "required": False}
            }
        )
        
        formatted = formatter.format_tool_info(tool_info)
        assert "test_tool" in formatted
        assert "ENABLED" in formatted
        assert "A test tool" in formatted
        assert "calculation" in formatted
        assert "param1" in formatted
        assert "param2" in formatted
        assert "required" in formatted
        assert "optional" in formatted
    
    def test_format_agent_state(self):
        """Test formatting agent state."""
        formatter = OutputFormatter(color_enabled=False)
        
        state = AgentState(
            current_task="Test task",
            is_active=True,
            reasoning_history=["Thought 1", "Thought 2", "Thought 3"],
            action_history=[{"action": "test"}]
        )
        
        formatted = formatter.format_agent_state(state)
        assert "ACTIVE" in formatted
        assert "Test task" in formatted
        assert "Recent Reasoning:" in formatted
        assert "Actions Taken: 1" in formatted
    
    def test_format_json(self):
        """Test JSON formatting."""
        formatter = OutputFormatter(color_enabled=False)
        
        data = {"key": "value", "number": 42, "boolean": True}
        result = formatter.format_json(data)
        
        assert '"key"' in result
        assert '"value"' in result
        assert "42" in result
        assert "true" in result
    
    def test_format_json_with_color(self):
        """Test JSON formatting with color."""
        formatter = OutputFormatter(color_enabled=True)
        
        data = {"key": "value", "boolean": True}
        result = formatter.format_json(data)
        
        # Should contain color codes for syntax highlighting
        assert ColorCodes.CYAN in result or ColorCodes.GREEN in result
    
    def test_format_list_numbered(self):
        """Test formatting a numbered list."""
        formatter = OutputFormatter(color_enabled=False)
        
        items = ["Item 1", "Item 2", "Item 3"]
        result = formatter.format_list(items, numbered=True)
        
        assert "1. Item 1" in result
        assert "2. Item 2" in result
        assert "3. Item 3" in result
    
    def test_format_list_bulleted(self):
        """Test formatting a bulleted list."""
        formatter = OutputFormatter(color_enabled=False)
        
        items = ["Item 1", "Item 2"]
        result = formatter.format_list(items, numbered=False)
        
        assert "• Item 1" in result
        assert "• Item 2" in result
    
    def test_format_list_empty(self):
        """Test formatting an empty list."""
        formatter = OutputFormatter(color_enabled=False)
        
        result = formatter.format_list([])
        assert "(empty list)" in result
    
    def test_format_table(self):
        """Test formatting a table."""
        formatter = OutputFormatter(color_enabled=False)
        
        headers = ["Name", "Age", "City"]
        rows = [
            ["Alice", "25", "New York"],
            ["Bob", "30", "London"]
        ]
        
        result = formatter.format_table(headers, rows)
        
        assert "Name" in result
        assert "Age" in result
        assert "City" in result
        assert "Alice" in result
        assert "Bob" in result
        assert "|" in result  # Table separator
        assert "-" in result  # Header separator
    
    def test_format_table_empty(self):
        """Test formatting an empty table."""
        formatter = OutputFormatter(color_enabled=False)
        
        result = formatter.format_table([], [])
        assert "(empty table)" in result
    
    def test_create_separator(self):
        """Test creating a separator."""
        formatter = OutputFormatter(color_enabled=False)
        
        result = formatter.create_separator("-", 10)
        assert result == "----------"
        
        result = formatter.create_separator("=", 5)
        assert result == "====="
    
    def test_create_header(self):
        """Test creating a header."""
        formatter = OutputFormatter(color_enabled=False)
        
        result = formatter.create_header("Test Header")
        lines = result.split('\n')
        
        assert len(lines) == 3
        assert "Test Header" in lines[1]
        assert "=" * len("Test Header") in lines[0]
        assert "=" * len("Test Header") in lines[2]


class TestDisplayManager:
    """Test cases for the DisplayManager class."""
    
    def test_init_with_color_auto_detect(self):
        """Test display manager initialization with auto color detection."""
        with patch.object(DisplayManager, '_supports_color', return_value=True):
            manager = DisplayManager()
            assert manager.color_enabled is True
        
        with patch.object(DisplayManager, '_supports_color', return_value=False):
            manager = DisplayManager()
            assert manager.color_enabled is False
    
    def test_init_with_explicit_color_setting(self):
        """Test display manager initialization with explicit color setting."""
        manager = DisplayManager(color_enabled=True)
        assert manager.color_enabled is True
        
        manager = DisplayManager(color_enabled=False)
        assert manager.color_enabled is False
    
    @patch('builtins.print')
    def test_show_message(self, mock_print):
        """Test showing a message."""
        manager = DisplayManager(color_enabled=False)
        
        manager.show_message("Test message", OutputStyle.INFO)
        mock_print.assert_called_once()
        
        # Check that the message was formatted
        call_args = mock_print.call_args[0][0]
        assert "Test message" in call_args
    
    @patch('builtins.print')
    def test_show_progress(self, mock_print):
        """Test showing progress."""
        manager = DisplayManager(color_enabled=False)
        
        manager.show_progress("Testing", 0.5)
        mock_print.assert_called()
        
        # Check that progress was displayed
        call_args = mock_print.call_args[1]  # kwargs
        assert call_args.get('end') == ""  # Should use carriage return
        assert call_args.get('flush') is True
    
    @patch('builtins.print')
    def test_show_progress_complete(self, mock_print):
        """Test showing complete progress."""
        manager = DisplayManager(color_enabled=False)
        
        manager.show_progress("Testing", 1.0)
        
        # Should print newline when complete
        calls = mock_print.call_args_list
        assert len(calls) >= 1
    
    @patch('builtins.print')
    def test_show_animation(self, mock_print):
        """Test showing animation."""
        manager = DisplayManager(color_enabled=False)
        
        manager.show_animation(AnimationType.SPINNER, "Loading...")
        mock_print.assert_called_once()
        
        call_args = mock_print.call_args[0][0]
        assert "Loading..." in call_args
    
    def test_format_output_json(self):
        """Test formatting output as JSON."""
        manager = DisplayManager(color_enabled=False)
        
        data = {"key": "value"}
        result = manager.format_output(data, "json")
        
        assert '"key"' in result
        assert '"value"' in result
    
    def test_format_output_list(self):
        """Test formatting output as list."""
        manager = DisplayManager(color_enabled=False)
        
        data = ["item1", "item2"]
        result = manager.format_output(data, "list")
        
        assert "1. item1" in result
        assert "2. item2" in result
    
    def test_format_output_message(self):
        """Test formatting output as message."""
        manager = DisplayManager(color_enabled=False)
        
        message = Message(content="Test", role=MessageRole.USER)
        result = manager.format_output(message, "message")
        
        assert "USER:" in result
        assert "Test" in result
    
    @patch('os.system')
    def test_clear_screen_windows(self, mock_system):
        """Test clearing screen on Windows."""
        with patch('os.name', 'nt'):
            manager = DisplayManager()
            manager.clear_screen()
            mock_system.assert_called_once_with('cls')
    
    @patch('os.system')
    def test_clear_screen_unix(self, mock_system):
        """Test clearing screen on Unix."""
        with patch('os.name', 'posix'):
            manager = DisplayManager()
            manager.clear_screen()
            mock_system.assert_called_once_with('clear')
    
    def test_set_color_enabled(self):
        """Test setting color enabled."""
        manager = DisplayManager(color_enabled=True)
        
        manager.set_color_enabled(False)
        assert manager.color_enabled is False
        assert manager.formatter.color_enabled is False
        
        manager.set_color_enabled(True)
        assert manager.color_enabled is True
        assert manager.formatter.color_enabled is True
    
    @patch('builtins.print')
    def test_convenience_methods(self, mock_print):
        """Test convenience methods for different message types."""
        manager = DisplayManager(color_enabled=False)
        
        manager.print_success("Success!")
        manager.print_error("Error!")
        manager.print_warning("Warning!")
        manager.print_info("Info!")
        
        assert mock_print.call_count == 4
    
    @patch('builtins.input', return_value='test input')
    @patch('builtins.print')
    def test_input_prompt(self, mock_print, mock_input):
        """Test input prompt."""
        manager = DisplayManager(color_enabled=False)
        
        result = manager.input_prompt("Enter something")
        
        assert result == "test input"
        mock_input.assert_called_once()
    
    @patch('builtins.input', side_effect=['y'])
    def test_confirm_prompt_yes(self, mock_input):
        """Test confirmation prompt with yes response."""
        manager = DisplayManager(color_enabled=False)
        
        result = manager.confirm_prompt("Are you sure?")
        assert result is True
    
    @patch('builtins.input', side_effect=['n'])
    def test_confirm_prompt_no(self, mock_input):
        """Test confirmation prompt with no response."""
        manager = DisplayManager(color_enabled=False)
        
        result = manager.confirm_prompt("Are you sure?")
        assert result is False
    
    @patch('builtins.input', side_effect=['', ''])
    def test_confirm_prompt_default(self, mock_input):
        """Test confirmation prompt with default response."""
        manager = DisplayManager(color_enabled=False)
        
        result = manager.confirm_prompt("Are you sure?", default=True)
        assert result is True
        
        result = manager.confirm_prompt("Are you sure?", default=False)
        assert result is False
    
    def test_supports_color_detection(self):
        """Test color support detection logic."""
        manager = DisplayManager()
        
        # Test with NO_COLOR environment variable
        with patch.dict('os.environ', {'NO_COLOR': '1'}, clear=True):
            with patch('sys.stdout.isatty', return_value=True):
                assert manager._supports_color() is False
        
        # Test with FORCE_COLOR environment variable
        with patch.dict('os.environ', {'FORCE_COLOR': '1'}, clear=True):
            with patch('sys.stdout.isatty', return_value=True):
                assert manager._supports_color() is True
        
        # Test with color-supporting TERM
        with patch.dict('os.environ', {'TERM': 'xterm-256color'}, clear=True):
            with patch('sys.stdout.isatty', return_value=True):
                assert manager._supports_color() is True


if __name__ == "__main__":
    pytest.main([__file__])