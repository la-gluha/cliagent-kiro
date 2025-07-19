"""Unit tests for the CLIApplication class."""

import pytest
from unittest.mock import Mock, patch, call
from io import StringIO
import sys
from src.cli.cli_application import CLIApplication
from src.interfaces.display_interface import DisplayInterface


class MockDisplayManager(DisplayInterface):
    """Mock display manager for testing."""
    
    def __init__(self):
        self.messages = []
        self.progress_calls = []
        self.animations = []
        self.color_enabled = True
    
    def show_message(self, message: str, style = None) -> None:
        self.messages.append((message, style))
    
    def show_progress(self, operation: str, progress: float) -> None:
        self.progress_calls.append((operation, progress))
    
    def show_animation(self, animation_type, message: str = "") -> None:
        self.animations.append((animation_type, message))
    
    def format_output(self, content, format_type: str) -> str:
        return str(content)
    
    def clear_screen(self) -> None:
        pass
    
    def set_color_enabled(self, enabled: bool) -> None:
        self.color_enabled = enabled


class TestCLIApplication:
    """Test cases for CLIApplication."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_display = MockDisplayManager()
        self.cli_app = CLIApplication(self.mock_display)
    
    def test_init_without_display_manager(self):
        """Test initialization without display manager."""
        app = CLIApplication()
        assert app.display_manager is None
        assert app.command_parser is not None
        assert app.session_manager is not None
        assert app.is_running is False
    
    def test_init_with_display_manager(self):
        """Test initialization with display manager."""
        assert self.cli_app.display_manager == self.mock_display
        assert self.cli_app.command_parser is not None
        assert self.cli_app.session_manager is not None
    
    def test_display_output_with_display_manager(self):
        """Test display output with display manager."""
        self.cli_app.display_output("test message", "info")
        
        assert len(self.mock_display.messages) == 1
        assert self.mock_display.messages[0] == ("test message", "info")
    
    def test_display_output_without_display_manager(self):
        """Test display output without display manager."""
        app = CLIApplication()
        
        # Capture stdout
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            app.display_output("test message", "info")
        
        assert "test message" in captured_output.getvalue()
    
    def test_display_output_error_formatting(self):
        """Test error message formatting without display manager."""
        app = CLIApplication()
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            app.display_output("error message", "error")
        
        output = captured_output.getvalue()
        assert "❌ ERROR:" in output
        assert "error message" in output
    
    def test_show_progress_with_display_manager(self):
        """Test show progress with display manager."""
        self.cli_app.show_progress("Processing", 0.5)
        
        assert len(self.mock_display.progress_calls) == 1
        assert self.mock_display.progress_calls[0] == ("Processing", 0.5)
    
    def test_show_progress_without_display_manager(self):
        """Test show progress without display manager."""
        app = CLIApplication()
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            app.show_progress("Processing", 0.5)
        
        output = captured_output.getvalue()
        assert "Processing" in output
        assert "50%" in output
    
    def test_handle_input_empty(self):
        """Test handling empty input."""
        self.cli_app.handle_input("")
        self.cli_app.handle_input("   ")
        
        # Should not produce any output for empty input
        assert len(self.mock_display.messages) == 0
    
    def test_handle_input_valid_command(self):
        """Test handling valid command."""
        self.cli_app.handle_input("help")
        
        # Should have at least one message (the help text)
        assert len(self.mock_display.messages) >= 1
        help_message = next((msg for msg, style in self.mock_display.messages if "Available commands" in msg), None)
        assert help_message is not None
    
    def test_handle_input_invalid_command(self):
        """Test handling invalid command."""
        self.cli_app.handle_input("invalid_command")
        
        # Should show error message
        error_messages = [msg for msg, style in self.mock_display.messages if style == "error"]
        assert len(error_messages) > 0
        assert "Unknown command" in error_messages[0]
    
    def test_handle_input_command_suggestion(self):
        """Test command suggestion for invalid commands."""
        self.cli_app.handle_input("hep")  # Should suggest "help"
        
        # Should show error and suggestion
        messages = [msg for msg, style in self.mock_display.messages]
        error_msg = next((msg for msg in messages if "Unknown command" in msg), None)
        suggestion_msg = next((msg for msg in messages if "Did you mean" in msg), None)
        
        assert error_msg is not None
        assert suggestion_msg is not None
        assert "help" in suggestion_msg
    
    def test_handle_input_parse_error(self):
        """Test handling parse errors."""
        self.cli_app.handle_input('task "unclosed quote')
        
        # Should show parse error
        error_messages = [msg for msg, style in self.mock_display.messages if style == "error"]
        assert len(error_messages) > 0
        assert "Parse error" in error_messages[0]
    
    def test_execute_help_command(self):
        """Test executing help command."""
        self.cli_app.handle_input("help")
        
        messages = [msg for msg, style in self.mock_display.messages]
        help_message = next((msg for msg in messages if "Available commands" in msg), None)
        assert help_message is not None
    
    def test_execute_help_command_specific(self):
        """Test executing help command for specific command."""
        self.cli_app.handle_input("help task")
        
        messages = [msg for msg, style in self.mock_display.messages]
        help_message = next((msg for msg in messages if "task:" in msg), None)
        assert help_message is not None
    
    @patch('os.system')
    def test_execute_clear_command(self, mock_system):
        """Test executing clear command."""
        self.cli_app.handle_input("clear")
        
        # Should call os.system with appropriate clear command
        mock_system.assert_called_once()
        call_args = mock_system.call_args[0][0]
        assert call_args in ['cls', 'clear']
    
    def test_execute_history_command_empty(self):
        """Test executing history command with no history."""
        self.cli_app.handle_input("history")
        
        messages = [msg for msg, style in self.mock_display.messages]
        history_message = next((msg for msg in messages if "No command history" in msg), None)
        assert history_message is not None
    
    def test_execute_history_command_with_data(self):
        """Test executing history command with command history."""
        # Add some commands to history
        self.cli_app.handle_input("help")
        self.cli_app.handle_input("task test")
        
        # Clear previous messages
        self.mock_display.messages.clear()
        
        self.cli_app.handle_input("history")
        
        messages = [msg for msg, style in self.mock_display.messages]
        history_header = next((msg for msg in messages if "Command History" in msg), None)
        assert history_header is not None
        
        # Should show numbered history items
        numbered_items = [msg for msg in messages if ". " in msg and any(char.isdigit() for char in msg)]
        assert len(numbered_items) > 0
    
    def test_execute_task_command(self):
        """Test executing task command."""
        self.cli_app.handle_input("task analyze data")
        
        messages = [msg for msg, style in self.mock_display.messages]
        task_message = next((msg for msg in messages if "Task execution requested" in msg), None)
        assert task_message is not None
        assert "analyze data" in task_message
    
    def test_execute_memory_command(self):
        """Test executing memory command."""
        self.cli_app.handle_input("memory")
        
        messages = [msg for msg, style in self.mock_display.messages]
        memory_message = next((msg for msg in messages if "Session Information" in msg), None)
        assert memory_message is not None
    
    def test_execute_config_command_no_args(self):
        """Test executing config command without arguments."""
        self.cli_app.handle_input("config")
        
        messages = [msg for msg, style in self.mock_display.messages]
        config_message = next((msg for msg in messages if "Configuration management" in msg), None)
        assert config_message is not None
    
    def test_execute_config_command_get(self):
        """Test executing config command to get a value."""
        # First set a value
        self.cli_app.handle_input("config test_key test_value")
        
        # Clear messages
        self.mock_display.messages.clear()
        
        # Then get the value
        self.cli_app.handle_input("config test_key")
        
        messages = [msg for msg, style in self.mock_display.messages]
        config_message = next((msg for msg in messages if "test_key: test_value" in msg), None)
        assert config_message is not None
    
    def test_execute_config_command_set(self):
        """Test executing config command to set a value."""
        self.cli_app.handle_input("config test_key test_value")
        
        messages = [msg for msg, style in self.mock_display.messages]
        success_message = next((msg for msg, style in self.mock_display.messages if style == "success"), None)
        assert success_message is not None
        assert "Set test_key = test_value" in success_message
    
    def test_execute_config_command_get_nonexistent(self):
        """Test executing config command to get nonexistent value."""
        self.cli_app.handle_input("config nonexistent_key")
        
        messages = [msg for msg, style in self.mock_display.messages]
        warning_message = next((msg for msg, style in self.mock_display.messages if style == "warning"), None)
        assert warning_message is not None
        assert "not found" in warning_message
    
    def test_shutdown(self):
        """Test shutdown functionality."""
        self.cli_app.is_running = True
        self.cli_app.shutdown()
        
        assert self.cli_app.is_running is False
        
        # Should display goodbye message
        messages = [msg for msg, style in self.mock_display.messages]
        goodbye_message = next((msg for msg in messages if "Goodbye" in msg), None)
        assert goodbye_message is not None
    
    def test_execute_exit_command(self):
        """Test executing exit command."""
        self.cli_app.is_running = True
        self.cli_app.handle_input("exit")
        
        assert self.cli_app.is_running is False
    
    def test_execute_quit_command(self):
        """Test executing quit command."""
        self.cli_app.is_running = True
        self.cli_app.handle_input("quit")
        
        assert self.cli_app.is_running is False
    
    def test_session_activity_tracking(self):
        """Test that commands are tracked in session activity."""
        # Execute some commands
        self.cli_app.handle_input("help")
        self.cli_app.handle_input("task test")
        
        # Check session history
        history = self.cli_app.session_manager.get_command_history()
        assert "help" in history
        assert "task test" in history
    
    def test_command_validation_integration(self):
        """Test integration with command validation."""
        # Test command that doesn't accept arguments
        self.cli_app.handle_input("clear extra_arg")
        
        error_messages = [msg for msg, style in self.mock_display.messages if style == "error"]
        assert len(error_messages) > 0
        assert "does not accept arguments" in error_messages[0]
    
    def test_unimplemented_commands(self):
        """Test handling of recognized but unimplemented commands."""
        unimplemented_commands = ["chat", "tools"]
        
        for cmd in unimplemented_commands:
            self.mock_display.messages.clear()
            self.cli_app.handle_input(cmd)
            
            warning_messages = [msg for msg, style in self.mock_display.messages if style == "warning"]
            assert len(warning_messages) > 0
            assert "not yet implemented" in warning_messages[-1]