"""Integration tests for CLI interaction flows."""

import pytest
from unittest.mock import Mock, patch, call
from io import StringIO
import sys
from src.cli.cli_application import CLIApplication
from src.interfaces.display_interface import DisplayInterface


class MockDisplayManager(DisplayInterface):
    """Mock display manager for integration testing."""
    
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


class TestCLIIntegration:
    """Integration test cases for CLI workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_display = MockDisplayManager()
        self.cli_app = CLIApplication(self.mock_display)
    
    def test_complete_help_workflow(self):
        """Test complete help command workflow."""
        # Test general help
        self.cli_app.handle_input("help")
        
        help_messages = [msg for msg, style in self.mock_display.messages if "Available commands" in msg]
        assert len(help_messages) > 0
        
        # Clear messages
        self.mock_display.messages.clear()
        
        # Test specific help
        self.cli_app.handle_input("help task")
        
        task_help_messages = [msg for msg, style in self.mock_display.messages if "task:" in msg or "Execute a task" in msg]
        assert len(task_help_messages) > 0
    
    def test_command_suggestion_workflow(self):
        """Test command suggestion and correction workflow."""
        # Test typo that should get suggestion
        self.cli_app.handle_input("hep")
        
        messages = [msg for msg, style in self.mock_display.messages]
        
        # Should have error message
        error_msg = next((msg for msg in messages if "Unknown command" in msg), None)
        assert error_msg is not None
        
        # Should have suggestion
        suggestion_msg = next((msg for msg in messages if "Did you mean" in msg), None)
        assert suggestion_msg is not None
        assert "help" in suggestion_msg
        
        # Should have help info
        help_info_msg = next((msg for msg in messages if "Type 'help help'" in msg), None)
        assert help_info_msg is not None
    
    def test_session_management_workflow(self):
        """Test session creation and management workflow."""
        # Session should be created when first command is executed
        session = self.cli_app.session_manager.get_current_session()
        assert session is None  # No session initially
        
        # Execute a command to create session
        self.cli_app.handle_input("help")
        session = self.cli_app.session_manager.get_current_session()
        assert session is not None
        
        # Execute some more commands to build history
        additional_commands = ["memory", "config test_key test_value"]
        for cmd in additional_commands:
            self.cli_app.handle_input(cmd)
        
        # Check history (should include initial help + additional commands)
        history = self.cli_app.session_manager.get_command_history()
        expected_commands = ["help"] + additional_commands
        assert len(history) == len(expected_commands)
        for cmd in expected_commands:
            assert cmd in history
        
        # Test memory command shows session info
        self.mock_display.messages.clear()
        self.cli_app.handle_input("memory")
        
        memory_messages = [msg for msg, style in self.mock_display.messages]
        session_info_msg = next((msg for msg in memory_messages if "Session Information" in msg), None)
        assert session_info_msg is not None
    
    def test_configuration_workflow(self):
        """Test configuration management workflow."""
        # Test config help
        self.cli_app.handle_input("config")
        
        config_help_messages = [msg for msg, style in self.mock_display.messages if "Configuration management" in msg]
        assert len(config_help_messages) > 0
        
        # Clear messages
        self.mock_display.messages.clear()
        
        # Test setting a config value
        self.cli_app.handle_input("config test_key test_value")
        
        success_messages = [msg for msg, style in self.mock_display.messages if style == "success"]
        assert len(success_messages) > 0
        assert "Set test_key = test_value" in success_messages[0]
        
        # Clear messages
        self.mock_display.messages.clear()
        
        # Test getting the config value
        self.cli_app.handle_input("config test_key")
        
        value_messages = [msg for msg, style in self.mock_display.messages if "test_key: test_value" in msg]
        assert len(value_messages) > 0
    
    def test_error_handling_workflow(self):
        """Test error handling and recovery workflow."""
        # Test parse error
        self.cli_app.handle_input('task "unclosed quote')
        
        error_messages = [msg for msg, style in self.mock_display.messages if style == "error"]
        assert len(error_messages) > 0
        assert "Parse error" in error_messages[0]
        
        # Clear messages
        self.mock_display.messages.clear()
        
        # Test validation error
        self.cli_app.handle_input("clear extra_arg")
        
        error_messages = [msg for msg, style in self.mock_display.messages if style == "error"]
        assert len(error_messages) > 0
        assert "does not accept arguments" in error_messages[0]
        
        # Should also have help suggestion
        help_messages = [msg for msg, style in self.mock_display.messages if "Type 'help" in msg]
        assert len(help_messages) > 0
    
    def test_task_execution_workflow(self):
        """Test task execution workflow."""
        # Test task command
        self.cli_app.handle_input("task analyze the data")
        
        messages = [msg for msg, style in self.mock_display.messages]
        
        # Should show task execution request
        task_msg = next((msg for msg in messages if "Task execution requested" in msg), None)
        assert task_msg is not None
        assert "analyze the data" in task_msg
        
        # Should show not implemented warning
        warning_msg = next((msg for msg in messages if "not yet implemented" in msg), None)
        assert warning_msg is not None
    
    def test_history_workflow(self):
        """Test command history workflow."""
        # Initially no history
        self.cli_app.handle_input("history")
        
        no_history_messages = [msg for msg, style in self.mock_display.messages if "No command history" in msg]
        assert len(no_history_messages) > 0
        
        # Execute some commands
        commands = ["help", "task test", "memory"]
        for cmd in commands:
            self.cli_app.handle_input(cmd)
        
        # Clear messages
        self.mock_display.messages.clear()
        
        # Check history now shows commands
        self.cli_app.handle_input("history")
        
        messages = [msg for msg, style in self.mock_display.messages]
        history_header = next((msg for msg in messages if "Command History" in msg), None)
        assert history_header is not None
        
        # Should show numbered history items
        numbered_items = [msg for msg in messages if ". " in msg and any(char.isdigit() for char in msg)]
        assert len(numbered_items) > 0
    
    @patch('os.system')
    def test_clear_screen_workflow(self, mock_system):
        """Test screen clearing workflow."""
        self.cli_app.handle_input("clear")
        
        # Should call os.system with appropriate clear command
        mock_system.assert_called_once()
        call_args = mock_system.call_args[0][0]
        assert call_args in ['cls', 'clear']
    
    def test_exit_workflow(self):
        """Test application exit workflow."""
        # Test exit command
        self.cli_app.is_running = True
        self.cli_app.handle_input("exit")
        
        assert self.cli_app.is_running is False
        
        # Should show goodbye message
        goodbye_messages = [msg for msg, style in self.mock_display.messages if "Goodbye" in msg]
        assert len(goodbye_messages) > 0
    
    def test_quit_workflow(self):
        """Test application quit workflow."""
        # Test quit command
        self.cli_app.is_running = True
        self.cli_app.handle_input("quit")
        
        assert self.cli_app.is_running is False
        
        # Should show goodbye message
        goodbye_messages = [msg for msg, style in self.mock_display.messages if "Goodbye" in msg]
        assert len(goodbye_messages) > 0
    
    def test_multiple_command_workflow(self):
        """Test executing multiple commands in sequence."""
        commands_and_expectations = [
            ("help", "Available commands"),
            ("memory", "Session Information"),
            ("config test_key test_value", "Set test_key = test_value"),
            ("config test_key", "test_key: test_value"),
            ("task test task", "Task execution requested"),
            ("history", "Command History")
        ]
        
        for command, expected_content in commands_and_expectations:
            self.mock_display.messages.clear()
            self.cli_app.handle_input(command)
            
            messages = [msg for msg, style in self.mock_display.messages]
            matching_msg = next((msg for msg in messages if expected_content in msg), None)
            assert matching_msg is not None, f"Expected '{expected_content}' not found in response to '{command}'"
    
    def test_empty_input_handling(self):
        """Test handling of empty input."""
        initial_message_count = len(self.mock_display.messages)
        
        # Test various empty inputs
        empty_inputs = ["", "   ", "\t", "\n"]
        for empty_input in empty_inputs:
            self.cli_app.handle_input(empty_input)
        
        # Should not produce any new messages
        assert len(self.mock_display.messages) == initial_message_count
    
    def test_case_insensitive_commands(self):
        """Test that commands work regardless of case."""
        case_variations = [
            ("HELP", "Available commands"),
            ("Help", "Available commands"),
            ("hElP", "Available commands"),
            ("MEMORY", "Session Information"),
            ("Memory", "Session Information")
        ]
        
        for command, expected_content in case_variations:
            self.mock_display.messages.clear()
            self.cli_app.handle_input(command)
            
            messages = [msg for msg, style in self.mock_display.messages]
            matching_msg = next((msg for msg in messages if expected_content in msg), None)
            assert matching_msg is not None, f"Case-insensitive command '{command}' failed"
    
    def test_quoted_arguments_workflow(self):
        """Test handling of quoted arguments."""
        # Test task with quoted argument
        self.cli_app.handle_input('task "analyze the complex data file"')
        
        messages = [msg for msg, style in self.mock_display.messages]
        task_msg = next((msg for msg in messages if "Task execution requested" in msg), None)
        assert task_msg is not None
        assert "analyze the complex data file" in task_msg
        
        # Clear messages
        self.mock_display.messages.clear()
        
        # Test config with quoted values
        self.cli_app.handle_input('config description "This is a test description"')
        
        success_messages = [msg for msg, style in self.mock_display.messages if style == "success"]
        assert len(success_messages) > 0
        assert "This is a test description" in success_messages[0]