"""Unit tests for the CommandParser class."""

import pytest
from src.cli.command_parser import CommandParser, ParsedCommand


class TestCommandParser:
    """Test cases for CommandParser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CommandParser()
    
    def test_parse_simple_command(self):
        """Test parsing a simple command without arguments."""
        result = self.parser.parse("help")
        
        assert result.command == "help"
        assert result.args == []
        assert result.kwargs == {}
        assert result.raw_input == "help"
    
    def test_parse_command_with_args(self):
        """Test parsing a command with positional arguments."""
        result = self.parser.parse("task analyze data file")
        
        assert result.command == "task"
        assert result.args == ["analyze", "data", "file"]
        assert result.kwargs == {}
    
    def test_parse_command_with_kwargs(self):
        """Test parsing a command with keyword arguments."""
        result = self.parser.parse("config model=openai timeout=30")
        
        assert result.command == "config"
        assert result.args == []
        assert result.kwargs == {"model": "openai", "timeout": "30"}
    
    def test_parse_command_with_mixed_args(self):
        """Test parsing a command with both positional and keyword arguments."""
        result = self.parser.parse("task process file.txt format=json verbose=true")
        
        assert result.command == "task"
        assert result.args == ["process", "file.txt"]
        assert result.kwargs == {"format": "json", "verbose": "true"}
    
    def test_parse_quoted_arguments(self):
        """Test parsing commands with quoted arguments."""
        result = self.parser.parse('task "analyze the data file" output="results.txt"')
        
        assert result.command == "task"
        assert result.args == ["analyze the data file"]
        assert result.kwargs == {"output": "results.txt"}
    
    def test_parse_empty_input(self):
        """Test parsing empty input raises ValueError."""
        with pytest.raises(ValueError, match="Empty command"):
            self.parser.parse("")
        
        with pytest.raises(ValueError, match="Empty command"):
            self.parser.parse("   ")
    
    def test_parse_invalid_quotes(self):
        """Test parsing input with invalid quotes raises ValueError."""
        with pytest.raises(ValueError, match="Invalid command syntax"):
            self.parser.parse('task "unclosed quote')
    
    def test_validate_valid_commands(self):
        """Test validation of valid commands."""
        valid_commands = [
            ParsedCommand("help", [], {}, "help"),
            ParsedCommand("exit", [], {}, "exit"),
            ParsedCommand("task", ["do something"], {}, "task do something"),
            ParsedCommand("config", ["key", "value"], {}, "config key value")
        ]
        
        for cmd in valid_commands:
            is_valid, error = self.parser.validate_command(cmd)
            assert is_valid, f"Command {cmd.command} should be valid, got error: {error}"
    
    def test_validate_invalid_commands(self):
        """Test validation of invalid commands."""
        invalid_commands = [
            ParsedCommand("unknown", [], {}, "unknown"),
            ParsedCommand("help", ["arg1", "arg2"], {}, "help arg1 arg2"),  # help takes max 1 arg
            ParsedCommand("task", [], {}, "task"),  # task requires args
            ParsedCommand("config", ["a", "b", "c"], {}, "config a b c")  # too many args
        ]
        
        for cmd in invalid_commands:
            is_valid, error = self.parser.validate_command(cmd)
            assert not is_valid, f"Command {cmd.command} should be invalid"
            assert error is not None
    
    def test_get_help_text_general(self):
        """Test getting general help text."""
        help_text = self.parser.get_help_text()
        
        assert "Available commands:" in help_text
        assert "help" in help_text
        assert "task" in help_text
        assert "Usage examples:" in help_text
    
    def test_get_help_text_specific(self):
        """Test getting help text for a specific command."""
        help_text = self.parser.get_help_text("help")
        
        assert "help:" in help_text
        assert "Show help information" in help_text
    
    def test_suggest_command_exact_match(self):
        """Test command suggestion for close matches."""
        suggestions = [
            ("hep", "help"),
            ("tak", "task"),
            ("exti", "exit"),
            ("claer", "clear")
        ]
        
        for invalid, expected in suggestions:
            suggestion = self.parser.suggest_command(invalid)
            assert suggestion == expected, f"Expected '{expected}' for '{invalid}', got '{suggestion}'"
    
    def test_suggest_command_no_match(self):
        """Test command suggestion when no good match exists."""
        suggestion = self.parser.suggest_command("completely_different")
        assert suggestion is None
    
    def test_case_insensitive_parsing(self):
        """Test that command parsing is case insensitive."""
        result = self.parser.parse("HELP")
        assert result.command == "help"
        
        result = self.parser.parse("Task Do Something")
        assert result.command == "task"
        assert result.args == ["Do", "Something"]
    
    def test_whitespace_handling(self):
        """Test proper handling of various whitespace scenarios."""
        test_cases = [
            "  help  ",
            "\ttask\tdo\tsomething\t",
            "config   key=value   ",
        ]
        
        for test_input in test_cases:
            result = self.parser.parse(test_input)
            assert result.raw_input == test_input.strip()
    
    def test_special_characters_in_args(self):
        """Test handling of special characters in arguments."""
        result = self.parser.parse('task "file with spaces.txt" path=/home/user')
        
        assert result.args == ["file with spaces.txt"]
        assert result.kwargs == {"path": "/home/user"}
    
    def test_equals_in_args(self):
        """Test handling of equals signs in positional arguments."""
        result = self.parser.parse('task equation "x=y+z"')
        
        assert result.command == "task"
        assert result.args == ["equation", "x=y+z"]
        assert result.kwargs == {}
    
    def test_empty_kwargs_values(self):
        """Test handling of empty keyword argument values."""
        result = self.parser.parse('config key=""')
        
        assert result.kwargs == {"key": ""}
    
    def test_multiple_equals_in_kwargs(self):
        """Test handling of multiple equals signs in keyword arguments."""
        result = self.parser.parse('config equation="x=y=z"')
        
        assert result.kwargs == {"equation": "x=y=z"}