"""Command parser for processing and validating CLI input."""

import re
import shlex
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class ParsedCommand:
    """Represents a parsed command with its arguments."""
    command: str
    args: List[str]
    kwargs: Dict[str, str]
    raw_input: str


class CommandParser:
    """Handles parsing and validation of CLI commands."""
    
    def __init__(self):
        self.valid_commands = {
            'help': 'Show help information',
            'exit': 'Exit the application',
            'quit': 'Exit the application',
            'clear': 'Clear the screen',
            'history': 'Show command history',
            'task': 'Execute a task',
            'chat': 'Start a chat session',
            'tools': 'List available tools',
            'memory': 'Show memory information',
            'config': 'Show or modify configuration'
        }
    
    def parse(self, user_input: str) -> ParsedCommand:
        """
        Parse user input into a structured command.
        
        Args:
            user_input: Raw user input string
            
        Returns:
            ParsedCommand object with parsed components
            
        Raises:
            ValueError: If command parsing fails
        """
        if not user_input or not user_input.strip():
            raise ValueError("Empty command")
        
        # Clean and normalize input
        cleaned_input = user_input.strip()
        
        # Track which tokens were originally quoted to preserve them as positional args
        quoted_tokens = set()
        
        # Find quoted sections in the original input
        import re
        quote_pattern = r'"([^"]*)"'
        quoted_matches = re.findall(quote_pattern, cleaned_input)
        for match in quoted_matches:
            quoted_tokens.add(match)
        
        try:
            # Use shlex to properly handle quoted arguments
            tokens = shlex.split(cleaned_input)
        except ValueError as e:
            raise ValueError(f"Invalid command syntax: {e}")
        
        if not tokens:
            raise ValueError("Empty command")
        
        command = tokens[0].lower()
        remaining_tokens = tokens[1:]
        
        # Separate positional args from keyword args
        args = []
        kwargs = {}
        
        for token in remaining_tokens:
            # If this token was originally quoted, treat it as positional argument
            if token in quoted_tokens:
                args.append(token)
            elif '=' in token and not token.startswith('='):
                # Check if this looks like a keyword argument (key=value format)
                # Only treat as kwarg if the part before = looks like a valid identifier
                key_part = token.split('=', 1)[0]
                if key_part.isidentifier():
                    # This is a keyword argument
                    key, value = token.split('=', 1)
                    kwargs[key] = value
                else:
                    # This is a positional argument that happens to contain =
                    args.append(token)
            else:
                # This is a positional argument
                args.append(token)
        
        return ParsedCommand(
            command=command,
            args=args,
            kwargs=kwargs,
            raw_input=cleaned_input
        )
    
    def validate_command(self, parsed_command: ParsedCommand) -> Tuple[bool, Optional[str]]:
        """
        Validate a parsed command.
        
        Args:
            parsed_command: The parsed command to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        command = parsed_command.command
        
        # Check if command exists
        if command not in self.valid_commands:
            return False, f"Unknown command: '{command}'. Type 'help' for available commands."
        
        # Command-specific validation
        if command in ['exit', 'quit', 'clear', 'history', 'tools']:
            # These commands don't take arguments
            if parsed_command.args or parsed_command.kwargs:
                return False, f"Command '{command}' does not accept arguments"
        
        elif command == 'help':
            # Help command can take 0 or 1 argument
            if len(parsed_command.args) > 1 or parsed_command.kwargs:
                return False, f"Help command accepts at most 1 argument"
        
        elif command == 'task':
            # Task command requires at least one argument
            if not parsed_command.args:
                return False, "Task command requires a task description"
        
        elif command == 'config':
            # Config command can have 0, 1, or 2 arguments
            if len(parsed_command.args) > 2:
                return False, "Config command accepts at most 2 arguments: [key] [value]"
        
        return True, None
    
    def get_help_text(self, command: Optional[str] = None) -> str:
        """
        Get help text for commands.
        
        Args:
            command: Specific command to get help for, or None for general help
            
        Returns:
            Help text string
        """
        if command:
            if command in self.valid_commands:
                return self._get_detailed_help(command)
            else:
                return f"Unknown command: '{command}'. Type 'help' to see available commands."
        
        help_lines = ["Available commands:"]
        for cmd, desc in self.valid_commands.items():
            help_lines.append(f"  {cmd:<10} - {desc}")
        
        help_lines.extend([
            "",
            "Usage examples:",
            "  task 'analyze the data file'",
            "  config model_provider openai",
            "  help task",
            "",
            "Use quotes for arguments containing spaces.",
            "Type 'help <command>' for detailed information about a specific command."
        ])
        
        return "\n".join(help_lines)
    
    def _get_detailed_help(self, command: str) -> str:
        """
        Get detailed help for a specific command.
        
        Args:
            command: The command to get detailed help for
            
        Returns:
            Detailed help text
        """
        help_details = {
            'help': [
                "help: Show help information",
                "",
                "Usage:",
                "  help           - Show all available commands",
                "  help <command> - Show detailed help for a specific command",
                "",
                "Examples:",
                "  help",
                "  help task",
                "  help config"
            ],
            'exit': [
                "exit: Exit the application",
                "",
                "Usage:",
                "  exit",
                "",
                "Aliases: quit",
                "This command gracefully shuts down the CLI application."
            ],
            'quit': [
                "quit: Exit the application",
                "",
                "Usage:",
                "  quit",
                "",
                "Aliases: exit",
                "This command gracefully shuts down the CLI application."
            ],
            'clear': [
                "clear: Clear the screen",
                "",
                "Usage:",
                "  clear",
                "",
                "Clears the terminal screen for a fresh view."
            ],
            'history': [
                "history: Show command history",
                "",
                "Usage:",
                "  history",
                "",
                "Shows the last 10 commands executed in the current session."
            ],
            'task': [
                "task: Execute a task",
                "",
                "Usage:",
                "  task <description>",
                "",
                "Examples:",
                "  task 'analyze the data file'",
                "  task create a summary report",
                "  task \"find all TODO items in the code\"",
                "",
                "The task description will be processed by the ReAct agent."
            ],
            'chat': [
                "chat: Start a chat session",
                "",
                "Usage:",
                "  chat",
                "",
                "Activates chat mode for conversational interaction with the AI agent."
            ],
            'tools': [
                "tools: List available tools",
                "",
                "Usage:",
                "  tools",
                "",
                "Shows all tools available to the AI agent for task execution."
            ],
            'memory': [
                "memory: Show memory information",
                "",
                "Usage:",
                "  memory",
                "",
                "Displays current session information and memory usage."
            ],
            'config': [
                "config: Show or modify configuration",
                "",
                "Usage:",
                "  config                - Show configuration help",
                "  config <key>          - Show value for configuration key",
                "  config <key> <value>  - Set configuration key to value",
                "",
                "Examples:",
                "  config",
                "  config model_provider",
                "  config model_provider openai",
                "  config timeout 30"
            ]
        }
        
        if command in help_details:
            return "\n".join(help_details[command])
        else:
            return f"{command}: {self.valid_commands[command]}"
    
    def suggest_command(self, invalid_command: str) -> Optional[str]:
        """
        Suggest a similar valid command for an invalid one.
        
        Args:
            invalid_command: The invalid command entered
            
        Returns:
            Suggested command or None if no good match
        """
        invalid_lower = invalid_command.lower()
        
        # Direct substring matches
        for cmd in self.valid_commands:
            if invalid_lower in cmd or cmd in invalid_lower:
                return cmd
        
        # Simple edit distance for typos
        def simple_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return simple_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        best_match = None
        best_distance = float('inf')
        
        for cmd in self.valid_commands:
            distance = simple_distance(invalid_lower, cmd)
            if distance < best_distance and distance <= 2:  # Allow up to 2 character differences
                best_distance = distance
                best_match = cmd
        
        return best_match