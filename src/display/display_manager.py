"""
Display manager for the AI Agent System.

This module provides the main display management functionality,
coordinating output formatting and user interface elements.
"""

import sys
import os
from typing import Any, Optional
from ..interfaces.display_interface import DisplayInterface, OutputStyle, AnimationType
from .output_formatter import OutputFormatter


class DisplayManager(DisplayInterface):
    """
    Main display manager that coordinates all output operations.
    
    Handles message display, formatting, and basic terminal operations
    while providing a clean interface for the rest of the system.
    """
    
    def __init__(self, color_enabled: Optional[bool] = None):
        """
        Initialize the display manager.
        
        Args:
            color_enabled: Whether to enable color output. If None, auto-detect.
        """
        if color_enabled is None:
            # Auto-detect color support
            color_enabled = self._supports_color()
        
        self.color_enabled = color_enabled
        self.formatter = OutputFormatter(color_enabled=color_enabled)
        self._current_animation = None
    
    def _supports_color(self) -> bool:
        """
        Check if the terminal supports color output.
        
        Returns:
            True if color is supported, False otherwise
        """
        # Check if we're in a terminal
        if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
            return False
        
        # Check environment variables
        if os.environ.get('NO_COLOR'):
            return False
        
        if os.environ.get('FORCE_COLOR'):
            return True
        
        # Check TERM environment variable
        term = os.environ.get('TERM', '').lower()
        if 'color' in term or term in ['xterm', 'xterm-256color', 'screen', 'linux']:
            return True
        
        # Check if running on Windows with modern terminal
        if os.name == 'nt':
            # Windows 10 version 1607 and later support ANSI escape sequences
            try:
                import platform
                version = platform.version()
                if version and len(version.split('.')) >= 3:
                    major, minor, build = version.split('.')[:3]
                    if int(major) >= 10 and int(build) >= 14393:
                        return True
            except (ValueError, AttributeError):
                pass
        
        return False
    
    def show_message(self, message: str, style: OutputStyle = OutputStyle.NORMAL) -> None:
        """
        Display a message with the specified style.
        
        Args:
            message: The message to display
            style: The style to apply to the message
        """
        formatted_message = self.formatter.apply_style(message, style)
        print(formatted_message)
        sys.stdout.flush()
    
    def show_progress(self, operation: str, progress: float) -> None:
        """
        Display progress for a long-running operation.
        
        Args:
            operation: Description of the operation
            progress: Progress value between 0.0 and 1.0
        """
        # Clamp progress to valid range
        progress = max(0.0, min(1.0, progress))
        
        # Create a simple progress bar
        bar_width = 30
        filled_width = int(bar_width * progress)
        bar = "█" * filled_width + "░" * (bar_width - filled_width)
        percentage = int(progress * 100)
        
        # Format the progress display
        progress_text = f"{operation}: [{bar}] {percentage}%"
        
        # Use carriage return to overwrite the line
        print(f"\r{progress_text}", end="", flush=True)
        
        # Print newline when complete
        if progress >= 1.0:
            print()
    
    def show_animation(self, animation_type: AnimationType, message: str = "") -> None:
        """
        Display an animation with optional message.
        
        Args:
            animation_type: Type of animation to show
            message: Optional message to display with animation
        """
        # For now, just display a simple indicator
        # This will be enhanced in the animation engine
        if animation_type == AnimationType.SPINNER:
            indicator = "⠋"
        elif animation_type == AnimationType.DOTS:
            indicator = "..."
        elif animation_type == AnimationType.PULSE:
            indicator = "●"
        else:
            indicator = "▶"
        
        display_text = f"{indicator} {message}" if message else indicator
        self.show_message(display_text, OutputStyle.INFO)
    
    def format_output(self, content: Any, format_type: str) -> str:
        """
        Format content for display.
        
        Args:
            content: The content to format
            format_type: The type of formatting to apply
            
        Returns:
            Formatted string ready for display
        """
        if format_type == "json":
            return self.formatter.format_json(content)
        elif format_type == "list":
            if isinstance(content, list):
                return self.formatter.format_list(content)
            else:
                return str(content)
        elif format_type == "message":
            from ..models.data_models import Message
            if isinstance(content, Message):
                return self.formatter.format_message(content)
            else:
                return str(content)
        elif format_type == "task_result":
            from ..models.data_models import TaskResult
            if isinstance(content, TaskResult):
                return self.formatter.format_task_result(content)
            else:
                return str(content)
        elif format_type == "tool_info":
            from ..models.data_models import ToolInfo
            if isinstance(content, ToolInfo):
                return self.formatter.format_tool_info(content)
            else:
                return str(content)
        elif format_type == "agent_state":
            from ..models.data_models import AgentState
            if isinstance(content, AgentState):
                return self.formatter.format_agent_state(content)
            else:
                return str(content)
        else:
            # Default formatting
            return self.formatter._format_any_content(content)
    
    def clear_screen(self) -> None:
        """Clear the display screen."""
        if os.name == 'nt':  # Windows
            os.system('cls')
        else:  # Unix/Linux/MacOS
            os.system('clear')
    
    def set_color_enabled(self, enabled: bool) -> None:
        """
        Enable or disable color output.
        
        Args:
            enabled: Whether to enable color output
        """
        self.color_enabled = enabled
        self.formatter.set_color_enabled(enabled)
    
    def print_header(self, title: str) -> None:
        """
        Print a formatted header.
        
        Args:
            title: The header title
        """
        header = self.formatter.create_header(title)
        print(header)
    
    def print_separator(self, char: str = "-", length: int = 50) -> None:
        """
        Print a separator line.
        
        Args:
            char: Character to use for the separator
            length: Length of the separator
        """
        separator = self.formatter.create_separator(char, length)
        print(separator)
    
    def print_success(self, message: str) -> None:
        """
        Print a success message.
        
        Args:
            message: The success message
        """
        self.show_message(message, OutputStyle.SUCCESS)
    
    def print_error(self, message: str) -> None:
        """
        Print an error message.
        
        Args:
            message: The error message
        """
        self.show_message(message, OutputStyle.ERROR)
    
    def print_warning(self, message: str) -> None:
        """
        Print a warning message.
        
        Args:
            message: The warning message
        """
        self.show_message(message, OutputStyle.WARNING)
    
    def print_info(self, message: str) -> None:
        """
        Print an info message.
        
        Args:
            message: The info message
        """
        self.show_message(message, OutputStyle.INFO)
    
    def print_formatted(self, content: Any, format_type: str = "default") -> None:
        """
        Print formatted content.
        
        Args:
            content: The content to print
            format_type: The formatting type to use
        """
        formatted_content = self.format_output(content, format_type)
        print(formatted_content)
    
    def input_prompt(self, prompt: str, style: OutputStyle = OutputStyle.INFO) -> str:
        """
        Display a prompt and get user input.
        
        Args:
            prompt: The prompt message
            style: Style for the prompt
            
        Returns:
            User input string
        """
        formatted_prompt = self.formatter.apply_style(f"{prompt}: ", style)
        return input(formatted_prompt)
    
    def confirm_prompt(self, message: str, default: bool = False) -> bool:
        """
        Display a yes/no confirmation prompt.
        
        Args:
            message: The confirmation message
            default: Default value if user just presses enter
            
        Returns:
            True for yes, False for no
        """
        default_text = "Y/n" if default else "y/N"
        prompt = f"{message} ({default_text})"
        
        while True:
            response = self.input_prompt(prompt).strip().lower()
            
            if not response:
                return default
            elif response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                self.print_error("Please enter 'y' for yes or 'n' for no.")