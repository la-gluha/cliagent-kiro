"""
Display interface definition for the AI Agent System.

This module defines the abstract interface that all display implementations
must follow to ensure consistent output formatting and user experience.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from enum import Enum


class OutputStyle(Enum):
    """Enumeration for output styles."""
    NORMAL = "normal"
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HIGHLIGHT = "highlight"
    MUTED = "muted"


class AnimationType(Enum):
    """Enumeration for animation types."""
    SPINNER = "spinner"
    PROGRESS_BAR = "progress_bar"
    DOTS = "dots"
    PULSE = "pulse"


class DisplayInterface(ABC):
    """
    Abstract interface for display management in the AI Agent System.
    
    This interface defines the contract for formatting and displaying
    various types of content to the user.
    """
    
    @abstractmethod
    def show_message(self, message: str, style: OutputStyle = OutputStyle.NORMAL) -> None:
        """
        Display a message with the specified style.
        
        Args:
            message: The message to display
            style: The style to apply to the message
        """
        pass
    
    @abstractmethod
    def show_progress(self, operation: str, progress: float) -> None:
        """
        Display progress for a long-running operation.
        
        Args:
            operation: Description of the operation
            progress: Progress value between 0.0 and 1.0
        """
        pass
    
    @abstractmethod
    def show_animation(self, animation_type: AnimationType, message: str = "") -> None:
        """
        Display an animation with optional message.
        
        Args:
            animation_type: Type of animation to show
            message: Optional message to display with animation
        """
        pass
    
    @abstractmethod
    def format_output(self, content: Any, format_type: str) -> str:
        """
        Format content for display.
        
        Args:
            content: The content to format
            format_type: The type of formatting to apply
            
        Returns:
            Formatted string ready for display
        """
        pass
    
    @abstractmethod
    def clear_screen(self) -> None:
        """Clear the display screen."""
        pass
    
    @abstractmethod
    def set_color_enabled(self, enabled: bool) -> None:
        """
        Enable or disable color output.
        
        Args:
            enabled: Whether to enable color output
        """
        pass


class ProgressInterface(ABC):
    """
    Abstract interface for progress tracking.
    """
    
    @abstractmethod
    def start_progress(self, operation: str, total_steps: Optional[int] = None) -> None:
        """
        Start tracking progress for an operation.
        
        Args:
            operation: Description of the operation
            total_steps: Total number of steps (if known)
        """
        pass
    
    @abstractmethod
    def update_progress(self, current_step: int, message: str = "") -> None:
        """
        Update the progress.
        
        Args:
            current_step: Current step number
            message: Optional status message
        """
        pass
    
    @abstractmethod
    def finish_progress(self, success: bool = True, message: str = "") -> None:
        """
        Finish the progress tracking.
        
        Args:
            success: Whether the operation was successful
            message: Final message to display
        """
        pass


class AnimationInterface(ABC):
    """
    Abstract interface for animations.
    """
    
    @abstractmethod
    def start_animation(self, animation_type: AnimationType, message: str = "") -> None:
        """
        Start an animation.
        
        Args:
            animation_type: Type of animation
            message: Message to display with animation
        """
        pass
    
    @abstractmethod
    def update_animation_message(self, message: str) -> None:
        """
        Update the animation message.
        
        Args:
            message: New message to display
        """
        pass
    
    @abstractmethod
    def stop_animation(self) -> None:
        """Stop the current animation."""
        pass