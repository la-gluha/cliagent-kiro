"""CLI interface definition for the AI Agent System."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class CLIInterface(ABC):
    """Abstract base class for CLI implementations."""
    
    @abstractmethod
    def start_session(self) -> None:
        """Start a new CLI session."""
        pass
    
    @abstractmethod
    def handle_input(self, user_input: str) -> None:
        """Handle user input and process commands."""
        pass
    
    @abstractmethod
    def display_output(self, content: str, output_type: str = "info") -> None:
        """Display output to the user."""
        pass
    
    @abstractmethod
    def show_progress(self, message: str, progress: float) -> None:
        """Show progress information to the user."""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Gracefully shutdown the CLI session."""
        pass