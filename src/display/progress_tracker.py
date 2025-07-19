"""
Progress tracker for the AI Agent System.

This module provides progress tracking capabilities for long-running operations
with support for steps, sub-operations, and engaging status messages.
"""

import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from ..interfaces.display_interface import ProgressInterface
from .animation_engine import AnimationEngine, ProgressBarStyle


@dataclass
class ProgressStep:
    """Represents a single step in a progress operation."""
    name: str
    description: str
    weight: float = 1.0  # Relative weight of this step
    completed: bool = False
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None


class ProgressTracker(ProgressInterface):
    """
    Tracks progress for long-running operations.
    
    Provides step-by-step progress tracking with time estimation,
    engaging messages, and visual progress indicators.
    """
    
    def __init__(self, animation_engine: Optional[AnimationEngine] = None, 
                 color_enabled: bool = True):
        """
        Initialize the progress tracker.
        
        Args:
            animation_engine: Animation engine for visual effects
            color_enabled: Whether to use colors in output
        """
        self.animation_engine = animation_engine or AnimationEngine(color_enabled)
        self.color_enabled = color_enabled
        
        # Progress state
        self._operation_name = ""
        self._steps: List[ProgressStep] = []
        self._current_step_index = 0
        self._start_time: Optional[datetime] = None
        self._total_weight = 0.0
        self._completed_weight = 0.0
        self._is_active = False
        
        # Messages and tips
        self._status_messages: List[str] = []
        self._tips: List[str] = [
            "The agent is analyzing the problem step by step",
            "Complex tasks are broken down into manageable pieces",
            "Each step builds on the previous ones for better results",
            "The agent learns from each action to improve its approach",
            "Progress tracking helps you understand what's happening",
            "Patience leads to better outcomes with complex reasoning"
        ]
        
        # Color codes
        self._colors = {
            'green': '\033[92m',
            'blue': '\033[94m',
            'yellow': '\033[93m',
            'cyan': '\033[96m',
            'magenta': '\033[95m',
            'red': '\033[91m',
            'gray': '\033[90m',
            'reset': '\033[0m'
        } if color_enabled else {key: '' for key in ['green', 'blue', 'yellow', 'cyan', 'magenta', 'red', 'gray', 'reset']}
    
    def start_progress(self, operation: str, total_steps: Optional[int] = None) -> None:
        """
        Start tracking progress for an operation.
        
        Args:
            operation: Description of the operation
            total_steps: Total number of steps (if known)
        """
        self._operation_name = operation
        self._steps = []
        self._current_step_index = 0
        self._start_time = datetime.now()
        self._total_weight = 0.0
        self._completed_weight = 0.0
        self._is_active = True
        self._status_messages = []
        
        # Create placeholder steps if total is known
        if total_steps:
            for i in range(total_steps):
                self._steps.append(ProgressStep(
                    name=f"Step {i + 1}",
                    description=f"Executing step {i + 1} of {total_steps}"
                ))
            self._total_weight = float(total_steps)
        
        # Show initial progress
        self._display_progress()
    
    def add_step(self, name: str, description: str, weight: float = 1.0) -> None:
        """
        Add a step to the progress tracker.
        
        Args:
            name: Name of the step
            description: Description of the step
            weight: Relative weight of this step
        """
        step = ProgressStep(name=name, description=description, weight=weight)
        self._steps.append(step)
        self._total_weight += weight
        
        if self._is_active:
            self._display_progress()
    
    def update_progress(self, current_step: int, message: str = "") -> None:
        """
        Update the progress to a specific step.
        
        Args:
            current_step: Current step number (0-based)
            message: Optional status message
        """
        if not self._is_active:
            return
        
        # Update current step index
        old_step_index = self._current_step_index
        self._current_step_index = min(current_step, len(self._steps) - 1)
        
        # Mark completed steps
        for i in range(old_step_index, self._current_step_index + 1):
            if i < len(self._steps) and not self._steps[i].completed:
                self._steps[i].completed = True
                self._steps[i].end_time = datetime.now()
                if not self._steps[i].start_time:
                    self._steps[i].start_time = self._steps[i].end_time
                self._completed_weight += self._steps[i].weight
        
        # Start current step if not started
        if (self._current_step_index < len(self._steps) and 
            not self._steps[self._current_step_index].start_time):
            self._steps[self._current_step_index].start_time = datetime.now()
        
        # Add status message
        if message:
            self._status_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
            # Keep only last 5 messages
            self._status_messages = self._status_messages[-5:]
        
        self._display_progress()
    
    def update_current_step(self, message: str) -> None:
        """
        Update the current step with a status message.
        
        Args:
            message: Status message for the current step
        """
        if not self._is_active or self._current_step_index >= len(self._steps):
            return
        
        current_step = self._steps[self._current_step_index]
        current_step.description = message
        
        if not current_step.start_time:
            current_step.start_time = datetime.now()
        
        self._display_progress()
    
    def mark_step_error(self, step_index: int, error_message: str) -> None:
        """
        Mark a step as having an error.
        
        Args:
            step_index: Index of the step with error
            error_message: Error message
        """
        if 0 <= step_index < len(self._steps):
            self._steps[step_index].error = error_message
            self._steps[step_index].end_time = datetime.now()
            if not self._steps[step_index].start_time:
                self._steps[step_index].start_time = self._steps[step_index].end_time
    
    def finish_progress(self, success: bool = True, message: str = "") -> None:
        """
        Finish the progress tracking.
        
        Args:
            success: Whether the operation was successful
            message: Final message to display
        """
        if not self._is_active:
            return
        
        self._is_active = False
        
        # Mark all remaining steps as completed if successful
        if success:
            for step in self._steps:
                if not step.completed:
                    step.completed = True
                    step.end_time = datetime.now()
                    if not step.start_time:
                        step.start_time = step.end_time
                    self._completed_weight += step.weight
        
        # Show final progress
        self._display_final_progress(success, message)
    
    def get_progress_percentage(self) -> float:
        """
        Get the current progress as a percentage.
        
        Returns:
            Progress percentage (0.0 to 1.0)
        """
        if self._total_weight == 0:
            return 0.0
        return min(1.0, self._completed_weight / self._total_weight)
    
    def get_estimated_time_remaining(self) -> Optional[timedelta]:
        """
        Get estimated time remaining for the operation.
        
        Returns:
            Estimated time remaining, or None if cannot estimate
        """
        if not self._start_time or self._completed_weight == 0:
            return None
        
        elapsed = datetime.now() - self._start_time
        progress = self.get_progress_percentage()
        
        if progress == 0:
            return None
        
        total_estimated = elapsed / progress
        remaining = total_estimated - elapsed
        
        return remaining if remaining.total_seconds() > 0 else timedelta(0)
    
    def _display_progress(self) -> None:
        """Display the current progress."""
        if not self._is_active:
            return
        
        progress = self.get_progress_percentage()
        
        # Create progress bar
        bar = self.animation_engine.create_progress_bar(
            progress, width=40, style=ProgressBarStyle.BLOCKS
        )
        
        # Get current step info
        current_step_info = ""
        if self._current_step_index < len(self._steps):
            step = self._steps[self._current_step_index]
            current_step_info = f" - {step.description}"
        
        # Get time info
        time_info = ""
        if self._start_time:
            elapsed = datetime.now() - self._start_time
            elapsed_str = self._format_duration(elapsed)
            
            eta = self.get_estimated_time_remaining()
            if eta and progress > 0.1:  # Only show ETA if we have some progress
                eta_str = self._format_duration(eta)
                time_info = f" | Elapsed: {elapsed_str} | ETA: {eta_str}"
            else:
                time_info = f" | Elapsed: {elapsed_str}"
        
        # Format the display
        operation_color = self._colors['cyan']
        reset = self._colors['reset']
        
        display_text = f"\r{operation_color}{self._operation_name}{reset}: {bar}{current_step_info}{time_info}"
        print(display_text, end="", flush=True)
    
    def _display_final_progress(self, success: bool, message: str) -> None:
        """Display the final progress result."""
        # Clear the progress line
        print("\r" + " " * 120 + "\r", end="", flush=True)
        
        # Show completion message
        if success:
            self.animation_engine.show_completion_message(
                f"{self._operation_name} completed successfully" + (f": {message}" if message else ""),
                success=True
            )
        else:
            self.animation_engine.show_completion_message(
                f"{self._operation_name} failed" + (f": {message}" if message else ""),
                success=False
            )
        
        # Show summary if we have steps
        if self._steps:
            self._show_summary()
    
    def _show_summary(self) -> None:
        """Show a summary of the completed operation."""
        if not self._steps:
            return
        
        total_steps = len(self._steps)
        completed_steps = sum(1 for step in self._steps if step.completed)
        error_steps = sum(1 for step in self._steps if step.error)
        
        summary_color = self._colors['blue']
        success_color = self._colors['green']
        error_color = self._colors['red']
        reset = self._colors['reset']
        
        print(f"\n{summary_color}Summary:{reset}")
        print(f"  Total steps: {total_steps}")
        print(f"  {success_color}Completed: {completed_steps}{reset}")
        
        if error_steps > 0:
            print(f"  {error_color}Errors: {error_steps}{reset}")
        
        if self._start_time:
            total_time = datetime.now() - self._start_time
            print(f"  Total time: {self._format_duration(total_time)}")
        
        # Show recent status messages
        if self._status_messages:
            print(f"\n{summary_color}Recent activity:{reset}")
            for msg in self._status_messages[-3:]:  # Show last 3 messages
                print(f"  {msg}")
    
    def _format_duration(self, duration: timedelta) -> str:
        """
        Format a duration for display.
        
        Args:
            duration: Duration to format
            
        Returns:
            Formatted duration string
        """
        total_seconds = int(duration.total_seconds())
        
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    def show_step_details(self) -> None:
        """Show detailed information about all steps."""
        if not self._steps:
            return
        
        details_color = self._colors['blue']
        success_color = self._colors['green']
        error_color = self._colors['red']
        pending_color = self._colors['gray']
        reset = self._colors['reset']
        
        print(f"\n{details_color}Step Details:{reset}")
        
        for i, step in enumerate(self._steps):
            status_icon = ""
            status_color = pending_color
            
            if step.error:
                status_icon = "❌"
                status_color = error_color
            elif step.completed:
                status_icon = "✅"
                status_color = success_color
            elif i == self._current_step_index:
                status_icon = "🔄"
                status_color = self._colors['yellow']
            else:
                status_icon = "⏳"
                status_color = pending_color
            
            duration_str = ""
            if step.start_time and step.end_time:
                duration = step.end_time - step.start_time
                duration_str = f" ({self._format_duration(duration)})"
            
            print(f"  {status_color}{status_icon} {step.name}: {step.description}{duration_str}{reset}")
            
            if step.error:
                print(f"    {error_color}Error: {step.error}{reset}")
    
    def add_tip(self, tip: str) -> None:
        """
        Add a custom tip to the tip rotation.
        
        Args:
            tip: Tip message to add
        """
        self._tips.append(tip)
    
    def show_engaging_message(self, base_message: str) -> None:
        """
        Show an engaging message with tips.
        
        Args:
            base_message: Base message to enhance
        """
        enhanced_message = self.animation_engine.create_loading_message(
            base_message, self._tips
        )
        print(enhanced_message)