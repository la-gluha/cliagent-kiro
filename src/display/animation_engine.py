"""
Animation engine for the AI Agent System.

This module provides animated progress indicators, spinners, and other
engaging visual elements for the terminal interface.
"""

import time
import threading
from typing import Optional, List, Dict, Any
from enum import Enum
from ..interfaces.display_interface import AnimationType, AnimationInterface


class SpinnerType(Enum):
    """Different types of spinner animations."""
    DOTS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    CLOCK = ["🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘", "🕙", "🕚", "🕛"]
    ARROWS = ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"]
    BARS = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃", "▁"]
    BOUNCE = ["⠁", "⠂", "⠄", "⠂"]
    PULSE = ["●", "◐", "◑", "◒", "◓", "◔", "◕", "◖", "◗", "◘"]


class ProgressBarStyle(Enum):
    """Different styles of progress bars."""
    BLOCKS = {"filled": "█", "empty": "░", "partial": ["▏", "▎", "▍", "▌", "▋", "▊", "▉"]}
    DOTS = {"filled": "●", "empty": "○", "partial": ["◔", "◑", "◕"]}
    ARROWS = {"filled": "►", "empty": "▷", "partial": ["▷"]}
    CLASSIC = {"filled": "#", "empty": "-", "partial": ["#"]}


class AnimationEngine(AnimationInterface):
    """
    Manages animated elements for the terminal interface.
    
    Provides spinners, progress bars, and other animated indicators
    to create an engaging user experience during operations.
    """
    
    def __init__(self, color_enabled: bool = True):
        """
        Initialize the animation engine.
        
        Args:
            color_enabled: Whether to use colors in animations
        """
        self.color_enabled = color_enabled
        self._current_animation: Optional[threading.Thread] = None
        self._animation_stop_event = threading.Event()
        self._animation_lock = threading.Lock()
        self._current_message = ""
        self._animation_type = None
        
        # Color codes for animations
        self._colors = {
            'green': '\033[92m',
            'blue': '\033[94m',
            'yellow': '\033[93m',
            'cyan': '\033[96m',
            'magenta': '\033[95m',
            'red': '\033[91m',
            'reset': '\033[0m'
        } if color_enabled else {key: '' for key in ['green', 'blue', 'yellow', 'cyan', 'magenta', 'red', 'reset']}
    
    def start_animation(self, animation_type: AnimationType, message: str = "") -> None:
        """
        Start an animation.
        
        Args:
            animation_type: Type of animation to start
            message: Message to display with the animation
        """
        self.stop_animation()  # Stop any existing animation
        
        with self._animation_lock:
            self._current_message = message
            self._animation_type = animation_type
            self._animation_stop_event.clear()
            
            if animation_type == AnimationType.SPINNER:
                self._current_animation = threading.Thread(
                    target=self._run_spinner_animation,
                    daemon=True
                )
            elif animation_type == AnimationType.DOTS:
                self._current_animation = threading.Thread(
                    target=self._run_dots_animation,
                    daemon=True
                )
            elif animation_type == AnimationType.PULSE:
                self._current_animation = threading.Thread(
                    target=self._run_pulse_animation,
                    daemon=True
                )
            elif animation_type == AnimationType.PROGRESS_BAR:
                # Progress bar is handled differently, not as a continuous animation
                return
            
            if self._current_animation:
                self._current_animation.start()
    
    def update_animation_message(self, message: str) -> None:
        """
        Update the message displayed with the animation.
        
        Args:
            message: New message to display
        """
        with self._animation_lock:
            self._current_message = message
    
    def stop_animation(self) -> None:
        """Stop the current animation."""
        if self._current_animation and self._current_animation.is_alive():
            self._animation_stop_event.set()
            self._current_animation.join(timeout=1.0)
            
            # Clear the animation line
            print("\r" + " " * 80 + "\r", end="", flush=True)
        
        self._current_animation = None
        self._animation_type = None
    
    def _run_spinner_animation(self) -> None:
        """Run a spinner animation."""
        spinner_chars = SpinnerType.DOTS.value
        i = 0
        
        while not self._animation_stop_event.is_set():
            with self._animation_lock:
                spinner = spinner_chars[i % len(spinner_chars)]
                color = self._colors['cyan']
                reset = self._colors['reset']
                
                display_text = f"\r{color}{spinner}{reset} {self._current_message}"
                print(display_text, end="", flush=True)
            
            i += 1
            time.sleep(0.1)
    
    def _run_dots_animation(self) -> None:
        """Run a dots animation."""
        max_dots = 3
        i = 0
        
        while not self._animation_stop_event.is_set():
            with self._animation_lock:
                dots = "." * ((i % (max_dots + 1)))
                spaces = " " * (max_dots - len(dots))
                color = self._colors['blue']
                reset = self._colors['reset']
                
                display_text = f"\r{color}{self._current_message}{dots}{spaces}{reset}"
                print(display_text, end="", flush=True)
            
            i += 1
            time.sleep(0.5)
    
    def _run_pulse_animation(self) -> None:
        """Run a pulse animation."""
        pulse_chars = SpinnerType.PULSE.value
        i = 0
        
        while not self._animation_stop_event.is_set():
            with self._animation_lock:
                pulse = pulse_chars[i % len(pulse_chars)]
                color = self._colors['magenta']
                reset = self._colors['reset']
                
                display_text = f"\r{color}{pulse}{reset} {self._current_message}"
                print(display_text, end="", flush=True)
            
            i += 1
            time.sleep(0.2)
    
    def create_progress_bar(self, progress: float, width: int = 30, 
                          style: ProgressBarStyle = ProgressBarStyle.BLOCKS,
                          show_percentage: bool = True) -> str:
        """
        Create a progress bar string.
        
        Args:
            progress: Progress value between 0.0 and 1.0
            width: Width of the progress bar
            style: Style of the progress bar
            show_percentage: Whether to show percentage
            
        Returns:
            Formatted progress bar string
        """
        # Clamp progress to valid range
        progress = max(0.0, min(1.0, progress))
        
        # Get style characters
        style_chars = style.value
        filled_char = style_chars["filled"]
        empty_char = style_chars["empty"]
        partial_chars = style_chars.get("partial", [filled_char])
        
        # Calculate filled width
        filled_width = progress * width
        full_blocks = int(filled_width)
        partial_block = filled_width - full_blocks
        
        # Build the bar
        bar_parts = []
        
        # Add full blocks
        if full_blocks > 0:
            color = self._colors['green']
            reset = self._colors['reset']
            bar_parts.append(f"{color}{filled_char * full_blocks}{reset}")
        
        # Add partial block if needed
        if partial_block > 0 and full_blocks < width and len(partial_chars) > 1:
            partial_index = int(partial_block * len(partial_chars))
            if partial_index < len(partial_chars):
                color = self._colors['yellow']
                reset = self._colors['reset']
                bar_parts.append(f"{color}{partial_chars[partial_index]}{reset}")
                full_blocks += 1
        
        # Add empty blocks
        empty_blocks = width - full_blocks
        if empty_blocks > 0:
            bar_parts.append(empty_char * empty_blocks)
        
        bar = "".join(bar_parts)
        
        if show_percentage:
            percentage = int(progress * 100)
            return f"[{bar}] {percentage}%"
        else:
            return f"[{bar}]"
    
    def show_progress_bar(self, operation: str, progress: float, 
                         width: int = 30, style: ProgressBarStyle = ProgressBarStyle.BLOCKS) -> None:
        """
        Display a progress bar for an operation.
        
        Args:
            operation: Description of the operation
            progress: Progress value between 0.0 and 1.0
            width: Width of the progress bar
            style: Style of the progress bar
        """
        bar = self.create_progress_bar(progress, width, style)
        display_text = f"\r{operation}: {bar}"
        print(display_text, end="", flush=True)
        
        # Print newline when complete
        if progress >= 1.0:
            print()
    
    def create_loading_message(self, base_message: str, tips: Optional[List[str]] = None) -> str:
        """
        Create an engaging loading message with optional tips.
        
        Args:
            base_message: Base message to display
            tips: Optional list of tips to cycle through
            
        Returns:
            Formatted loading message
        """
        if not tips:
            return base_message
        
        # Cycle through tips based on current time
        tip_index = int(time.time()) % len(tips)
        tip = tips[tip_index]
        
        color = self._colors['cyan']
        reset = self._colors['reset']
        
        return f"{base_message}\n{color}💡 Tip: {tip}{reset}"
    
    def show_completion_message(self, message: str, success: bool = True) -> None:
        """
        Show a completion message with appropriate styling.
        
        Args:
            message: Completion message
            success: Whether the operation was successful
        """
        if success:
            icon = "✅"
            color = self._colors['green']
        else:
            icon = "❌"
            color = self._colors['red']
        
        reset = self._colors['reset']
        print(f"{color}{icon} {message}{reset}")
    
    def create_status_line(self, status: str, details: Optional[str] = None) -> str:
        """
        Create a status line with optional details.
        
        Args:
            status: Main status message
            details: Optional additional details
            
        Returns:
            Formatted status line
        """
        color = self._colors['blue']
        reset = self._colors['reset']
        
        if details:
            return f"{color}[{status}]{reset} {details}"
        else:
            return f"{color}[{status}]{reset}"
    
    def show_thinking_animation(self, message: str = "Thinking") -> None:
        """
        Show a thinking animation with a custom message.
        
        Args:
            message: Message to display while thinking
        """
        self.start_animation(AnimationType.DOTS, f"{message}")
    
    def show_working_animation(self, message: str = "Working") -> None:
        """
        Show a working animation with a custom message.
        
        Args:
            message: Message to display while working
        """
        self.start_animation(AnimationType.SPINNER, f"{message}")
    
    def show_processing_animation(self, message: str = "Processing") -> None:
        """
        Show a processing animation with a custom message.
        
        Args:
            message: Message to display while processing
        """
        self.start_animation(AnimationType.PULSE, f"{message}")