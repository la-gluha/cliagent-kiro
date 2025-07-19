"""
Unit tests for the animation engine.

Tests the AnimationEngine class and related animation functionality
to ensure proper visual effects and timing.
"""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock
from src.display.animation_engine import AnimationEngine, SpinnerType, ProgressBarStyle
from src.interfaces.display_interface import AnimationType


class TestAnimationEngine:
    """Test cases for the AnimationEngine class."""
    
    def test_init_with_color_enabled(self):
        """Test animation engine initialization with color enabled."""
        engine = AnimationEngine(color_enabled=True)
        assert engine.color_enabled is True
        assert engine._colors['green'] == '\033[92m'
        assert engine._colors['reset'] == '\033[0m'
    
    def test_init_with_color_disabled(self):
        """Test animation engine initialization with color disabled."""
        engine = AnimationEngine(color_enabled=False)
        assert engine.color_enabled is False
        assert engine._colors['green'] == ''
        assert engine._colors['reset'] == ''
    
    @patch('builtins.print')
    def test_start_spinner_animation(self, mock_print):
        """Test starting a spinner animation."""
        engine = AnimationEngine(color_enabled=False)
        
        engine.start_animation(AnimationType.SPINNER, "Loading...")
        
        # Give the animation a moment to start
        time.sleep(0.15)
        
        # Check that animation thread is running
        assert engine._current_animation is not None
        assert engine._current_animation.is_alive()
        assert engine._animation_type == AnimationType.SPINNER
        assert engine._current_message == "Loading..."
        
        # Stop the animation
        engine.stop_animation()
        
        # Verify print was called (animation should have printed)
        assert mock_print.called
    
    @patch('builtins.print')
    def test_start_dots_animation(self, mock_print):
        """Test starting a dots animation."""
        engine = AnimationEngine(color_enabled=False)
        
        engine.start_animation(AnimationType.DOTS, "Processing")
        
        # Give the animation a moment to start
        time.sleep(0.1)
        
        # Check that animation is running
        assert engine._current_animation is not None
        assert engine._current_animation.is_alive()
        assert engine._animation_type == AnimationType.DOTS
        
        engine.stop_animation()
    
    @patch('builtins.print')
    def test_start_pulse_animation(self, mock_print):
        """Test starting a pulse animation."""
        engine = AnimationEngine(color_enabled=False)
        
        engine.start_animation(AnimationType.PULSE, "Thinking")
        
        # Give the animation a moment to start
        time.sleep(0.1)
        
        # Check that animation is running
        assert engine._current_animation is not None
        assert engine._current_animation.is_alive()
        assert engine._animation_type == AnimationType.PULSE
        
        engine.stop_animation()
    
    def test_stop_animation(self):
        """Test stopping an animation."""
        engine = AnimationEngine(color_enabled=False)
        
        # Start an animation
        engine.start_animation(AnimationType.SPINNER, "Test")
        time.sleep(0.1)
        
        # Verify it's running
        assert engine._current_animation is not None
        assert engine._current_animation.is_alive()
        
        # Stop the animation
        engine.stop_animation()
        
        # Verify it's stopped
        assert engine._current_animation is None
        assert engine._animation_type is None
    
    def test_update_animation_message(self):
        """Test updating animation message."""
        engine = AnimationEngine(color_enabled=False)
        
        engine.start_animation(AnimationType.SPINNER, "Initial message")
        time.sleep(0.1)
        
        # Update the message
        engine.update_animation_message("Updated message")
        
        # Verify the message was updated
        assert engine._current_message == "Updated message"
        
        engine.stop_animation()
    
    def test_create_progress_bar_basic(self):
        """Test creating a basic progress bar."""
        engine = AnimationEngine(color_enabled=False)
        
        # Test 0% progress
        bar = engine.create_progress_bar(0.0, width=10)
        assert "[░░░░░░░░░░] 0%" in bar
        
        # Test 50% progress
        bar = engine.create_progress_bar(0.5, width=10)
        assert "█████" in bar
        assert "50%" in bar
        
        # Test 100% progress
        bar = engine.create_progress_bar(1.0, width=10)
        assert "██████████" in bar
        assert "100%" in bar
    
    def test_create_progress_bar_without_percentage(self):
        """Test creating a progress bar without percentage."""
        engine = AnimationEngine(color_enabled=False)
        
        bar = engine.create_progress_bar(0.5, width=10, show_percentage=False)
        assert "%" not in bar
        assert "[" in bar and "]" in bar
    
    def test_create_progress_bar_different_styles(self):
        """Test creating progress bars with different styles."""
        engine = AnimationEngine(color_enabled=False)
        
        # Test blocks style
        bar = engine.create_progress_bar(0.5, style=ProgressBarStyle.BLOCKS)
        assert "█" in bar or "░" in bar
        
        # Test dots style
        bar = engine.create_progress_bar(0.5, style=ProgressBarStyle.DOTS)
        assert "●" in bar or "○" in bar
        
        # Test classic style
        bar = engine.create_progress_bar(0.5, style=ProgressBarStyle.CLASSIC)
        assert "#" in bar or "-" in bar
    
    def test_create_progress_bar_edge_cases(self):
        """Test progress bar with edge cases."""
        engine = AnimationEngine(color_enabled=False)
        
        # Test negative progress (should clamp to 0)
        bar = engine.create_progress_bar(-0.5, width=10)
        assert "0%" in bar
        
        # Test progress > 1 (should clamp to 100%)
        bar = engine.create_progress_bar(1.5, width=10)
        assert "100%" in bar
    
    @patch('builtins.print')
    def test_show_progress_bar(self, mock_print):
        """Test showing a progress bar."""
        engine = AnimationEngine(color_enabled=False)
        
        engine.show_progress_bar("Testing", 0.5, width=20)
        
        # Verify print was called
        mock_print.assert_called()
        call_args = mock_print.call_args
        
        # Check that it uses carriage return for overwriting
        assert call_args[1]['end'] == ""
        assert call_args[1]['flush'] is True
        
        # Check content
        output = call_args[0][0]
        assert "Testing:" in output
        assert "50%" in output
    
    @patch('builtins.print')
    def test_show_progress_bar_completion(self, mock_print):
        """Test showing a completed progress bar."""
        engine = AnimationEngine(color_enabled=False)
        
        engine.show_progress_bar("Testing", 1.0, width=20)
        
        # Should print newline when complete
        calls = mock_print.call_args_list
        assert len(calls) >= 1
    
    def test_create_loading_message_without_tips(self):
        """Test creating loading message without tips."""
        engine = AnimationEngine(color_enabled=False)
        
        message = engine.create_loading_message("Loading data")
        assert message == "Loading data"
    
    def test_create_loading_message_with_tips(self):
        """Test creating loading message with tips."""
        engine = AnimationEngine(color_enabled=False)
        
        tips = ["Tip 1", "Tip 2", "Tip 3"]
        message = engine.create_loading_message("Loading data", tips)
        
        assert "Loading data" in message
        assert "Tip:" in message
        # Should contain one of the tips
        assert any(tip in message for tip in tips)
    
    @patch('builtins.print')
    def test_show_completion_message_success(self, mock_print):
        """Test showing a success completion message."""
        engine = AnimationEngine(color_enabled=False)
        
        engine.show_completion_message("Task completed", success=True)
        
        mock_print.assert_called_once()
        output = mock_print.call_args[0][0]
        assert "✅" in output
        assert "Task completed" in output
    
    @patch('builtins.print')
    def test_show_completion_message_failure(self, mock_print):
        """Test showing a failure completion message."""
        engine = AnimationEngine(color_enabled=False)
        
        engine.show_completion_message("Task failed", success=False)
        
        mock_print.assert_called_once()
        output = mock_print.call_args[0][0]
        assert "❌" in output
        assert "Task failed" in output
    
    def test_create_status_line_basic(self):
        """Test creating a basic status line."""
        engine = AnimationEngine(color_enabled=False)
        
        status = engine.create_status_line("RUNNING")
        assert "[RUNNING]" in status
    
    def test_create_status_line_with_details(self):
        """Test creating a status line with details."""
        engine = AnimationEngine(color_enabled=False)
        
        status = engine.create_status_line("RUNNING", "Processing file.txt")
        assert "[RUNNING]" in status
        assert "Processing file.txt" in status
    
    @patch('builtins.print')
    def test_convenience_animation_methods(self, mock_print):
        """Test convenience methods for common animations."""
        engine = AnimationEngine(color_enabled=False)
        
        # Test thinking animation
        engine.show_thinking_animation("Analyzing")
        time.sleep(0.1)
        assert engine._animation_type == AnimationType.DOTS
        assert "Analyzing" in engine._current_message
        engine.stop_animation()
        
        # Test working animation
        engine.show_working_animation("Building")
        time.sleep(0.1)
        assert engine._animation_type == AnimationType.SPINNER
        assert "Building" in engine._current_message
        engine.stop_animation()
        
        # Test processing animation
        engine.show_processing_animation("Computing")
        time.sleep(0.1)
        assert engine._animation_type == AnimationType.PULSE
        assert "Computing" in engine._current_message
        engine.stop_animation()
    
    def test_animation_thread_safety(self):
        """Test that animation operations are thread-safe."""
        engine = AnimationEngine(color_enabled=False)
        
        # Start multiple animations rapidly
        for i in range(5):
            engine.start_animation(AnimationType.SPINNER, f"Test {i}")
            time.sleep(0.01)
        
        # Should only have one active animation
        assert engine._current_animation is not None
        assert engine._current_animation.is_alive()
        
        engine.stop_animation()
        assert engine._current_animation is None
    
    def test_progress_bar_with_color(self):
        """Test progress bar with color enabled."""
        engine = AnimationEngine(color_enabled=True)
        
        bar = engine.create_progress_bar(0.5, width=10)
        
        # Should contain ANSI color codes
        assert '\033[' in bar  # ANSI escape sequence
        assert engine._colors['green'] in bar or engine._colors['yellow'] in bar
    
    def test_spinner_types_enum(self):
        """Test that spinner types are properly defined."""
        # Test that all spinner types have valid character lists
        assert len(SpinnerType.DOTS.value) > 0
        assert len(SpinnerType.CLOCK.value) > 0
        assert len(SpinnerType.ARROWS.value) > 0
        assert len(SpinnerType.BARS.value) > 0
        assert len(SpinnerType.BOUNCE.value) > 0
        assert len(SpinnerType.PULSE.value) > 0
        
        # Test that all characters are strings
        for spinner_type in SpinnerType:
            for char in spinner_type.value:
                assert isinstance(char, str)
                assert len(char) > 0
    
    def test_progress_bar_styles_enum(self):
        """Test that progress bar styles are properly defined."""
        for style in ProgressBarStyle:
            style_dict = style.value
            assert 'filled' in style_dict
            assert 'empty' in style_dict
            assert isinstance(style_dict['filled'], str)
            assert isinstance(style_dict['empty'], str)
            
            if 'partial' in style_dict:
                assert isinstance(style_dict['partial'], list)
                assert len(style_dict['partial']) > 0


if __name__ == "__main__":
    pytest.main([__file__])