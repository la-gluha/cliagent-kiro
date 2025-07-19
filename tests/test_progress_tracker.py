"""
Unit tests for the progress tracker.

Tests the ProgressTracker class and related progress tracking functionality
to ensure proper step management and time estimation.
"""

import pytest
import time
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from src.display.progress_tracker import ProgressTracker, ProgressStep
from src.display.animation_engine import AnimationEngine


class TestProgressStep:
    """Test cases for the ProgressStep dataclass."""
    
    def test_progress_step_creation(self):
        """Test creating a progress step."""
        step = ProgressStep(
            name="Test Step",
            description="A test step",
            weight=2.0
        )
        
        assert step.name == "Test Step"
        assert step.description == "A test step"
        assert step.weight == 2.0
        assert step.completed is False
        assert step.start_time is None
        assert step.end_time is None
        assert step.error is None
    
    def test_progress_step_defaults(self):
        """Test progress step with default values."""
        step = ProgressStep(name="Test", description="Test description")
        
        assert step.weight == 1.0
        assert step.completed is False


class TestProgressTracker:
    """Test cases for the ProgressTracker class."""
    
    def test_init_with_animation_engine(self):
        """Test progress tracker initialization with animation engine."""
        engine = AnimationEngine(color_enabled=False)
        tracker = ProgressTracker(animation_engine=engine, color_enabled=False)
        
        assert tracker.animation_engine is engine
        assert tracker.color_enabled is False
        assert tracker._is_active is False
    
    def test_init_without_animation_engine(self):
        """Test progress tracker initialization without animation engine."""
        tracker = ProgressTracker(color_enabled=False)
        
        assert tracker.animation_engine is not None
        assert isinstance(tracker.animation_engine, AnimationEngine)
        assert tracker.color_enabled is False
    
    @patch('builtins.print')
    def test_start_progress_without_steps(self, mock_print):
        """Test starting progress without predefined steps."""
        tracker = ProgressTracker(color_enabled=False)
        
        tracker.start_progress("Test Operation")
        
        assert tracker._operation_name == "Test Operation"
        assert tracker._is_active is True
        assert tracker._start_time is not None
        assert len(tracker._steps) == 0
        assert tracker._total_weight == 0.0
        
        # Should have displayed initial progress
        mock_print.assert_called()
    
    @patch('builtins.print')
    def test_start_progress_with_total_steps(self, mock_print):
        """Test starting progress with total steps count."""
        tracker = ProgressTracker(color_enabled=False)
        
        tracker.start_progress("Test Operation", total_steps=3)
        
        assert tracker._operation_name == "Test Operation"
        assert tracker._is_active is True
        assert len(tracker._steps) == 3
        assert tracker._total_weight == 3.0
        
        # Check that placeholder steps were created
        for i, step in enumerate(tracker._steps):
            assert step.name == f"Step {i + 1}"
            assert f"step {i + 1} of 3" in step.description.lower()
    
    @patch('builtins.print')
    def test_add_step(self, mock_print):
        """Test adding a step to the tracker."""
        tracker = ProgressTracker(color_enabled=False)
        tracker.start_progress("Test Operation")
        
        tracker.add_step("Custom Step", "A custom step", weight=2.0)
        
        assert len(tracker._steps) == 1
        assert tracker._steps[0].name == "Custom Step"
        assert tracker._steps[0].description == "A custom step"
        assert tracker._steps[0].weight == 2.0
        assert tracker._total_weight == 2.0
    
    @patch('builtins.print')
    def test_update_progress(self, mock_print):
        """Test updating progress to a specific step."""
        tracker = ProgressTracker(color_enabled=False)
        tracker.start_progress("Test Operation", total_steps=3)
        
        # Update to step 1 (0-based indexing)
        tracker.update_progress(1, "Completed step 1")
        
        # Check that steps 0 and 1 are marked as completed
        assert tracker._steps[0].completed is True
        assert tracker._steps[1].completed is True
        assert tracker._steps[2].completed is False
        
        # Check that completed weight is updated
        assert tracker._completed_weight == 2.0
        
        # Check that status message was added
        assert len(tracker._status_messages) == 1
        assert "Completed step 1" in tracker._status_messages[0]
    
    @patch('builtins.print')
    def test_update_current_step(self, mock_print):
        """Test updating the current step description."""
        tracker = ProgressTracker(color_enabled=False)
        tracker.start_progress("Test Operation", total_steps=2)
        
        tracker.update_current_step("Processing data...")
        
        # Current step (index 0) should have updated description
        assert tracker._steps[0].description == "Processing data..."
        assert tracker._steps[0].start_time is not None
    
    def test_mark_step_error(self):
        """Test marking a step as having an error."""
        tracker = ProgressTracker(color_enabled=False)
        tracker.start_progress("Test Operation", total_steps=2)
        
        tracker.mark_step_error(0, "Something went wrong")
        
        step = tracker._steps[0]
        assert step.error == "Something went wrong"
        assert step.end_time is not None
        assert step.start_time is not None
    
    @patch('builtins.print')
    def test_finish_progress_success(self, mock_print):
        """Test finishing progress successfully."""
        tracker = ProgressTracker(color_enabled=False)
        tracker.start_progress("Test Operation", total_steps=2)
        
        tracker.finish_progress(success=True, message="All done!")
        
        assert tracker._is_active is False
        
        # All steps should be marked as completed
        for step in tracker._steps:
            assert step.completed is True
            assert step.end_time is not None
        
        assert tracker._completed_weight == tracker._total_weight
        
        # Should have printed completion message
        mock_print.assert_called()
    
    @patch('builtins.print')
    def test_finish_progress_failure(self, mock_print):
        """Test finishing progress with failure."""
        tracker = ProgressTracker(color_enabled=False)
        tracker.start_progress("Test Operation", total_steps=2)
        
        tracker.finish_progress(success=False, message="Failed!")
        
        assert tracker._is_active is False
        
        # Steps should not be automatically completed on failure
        incomplete_steps = [step for step in tracker._steps if not step.completed]
        assert len(incomplete_steps) > 0
    
    def test_get_progress_percentage(self):
        """Test getting progress percentage."""
        tracker = ProgressTracker(color_enabled=False)
        tracker.start_progress("Test Operation", total_steps=4)
        
        # Initially 0%
        assert tracker.get_progress_percentage() == 0.0
        
        # Complete 2 out of 4 steps
        tracker._completed_weight = 2.0
        assert tracker.get_progress_percentage() == 0.5
        
        # Complete all steps
        tracker._completed_weight = 4.0
        assert tracker.get_progress_percentage() == 1.0
    
    def test_get_progress_percentage_no_steps(self):
        """Test getting progress percentage with no steps."""
        tracker = ProgressTracker(color_enabled=False)
        tracker.start_progress("Test Operation")
        
        # Should return 0% when no steps are defined
        assert tracker.get_progress_percentage() == 0.0
    
    def test_get_estimated_time_remaining(self):
        """Test getting estimated time remaining."""
        tracker = ProgressTracker(color_enabled=False)
        tracker.start_progress("Test Operation", total_steps=4)
        
        # No estimate initially
        eta = tracker.get_estimated_time_remaining()
        assert eta is None
        
        # Simulate some progress and elapsed time
        tracker._start_time = datetime.now() - timedelta(seconds=10)
        tracker._completed_weight = 1.0  # 25% complete
        
        eta = tracker.get_estimated_time_remaining()
        assert eta is not None
        assert isinstance(eta, timedelta)
        # Should estimate about 30 more seconds (10s for 25% = 40s total - 10s elapsed)
        assert 20 <= eta.total_seconds() <= 40
    
    def test_get_estimated_time_remaining_edge_cases(self):
        """Test ETA calculation edge cases."""
        tracker = ProgressTracker(color_enabled=False)
        tracker.start_progress("Test Operation", total_steps=2)
        
        # No start time
        tracker._start_time = None
        assert tracker.get_estimated_time_remaining() is None
        
        # No progress made
        tracker._start_time = datetime.now()
        tracker._completed_weight = 0.0
        assert tracker.get_estimated_time_remaining() is None
        
        # Already completed
        tracker._completed_weight = 2.0
        eta = tracker.get_estimated_time_remaining()
        assert eta is not None
        assert eta.total_seconds() == 0
    
    def test_format_duration(self):
        """Test duration formatting."""
        tracker = ProgressTracker(color_enabled=False)
        
        # Test seconds
        duration = timedelta(seconds=45)
        formatted = tracker._format_duration(duration)
        assert formatted == "45s"
        
        # Test minutes and seconds
        duration = timedelta(minutes=2, seconds=30)
        formatted = tracker._format_duration(duration)
        assert formatted == "2m 30s"
        
        # Test hours and minutes
        duration = timedelta(hours=1, minutes=30, seconds=45)
        formatted = tracker._format_duration(duration)
        assert formatted == "1h 30m"
    
    @patch('builtins.print')
    def test_show_step_details(self, mock_print):
        """Test showing detailed step information."""
        tracker = ProgressTracker(color_enabled=False)
        tracker.start_progress("Test Operation", total_steps=3)
        
        # Mark some steps as completed and one with error
        tracker._steps[0].completed = True
        tracker._steps[0].start_time = datetime.now() - timedelta(seconds=5)
        tracker._steps[0].end_time = datetime.now()
        
        tracker._steps[1].error = "Test error"
        tracker._steps[1].start_time = datetime.now() - timedelta(seconds=3)
        tracker._steps[1].end_time = datetime.now()
        
        tracker.show_step_details()
        
        # Should have printed step details
        mock_print.assert_called()
        
        # Check that the output contains step information
        calls = mock_print.call_args_list
        output = " ".join([str(call[0][0]) for call in calls])
        
        assert "Step Details:" in output
        assert "✅" in output  # Completed step
        assert "❌" in output  # Error step
        assert "⏳" in output  # Pending step
    
    def test_add_tip(self):
        """Test adding custom tips."""
        tracker = ProgressTracker(color_enabled=False)
        
        initial_tip_count = len(tracker._tips)
        tracker.add_tip("Custom tip for testing")
        
        assert len(tracker._tips) == initial_tip_count + 1
        assert "Custom tip for testing" in tracker._tips
    
    @patch('builtins.print')
    def test_show_engaging_message(self, mock_print):
        """Test showing engaging messages with tips."""
        tracker = ProgressTracker(color_enabled=False)
        
        tracker.show_engaging_message("Processing your request")
        
        mock_print.assert_called_once()
        output = mock_print.call_args[0][0]
        
        assert "Processing your request" in output
        # Should contain a tip
        assert "Tip:" in output
    
    @patch('builtins.print')
    def test_status_messages_limit(self, mock_print):
        """Test that status messages are limited to prevent memory growth."""
        tracker = ProgressTracker(color_enabled=False)
        tracker.start_progress("Test Operation", total_steps=10)
        
        # Add many status messages
        for i in range(10):
            tracker.update_progress(i, f"Message {i}")
        
        # Should only keep the last 5 messages
        assert len(tracker._status_messages) == 5
        assert "Message 9" in tracker._status_messages[-1]
        assert "Message 5" in tracker._status_messages[0]
    
    @patch('builtins.print')
    def test_inactive_tracker_operations(self, mock_print):
        """Test operations on inactive tracker."""
        tracker = ProgressTracker(color_enabled=False)
        
        # Try operations without starting progress
        tracker.update_progress(1, "Should be ignored")
        tracker.update_current_step("Should be ignored")
        tracker.finish_progress(True, "Should be ignored")
        
        # Should not crash and should not print anything
        assert not mock_print.called
        assert tracker._is_active is False
    
    def test_color_codes_with_color_disabled(self):
        """Test that color codes are empty when color is disabled."""
        tracker = ProgressTracker(color_enabled=False)
        
        for color_name, color_code in tracker._colors.items():
            assert color_code == ""
    
    def test_color_codes_with_color_enabled(self):
        """Test that color codes are set when color is enabled."""
        tracker = ProgressTracker(color_enabled=True)
        
        assert tracker._colors['green'] == '\033[92m'
        assert tracker._colors['reset'] == '\033[0m'
        assert len(tracker._colors) > 0
    
    @patch('builtins.print')
    def test_progress_display_with_time_info(self, mock_print):
        """Test progress display includes time information."""
        tracker = ProgressTracker(color_enabled=False)
        tracker.start_progress("Test Operation", total_steps=2)
        
        # Simulate some elapsed time
        tracker._start_time = datetime.now() - timedelta(seconds=30)
        tracker.update_progress(0, "First step done")
        
        # Check that time information is displayed
        calls = mock_print.call_args_list
        output = " ".join([str(call[0][0]) for call in calls if call[0]])
        
        assert "Elapsed:" in output
    
    @patch('builtins.print')
    def test_summary_display(self, mock_print):
        """Test that summary is displayed after completion."""
        tracker = ProgressTracker(color_enabled=False)
        tracker.start_progress("Test Operation", total_steps=3)
        
        # Add some status messages
        tracker.update_progress(1, "Step 1 complete")
        tracker.update_progress(2, "Step 2 complete")
        
        # Mark one step with error
        tracker.mark_step_error(1, "Test error")
        
        tracker.finish_progress(success=True, message="Done")
        
        # Check that summary was displayed
        calls = mock_print.call_args_list
        output = " ".join([str(call[0][0]) for call in calls])
        
        assert "Summary:" in output
        assert "Total steps:" in output
        assert "Completed:" in output
        assert "Total time:" in output
        assert "Recent activity:" in output


if __name__ == "__main__":
    pytest.main([__file__])