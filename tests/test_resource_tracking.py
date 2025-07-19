"""
Tests for the resource tracking system.

Tests system resource monitoring, threshold management,
and resource constraint detection functionality.
"""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock
from collections import deque

from src.monitoring.resource_tracker import (
    ResourceTracker, ResourceSnapshot, ResourceThreshold
)


class TestResourceSnapshot:
    """Test ResourceSnapshot dataclass."""
    
    def test_snapshot_creation(self):
        """Test creating ResourceSnapshot."""
        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=25.5,
            memory_percent=60.0,
            memory_used_mb=4096.0,
            memory_available_mb=4096.0,
            disk_usage_percent=45.0,
            disk_free_gb=100.0,
            network_bytes_sent=1024,
            network_bytes_recv=2048,
            process_count=150,
            thread_count=8
        )
        
        assert snapshot.cpu_percent == 25.5
        assert snapshot.memory_percent == 60.0
        assert snapshot.memory_used_mb == 4096.0
        assert snapshot.disk_usage_percent == 45.0
        assert snapshot.process_count == 150
        assert snapshot.thread_count == 8


class TestResourceThreshold:
    """Test ResourceThreshold dataclass."""
    
    def test_threshold_creation(self):
        """Test creating ResourceThreshold."""
        def dummy_callback(snapshot):
            pass
        
        threshold = ResourceThreshold(
            resource_type="cpu",
            threshold_percent=80.0,
            action="warn",
            callback=dummy_callback
        )
        
        assert threshold.resource_type == "cpu"
        assert threshold.threshold_percent == 80.0
        assert threshold.action == "warn"
        assert threshold.callback == dummy_callback
    
    def test_threshold_without_callback(self):
        """Test creating ResourceThreshold without callback."""
        threshold = ResourceThreshold(
            resource_type="memory",
            threshold_percent=90.0,
            action="alert"
        )
        
        assert threshold.callback is None


class TestResourceTracker:
    """Test ResourceTracker class."""
    
    @patch('src.monitoring.resource_tracker.psutil')
    def test_tracker_initialization(self, mock_psutil):
        """Test ResourceTracker initialization."""
        # Mock psutil components
        mock_process = MagicMock()
        mock_psutil.Process.return_value = mock_process
        mock_psutil.net_io_counters.return_value = MagicMock(bytes_sent=0, bytes_recv=0)
        
        tracker = ResourceTracker(
            monitoring_interval=1.0,
            history_size=100,
            enable_continuous_monitoring=False
        )
        
        assert tracker.monitoring_interval == 1.0
        assert tracker.history_size == 100
        assert tracker._monitoring_thread is None
    
    @patch('src.monitoring.resource_tracker.psutil')
    def test_capture_snapshot(self, mock_psutil):
        """Test capturing resource snapshot."""
        # Mock psutil functions
        mock_psutil.cpu_percent.return_value = 25.0
        mock_psutil.virtual_memory.return_value = MagicMock(
            percent=60.0,
            used=4 * 1024 * 1024 * 1024,  # 4GB
            available=4 * 1024 * 1024 * 1024  # 4GB
        )
        mock_psutil.disk_usage.return_value = MagicMock(
            percent=45.0,
            free=100 * 1024 * 1024 * 1024  # 100GB
        )
        mock_psutil.net_io_counters.return_value = MagicMock(
            bytes_sent=1024,
            bytes_recv=2048
        )
        mock_psutil.pids.return_value = list(range(150))
        
        # Mock process and initial network stats
        mock_process = MagicMock()
        mock_psutil.Process.return_value = mock_process
        
        tracker = ResourceTracker(enable_continuous_monitoring=False)
        tracker._initial_network_stats = MagicMock(bytes_sent=0, bytes_recv=0)
        
        snapshot = tracker.capture_snapshot()
        
        assert isinstance(snapshot, ResourceSnapshot)
        assert snapshot.cpu_percent == 25.0
        assert snapshot.memory_percent == 60.0
        assert snapshot.disk_usage_percent == 45.0
        assert snapshot.process_count == 150
    
    @patch('src.monitoring.resource_tracker.psutil')
    def test_get_current_usage(self, mock_psutil):
        """Test getting current resource usage."""
        # Setup mocks
        mock_psutil.cpu_percent.return_value = 30.0
        mock_psutil.virtual_memory.return_value = MagicMock(
            percent=50.0, used=2**30, available=2**30
        )
        mock_psutil.disk_usage.return_value = MagicMock(percent=40.0, free=2**30)
        mock_psutil.net_io_counters.return_value = MagicMock(bytes_sent=100, bytes_recv=200)
        mock_psutil.pids.return_value = list(range(100))
        mock_psutil.Process.return_value = MagicMock()
        
        tracker = ResourceTracker(enable_continuous_monitoring=False)
        tracker._initial_network_stats = MagicMock(bytes_sent=0, bytes_recv=0)
        
        # Capture a snapshot first
        tracker.capture_snapshot()
        
        current = tracker.get_current_usage()
        assert current is not None
        assert current.cpu_percent == 30.0
    
    @patch('src.monitoring.resource_tracker.psutil')
    def test_get_usage_history(self, mock_psutil):
        """Test getting usage history."""
        # Setup mocks
        mock_psutil.cpu_percent.return_value = 25.0
        mock_psutil.virtual_memory.return_value = MagicMock(
            percent=60.0, used=2**30, available=2**30
        )
        mock_psutil.disk_usage.return_value = MagicMock(percent=45.0, free=2**30)
        mock_psutil.net_io_counters.return_value = MagicMock(bytes_sent=100, bytes_recv=200)
        mock_psutil.pids.return_value = list(range(100))
        mock_psutil.Process.return_value = MagicMock()
        
        tracker = ResourceTracker(enable_continuous_monitoring=False)
        tracker._initial_network_stats = MagicMock(bytes_sent=0, bytes_recv=0)
        
        # Capture multiple snapshots
        for i in range(5):
            tracker.capture_snapshot()
            time.sleep(0.01)
        
        history = tracker.get_usage_history()
        assert len(history) == 5
        
        # Test time-based filtering
        recent_history = tracker.get_usage_history(minutes=1)
        assert len(recent_history) <= 5
    
    @patch('src.monitoring.resource_tracker.psutil')
    def test_get_usage_stats(self, mock_psutil):
        """Test getting usage statistics."""
        # Setup mocks with varying values
        cpu_values = [20.0, 30.0, 40.0, 50.0, 60.0]
        memory_values = [40.0, 50.0, 60.0, 70.0, 80.0]
        disk_values = [30.0, 35.0, 40.0, 45.0, 50.0]
        
        mock_psutil.virtual_memory.return_value = MagicMock(
            used=2**30, available=2**30
        )
        mock_psutil.disk_usage.return_value = MagicMock(free=2**30)
        mock_psutil.net_io_counters.return_value = MagicMock(bytes_sent=100, bytes_recv=200)
        mock_psutil.pids.return_value = list(range(100))
        mock_psutil.Process.return_value = MagicMock()
        
        tracker = ResourceTracker(enable_continuous_monitoring=False)
        tracker._initial_network_stats = MagicMock(bytes_sent=0, bytes_recv=0)
        
        # Capture snapshots with varying values
        for i, (cpu, mem, disk) in enumerate(zip(cpu_values, memory_values, disk_values)):
            mock_psutil.cpu_percent.return_value = cpu
            mock_psutil.virtual_memory.return_value.percent = mem
            mock_psutil.disk_usage.return_value.percent = disk
            tracker.capture_snapshot()
        
        stats = tracker.get_usage_stats()
        
        assert 'cpu' in stats
        assert 'memory' in stats
        assert 'disk' in stats
        assert stats['cpu']['avg'] == sum(cpu_values) / len(cpu_values)
        assert stats['cpu']['min'] == min(cpu_values)
        assert stats['cpu']['max'] == max(cpu_values)
        assert stats['samples'] == 5
    
    @patch('src.monitoring.resource_tracker.psutil')
    def test_add_threshold(self, mock_psutil):
        """Test adding resource thresholds."""
        mock_psutil.Process.return_value = MagicMock()
        mock_psutil.net_io_counters.return_value = MagicMock(bytes_sent=0, bytes_recv=0)
        
        tracker = ResourceTracker(enable_continuous_monitoring=False)
        
        threshold = ResourceThreshold(
            resource_type="cpu",
            threshold_percent=80.0,
            action="warn"
        )
        
        tracker.add_threshold(threshold)
        assert len(tracker._thresholds) == 1
        assert tracker._thresholds[0] == threshold
    
    @patch('src.monitoring.resource_tracker.psutil')
    def test_threshold_checking(self, mock_psutil):
        """Test threshold checking and callback execution."""
        # Setup mocks
        mock_psutil.cpu_percent.return_value = 85.0  # Above threshold
        mock_psutil.virtual_memory.return_value = MagicMock(
            percent=60.0, used=2**30, available=2**30
        )
        mock_psutil.disk_usage.return_value = MagicMock(percent=45.0, free=2**30)
        mock_psutil.net_io_counters.return_value = MagicMock(bytes_sent=100, bytes_recv=200)
        mock_psutil.pids.return_value = list(range(100))
        mock_psutil.Process.return_value = MagicMock()
        
        tracker = ResourceTracker(enable_continuous_monitoring=False)
        tracker._initial_network_stats = MagicMock(bytes_sent=0, bytes_recv=0)
        
        # Add threshold with callback
        callback_called = []
        def test_callback(snapshot):
            callback_called.append(snapshot)
        
        threshold = ResourceThreshold(
            resource_type="cpu",
            threshold_percent=80.0,
            action="warn",
            callback=test_callback
        )
        tracker.add_threshold(threshold)
        
        # Capture snapshot - should trigger threshold
        tracker.capture_snapshot()
        
        # Callback should have been called
        assert len(callback_called) == 1
        assert callback_called[0].cpu_percent == 85.0
    
    @patch('src.monitoring.resource_tracker.psutil')
    def test_is_resource_constrained(self, mock_psutil):
        """Test resource constraint detection."""
        # Setup mocks with high resource usage
        mock_psutil.cpu_percent.return_value = 85.0
        mock_psutil.virtual_memory.return_value = MagicMock(
            percent=90.0, used=2**30, available=2**30
        )
        mock_psutil.disk_usage.return_value = MagicMock(percent=95.0, free=2**30)
        mock_psutil.net_io_counters.return_value = MagicMock(bytes_sent=100, bytes_recv=200)
        mock_psutil.pids.return_value = list(range(100))
        mock_psutil.Process.return_value = MagicMock()
        
        tracker = ResourceTracker(enable_continuous_monitoring=False)
        tracker._initial_network_stats = MagicMock(bytes_sent=0, bytes_recv=0)
        
        # Capture snapshot
        tracker.capture_snapshot()
        
        constraints = tracker.is_resource_constrained(
            cpu_threshold=80.0,
            memory_threshold=85.0,
            disk_threshold=90.0
        )
        
        assert constraints['cpu'] is True
        assert constraints['memory'] is True
        assert constraints['disk'] is True
    
    @patch('src.monitoring.resource_tracker.psutil')
    def test_suggest_resource_optimization(self, mock_psutil):
        """Test resource optimization suggestions."""
        # Setup mocks with high resource usage
        mock_psutil.cpu_percent.return_value = 85.0
        mock_psutil.virtual_memory.return_value = MagicMock(
            percent=90.0, used=2**30, available=2**30
        )
        mock_psutil.disk_usage.return_value = MagicMock(percent=95.0, free=2**30)
        mock_psutil.net_io_counters.return_value = MagicMock(bytes_sent=100, bytes_recv=200)
        mock_psutil.pids.return_value = list(range(100))
        mock_psutil.Process.return_value = MagicMock()
        
        tracker = ResourceTracker(enable_continuous_monitoring=False)
        tracker._initial_network_stats = MagicMock(bytes_sent=0, bytes_recv=0)
        
        # Mock high thread count
        with patch('threading.active_count', return_value=60):
            tracker.capture_snapshot()
            
            suggestions = tracker.suggest_resource_optimization()
            
            assert len(suggestions) > 0
            assert any("CPU" in suggestion for suggestion in suggestions)
            assert any("memory" in suggestion for suggestion in suggestions)
            assert any("disk" in suggestion for suggestion in suggestions)
            assert any("thread" in suggestion for suggestion in suggestions)
    
    @patch('src.monitoring.resource_tracker.psutil')
    def test_get_process_info(self, mock_psutil):
        """Test getting process information."""
        # Setup mock process
        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_process.cpu_percent.return_value = 15.0
        mock_process.memory_info.return_value = MagicMock(
            _asdict=lambda: {'rss': 1024*1024, 'vms': 2048*1024}
        )
        mock_process.memory_percent.return_value = 5.0
        mock_process.num_threads.return_value = 4
        mock_process.create_time.return_value = time.time() - 3600
        mock_process.status.return_value = "running"
        
        mock_psutil.Process.return_value = mock_process
        mock_psutil.net_io_counters.return_value = MagicMock(bytes_sent=0, bytes_recv=0)
        
        tracker = ResourceTracker(enable_continuous_monitoring=False)
        
        process_info = tracker.get_process_info()
        
        assert process_info['pid'] == 1234
        assert process_info['cpu_percent'] == 15.0
        assert process_info['memory_percent'] == 5.0
        assert process_info['num_threads'] == 4
        assert process_info['status'] == "running"
    
    @patch('src.monitoring.resource_tracker.psutil')
    def test_monitoring_thread_lifecycle(self, mock_psutil):
        """Test monitoring thread start and stop."""
        mock_psutil.Process.return_value = MagicMock()
        mock_psutil.net_io_counters.return_value = MagicMock(bytes_sent=0, bytes_recv=0)
        
        # Mock psutil functions for monitoring loop
        mock_psutil.cpu_percent.return_value = 25.0
        mock_psutil.virtual_memory.return_value = MagicMock(
            percent=60.0, used=2**30, available=2**30
        )
        mock_psutil.disk_usage.return_value = MagicMock(percent=45.0, free=2**30)
        mock_psutil.pids.return_value = list(range(100))
        
        tracker = ResourceTracker(
            monitoring_interval=0.1,
            enable_continuous_monitoring=False
        )
        
        # Start monitoring
        tracker.start_monitoring()
        assert tracker._monitoring_thread is not None
        assert tracker._monitoring_thread.is_alive()
        
        # Let it run briefly
        time.sleep(0.2)
        
        # Stop monitoring
        tracker.stop_monitoring()
        assert not tracker._monitoring_thread.is_alive()
    
    @patch('src.monitoring.resource_tracker.psutil')
    def test_error_handling_in_capture_snapshot(self, mock_psutil):
        """Test error handling in snapshot capture."""
        mock_psutil.Process.return_value = MagicMock()
        mock_psutil.net_io_counters.return_value = MagicMock(bytes_sent=0, bytes_recv=0)
        
        # Make cpu_percent raise an exception
        mock_psutil.cpu_percent.side_effect = Exception("CPU error")
        
        tracker = ResourceTracker(enable_continuous_monitoring=False)
        
        with pytest.raises(Exception):
            tracker.capture_snapshot()
    
    @patch('src.monitoring.resource_tracker.psutil')
    def test_threshold_callback_error_handling(self, mock_psutil):
        """Test error handling in threshold callbacks."""
        # Setup mocks
        mock_psutil.cpu_percent.return_value = 85.0
        mock_psutil.virtual_memory.return_value = MagicMock(
            percent=60.0, used=2**30, available=2**30
        )
        mock_psutil.disk_usage.return_value = MagicMock(percent=45.0, free=2**30)
        mock_psutil.net_io_counters.return_value = MagicMock(bytes_sent=100, bytes_recv=200)
        mock_psutil.pids.return_value = list(range(100))
        mock_psutil.Process.return_value = MagicMock()
        
        tracker = ResourceTracker(enable_continuous_monitoring=False)
        tracker._initial_network_stats = MagicMock(bytes_sent=0, bytes_recv=0)
        
        # Add threshold with failing callback
        def failing_callback(snapshot):
            raise Exception("Callback error")
        
        threshold = ResourceThreshold(
            resource_type="cpu",
            threshold_percent=80.0,
            action="warn",
            callback=failing_callback
        )
        tracker.add_threshold(threshold)
        
        # This should not raise an exception despite callback failure
        tracker.capture_snapshot()
        
        # Snapshot should still be captured
        current = tracker.get_current_usage()
        assert current is not None
        assert current.cpu_percent == 85.0