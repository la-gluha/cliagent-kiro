"""
Tests for the performance monitoring system.

Tests operation timing, metrics collection, statistics calculation,
and threshold monitoring functionality.
"""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock
from collections import deque

from src.monitoring.performance_monitor import (
    PerformanceMonitor, PerformanceMetric, OperationStats
)


class TestPerformanceMetric:
    """Test PerformanceMetric dataclass."""
    
    def test_metric_creation(self):
        """Test creating PerformanceMetric."""
        metric = PerformanceMetric(
            name="test_metric",
            value=1.5,
            unit="seconds",
            timestamp=time.time(),
            context={"key": "value"}
        )
        
        assert metric.name == "test_metric"
        assert metric.value == 1.5
        assert metric.unit == "seconds"
        assert metric.context == {"key": "value"}
        assert isinstance(metric.timestamp, float)
    
    def test_metric_without_context(self):
        """Test creating PerformanceMetric without context."""
        metric = PerformanceMetric(
            name="simple_metric",
            value=2.0,
            unit="count",
            timestamp=time.time()
        )
        
        assert metric.context is None


class TestOperationStats:
    """Test OperationStats dataclass and properties."""
    
    def test_stats_creation(self):
        """Test creating OperationStats."""
        stats = OperationStats("test_operation")
        
        assert stats.name == "test_operation"
        assert stats.count == 0
        assert stats.total_time == 0.0
        assert stats.min_time == float('inf')
        assert stats.max_time == 0.0
        assert isinstance(stats.recent_times, deque)
    
    def test_avg_time_calculation(self):
        """Test average time calculation."""
        stats = OperationStats("test_op")
        stats.count = 3
        stats.total_time = 6.0
        
        assert stats.avg_time == 2.0
    
    def test_avg_time_zero_count(self):
        """Test average time with zero count."""
        stats = OperationStats("test_op")
        assert stats.avg_time == 0.0
    
    def test_median_time_calculation(self):
        """Test median time calculation."""
        stats = OperationStats("test_op")
        stats.recent_times.extend([1.0, 2.0, 3.0, 4.0, 5.0])
        
        assert stats.median_time == 3.0
    
    def test_median_time_empty(self):
        """Test median time with empty recent times."""
        stats = OperationStats("test_op")
        assert stats.median_time == 0.0
    
    def test_p95_time_calculation(self):
        """Test 95th percentile time calculation."""
        stats = OperationStats("test_op")
        times = [i * 0.1 for i in range(1, 101)]  # 0.1 to 10.0
        stats.recent_times.extend(times)
        
        p95 = stats.p95_time
        assert p95 >= 9.0  # Should be around 95th percentile
    
    def test_p95_time_empty(self):
        """Test 95th percentile with empty recent times."""
        stats = OperationStats("test_op")
        assert stats.p95_time == 0.0


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""
    
    def test_monitor_initialization(self):
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor(
            enable_detailed_logging=True,
            metric_retention_seconds=1800
        )
        
        assert monitor.enable_detailed_logging is True
        assert monitor.metric_retention_seconds == 1800
        assert len(monitor._operation_stats) == 0
        assert len(monitor._metrics) == 0
    
    def test_measure_operation_context_manager(self):
        """Test measuring operation with context manager."""
        monitor = PerformanceMonitor(enable_detailed_logging=False)
        
        with monitor.measure_operation("test_operation"):
            time.sleep(0.01)  # Small delay
        
        stats = monitor.get_operation_stats("test_operation")
        assert "test_operation" in stats
        assert stats["test_operation"].count == 1
        assert stats["test_operation"].total_time > 0
    
    def test_measure_operation_with_context(self):
        """Test measuring operation with context data."""
        monitor = PerformanceMonitor(enable_detailed_logging=False)
        context = {"user_id": "123", "request_type": "test"}
        
        with monitor.measure_operation("context_operation", context):
            time.sleep(0.01)
        
        metrics = monitor.get_metrics("context_operation")
        assert len(metrics) > 0
        assert metrics[0].context == context
    
    def test_measure_operation_exception_handling(self):
        """Test that operation measurement works even with exceptions."""
        monitor = PerformanceMonitor(enable_detailed_logging=False)
        
        try:
            with monitor.measure_operation("exception_operation"):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        stats = monitor.get_operation_stats("exception_operation")
        assert "exception_operation" in stats
        assert stats["exception_operation"].count == 1
    
    def test_record_custom_metric(self):
        """Test recording custom metrics."""
        monitor = PerformanceMonitor(enable_detailed_logging=False)
        
        monitor.record_metric("custom_metric", 42.0, "units", {"type": "test"})
        
        metrics = monitor.get_metrics("custom_metric")
        assert len(metrics) == 1
        assert metrics[0].name == "custom_metric"
        assert metrics[0].value == 42.0
        assert metrics[0].unit == "units"
        assert metrics[0].context == {"type": "test"}
    
    def test_get_operation_stats_all(self):
        """Test getting all operation statistics."""
        monitor = PerformanceMonitor(enable_detailed_logging=False)
        
        with monitor.measure_operation("op1"):
            time.sleep(0.01)
        
        with monitor.measure_operation("op2"):
            time.sleep(0.01)
        
        all_stats = monitor.get_operation_stats()
        assert "op1" in all_stats
        assert "op2" in all_stats
        assert len(all_stats) == 2
    
    def test_get_metrics_filtering(self):
        """Test metric filtering by name and timestamp."""
        monitor = PerformanceMonitor(enable_detailed_logging=False)
        
        start_time = time.time()
        monitor.record_metric("metric1", 1.0)
        time.sleep(0.01)
        monitor.record_metric("metric2", 2.0)
        mid_time = time.time()
        time.sleep(0.01)
        monitor.record_metric("metric1", 3.0)
        
        # Test name filtering
        metric1_only = monitor.get_metrics("metric1")
        assert len(metric1_only) == 2
        assert all("metric1" in m.name for m in metric1_only)
        
        # Test timestamp filtering
        recent_metrics = monitor.get_metrics(since_timestamp=mid_time)
        assert len(recent_metrics) == 1
        assert recent_metrics[0].value == 3.0
    
    def test_get_active_operations(self):
        """Test getting active operations."""
        monitor = PerformanceMonitor(enable_detailed_logging=False)
        
        def slow_operation():
            with monitor.measure_operation("slow_op"):
                time.sleep(0.1)
        
        # Start operation in background thread
        thread = threading.Thread(target=slow_operation)
        thread.start()
        
        time.sleep(0.05)  # Let operation start
        active_ops = monitor.get_active_operations()
        
        # Should have one active operation
        assert len(active_ops) > 0
        
        thread.join()  # Wait for completion
        
        # Should have no active operations
        active_ops = monitor.get_active_operations()
        assert len(active_ops) == 0
    
    def test_threshold_callbacks(self):
        """Test threshold callback functionality."""
        monitor = PerformanceMonitor(enable_detailed_logging=False)
        callback_called = []
        
        def threshold_callback(op_name, execution_time, stats):
            callback_called.append((op_name, execution_time))
        
        monitor.add_threshold_callback("slow_operation", 0.05, threshold_callback)
        
        # Fast operation - should not trigger callback
        with monitor.measure_operation("slow_operation"):
            time.sleep(0.01)
        
        assert len(callback_called) == 0
        
        # Slow operation - should trigger callback
        with monitor.measure_operation("slow_operation"):
            time.sleep(0.06)
        
        assert len(callback_called) == 1
        assert callback_called[0][0] == "slow_operation"
        assert callback_called[0][1] > 0.05
    
    def test_get_summary(self):
        """Test getting performance summary."""
        monitor = PerformanceMonitor(enable_detailed_logging=False)
        
        # Add some operations
        with monitor.measure_operation("op1"):
            time.sleep(0.01)
        
        with monitor.measure_operation("op2"):
            time.sleep(0.01)
        
        monitor.record_metric("custom", 1.0)
        
        summary = monitor.get_summary()
        
        assert "total_operations" in summary
        assert "active_operations" in summary
        assert "total_metrics" in summary
        assert "operation_stats" in summary
        
        assert summary["total_operations"] == 2
        assert "op1" in summary["operation_stats"]
        assert "op2" in summary["operation_stats"]
    
    def test_reset_stats(self):
        """Test resetting performance statistics."""
        monitor = PerformanceMonitor(enable_detailed_logging=False)
        
        # Add some data
        with monitor.measure_operation("test_op"):
            time.sleep(0.01)
        
        monitor.record_metric("test_metric", 1.0)
        
        # Verify data exists
        assert len(monitor.get_operation_stats()) > 0
        assert len(monitor.get_metrics()) > 0
        
        # Reset and verify data is cleared
        monitor.reset_stats()
        
        assert len(monitor.get_operation_stats()) == 0
        assert len(monitor.get_metrics()) == 0
    
    def test_metric_cleanup(self):
        """Test automatic cleanup of old metrics."""
        monitor = PerformanceMonitor(
            enable_detailed_logging=False,
            metric_retention_seconds=0.1  # Very short retention
        )
        
        # Add metric
        monitor.record_metric("old_metric", 1.0)
        assert len(monitor.get_metrics()) == 1
        
        # Wait for retention period
        time.sleep(0.2)
        
        # Add another metric to trigger cleanup
        monitor.record_metric("new_metric", 2.0)
        
        # Old metric should be cleaned up
        metrics = monitor.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].name == "new_metric"
    
    def test_concurrent_operations(self):
        """Test concurrent operation measurement."""
        monitor = PerformanceMonitor(enable_detailed_logging=False)
        
        def concurrent_operation(op_name):
            with monitor.measure_operation(op_name):
                time.sleep(0.02)
        
        # Start multiple concurrent operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=concurrent_operation, args=(f"concurrent_op_{i}",))
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # Check that all operations were recorded
        all_stats = monitor.get_operation_stats()
        assert len(all_stats) == 5
        
        for i in range(5):
            op_name = f"concurrent_op_{i}"
            assert op_name in all_stats
            assert all_stats[op_name].count == 1


class TestPerformanceMonitorIntegration:
    """Integration tests for PerformanceMonitor."""
    
    def test_detailed_logging_integration(self):
        """Test integration with logging system."""
        with patch('src.monitoring.performance_monitor.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            monitor = PerformanceMonitor(enable_detailed_logging=True)
            
            with monitor.measure_operation("logged_operation"):
                time.sleep(0.01)
            
            # Should have called logger
            assert mock_logger.info.called or any(
                call[0][0] == "logged_operation" 
                for call in mock_logger.method_calls
            )
    
    def test_threshold_callback_error_handling(self):
        """Test error handling in threshold callbacks."""
        monitor = PerformanceMonitor(enable_detailed_logging=False)
        
        def failing_callback(op_name, execution_time, stats):
            raise Exception("Callback error")
        
        monitor.add_threshold_callback("error_operation", 0.01, failing_callback)
        
        # This should not raise an exception despite callback failure
        with monitor.measure_operation("error_operation"):
            time.sleep(0.02)
        
        # Operation should still be recorded
        stats = monitor.get_operation_stats("error_operation")
        assert "error_operation" in stats
        assert stats["error_operation"].count == 1