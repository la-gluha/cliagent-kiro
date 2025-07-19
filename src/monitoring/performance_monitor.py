"""
Performance monitoring system for the AI Agent System.

Tracks operation timing, throughput, and system performance metrics
to identify bottlenecks and optimize system performance.
"""

import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
import statistics
import logging

from .logger import get_logger, LogContext, log_with_context


@dataclass
class PerformanceMetric:
    """Represents a performance metric measurement."""
    name: str
    value: float
    unit: str
    timestamp: float
    context: Optional[Dict[str, Any]] = None


@dataclass
class OperationStats:
    """Statistics for a specific operation."""
    name: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def avg_time(self) -> float:
        """Average execution time."""
        return self.total_time / self.count if self.count > 0 else 0.0
    
    @property
    def median_time(self) -> float:
        """Median execution time from recent measurements."""
        if not self.recent_times:
            return 0.0
        return statistics.median(self.recent_times)
    
    @property
    def p95_time(self) -> float:
        """95th percentile execution time from recent measurements."""
        if not self.recent_times:
            return 0.0
        sorted_times = sorted(self.recent_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[min(index, len(sorted_times) - 1)]


class PerformanceMonitor:
    """Monitors and tracks system performance metrics."""
    
    def __init__(self, 
                 enable_detailed_logging: bool = True,
                 metric_retention_seconds: int = 3600):
        """
        Initialize the performance monitor.
        
        Args:
            enable_detailed_logging: Whether to log detailed performance metrics
            metric_retention_seconds: How long to retain metrics in memory
        """
        self.enable_detailed_logging = enable_detailed_logging
        self.metric_retention_seconds = metric_retention_seconds
        self.logger = get_logger("performance")
        
        # Thread-safe storage for metrics
        self._lock = threading.RLock()
        self._operation_stats: Dict[str, OperationStats] = {}
        self._metrics: List[PerformanceMetric] = []
        self._active_operations: Dict[str, float] = {}
        
        # Callbacks for metric thresholds
        self._threshold_callbacks: Dict[str, List[Callable]] = defaultdict(list)
    
    @contextmanager
    def measure_operation(self, operation_name: str, context: Optional[Dict[str, Any]] = None):
        """
        Context manager to measure operation execution time.
        
        Args:
            operation_name: Name of the operation being measured
            context: Additional context information
        """
        start_time = time.time()
        operation_id = f"{operation_name}_{threading.get_ident()}_{start_time}"
        
        with self._lock:
            self._active_operations[operation_id] = start_time
        
        try:
            yield
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            
            with self._lock:
                self._active_operations.pop(operation_id, None)
                self._record_operation_time(operation_name, execution_time, context)
    
    def _record_operation_time(self, 
                              operation_name: str, 
                              execution_time: float,
                              context: Optional[Dict[str, Any]] = None):
        """Record execution time for an operation."""
        # Update operation statistics
        if operation_name not in self._operation_stats:
            self._operation_stats[operation_name] = OperationStats(operation_name)
        
        stats = self._operation_stats[operation_name]
        stats.count += 1
        stats.total_time += execution_time
        stats.min_time = min(stats.min_time, execution_time)
        stats.max_time = max(stats.max_time, execution_time)
        stats.recent_times.append(execution_time)
        
        # Record metric
        metric = PerformanceMetric(
            name=f"{operation_name}_duration",
            value=execution_time,
            unit="seconds",
            timestamp=time.time(),
            context=context
        )
        self._metrics.append(metric)
        
        # Log performance metric
        if self.enable_detailed_logging:
            log_context = LogContext(
                component="performance_monitor",
                operation=operation_name,
                extra=context
            )
            log_with_context(
                self.logger,
                logging.INFO,
                f"Operation '{operation_name}' completed in {execution_time:.4f}s",
                log_context,
                execution_time=execution_time,
                operation_stats=stats.__dict__
            )
        
        # Check thresholds
        self._check_thresholds(operation_name, execution_time, stats)
        
        # Clean up old metrics
        self._cleanup_old_metrics()
    
    def record_metric(self, 
                     name: str, 
                     value: float, 
                     unit: str = "count",
                     context: Optional[Dict[str, Any]] = None):
        """Record a custom metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            context=context
        )
        
        with self._lock:
            self._metrics.append(metric)
        
        if self.enable_detailed_logging:
            log_context = LogContext(
                component="performance_monitor",
                operation="custom_metric",
                extra=context
            )
            log_with_context(
                self.logger,
                logging.INFO,
                f"Metric '{name}': {value} {unit}",
                log_context,
                metric_name=name,
                metric_value=value,
                metric_unit=unit
            )
    
    def get_operation_stats(self, operation_name: Optional[str] = None) -> Dict[str, OperationStats]:
        """Get operation statistics."""
        with self._lock:
            if operation_name:
                return {operation_name: self._operation_stats.get(operation_name)}
            return dict(self._operation_stats)
    
    def get_metrics(self, 
                   name_filter: Optional[str] = None,
                   since_timestamp: Optional[float] = None) -> List[PerformanceMetric]:
        """Get recorded metrics with optional filtering."""
        with self._lock:
            metrics = self._metrics.copy()
        
        if name_filter:
            metrics = [m for m in metrics if name_filter in m.name]
        
        if since_timestamp:
            metrics = [m for m in metrics if m.timestamp >= since_timestamp]
        
        return metrics
    
    def get_active_operations(self) -> Dict[str, float]:
        """Get currently active operations and their start times."""
        with self._lock:
            current_time = time.time()
            return {
                op_id: current_time - start_time 
                for op_id, start_time in self._active_operations.items()
            }
    
    def add_threshold_callback(self, 
                              operation_name: str,
                              threshold_seconds: float,
                              callback: Callable[[str, float, OperationStats], None]):
        """Add a callback to be triggered when operation exceeds threshold."""
        def threshold_check(op_name: str, execution_time: float, stats: OperationStats):
            if execution_time > threshold_seconds:
                callback(op_name, execution_time, stats)
        
        self._threshold_callbacks[operation_name].append(threshold_check)
    
    def _check_thresholds(self, operation_name: str, execution_time: float, stats: OperationStats):
        """Check if operation execution time exceeds any thresholds."""
        for callback in self._threshold_callbacks.get(operation_name, []):
            try:
                callback(operation_name, execution_time, stats)
            except Exception as e:
                self.logger.error(f"Error in threshold callback: {e}", exc_info=True)
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff_time = time.time() - self.metric_retention_seconds
        self._metrics = [m for m in self._metrics if m.timestamp >= cutoff_time]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        with self._lock:
            summary = {
                'total_operations': sum(stats.count for stats in self._operation_stats.values()),
                'active_operations': len(self._active_operations),
                'total_metrics': len(self._metrics),
                'operation_stats': {}
            }
            
            for name, stats in self._operation_stats.items():
                summary['operation_stats'][name] = {
                    'count': stats.count,
                    'avg_time': stats.avg_time,
                    'median_time': stats.median_time,
                    'p95_time': stats.p95_time,
                    'min_time': stats.min_time,
                    'max_time': stats.max_time
                }
            
            return summary
    
    def reset_stats(self):
        """Reset all performance statistics."""
        with self._lock:
            self._operation_stats.clear()
            self._metrics.clear()
            self._active_operations.clear()
        
        self.logger.info("Performance statistics reset")