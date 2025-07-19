"""
Resource tracking and monitoring for the AI Agent System.

Monitors system resources like CPU, memory, and disk usage to ensure
optimal performance and implement resource management policies.
"""

import psutil
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from collections import deque
import logging

from .logger import get_logger, LogContext, log_with_context
from .performance_monitor import PerformanceMonitor


@dataclass
class ResourceSnapshot:
    """Snapshot of system resource usage at a point in time."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    thread_count: int


@dataclass
class ResourceThreshold:
    """Defines a resource usage threshold and associated action."""
    resource_type: str  # 'cpu', 'memory', 'disk'
    threshold_percent: float
    action: str  # 'warn', 'throttle', 'alert'
    callback: Optional[Callable[[ResourceSnapshot], None]] = None


class ResourceTracker:
    """Tracks system resource usage and implements resource management."""
    
    def __init__(self, 
                 monitoring_interval: float = 5.0,
                 history_size: int = 720,  # 1 hour at 5s intervals
                 enable_continuous_monitoring: bool = True):
        """
        Initialize the resource tracker.
        
        Args:
            monitoring_interval: Seconds between resource measurements
            history_size: Number of historical snapshots to keep
            enable_continuous_monitoring: Whether to start background monitoring
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.logger = get_logger("resource_tracker")
        
        # Resource history
        self._resource_history: deque = deque(maxlen=history_size)
        self._lock = threading.RLock()
        
        # Monitoring thread
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Thresholds and callbacks
        self._thresholds: List[ResourceThreshold] = []
        self._resource_alerts: Dict[str, float] = {}  # Last alert time for each resource
        
        # Process tracking
        self._process = psutil.Process()
        self._initial_network_stats = psutil.net_io_counters()
        
        if enable_continuous_monitoring:
            self.start_monitoring()
    
    def start_monitoring(self):
        """Start continuous resource monitoring in background thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="ResourceMonitor"
        )
        self._monitoring_thread.start()
        
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous resource monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread."""
        while not self._stop_monitoring.wait(self.monitoring_interval):
            try:
                snapshot = self.capture_snapshot()
                self._check_thresholds(snapshot)
            except Exception as e:
                self.logger.error(f"Error in resource monitoring loop: {e}", exc_info=True)
    
    def capture_snapshot(self) -> ResourceSnapshot:
        """Capture current system resource usage."""
        try:
            # System-wide metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Process-specific metrics
            process_count = len(psutil.pids())
            thread_count = threading.active_count()
            
            snapshot = ResourceSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / (1024 * 1024 * 1024),
                network_bytes_sent=network.bytes_sent - self._initial_network_stats.bytes_sent,
                network_bytes_recv=network.bytes_recv - self._initial_network_stats.bytes_recv,
                process_count=process_count,
                thread_count=thread_count
            )
            
            with self._lock:
                self._resource_history.append(snapshot)
            
            # Log resource usage periodically
            if len(self._resource_history) % 12 == 0:  # Every minute at 5s intervals
                log_context = LogContext(
                    component="resource_tracker",
                    operation="resource_snapshot"
                )
                log_with_context(
                    self.logger,
                    logging.INFO,
                    f"Resource usage - CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%, Disk: {disk.percent:.1f}%",
                    log_context,
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_mb=snapshot.memory_used_mb,
                    disk_percent=disk.percent,
                    thread_count=thread_count
                )
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Error capturing resource snapshot: {e}", exc_info=True)
            raise
    
    def get_current_usage(self) -> Optional[ResourceSnapshot]:
        """Get the most recent resource usage snapshot."""
        with self._lock:
            return self._resource_history[-1] if self._resource_history else None
    
    def get_usage_history(self, 
                         minutes: Optional[int] = None) -> List[ResourceSnapshot]:
        """Get resource usage history."""
        with self._lock:
            history = list(self._resource_history)
        
        if minutes:
            cutoff_time = time.time() - (minutes * 60)
            history = [s for s in history if s.timestamp >= cutoff_time]
        
        return history
    
    def get_usage_stats(self, minutes: Optional[int] = None) -> Dict[str, Any]:
        """Get statistical summary of resource usage."""
        history = self.get_usage_history(minutes)
        
        if not history:
            return {}
        
        cpu_values = [s.cpu_percent for s in history]
        memory_values = [s.memory_percent for s in history]
        disk_values = [s.disk_usage_percent for s in history]
        
        return {
            'period_minutes': minutes or (len(history) * self.monitoring_interval / 60),
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values),
                'current': cpu_values[-1] if cpu_values else 0
            },
            'memory': {
                'avg': sum(memory_values) / len(memory_values),
                'min': min(memory_values),
                'max': max(memory_values),
                'current': memory_values[-1] if memory_values else 0
            },
            'disk': {
                'avg': sum(disk_values) / len(disk_values),
                'min': min(disk_values),
                'max': max(disk_values),
                'current': disk_values[-1] if disk_values else 0
            },
            'samples': len(history)
        }
    
    def add_threshold(self, threshold: ResourceThreshold):
        """Add a resource usage threshold."""
        self._thresholds.append(threshold)
        self.logger.info(f"Added {threshold.resource_type} threshold: {threshold.threshold_percent}% -> {threshold.action}")
    
    def _check_thresholds(self, snapshot: ResourceSnapshot):
        """Check if current resource usage exceeds any thresholds."""
        resource_values = {
            'cpu': snapshot.cpu_percent,
            'memory': snapshot.memory_percent,
            'disk': snapshot.disk_usage_percent
        }
        
        current_time = time.time()
        
        for threshold in self._thresholds:
            resource_value = resource_values.get(threshold.resource_type)
            if resource_value is None:
                continue
            
            if resource_value >= threshold.threshold_percent:
                # Check if we've already alerted recently (avoid spam)
                last_alert_key = f"{threshold.resource_type}_{threshold.threshold_percent}"
                last_alert_time = self._resource_alerts.get(last_alert_key, 0)
                
                if current_time - last_alert_time > 300:  # 5 minutes between alerts
                    self._handle_threshold_exceeded(threshold, resource_value, snapshot)
                    self._resource_alerts[last_alert_key] = current_time
    
    def _handle_threshold_exceeded(self, 
                                  threshold: ResourceThreshold,
                                  current_value: float,
                                  snapshot: ResourceSnapshot):
        """Handle a resource threshold being exceeded."""
        log_context = LogContext(
            component="resource_tracker",
            operation="threshold_exceeded"
        )
        
        message = (f"{threshold.resource_type.upper()} usage exceeded threshold: "
                  f"{current_value:.1f}% >= {threshold.threshold_percent}%")
        
        if threshold.action == 'warn':
            log_with_context(
                self.logger,
                logging.WARNING,
                message,
                log_context,
                resource_type=threshold.resource_type,
                current_value=current_value,
                threshold=threshold.threshold_percent,
                action=threshold.action
            )
        elif threshold.action == 'alert':
            log_with_context(
                self.logger,
                logging.ERROR,
                f"ALERT: {message}",
                log_context,
                resource_type=threshold.resource_type,
                current_value=current_value,
                threshold=threshold.threshold_percent,
                action=threshold.action
            )
        
        # Execute callback if provided
        if threshold.callback:
            try:
                threshold.callback(snapshot)
            except Exception as e:
                self.logger.error(f"Error in threshold callback: {e}", exc_info=True)
    
    def get_process_info(self) -> Dict[str, Any]:
        """Get information about the current process."""
        try:
            with self._process.oneshot():
                return {
                    'pid': self._process.pid,
                    'cpu_percent': self._process.cpu_percent(),
                    'memory_info': self._process.memory_info()._asdict(),
                    'memory_percent': self._process.memory_percent(),
                    'num_threads': self._process.num_threads(),
                    'create_time': self._process.create_time(),
                    'status': self._process.status()
                }
        except Exception as e:
            self.logger.error(f"Error getting process info: {e}", exc_info=True)
            return {}
    
    def is_resource_constrained(self, 
                               cpu_threshold: float = 80.0,
                               memory_threshold: float = 85.0,
                               disk_threshold: float = 90.0) -> Dict[str, bool]:
        """Check if system resources are currently constrained."""
        current = self.get_current_usage()
        if not current:
            return {'cpu': False, 'memory': False, 'disk': False}
        
        return {
            'cpu': current.cpu_percent >= cpu_threshold,
            'memory': current.memory_percent >= memory_threshold,
            'disk': current.disk_usage_percent >= disk_threshold
        }
    
    def suggest_resource_optimization(self) -> List[str]:
        """Suggest optimizations based on current resource usage."""
        suggestions = []
        current = self.get_current_usage()
        
        if not current:
            return suggestions
        
        if current.cpu_percent > 80:
            suggestions.append("High CPU usage detected. Consider reducing concurrent operations.")
        
        if current.memory_percent > 85:
            suggestions.append("High memory usage detected. Consider clearing caches or reducing memory-intensive operations.")
        
        if current.disk_usage_percent > 90:
            suggestions.append("Low disk space detected. Consider cleaning up temporary files or logs.")
        
        if current.thread_count > 50:
            suggestions.append("High thread count detected. Consider using thread pools or reducing concurrent operations.")
        
        return suggestions