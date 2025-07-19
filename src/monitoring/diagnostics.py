"""
Diagnostic utilities and commands for the AI Agent System.

Provides debugging tools, system health checks, and diagnostic commands
to help troubleshoot issues and monitor system health.
"""

import sys
import platform
import threading
import time
import traceback
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging

from .logger import get_logger, LogContext, log_with_context
from .performance_monitor import PerformanceMonitor
from .resource_tracker import ResourceTracker


@dataclass
class SystemInfo:
    """System information for diagnostics."""
    python_version: str
    platform: str
    architecture: str
    processor: str
    hostname: str
    total_memory_gb: float
    available_memory_gb: float
    cpu_count: int
    thread_count: int
    uptime_seconds: float


@dataclass
class HealthCheckResult:
    """Result of a system health check."""
    component: str
    status: str  # 'healthy', 'warning', 'error'
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class DiagnosticManager:
    """Manages system diagnostics and health monitoring."""
    
    def __init__(self, 
                 performance_monitor: Optional[PerformanceMonitor] = None,
                 resource_tracker: Optional[ResourceTracker] = None):
        """
        Initialize the diagnostic manager.
        
        Args:
            performance_monitor: Performance monitoring instance
            resource_tracker: Resource tracking instance
        """
        self.logger = get_logger("diagnostics")
        self.performance_monitor = performance_monitor
        self.resource_tracker = resource_tracker
        
        # Health check registry
        self._health_checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._register_default_health_checks()
        
        # System start time for uptime calculation
        self._start_time = time.time()
        
        # Diagnostic data collection
        self._diagnostic_data: Dict[str, Any] = {}
    
    def _register_default_health_checks(self):
        """Register default system health checks."""
        self._health_checks.update({
            'memory': self._check_memory_health,
            'disk_space': self._check_disk_health,
            'thread_count': self._check_thread_health,
            'logging': self._check_logging_health,
            'performance_monitor': self._check_performance_monitor_health,
            'resource_tracker': self._check_resource_tracker_health
        })
    
    def register_health_check(self, name: str, check_func: Callable[[], HealthCheckResult]):
        """Register a custom health check."""
        self._health_checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")
    
    def run_health_check(self, component: Optional[str] = None) -> Dict[str, HealthCheckResult]:
        """Run health checks for specified component or all components."""
        results = {}
        
        checks_to_run = {component: self._health_checks[component]} if component else self._health_checks
        
        for check_name, check_func in checks_to_run.items():
            try:
                result = check_func()
                results[check_name] = result
                
                log_level = logging.INFO
                if result.status == 'warning':
                    log_level = logging.WARNING
                elif result.status == 'error':
                    log_level = logging.ERROR
                
                log_context = LogContext(
                    component="diagnostics",
                    operation="health_check"
                )
                log_with_context(
                    self.logger,
                    log_level,
                    f"Health check '{check_name}': {result.status} - {result.message}",
                    log_context,
                    check_name=check_name,
                    status=result.status,
                    details=result.details
                )
                
            except Exception as e:
                error_result = HealthCheckResult(
                    component=check_name,
                    status='error',
                    message=f"Health check failed: {str(e)}",
                    details={'exception': str(e), 'traceback': traceback.format_exc()}
                )
                results[check_name] = error_result
                
                self.logger.error(f"Health check '{check_name}' failed: {e}", exc_info=True)
        
        return results
    
    def _check_memory_health(self) -> HealthCheckResult:
        """Check system memory health."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            if memory.percent > 90:
                return HealthCheckResult(
                    component='memory',
                    status='error',
                    message=f"Critical memory usage: {memory.percent:.1f}%",
                    details={'memory_percent': memory.percent, 'available_gb': memory.available / (1024**3)}
                )
            elif memory.percent > 80:
                return HealthCheckResult(
                    component='memory',
                    status='warning',
                    message=f"High memory usage: {memory.percent:.1f}%",
                    details={'memory_percent': memory.percent, 'available_gb': memory.available / (1024**3)}
                )
            else:
                return HealthCheckResult(
                    component='memory',
                    status='healthy',
                    message=f"Memory usage normal: {memory.percent:.1f}%",
                    details={'memory_percent': memory.percent, 'available_gb': memory.available / (1024**3)}
                )
        except Exception as e:
            return HealthCheckResult(
                component='memory',
                status='error',
                message=f"Unable to check memory: {str(e)}"
            )
    
    def _check_disk_health(self) -> HealthCheckResult:
        """Check disk space health."""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            
            if disk.percent > 95:
                return HealthCheckResult(
                    component='disk_space',
                    status='error',
                    message=f"Critical disk usage: {disk.percent:.1f}%",
                    details={'disk_percent': disk.percent, 'free_gb': disk.free / (1024**3)}
                )
            elif disk.percent > 85:
                return HealthCheckResult(
                    component='disk_space',
                    status='warning',
                    message=f"High disk usage: {disk.percent:.1f}%",
                    details={'disk_percent': disk.percent, 'free_gb': disk.free / (1024**3)}
                )
            else:
                return HealthCheckResult(
                    component='disk_space',
                    status='healthy',
                    message=f"Disk usage normal: {disk.percent:.1f}%",
                    details={'disk_percent': disk.percent, 'free_gb': disk.free / (1024**3)}
                )
        except Exception as e:
            return HealthCheckResult(
                component='disk_space',
                status='error',
                message=f"Unable to check disk space: {str(e)}"
            )
    
    def _check_thread_health(self) -> HealthCheckResult:
        """Check thread count health."""
        thread_count = threading.active_count()
        
        if thread_count > 100:
            return HealthCheckResult(
                component='thread_count',
                status='warning',
                message=f"High thread count: {thread_count}",
                details={'thread_count': thread_count}
            )
        else:
            return HealthCheckResult(
                component='thread_count',
                status='healthy',
                message=f"Thread count normal: {thread_count}",
                details={'thread_count': thread_count}
            )
    
    def _check_logging_health(self) -> HealthCheckResult:
        """Check logging system health."""
        try:
            # Test if we can write to log
            test_logger = get_logger("health_check_test")
            test_logger.info("Health check test message")
            
            return HealthCheckResult(
                component='logging',
                status='healthy',
                message="Logging system operational"
            )
        except Exception as e:
            return HealthCheckResult(
                component='logging',
                status='error',
                message=f"Logging system error: {str(e)}"
            )
    
    def _check_performance_monitor_health(self) -> HealthCheckResult:
        """Check performance monitor health."""
        if not self.performance_monitor:
            return HealthCheckResult(
                component='performance_monitor',
                status='warning',
                message="Performance monitor not initialized"
            )
        
        try:
            stats = self.performance_monitor.get_summary()
            active_ops = len(self.performance_monitor.get_active_operations())
            
            return HealthCheckResult(
                component='performance_monitor',
                status='healthy',
                message=f"Performance monitor operational",
                details={
                    'total_operations': stats.get('total_operations', 0),
                    'active_operations': active_ops,
                    'total_metrics': stats.get('total_metrics', 0)
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component='performance_monitor',
                status='error',
                message=f"Performance monitor error: {str(e)}"
            )
    
    def _check_resource_tracker_health(self) -> HealthCheckResult:
        """Check resource tracker health."""
        if not self.resource_tracker:
            return HealthCheckResult(
                component='resource_tracker',
                status='warning',
                message="Resource tracker not initialized"
            )
        
        try:
            current_usage = self.resource_tracker.get_current_usage()
            if not current_usage:
                return HealthCheckResult(
                    component='resource_tracker',
                    status='warning',
                    message="No resource data available"
                )
            
            return HealthCheckResult(
                component='resource_tracker',
                status='healthy',
                message="Resource tracker operational",
                details={
                    'last_update': current_usage.timestamp,
                    'cpu_percent': current_usage.cpu_percent,
                    'memory_percent': current_usage.memory_percent
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component='resource_tracker',
                status='error',
                message=f"Resource tracker error: {str(e)}"
            )
    
    def get_system_info(self) -> SystemInfo:
        """Get comprehensive system information."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            uptime = time.time() - self._start_time
            
            return SystemInfo(
                python_version=sys.version,
                platform=platform.platform(),
                architecture=platform.architecture()[0],
                processor=platform.processor() or "Unknown",
                hostname=platform.node(),
                total_memory_gb=memory.total / (1024**3),
                available_memory_gb=memory.available / (1024**3),
                cpu_count=psutil.cpu_count(),
                thread_count=threading.active_count(),
                uptime_seconds=uptime
            )
        except Exception as e:
            self.logger.error(f"Error getting system info: {e}", exc_info=True)
            raise
    
    def generate_diagnostic_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report."""
        report = {
            'timestamp': time.time(),
            'system_info': asdict(self.get_system_info()),
            'health_checks': {},
            'performance_summary': {},
            'resource_summary': {}
        }
        
        # Run all health checks
        health_results = self.run_health_check()
        report['health_checks'] = {
            name: asdict(result) for name, result in health_results.items()
        }
        
        # Add performance summary
        if self.performance_monitor:
            report['performance_summary'] = self.performance_monitor.get_summary()
        
        # Add resource summary
        if self.resource_tracker:
            report['resource_summary'] = self.resource_tracker.get_usage_stats(minutes=60)
            report['resource_constraints'] = self.resource_tracker.is_resource_constrained()
            report['optimization_suggestions'] = self.resource_tracker.suggest_resource_optimization()
        
        return report
    
    def save_diagnostic_report(self, filepath: Optional[Path] = None) -> Path:
        """Save diagnostic report to file."""
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = Path(f"diagnostic_report_{timestamp}.json")
        
        report = self.generate_diagnostic_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Diagnostic report saved to {filepath}")
        return filepath
    
    def get_debug_info(self, include_traceback: bool = True) -> Dict[str, Any]:
        """Get debug information for troubleshooting."""
        debug_info = {
            'timestamp': time.time(),
            'python_version': sys.version,
            'platform': platform.platform(),
            'thread_count': threading.active_count(),
            'thread_names': [t.name for t in threading.enumerate()],
        }
        
        if include_traceback:
            debug_info['stack_traces'] = {}
            for thread_id, frame in sys._current_frames().items():
                debug_info['stack_traces'][thread_id] = traceback.format_stack(frame)
        
        return debug_info
    
    def log_system_status(self):
        """Log current system status."""
        try:
            system_info = self.get_system_info()
            health_results = self.run_health_check()
            
            # Count health status
            status_counts = {'healthy': 0, 'warning': 0, 'error': 0}
            for result in health_results.values():
                status_counts[result.status] = status_counts.get(result.status, 0) + 1
            
            log_context = LogContext(
                component="diagnostics",
                operation="system_status"
            )
            
            log_with_context(
                self.logger,
                logging.INFO,
                f"System Status - Uptime: {system_info.uptime_seconds:.0f}s, "
                f"Health: {status_counts['healthy']} OK, {status_counts['warning']} warnings, {status_counts['error']} errors",
                log_context,
                system_info=asdict(system_info),
                health_summary=status_counts
            )
            
        except Exception as e:
            self.logger.error(f"Error logging system status: {e}", exc_info=True)