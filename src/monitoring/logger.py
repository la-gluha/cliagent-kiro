"""
Structured logging system for the AI Agent System.

Provides centralized logging with structured output, multiple handlers,
and configurable log levels for different components.
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class LogContext:
    """Context information for structured logging."""
    component: str
    operation: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add context if available
        if hasattr(record, 'context') and record.context:
            if isinstance(record.context, LogContext):
                log_entry['context'] = asdict(record.context)
            else:
                log_entry['context'] = record.context
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info', 'context']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class SystemLogger:
    """Centralized logging system for the AI Agent System."""
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_dir: Optional[Path] = None,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        """
        Initialize the logging system.
        
        Args:
            log_level: Minimum log level to capture
            log_dir: Directory for log files (defaults to ./logs)
            enable_console: Whether to log to console
            enable_file: Whether to log to files
            max_file_size: Maximum size of log files before rotation
            backup_count: Number of backup files to keep
        """
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = log_dir or Path("logs")
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging configuration."""
        # Create log directory if it doesn't exist
        if self.enable_file:
            self.log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Add console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # Add file handlers
        if self.enable_file:
            # Main application log
            app_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / "app.log",
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            app_handler.setLevel(self.log_level)
            app_handler.setFormatter(StructuredFormatter())
            root_logger.addHandler(app_handler)
            
            # Error log
            error_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / "error.log",
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(StructuredFormatter())
            root_logger.addHandler(error_handler)
            
            # Performance log
            perf_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / "performance.log",
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            perf_handler.setLevel(logging.INFO)
            perf_handler.setFormatter(StructuredFormatter())
            
            # Create performance logger
            perf_logger = logging.getLogger("performance")
            perf_logger.addHandler(perf_handler)
            perf_logger.setLevel(logging.INFO)
            perf_logger.propagate = False
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance for a specific component."""
        return logging.getLogger(name)
    
    def log_with_context(self, 
                        logger: logging.Logger,
                        level: int,
                        message: str,
                        context: Optional[LogContext] = None,
                        **kwargs):
        """Log a message with structured context."""
        extra = {'context': context} if context else {}
        extra.update(kwargs)
        logger.log(level, message, extra=extra)


# Global logger instance
_system_logger: Optional[SystemLogger] = None


def initialize_logging(log_level: str = "INFO",
                      log_dir: Optional[Path] = None,
                      enable_console: bool = True,
                      enable_file: bool = True) -> SystemLogger:
    """Initialize the global logging system."""
    global _system_logger
    _system_logger = SystemLogger(
        log_level=log_level,
        log_dir=log_dir,
        enable_console=enable_console,
        enable_file=enable_file
    )
    return _system_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific component."""
    if _system_logger is None:
        initialize_logging()
    return _system_logger.get_logger(name)


def log_with_context(logger: logging.Logger,
                    level: int,
                    message: str,
                    context: Optional[LogContext] = None,
                    **kwargs):
    """Log a message with structured context."""
    if _system_logger is None:
        initialize_logging()
    _system_logger.log_with_context(logger, level, message, context, **kwargs)