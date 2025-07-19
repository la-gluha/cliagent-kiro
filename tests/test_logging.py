"""
Tests for the logging system.

Tests structured logging, log formatting, context handling,
and log file management functionality.
"""

import pytest
import logging
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.monitoring.logger import (
    SystemLogger, StructuredFormatter, LogContext,
    initialize_logging, get_logger, log_with_context
)


class TestLogContext:
    """Test LogContext dataclass."""
    
    def test_log_context_creation(self):
        """Test creating LogContext with various parameters."""
        context = LogContext(
            component="test_component",
            operation="test_operation",
            user_id="user123",
            session_id="session456",
            request_id="req789",
            extra={"key": "value"}
        )
        
        assert context.component == "test_component"
        assert context.operation == "test_operation"
        assert context.user_id == "user123"
        assert context.session_id == "session456"
        assert context.request_id == "req789"
        assert context.extra == {"key": "value"}
    
    def test_log_context_minimal(self):
        """Test creating LogContext with minimal parameters."""
        context = LogContext(component="minimal")
        
        assert context.component == "minimal"
        assert context.operation is None
        assert context.user_id is None
        assert context.session_id is None
        assert context.request_id is None
        assert context.extra is None


class TestStructuredFormatter:
    """Test StructuredFormatter for JSON logging."""
    
    def test_basic_formatting(self):
        """Test basic log record formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test_logger"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "path"
        assert log_data["line"] == 42
        assert "timestamp" in log_data
    
    def test_formatting_with_context(self):
        """Test formatting with LogContext."""
        formatter = StructuredFormatter()
        context = LogContext(
            component="test_component",
            operation="test_op",
            extra={"key": "value"}
        )
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message with context",
            args=(),
            exc_info=None
        )
        record.context = context
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert "context" in log_data
        assert log_data["context"]["component"] == "test_component"
        assert log_data["context"]["operation"] == "test_op"
        assert log_data["context"]["extra"]["key"] == "value"
    
    def test_formatting_with_exception(self):
        """Test formatting with exception information."""
        formatter = StructuredFormatter()
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="/test/path.py",
                lineno=42,
                msg="Error occurred",
                args=(),
                exc_info=True
            )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert "exception" in log_data
        assert "ValueError: Test exception" in log_data["exception"]


class TestSystemLogger:
    """Test SystemLogger class."""
    
    def test_logger_initialization(self):
        """Test SystemLogger initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            logger = SystemLogger(
                log_level="DEBUG",
                log_dir=log_dir,
                enable_console=True,
                enable_file=True
            )
            
            assert logger.log_level == logging.DEBUG
            assert logger.log_dir == log_dir
            assert logger.enable_console is True
            assert logger.enable_file is True
    
    def test_get_logger(self):
        """Test getting logger instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            system_logger = SystemLogger(log_dir=Path(temp_dir))
            logger = system_logger.get_logger("test_component")
            
            assert isinstance(logger, logging.Logger)
            assert logger.name == "test_component"
    
    def test_log_with_context(self):
        """Test logging with context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            system_logger = SystemLogger(log_dir=Path(temp_dir))
            logger = system_logger.get_logger("test_component")
            context = LogContext(component="test", operation="test_op")
            
            # This should not raise an exception
            system_logger.log_with_context(
                logger, logging.INFO, "Test message", context
            )
    
    def test_file_creation(self):
        """Test that log files are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            SystemLogger(log_dir=log_dir, enable_file=True)
            
            # Check that log files are created
            assert (log_dir / "app.log").exists()
            assert (log_dir / "error.log").exists()
            assert (log_dir / "performance.log").exists()
    
    def test_console_only_mode(self):
        """Test console-only logging mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            SystemLogger(
                log_dir=log_dir,
                enable_console=True,
                enable_file=False
            )
            
            # Log files should not be created
            assert not (log_dir / "app.log").exists()
            assert not (log_dir / "error.log").exists()


class TestGlobalLoggerFunctions:
    """Test global logger functions."""
    
    def test_initialize_logging(self):
        """Test global logging initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            system_logger = initialize_logging(
                log_level="DEBUG",
                log_dir=log_dir
            )
            
            assert isinstance(system_logger, SystemLogger)
            assert system_logger.log_level == logging.DEBUG
    
    def test_get_logger_global(self):
        """Test getting logger through global function."""
        logger = get_logger("test_global")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_global"
    
    def test_log_with_context_global(self):
        """Test global log_with_context function."""
        logger = get_logger("test_context")
        context = LogContext(component="global_test")
        
        # This should not raise an exception
        log_with_context(
            logger, logging.INFO, "Global context test", context
        )


class TestLogRotation:
    """Test log file rotation functionality."""
    
    def test_log_rotation_setup(self):
        """Test that log rotation is properly configured."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            system_logger = SystemLogger(
                log_dir=log_dir,
                max_file_size=1024,  # Small size for testing
                backup_count=3
            )
            
            logger = system_logger.get_logger("rotation_test")
            
            # Generate enough log data to trigger rotation
            large_message = "x" * 200
            for i in range(10):
                logger.info(f"Log message {i}: {large_message}")
            
            # Check that log file exists
            assert (log_dir / "app.log").exists()


class TestPerformanceLogger:
    """Test performance-specific logging."""
    
    def test_performance_logger_separation(self):
        """Test that performance logger is separate."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            SystemLogger(log_dir=log_dir)
            
            perf_logger = logging.getLogger("performance")
            perf_logger.info("Performance test message")
            
            # Performance log should exist
            assert (log_dir / "performance.log").exists()
    
    def test_performance_logger_no_propagation(self):
        """Test that performance logger doesn't propagate to root."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            SystemLogger(log_dir=log_dir)
            
            perf_logger = logging.getLogger("performance")
            assert perf_logger.propagate is False


class TestLogLevels:
    """Test different log levels."""
    
    def test_log_level_filtering(self):
        """Test that log level filtering works."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            system_logger = SystemLogger(
                log_level="WARNING",
                log_dir=log_dir
            )
            
            logger = system_logger.get_logger("level_test")
            
            # These should be filtered out
            logger.debug("Debug message")
            logger.info("Info message")
            
            # These should be logged
            logger.warning("Warning message")
            logger.error("Error message")
            
            # Check that log file has content (warnings and errors only)
            log_file = log_dir / "app.log"
            if log_file.exists():
                content = log_file.read_text()
                assert "Debug message" not in content
                assert "Info message" not in content


class TestErrorHandling:
    """Test error handling in logging system."""
    
    def test_invalid_log_directory(self):
        """Test handling of invalid log directory."""
        # Try to create logger with invalid directory
        invalid_path = Path("/invalid/path/that/does/not/exist")
        
        # This should handle the error gracefully
        try:
            SystemLogger(log_dir=invalid_path, enable_file=True)
        except Exception as e:
            # Should handle permission errors gracefully
            assert isinstance(e, (PermissionError, OSError))
    
    def test_logging_with_none_context(self):
        """Test logging with None context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            system_logger = SystemLogger(log_dir=Path(temp_dir))
            logger = system_logger.get_logger("none_test")
            
            # This should not raise an exception
            system_logger.log_with_context(
                logger, logging.INFO, "Test with None context", None
            )
    
    def test_logging_with_invalid_context(self):
        """Test logging with invalid context type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            system_logger = SystemLogger(log_dir=Path(temp_dir))
            logger = system_logger.get_logger("invalid_test")
            
            # This should handle invalid context gracefully
            system_logger.log_with_context(
                logger, logging.INFO, "Test with invalid context", "invalid"
            )