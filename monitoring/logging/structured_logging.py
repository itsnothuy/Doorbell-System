#!/usr/bin/env python3
"""
Structured Logging

JSON-based structured logging for production environments with proper correlation
and contextual information.
"""

import json
import logging
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, Optional


class StructuredLogger:
    """
    Structured logger that outputs JSON-formatted logs for easy parsing and analysis.

    Features:
    - JSON output format
    - Request correlation IDs
    - Structured metadata
    - Performance metrics
    - Error tracking
    """

    def __init__(self, name: str, level: int = logging.INFO):
        """
        Initialize structured logger.

        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Add JSON formatter if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(self._create_formatter())
            self.logger.addHandler(handler)

        self.context = {}

    def _create_formatter(self) -> logging.Formatter:
        """Create JSON formatter for structured logging."""

        class JSONFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                }

                # Add exception info if present
                if record.exc_info:
                    log_data["exception"] = {
                        "type": record.exc_info[0].__name__,
                        "message": str(record.exc_info[1]),
                        "traceback": traceback.format_exception(*record.exc_info),
                    }

                # Add extra fields
                if hasattr(record, "context"):
                    log_data["context"] = record.context

                return json.dumps(log_data)

        return JSONFormatter()

    def set_context(self, **kwargs: Any) -> None:
        """
        Set context for subsequent log messages.

        Args:
            **kwargs: Context key-value pairs
        """
        self.context.update(kwargs)

    def clear_context(self) -> None:
        """Clear logging context."""
        self.context.clear()

    def _log(
        self,
        level: int,
        message: str,
        extra_data: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Internal logging method.

        Args:
            level: Logging level
            message: Log message
            extra_data: Additional data to include
            **kwargs: Additional context
        """
        context = self.context.copy()
        if extra_data:
            context.update(extra_data)
        if kwargs:
            context.update(kwargs)

        extra = {"context": context} if context else {}
        self.logger.log(level, message, extra=extra)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, exception: Optional[Exception] = None, **kwargs: Any) -> None:
        """
        Log error message.

        Args:
            message: Error message
            exception: Optional exception object
            **kwargs: Additional context
        """
        if exception:
            exc_info = (type(exception), exception, exception.__traceback__)
            self.logger.error(message, exc_info=exc_info, extra={"context": kwargs})
        else:
            self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs: Any) -> None:
        """
        Log critical message.

        Args:
            message: Critical message
            exception: Optional exception object
            **kwargs: Additional context
        """
        if exception:
            exc_info = (type(exception), exception, exception.__traceback__)
            self.logger.critical(message, exc_info=exc_info, extra={"context": kwargs})
        else:
            self._log(logging.CRITICAL, message, **kwargs)

    def log_performance(
        self, operation: str, duration_ms: float, success: bool = True, **kwargs: Any
    ) -> None:
        """
        Log performance metrics.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            success: Whether operation was successful
            **kwargs: Additional context
        """
        data = {
            "operation": operation,
            "duration_ms": duration_ms,
            "success": success,
            **kwargs,
        }
        self._log(logging.INFO, f"Performance: {operation}", extra_data=data)

    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        **kwargs: Any,
    ) -> None:
        """
        Log HTTP request.

        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            duration_ms: Request duration in milliseconds
            **kwargs: Additional context
        """
        data = {
            "http_method": method,
            "http_path": path,
            "http_status": status_code,
            "duration_ms": duration_ms,
            **kwargs,
        }
        self._log(logging.INFO, f"{method} {path} {status_code}", extra_data=data)


# Convenience function to get a structured logger
def get_structured_logger(name: str, level: int = logging.INFO) -> StructuredLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, level)
