#!/usr/bin/env python3
"""
Error Handling Infrastructure for Message Bus

Provides comprehensive error handling, recovery strategies, circuit breaker
patterns, and error logging for the message bus system.
"""

import json
import logging
import threading
import time
import traceback
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for proper handling."""
    CONNECTION_ERROR = "connection_error"
    MESSAGE_PROCESSING_ERROR = "message_processing_error"
    SERIALIZATION_ERROR = "serialization_error"
    QUEUE_OVERFLOW = "queue_overflow"
    TIMEOUT_ERROR = "timeout_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION_ERROR = "configuration_error"


@dataclass
class ErrorReport:
    """Comprehensive error report structure."""
    error_id: str
    timestamp: float
    category: ErrorCategory
    severity: ErrorSeverity
    component: str
    message: str
    exception_type: str
    traceback: str
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error report to dictionary."""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp,
            'category': self.category.value,
            'severity': self.severity.value,
            'component': self.component,
            'message': self.message,
            'exception_type': self.exception_type,
            'traceback': self.traceback,
            'context': self.context,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful,
            'retry_count': self.retry_count
        }


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
        self._lock = threading.Lock()
    
    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result
            
        Raises:
            RuntimeError: If circuit breaker is OPEN
        """
        with self._lock:
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = 'HALF_OPEN'
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise RuntimeError("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == 'HALF_OPEN':
                    self._reset()
                    logger.info("Circuit breaker transitioning to CLOSED")
                
                return result
                
            except Exception as e:
                self._record_failure()
                raise
    
    def _record_failure(self) -> None:
        """Record a failure and update circuit breaker state."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def _reset(self) -> None:
        """Reset circuit breaker to healthy state."""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def get_state(self) -> str:
        """Get current circuit breaker state."""
        with self._lock:
            return self.state


class ErrorRecoveryManager:
    """Manages error recovery strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize error recovery manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {
            ErrorCategory.CONNECTION_ERROR: self._recover_connection_error,
            ErrorCategory.MESSAGE_PROCESSING_ERROR: self._recover_processing_error,
            ErrorCategory.QUEUE_OVERFLOW: self._recover_queue_overflow,
            ErrorCategory.TIMEOUT_ERROR: self._recover_timeout_error,
            ErrorCategory.SERIALIZATION_ERROR: self._recover_serialization_error
        }
        
        self.circuit_breakers: Dict[str, CircuitBreaker] = defaultdict(lambda: CircuitBreaker())
        self.retry_configs = {
            ErrorCategory.CONNECTION_ERROR: {'max_retries': 3, 'backoff': 2.0},
            ErrorCategory.MESSAGE_PROCESSING_ERROR: {'max_retries': 2, 'backoff': 1.0},
            ErrorCategory.TIMEOUT_ERROR: {'max_retries': 2, 'backoff': 1.5},
            ErrorCategory.SERIALIZATION_ERROR: {'max_retries': 1, 'backoff': 0.5}
        }
    
    def attempt_recovery(self, error_report: ErrorReport, context: Dict[str, Any]) -> bool:
        """
        Attempt to recover from error.
        
        Args:
            error_report: Error information
            context: Recovery context (connections, queues, etc.)
            
        Returns:
            True if recovery successful, False otherwise
        """
        try:
            strategy = self.recovery_strategies.get(error_report.category)
            if not strategy:
                logger.warning(f"No recovery strategy for {error_report.category}")
                return False
            
            # Use circuit breaker for recovery attempts
            breaker_key = f"{error_report.component}_{error_report.category.value}"
            circuit_breaker = self.circuit_breakers[breaker_key]
            
            result = circuit_breaker.call(strategy, error_report, context)
            
            error_report.recovery_attempted = True
            error_report.recovery_successful = result
            
            if result:
                logger.info(f"Successfully recovered from {error_report.category.value}")
            else:
                logger.warning(f"Recovery failed for {error_report.category.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            error_report.recovery_attempted = True
            error_report.recovery_successful = False
            return False
    
    def _recover_connection_error(self, error_report: ErrorReport, context: Dict[str, Any]) -> bool:
        """Recover from connection errors."""
        try:
            connection_manager = context.get('connection_manager')
            if not connection_manager:
                return False
            
            # Attempt to reconnect
            connection_id = error_report.context.get('connection_id')
            if connection_id and hasattr(connection_manager, 'reconnect'):
                return connection_manager.reconnect(connection_id)
            
            return False
            
        except Exception as e:
            logger.error(f"Connection recovery failed: {e}")
            return False
    
    def _recover_processing_error(self, error_report: ErrorReport, context: Dict[str, Any]) -> bool:
        """Recover from message processing errors."""
        try:
            message_queue = context.get('message_queue')
            failed_message = error_report.context.get('message')
            
            if not message_queue or not failed_message:
                return False
            
            # Check if message can be retried
            retry_config = self.retry_configs.get(error_report.category, {})
            max_retries = retry_config.get('max_retries', 1)
            
            if error_report.retry_count < max_retries:
                # Add delay before retry
                backoff = retry_config.get('backoff', 1.0)
                time.sleep(backoff * (2 ** error_report.retry_count))
                
                # Retry message processing
                error_report.retry_count += 1
                # Note: requeue_message expects a Message object, but we only have string representation
                # For now, we'll skip requeue and move to dead letter queue
                logger.info(f"Message retry not supported for string representation")
            
            # Move to dead letter queue if max retries exceeded
            dead_letter_queue = context.get('dead_letter_queue')
            if dead_letter_queue and hasattr(dead_letter_queue, 'add_message'):
                dead_letter_queue.add_message(failed_message, error_report)
                logger.info(f"Message moved to dead letter queue after {max_retries} retries")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Processing recovery failed: {e}")
            return False
    
    def _recover_queue_overflow(self, error_report: ErrorReport, context: Dict[str, Any]) -> bool:
        """Recover from queue overflow conditions."""
        try:
            queue_manager = context.get('queue_manager')
            if not queue_manager:
                return False
            
            queue_name = error_report.context.get('queue_name')
            if not queue_name:
                return False
            
            # Implement backpressure by dropping low-priority messages
            if hasattr(queue_manager, 'apply_backpressure'):
                dropped_count = queue_manager.apply_backpressure(queue_name)
                
                if dropped_count > 0:
                    logger.info(f"Applied backpressure: dropped {dropped_count} low-priority messages")
                    return True
            
            # If no low-priority messages, try to increase queue capacity temporarily
            if hasattr(queue_manager, 'can_expand_capacity') and hasattr(queue_manager, 'expand_capacity'):
                if queue_manager.can_expand_capacity(queue_name):
                    queue_manager.expand_capacity(queue_name, factor=1.5)
                    logger.info(f"Temporarily expanded capacity for queue {queue_name}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Queue overflow recovery failed: {e}")
            return False
    
    def _recover_timeout_error(self, error_report: ErrorReport, context: Dict[str, Any]) -> bool:
        """Recover from timeout errors."""
        try:
            # Increase timeout for retries
            timeout_manager = context.get('timeout_manager')
            if timeout_manager and hasattr(timeout_manager, 'increase_timeout'):
                operation = error_report.context.get('operation')
                if operation:
                    timeout_manager.increase_timeout(operation, factor=1.5)
                    logger.info(f"Increased timeout for operation {operation}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Timeout recovery failed: {e}")
            return False
    
    def _recover_serialization_error(self, error_report: ErrorReport, context: Dict[str, Any]) -> bool:
        """Recover from serialization errors."""
        try:
            # Try alternative serialization method
            serializer = context.get('serializer')
            message_data = error_report.context.get('message_data')
            
            if serializer and message_data:
                # Try JSON serialization as fallback
                if hasattr(serializer, 'fallback_serialize'):
                    return serializer.fallback_serialize(message_data)
            
            return False
            
        except Exception as e:
            logger.error(f"Serialization recovery failed: {e}")
            return False
    
    def get_circuit_breaker_status(self) -> Dict[str, str]:
        """Get status of all circuit breakers."""
        return {key: breaker.get_state() for key, breaker in self.circuit_breakers.items()}


class ErrorLogger:
    """Centralized error logging and reporting."""
    
    def __init__(self, log_dir: str, max_log_size: int = 10 * 1024 * 1024):
        """
        Initialize error logger.
        
        Args:
            log_dir: Directory for error logs
            max_log_size: Maximum log file size before rotation
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_log_size = max_log_size
        self.error_log_file = self.log_dir / "message_bus_errors.json"
        self.metrics_file = self.log_dir / "error_metrics.json"
        
        self.error_counts: Dict[Any, int] = defaultdict(int)
        self.error_history: deque = deque(maxlen=1000)  # Keep last 1000 errors
        
        self._lock = threading.Lock()
    
    def log_error(self, error_report: ErrorReport) -> None:
        """Log error report to file and update metrics."""
        with self._lock:
            try:
                # Add to history
                self.error_history.append(error_report)
                
                # Update counts
                self.error_counts[error_report.category] += 1
                self.error_counts[f"{error_report.category}_{error_report.severity}"] += 1
                
                # Write to log file
                self._write_error_log(error_report)
                
                # Update metrics
                self._update_metrics()
                
                # Log to standard logger based on severity
                log_message = f"Error {error_report.error_id}: {error_report.message}"
                
                if error_report.severity == ErrorSeverity.CRITICAL:
                    logger.critical(log_message)
                elif error_report.severity == ErrorSeverity.HIGH:
                    logger.error(log_message)
                elif error_report.severity == ErrorSeverity.MEDIUM:
                    logger.warning(log_message)
                else:
                    logger.info(log_message)
                
            except Exception as e:
                # Fallback logging to prevent error logging failures
                logger.error(f"Failed to log error report: {e}")
    
    def _write_error_log(self, error_report: ErrorReport) -> None:
        """Write error to JSON log file."""
        try:
            # Check file size and rotate if needed
            if self.error_log_file.exists() and self.error_log_file.stat().st_size > self.max_log_size:
                self._rotate_log_file()
            
            # Append error to log file
            with open(self.error_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_report.to_dict()) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to write error log: {e}")
    
    def _rotate_log_file(self) -> None:
        """Rotate log file when it gets too large."""
        try:
            timestamp = int(time.time())
            backup_file = self.log_dir / f"message_bus_errors_{timestamp}.json"
            self.error_log_file.rename(backup_file)
            
            logger.info(f"Rotated error log to {backup_file}")
            
        except Exception as e:
            logger.error(f"Failed to rotate log file: {e}")
    
    def _update_metrics(self) -> None:
        """Update error metrics file."""
        try:
            # Convert error_counts keys to strings for JSON serialization
            serializable_counts = {}
            for key, value in self.error_counts.items():
                if isinstance(key, ErrorCategory):
                    serializable_counts[key.value] = value
                else:
                    serializable_counts[str(key)] = value
            
            metrics = {
                'total_errors': len(self.error_history),
                'error_counts_by_category': serializable_counts,
                'last_updated': time.time(),
                'recent_errors': [
                    {
                        'category': err.category.value,
                        'severity': err.severity.value,
                        'timestamp': err.timestamp,
                        'component': err.component
                    }
                    for err in list(self.error_history)[-10:]  # Last 10 errors
                ]
            }
            
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self._lock:
            if not self.error_history:
                return {'total_errors': 0, 'categories': {}, 'severity_distribution': {}}
            
            # Calculate statistics
            total_errors = len(self.error_history)
            recent_errors = [err for err in self.error_history if time.time() - err.timestamp < 3600]  # Last hour
            
            category_stats: Dict[str, int] = defaultdict(int)
            severity_stats: Dict[str, int] = defaultdict(int)
            
            for error in self.error_history:
                category_stats[error.category.value] += 1
                severity_stats[error.severity.value] += 1
            
            recovery_successful_count = sum(1 for err in self.error_history if err.recovery_successful)
            
            return {
                'total_errors': total_errors,
                'recent_errors_1h': len(recent_errors),
                'categories': dict(category_stats),
                'severity_distribution': dict(severity_stats),
                'error_rate_1h': len(recent_errors) / 60.0,  # Errors per minute
                'recovery_success_rate': recovery_successful_count / total_errors if total_errors > 0 else 0.0
            }


class DeadLetterQueue:
    """Dead letter queue for failed messages."""
    
    def __init__(self, storage_dir: str):
        """
        Initialize dead letter queue.
        
        Args:
            storage_dir: Directory for storing failed messages
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.messages: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def add_message(self, message: Any, error_report: ErrorReport) -> None:
        """
        Add message to dead letter queue.
        
        Args:
            message: Failed message
            error_report: Associated error report
        """
        with self._lock:
            try:
                dead_letter = {
                    'message': str(message),
                    'error_id': error_report.error_id,
                    'timestamp': time.time(),
                    'error_category': error_report.category.value,
                    'retry_count': error_report.retry_count
                }
                
                self.messages.append(dead_letter)
                
                # Persist to disk
                self._persist_message(dead_letter)
                
                logger.info(f"Added message to dead letter queue: {error_report.error_id}")
                
            except Exception as e:
                logger.error(f"Failed to add message to dead letter queue: {e}")
    
    def _persist_message(self, dead_letter: Dict[str, Any]) -> None:
        """Persist failed message to disk."""
        try:
            filename = self.storage_dir / f"dlq_{int(time.time())}_{uuid.uuid4().hex[:8]}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(dead_letter, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist dead letter message: {e}")
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in dead letter queue."""
        with self._lock:
            return self.messages.copy()
    
    def clear(self) -> None:
        """Clear all messages from dead letter queue."""
        with self._lock:
            self.messages.clear()
