#!/usr/bin/env python3
"""
Error Handling Test Suite

Comprehensive tests for error handling infrastructure including circuit breakers,
error recovery, and error logging.
"""

import time
import tempfile
import pytest
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.communication.error_handling import (
    ErrorLogger,
    ErrorRecoveryManager,
    ErrorReport,
    ErrorCategory,
    ErrorSeverity,
    CircuitBreaker,
    DeadLetterQueue,
)


class TestErrorReport:
    """Test suite for ErrorReport dataclass."""
    
    def test_error_report_creation(self):
        """Test basic error report creation."""
        report = ErrorReport(
            error_id="test-123",
            timestamp=time.time(),
            category=ErrorCategory.MESSAGE_PROCESSING_ERROR,
            severity=ErrorSeverity.HIGH,
            component="test_component",
            message="Test error",
            exception_type="ValueError",
            traceback="traceback here",
            context={"key": "value"}
        )
        
        assert report.error_id == "test-123"
        assert report.category == ErrorCategory.MESSAGE_PROCESSING_ERROR
        assert report.severity == ErrorSeverity.HIGH
        assert report.recovery_attempted is False
        assert report.recovery_successful is False
    
    def test_error_report_to_dict(self):
        """Test error report conversion to dictionary."""
        report = ErrorReport(
            error_id="test-123",
            timestamp=1234567890.0,
            category=ErrorCategory.CONNECTION_ERROR,
            severity=ErrorSeverity.CRITICAL,
            component="test",
            message="test",
            exception_type="Exception",
            traceback="",
            context={}
        )
        
        result = report.to_dict()
        
        assert isinstance(result, dict)
        assert result['error_id'] == "test-123"
        assert result['category'] == "connection_error"
        assert result['severity'] == "critical"


class TestCircuitBreaker:
    """Test suite for CircuitBreaker."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
        
        assert breaker.failure_threshold == 3
        assert breaker.recovery_timeout == 10.0
        assert breaker.state == 'CLOSED'
        assert breaker.failure_count == 0
    
    def test_circuit_breaker_successful_call(self):
        """Test successful function call through circuit breaker."""
        breaker = CircuitBreaker()
        
        def successful_func(x, y):
            return x + y
        
        result = breaker.call(successful_func, 2, 3)
        
        assert result == 5
        assert breaker.state == 'CLOSED'
        assert breaker.failure_count == 0
    
    def test_circuit_breaker_failure_recording(self):
        """Test that circuit breaker records failures."""
        breaker = CircuitBreaker(failure_threshold=3)
        
        def failing_func():
            raise ValueError("Test error")
        
        # Record failures
        for i in range(2):
            with pytest.raises(ValueError):
                breaker.call(failing_func)
        
        assert breaker.failure_count == 2
        assert breaker.state == 'CLOSED'
    
    def test_circuit_breaker_opens_after_threshold(self):
        """Test that circuit breaker opens after failure threshold."""
        breaker = CircuitBreaker(failure_threshold=3)
        
        def failing_func():
            raise ValueError("Test error")
        
        # Exceed failure threshold
        for i in range(3):
            with pytest.raises(ValueError):
                breaker.call(failing_func)
        
        assert breaker.state == 'OPEN'
        
        # Should raise RuntimeError when open
        with pytest.raises(RuntimeError, match="Circuit breaker is OPEN"):
            breaker.call(failing_func)
    
    def test_circuit_breaker_half_open_transition(self):
        """Test circuit breaker transition to HALF_OPEN state."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        
        def failing_func():
            raise ValueError("Test error")
        
        # Open the circuit
        for i in range(2):
            with pytest.raises(ValueError):
                breaker.call(failing_func)
        
        assert breaker.state == 'OPEN'
        
        # Wait for recovery timeout
        time.sleep(0.15)
        
        # Next call should transition to HALF_OPEN
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        
        # State should have been HALF_OPEN before the call failed
        assert breaker.state == 'OPEN'  # Back to OPEN after failure
    
    def test_circuit_breaker_closes_after_success(self):
        """Test circuit breaker closes after successful call in HALF_OPEN state."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        
        call_count = [0]
        
        def sometimes_failing_func():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ValueError("Test error")
            return "success"
        
        # Open the circuit
        for i in range(2):
            with pytest.raises(ValueError):
                breaker.call(sometimes_failing_func)
        
        assert breaker.state == 'OPEN'
        
        # Wait for recovery timeout
        time.sleep(0.15)
        
        # Successful call should close the circuit
        result = breaker.call(sometimes_failing_func)
        
        assert result == "success"
        assert breaker.state == 'CLOSED'
        assert breaker.failure_count == 0


class TestErrorRecoveryManager:
    """Test suite for ErrorRecoveryManager."""
    
    def test_recovery_manager_initialization(self):
        """Test recovery manager initialization."""
        manager = ErrorRecoveryManager()
        
        assert manager.recovery_strategies is not None
        assert len(manager.recovery_strategies) > 0
        assert ErrorCategory.CONNECTION_ERROR in manager.recovery_strategies
    
    def test_recovery_manager_no_strategy(self):
        """Test recovery when no strategy exists."""
        manager = ErrorRecoveryManager()
        
        report = ErrorReport(
            error_id="test",
            timestamp=time.time(),
            category=ErrorCategory.CONFIGURATION_ERROR,
            severity=ErrorSeverity.LOW,
            component="test",
            message="test",
            exception_type="Exception",
            traceback="",
            context={}
        )
        
        result = manager.attempt_recovery(report, {})
        
        assert result is False
        assert report.recovery_attempted is False
    
    def test_recovery_manager_connection_error(self):
        """Test connection error recovery."""
        manager = ErrorRecoveryManager()
        
        # Mock connection manager
        mock_connection_manager = Mock()
        mock_connection_manager.reconnect = Mock(return_value=True)
        
        report = ErrorReport(
            error_id="test",
            timestamp=time.time(),
            category=ErrorCategory.CONNECTION_ERROR,
            severity=ErrorSeverity.HIGH,
            component="test",
            message="Connection failed",
            exception_type="ConnectionError",
            traceback="",
            context={'connection_id': 'conn-123'}
        )
        
        result = manager.attempt_recovery(report, {'connection_manager': mock_connection_manager})
        
        assert result is True
        assert report.recovery_attempted is True
        assert report.recovery_successful is True
        mock_connection_manager.reconnect.assert_called_once_with('conn-123')
    
    def test_recovery_manager_processing_error_retry(self):
        """Test processing error recovery with retry."""
        manager = ErrorRecoveryManager()
        
        # Mock message queue
        mock_queue = Mock()
        mock_queue.requeue_message = Mock(return_value=True)
        
        report = ErrorReport(
            error_id="test",
            timestamp=time.time(),
            category=ErrorCategory.MESSAGE_PROCESSING_ERROR,
            severity=ErrorSeverity.MEDIUM,
            component="test",
            message="Processing failed",
            exception_type="ValueError",
            traceback="",
            context={'message': 'test_message'}
        )
        
        result = manager.attempt_recovery(report, {'message_queue': mock_queue})
        
        assert result is True
        assert report.retry_count == 1
    
    def test_recovery_manager_circuit_breaker_integration(self):
        """Test recovery manager integrates with circuit breakers."""
        manager = ErrorRecoveryManager()
        
        report = ErrorReport(
            error_id="test",
            timestamp=time.time(),
            category=ErrorCategory.TIMEOUT_ERROR,
            severity=ErrorSeverity.MEDIUM,
            component="test_component",
            message="Timeout",
            exception_type="TimeoutError",
            traceback="",
            context={}
        )
        
        # Attempt recovery multiple times
        for i in range(3):
            manager.attempt_recovery(report, {})
        
        # Check circuit breaker was created
        breaker_key = "test_component_timeout_error"
        assert breaker_key in manager.circuit_breakers


class TestErrorLogger:
    """Test suite for ErrorLogger."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_error_logger_initialization(self, temp_log_dir):
        """Test error logger initialization."""
        logger = ErrorLogger(temp_log_dir)
        
        assert logger.log_dir.exists()
        assert logger.max_log_size > 0
        assert len(logger.error_history) == 0
    
    def test_error_logger_log_error(self, temp_log_dir):
        """Test logging an error."""
        logger = ErrorLogger(temp_log_dir)
        
        report = ErrorReport(
            error_id="test-123",
            timestamp=time.time(),
            category=ErrorCategory.MESSAGE_PROCESSING_ERROR,
            severity=ErrorSeverity.HIGH,
            component="test",
            message="Test error",
            exception_type="ValueError",
            traceback="test traceback",
            context={"key": "value"}
        )
        
        logger.log_error(report)
        
        assert len(logger.error_history) == 1
        assert logger.error_counts[ErrorCategory.MESSAGE_PROCESSING_ERROR] == 1
    
    def test_error_logger_statistics(self, temp_log_dir):
        """Test error statistics generation."""
        logger = ErrorLogger(temp_log_dir)
        
        # Log multiple errors
        for i in range(5):
            report = ErrorReport(
                error_id=f"test-{i}",
                timestamp=time.time(),
                category=ErrorCategory.MESSAGE_PROCESSING_ERROR,
                severity=ErrorSeverity.HIGH if i < 2 else ErrorSeverity.LOW,
                component="test",
                message=f"Test error {i}",
                exception_type="ValueError",
                traceback="",
                context={}
            )
            logger.log_error(report)
        
        stats = logger.get_error_statistics()
        
        assert stats['total_errors'] == 5
        assert ErrorCategory.MESSAGE_PROCESSING_ERROR.value in stats['categories']
        assert stats['categories'][ErrorCategory.MESSAGE_PROCESSING_ERROR.value] == 5
    
    def test_error_logger_file_rotation(self, temp_log_dir):
        """Test log file rotation."""
        # Create logger with very small max size
        logger = ErrorLogger(temp_log_dir, max_log_size=100)
        
        # Log many errors to trigger rotation
        for i in range(10):
            report = ErrorReport(
                error_id=f"test-{i}",
                timestamp=time.time(),
                category=ErrorCategory.MESSAGE_PROCESSING_ERROR,
                severity=ErrorSeverity.HIGH,
                component="test",
                message="Test error with long message " * 10,
                exception_type="ValueError",
                traceback="long traceback " * 10,
                context={"data": "x" * 100}
            )
            logger.log_error(report)
        
        # Check that files exist
        log_files = list(Path(temp_log_dir).glob("message_bus_errors*.json"))
        assert len(log_files) >= 1


class TestDeadLetterQueue:
    """Test suite for DeadLetterQueue."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary directory for dead letter queue."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_dead_letter_queue_initialization(self, temp_storage_dir):
        """Test dead letter queue initialization."""
        dlq = DeadLetterQueue(temp_storage_dir)
        
        assert dlq.storage_dir.exists()
        assert len(dlq.messages) == 0
    
    def test_dead_letter_queue_add_message(self, temp_storage_dir):
        """Test adding message to dead letter queue."""
        dlq = DeadLetterQueue(temp_storage_dir)
        
        report = ErrorReport(
            error_id="test-123",
            timestamp=time.time(),
            category=ErrorCategory.MESSAGE_PROCESSING_ERROR,
            severity=ErrorSeverity.HIGH,
            component="test",
            message="Test error",
            exception_type="ValueError",
            traceback="",
            context={}
        )
        
        dlq.add_message("test message", report)
        
        assert len(dlq.messages) == 1
        assert dlq.messages[0]['error_id'] == "test-123"
    
    def test_dead_letter_queue_persistence(self, temp_storage_dir):
        """Test dead letter queue persists messages to disk."""
        dlq = DeadLetterQueue(temp_storage_dir)
        
        report = ErrorReport(
            error_id="test-123",
            timestamp=time.time(),
            category=ErrorCategory.MESSAGE_PROCESSING_ERROR,
            severity=ErrorSeverity.HIGH,
            component="test",
            message="Test error",
            exception_type="ValueError",
            traceback="",
            context={}
        )
        
        dlq.add_message("test message", report)
        
        # Check file was created
        files = list(Path(temp_storage_dir).glob("dlq_*.json"))
        assert len(files) >= 1
    
    def test_dead_letter_queue_get_messages(self, temp_storage_dir):
        """Test retrieving messages from dead letter queue."""
        dlq = DeadLetterQueue(temp_storage_dir)
        
        # Add multiple messages
        for i in range(3):
            report = ErrorReport(
                error_id=f"test-{i}",
                timestamp=time.time(),
                category=ErrorCategory.MESSAGE_PROCESSING_ERROR,
                severity=ErrorSeverity.HIGH,
                component="test",
                message="Test error",
                exception_type="ValueError",
                traceback="",
                context={}
            )
            dlq.add_message(f"test message {i}", report)
        
        messages = dlq.get_messages()
        
        assert len(messages) == 3
    
    def test_dead_letter_queue_clear(self, temp_storage_dir):
        """Test clearing dead letter queue."""
        dlq = DeadLetterQueue(temp_storage_dir)
        
        report = ErrorReport(
            error_id="test-123",
            timestamp=time.time(),
            category=ErrorCategory.MESSAGE_PROCESSING_ERROR,
            severity=ErrorSeverity.HIGH,
            component="test",
            message="Test error",
            exception_type="ValueError",
            traceback="",
            context={}
        )
        
        dlq.add_message("test message", report)
        assert len(dlq.messages) == 1
        
        dlq.clear()
        assert len(dlq.messages) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
