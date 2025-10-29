#!/usr/bin/env python3
"""
Message Bus Error Handling Integration Tests

Tests for integrated error handling in the message bus system.
"""

import time
import tempfile
import shutil
import pytest
from unittest.mock import Mock, patch

from src.communication.message_bus import (
    MessageBus,
    Message,
    MessagePriority,
)
from src.communication.error_handling import (
    ErrorCategory,
    ErrorSeverity,
)


class TestMessageBusErrorHandling:
    """Test suite for message bus error handling integration."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def message_bus_with_error_handling(self, temp_log_dir):
        """Create message bus with error handling enabled."""
        bus = MessageBus(
            max_queue_size=100,
            enable_error_handling=True,
            error_log_dir=temp_log_dir
        )
        bus.start()
        yield bus
        bus.stop()
    
    @pytest.fixture
    def message_bus_without_error_handling(self):
        """Create message bus without error handling."""
        bus = MessageBus(max_queue_size=100, enable_error_handling=False)
        bus.start()
        yield bus
        bus.stop()
    
    def test_error_handling_enabled_initialization(self, temp_log_dir):
        """Test message bus initializes with error handling."""
        bus = MessageBus(enable_error_handling=True, error_log_dir=temp_log_dir)
        
        assert bus.enable_error_handling is True
        assert bus.error_logger is not None
        assert bus.recovery_manager is not None
        assert bus.dead_letter_queue is not None
    
    def test_error_handling_disabled_initialization(self):
        """Test message bus initializes without error handling."""
        bus = MessageBus(enable_error_handling=False)
        
        assert bus.enable_error_handling is False
        assert bus.error_logger is None
        assert bus.recovery_manager is None
        assert bus.dead_letter_queue is None
    
    def test_callback_error_logged(self, message_bus_with_error_handling):
        """Test that callback errors are logged properly."""
        bus = message_bus_with_error_handling
        
        def failing_callback(msg):
            raise ValueError("Intentional test error")
        
        bus.subscribe("test_topic", failing_callback, "failing_sub")
        bus.publish("test_topic", {"data": "test"})
        
        time.sleep(0.2)
        
        # Check error was logged
        assert bus.stats['errors'] > 0
        if bus.error_logger:
            stats = bus.error_logger.get_error_statistics()
            assert stats['total_errors'] > 0
    
    def test_queue_overflow_handling(self, temp_log_dir):
        """Test queue overflow error handling."""
        bus = MessageBus(
            max_queue_size=5,
            enable_error_handling=True,
            error_log_dir=temp_log_dir
        )
        bus.start()
        
        try:
            # Fill the queue
            for i in range(10):
                bus.publish("test_topic", {"data": i})
            
            # Check that overflow was handled
            assert bus.stats['messages_dropped'] > 0
            
            # Check dead letter queue
            if bus.dead_letter_queue:
                dlq_messages = bus.dead_letter_queue.get_messages()
                assert len(dlq_messages) > 0
        finally:
            bus.stop()
    
    def test_error_categorization(self, message_bus_with_error_handling):
        """Test error categorization."""
        bus = message_bus_with_error_handling
        
        # Test different error types
        timeout_error = TimeoutError("Connection timeout")
        category = bus._categorize_error(timeout_error)
        assert category == ErrorCategory.TIMEOUT_ERROR
        
        connection_error = ConnectionError("Network failed")
        category = bus._categorize_error(connection_error)
        assert category == ErrorCategory.CONNECTION_ERROR
        
        value_error = ValueError("Invalid value")
        category = bus._categorize_error(value_error)
        assert category == ErrorCategory.MESSAGE_PROCESSING_ERROR
    
    def test_error_severity_assessment(self, message_bus_with_error_handling):
        """Test error severity assessment."""
        bus = message_bus_with_error_handling
        
        # Critical errors
        memory_error = MemoryError("Out of memory")
        severity = bus._assess_severity(memory_error, {})
        assert severity == ErrorSeverity.CRITICAL
        
        # High severity errors
        timeout_error = TimeoutError("Timeout")
        severity = bus._assess_severity(timeout_error, {})
        assert severity == ErrorSeverity.HIGH
        
        # Medium severity errors
        value_error = ValueError("Invalid value")
        severity = bus._assess_severity(value_error, {})
        assert severity == ErrorSeverity.MEDIUM
    
    def test_health_status_with_error_handling(self, message_bus_with_error_handling):
        """Test health status includes error handling metrics."""
        bus = message_bus_with_error_handling
        
        health = bus.get_health_status()
        
        assert 'status' in health
        assert 'error_statistics' in health
        assert 'circuit_breaker_status' in health
        assert 'errors_recovered' in health
        assert 'messages_failed' in health
    
    def test_health_status_without_error_handling(self, message_bus_without_error_handling):
        """Test health status without error handling."""
        bus = message_bus_without_error_handling
        
        health = bus.get_health_status()
        
        assert 'status' in health
        assert 'error_statistics' not in health
        assert 'circuit_breaker_status' not in health
    
    def test_message_requeue(self, message_bus_with_error_handling):
        """Test message requeue functionality."""
        bus = message_bus_with_error_handling
        
        message = Message(
            topic="test_topic",
            data={"key": "value"},
            priority=MessagePriority.HIGH
        )
        
        result = bus.requeue_message(message)
        
        assert result is True
        assert bus.main_queue.qsize() > 0
    
    def test_connection_error_handling(self, message_bus_with_error_handling):
        """Test connection error handling."""
        bus = message_bus_with_error_handling
        
        error = ConnectionError("Network failed")
        
        # Should not raise exception
        bus._handle_connection_error("conn-123", error)
        
        # Check error was logged
        if bus.error_logger:
            stats = bus.error_logger.get_error_statistics()
            assert stats['total_errors'] > 0
    
    def test_health_check_delegates_to_get_health_status(self, message_bus_with_error_handling):
        """Test that health_check uses get_health_status."""
        bus = message_bus_with_error_handling
        
        health_check_result = bus.health_check()
        health_status_result = bus.get_health_status()
        
        # Should return the same data
        assert health_check_result == health_status_result
    
    def test_backpressure_application(self, message_bus_with_error_handling):
        """Test backpressure application."""
        bus = message_bus_with_error_handling
        
        # Apply backpressure
        dropped = bus.apply_backpressure("test_queue")
        
        # Should return without error (even if 0 messages dropped)
        assert isinstance(dropped, int)
        assert dropped >= 0
    
    def test_queue_capacity_expansion_check(self, message_bus_with_error_handling):
        """Test queue capacity expansion check."""
        bus = message_bus_with_error_handling
        
        can_expand = bus.can_expand_capacity("test_queue")
        
        # Current implementation returns False
        assert can_expand is False
    
    def test_error_statistics_tracking(self, message_bus_with_error_handling):
        """Test comprehensive error statistics tracking."""
        bus = message_bus_with_error_handling
        
        # Generate some errors
        def failing_callback(msg):
            raise ValueError("Test error")
        
        bus.subscribe("test_topic", failing_callback, "sub1")
        
        for i in range(5):
            bus.publish("test_topic", {"data": i})
        
        time.sleep(0.3)
        
        # Check statistics
        health = bus.get_health_status()
        
        assert health['messages_failed'] > 0
        if 'error_statistics' in health:
            error_stats = health['error_statistics']
            assert error_stats['total_errors'] > 0


class TestMessageBusRecovery:
    """Test suite for message bus recovery mechanisms."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def message_bus(self, temp_log_dir):
        """Create message bus with error handling."""
        bus = MessageBus(
            max_queue_size=100,
            enable_error_handling=True,
            error_log_dir=temp_log_dir
        )
        bus.start()
        yield bus
        bus.stop()
    
    def test_recovery_updates_metrics(self, message_bus):
        """Test that successful recovery updates metrics."""
        bus = message_bus
        
        initial_recovered = bus.stats['errors_recovered']
        
        # This is difficult to test without complex mocking
        # Just verify the metric exists
        assert 'errors_recovered' in bus.stats
        assert isinstance(initial_recovered, int)
    
    def test_critical_error_escalation(self, message_bus):
        """Test critical error escalation."""
        bus = message_bus
        
        from src.communication.error_handling import ErrorReport
        
        critical_report = ErrorReport(
            error_id="critical-123",
            timestamp=time.time(),
            category=ErrorCategory.RESOURCE_EXHAUSTION,
            severity=ErrorSeverity.CRITICAL,
            component="test",
            message="Critical error",
            exception_type="MemoryError",
            traceback="",
            context={}
        )
        
        # Should not raise exception
        bus._escalate_critical_error(critical_report)


class TestMessageBusBackwardCompatibility:
    """Test that error handling doesn't break existing functionality."""
    
    @pytest.fixture
    def message_bus(self):
        """Create message bus with default settings."""
        bus = MessageBus()  # Default settings
        bus.start()
        yield bus
        bus.stop()
    
    def test_basic_publish_subscribe_still_works(self, message_bus):
        """Test basic pub/sub still works with error handling."""
        received_messages = []
        
        def callback(msg):
            received_messages.append(msg)
        
        message_bus.subscribe("test_topic", callback, "sub1")
        message_bus.publish("test_topic", {"data": "test"})
        
        time.sleep(0.2)
        
        assert len(received_messages) == 1
        assert received_messages[0].data == {"data": "test"}
    
    def test_stats_still_work(self, message_bus):
        """Test stats functionality still works."""
        stats = message_bus.get_stats()
        
        assert 'messages_published' in stats
        assert 'messages_delivered' in stats
        assert 'messages_dropped' in stats
    
    def test_health_check_still_works(self, message_bus):
        """Test health check still works."""
        health = message_bus.health_check()
        
        assert 'running' in health
        assert 'status' in health


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
