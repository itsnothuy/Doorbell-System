#!/usr/bin/env python3
"""
Final Validation Script for Error Handling Implementation

This script performs comprehensive validation of the error handling system.
Run this before merging to ensure everything works correctly.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.communication.message_bus import MessageBus, MessagePriority
from src.communication.error_handling import (
    ErrorCategory,
    ErrorSeverity,
    CircuitBreaker,
    ErrorRecoveryManager,
    ErrorLogger,
    DeadLetterQueue,
)


def test_backward_compatibility():
    """Test that existing functionality still works."""
    print("Testing backward compatibility...")
    
    bus = MessageBus()
    bus.start()
    
    received = []
    def callback(msg):
        received.append(msg)
    
    bus.subscribe('test', callback, 'sub1')
    bus.publish('test', {'data': 'test'})
    time.sleep(0.2)
    
    bus.stop()
    
    assert len(received) == 1, "Message not received"
    assert received[0].data == {'data': 'test'}, "Wrong data"
    
    print("  ✓ Backward compatibility OK")


def test_error_handling_enabled():
    """Test error handling when enabled."""
    print("Testing error handling enabled...")
    
    bus = MessageBus(enable_error_handling=True)
    bus.start()
    
    # Test components initialized
    assert bus.error_logger is not None
    assert bus.recovery_manager is not None
    assert bus.dead_letter_queue is not None
    
    # Test error categorization
    error = ValueError("test")
    category = bus._categorize_error(error)
    assert category == ErrorCategory.MESSAGE_PROCESSING_ERROR
    
    # Test severity assessment
    severity = bus._assess_severity(error, {})
    assert severity == ErrorSeverity.MEDIUM
    
    # Test health status
    health = bus.get_health_status()
    assert 'status' in health
    assert 'error_statistics' in health
    
    bus.stop()
    
    print("  ✓ Error handling enabled OK")


def test_error_handling_disabled():
    """Test error handling when disabled."""
    print("Testing error handling disabled...")
    
    bus = MessageBus(enable_error_handling=False)
    bus.start()
    
    # Test components not initialized
    assert bus.error_logger is None
    assert bus.recovery_manager is None
    assert bus.dead_letter_queue is None
    
    # Test health status without error handling
    health = bus.get_health_status()
    assert 'status' in health
    assert 'error_statistics' not in health
    
    bus.stop()
    
    print("  ✓ Error handling disabled OK")


def test_circuit_breaker():
    """Test circuit breaker functionality."""
    print("Testing circuit breaker...")
    
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
    
    # Test successful call
    result = breaker.call(lambda: "success")
    assert result == "success"
    assert breaker.state == 'CLOSED'
    
    # Test failures open circuit
    for _ in range(2):
        try:
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        except ValueError:
            pass
    
    assert breaker.state == 'OPEN'
    
    # Test circuit stays open
    try:
        breaker.call(lambda: "test")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Circuit breaker is OPEN" in str(e)
    
    # Test recovery after timeout
    time.sleep(0.15)
    result = breaker.call(lambda: "recovered")
    assert result == "recovered"
    assert breaker.state == 'CLOSED'
    
    print("  ✓ Circuit breaker OK")


def test_error_logger():
    """Test error logger functionality."""
    print("Testing error logger...")
    
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    from src.communication.error_handling import ErrorReport
    
    logger = ErrorLogger(temp_dir)
    
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
    
    # Check error was logged
    assert len(logger.error_history) == 1
    stats = logger.get_error_statistics()
    assert stats['total_errors'] == 1
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    print("  ✓ Error logger OK")


def test_dead_letter_queue():
    """Test dead letter queue functionality."""
    print("Testing dead letter queue...")
    
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    from src.communication.error_handling import ErrorReport
    
    dlq = DeadLetterQueue(temp_dir)
    
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
    
    # Check message was added
    messages = dlq.get_messages()
    assert len(messages) == 1
    assert messages[0]['error_id'] == "test-123"
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    print("  ✓ Dead letter queue OK")


def test_error_recovery():
    """Test error recovery manager."""
    print("Testing error recovery manager...")
    
    from src.communication.error_handling import ErrorReport
    
    manager = ErrorRecoveryManager()
    
    # Test connection error recovery (will fail without real connection manager)
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
    
    # Should not crash even without connection manager
    result = manager.attempt_recovery(report, {})
    assert isinstance(result, bool)
    
    print("  ✓ Error recovery manager OK")


def run_all_validations():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("Final Validation for Error Handling Implementation")
    print("=" * 70 + "\n")
    
    tests = [
        ("Backward Compatibility", test_backward_compatibility),
        ("Error Handling Enabled", test_error_handling_enabled),
        ("Error Handling Disabled", test_error_handling_disabled),
        ("Circuit Breaker", test_circuit_breaker),
        ("Error Logger", test_error_logger),
        ("Dead Letter Queue", test_dead_letter_queue),
        ("Error Recovery", test_error_recovery),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"Validation Results: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")
    
    if failed == 0:
        print("✅ ALL VALIDATIONS PASSED!")
        print("✅ Implementation is ready for merge!")
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print("❌ Please fix the issues before merging.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_validations())
