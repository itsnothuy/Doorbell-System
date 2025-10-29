# Message Bus Error Handling System

## Overview

The Message Bus Error Handling System provides comprehensive error detection, classification, recovery, and monitoring capabilities for the communication infrastructure. It implements industry-standard patterns like Circuit Breaker and Dead Letter Queue to ensure system reliability and fault tolerance.

## Architecture

### Core Components

#### 1. Error Classification
- **ErrorCategory**: Categorizes errors by type (connection, processing, timeout, etc.)
- **ErrorSeverity**: Classifies error severity (LOW, MEDIUM, HIGH, CRITICAL)
- **ErrorReport**: Comprehensive error report with context, traceback, and recovery status

#### 2. Circuit Breaker
Prevents cascading failures by monitoring error rates and opening the circuit when thresholds are exceeded.

**States:**
- `CLOSED`: Normal operation, requests pass through
- `OPEN`: Circuit broken, requests fail immediately
- `HALF_OPEN`: Testing if service has recovered

**Configuration:**
```python
CircuitBreaker(
    failure_threshold=5,      # Open after 5 failures
    recovery_timeout=60.0     # Wait 60s before trying recovery
)
```

#### 3. Error Recovery Manager
Manages recovery strategies for different error categories with automatic retry and backoff.

**Recovery Strategies:**
- **Connection Errors**: Automatic reconnection
- **Processing Errors**: Retry with exponential backoff
- **Queue Overflow**: Backpressure and capacity management
- **Timeout Errors**: Adaptive timeout adjustment
- **Serialization Errors**: Fallback serialization methods

#### 4. Error Logger
Centralized error logging with metrics, history tracking, and file rotation.

**Features:**
- JSON-based error logs
- Automatic log file rotation
- Error history (last 1000 errors)
- Real-time metrics updates
- Severity-based standard logging

#### 5. Dead Letter Queue
Persists failed messages that exceed retry limits for manual inspection and recovery.

**Features:**
- Disk-based persistence
- Error context tracking
- Retry count tracking
- Message recovery support

## Usage

### Basic Usage

```python
from src.communication.message_bus import MessageBus

# Create message bus with error handling (enabled by default)
bus = MessageBus(
    max_queue_size=10000,
    enable_error_handling=True,  # Default
    error_log_dir="data/logs/message_bus"
)

bus.start()

# Use normally - error handling is automatic
bus.publish("topic", {"data": "value"})

# Check health status including error metrics
health = bus.get_health_status()
print(f"Status: {health['status']}")
print(f"Errors Recovered: {health['errors_recovered']}")

bus.stop()
```

### Disabling Error Handling

For scenarios where overhead must be minimized:

```python
bus = MessageBus(enable_error_handling=False)
```

### Health Monitoring

```python
health = bus.get_health_status()

# Overall status
print(f"Status: {health['status']}")  # healthy, degraded, or critical
print(f"Running: {health['running']}")

# Performance metrics
print(f"Messages Processed: {health['messages_processed']}")
print(f"Messages Failed: {health['messages_failed']}")
print(f"Success Rate: {health['success_rate']:.2%}")

# Error statistics (if error handling enabled)
if 'error_statistics' in health:
    stats = health['error_statistics']
    print(f"Total Errors: {stats['total_errors']}")
    print(f"Error Rate (1h): {stats['error_rate_1h']:.2f} errors/min")
    print(f"Recovery Success Rate: {stats['recovery_success_rate']:.2%}")

# Circuit breaker status
if 'circuit_breaker_status' in health:
    for breaker, state in health['circuit_breaker_status'].items():
        print(f"{breaker}: {state}")
```

### Error Handling Configuration

The error recovery manager supports configuration of retry behavior:

```python
from src.communication.error_handling import ErrorRecoveryManager, ErrorCategory

config = {}
manager = ErrorRecoveryManager(config)

# Retry configurations are pre-defined but can be customized
manager.retry_configs[ErrorCategory.CONNECTION_ERROR] = {
    'max_retries': 5,
    'backoff': 3.0  # seconds
}
```

### Dead Letter Queue Access

```python
# Get failed messages
if bus.dead_letter_queue:
    failed_messages = bus.dead_letter_queue.get_messages()
    
    for msg in failed_messages:
        print(f"Error ID: {msg['error_id']}")
        print(f"Message: {msg['message']}")
        print(f"Category: {msg['error_category']}")
        print(f"Retry Count: {msg['retry_count']}")
    
    # Clear processed messages
    bus.dead_letter_queue.clear()
```

## Error Categories

### CONNECTION_ERROR
Network or connection-related failures.

**Recovery Strategy:**
- Automatic reconnection attempts
- Circuit breaker protection

### MESSAGE_PROCESSING_ERROR
Errors during message processing or callback execution.

**Recovery Strategy:**
- Retry with exponential backoff (max 2 retries)
- Move to dead letter queue after exhausting retries

### SERIALIZATION_ERROR
Errors during message serialization/deserialization.

**Recovery Strategy:**
- Fallback to alternative serialization method
- Limited retry (max 1 attempt)

### QUEUE_OVERFLOW
Queue capacity exceeded.

**Recovery Strategy:**
- Apply backpressure (drop low-priority messages)
- Temporary capacity expansion if supported
- Move to dead letter queue

### TIMEOUT_ERROR
Operation timeout.

**Recovery Strategy:**
- Increase timeout adaptively
- Retry with longer timeout

### RESOURCE_EXHAUSTION
System resource limits reached (memory, CPU, etc.).

**Recovery Strategy:**
- Escalate as CRITICAL
- No automatic recovery

### AUTHENTICATION_ERROR
Authentication or authorization failures.

**Recovery Strategy:**
- No automatic recovery
- Manual intervention required

### CONFIGURATION_ERROR
Configuration or setup issues.

**Recovery Strategy:**
- No automatic recovery
- Manual intervention required

## Error Severity Levels

### LOW
Minor issues that don't affect system operation.
- Logged as INFO
- No immediate action required

### MEDIUM
Issues that affect individual operations but not overall system health.
- Logged as WARNING
- Recovery attempted
- Monitoring recommended

### HIGH
Significant issues affecting system reliability.
- Logged as ERROR
- Recovery attempted
- Immediate investigation recommended

### CRITICAL
System-threatening issues requiring immediate attention.
- Logged as CRITICAL
- Recovery attempted
- Escalated to administrators
- May trigger emergency procedures

## Files and Directories

### Error Logs
```
data/logs/message_bus/
├── message_bus_errors.json          # Current error log
├── message_bus_errors_*.json        # Rotated logs
├── error_metrics.json               # Real-time metrics
└── dead_letter/
    └── dlq_*.json                   # Failed messages
```

### Error Log Format
```json
{
  "error_id": "uuid",
  "timestamp": 1234567890.0,
  "category": "message_processing_error",
  "severity": "high",
  "component": "message_processor",
  "message": "Error description",
  "exception_type": "ValueError",
  "traceback": "Full traceback...",
  "context": {
    "message": "truncated message",
    "subscriber_id": "sub1"
  },
  "recovery_attempted": true,
  "recovery_successful": false,
  "retry_count": 2
}
```

### Metrics File Format
```json
{
  "total_errors": 42,
  "error_counts_by_category": {
    "message_processing_error": 30,
    "queue_overflow": 10,
    "timeout_error": 2
  },
  "last_updated": 1234567890.0,
  "recent_errors": [
    {
      "category": "message_processing_error",
      "severity": "medium",
      "timestamp": 1234567890.0,
      "component": "message_processor"
    }
  ]
}
```

## Performance Considerations

### Overhead
Error handling adds minimal overhead (measured on Raspberry Pi 4, 4GB RAM):
- Error classification: < 1ms (average case, no I/O)
- Circuit breaker check: < 0.1ms (in-memory state check)
- Error logging: Asynchronous, non-blocking (background thread)
- Health status: Cached, computed on-demand

### Resource Usage
- Memory: ~1KB per error report on average (includes message preview up to 1000 chars; tracebacks can add 1-5KB depending on call stack depth)
- Disk: ~10MB default max log size with automatic rotation
- CPU: < 1% additional CPU usage during normal operations (spikes to 2-5% during error bursts)

### Scaling
- Thread-safe implementation
- Lock-free read operations where possible
- Efficient data structures (deque for history, defaultdict for counts)

## Best Practices

### 1. Monitor Health Status Regularly
```python
import threading
import time

def check_health():
    """Check health status periodically."""
    while True:
        health = bus.get_health_status()
        if health['status'] != 'healthy':
            # Implement your alerting mechanism here
            print(f"WARNING: System health is {health['status']}")
        time.sleep(60)  # Check every minute

# Start monitoring in background thread
monitor_thread = threading.Thread(target=check_health, daemon=True)
monitor_thread.start()
```

### 2. Review Error Logs
- Check error logs daily
- Monitor error rate trends
- Investigate recurring errors

### 3. Handle Dead Letter Queue
- Review failed messages regularly
- Implement retry mechanisms for recoverable failures
- Archive or purge old messages

### 4. Circuit Breaker Monitoring
- Monitor circuit breaker states
- Adjust thresholds based on system behavior
- Alert on OPEN circuits

### 5. Custom Recovery Strategies
For specialized error handling, you can add custom recovery strategies:
```python
from src.communication.error_handling import ErrorRecoveryManager, ErrorCategory

manager = ErrorRecoveryManager()

def custom_recovery(error_report, context):
    """Custom recovery logic for specific scenarios."""
    # Implement your custom recovery logic
    logger.info(f"Custom recovery for {error_report.error_id}")
    return True

# Override existing strategy or add for a supported category
manager.recovery_strategies[ErrorCategory.TIMEOUT_ERROR] = custom_recovery
```

**Note:** The ErrorCategory enum is fixed. To handle new error types, categorize them under existing categories or extend the enum in `error_handling.py`.

## Testing

### Unit Tests
```bash
pytest tests/test_error_handling.py -v
```

### Integration Tests
```bash
pytest tests/test_message_bus_error_handling.py -v
```

### Demonstration
```bash
python examples/demonstrate_error_handling.py
```

## Troubleshooting

### High Error Rate
**Symptoms:** error_rate_1h > threshold (e.g., 10 errors/minute - adjust based on your system's normal error rate)

**Diagnosis:**
1. Check error categories in health status
2. Review recent errors in error logs
3. Check circuit breaker states

**Solutions:**
- Increase retry counts for transient errors
- Adjust circuit breaker thresholds
- Scale up resources if resource exhaustion

**Note:** The "10 errors/minute" threshold is an example. Determine your baseline error rate during normal operations and set alerts at 2-3x that rate.

### Circuit Breaker Stuck OPEN
**Symptoms:** Circuit remains OPEN despite service recovery

**Diagnosis:**
1. Check `recovery_timeout` setting
2. Verify underlying service is healthy
3. Review error patterns

**Solutions:**
- Increase `recovery_timeout`
- Fix underlying service issues
- Reset circuit breaker if appropriate

### Dead Letter Queue Growing
**Symptoms:** Dead letter queue size increasing

**Diagnosis:**
1. Review failed message patterns
2. Check error categories
3. Verify recovery strategies

**Solutions:**
- Fix root cause of failures
- Implement message replay mechanism
- Archive old messages

### Log File Growing Too Large
**Symptoms:** Disk space issues

**Diagnosis:**
1. Check log rotation settings
2. Review error frequency
3. Verify cleanup of old logs

**Solutions:**
- Reduce `max_log_size`
- Implement log archival
- Reduce error rate

## Future Enhancements

- Metrics export to Prometheus
- Integration with external alerting systems
- Advanced backpressure algorithms
- Distributed tracing support
- Machine learning-based error prediction
- Automatic recovery strategy tuning

## References

- Issue #18: Communication Message Bus Error Handling Implementation
- Frigate NVR architecture patterns
- Martin Fowler's Circuit Breaker pattern
- Enterprise Integration Patterns (Dead Letter Channel)
