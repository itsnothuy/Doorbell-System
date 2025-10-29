#!/usr/bin/env python3
"""
Message Bus Error Handling Demonstration

This script demonstrates the comprehensive error handling capabilities
of the message bus system.
"""

import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.communication.message_bus import MessageBus, MessagePriority
from src.communication.error_handling import ErrorCategory, ErrorSeverity


def demonstrate_error_handling():
    """Demonstrate error handling features."""
    print("=" * 70)
    print("Message Bus Error Handling Demonstration")
    print("=" * 70)
    print()
    
    # Create message bus with error handling enabled
    print("1. Initializing message bus with error handling...")
    bus = MessageBus(
        max_queue_size=10,
        enable_error_handling=True,
        error_log_dir="/tmp/message_bus_demo"
    )
    bus.start()
    print("   ✓ Message bus started with error handling enabled")
    print()
    
    # Demonstrate normal operation
    print("2. Testing normal message publishing and delivery...")
    received_messages = []
    
    def normal_callback(msg):
        received_messages.append(msg)
        print(f"   ✓ Received message: {msg.data}")
    
    bus.subscribe("normal_topic", normal_callback, "normal_sub")
    bus.publish("normal_topic", {"message": "Hello World"})
    time.sleep(0.2)
    print(f"   ✓ Successfully delivered {len(received_messages)} message(s)")
    print()
    
    # Demonstrate error handling with failing callback
    print("3. Testing error handling with failing callback...")
    error_count = 0
    
    def failing_callback(msg):
        nonlocal error_count
        error_count += 1
        raise ValueError(f"Simulated error #{error_count}")
    
    bus.subscribe("error_topic", failing_callback, "failing_sub")
    bus.publish("error_topic", {"message": "This will cause an error"})
    time.sleep(0.2)
    
    print(f"   ✓ Error handled gracefully (errors logged: {bus.stats['errors']})")
    print()
    
    # Demonstrate queue overflow handling
    print("4. Testing queue overflow handling...")
    initial_dropped = bus.stats['messages_dropped']
    
    # Try to overflow the queue (max_queue_size=10)
    for i in range(15):
        bus.publish("overflow_topic", {"message": f"Message {i}"}, priority=MessagePriority.LOW)
    
    dropped_count = bus.stats['messages_dropped'] - initial_dropped
    print(f"   ✓ Queue overflow handled ({dropped_count} message(s) dropped)")
    
    if bus.dead_letter_queue:
        dlq_count = len(bus.dead_letter_queue.get_messages())
        print(f"   ✓ Dead letter queue contains {dlq_count} failed message(s)")
    print()
    
    # Show health status
    print("5. Checking system health status...")
    health = bus.get_health_status()
    
    print(f"   Status: {health['status'].upper()}")
    print(f"   Running: {health['running']}")
    print(f"   Messages Published: {bus.stats['messages_published']}")
    print(f"   Messages Delivered: {bus.stats['messages_delivered']}")
    print(f"   Messages Failed: {health['messages_failed']}")
    print(f"   Messages Dropped: {bus.stats['messages_dropped']}")
    print(f"   Errors Recovered: {health['errors_recovered']}")
    print(f"   Active Subscriptions: {health['active_subscriptions']}")
    print()
    
    # Show error statistics if available
    if 'error_statistics' in health:
        print("6. Error Statistics:")
        error_stats = health['error_statistics']
        print(f"   Total Errors: {error_stats['total_errors']}")
        print(f"   Recent Errors (1h): {error_stats['recent_errors_1h']}")
        print(f"   Error Rate (1h): {error_stats['error_rate_1h']:.2f} errors/min")
        
        if error_stats['categories']:
            print("   Error Categories:")
            for category, count in error_stats['categories'].items():
                print(f"     - {category}: {count}")
        print()
    
    # Show circuit breaker status
    if 'circuit_breaker_status' in health and health['circuit_breaker_status']:
        print("7. Circuit Breaker Status:")
        for breaker_id, state in health['circuit_breaker_status'].items():
            print(f"   {breaker_id}: {state}")
        print()
    
    # Demonstrate error categorization
    print("8. Error Categorization Examples:")
    test_errors = [
        (TimeoutError("Connection timeout"), "Timeout"),
        (ConnectionError("Network failed"), "Connection"),
        (ValueError("Invalid value"), "Processing"),
        (MemoryError("Out of memory"), "Resource Exhaustion")
    ]
    
    for error, error_type in test_errors:
        category = bus._categorize_error(error)
        severity = bus._assess_severity(error, {})
        print(f"   {error_type} Error -> Category: {category.value}, Severity: {severity.value}")
    print()
    
    # Cleanup
    print("9. Shutting down message bus...")
    bus.stop()
    print("   ✓ Message bus stopped cleanly")
    print()
    
    print("=" * 70)
    print("Demonstration Complete!")
    print("=" * 70)
    print()
    print(f"Error logs saved to: /tmp/message_bus_demo/")
    print()


if __name__ == "__main__":
    try:
        demonstrate_error_handling()
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user")
    except Exception as e:
        print(f"\n\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
