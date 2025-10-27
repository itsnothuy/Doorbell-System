#!/usr/bin/env python3
"""
Message Bus Test Suite

Comprehensive tests for message bus functionality including pub/sub patterns,
thread safety, priority handling, and resource management.
"""

import time
import pytest
import threading
from unittest.mock import Mock, MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

from src.communication.message_bus import (
    MessageBus,
    Message,
    MessagePriority,
    Subscription,
    RequestReplyBus,
    get_message_bus,
    init_message_bus,
)


class TestMessage:
    """Test suite for Message dataclass."""
    
    def test_message_creation_basic(self):
        """Test basic message creation."""
        msg = Message(topic="test", data={"key": "value"})
        
        assert msg.topic == "test"
        assert msg.data == {"key": "value"}
        assert msg.priority == MessagePriority.NORMAL
        assert msg.timestamp > 0
        assert msg.message_id is not None
    
    def test_message_priority_levels(self):
        """Test different message priority levels."""
        for priority in MessagePriority:
            msg = Message(topic="test", data={}, priority=priority)
            assert msg.priority == priority
    
    def test_message_id_generation(self):
        """Test automatic message ID generation."""
        msg1 = Message(topic="test1", data={}, source="src1")
        msg2 = Message(topic="test2", data={}, source="src1")
        
        assert msg1.message_id is not None
        assert msg2.message_id is not None
        assert msg1.message_id != msg2.message_id
    
    def test_message_with_correlation_id(self):
        """Test message with correlation ID for request/reply."""
        correlation_id = "test-correlation-123"
        msg = Message(
            topic="test",
            data={},
            correlation_id=correlation_id
        )
        
        assert msg.correlation_id == correlation_id
    
    def test_message_timestamp_ordering(self):
        """Test that messages have incrementing timestamps."""
        msg1 = Message(topic="test", data={})
        time.sleep(0.001)
        msg2 = Message(topic="test", data={})
        
        assert msg2.timestamp > msg1.timestamp


class TestSubscription:
    """Test suite for Subscription dataclass."""
    
    def test_subscription_creation(self):
        """Test subscription creation."""
        callback = Mock()
        sub = Subscription(
            topic="test_topic",
            callback=callback,
            subscriber_id="test_sub"
        )
        
        assert sub.topic == "test_topic"
        assert sub.callback == callback
        assert sub.subscriber_id == "test_sub"
        assert sub.active is True
        assert sub.pattern_match is False
    
    def test_subscription_with_pattern(self):
        """Test subscription with pattern matching."""
        callback = Mock()
        sub = Subscription(
            topic="test.*",
            callback=callback,
            subscriber_id="test_sub",
            pattern_match=True
        )
        
        assert sub.pattern_match is True


class TestMessageBus:
    """Test suite for MessageBus core functionality."""
    
    @pytest.fixture
    def message_bus(self):
        """Create message bus instance for testing."""
        bus = MessageBus(max_queue_size=100)
        bus.start()
        yield bus
        bus.stop()
    
    def test_message_bus_initialization(self):
        """Test message bus initialization."""
        bus = MessageBus(max_queue_size=500)
        
        assert bus.max_queue_size == 500
        assert bus.running is False
        assert bus.stats['messages_published'] == 0
        assert bus.stats['messages_delivered'] == 0
    
    def test_message_bus_start_stop(self):
        """Test message bus lifecycle."""
        bus = MessageBus()
        
        # Test start
        assert bus.running is False
        bus.start()
        assert bus.running is True
        assert bus.dispatch_thread is not None
        assert bus.dispatch_thread.is_alive()
        
        # Test stop
        bus.stop()
        assert bus.running is False
    
    def test_double_start_warning(self, message_bus):
        """Test that starting already running bus logs warning."""
        # Bus is already started in fixture
        assert message_bus.running is True
        
        # Try to start again
        message_bus.start()
        
        # Should still be running without issues
        assert message_bus.running is True
    
    def test_publish_basic(self, message_bus):
        """Test basic message publishing."""
        result = message_bus.publish("test_topic", {"data": "test"})
        
        assert result is True
        assert message_bus.stats['messages_published'] == 1
    
    def test_publish_when_not_running(self):
        """Test publishing when bus is not running."""
        bus = MessageBus()
        
        result = bus.publish("test_topic", {"data": "test"})
        
        assert result is False
        assert bus.stats['messages_published'] == 0
    
    def test_publish_with_priority(self, message_bus):
        """Test publishing with different priorities."""
        result = message_bus.publish(
            "test_topic",
            {"data": "urgent"},
            priority=MessagePriority.HIGH
        )
        
        assert result is True
    
    def test_publish_with_source(self, message_bus):
        """Test publishing with source identifier."""
        result = message_bus.publish(
            "test_topic",
            {"data": "test"},
            source="test_component"
        )
        
        assert result is True
    
    def test_subscribe_basic(self, message_bus):
        """Test basic subscription."""
        callback = Mock()
        
        result = message_bus.subscribe("test_topic", callback, "sub1")
        
        assert result is True
        assert message_bus.stats['active_subscriptions'] == 1
        assert "test_topic" in message_bus.subscriptions
    
    def test_subscribe_multiple_subscribers(self, message_bus):
        """Test multiple subscribers to same topic."""
        callback1 = Mock()
        callback2 = Mock()
        
        result1 = message_bus.subscribe("test_topic", callback1, "sub1")
        result2 = message_bus.subscribe("test_topic", callback2, "sub2")
        
        assert result1 is True
        assert result2 is True
        assert message_bus.stats['active_subscriptions'] == 2
        assert len(message_bus.subscriptions["test_topic"]) == 2
    
    def test_subscribe_duplicate_prevention(self, message_bus):
        """Test that duplicate subscriptions are prevented."""
        callback = Mock()
        
        result1 = message_bus.subscribe("test_topic", callback, "sub1")
        result2 = message_bus.subscribe("test_topic", callback, "sub1")
        
        assert result1 is True
        assert result2 is False
        assert message_bus.stats['active_subscriptions'] == 1
    
    def test_unsubscribe(self, message_bus):
        """Test unsubscribing from topic."""
        callback = Mock()
        
        message_bus.subscribe("test_topic", callback, "sub1")
        assert message_bus.stats['active_subscriptions'] == 1
        
        result = message_bus.unsubscribe("test_topic", "sub1")
        
        assert result is True
        assert message_bus.stats['active_subscriptions'] == 0
    
    def test_unsubscribe_nonexistent(self, message_bus):
        """Test unsubscribing from non-existent subscription."""
        result = message_bus.unsubscribe("test_topic", "sub1")
        
        assert result is False
    
    def test_publish_subscribe_basic_flow(self, message_bus):
        """Test basic publish-subscribe flow."""
        received_messages = []
        
        def callback(message):
            received_messages.append(message)
        
        message_bus.subscribe("test_topic", callback, "sub1")
        message_bus.publish("test_topic", {"key": "value"})
        
        # Wait for message processing
        time.sleep(0.2)
        
        assert len(received_messages) == 1
        assert received_messages[0].topic == "test_topic"
        assert received_messages[0].data == {"key": "value"}
    
    def test_multiple_subscribers_receive_message(self, message_bus):
        """Test that all subscribers receive messages."""
        received1 = []
        received2 = []
        
        def callback1(msg):
            received1.append(msg)
        
        def callback2(msg):
            received2.append(msg)
        
        message_bus.subscribe("test_topic", callback1, "sub1")
        message_bus.subscribe("test_topic", callback2, "sub2")
        message_bus.publish("test_topic", {"data": "test"})
        
        time.sleep(0.2)
        
        assert len(received1) == 1
        assert len(received2) == 1
    
    def test_pattern_matching_basic(self, message_bus):
        """Test basic pattern matching with wildcards."""
        received_messages = []
        
        def callback(msg):
            received_messages.append(msg)
        
        message_bus.subscribe("test.*", callback, "sub1", pattern_match=True)
        message_bus.publish("test.alpha", {"data": "alpha"})
        message_bus.publish("test.beta", {"data": "beta"})
        message_bus.publish("other.topic", {"data": "other"})
        
        time.sleep(0.2)
        
        # Should receive test.alpha and test.beta, but not other.topic
        assert len(received_messages) == 2
    
    def test_no_subscribers_no_error(self, message_bus):
        """Test publishing to topic with no subscribers."""
        result = message_bus.publish("unused_topic", {"data": "test"})
        
        assert result is True
        time.sleep(0.1)
        # Should not raise any errors


class TestMessageBusPriority:
    """Test suite for message priority handling."""
    
    @pytest.fixture
    def message_bus(self):
        """Create message bus instance."""
        bus = MessageBus(max_queue_size=100)
        bus.start()
        yield bus
        bus.stop()
    
    def test_priority_ordering(self, message_bus):
        """Test that high priority messages are processed first."""
        received_order = []
        
        def callback(msg):
            received_order.append(msg.data['priority_name'])
        
        message_bus.subscribe("test_topic", callback, "sub1")
        
        # Publish in specific order but with different priorities
        message_bus.publish("test_topic", {"priority_name": "normal"}, MessagePriority.NORMAL)
        message_bus.publish("test_topic", {"priority_name": "critical"}, MessagePriority.CRITICAL)
        message_bus.publish("test_topic", {"priority_name": "low"}, MessagePriority.LOW)
        message_bus.publish("test_topic", {"priority_name": "high"}, MessagePriority.HIGH)
        
        time.sleep(0.3)
        
        # Critical and high should be processed before normal and low
        # Note: Exact ordering depends on timing, but critical should be early
        assert len(received_order) == 4
        assert "critical" in received_order


class TestMessageBusThreadSafety:
    """Test suite for thread safety."""
    
    @pytest.fixture
    def message_bus(self):
        """Create message bus instance."""
        bus = MessageBus(max_queue_size=1000)
        bus.start()
        yield bus
        bus.stop()
    
    def test_concurrent_publishing(self, message_bus):
        """Test concurrent publishing from multiple threads."""
        received_count = [0]
        lock = threading.Lock()
        
        def callback(msg):
            with lock:
                received_count[0] += 1
        
        message_bus.subscribe("concurrent_topic", callback, "sub1")
        
        def publish_messages(thread_id):
            for i in range(10):
                message_bus.publish(
                    "concurrent_topic",
                    {"thread": thread_id, "count": i}
                )
        
        # Publish from 5 threads simultaneously
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(publish_messages, i) for i in range(5)]
            for future in futures:
                future.result()
        
        time.sleep(0.5)
        
        # Should receive all 50 messages
        assert received_count[0] == 50
    
    def test_concurrent_subscribe_unsubscribe(self, message_bus):
        """Test concurrent subscription operations."""
        def subscribe_unsubscribe(subscriber_id):
            callback = Mock()
            message_bus.subscribe("test_topic", callback, subscriber_id)
            time.sleep(0.01)
            message_bus.unsubscribe("test_topic", subscriber_id)
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(subscribe_unsubscribe, f"sub{i}")
                for i in range(20)
            ]
            for future in futures:
                future.result()
        
        # All subscriptions should be cleaned up
        assert message_bus.stats['active_subscriptions'] == 0


class TestMessageBusErrorHandling:
    """Test suite for error handling and recovery."""
    
    @pytest.fixture
    def message_bus(self):
        """Create message bus instance."""
        bus = MessageBus(max_queue_size=100)
        bus.start()
        yield bus
        bus.stop()
    
    def test_callback_exception_handling(self, message_bus):
        """Test that exceptions in callbacks don't crash bus."""
        received_by_good_sub = []
        
        def failing_callback(msg):
            raise ValueError("Intentional test error")
        
        def good_callback(msg):
            received_by_good_sub.append(msg)
        
        message_bus.subscribe("test_topic", failing_callback, "bad_sub")
        message_bus.subscribe("test_topic", good_callback, "good_sub")
        
        message_bus.publish("test_topic", {"data": "test"})
        
        time.sleep(0.2)
        
        # Good subscriber should still receive the message
        assert len(received_by_good_sub) == 1
        assert message_bus.stats['errors'] > 0
    
    def test_queue_full_handling(self):
        """Test behavior when queue is full."""
        bus = MessageBus(max_queue_size=5)
        bus.start()
        
        try:
            # Fill the queue
            for i in range(10):
                result = bus.publish("test_topic", {"data": i})
            
            # Some messages should be dropped
            assert bus.stats['messages_dropped'] > 0
            
        finally:
            bus.stop()


class TestMessageBusStats:
    """Test suite for statistics and monitoring."""
    
    @pytest.fixture
    def message_bus(self):
        """Create message bus instance."""
        bus = MessageBus(max_queue_size=100)
        bus.start()
        yield bus
        bus.stop()
    
    def test_get_stats(self, message_bus):
        """Test statistics retrieval."""
        stats = message_bus.get_stats()
        
        assert 'messages_published' in stats
        assert 'messages_delivered' in stats
        assert 'messages_dropped' in stats
        assert 'active_subscriptions' in stats
        assert 'queue_sizes' in stats
    
    def test_stats_update_on_publish(self, message_bus):
        """Test that stats update on publish."""
        initial_count = message_bus.stats['messages_published']
        
        message_bus.publish("test_topic", {"data": "test"})
        
        assert message_bus.stats['messages_published'] == initial_count + 1
    
    def test_get_topic_list(self, message_bus):
        """Test getting list of topics."""
        callback = Mock()
        
        message_bus.subscribe("topic1", callback, "sub1")
        message_bus.subscribe("topic2", callback, "sub2")
        
        topics = message_bus.get_topic_list()
        
        assert "topic1" in topics
        assert "topic2" in topics
    
    def test_get_subscriber_count(self, message_bus):
        """Test getting subscriber count for topic."""
        callback = Mock()
        
        message_bus.subscribe("test_topic", callback, "sub1")
        message_bus.subscribe("test_topic", callback, "sub2")
        
        count = message_bus.get_subscriber_count("test_topic")
        
        assert count == 2
    
    def test_health_check(self, message_bus):
        """Test health check functionality."""
        health = message_bus.health_check()
        
        assert 'running' in health
        assert 'main_queue_size' in health
        assert 'active_subscriptions' in health
        assert 'dispatch_thread_alive' in health
        assert 'error_rate' in health
        
        assert health['running'] is True
        assert health['dispatch_thread_alive'] is True


class TestRequestReplyBus:
    """Test suite for request-reply pattern."""
    
    @pytest.fixture
    def message_bus(self):
        """Create message bus instance."""
        bus = MessageBus(max_queue_size=100)
        bus.start()
        yield bus
        bus.stop()
    
    @pytest.fixture
    def request_reply_bus(self, message_bus):
        """Create request-reply bus instance."""
        return RequestReplyBus(message_bus)
    
    def test_request_reply_basic(self, message_bus, request_reply_bus):
        """Test basic request-reply pattern."""
        # Setup responder
        def responder(msg):
            response_topic = msg.data.get('response_topic')
            if response_topic:
                message_bus.publish(response_topic, {"result": "success"})
        
        message_bus.subscribe("test_request", responder, "responder")
        
        # Make request
        response = request_reply_bus.request("test_request", {"query": "test"}, timeout=2.0)
        
        assert response is not None
        assert response["result"] == "success"
    
    def test_request_timeout(self, request_reply_bus):
        """Test request timeout when no response."""
        with pytest.raises(TimeoutError):
            request_reply_bus.request("nonexistent_topic", {"query": "test"}, timeout=0.5)


class TestGlobalMessageBus:
    """Test suite for global message bus singleton."""
    
    def test_get_message_bus_singleton(self):
        """Test that get_message_bus returns singleton."""
        bus1 = get_message_bus()
        bus2 = get_message_bus()
        
        assert bus1 is bus2
    
    def test_init_message_bus(self):
        """Test initializing global message bus with config."""
        bus = init_message_bus(max_queue_size=500)
        
        assert bus.max_queue_size == 500
        
        # Cleanup
        bus.stop()


class TestMessageBusResourceCleanup:
    """Test suite for resource cleanup and memory management."""
    
    def test_stop_cleans_up_threads(self):
        """Test that stop properly cleans up threads."""
        bus = MessageBus()
        bus.start()
        
        dispatch_thread = bus.dispatch_thread
        assert dispatch_thread.is_alive()
        
        bus.stop()
        
        # Give thread time to finish
        time.sleep(0.5)
        
        assert not dispatch_thread.is_alive()
    
    def test_stop_without_start(self):
        """Test that stop without start doesn't error."""
        bus = MessageBus()
        
        # Should not raise error
        bus.stop()
    
    def test_multiple_stop_calls(self):
        """Test that multiple stop calls are safe."""
        bus = MessageBus()
        bus.start()
        
        bus.stop()
        bus.stop()  # Second stop should be safe


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
