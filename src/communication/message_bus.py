#!/usr/bin/env python3
"""
Message Bus - ZeroMQ-inspired High-Performance IPC

This module provides high-performance inter-process communication
similar to ZeroMQ but simplified for our use case.
"""

import time
import queue
import threading
import logging
from typing import Dict, Any, Callable, Optional, List, Set
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority levels for queue ordering."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Message:
    """Represents a message in the message bus."""
    topic: str
    data: Any
    timestamp: float = field(default_factory=time.time)
    priority: MessagePriority = MessagePriority.NORMAL
    source: Optional[str] = None
    message_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        if self.message_id is None:
            self.message_id = f"{self.source}_{int(self.timestamp * 1000000)}"


@dataclass
class Subscription:
    """Represents a subscription to a topic."""
    topic: str
    callback: Callable[[Message], None]
    subscriber_id: str
    pattern_match: bool = False
    active: bool = True


class MessageBus:
    """
    High-performance message bus for inter-component communication.
    
    Inspired by ZeroMQ patterns but simplified for our specific use case.
    Supports publish/subscribe, request/reply, and push/pull patterns.
    """
    
    def __init__(self, max_queue_size: int = 10000):
        """Initialize the message bus."""
        self.max_queue_size = max_queue_size
        self.running = False
        
        # Message queues by topic
        self.queues: Dict[str, queue.PriorityQueue] = {}
        self.queue_locks: Dict[str, threading.Lock] = {}
        
        # Subscription management
        self.subscriptions: Dict[str, List[Subscription]] = {}
        self.subscription_lock = threading.Lock()
        
        # Worker threads for message processing
        self.worker_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="MessageBus")
        self.dispatch_thread: Optional[threading.Thread] = None
        
        # Statistics and monitoring
        self.stats = {
            'messages_published': 0,
            'messages_delivered': 0,
            'messages_dropped': 0,
            'active_subscriptions': 0,
            'queue_sizes': {},
            'errors': 0
        }
        self.stats_lock = threading.Lock()
        
        # Internal queue for all messages
        self.main_queue = queue.PriorityQueue(maxsize=max_queue_size)
        
        logger.info("Message bus initialized")
    
    def start(self) -> None:
        """Start the message bus and begin processing messages."""
        if self.running:
            logger.warning("Message bus already running")
            return
        
        self.running = True
        
        # Start main dispatch thread
        self.dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            name="MessageBus-Dispatcher",
            daemon=False
        )
        self.dispatch_thread.start()
        
        logger.info("Message bus started")
    
    def stop(self) -> None:
        """Stop the message bus and cleanup resources."""
        if not self.running:
            return
        
        logger.info("Stopping message bus...")
        self.running = False
        
        # Signal shutdown with a special message
        try:
            self.main_queue.put_nowait((MessagePriority.CRITICAL.value, time.time(), None))
        except queue.Full:
            pass
        
        # Wait for dispatch thread to finish
        if self.dispatch_thread and self.dispatch_thread.is_alive():
            self.dispatch_thread.join(timeout=5.0)
        
        # Shutdown worker pool
        self.worker_executor.shutdown(wait=True)
        
        logger.info("Message bus stopped")
    
    def publish(self, topic: str, data: Any, 
                priority: MessagePriority = MessagePriority.NORMAL,
                source: Optional[str] = None) -> bool:
        """
        Publish a message to a topic.
        
        Args:
            topic: Topic name for routing
            data: Message payload
            priority: Message priority for ordering
            source: Source component identifier
            
        Returns:
            bool: True if message was queued successfully
        """
        if not self.running:
            logger.warning("Cannot publish - message bus not running")
            return False
        
        try:
            message = Message(
                topic=topic,
                data=data,
                priority=priority,
                source=source
            )
            
            # Add to main queue with priority
            queue_item = (priority.value, message.timestamp, message)
            self.main_queue.put_nowait(queue_item)
            
            # Update statistics
            with self.stats_lock:
                self.stats['messages_published'] += 1
            
            logger.debug(f"Published message to topic '{topic}' from {source}")
            return True
            
        except queue.Full:
            logger.warning(f"Message queue full - dropping message for topic '{topic}'")
            with self.stats_lock:
                self.stats['messages_dropped'] += 1
            return False
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            with self.stats_lock:
                self.stats['errors'] += 1
            return False
    
    def subscribe(self, topic: str, callback: Callable[[Message], None],
                  subscriber_id: str, pattern_match: bool = False) -> bool:
        """
        Subscribe to a topic with a callback function.
        
        Args:
            topic: Topic to subscribe to (can be pattern if pattern_match=True)
            callback: Function to call when message received
            subscriber_id: Unique identifier for this subscription
            pattern_match: Whether topic is a pattern (supports wildcards)
            
        Returns:
            bool: True if subscription was successful
        """
        try:
            subscription = Subscription(
                topic=topic,
                callback=callback,
                subscriber_id=subscriber_id,
                pattern_match=pattern_match
            )
            
            with self.subscription_lock:
                if topic not in self.subscriptions:
                    self.subscriptions[topic] = []
                
                # Check for duplicate subscriptions
                existing = [s for s in self.subscriptions[topic] 
                           if s.subscriber_id == subscriber_id]
                if existing:
                    logger.warning(f"Subscription already exists for {subscriber_id} on {topic}")
                    return False
                
                self.subscriptions[topic].append(subscription)
                
                with self.stats_lock:
                    self.stats['active_subscriptions'] += 1
            
            logger.info(f"Subscribed {subscriber_id} to topic '{topic}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe: {e}")
            return False
    
    def unsubscribe(self, topic: str, subscriber_id: str) -> bool:
        """
        Unsubscribe from a topic.
        
        Args:
            topic: Topic to unsubscribe from
            subscriber_id: Subscriber identifier
            
        Returns:
            bool: True if unsubscription was successful
        """
        try:
            with self.subscription_lock:
                if topic not in self.subscriptions:
                    return False
                
                original_count = len(self.subscriptions[topic])
                self.subscriptions[topic] = [
                    s for s in self.subscriptions[topic] 
                    if s.subscriber_id != subscriber_id
                ]
                
                removed_count = original_count - len(self.subscriptions[topic])
                if removed_count > 0:
                    with self.stats_lock:
                        self.stats['active_subscriptions'] -= removed_count
                    logger.info(f"Unsubscribed {subscriber_id} from topic '{topic}'")
                    return True
                
            return False
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe: {e}")
            return False
    
    def _dispatch_loop(self) -> None:
        """Main message dispatch loop running in dedicated thread."""
        logger.info("Message dispatch loop started")
        
        while self.running:
            try:
                # Get next message from queue (blocking with timeout)
                try:
                    priority, timestamp, message = self.main_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Check for shutdown signal
                if message is None:
                    break
                
                # Dispatch message to subscribers
                self._dispatch_message(message)
                
                # Mark queue task as done
                self.main_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in dispatch loop: {e}")
                with self.stats_lock:
                    self.stats['errors'] += 1
                time.sleep(0.1)  # Brief pause on error
        
        logger.info("Message dispatch loop stopped")
    
    def _dispatch_message(self, message: Message) -> None:
        """Dispatch a message to all matching subscribers."""
        try:
            # Find all matching subscriptions
            matching_subscriptions = []
            
            with self.subscription_lock:
                # Exact topic matches
                if message.topic in self.subscriptions:
                    matching_subscriptions.extend(
                        s for s in self.subscriptions[message.topic] if s.active
                    )
                
                # Pattern matches (simple wildcard support)
                for topic_pattern, subs in self.subscriptions.items():
                    for sub in subs:
                        if sub.pattern_match and sub.active and self._topic_matches(message.topic, topic_pattern):
                            matching_subscriptions.append(sub)
            
            if not matching_subscriptions:
                logger.debug(f"No subscribers for topic '{message.topic}'")
                return
            
            # Dispatch to all matching subscribers in parallel
            for subscription in matching_subscriptions:
                self.worker_executor.submit(self._deliver_message, subscription, message)
            
            logger.debug(f"Dispatched message to {len(matching_subscriptions)} subscribers")
            
        except Exception as e:
            logger.error(f"Failed to dispatch message: {e}")
            with self.stats_lock:
                self.stats['errors'] += 1
    
    def _deliver_message(self, subscription: Subscription, message: Message) -> None:
        """Deliver a message to a specific subscriber."""
        try:
            subscription.callback(message)
            
            with self.stats_lock:
                self.stats['messages_delivered'] += 1
                
        except Exception as e:
            logger.error(f"Error delivering message to {subscription.subscriber_id}: {e}")
            with self.stats_lock:
                self.stats['errors'] += 1
            
            # Consider deactivating problematic subscribers
            if self.stats['errors'] > 100:  # Threshold for problematic subscriber
                logger.warning(f"Deactivating problematic subscriber: {subscription.subscriber_id}")
                subscription.active = False
    
    def _topic_matches(self, message_topic: str, pattern: str) -> bool:
        """Check if a message topic matches a subscription pattern."""
        # Simple wildcard matching (* and ?)
        import fnmatch
        return fnmatch.fnmatch(message_topic, pattern)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        with self.stats_lock:
            stats = self.stats.copy()
        
        # Add current queue sizes
        stats['queue_sizes']['main_queue'] = self.main_queue.qsize()
        stats['queue_sizes']['total_subscriptions'] = sum(
            len(subs) for subs in self.subscriptions.values()
        )
        
        return stats
    
    def get_topic_list(self) -> List[str]:
        """Get list of all topics with active subscriptions."""
        with self.subscription_lock:
            return list(self.subscriptions.keys())
    
    def get_subscriber_count(self, topic: str) -> int:
        """Get number of active subscribers for a topic."""
        with self.subscription_lock:
            if topic not in self.subscriptions:
                return 0
            return len([s for s in self.subscriptions[topic] if s.active])
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        return {
            'running': self.running,
            'main_queue_size': self.main_queue.qsize(),
            'main_queue_full': self.main_queue.qsize() >= self.max_queue_size * 0.9,
            'active_subscriptions': sum(
                len([s for s in subs if s.active]) 
                for subs in self.subscriptions.values()
            ),
            'dispatch_thread_alive': self.dispatch_thread.is_alive() if self.dispatch_thread else False,
            'error_rate': self.stats['errors'] / max(1, self.stats['messages_published']),
            'stats': self.get_stats()
        }


# Convenience functions for common messaging patterns

class RequestReplyBus:
    """Request-Reply pattern implementation using the message bus."""
    
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.pending_requests: Dict[str, threading.Event] = {}
        self.responses: Dict[str, Any] = {}
        self.timeout_default = 10.0  # seconds
    
    def request(self, topic: str, data: Any, timeout: Optional[float] = None) -> Any:
        """Send a request and wait for response."""
        correlation_id = f"req_{int(time.time() * 1000000)}"
        response_topic = f"{topic}_response_{correlation_id}"
        timeout = timeout or self.timeout_default
        
        # Setup response handler
        response_event = threading.Event()
        self.pending_requests[correlation_id] = response_event
        
        def response_handler(message: Message):
            self.responses[correlation_id] = message.data
            response_event.set()
        
        # Subscribe to response topic
        self.message_bus.subscribe(response_topic, response_handler, f"requester_{correlation_id}")
        
        try:
            # Send request
            request_message = Message(
                topic=topic,
                data=data,
                correlation_id=correlation_id,
                priority=MessagePriority.HIGH
            )
            request_message.data['response_topic'] = response_topic
            
            self.message_bus.publish(topic, request_message.data, MessagePriority.HIGH)
            
            # Wait for response
            if response_event.wait(timeout):
                return self.responses.pop(correlation_id, None)
            else:
                raise TimeoutError(f"Request timeout after {timeout}s")
                
        finally:
            # Cleanup
            self.message_bus.unsubscribe(response_topic, f"requester_{correlation_id}")
            self.pending_requests.pop(correlation_id, None)
            self.responses.pop(correlation_id, None)


# Global message bus instance (singleton pattern)
_global_message_bus: Optional[MessageBus] = None
_bus_lock = threading.Lock()


def get_message_bus() -> MessageBus:
    """Get or create the global message bus instance."""
    global _global_message_bus
    
    with _bus_lock:
        if _global_message_bus is None:
            _global_message_bus = MessageBus()
        return _global_message_bus


def init_message_bus(max_queue_size: int = 10000) -> MessageBus:
    """Initialize the global message bus with specific configuration."""
    global _global_message_bus
    
    with _bus_lock:
        if _global_message_bus is not None:
            _global_message_bus.stop()
        
        _global_message_bus = MessageBus(max_queue_size)
        return _global_message_bus