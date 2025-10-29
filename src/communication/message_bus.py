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
import traceback
import uuid
from typing import Dict, Any, Callable, Optional, List, Set
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum

from src.communication.error_handling import (
    ErrorLogger,
    ErrorRecoveryManager,
    ErrorReport,
    ErrorCategory,
    ErrorSeverity,
    DeadLetterQueue,
)

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
    
    def __init__(self, max_queue_size: int = 10000, enable_error_handling: bool = True,
                 error_log_dir: str = "data/logs/message_bus"):
        """
        Initialize the message bus.
        
        Args:
            max_queue_size: Maximum queue size
            enable_error_handling: Enable comprehensive error handling
            error_log_dir: Directory for error logs
        """
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
            'errors': 0,
            'messages_failed': 0,
            'errors_recovered': 0,
            'start_time': time.time()
        }
        self.stats_lock = threading.Lock()
        
        # Internal queue for all messages
        self.main_queue = queue.PriorityQueue(maxsize=max_queue_size)
        
        # Error handling infrastructure
        self.enable_error_handling = enable_error_handling
        if enable_error_handling:
            self.error_logger = ErrorLogger(error_log_dir)
            self.recovery_manager = ErrorRecoveryManager()
            self.dead_letter_queue = DeadLetterQueue(f"{error_log_dir}/dead_letter")
        else:
            self.error_logger = None
            self.recovery_manager = None
            self.dead_letter_queue = None
        
        logger.info("Message bus initialized with error handling enabled" if enable_error_handling else "Message bus initialized")
    
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
            
            # Handle queue overflow with error handling system
            if self.enable_error_handling:
                self._handle_queue_overflow('main_queue', message)
            
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
            
            # Use error handling system if enabled
            if self.enable_error_handling:
                self._handle_message_processing_error(
                    message, 
                    e, 
                    {
                        'subscriber_id': subscription.subscriber_id,
                        'topic': message.topic,
                        'queue_name': 'main_queue'
                    }
                )
            
            # Consider deactivating problematic subscribers
            if self.stats['errors'] > 100:  # Threshold for problematic subscriber
                logger.warning(f"Deactivating problematic subscriber: {subscription.subscriber_id}")
                subscription.active = False
    
    def _topic_matches(self, message_topic: str, pattern: str) -> bool:
        """Check if a message topic matches a subscription pattern."""
        # Simple wildcard matching (* and ?)
        import fnmatch
        return fnmatch.fnmatch(message_topic, pattern)
    
    def _handle_message_processing_error(self, message: Message, error: Exception, 
                                        context: Dict[str, Any]) -> None:
        """
        Handle errors during message processing with comprehensive recovery.
        
        Args:
            message: The message that failed processing
            error: The exception that occurred
            context: Additional context about the error
        """
        if not self.enable_error_handling:
            return
        
        try:
            # Create error report
            error_report = ErrorReport(
                error_id=str(uuid.uuid4()),
                timestamp=time.time(),
                category=self._categorize_error(error),
                severity=self._assess_severity(error, context),
                component='message_processor',
                message=str(error),
                exception_type=type(error).__name__,
                traceback=traceback.format_exc(),
                context={
                    'message': str(message)[:1000],  # Truncate large messages
                    'message_type': type(message).__name__,
                    'message_topic': message.topic,
                    'queue_name': context.get('queue_name'),
                    'subscriber_id': context.get('subscriber_id'),
                    'subscriber_count': len(self.subscriptions.get(message.topic, []))
                }
            )
            
            # Log error
            self.error_logger.log_error(error_report)
            
            # Update metrics
            with self.stats_lock:
                self.stats['messages_failed'] += 1
            
            # Attempt recovery
            recovery_context = {
                'message_queue': self,
                'connection_manager': self,
                'dead_letter_queue': self.dead_letter_queue,
                'queue_manager': self,
                'timeout_manager': self,
                'serializer': None
            }
            
            recovery_successful = self.recovery_manager.attempt_recovery(error_report, recovery_context)
            
            if recovery_successful:
                with self.stats_lock:
                    self.stats['errors_recovered'] += 1
                logger.info(f"Successfully recovered from message processing error: {error_report.error_id}")
            else:
                logger.error(f"Failed to recover from message processing error: {error_report.error_id}")
                
                # Escalate critical errors
                if error_report.severity == ErrorSeverity.CRITICAL:
                    self._escalate_critical_error(error_report)
            
        except Exception as recovery_error:
            # Fallback error handling to prevent infinite loops
            logger.critical(f"Error in error handling: {recovery_error}")
    
    def _handle_connection_error(self, connection_id: str, error: Exception) -> None:
        """
        Handle connection-related errors with automatic recovery.
        
        Args:
            connection_id: Identifier of the connection that failed
            error: The exception that occurred
        """
        if not self.enable_error_handling:
            return
        
        try:
            error_report = ErrorReport(
                error_id=str(uuid.uuid4()),
                timestamp=time.time(),
                category=ErrorCategory.CONNECTION_ERROR,
                severity=self._assess_connection_severity(connection_id, error),
                component='connection_manager',
                message=str(error),
                exception_type=type(error).__name__,
                traceback=traceback.format_exc(),
                context={
                    'connection_id': connection_id,
                    'active_connections': 0,  # Would be len(self.connections) if we had connections
                    'error_details': str(error)
                }
            )
            
            self.error_logger.log_error(error_report)
            
            logger.info(f"Connection error logged for {connection_id}: {error_report.error_id}")
            
        except Exception as e:
            logger.critical(f"Critical error in connection error handling: {e}")
    
    def _handle_queue_overflow(self, queue_name: str, message: Message) -> None:
        """
        Handle queue capacity overflow with backpressure and prioritization.
        
        Args:
            queue_name: Name of the queue that overflowed
            message: Message that couldn't be queued
        """
        if not self.enable_error_handling:
            return
        
        try:
            error_report = ErrorReport(
                error_id=str(uuid.uuid4()),
                timestamp=time.time(),
                category=ErrorCategory.QUEUE_OVERFLOW,
                severity=ErrorSeverity.HIGH,
                component='queue_manager',
                message=f"Queue {queue_name} overflow",
                exception_type='QueueOverflow',
                traceback='',
                context={
                    'queue_name': queue_name,
                    'queue_size': self.main_queue.qsize(),
                    'message_priority': message.priority.name if hasattr(message.priority, 'name') else 'unknown',
                    'message_topic': message.topic,
                    'message_size': len(str(message))
                }
            )
            
            self.error_logger.log_error(error_report)
            
            # Apply backpressure strategies
            logger.warning(f"Queue overflow detected for {queue_name}, message dropped")
            
            # Add to dead letter queue if available
            if self.dead_letter_queue:
                self.dead_letter_queue.add_message(message, error_report)
            
        except Exception as e:
            logger.critical(f"Critical error in queue overflow handling: {e}")
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """
        Categorize error based on exception type and context.
        
        Args:
            error: The exception to categorize
            
        Returns:
            ErrorCategory classification
        """
        error_type = type(error).__name__
        error_str = str(error).lower()
        
        if 'connection' in error_type.lower() or 'network' in error_type.lower():
            return ErrorCategory.CONNECTION_ERROR
        elif 'timeout' in error_type.lower() or 'timeout' in error_str:
            return ErrorCategory.TIMEOUT_ERROR
        elif 'serializ' in error_type.lower() or 'pickle' in error_type.lower():
            return ErrorCategory.SERIALIZATION_ERROR
        elif 'memory' in error_type.lower() or 'resource' in error_type.lower():
            return ErrorCategory.RESOURCE_EXHAUSTION
        elif 'auth' in error_type.lower() or 'permission' in error_type.lower():
            return ErrorCategory.AUTHENTICATION_ERROR
        else:
            return ErrorCategory.MESSAGE_PROCESSING_ERROR
    
    def _assess_severity(self, error: Exception, context: Dict[str, Any]) -> ErrorSeverity:
        """
        Assess error severity based on error type and context.
        
        Args:
            error: The exception
            context: Additional context
            
        Returns:
            ErrorSeverity classification
        """
        error_type = type(error).__name__
        
        # Critical errors that could crash the system
        if error_type in ['MemoryError', 'SystemError', 'KeyboardInterrupt']:
            return ErrorSeverity.CRITICAL
        
        # High severity for infrastructure errors
        if error_type in ['ConnectionError', 'TimeoutError', 'OSError']:
            return ErrorSeverity.HIGH
        
        # Medium severity for processing errors
        if error_type in ['ValueError', 'TypeError', 'AttributeError']:
            return ErrorSeverity.MEDIUM
        
        # Low severity for minor issues
        return ErrorSeverity.LOW
    
    def _assess_connection_severity(self, connection_id: str, error: Exception) -> ErrorSeverity:
        """
        Assess severity of connection errors.
        
        Args:
            connection_id: Connection identifier
            error: The exception
            
        Returns:
            ErrorSeverity classification
        """
        # For now, treat all connection errors as HIGH severity
        # In a more sophisticated system, we might check if there are backup connections
        return ErrorSeverity.HIGH
    
    def _escalate_critical_error(self, error_report: ErrorReport) -> None:
        """
        Escalate critical errors to system administrators.
        
        Args:
            error_report: The critical error report
        """
        try:
            # Log critical error
            logger.critical(f"CRITICAL ERROR ESCALATION: {error_report.error_id}")
            logger.critical(f"Category: {error_report.category.value}, Message: {error_report.message}")
            
            # In production, this would send notifications
            # self.notification_system.send_critical_alert(error_report)
            
        except Exception as e:
            logger.critical(f"Failed to escalate critical error: {e}")
    
    def requeue_message(self, message: Message) -> bool:
        """
        Requeue a message for reprocessing.
        
        Args:
            message: Message to requeue
            
        Returns:
            True if requeue was successful
        """
        try:
            queue_item = (message.priority.value, message.timestamp, message)
            self.main_queue.put_nowait(queue_item)
            logger.debug(f"Requeued message {message.message_id}")
            return True
        except queue.Full:
            logger.warning(f"Cannot requeue message {message.message_id}: queue full")
            return False
    
    def apply_backpressure(self, queue_name: str) -> int:
        """
        Apply backpressure by dropping low-priority messages.
        
        Args:
            queue_name: Name of the queue to apply backpressure
            
        Returns:
            Number of messages dropped
        """
        # This is a simplified implementation
        # In production, we would implement more sophisticated backpressure
        dropped_count = 0
        logger.info(f"Applying backpressure to {queue_name}")
        return dropped_count
    
    def can_expand_capacity(self, queue_name: str) -> bool:
        """
        Check if queue capacity can be expanded.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            True if expansion is possible
        """
        # For now, return False as we have a fixed queue size
        return False
    
    def expand_capacity(self, queue_name: str, factor: float = 1.5) -> None:
        """
        Expand queue capacity temporarily.
        
        Args:
            queue_name: Name of the queue
            factor: Expansion factor
        """
        # Not implemented for fixed-size queues
        logger.warning(f"Queue expansion not supported for {queue_name}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of message bus.
        
        Returns:
            Health status dictionary with metrics and error statistics
        """
        try:
            with self.stats_lock:
                # Use messages_delivered as messages_processed since they're essentially the same
                messages_processed = self.stats.get('messages_processed', self.stats['messages_delivered'])
                messages_failed = self.stats['messages_failed']
                total_messages = messages_processed + messages_failed
                success_rate = (messages_processed / total_messages) if total_messages > 0 else 1.0
                uptime = time.time() - self.stats['start_time']
            
            status = {
                'status': 'healthy' if success_rate > 0.95 else 'degraded' if success_rate > 0.8 else 'critical',
                'running': self.running,
                'uptime_seconds': uptime,
                'messages_processed': messages_processed,
                'messages_failed': messages_failed,
                'success_rate': success_rate,
                'errors_recovered': self.stats['errors_recovered'],
                'main_queue_size': self.main_queue.qsize(),
                'main_queue_full': self.main_queue.qsize() >= self.max_queue_size * 0.9,
                'active_subscriptions': sum(
                    len([s for s in subs if s.active]) 
                    for subs in self.subscriptions.values()
                ),
                'dispatch_thread_alive': self.dispatch_thread.is_alive() if self.dispatch_thread else False,
            }
            
            # Add error statistics if error handling is enabled
            if self.enable_error_handling and self.error_logger:
                error_stats = self.error_logger.get_error_statistics()
                status['error_statistics'] = error_stats
                
                if self.recovery_manager:
                    status['circuit_breaker_status'] = self.recovery_manager.get_circuit_breaker_status()
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'unknown', 'error': str(e)}
    
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
        """
        Perform health check and return status.
        
        Returns comprehensive health status including error handling metrics.
        """
        return self.get_health_status()


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


def init_message_bus(max_queue_size: int = 10000, enable_error_handling: bool = True,
                     error_log_dir: str = "data/logs/message_bus") -> MessageBus:
    """
    Initialize the global message bus with specific configuration.
    
    Args:
        max_queue_size: Maximum queue size
        enable_error_handling: Enable comprehensive error handling
        error_log_dir: Directory for error logs
    """
    global _global_message_bus
    
    with _bus_lock:
        if _global_message_bus is not None:
            _global_message_bus.stop()
        
        _global_message_bus = MessageBus(max_queue_size, enable_error_handling, error_log_dir)
        return _global_message_bus