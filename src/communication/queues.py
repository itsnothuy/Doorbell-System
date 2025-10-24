#!/usr/bin/env python3
"""
Queue Management System - High-Performance Queue Orchestration

This module provides queue management and coordination for the pipeline,
ensuring proper message flow and backpressure handling.
"""

import time
import queue
import threading
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class QueueType(Enum):
    """Types of queues in the system."""
    PRIORITY = "priority"
    FIFO = "fifo"
    LIFO = "lifo"
    RING_BUFFER = "ring_buffer"


class BackpressureStrategy(Enum):
    """Strategies for handling queue backpressure."""
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block"
    REJECT = "reject"


@dataclass
class QueueConfig:
    """Configuration for a managed queue."""
    name: str
    queue_type: QueueType = QueueType.FIFO
    max_size: int = 1000
    backpressure_strategy: BackpressureStrategy = BackpressureStrategy.DROP_OLDEST
    high_water_mark: float = 0.8  # Percentage at which to trigger warnings
    low_water_mark: float = 0.2   # Percentage at which backpressure is relieved
    
    # Performance tuning
    batch_size: int = 1
    timeout: float = 1.0
    
    # Monitoring
    enable_metrics: bool = True
    metric_interval: float = 60.0


@dataclass
class QueueMetrics:
    """Metrics for queue performance monitoring."""
    queue_name: str
    current_size: int = 0
    max_size: int = 0
    total_enqueued: int = 0
    total_dequeued: int = 0
    total_dropped: int = 0
    avg_wait_time: float = 0.0
    max_wait_time: float = 0.0
    throughput_per_second: float = 0.0
    last_updated: float = field(default_factory=time.time)


class ManagedQueue:
    """
    A managed queue with monitoring, backpressure handling, and metrics.
    """
    
    def __init__(self, config: QueueConfig):
        self.config = config
        self.logger = logging.getLogger(f"Queue-{config.name}")
        
        # Create the underlying queue
        self._queue = self._create_queue()
        self._lock = threading.RLock()
        
        # Metrics and monitoring
        self.metrics = QueueMetrics(
            queue_name=config.name,
            max_size=config.max_size
        )
        
        # Backpressure state
        self._backpressure_active = False
        self._last_metric_update = time.time()
        
        # Wait time tracking
        self._wait_times: List[float] = []
        self._wait_times_lock = threading.Lock()
        
        self.logger.info(f"Created managed queue: {config.name} "
                        f"(type={config.queue_type.value}, max_size={config.max_size})")
    
    def _create_queue(self) -> queue.Queue:
        """Create the appropriate queue type."""
        if self.config.queue_type == QueueType.PRIORITY:
            return queue.PriorityQueue(maxsize=self.config.max_size)
        elif self.config.queue_type == QueueType.LIFO:
            return queue.LifoQueue(maxsize=self.config.max_size)
        elif self.config.queue_type == QueueType.RING_BUFFER:
            # Ring buffer implementation using collections.deque
            from collections import deque
            return deque(maxlen=self.config.max_size)
        else:  # FIFO
            return queue.Queue(maxsize=self.config.max_size)
    
    def put(self, item: Any, timeout: Optional[float] = None) -> bool:
        """
        Put an item in the queue with backpressure handling.
        
        Args:
            item: Item to enqueue
            timeout: Timeout for the operation
            
        Returns:
            bool: True if item was enqueued successfully
        """
        timeout = timeout or self.config.timeout
        start_time = time.time()
        
        try:
            with self._lock:
                # Check backpressure
                if self._should_apply_backpressure():
                    if self.config.backpressure_strategy == BackpressureStrategy.REJECT:
                        self.metrics.total_dropped += 1
                        self.logger.warning(f"Queue {self.config.name} rejecting item due to backpressure")
                        return False
                    elif self.config.backpressure_strategy == BackpressureStrategy.DROP_OLDEST:
                        self._drop_oldest_item()
                    elif self.config.backpressure_strategy == BackpressureStrategy.DROP_NEWEST:
                        self.metrics.total_dropped += 1
                        self.logger.debug(f"Queue {self.config.name} dropping newest item")
                        return False
                
                # Handle different queue types
                if isinstance(self._queue, queue.Queue):
                    if timeout == 0:
                        self._queue.put_nowait(item)
                    else:
                        self._queue.put(item, timeout=timeout)
                else:  # Ring buffer (deque)
                    self._queue.append(item)
                
                # Update metrics
                self.metrics.total_enqueued += 1
                self.metrics.current_size = self.qsize()
                
                # Record wait time
                wait_time = time.time() - start_time
                self._record_wait_time(wait_time)
                
                self.logger.debug(f"Enqueued item to {self.config.name} "
                                f"(size={self.metrics.current_size}/{self.config.max_size})")
                
                return True
                
        except queue.Full:
            if self.config.backpressure_strategy == BackpressureStrategy.BLOCK:
                # This shouldn't happen with timeout, but handle gracefully
                self.logger.warning(f"Queue {self.config.name} is full, blocking not supported")
            
            self.metrics.total_dropped += 1
            self.logger.warning(f"Queue {self.config.name} is full, dropping item")
            return False
        except Exception as e:
            self.logger.error(f"Error enqueueing to {self.config.name}: {e}")
            return False
    
    def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        """
        Get an item from the queue.
        
        Args:
            timeout: Timeout for the operation
            
        Returns:
            The dequeued item or None if timeout
        """
        timeout = timeout or self.config.timeout
        start_time = time.time()
        
        try:
            with self._lock:
                if isinstance(self._queue, queue.Queue):
                    if timeout == 0:
                        item = self._queue.get_nowait()
                    else:
                        item = self._queue.get(timeout=timeout)
                else:  # Ring buffer (deque)
                    if len(self._queue) == 0:
                        return None
                    item = self._queue.popleft()
                
                # Update metrics
                self.metrics.total_dequeued += 1
                self.metrics.current_size = self.qsize()
                
                # Record wait time
                wait_time = time.time() - start_time
                self._record_wait_time(wait_time)
                
                # Check if backpressure should be relieved
                self._check_backpressure_relief()
                
                self.logger.debug(f"Dequeued item from {self.config.name} "
                                f"(size={self.metrics.current_size}/{self.config.max_size})")
                
                return item
                
        except queue.Empty:
            return None
        except Exception as e:
            self.logger.error(f"Error dequeuing from {self.config.name}: {e}")
            return None
    
    def get_batch(self, batch_size: Optional[int] = None, 
                  timeout: Optional[float] = None) -> List[Any]:
        """
        Get multiple items from the queue at once.
        
        Args:
            batch_size: Number of items to retrieve
            timeout: Timeout for the operation
            
        Returns:
            List of items (may be fewer than requested)
        """
        batch_size = batch_size or self.config.batch_size
        timeout = timeout or self.config.timeout
        
        items = []
        start_time = time.time()
        
        for _ in range(batch_size):
            # Adjust remaining timeout
            elapsed = time.time() - start_time
            remaining_timeout = max(0, timeout - elapsed)
            
            if remaining_timeout <= 0:
                break
            
            item = self.get(timeout=remaining_timeout)
            if item is None:
                break
            
            items.append(item)
        
        self.logger.debug(f"Dequeued batch of {len(items)} items from {self.config.name}")
        return items
    
    def qsize(self) -> int:
        """Get the current queue size."""
        with self._lock:
            if isinstance(self._queue, queue.Queue):
                return self._queue.qsize()
            else:  # Ring buffer (deque)
                return len(self._queue)
    
    def empty(self) -> bool:
        """Check if the queue is empty."""
        return self.qsize() == 0
    
    def full(self) -> bool:
        """Check if the queue is full."""
        return self.qsize() >= self.config.max_size
    
    def _should_apply_backpressure(self) -> bool:
        """Check if backpressure should be applied."""
        current_size = self.qsize()
        threshold = int(self.config.max_size * self.config.high_water_mark)
        
        if current_size >= threshold and not self._backpressure_active:
            self._backpressure_active = True
            self.logger.warning(f"Backpressure activated for {self.config.name} "
                              f"(size={current_size}/{self.config.max_size})")
        
        return self._backpressure_active
    
    def _check_backpressure_relief(self) -> None:
        """Check if backpressure can be relieved."""
        if not self._backpressure_active:
            return
        
        current_size = self.qsize()
        threshold = int(self.config.max_size * self.config.low_water_mark)
        
        if current_size <= threshold:
            self._backpressure_active = False
            self.logger.info(f"Backpressure relieved for {self.config.name} "
                           f"(size={current_size}/{self.config.max_size})")
    
    def _drop_oldest_item(self) -> None:
        """Drop the oldest item from the queue."""
        try:
            if isinstance(self._queue, queue.Queue):
                # For standard queues, we need to recreate to drop oldest
                old_items = []
                while not self._queue.empty():
                    try:
                        old_items.append(self._queue.get_nowait())
                    except queue.Empty:
                        break
                
                # Put back all but the oldest
                for item in old_items[1:]:
                    try:
                        self._queue.put_nowait(item)
                    except queue.Full:
                        break
                        
                self.metrics.total_dropped += 1
                self.logger.debug(f"Dropped oldest item from {self.config.name}")
                
            else:  # Ring buffer (deque)
                if len(self._queue) > 0:
                    self._queue.popleft()
                    self.metrics.total_dropped += 1
                    self.logger.debug(f"Dropped oldest item from {self.config.name}")
                    
        except Exception as e:
            self.logger.error(f"Error dropping oldest item from {self.config.name}: {e}")
    
    def _record_wait_time(self, wait_time: float) -> None:
        """Record wait time for metrics."""
        with self._wait_times_lock:
            self._wait_times.append(wait_time)
            
            # Keep only recent wait times (last 1000)
            if len(self._wait_times) > 1000:
                self._wait_times = self._wait_times[-1000:]
            
            # Update metrics
            self.metrics.avg_wait_time = sum(self._wait_times) / len(self._wait_times)
            self.metrics.max_wait_time = max(self._wait_times)
    
    def get_metrics(self) -> QueueMetrics:
        """Get current queue metrics."""
        with self._lock:
            # Update current size
            self.metrics.current_size = self.qsize()
            
            # Calculate throughput
            now = time.time()
            time_diff = now - self._last_metric_update
            
            if time_diff >= self.config.metric_interval:
                ops_in_period = (self.metrics.total_enqueued + 
                               self.metrics.total_dequeued)
                self.metrics.throughput_per_second = ops_in_period / time_diff
                self._last_metric_update = now
            
            self.metrics.last_updated = now
            
            return self.metrics
    
    def clear(self) -> int:
        """Clear all items from the queue and return the count of removed items."""
        with self._lock:
            removed_count = self.qsize()
            
            if isinstance(self._queue, queue.Queue):
                # Clear standard queue
                with self._queue.mutex:
                    self._queue.queue.clear()
            else:  # Ring buffer (deque)
                self._queue.clear()
            
            self.metrics.current_size = 0
            self.logger.info(f"Cleared {removed_count} items from {self.config.name}")
            
            return removed_count


class QueueManager:
    """
    Central manager for all pipeline queues.
    
    Provides queue creation, monitoring, and coordination.
    """
    
    def __init__(self):
        self.queues: Dict[str, ManagedQueue] = {}
        self.queue_configs: Dict[str, QueueConfig] = {}
        self._lock = threading.RLock()
        
        # Monitoring
        self._monitoring_enabled = True
        self._monitoring_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        logger.info("Queue manager initialized")
    
    def create_queue(self, config: QueueConfig) -> ManagedQueue:
        """Create a new managed queue."""
        with self._lock:
            if config.name in self.queues:
                logger.warning(f"Queue {config.name} already exists, returning existing queue")
                return self.queues[config.name]
            
            queue_instance = ManagedQueue(config)
            self.queues[config.name] = queue_instance
            self.queue_configs[config.name] = config
            
            logger.info(f"Created queue: {config.name}")
            return queue_instance
    
    def get_queue(self, name: str) -> Optional[ManagedQueue]:
        """Get an existing queue by name."""
        with self._lock:
            return self.queues.get(name)
    
    def remove_queue(self, name: str) -> bool:
        """Remove a queue."""
        with self._lock:
            if name not in self.queues:
                return False
            
            # Clear the queue first
            queue_instance = self.queues[name]
            cleared_count = queue_instance.clear()
            
            # Remove from manager
            del self.queues[name]
            del self.queue_configs[name]
            
            logger.info(f"Removed queue {name} (cleared {cleared_count} items)")
            return True
    
    def start(self) -> None:
        """Start the queue manager and monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Queue manager already running")
            return
        
        self._shutdown_event.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="QueueManager-Monitor",
            daemon=True
        )
        self._monitoring_thread.start()
        
        logger.info("Queue manager started")
    
    def stop(self) -> None:
        """Stop the queue manager."""
        logger.info("Stopping queue manager...")
        self._shutdown_event.set()
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
        
        # Clear all queues
        with self._lock:
            for name, queue_instance in self.queues.items():
                cleared_count = queue_instance.clear()
                logger.info(f"Cleared queue {name}: {cleared_count} items")
        
        logger.info("Queue manager stopped")
    
    def _monitoring_loop(self) -> None:
        """Monitor all queues for health and performance."""
        logger.info("Queue monitoring started")
        
        while not self._shutdown_event.is_set():
            try:
                self._check_queue_health()
                time.sleep(30.0)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in queue monitoring: {e}")
                time.sleep(5.0)
        
        logger.info("Queue monitoring stopped")
    
    def _check_queue_health(self) -> None:
        """Check health of all queues."""
        with self._lock:
            for name, queue_instance in self.queues.items():
                metrics = queue_instance.get_metrics()
                
                # Check for high queue usage
                usage_ratio = metrics.current_size / max(1, metrics.max_size)
                if usage_ratio > queue_instance.config.high_water_mark:
                    logger.warning(f"Queue {name} is {usage_ratio:.1%} full "
                                 f"({metrics.current_size}/{metrics.max_size})")
                
                # Check for high drop rate
                if metrics.total_enqueued > 0:
                    drop_rate = metrics.total_dropped / metrics.total_enqueued
                    if drop_rate > 0.05:  # 5% drop rate
                        logger.warning(f"Queue {name} has high drop rate: {drop_rate:.1%}")
                
                # Log performance metrics
                logger.debug(f"Queue {name}: size={metrics.current_size}, "
                           f"throughput={metrics.throughput_per_second:.1f} ops/s, "
                           f"avg_wait={metrics.avg_wait_time*1000:.1f}ms")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all queues."""
        with self._lock:
            status = {
                'total_queues': len(self.queues),
                'monitoring_enabled': self._monitoring_enabled,
                'queues': {}
            }
            
            for name, queue_instance in self.queues.items():
                metrics = queue_instance.get_metrics()
                status['queues'][name] = {
                    'type': queue_instance.config.queue_type.value,
                    'size': metrics.current_size,
                    'max_size': metrics.max_size,
                    'usage_percent': (metrics.current_size / max(1, metrics.max_size)) * 100,
                    'total_enqueued': metrics.total_enqueued,
                    'total_dequeued': metrics.total_dequeued,
                    'total_dropped': metrics.total_dropped,
                    'throughput_per_second': metrics.throughput_per_second,
                    'backpressure_active': queue_instance._backpressure_active
                }
            
            return status
    
    def get_queue_list(self) -> List[str]:
        """Get list of all queue names."""
        with self._lock:
            return list(self.queues.keys())


# Convenience functions for common queue operations

def create_frame_buffer(name: str, max_frames: int = 30) -> ManagedQueue:
    """Create a ring buffer for frame data."""
    config = QueueConfig(
        name=name,
        queue_type=QueueType.RING_BUFFER,
        max_size=max_frames,
        backpressure_strategy=BackpressureStrategy.DROP_OLDEST
    )
    
    manager = QueueManager()
    return manager.create_queue(config)


def create_priority_queue(name: str, max_size: int = 1000) -> ManagedQueue:
    """Create a priority queue for urgent processing."""
    config = QueueConfig(
        name=name,
        queue_type=QueueType.PRIORITY,
        max_size=max_size,
        backpressure_strategy=BackpressureStrategy.DROP_NEWEST
    )
    
    manager = QueueManager()
    return manager.create_queue(config)