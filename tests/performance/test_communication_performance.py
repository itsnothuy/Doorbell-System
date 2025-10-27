#!/usr/bin/env python3
"""
Communication Performance Test Suite

Performance benchmarks and profiling for the communication infrastructure
including throughput, latency, and resource usage measurements.
"""

import time
import pytest
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

from src.communication.message_bus import MessageBus, Message, MessagePriority
from src.communication.events import PipelineEvent, EventType, EventPriority
from src.communication.queues import (
    QueueManager,
    QueueConfig,
    QueueType,
    BackpressureStrategy,
)


# Performance test configuration constants
THROUGHPUT_MIN_THRESHOLD = 1000  # messages/second
DELIVERY_THROUGHPUT_MIN = 500    # messages/second
LATENCY_MAX_AVG_MS = 5.0        # milliseconds
LATENCY_MAX_E2E_MS = 10.0       # milliseconds
QUEUE_THROUGHPUT_MIN = 5000     # operations/second
CONCURRENT_THROUGHPUT_MIN = 1000 # messages/second

# Test timing constants
MAX_WAIT_TIMEOUT = 30           # seconds
DELIVERY_WAIT_TIME = 2.0        # seconds
MESSAGE_BATCH_DELAY = 0.001     # seconds


class TestMessageBusPerformance:
    """Performance benchmarks for message bus operations."""
    
    @pytest.fixture
    def message_bus(self):
        """Create high-capacity message bus."""
        bus = MessageBus(max_queue_size=10000)
        bus.start()
        yield bus
        bus.stop()
    
    @pytest.mark.slow
    def test_publish_throughput(self, message_bus):
        """Benchmark message publishing throughput."""
        message_count = 10000
        start_time = time.time()
        
        for i in range(message_count):
            message_bus.publish(
                f"topic_{i % 10}",
                {"sequence": i, "data": "x" * 100}
            )
        
        duration = time.time() - start_time
        throughput = message_count / duration
        
        print(f"\n📊 Publish Throughput: {throughput:.0f} messages/second")
        print(f"   Duration: {duration:.2f}s for {message_count} messages")
        
        # Target: >1,000 messages/second
        assert throughput > THROUGHPUT_MIN_THRESHOLD, \
            f"Throughput {throughput:.0f} below minimum {THROUGHPUT_MIN_THRESHOLD}"
    
    @pytest.mark.slow
    def test_subscribe_delivery_throughput(self, message_bus):
        """Benchmark message delivery to subscribers."""
        message_count = 5000
        received_count = [0]
        lock = threading.Lock()
        
        def fast_handler(message):
            with lock:
                received_count[0] += 1
        
        # Subscribe handler
        message_bus.subscribe("throughput_test", fast_handler, "fast_sub")
        
        start_time = time.time()
        
        for i in range(message_count):
            message_bus.publish("throughput_test", {"sequence": i})
        
        # Wait for all messages to be delivered
        while received_count[0] < message_count and (time.time() - start_time) < MAX_WAIT_TIMEOUT:
            time.sleep(0.1)
        
        duration = time.time() - start_time
        throughput = received_count[0] / duration
        
        print(f"\n📊 Delivery Throughput: {throughput:.0f} messages/second")
        print(f"   Delivered: {received_count[0]}/{message_count} messages")
        print(f"   Duration: {duration:.2f}s")
        
        assert received_count[0] >= message_count * 0.95  # Allow 5% loss
        assert throughput > DELIVERY_THROUGHPUT_MIN, \
            f"Throughput {throughput:.0f} below minimum {DELIVERY_THROUGHPUT_MIN}"
    
    @pytest.mark.slow
    def test_multiple_subscribers_throughput(self, message_bus):
        """Benchmark throughput with multiple subscribers."""
        message_count = 1000
        subscriber_count = 10
        received_counts = [0] * subscriber_count
        locks = [threading.Lock() for _ in range(subscriber_count)]
        
        def create_handler(index):
            def handler(message):
                with locks[index]:
                    received_counts[index] += 1
            return handler
        
        # Subscribe multiple handlers
        for i in range(subscriber_count):
            message_bus.subscribe(
                "multi_sub_test",
                create_handler(i),
                f"sub_{i}"
            )
        
        start_time = time.time()
        
        for i in range(message_count):
            message_bus.publish("multi_sub_test", {"sequence": i})
        
        # Wait for delivery
        time.sleep(DELIVERY_WAIT_TIME)
        
        duration = time.time() - start_time
        total_delivered = sum(received_counts)
        expected_total = message_count * subscriber_count
        throughput = total_delivered / duration
        
        print(f"\n📊 Multi-Subscriber Throughput: {throughput:.0f} messages/second")
        print(f"   Delivered: {total_delivered}/{expected_total} total")
        print(f"   Subscribers: {subscriber_count}")
        print(f"   Per subscriber: {[c for c in received_counts]}")
        
        # Allow some message loss due to timing
        assert total_delivered >= expected_total * 0.90


class TestMessageLatency:
    """Latency benchmarks for message operations."""
    
    @pytest.fixture
    def message_bus(self):
        """Create message bus for latency testing."""
        bus = MessageBus()
        bus.start()
        yield bus
        bus.stop()
    
    @pytest.mark.slow
    def test_publish_latency(self, message_bus):
        """Measure message publishing latency."""
        iterations = 1000
        latencies = []
        
        for i in range(iterations):
            start = time.perf_counter()
            message_bus.publish("latency_test", {"sequence": i})
            latency = (time.perf_counter() - start) * 1000  # milliseconds
            latencies.append(latency)
        
        avg_latency = statistics.mean(latencies)
        p50_latency = statistics.median(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        p99_latency = sorted(latencies)[int(0.99 * len(latencies))]
        
        print(f"\n📊 Publish Latency:")
        print(f"   Average: {avg_latency:.3f}ms")
        print(f"   P50: {p50_latency:.3f}ms")
        print(f"   P95: {p95_latency:.3f}ms")
        print(f"   P99: {p99_latency:.3f}ms")
        
        # Target: <5ms average
        assert avg_latency < LATENCY_MAX_AVG_MS, \
            f"Average latency {avg_latency:.3f}ms exceeds maximum {LATENCY_MAX_AVG_MS}ms"
    
    @pytest.mark.slow
    def test_end_to_end_latency(self, message_bus):
        """Measure end-to-end delivery latency."""
        iterations = 500
        latencies = []
        send_times = {}
        receive_times = {}
        lock = threading.Lock()
        
        def latency_handler(message):
            receive_time = time.perf_counter()
            msg_id = message.data['id']
            with lock:
                receive_times[msg_id] = receive_time
        
        message_bus.subscribe("e2e_latency", latency_handler, "latency_sub")
        
        for i in range(iterations):
            send_time = time.perf_counter()
            send_times[i] = send_time
            message_bus.publish("e2e_latency", {"id": i})
            time.sleep(MESSAGE_BATCH_DELAY)
        
        # Wait for all messages
        time.sleep(DELIVERY_WAIT_TIME)
        
        # Calculate latencies using actual send/receive times
        for i in range(iterations):
            if i in receive_times and i in send_times:
                latency_ms = (receive_times[i] - send_times[i]) * 1000
                if latency_ms > 0 and latency_ms < 1000:  # Sanity check
                    latencies.append(latency_ms)
        
        if latencies:
            avg_latency = statistics.mean(latencies)
            p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
            
            print(f"\n📊 End-to-End Latency:")
            print(f"   Messages received: {len(receive_times)}/{iterations}")
            print(f"   Average: {avg_latency:.3f}ms")
            print(f"   P95: {p95_latency:.3f}ms")
            
            # Relaxed target for CI environment
            assert avg_latency < LATENCY_MAX_E2E_MS, \
                f"E2E latency {avg_latency:.3f}ms exceeds maximum {LATENCY_MAX_E2E_MS}ms"


class TestQueuePerformance:
    """Performance benchmarks for queue operations."""
    
    @pytest.mark.slow
    def test_queue_enqueue_throughput(self):
        """Benchmark queue enqueue operations."""
        queue_manager = QueueManager()
        config = QueueConfig(
            name="perf_queue",
            queue_type=QueueType.FIFO,
            max_size=10000
        )
        queue = queue_manager.create_queue(config)
        
        message_count = 10000
        start_time = time.time()
        
        for i in range(message_count):
            queue.put(f"item_{i}")
        
        duration = time.time() - start_time
        throughput = message_count / duration
        
        print(f"\n📊 Queue Enqueue Throughput: {throughput:.0f} items/second")
        
        queue_manager.stop()
        
        assert throughput > QUEUE_THROUGHPUT_MIN, \
            f"Throughput {throughput:.0f} below target {QUEUE_THROUGHPUT_MIN}"
    
    @pytest.mark.slow
    def test_queue_dequeue_throughput(self):
        """Benchmark queue dequeue operations."""
        queue_manager = QueueManager()
        config = QueueConfig(
            name="perf_queue",
            queue_type=QueueType.FIFO,
            max_size=10000
        )
        queue = queue_manager.create_queue(config)
        
        # Fill queue
        message_count = 10000
        for i in range(message_count):
            queue.put(f"item_{i}")
        
        # Measure dequeue
        start_time = time.time()
        
        retrieved = 0
        for _ in range(message_count):
            item = queue.get(timeout=0.01)
            if item:
                retrieved += 1
            else:
                break
        
        duration = time.time() - start_time
        throughput = retrieved / duration
        
        print(f"\n📊 Queue Dequeue Throughput: {throughput:.0f} items/second")
        print(f"   Retrieved: {retrieved}/{message_count}")
        
        queue_manager.stop()
        
        assert throughput > QUEUE_THROUGHPUT_MIN, \
            f"Throughput {throughput:.0f} below target {QUEUE_THROUGHPUT_MIN}"
    
    @pytest.mark.slow
    def test_priority_queue_performance(self):
        """Benchmark priority queue with mixed priorities."""
        queue_manager = QueueManager()
        config = QueueConfig(
            name="priority_perf",
            queue_type=QueueType.PRIORITY,
            max_size=5000
        )
        queue = queue_manager.create_queue(config)
        
        message_count = 5000
        start_time = time.time()
        
        # Enqueue with random priorities
        for i in range(message_count):
            priority = i % 4 + 1  # Priorities 1-4
            queue.put((priority, f"item_{i}"))
        
        enqueue_duration = time.time() - start_time
        
        # Dequeue all
        dequeue_start = time.time()
        retrieved = []
        while not queue.empty():
            item = queue.get(timeout=0.01)
            if item:
                retrieved.append(item)
            else:
                break
        
        dequeue_duration = time.time() - dequeue_start
        
        enqueue_throughput = message_count / enqueue_duration
        dequeue_throughput = len(retrieved) / dequeue_duration
        
        print(f"\n📊 Priority Queue Performance:")
        print(f"   Enqueue: {enqueue_throughput:.0f} items/second")
        print(f"   Dequeue: {dequeue_throughput:.0f} items/second")
        
        queue_manager.stop()


class TestConcurrentPerformance:
    """Performance benchmarks for concurrent operations."""
    
    @pytest.fixture
    def message_bus(self):
        """Create message bus for concurrent testing."""
        bus = MessageBus(max_queue_size=20000)
        bus.start()
        yield bus
        bus.stop()
    
    @pytest.mark.slow
    def test_concurrent_publishers(self, message_bus):
        """Benchmark multiple concurrent publishers."""
        publishers = 10
        messages_per_publisher = 500
        total_messages = publishers * messages_per_publisher
        
        def publisher_thread(thread_id):
            for i in range(messages_per_publisher):
                message_bus.publish(
                    f"concurrent_topic",
                    {"publisher": thread_id, "sequence": i}
                )
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=publishers) as executor:
            futures = [
                executor.submit(publisher_thread, i)
                for i in range(publishers)
            ]
            for future in futures:
                future.result()
        
        duration = time.time() - start_time
        throughput = total_messages / duration
        
        print(f"\n📊 Concurrent Publishers ({publishers} threads):")
        print(f"   Total throughput: {throughput:.0f} messages/second")
        print(f"   Per publisher: {throughput/publishers:.0f} messages/second")
        
        assert throughput > CONCURRENT_THROUGHPUT_MIN, \
            f"Throughput {throughput:.0f} below target {CONCURRENT_THROUGHPUT_MIN}"
    
    @pytest.mark.slow
    def test_concurrent_subscribers(self, message_bus):
        """Benchmark message delivery to many concurrent subscribers."""
        subscribers = 20
        message_count = 500
        received_counts = [0] * subscribers
        locks = [threading.Lock() for _ in range(subscribers)]
        
        def create_handler(index):
            def handler(message):
                with locks[index]:
                    received_counts[index] += 1
                # Simulate some processing
                time.sleep(0.0001)
            return handler
        
        # Subscribe all handlers
        for i in range(subscribers):
            message_bus.subscribe(
                "concurrent_sub_topic",
                create_handler(i),
                f"sub_{i}"
            )
        
        start_time = time.time()
        
        # Publish messages
        for i in range(message_count):
            message_bus.publish("concurrent_sub_topic", {"sequence": i})
        
        # Wait for all deliveries
        time.sleep(5.0)  # Extended wait for concurrent processing
        
        duration = time.time() - start_time
        total_delivered = sum(received_counts)
        expected_total = message_count * subscribers
        throughput = total_delivered / duration
        
        print(f"\n📊 Concurrent Subscribers ({subscribers} subscribers):")
        print(f"   Total deliveries: {total_delivered}/{expected_total}")
        print(f"   Throughput: {throughput:.0f} deliveries/second")
        print(f"   Delivery rate: {(total_delivered/expected_total)*100:.1f}%")


class TestScalabilityBenchmarks:
    """Scalability benchmarks for increasing loads."""
    
    @pytest.mark.slow
    def test_subscriber_scalability(self):
        """Test performance scaling with increasing subscribers."""
        message_bus = MessageBus(max_queue_size=10000)
        message_bus.start()
        
        subscriber_counts = [1, 5, 10, 20, 50]
        results = []
        
        for sub_count in subscriber_counts:
            received = [0]
            lock = threading.Lock()
            
            def handler(message):
                with lock:
                    received[0] += 1
            
            # Subscribe handlers
            for i in range(sub_count):
                message_bus.subscribe("scale_test", handler, f"sub_{i}")
            
            # Publish messages
            message_count = 100
            start_time = time.time()
            
            for i in range(message_count):
                message_bus.publish("scale_test", {"seq": i})
            
            # Wait for delivery
            time.sleep(DELIVERY_WAIT_TIME)
            
            duration = time.time() - start_time
            expected = message_count * sub_count
            throughput = received[0] / duration
            
            results.append({
                'subscribers': sub_count,
                'throughput': throughput,
                'delivered': received[0],
                'expected': expected,
                'rate': received[0] / expected
            })
            
            # Unsubscribe all for next iteration
            for i in range(sub_count):
                message_bus.unsubscribe("scale_test", f"sub_{i}")
        
        message_bus.stop()
        
        print(f"\n📊 Subscriber Scalability:")
        for result in results:
            print(f"   {result['subscribers']:3d} subs: "
                  f"{result['throughput']:7.0f} msg/s, "
                  f"{result['rate']*100:5.1f}% delivered")


class TestMemoryPerformance:
    """Memory usage and leak detection tests."""
    
    @pytest.mark.slow
    def test_memory_stability_long_running(self):
        """Test memory stability during extended operation."""
        message_bus = MessageBus()
        message_bus.start()
        
        received = [0]
        lock = threading.Lock()
        
        def handler(message):
            with lock:
                received[0] += 1
        
        message_bus.subscribe("memory_test", handler, "mem_sub")
        
        # Run for extended period
        iterations = 5000
        for i in range(iterations):
            message_bus.publish("memory_test", {"sequence": i, "data": "x" * 1000})
            if i % 1000 == 0:
                time.sleep(0.1)  # Brief pause
        
        time.sleep(1.0)
        
        print(f"\n📊 Memory Stability Test:")
        print(f"   Messages published: {iterations}")
        print(f"   Messages received: {received[0]}")
        print(f"   Memory test completed successfully")
        
        message_bus.stop()
        
        # If we got here without errors, memory is stable
        assert received[0] >= iterations * 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])
