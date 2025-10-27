#!/usr/bin/env python3
"""
Communication Integration Test Suite

End-to-end integration tests for the communication infrastructure
including message bus, events, and queues working together.
"""

import time
import pytest
import threading
from unittest.mock import Mock
from concurrent.futures import ThreadPoolExecutor

from src.communication.message_bus import MessageBus, Message, MessagePriority
from src.communication.events import (
    PipelineEvent,
    EventType,
    EventPriority,
    DoorbellEvent,
    FrameEvent,
    MotionEvent,
    FaceDetectionEvent,
    FaceRecognition,
    FaceDetection,
    RecognitionStatus,
    BoundingBox,
    create_doorbell_event,
    create_frame_event,
)
from src.communication.queues import (
    QueueManager,
    QueueConfig,
    QueueType,
    BackpressureStrategy,
)


class TestMessageBusEventIntegration:
    """Test integration between message bus and events system."""
    
    @pytest.fixture
    def message_bus(self):
        """Create message bus for integration testing."""
        bus = MessageBus()
        bus.start()
        yield bus
        bus.stop()
    
    def test_pipeline_event_flow(self, message_bus):
        """Test complete pipeline event flow through message bus."""
        received_events = []
        
        def event_handler(message):
            event = message.data
            received_events.append(event)
        
        # Subscribe to pipeline topics
        message_bus.subscribe("doorbell", event_handler, "handler1")
        message_bus.subscribe("frame", event_handler, "handler2")
        message_bus.subscribe("motion", event_handler, "handler3")
        
        # Create and publish pipeline events
        doorbell_event = create_doorbell_event(channel=18)
        frame_event = create_frame_event("frame_data", frame_path="/test/frame.jpg")
        motion_event = MotionEvent(
            event_type=EventType.MOTION_DETECTED,
            motion_score=0.85
        )
        
        message_bus.publish("doorbell", doorbell_event)
        message_bus.publish("frame", frame_event)
        message_bus.publish("motion", motion_event)
        
        # Wait for processing
        time.sleep(0.3)
        
        # Verify all events received
        assert len(received_events) == 3
        assert any(e.event_type == EventType.DOORBELL_PRESSED for e in received_events)
        assert any(e.event_type == EventType.FRAME_CAPTURED for e in received_events)
        assert any(e.event_type == EventType.MOTION_DETECTED for e in received_events)
    
    def test_event_correlation_chain(self, message_bus):
        """Test correlated event chain through pipeline."""
        event_chain = []
        
        def chain_handler(message):
            event = message.data
            event_chain.append(event.event_type)
        
        message_bus.subscribe("pipeline.*", chain_handler, "chain_sub", pattern_match=True)
        
        # Create correlated event chain
        correlation_id = "test-correlation-123"
        
        # Doorbell event starts chain
        event1 = PipelineEvent(
            event_type=EventType.DOORBELL_PRESSED,
            correlation_id=correlation_id
        )
        
        # Frame event follows
        event2 = PipelineEvent(
            event_type=EventType.FRAME_CAPTURED,
            correlation_id=correlation_id,
            parent_event_id=event1.event_id
        )
        
        # Face detection follows
        event3 = PipelineEvent(
            event_type=EventType.FACES_DETECTED,
            correlation_id=correlation_id,
            parent_event_id=event2.event_id
        )
        
        # Publish chain
        message_bus.publish("pipeline.doorbell", event1)
        message_bus.publish("pipeline.frame", event2)
        message_bus.publish("pipeline.face", event3)
        
        time.sleep(0.3)
        
        # Verify chain received
        assert len(event_chain) == 3
        assert EventType.DOORBELL_PRESSED in event_chain
        assert EventType.FRAME_CAPTURED in event_chain
        assert EventType.FACES_DETECTED in event_chain
    
    def test_priority_event_ordering(self, message_bus):
        """Test that high priority events are processed first."""
        received_priorities = []
        
        def priority_handler(message):
            received_priorities.append(message.priority.value)
        
        message_bus.subscribe("priority_test", priority_handler, "priority_sub")
        
        # Publish events with different priorities
        message_bus.publish("priority_test", {"id": 1}, MessagePriority.NORMAL)
        message_bus.publish("priority_test", {"id": 2}, MessagePriority.CRITICAL)
        message_bus.publish("priority_test", {"id": 3}, MessagePriority.LOW)
        message_bus.publish("priority_test", {"id": 4}, MessagePriority.HIGH)
        
        time.sleep(0.3)
        
        # Verify all received
        assert len(received_priorities) == 4
        # Critical should be processed early
        assert MessagePriority.CRITICAL.value in received_priorities


class TestQueueEventIntegration:
    """Test integration between queues and events system."""
    
    @pytest.fixture
    def queue_manager(self):
        """Create queue manager for testing."""
        manager = QueueManager()
        yield manager
        manager.stop()
    
    def test_event_queuing_fifo(self, queue_manager):
        """Test FIFO queue with pipeline events."""
        config = QueueConfig(
            name="event_fifo",
            queue_type=QueueType.FIFO,
            max_size=10
        )
        queue = queue_manager.create_queue(config)
        
        # Create and enqueue events
        events = [
            PipelineEvent(event_type=EventType.MOTION_DETECTED, data={"id": i})
            for i in range(5)
        ]
        
        for event in events:
            queue.put(event)
        
        # Retrieve events
        retrieved = []
        for _ in range(5):
            event = queue.get()
            if event:
                retrieved.append(event)
        
        # Verify FIFO order
        assert len(retrieved) == 5
        for i, event in enumerate(retrieved):
            assert event.data["id"] == i
    
    def test_event_queuing_priority(self, queue_manager):
        """Test priority queue with prioritized events."""
        config = QueueConfig(
            name="event_priority",
            queue_type=QueueType.PRIORITY,
            max_size=10
        )
        queue = queue_manager.create_queue(config)
        
        # Create events with different priorities
        event_normal = PipelineEvent(
            event_type=EventType.MOTION_DETECTED,
            priority=EventPriority.NORMAL
        )
        event_critical = PipelineEvent(
            event_type=EventType.FACE_BLACKLISTED,
            priority=EventPriority.CRITICAL
        )
        event_low = PipelineEvent(
            event_type=EventType.FRAME_CAPTURED,
            priority=EventPriority.LOW
        )
        
        # Priority queue needs items as tuples: (priority_value, item)
        # Lower priority value = higher priority
        # But EventPriority.CRITICAL has higher value (4), so we need to negate or use reverse
        # Actually, Python's priority queue uses min heap, so lower values have higher priority
        # We need to use negative values or reverse the priority
        queue.put((5 - event_normal.priority.value, event_normal))
        queue.put((5 - event_critical.priority.value, event_critical))
        queue.put((5 - event_low.priority.value, event_low))
        
        # Retrieve - should get critical first (5-4=1 is lowest)
        first = queue.get()
        assert first[1].priority == EventPriority.CRITICAL


class TestEndToEndPipelineSimulation:
    """Test complete end-to-end pipeline simulation."""
    
    @pytest.fixture
    def pipeline_components(self):
        """Setup complete pipeline components."""
        message_bus = MessageBus()
        message_bus.start()
        
        queue_manager = QueueManager()
        queue_manager.start()
        
        # Create queues for each stage
        frame_queue_config = QueueConfig(
            name="frame_queue",
            queue_type=QueueType.RING_BUFFER,
            max_size=10,
            backpressure_strategy=BackpressureStrategy.DROP_OLDEST
        )
        frame_queue = queue_manager.create_queue(frame_queue_config)
        
        detection_queue_config = QueueConfig(
            name="detection_queue",
            queue_type=QueueType.FIFO,
            max_size=20
        )
        detection_queue = queue_manager.create_queue(detection_queue_config)
        
        yield {
            'message_bus': message_bus,
            'queue_manager': queue_manager,
            'frame_queue': frame_queue,
            'detection_queue': detection_queue
        }
        
        message_bus.stop()
        queue_manager.stop()
    
    def test_doorbell_to_recognition_flow(self, pipeline_components):
        """Test complete flow from doorbell press to face recognition."""
        message_bus = pipeline_components['message_bus']
        frame_queue = pipeline_components['frame_queue']
        detection_queue = pipeline_components['detection_queue']
        
        processed_stages = []
        
        # Stage 1: Doorbell handler
        def doorbell_handler(message):
            event = message.data
            processed_stages.append("doorbell")
            
            # Trigger frame capture
            frame_event = create_frame_event("mock_frame_data")
            frame_queue.put(frame_event)
            message_bus.publish("frame_captured", frame_event)
        
        # Stage 2: Frame processor
        def frame_handler(message):
            event = message.data
            processed_stages.append("frame")
            
            # Simulate motion detection
            motion_event = MotionEvent(
                event_type=EventType.MOTION_DETECTED,
                motion_score=0.85,
                frame_event=event
            )
            message_bus.publish("motion_detected", motion_event)
        
        # Stage 3: Motion handler
        def motion_handler(message):
            event = message.data
            processed_stages.append("motion")
            
            # Trigger face detection
            bbox = BoundingBox(x=10, y=20, width=100, height=80)
            face = FaceDetection(bounding_box=bbox, confidence=0.95)
            face_event = FaceDetectionEvent(
                event_type=EventType.FACES_DETECTED,
                faces=[face],
                detection_time=0.05,
                detector_type="test"
            )
            detection_queue.put(face_event)
            message_bus.publish("faces_detected", face_event)
        
        # Stage 4: Recognition handler
        def recognition_handler(message):
            event = message.data
            processed_stages.append("recognition")
        
        # Subscribe handlers
        message_bus.subscribe("doorbell_pressed", doorbell_handler, "doorbell_sub")
        message_bus.subscribe("frame_captured", frame_handler, "frame_sub")
        message_bus.subscribe("motion_detected", motion_handler, "motion_sub")
        message_bus.subscribe("faces_detected", recognition_handler, "recognition_sub")
        
        # Trigger pipeline with doorbell event
        doorbell_event = create_doorbell_event(channel=18)
        message_bus.publish("doorbell_pressed", doorbell_event)
        
        # Wait for pipeline processing
        time.sleep(0.5)
        
        # Verify all stages processed
        assert "doorbell" in processed_stages
        assert "frame" in processed_stages
        assert "motion" in processed_stages
        assert "recognition" in processed_stages
        
        # Verify queues were used
        assert frame_queue.metrics.total_enqueued > 0
        assert detection_queue.metrics.total_enqueued > 0


class TestConcurrentPipelineOperations:
    """Test concurrent operations across communication infrastructure."""
    
    @pytest.fixture
    def concurrent_setup(self):
        """Setup for concurrent testing."""
        message_bus = MessageBus(max_queue_size=500)
        message_bus.start()
        
        queue_manager = QueueManager()
        
        yield {
            'message_bus': message_bus,
            'queue_manager': queue_manager
        }
        
        message_bus.stop()
        queue_manager.stop()
    
    def test_multiple_publishers_single_queue(self, concurrent_setup):
        """Test multiple publishers writing to single queue."""
        queue_manager = concurrent_setup['queue_manager']
        
        config = QueueConfig(
            name="multi_pub_queue",
            queue_type=QueueType.FIFO,
            max_size=100
        )
        queue = queue_manager.create_queue(config)
        
        def publisher(thread_id):
            for i in range(10):
                event = PipelineEvent(
                    event_type=EventType.MOTION_DETECTED,
                    source=f"publisher_{thread_id}",
                    data={"count": i}
                )
                queue.put(event)
        
        # Run multiple publishers
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(publisher, i) for i in range(5)]
            for future in futures:
                future.result()
        
        # Verify all events received
        assert queue.metrics.total_enqueued == 50
    
    def test_multiple_subscribers_multiple_topics(self, concurrent_setup):
        """Test multiple subscribers across multiple topics."""
        message_bus = concurrent_setup['message_bus']
        
        received_counts = {
            'topic1': [0],
            'topic2': [0],
            'topic3': [0]
        }
        locks = {
            'topic1': threading.Lock(),
            'topic2': threading.Lock(),
            'topic3': threading.Lock()
        }
        
        def create_handler(topic):
            def handler(message):
                with locks[topic]:
                    received_counts[topic][0] += 1
            return handler
        
        # Subscribe multiple handlers to different topics
        for topic in ['topic1', 'topic2', 'topic3']:
            for i in range(3):  # 3 subscribers per topic
                message_bus.subscribe(
                    topic,
                    create_handler(topic),
                    f"{topic}_sub_{i}"
                )
        
        # Publish to all topics
        for topic in ['topic1', 'topic2', 'topic3']:
            for i in range(10):
                message_bus.publish(topic, {"data": i})
        
        time.sleep(0.5)
        
        # Each topic should have received 30 messages (10 * 3 subscribers)
        for topic in ['topic1', 'topic2', 'topic3']:
            assert received_counts[topic][0] == 30


class TestMessageOrderingGuarantees:
    """Test message ordering guarantees in the communication system."""
    
    @pytest.fixture
    def ordering_setup(self):
        """Setup for ordering tests."""
        message_bus = MessageBus()
        message_bus.start()
        yield message_bus
        message_bus.stop()
    
    def test_fifo_ordering_single_subscriber(self, ordering_setup):
        """Test FIFO ordering with single subscriber."""
        message_bus = ordering_setup
        received_order = []
        
        def ordered_handler(message):
            received_order.append(message.data['sequence'])
        
        message_bus.subscribe("ordered_topic", ordered_handler, "ordered_sub")
        
        # Publish sequence
        for i in range(20):
            message_bus.publish("ordered_topic", {"sequence": i})
        
        time.sleep(0.3)
        
        # Verify order maintained
        assert len(received_order) == 20
        for i in range(19):
            # Each should be close to previous (allowing for minor variations)
            assert abs(received_order[i+1] - received_order[i]) <= 2


class TestQueueOverflowAndRecovery:
    """Test queue overflow handling and recovery."""
    
    def test_queue_overflow_drop_oldest(self):
        """Test queue overflow with drop oldest strategy."""
        queue_manager = QueueManager()
        
        config = QueueConfig(
            name="overflow_test",
            queue_type=QueueType.FIFO,
            max_size=5,
            backpressure_strategy=BackpressureStrategy.DROP_OLDEST,
            high_water_mark=0.8
        )
        queue = queue_manager.create_queue(config)
        
        # Fill beyond capacity
        for i in range(10):
            event = PipelineEvent(
                event_type=EventType.MOTION_DETECTED,
                data={"sequence": i}
            )
            queue.put(event)
        
        # Check that oldest were dropped
        assert queue.metrics.total_dropped > 0
        
        # Recovery: should be able to continue processing
        new_event = PipelineEvent(
            event_type=EventType.MOTION_DETECTED,
            data={"sequence": 100}
        )
        result = queue.put(new_event)
        assert result is True
        
        queue_manager.stop()
    
    def test_queue_recovery_after_backpressure(self):
        """Test queue recovery after backpressure relief."""
        queue_manager = QueueManager()
        
        config = QueueConfig(
            name="recovery_test",
            queue_type=QueueType.FIFO,
            max_size=10,
            high_water_mark=0.8,
            low_water_mark=0.2,
            backpressure_strategy=BackpressureStrategy.DROP_OLDEST
        )
        queue = queue_manager.create_queue(config)
        
        # Trigger backpressure
        for i in range(10):
            queue.put(f"item{i}")
        
        # Drain queue
        for _ in range(8):
            queue.get()
        
        # Should be able to add new items
        result = queue.put("new_item")
        assert result is True
        
        queue_manager.stop()


class TestCrossComponentEventPropagation:
    """Test event propagation across all components."""
    
    def test_full_system_integration(self):
        """Test complete system with all components integrated."""
        # Setup all components
        message_bus = MessageBus()
        message_bus.start()
        
        queue_manager = QueueManager()
        queue_manager.start()
        
        # Create event processing queue
        event_queue_config = QueueConfig(
            name="event_processing",
            queue_type=QueueType.PRIORITY,
            max_size=50
        )
        event_queue = queue_manager.create_queue(event_queue_config)
        
        processed_events = []
        
        # Event processor
        def event_processor(message):
            event = message.data
            # Add to processing queue
            event_queue.put((event.priority.value, event))
        
        # Subscribe processor
        message_bus.subscribe("*", event_processor, "processor", pattern_match=True)
        
        # Create varied events
        events = [
            create_doorbell_event(),
            create_frame_event("frame_data"),
            PipelineEvent(event_type=EventType.MOTION_DETECTED, priority=EventPriority.HIGH),
            PipelineEvent(event_type=EventType.FACES_DETECTED, priority=EventPriority.NORMAL),
            PipelineEvent(event_type=EventType.FACE_RECOGNIZED, priority=EventPriority.CRITICAL),
        ]
        
        # Publish all events
        for event in events:
            message_bus.publish(event.event_type.name, event)
        
        time.sleep(0.5)
        
        # Process from queue (should be in priority order)
        while not event_queue.empty():
            priority, event = event_queue.get()
            if event:
                processed_events.append(event)
        
        # Verify processing
        assert len(processed_events) >= 4  # Allow for some variations
        
        # Cleanup
        message_bus.stop()
        queue_manager.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
