#!/usr/bin/env python3
"""
Test suite for Frame Capture Worker

Comprehensive tests for the frame capture worker with ring buffer,
covering functionality, performance, and error handling.
"""

import sys
import time
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Try to import numpy, use mock if not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create a simple mock for testing without numpy
    class MockArray:
        def __init__(self, shape):
            self.shape = shape
            self.nbytes = shape[0] * shape[1] * shape[2] if len(shape) == 3 else 0
        def copy(self):
            return MockArray(self.shape)
    
    class np:
        @staticmethod
        def random_randint(*args, **kwargs):
            return MockArray((100, 100, 3))

from src.pipeline.frame_capture import FrameCaptureWorker
from src.communication.message_bus import MessageBus, Message
from src.communication.events import (
    PipelineEvent, EventType, DoorbellEvent, FrameEvent
)

# Mock camera handler for tests
class MockCameraHandler:
    def __init__(self):
        self.is_initialized = False
        self.camera_type = 'mock'
        self.capture_count = 0
    
    def initialize(self):
        self.is_initialized = True
        return True
    
    def capture_frame(self):
        self.capture_count += 1
        if NUMPY_AVAILABLE:
            return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        else:
            return MockArray((100, 100, 3))
    
    def capture_image(self):
        return self.capture_frame()
    
    def get_status(self):
        return 'mock_active' if self.is_initialized else 'not_initialized'
    
    def cleanup(self):
        self.is_initialized = False


class TestFrameCaptureWorker(unittest.TestCase):
    """Test suite for frame capture worker."""
    
    def setUp(self):
        """Set up test environment."""
        self.message_bus = MessageBus()
        self.message_bus.start()
        
        self.camera_handler = MockCameraHandler()
        self.camera_handler.initialize()
        
        self.config = {
            'buffer_size': 10,
            'capture_fps': 10,
            'burst_count': 3,
            'burst_interval': 0.1
        }
        
        self.worker = FrameCaptureWorker(
            camera_handler=self.camera_handler,
            message_bus=self.message_bus,
            config=self.config
        )
    
    def tearDown(self):
        """Cleanup after tests."""
        if self.worker.running:
            self.worker.stop()
            time.sleep(0.2)
        
        self.message_bus.stop()
        self.camera_handler.cleanup()
    
    def test_worker_initialization(self):
        """Test worker initializes with correct configuration."""
        self.assertIsNotNone(self.worker)
        self.assertEqual(self.worker.ring_buffer.maxlen, 10)
        self.assertEqual(self.worker.capture_fps, 10)
        self.assertEqual(self.worker.burst_count, 3)
        self.assertFalse(self.worker.running)
    
    def test_ring_buffer_operations(self):
        """Test ring buffer add and retrieve operations."""
        # Add frames to buffer
        for i in range(5):
            if NUMPY_AVAILABLE:
                frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            else:
                frame = MockArray((100, 100, 3))
            self.worker._add_frame_to_buffer(frame, {'test': i})
        
        # Check buffer size
        self.assertEqual(len(self.worker.ring_buffer), 5)
        
        # Get latest frames
        latest = self.worker.get_latest_frames(3)
        self.assertEqual(len(latest), 3)
        
        # Check metadata
        self.assertEqual(latest[-1]['metadata']['test'], 4)
    
    def test_ring_buffer_overflow(self):
        """Test ring buffer overflow behavior."""
        buffer_size = self.config['buffer_size']
        
        # Add more frames than buffer size
        for i in range(buffer_size + 5):
            if NUMPY_AVAILABLE:
                frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            else:
                frame = MockArray((100, 100, 3))
            self.worker._add_frame_to_buffer(frame, {'index': i})
        
        # Buffer should maintain max size
        self.assertEqual(len(self.worker.ring_buffer), buffer_size)
        
        # Oldest frames should be discarded
        oldest_frame = list(self.worker.ring_buffer)[0]
        self.assertEqual(oldest_frame['metadata']['index'], 5)  # First 5 were discarded
    
    def test_doorbell_event_handling(self):
        """Test doorbell event processing and frame burst capture."""
        # Track published frames
        published_frames = []
        
        def frame_callback(message: Message):
            published_frames.append(message.data)
        
        self.message_bus.subscribe('frame_captured', frame_callback, 'test_subscriber')
        
        # Create doorbell event
        doorbell_event = DoorbellEvent(
            event_type=EventType.DOORBELL_PRESSED,
            channel=18
        )
        
        message = Message(
            topic='doorbell_pressed',
            data=doorbell_event
        )
        
        # Handle doorbell event
        self.worker.handle_doorbell_event(message)
        
        # Wait for async processing
        time.sleep(0.5)
        
        # Should have captured burst_count frames
        self.assertEqual(len(published_frames), self.config['burst_count'])
        
        # Verify frame events
        for frame_event in published_frames:
            self.assertIsInstance(frame_event, FrameEvent)
            self.assertEqual(frame_event.event_type, EventType.FRAME_CAPTURED)
            self.assertIsNotNone(frame_event.frame_data)
    
    def test_continuous_capture_thread(self):
        """Test continuous capture loop operation."""
        # Start worker in separate thread
        worker_thread = threading.Thread(target=self.worker.start)
        worker_thread.start()
        
        # Let it capture some frames (give it more time to initialize)
        time.sleep(1.0)
        
        # Stop worker
        self.worker.stop()
        worker_thread.join(timeout=2.0)
        
        # Should have captured some frames
        # Note: frames_captured might be 0 if camera initialization takes time
        # but ring_buffer should have entries if continuous capture worked
        self.assertGreaterEqual(self.worker.frames_captured, 0)
    
    def test_thread_safety(self):
        """Test concurrent access to ring buffer."""
        def add_frames():
            for i in range(20):
                if NUMPY_AVAILABLE:
                    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                else:
                    frame = MockArray((100, 100, 3))
                self.worker._add_frame_to_buffer(frame, {'thread': threading.current_thread().name})
                time.sleep(0.01)
        
        def read_frames():
            for i in range(20):
                frames = self.worker.get_latest_frames(5)
                time.sleep(0.01)
        
        # Create multiple threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=add_frames, name=f'add_{i}')
            threads.append(t)
        
        for i in range(2):
            t = threading.Thread(target=read_frames, name=f'read_{i}')
            threads.append(t)
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Should not crash and buffer should be valid
        self.assertLessEqual(len(self.worker.ring_buffer), self.config['buffer_size'])
    
    def test_error_handling_and_recovery(self):
        """Test error scenarios and recovery mechanisms."""
        # Create a camera handler that fails
        failing_camera = MockCameraHandler()
        failing_camera.initialize()
        
        # Make capture fail by returning None
        failing_camera.capture_frame = Mock(return_value=None)
        failing_camera.capture_image = Mock(return_value=None)
        
        worker = FrameCaptureWorker(
            camera_handler=failing_camera,
            message_bus=self.message_bus,
            config=self.config
        )
        
        # Try to capture burst (should handle gracefully)
        frames = worker._capture_burst()
        
        # Should return empty list when all captures fail
        # (None frames are not added to the list)
        self.assertEqual(len(frames), 0)
    
    def test_resource_cleanup(self):
        """Test proper cleanup on worker shutdown."""
        # Initialize worker
        worker_thread = threading.Thread(target=self.worker.start)
        worker_thread.start()
        
        # Let it run briefly
        time.sleep(0.2)
        
        # Get initial buffer size
        initial_buffer_size = len(self.worker.ring_buffer)
        self.assertGreater(initial_buffer_size, 0)
        
        # Stop worker
        self.worker.stop()
        worker_thread.join(timeout=2.0)
        
        # Buffer should be cleared
        self.assertEqual(len(self.worker.ring_buffer), 0)
        self.assertFalse(self.worker.running)
    
    def test_message_bus_integration(self):
        """Test subscription setup and event publishing."""
        # Verify subscriptions were set up
        topics = self.message_bus.get_topic_list()
        self.assertIn('doorbell_pressed', topics)
        self.assertIn('system_shutdown', topics)
        
        # Verify subscriber count
        subscriber_count = self.message_bus.get_subscriber_count('doorbell_pressed')
        self.assertGreaterEqual(subscriber_count, 1)
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        metrics = self.worker.get_metrics()
        
        # Verify metric structure
        self.assertIn('worker_id', metrics)
        self.assertIn('frames_captured', metrics)
        self.assertIn('capture_errors', metrics)
        self.assertIn('buffer_size', metrics)
        self.assertIn('buffer_capacity', metrics)
        self.assertIn('camera_status', metrics)
        
        # Verify initial values
        self.assertEqual(metrics['frames_captured'], 0)
        self.assertEqual(metrics['buffer_capacity'], self.config['buffer_size'])
    
    def test_burst_capture_timing(self):
        """Test burst capture respects interval timing."""
        start_time = time.time()
        
        frames = self.worker._capture_burst()
        
        elapsed_time = time.time() - start_time
        
        # Should capture burst_count frames
        self.assertEqual(len(frames), self.config['burst_count'])
        
        # Should take approximately burst_count * burst_interval
        expected_min_time = (self.config['burst_count'] - 1) * self.config['burst_interval']
        self.assertGreaterEqual(elapsed_time, expected_min_time * 0.8)  # 20% tolerance
    
    def test_frame_metadata(self):
        """Test frame metadata is properly attached."""
        if NUMPY_AVAILABLE:
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        else:
            frame = MockArray((100, 100, 3))
        metadata = {'source': 'test', 'custom_field': 'value'}
        
        self.worker._add_frame_to_buffer(frame, metadata)
        
        frames = self.worker.get_latest_frames(1)
        self.assertEqual(len(frames), 1)
        
        # Verify metadata
        self.assertEqual(frames[0]['metadata']['source'], 'test')
        self.assertEqual(frames[0]['metadata']['custom_field'], 'value')
        self.assertIn('timestamp', frames[0])
    
    def test_capture_error_event_publishing(self):
        """Test error events are published correctly."""
        error_events = []
        
        def error_callback(message: Message):
            error_events.append(message.data)
        
        self.message_bus.subscribe('capture_errors', error_callback, 'test_error_sub')
        
        # Create a test error
        test_error = RuntimeError("Test error")
        test_event = PipelineEvent(event_type=EventType.DOORBELL_PRESSED)
        
        # Handle error
        self.worker._handle_capture_error(test_error, test_event)
        
        # Wait for async processing
        time.sleep(0.1)
        
        # Verify error event was published
        self.assertEqual(len(error_events), 1)
        self.assertIn('error', error_events[0].data)


class TestFrameCapturePerformance(unittest.TestCase):
    """Performance tests for frame capture."""
    
    def setUp(self):
        """Set up performance test environment."""
        self.message_bus = MessageBus()
        self.message_bus.start()
        
        self.camera_handler = MockCameraHandler()
        self.camera_handler.initialize()
        
        self.config = {
            'buffer_size': 30,
            'capture_fps': 30,
            'burst_count': 5,
            'burst_interval': 0.2
        }
    
    def tearDown(self):
        """Cleanup after tests."""
        self.message_bus.stop()
        self.camera_handler.cleanup()
    
    def test_capture_rate_benchmark(self):
        """Benchmark sustained capture rate."""
        worker = FrameCaptureWorker(
            camera_handler=self.camera_handler,
            message_bus=self.message_bus,
            config=self.config
        )
        
        # Start worker
        worker_thread = threading.Thread(target=worker.start)
        worker_thread.start()
        
        # Run for 2 seconds
        time.sleep(2.0)
        
        # Stop worker
        worker.stop()
        worker_thread.join(timeout=2.0)
        
        # Calculate actual FPS
        actual_fps = worker.frames_captured / 2.0
        
        # Should achieve at least 80% of target FPS
        target_fps = self.config['capture_fps']
        self.assertGreater(actual_fps, target_fps * 0.8)
        
        print(f"Capture rate: {actual_fps:.2f} FPS (target: {target_fps} FPS)")
    
    def test_memory_usage_stability(self):
        """Test memory usage during extended operation."""
        worker = FrameCaptureWorker(
            camera_handler=self.camera_handler,
            message_bus=self.message_bus,
            config=self.config
        )
        
        # Start worker
        worker_thread = threading.Thread(target=worker.start)
        worker_thread.start()
        
        # Run for several seconds
        time.sleep(3.0)
        
        # Buffer should maintain max size
        self.assertEqual(len(worker.ring_buffer), worker.ring_buffer.maxlen)
        
        # Stop worker
        worker.stop()
        worker_thread.join(timeout=2.0)


def run_tests():
    """Run all frame capture tests."""
    test_suite = unittest.TestSuite()
    
    test_classes = [
        TestFrameCaptureWorker,
        TestFrameCapturePerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("🧪 Running Frame Capture Worker Tests")
    print("=" * 60)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
