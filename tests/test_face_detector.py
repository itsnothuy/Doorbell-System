#!/usr/bin/env python3
"""
Test suite for Face Detection Worker Pool

Tests for multi-process face detection worker implementation.
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class np:
        uint8 = int
        
        @staticmethod
        def zeros(shape, dtype=None):
            class MockArray:
                def __init__(self, shape):
                    self.shape = shape
                    self.nbytes = shape[0] * shape[1] * (shape[2] if len(shape) > 2 else 1)
            return MockArray(shape)

from src.pipeline.face_detector import FaceDetectionWorker, DetectionJob
from src.communication.message_bus import MessageBus, Message
from src.communication.events import FrameEvent, EventType
from src.detectors.detection_result import DetectionResult, DetectionMetrics


class TestDetectionJob(unittest.TestCase):
    """Test suite for DetectionJob class."""
    
    def test_job_creation(self):
        """Test detection job creation."""
        # Create mock frame event
        frame_event = FrameEvent(
            event_type=EventType.FRAME_CAPTURED,
            frame_data=None,
            event_id='test_frame_1'
        )
        
        timestamp = time.time()
        job = DetectionJob(priority=1, frame_event=frame_event, timestamp=timestamp)
        
        self.assertEqual(job.priority, 1)
        self.assertEqual(job.frame_event, frame_event)
        self.assertEqual(job.timestamp, timestamp)
        self.assertIn('test_frame_1', job.job_id)
    
    def test_job_priority_comparison(self):
        """Test job priority comparison for queue ordering."""
        frame_event = FrameEvent(
            event_type=EventType.FRAME_CAPTURED,
            frame_data=None
        )
        
        job1 = DetectionJob(priority=1, frame_event=frame_event, timestamp=time.time())
        job2 = DetectionJob(priority=2, frame_event=frame_event, timestamp=time.time())
        
        # Lower priority value should come first
        self.assertTrue(job1 < job2)
        self.assertFalse(job2 < job1)


class TestFaceDetectionWorker(unittest.TestCase):
    """Test suite for face detection worker."""
    
    def setUp(self):
        """Set up test environment."""
        self.message_bus = MessageBus()
        self.message_bus.start()
        
        self.config = {
            'worker_count': 2,
            'detector_type': 'mock',
            'max_queue_size': 10,
            'job_timeout': 30.0,
            'model': 'hog'
        }
    
    def tearDown(self):
        """Cleanup after tests."""
        if hasattr(self, 'worker') and self.worker.running:
            self.worker.stop()
        
        if self.message_bus.running:
            self.message_bus.stop()
    
    def test_worker_initialization(self):
        """Test worker pool initialization."""
        worker = FaceDetectionWorker(self.message_bus, self.config)
        
        # Check configuration
        self.assertEqual(worker.worker_count, 2)
        self.assertEqual(worker.detector_type, 'mock')
        self.assertEqual(worker.max_queue_size, 10)
        self.assertEqual(worker.job_timeout, 30.0)
        
        # Check initial metrics
        self.assertEqual(worker.detection_count, 0)
        self.assertEqual(worker.detection_errors, 0)
        self.assertEqual(len(worker.pending_jobs), 0)
    
    def test_subscriptions_setup(self):
        """Test that subscriptions are set up correctly."""
        worker = FaceDetectionWorker(self.message_bus, self.config)
        
        # Verify subscriptions exist
        self.assertIn('frame_captured', self.message_bus.subscriptions)
        self.assertIn('worker_health_check', self.message_bus.subscriptions)
        self.assertIn('system_shutdown', self.message_bus.subscriptions)
    
    @patch('src.pipeline.face_detector.DetectorFactory')
    def test_initialize_worker_validates_detector(self, mock_factory):
        """Test that worker initialization validates detector availability."""
        mock_detector_class = Mock()
        mock_detector_class.is_available.return_value = True
        mock_factory.get_detector_class.return_value = mock_detector_class
        
        mock_detector = Mock()
        mock_detector.health_check.return_value = {'status': 'healthy'}
        mock_factory.create.return_value = mock_detector
        
        worker = FaceDetectionWorker(self.message_bus, self.config)
        worker._initialize_worker()
        
        # Should have validated detector
        mock_detector_class.is_available.assert_called_once()
        mock_detector.health_check.assert_called_once()
    
    @patch('src.pipeline.face_detector.DetectorFactory')
    def test_fallback_to_cpu_when_detector_unavailable(self, mock_factory):
        """Test fallback to CPU when requested detector is unavailable."""
        mock_detector_class = Mock()
        mock_detector_class.is_available.return_value = False
        mock_factory.get_detector_class.return_value = mock_detector_class
        
        config = self.config.copy()
        config['detector_type'] = 'gpu'
        
        worker = FaceDetectionWorker(self.message_bus, config)
        
        # Should fall back to CPU
        self.assertEqual(worker.detector_type, 'cpu')
    
    def test_handle_frame_event_queues_job(self):
        """Test that frame events are queued for processing."""
        worker = FaceDetectionWorker(self.message_bus, self.config)
        
        # Create frame event
        if NUMPY_AVAILABLE:
            frame_data = np.zeros((100, 100, 3), dtype=np.uint8)
        else:
            frame_data = np.zeros((100, 100, 3))
        
        frame_event = FrameEvent(
            event_type=EventType.FRAME_CAPTURED,
            frame_data=frame_data,
            event_id='test_frame'
        )
        
        message = Message(topic='frame_captured', data=frame_event)
        
        # Mock _process_job_queue to prevent actual processing
        with patch.object(worker, '_process_job_queue'):
            worker.handle_frame_event(message)
            
            # Job should be in queue
            self.assertFalse(worker.job_queue.empty())
    
    def test_doorbell_events_get_higher_priority(self):
        """Test that doorbell events get priority 1."""
        worker = FaceDetectionWorker(self.message_bus, self.config)
        
        # Create doorbell frame event
        if NUMPY_AVAILABLE:
            frame_data = np.zeros((100, 100, 3), dtype=np.uint8)
        else:
            frame_data = np.zeros((100, 100, 3))
        
        frame_event = FrameEvent(
            event_type=EventType.FRAME_CAPTURED,
            frame_data=frame_data
        )
        frame_event.data['source'] = 'doorbell'
        
        message = Message(topic='frame_captured', data=frame_event)
        
        with patch.object(worker, '_process_job_queue'):
            worker.handle_frame_event(message)
            
            # Get job from queue
            job = worker.job_queue.get(block=False)
            self.assertEqual(job.priority, 1)
    
    def test_queue_full_drops_frame(self):
        """Test that frames are dropped when queue is full."""
        config = self.config.copy()
        config['max_queue_size'] = 2
        worker = FaceDetectionWorker(self.message_bus, config)
        
        # Fill queue
        for i in range(3):
            if NUMPY_AVAILABLE:
                frame_data = np.zeros((100, 100, 3), dtype=np.uint8)
            else:
                frame_data = np.zeros((100, 100, 3))
            
            frame_event = FrameEvent(
                event_type=EventType.FRAME_CAPTURED,
                frame_data=frame_data,
                event_id=f'frame_{i}'
            )
            
            message = Message(topic='frame_captured', data=frame_event)
            
            with patch.object(worker, '_process_job_queue'):
                worker.handle_frame_event(message)
        
        # Queue should be at capacity
        self.assertEqual(worker.job_queue.qsize(), 2)
    
    def test_expired_jobs_are_dropped(self):
        """Test that expired jobs are dropped from queue."""
        worker = FaceDetectionWorker(self.message_bus, self.config)
        worker.job_timeout = 0.1  # Very short timeout
        
        # Create old job
        if NUMPY_AVAILABLE:
            frame_data = np.zeros((100, 100, 3), dtype=np.uint8)
        else:
            frame_data = np.zeros((100, 100, 3))
        
        frame_event = FrameEvent(
            event_type=EventType.FRAME_CAPTURED,
            frame_data=frame_data
        )
        
        old_job = DetectionJob(
            priority=1,
            frame_event=frame_event,
            timestamp=time.time() - 1.0  # 1 second ago
        )
        
        worker.job_queue.put(old_job)
        
        # Mock worker pool
        worker.worker_pool = Mock()
        
        # Process queue - old job should be dropped
        worker._process_job_queue()
        
        # Worker pool should not have been called
        worker.worker_pool.submit.assert_not_called()
    
    def test_handle_health_check(self):
        """Test health check response."""
        worker = FaceDetectionWorker(self.message_bus, self.config)
        worker.start_time = time.time()
        worker.detection_count = 10
        worker.detection_errors = 1
        
        # Create mock message
        message = Message(topic='worker_health_check', data={})
        
        # Capture published events
        published_events = []
        original_publish = self.message_bus.publish
        
        def capture_publish(topic, data, **kwargs):
            published_events.append((topic, data))
            return original_publish(topic, data, **kwargs)
        
        with patch.object(self.message_bus, 'publish', side_effect=capture_publish):
            worker.handle_health_check(message)
        
        # Should have published health response
        self.assertTrue(any(topic == 'worker_health_responses' for topic, _ in published_events))
    
    def test_get_metrics(self):
        """Test metrics collection."""
        worker = FaceDetectionWorker(self.message_bus, self.config)
        worker.start_time = time.time() - 10.0  # 10 seconds ago
        worker.detection_count = 5
        worker.detection_errors = 1
        worker.total_detection_time = 2.5
        
        metrics = worker.get_metrics()
        
        # Check metrics structure
        self.assertIn('detection_count', metrics)
        self.assertIn('detection_errors', metrics)
        self.assertIn('avg_detection_time', metrics)
        self.assertIn('detection_rate', metrics)
        self.assertIn('error_rate', metrics)
        self.assertIn('queue_size', metrics)
        self.assertIn('pending_jobs', metrics)
        self.assertIn('worker_count', metrics)
        self.assertIn('detector_type', metrics)
        
        # Check values
        self.assertEqual(metrics['detection_count'], 5)
        self.assertEqual(metrics['detection_errors'], 1)
        self.assertEqual(metrics['worker_count'], 2)
        self.assertEqual(metrics['detector_type'], 'mock')
    
    def test_cleanup_worker(self):
        """Test worker cleanup."""
        worker = FaceDetectionWorker(self.message_bus, self.config)
        
        # Mock worker pool
        mock_pool = Mock()
        worker.worker_pool = mock_pool
        
        # Add some pending jobs
        worker.pending_jobs['job1'] = {
            'future': Mock(),
            'job': Mock(),
            'submit_time': time.time()
        }
        
        # Add jobs to queue
        if NUMPY_AVAILABLE:
            frame_data = np.zeros((100, 100, 3), dtype=np.uint8)
        else:
            frame_data = np.zeros((100, 100, 3))
        
        frame_event = FrameEvent(
            event_type=EventType.FRAME_CAPTURED,
            frame_data=frame_data
        )
        job = DetectionJob(priority=1, frame_event=frame_event, timestamp=time.time())
        worker.job_queue.put(job)
        
        # Cleanup
        worker._cleanup_worker()
        
        # Worker pool should be shut down
        mock_pool.shutdown.assert_called_once()
        
        # Queue should be empty
        self.assertTrue(worker.job_queue.empty())


if __name__ == '__main__':
    unittest.main()
