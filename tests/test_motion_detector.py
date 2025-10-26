#!/usr/bin/env python3
"""
Test suite for Motion Detection Worker

Comprehensive tests for the motion detection worker covering functionality,
performance, and error handling.
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Try to import OpenCV and numpy, use mocks if not available
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    
    # Create simple mocks for testing without OpenCV
    class MockBackgroundSubtractor:
        def apply(self, frame, learningRate=0.01):
            # Return a simple foreground mask
            return np.zeros((100, 100), dtype=np.uint8)
    
    class cv2:
        @staticmethod
        def createBackgroundSubtractorMOG2(*args, **kwargs):
            return MockBackgroundSubtractor()
        
        @staticmethod
        def createBackgroundSubtractorKNN(*args, **kwargs):
            return MockBackgroundSubtractor()
        
        @staticmethod
        def cvtColor(img, code):
            return img[:, :, 0] if len(img.shape) == 3 else img
        
        @staticmethod
        def GaussianBlur(img, kernel, sigma):
            return img
        
        @staticmethod
        def resize(img, size):
            return np.zeros((size[1], size[0], 3), dtype=np.uint8) if len(img.shape) == 3 else np.zeros((size[1], size[0]), dtype=np.uint8)
        
        @staticmethod
        def morphologyEx(img, op, kernel):
            return img
        
        @staticmethod
        def findContours(img, mode, method):
            # Return empty contours
            return [], None
        
        @staticmethod
        def contourArea(contour):
            return 100
        
        @staticmethod
        def boundingRect(contour):
            return (10, 10, 20, 20)
        
        @staticmethod
        def moments(img):
            return {"m00": 100, "m10": 5000, "m01": 5000}
        
        COLOR_BGR2GRAY = 6
        MORPH_CLOSE = 3
        MORPH_OPEN = 2
        RETR_EXTERNAL = 0
        CHAIN_APPROX_SIMPLE = 2
    
    class np:
        uint8 = 'uint8'
        
        @staticmethod
        def zeros(shape, dtype=None):
            class MockArray:
                def __init__(self, s):
                    self.shape = s if isinstance(s, tuple) else (s,)
                    self.size = 1
                    for dim in self.shape:
                        self.size *= dim
                    self.nbytes = self.size
                
                def __getitem__(self, key):
                    return MockArray((10, 10))
                
                def copy(self):
                    return MockArray(self.shape)
            
            return MockArray(shape)
        
        @staticmethod
        def ones(shape, dtype=None):
            return np.zeros(shape, dtype)
        
        @staticmethod
        def random_randint(*args, **kwargs):
            shape = args[3] if len(args) > 3 else (100, 100, 3)
            return np.zeros(shape)

from src.communication.message_bus import MessageBus, Message
from src.communication.events import FrameEvent, EventType, MotionResult, MotionHistory
from config.motion_config import MotionConfig

# Import motion detector with conditional import
if CV2_AVAILABLE:
    from src.pipeline.motion_detector import MotionDetector
else:
    # Mock the MotionDetector for environments without OpenCV
    class MotionDetector:
        def __init__(self, message_bus, config):
            self.motion_config = config
            self.message_bus = message_bus
            self.worker_id = "motion_detector_mock"
            self.running = False
        
        def start(self):
            self.running = True
        
        def stop(self):
            self.running = False
        
        def get_metrics(self):
            return {'worker_id': self.worker_id}


class TestMotionConfig(unittest.TestCase):
    """Test suite for motion configuration."""
    
    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = MotionConfig()
        
        self.assertTrue(config.enabled)
        self.assertEqual(config.motion_threshold, 25.0)
        self.assertEqual(config.min_contour_area, 500)
        self.assertEqual(config.bg_subtractor_type, "MOG2")
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            'enabled': True,
            'motion_threshold': 30.0,
            'min_contour_area': 600,
            'bg_subtractor_type': 'KNN'
        }
        
        config = MotionConfig.from_dict(config_dict)
        
        self.assertTrue(config.enabled)
        self.assertEqual(config.motion_threshold, 30.0)
        self.assertEqual(config.min_contour_area, 600)
        self.assertEqual(config.bg_subtractor_type, 'KNN')
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid motion threshold
        with self.assertRaises(ValueError):
            config = MotionConfig()
            config.motion_threshold = -10.0
            config._validate()
        
        # Test invalid contour area
        with self.assertRaises(ValueError):
            config = MotionConfig()
            config.min_contour_area = -100
            config._validate()
        
        # Test invalid blur kernel (must be odd)
        with self.assertRaises(ValueError):
            config = MotionConfig()
            config.gaussian_blur_kernel = (20, 20)
            config._validate()
    
    def test_config_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = MotionConfig()
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertIn('enabled', config_dict)
        self.assertIn('motion_threshold', config_dict)
        self.assertIn('bg_subtractor_type', config_dict)
    
    def test_platform_optimizations(self):
        """Test platform-specific optimizations."""
        config = MotionConfig()
        config.apply_platform_optimizations()
        
        # Configuration should be optimized based on platform
        self.assertIsNotNone(config.frame_resize_factor)
        self.assertGreater(config.frame_resize_factor, 0)
        self.assertLessEqual(config.frame_resize_factor, 1.0)


class TestMotionResult(unittest.TestCase):
    """Test suite for motion result data structure."""
    
    def test_motion_result_creation(self):
        """Test motion result creation."""
        result = MotionResult(
            motion_detected=True,
            motion_score=45.5,
            motion_regions=[(10, 10, 50, 50)],
            contour_count=1,
            largest_contour_area=2500,
            motion_center=(35, 35),
            frame_timestamp=time.time(),
            processing_time=0.05
        )
        
        self.assertTrue(result.motion_detected)
        self.assertEqual(result.motion_score, 45.5)
        self.assertEqual(len(result.motion_regions), 1)
        self.assertEqual(result.contour_count, 1)
    
    def test_motion_result_to_dict(self):
        """Test motion result serialization."""
        result = MotionResult(
            motion_detected=True,
            motion_score=45.5,
            motion_regions=[(10, 10, 50, 50)],
            contour_count=1,
            largest_contour_area=2500,
            motion_center=(35, 35),
            frame_timestamp=time.time(),
            processing_time=0.05
        )
        
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertTrue(result_dict['motion_detected'])
        self.assertEqual(result_dict['motion_score'], 45.5)
        self.assertIn('motion_regions', result_dict)


class TestMotionHistory(unittest.TestCase):
    """Test suite for motion history tracking."""
    
    def test_motion_history_creation(self):
        """Test motion history initialization."""
        history = MotionHistory()
        
        self.assertEqual(len(history.recent_scores), 0)
        self.assertEqual(len(history.motion_events), 0)
        self.assertEqual(history.trend_direction, "stable")
    
    def test_add_score(self):
        """Test adding scores to history."""
        history = MotionHistory()
        
        history.add_score(25.0, time.time(), True)
        history.add_score(30.0, time.time(), True)
        
        self.assertEqual(len(history.recent_scores), 2)
        self.assertEqual(len(history.motion_events), 2)
    
    def test_calculate_trend(self):
        """Test trend calculation."""
        history = MotionHistory()
        
        # Add increasing scores
        for i in range(10):
            history.add_score(float(i * 5), time.time(), i > 5)
        
        trend = history.calculate_trend()
        self.assertEqual(trend, "increasing")
    
    def test_trim_history(self):
        """Test history trimming."""
        history = MotionHistory()
        
        # Add many scores
        for i in range(20):
            history.add_score(float(i), time.time(), False)
        
        # Trim to max size
        history.trim_history(10)
        
        self.assertEqual(len(history.recent_scores), 10)


@unittest.skipIf(not CV2_AVAILABLE, "OpenCV not available")
class TestMotionDetector(unittest.TestCase):
    """Test suite for motion detector worker."""
    
    def setUp(self):
        """Set up test environment."""
        self.message_bus = MessageBus()
        self.message_bus.start()
        
        self.config = MotionConfig()
        self.config.enabled = True
        self.config.motion_threshold = 25.0
        self.config.skip_frame_count = 0
        
        # Create detector
        self.detector = MotionDetector(self.message_bus, self.config)
    
    def tearDown(self):
        """Cleanup after tests."""
        if self.detector.running:
            self.detector.stop()
        
        self.message_bus.stop()
        time.sleep(0.1)
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.motion_config.motion_threshold, 25.0)
        self.assertEqual(self.detector.frames_processed, 0)
    
    def test_background_subtractor_creation(self):
        """Test background subtractor creation."""
        # Test MOG2
        self.config.bg_subtractor_type = "MOG2"
        subtractor = self.detector._create_background_subtractor()
        self.assertIsNotNone(subtractor)
        
        # Test KNN
        self.config.bg_subtractor_type = "KNN"
        subtractor = self.detector._create_background_subtractor()
        self.assertIsNotNone(subtractor)
        
        # Test invalid type
        self.config.bg_subtractor_type = "INVALID"
        with self.assertRaises(ValueError):
            self.detector._create_background_subtractor()
    
    def test_motion_detection_with_static_frame(self):
        """Test motion detection on static frame."""
        # Create a simple static frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Initialize detector first
        self.detector._initialize_worker()
        
        # Detect motion
        result = self.detector.detect_motion(frame)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, MotionResult)
        self.assertFalse(result.motion_detected)
    
    def test_frame_preprocessing(self):
        """Test frame preprocessing."""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        processed = self.detector._preprocess_frame(frame)
        
        self.assertIsNotNone(processed)
        # Should be grayscale
        self.assertEqual(len(processed.shape), 2)
    
    def test_should_forward_frame_with_motion(self):
        """Test frame forwarding decision with motion."""
        # Create motion result with motion
        motion_result = MotionResult(
            motion_detected=True,
            motion_score=35.0,
            motion_regions=[(10, 10, 20, 20)],
            contour_count=1,
            largest_contour_area=500,
            motion_center=(20, 20),
            frame_timestamp=time.time(),
            processing_time=0.01
        )
        
        should_forward = self.detector._should_forward_frame(motion_result)
        self.assertTrue(should_forward)
    
    def test_should_forward_frame_no_motion(self):
        """Test frame forwarding decision without motion."""
        # Create motion result without motion
        motion_result = MotionResult(
            motion_detected=False,
            motion_score=5.0,
            motion_regions=[],
            contour_count=0,
            largest_contour_area=0,
            motion_center=None,
            frame_timestamp=time.time(),
            processing_time=0.01
        )
        
        should_forward = self.detector._should_forward_frame(motion_result)
        # Should not forward if no motion and within static duration
        self.assertFalse(should_forward)
    
    def test_should_forward_frame_max_static_duration(self):
        """Test frame forwarding after max static duration."""
        # Set last forwarded time to past
        self.detector.last_forwarded_time = time.time() - self.config.max_static_duration - 1
        
        # Create motion result without motion
        motion_result = MotionResult(
            motion_detected=False,
            motion_score=5.0,
            motion_regions=[],
            contour_count=0,
            largest_contour_area=0,
            motion_center=None,
            frame_timestamp=time.time(),
            processing_time=0.01
        )
        
        should_forward = self.detector._should_forward_frame(motion_result)
        # Should forward due to max static duration
        self.assertTrue(should_forward)
    
    def test_motion_history_update(self):
        """Test motion history updates."""
        motion_result = MotionResult(
            motion_detected=True,
            motion_score=35.0,
            motion_regions=[(10, 10, 20, 20)],
            contour_count=1,
            largest_contour_area=500,
            motion_center=(20, 20),
            frame_timestamp=time.time(),
            processing_time=0.01
        )
        
        self.detector._update_motion_history(motion_result)
        
        self.assertEqual(len(self.detector.motion_history.recent_scores), 1)
        self.assertEqual(self.detector.motion_history.recent_scores[0], 35.0)
    
    def test_get_metrics(self):
        """Test metrics retrieval."""
        metrics = self.detector.get_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('frames_processed', metrics)
        self.assertIn('frames_forwarded', metrics)
        self.assertIn('motion_events', metrics)
        self.assertIn('forward_ratio', metrics)
        self.assertIn('config', metrics)


class TestMotionDetectorIntegration(unittest.TestCase):
    """Integration tests for motion detector with message bus."""
    
    def setUp(self):
        """Set up test environment."""
        self.message_bus = MessageBus()
        self.message_bus.start()
        
        self.config = MotionConfig()
        self.received_events = []
        
        # Subscribe to motion analyzed events
        self.message_bus.subscribe(
            'motion_analyzed',
            self._capture_event,
            'test_subscriber'
        )
    
    def tearDown(self):
        """Cleanup after tests."""
        self.message_bus.stop()
        time.sleep(0.1)
    
    def _capture_event(self, message: Message):
        """Capture received events."""
        self.received_events.append(message)
    
    @unittest.skipIf(not CV2_AVAILABLE, "OpenCV not available")
    def test_frame_event_processing(self):
        """Test processing of frame events."""
        detector = MotionDetector(self.message_bus, self.config)
        detector._initialize_worker()
        
        # Create test frame event
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_event = FrameEvent(
            event_type=EventType.FRAME_CAPTURED,
            frame_data=frame,
            resolution=(100, 100)
        )
        
        # Publish frame event
        self.message_bus.publish('frame_captured', frame_event)
        
        # Wait for processing
        time.sleep(0.5)
        
        # Check if event was forwarded
        self.assertGreater(detector.frames_processed, 0)
    
    @unittest.skipIf(not CV2_AVAILABLE, "OpenCV not available")
    def test_error_handling(self):
        """Test error handling in motion detection."""
        detector = MotionDetector(self.message_bus, self.config)
        detector._initialize_worker()
        
        # Create invalid frame event (no frame data)
        frame_event = FrameEvent(
            event_type=EventType.FRAME_CAPTURED,
            frame_data=None,
            resolution=(100, 100)
        )
        
        # Publish frame event
        self.message_bus.publish('frame_captured', frame_event)
        
        # Wait for processing
        time.sleep(0.5)
        
        # Should handle error gracefully
        # Event should still be forwarded (fallback)
        self.assertGreater(len(self.received_events), 0)


if __name__ == '__main__':
    unittest.main()
