#!/usr/bin/env python3
"""
Test suite for Face Recognition Worker

Tests for the face recognition worker with database integration and caching.
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from src.pipeline.face_recognizer import FaceRecognitionWorker
from src.communication.message_bus import MessageBus, Message
from src.communication.events import (
    FaceDetectionEvent,
    FaceDetection,
    BoundingBox,
    EventType,
    RecognitionStatus
)
from src.storage.face_database import FaceDatabase
from src.recognition.recognition_result import PersonMatch


class TestFaceRecognitionWorker(unittest.TestCase):
    """Test suite for FaceRecognitionWorker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock message bus
        self.mock_message_bus = Mock(spec=MessageBus)
        self.mock_message_bus.subscribe = Mock()
        self.mock_message_bus.publish = Mock()
        
        # Create mock face database
        self.mock_database = Mock(spec=FaceDatabase)
        self.mock_database.is_initialized = Mock(return_value=True)
        self.mock_database.find_blacklist_matches = Mock(return_value=[])
        self.mock_database.find_known_matches = Mock(return_value=[])
        self.mock_database.get_frequent_faces = Mock(return_value=[])
        
        # Worker configuration
        self.worker_config = {
            'tolerance': 0.6,
            'blacklist_tolerance': 0.5,
            'min_confidence': 0.4,
            'batch_size': 5,
            'cache': {
                'enabled': True,
                'cache_size': 100,
                'ttl_seconds': 3600
            }
        }
        
        # Create worker
        self.worker = FaceRecognitionWorker(
            self.mock_message_bus,
            self.mock_database,
            self.worker_config
        )
    
    def test_worker_initialization(self):
        """Test worker initializes correctly."""
        self.assertEqual(self.worker.tolerance, 0.6)
        self.assertEqual(self.worker.blacklist_tolerance, 0.5)
        self.assertEqual(self.worker.batch_size, 5)
        self.assertIsNotNone(self.worker.face_encoder)
        self.assertIsNotNone(self.worker.similarity_matcher)
        self.assertIsNotNone(self.worker.recognition_cache)
        
        # Verify subscriptions were set up
        self.mock_message_bus.subscribe.assert_called()
    
    def test_worker_metrics_initialization(self):
        """Test worker metrics are initialized."""
        metrics = self.worker.get_metrics()
        
        self.assertEqual(metrics['recognition_count'], 0)
        self.assertEqual(metrics['recognition_errors'], 0)
        self.assertEqual(metrics['cache_hits'], 0)
        self.assertEqual(metrics['cache_misses'], 0)
        self.assertEqual(metrics['known_person_matches'], 0)
        self.assertEqual(metrics['blacklist_matches'], 0)
        self.assertEqual(metrics['unknown_faces'], 0)
    
    def test_handle_detection_event_unknown_face(self):
        """Test handling detection event with unknown face."""
        # Create mock face detection
        bbox = BoundingBox(x=10, y=10, width=100, height=100, confidence=0.9)
        face = FaceDetection(bounding_box=bbox, confidence=0.9)
        
        detection_event = FaceDetectionEvent(
            faces=[face],
            detection_time=0.1,
            detector_type="cpu"
        )
        
        message = Message(
            topic='faces_detected',
            data=detection_event,
            timestamp=time.time()
        )
        
        # Database returns no matches
        self.mock_database.find_blacklist_matches.return_value = []
        self.mock_database.find_known_matches.return_value = []
        
        # Handle event
        self.worker.handle_detection_event(message)
        
        # Verify recognition event was published
        self.mock_message_bus.publish.assert_called()
        call_args = self.mock_message_bus.publish.call_args
        self.assertEqual(call_args[0][0], 'faces_recognized')
        
        recognition_event = call_args[0][1]
        self.assertEqual(len(recognition_event.recognitions), 1)
        self.assertEqual(recognition_event.recognitions[0].status, RecognitionStatus.UNKNOWN)
    
    def test_handle_detection_event_known_face(self):
        """Test handling detection event with known face."""
        # Create mock face detection
        bbox = BoundingBox(x=10, y=10, width=100, height=100, confidence=0.9)
        face = FaceDetection(bounding_box=bbox, confidence=0.9)
        
        detection_event = FaceDetectionEvent(
            faces=[face],
            detection_time=0.1,
            detector_type="cpu"
        )
        
        message = Message(
            topic='faces_detected',
            data=detection_event,
            timestamp=time.time()
        )
        
        # Database returns known match
        known_match = PersonMatch(
            person_id="person_001",
            person_name="John Doe",
            confidence=0.85,
            similarity_score=0.90
        )
        self.mock_database.find_blacklist_matches.return_value = []
        self.mock_database.find_known_matches.return_value = [known_match]
        
        # Handle event
        self.worker.handle_detection_event(message)
        
        # Verify recognition event was published
        self.mock_message_bus.publish.assert_called()
        call_args = self.mock_message_bus.publish.call_args
        recognition_event = call_args[0][1]
        
        self.assertEqual(len(recognition_event.recognitions), 1)
        self.assertEqual(recognition_event.recognitions[0].status, RecognitionStatus.KNOWN)
        self.assertEqual(recognition_event.recognitions[0].identity, "John Doe")
    
    def test_handle_detection_event_blacklisted_face(self):
        """Test handling detection event with blacklisted face."""
        # Create mock face detection
        bbox = BoundingBox(x=10, y=10, width=100, height=100, confidence=0.9)
        face = FaceDetection(bounding_box=bbox, confidence=0.9)
        
        detection_event = FaceDetectionEvent(
            faces=[face],
            detection_time=0.1,
            detector_type="cpu"
        )
        
        message = Message(
            topic='faces_detected',
            data=detection_event,
            timestamp=time.time()
        )
        
        # Database returns blacklist match
        blacklist_match = PersonMatch(
            person_id="blacklist_001",
            person_name="Suspicious Person",
            confidence=0.95,
            is_blacklisted=True
        )
        self.mock_database.find_blacklist_matches.return_value = [blacklist_match]
        
        # Handle event
        self.worker.handle_detection_event(message)
        
        # Verify recognition event was published
        call_args = self.mock_message_bus.publish.call_args
        recognition_event = call_args[0][1]
        
        self.assertEqual(recognition_event.recognitions[0].status, RecognitionStatus.BLACKLISTED)
        self.assertEqual(recognition_event.recognitions[0].identity, "Suspicious Person")
    
    def test_handle_detection_event_multiple_faces(self):
        """Test handling detection event with multiple faces."""
        # Create multiple mock face detections
        faces = []
        for i in range(3):
            bbox = BoundingBox(x=10+i*50, y=10, width=100, height=100, confidence=0.9)
            face = FaceDetection(bounding_box=bbox, confidence=0.9)
            faces.append(face)
        
        detection_event = FaceDetectionEvent(
            faces=faces,
            detection_time=0.1,
            detector_type="cpu"
        )
        
        message = Message(
            topic='faces_detected',
            data=detection_event,
            timestamp=time.time()
        )
        
        # Database returns no matches
        self.mock_database.find_blacklist_matches.return_value = []
        self.mock_database.find_known_matches.return_value = []
        
        # Handle event
        self.worker.handle_detection_event(message)
        
        # Verify all faces were processed
        call_args = self.mock_message_bus.publish.call_args
        recognition_event = call_args[0][1]
        self.assertEqual(len(recognition_event.recognitions), 3)
    
    def test_handle_detection_event_with_error(self):
        """Test handling detection event with error."""
        # Create mock face detection
        bbox = BoundingBox(x=10, y=10, width=100, height=100, confidence=0.9)
        face = FaceDetection(bounding_box=bbox, confidence=0.9)
        
        detection_event = FaceDetectionEvent(
            faces=[face],
            detection_time=0.1,
            detector_type="cpu"
        )
        
        message = Message(
            topic='faces_detected',
            data=detection_event,
            timestamp=time.time()
        )
        
        # Make database raise exception
        self.mock_database.find_blacklist_matches.side_effect = Exception("Database error")
        
        # Handle event (should not raise exception)
        try:
            self.worker.handle_detection_event(message)
        except Exception as e:
            self.fail(f"Worker should handle errors gracefully: {e}")
        
        # Error count should be incremented
        self.assertEqual(self.worker.recognition_errors, 1)
    
    def test_cache_functionality(self):
        """Test recognition cache hit/miss behavior."""
        bbox = BoundingBox(x=10, y=10, width=100, height=100, confidence=0.9)
        face = FaceDetection(bounding_box=bbox, confidence=0.9)
        
        detection_event = FaceDetectionEvent(
            faces=[face],
            detection_time=0.1,
            detector_type="cpu"
        )
        
        message = Message(
            topic='faces_detected',
            data=detection_event,
            timestamp=time.time()
        )
        
        # First call - should miss cache
        self.mock_database.find_blacklist_matches.return_value = []
        self.mock_database.find_known_matches.return_value = []
        
        self.worker.handle_detection_event(message)
        cache_miss_count = self.worker.cache_misses
        
        # Second call with same face - should hit cache
        self.worker.handle_detection_event(message)
        
        # Cache hits should increase
        self.assertGreater(self.worker.cache_hits, 0)
    
    def test_handle_database_update(self):
        """Test handling database update events."""
        update_data = {'person_id': 'person_001'}
        message = Message(
            topic='face_database_updated',
            data=update_data,
            timestamp=time.time()
        )
        
        # Should not raise exception
        try:
            self.worker.handle_database_update(message)
        except Exception as e:
            self.fail(f"Database update handling failed: {e}")
    
    def test_handle_cache_clear(self):
        """Test handling cache clear requests."""
        clear_data = {'clear_all': True}
        message = Message(
            topic='recognition_cache_clear',
            data=clear_data,
            timestamp=time.time()
        )
        
        # Should not raise exception
        try:
            self.worker.handle_cache_clear(message)
        except Exception as e:
            self.fail(f"Cache clear handling failed: {e}")
    
    def test_worker_metrics_update(self):
        """Test worker metrics are updated correctly."""
        bbox = BoundingBox(x=10, y=10, width=100, height=100, confidence=0.9)
        face = FaceDetection(bounding_box=bbox, confidence=0.9)
        
        detection_event = FaceDetectionEvent(
            faces=[face],
            detection_time=0.1,
            detector_type="cpu"
        )
        
        message = Message(
            topic='faces_detected',
            data=detection_event,
            timestamp=time.time()
        )
        
        # Process event with known match
        known_match = PersonMatch(person_id="person_001", confidence=0.85)
        self.mock_database.find_blacklist_matches.return_value = []
        self.mock_database.find_known_matches.return_value = [known_match]
        
        self.worker.handle_detection_event(message)
        
        metrics = self.worker.get_metrics()
        
        self.assertEqual(metrics['recognition_count'], 1)
        self.assertEqual(metrics['known_person_matches'], 1)
        self.assertGreater(metrics['avg_recognition_time'], 0)


class TestFaceRecognitionWorkerIntegration(unittest.TestCase):
    """Integration tests for face recognition worker."""
    
    def test_worker_with_real_cache(self):
        """Test worker with real cache instance."""
        mock_message_bus = Mock(spec=MessageBus)
        mock_message_bus.subscribe = Mock()
        mock_message_bus.publish = Mock()
        
        mock_database = Mock(spec=FaceDatabase)
        mock_database.is_initialized = Mock(return_value=True)
        mock_database.find_blacklist_matches = Mock(return_value=[])
        mock_database.find_known_matches = Mock(return_value=[])
        mock_database.get_frequent_faces = Mock(return_value=[])
        
        config = {
            'tolerance': 0.6,
            'cache': {
                'enabled': True,
                'cache_size': 10,
                'ttl_seconds': 60
            }
        }
        
        worker = FaceRecognitionWorker(mock_message_bus, mock_database, config)
        
        # Verify cache is properly initialized
        self.assertIsNotNone(worker.recognition_cache)
        self.assertTrue(worker.recognition_cache.enabled)


if __name__ == '__main__':
    unittest.main()
