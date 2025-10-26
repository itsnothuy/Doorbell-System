#!/usr/bin/env python3
"""
Test suite for Alert Manager

Comprehensive tests for alert classification and management.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.enrichment.alert_manager import (
    AlertManager, Alert, AlertPriority, AlertType
)
from src.communication.events import (
    FaceRecognitionEvent, FaceRecognition, FaceDetection,
    BoundingBox, RecognitionStatus, EventType
)


class TestAlert(unittest.TestCase):
    """Test suite for Alert class."""
    
    def test_alert_creation(self):
        """Test alert creation with required fields."""
        alert = Alert(
            alert_type=AlertType.KNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.INFO
        )
        
        self.assertEqual(alert.alert_type, AlertType.KNOWN_PERSON_DETECTED.value)
        self.assertEqual(alert.priority, AlertPriority.INFO)
        self.assertIsNotNone(alert.alert_id)
        self.assertIsNotNone(alert.created_at)
    
    def test_alert_with_person_info(self):
        """Test alert with person identification."""
        alert = Alert(
            alert_type=AlertType.KNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.INFO,
            person_id='person_123',
            person_name='John Doe',
            confidence=0.95
        )
        
        self.assertEqual(alert.person_id, 'person_123')
        self.assertEqual(alert.person_name, 'John Doe')
        self.assertEqual(alert.confidence, 0.95)
    
    def test_system_alert_fields(self):
        """Test system alert specific fields."""
        alert = Alert(
            alert_type=AlertType.SYSTEM_ALERT.value,
            priority=AlertPriority.WARNING,
            system_message='Test system message',
            component='test_component',
            severity='warning'
        )
        
        self.assertEqual(alert.system_message, 'Test system message')
        self.assertEqual(alert.component, 'test_component')
        self.assertEqual(alert.severity, 'warning')


class TestAlertManager(unittest.TestCase):
    """Test suite for AlertManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'blacklist_detection': {'enabled': True},
            'known_person_detection': {'enabled': True},
            'unknown_person_detection': {'enabled': True},
            'system_alerts': {'enabled': True}
        }
        self.alert_manager = AlertManager(self.config)
    
    def test_alert_manager_initialization(self):
        """Test alert manager initialization."""
        self.assertTrue(self.alert_manager.blacklist_enabled)
        self.assertTrue(self.alert_manager.known_person_enabled)
        self.assertTrue(self.alert_manager.unknown_person_enabled)
        self.assertEqual(self.alert_manager.alerts_created, 0)
    
    def test_classify_blacklist_detection(self):
        """Test classification of blacklist detection."""
        # Create mock recognition event with blacklisted person
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        face_detection = FaceDetection(bounding_box=bbox)
        
        recognition = FaceRecognition(
            face_detection=face_detection,
            status=RecognitionStatus.BLACKLISTED,
            identity='blacklisted_person',
            similarity_score=0.85
        )
        
        recognition_event = FaceRecognitionEvent(
            event_type=EventType.FACE_BLACKLISTED,
            recognitions=[recognition],
            event_id='test_event_1'
        )
        
        # Classify the event
        alerts = self.alert_manager.classify_recognition_event(recognition_event)
        
        # Verify alert was created
        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        
        self.assertEqual(alert.alert_type, AlertType.BLACKLIST_DETECTED.value)
        self.assertEqual(alert.priority, AlertPriority.CRITICAL)
        self.assertEqual(alert.person_id, 'blacklisted_person')
        self.assertEqual(alert.confidence, 0.85)
    
    def test_classify_known_person_detection(self):
        """Test classification of known person detection."""
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        face_detection = FaceDetection(bounding_box=bbox)
        
        recognition = FaceRecognition(
            face_detection=face_detection,
            status=RecognitionStatus.KNOWN,
            identity='john_doe',
            similarity_score=0.92
        )
        
        recognition_event = FaceRecognitionEvent(
            event_type=EventType.FACE_RECOGNIZED,
            recognitions=[recognition],
            event_id='test_event_2'
        )
        
        alerts = self.alert_manager.classify_recognition_event(recognition_event)
        
        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        
        self.assertEqual(alert.alert_type, AlertType.KNOWN_PERSON_DETECTED.value)
        self.assertEqual(alert.priority, AlertPriority.INFO)
        self.assertEqual(alert.person_id, 'john_doe')
    
    def test_classify_unknown_person_detection(self):
        """Test classification of unknown person detection."""
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        face_detection = FaceDetection(bounding_box=bbox)
        
        recognition = FaceRecognition(
            face_detection=face_detection,
            status=RecognitionStatus.UNKNOWN,
            identity=None,
            similarity_score=0.3
        )
        
        recognition_event = FaceRecognitionEvent(
            event_type=EventType.FACE_UNKNOWN,
            recognitions=[recognition],
            event_id='test_event_3'
        )
        
        alerts = self.alert_manager.classify_recognition_event(recognition_event)
        
        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        
        self.assertEqual(alert.alert_type, AlertType.UNKNOWN_PERSON_DETECTED.value)
        self.assertEqual(alert.priority, AlertPriority.WARNING)
        self.assertIsNone(alert.person_id)
    
    def test_classify_multiple_recognitions(self):
        """Test classification with multiple recognitions in one event."""
        bbox1 = BoundingBox(x=0, y=0, width=100, height=100)
        bbox2 = BoundingBox(x=200, y=0, width=100, height=100)
        
        face_detection1 = FaceDetection(bounding_box=bbox1)
        face_detection2 = FaceDetection(bounding_box=bbox2)
        
        recognition1 = FaceRecognition(
            face_detection=face_detection1,
            status=RecognitionStatus.KNOWN,
            identity='person_1',
            similarity_score=0.9
        )
        
        recognition2 = FaceRecognition(
            face_detection=face_detection2,
            status=RecognitionStatus.UNKNOWN,
            identity=None,
            similarity_score=0.4
        )
        
        recognition_event = FaceRecognitionEvent(
            event_type=EventType.FACE_RECOGNIZED,
            recognitions=[recognition1, recognition2],
            event_id='test_event_4'
        )
        
        alerts = self.alert_manager.classify_recognition_event(recognition_event)
        
        # Should create 2 alerts
        self.assertEqual(len(alerts), 2)
        
        # Verify alert types
        alert_types = [alert.alert_type for alert in alerts]
        self.assertIn(AlertType.KNOWN_PERSON_DETECTED.value, alert_types)
        self.assertIn(AlertType.UNKNOWN_PERSON_DETECTED.value, alert_types)
    
    def test_disabled_alert_types(self):
        """Test that disabled alert types are not generated."""
        config = {
            'blacklist_detection': {'enabled': True},
            'known_person_detection': {'enabled': False},  # Disabled
            'unknown_person_detection': {'enabled': True},
            'system_alerts': {'enabled': True}
        }
        
        manager = AlertManager(config)
        
        # Create known person recognition
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        face_detection = FaceDetection(bounding_box=bbox)
        
        recognition = FaceRecognition(
            face_detection=face_detection,
            status=RecognitionStatus.KNOWN,
            identity='test_person',
            similarity_score=0.9
        )
        
        recognition_event = FaceRecognitionEvent(
            event_type=EventType.FACE_RECOGNIZED,
            recognitions=[recognition],
            event_id='test_event_5'
        )
        
        alerts = manager.classify_recognition_event(recognition_event)
        
        # Should not create alert for known person
        self.assertEqual(len(alerts), 0)
    
    def test_create_system_alert(self):
        """Test creating a system alert."""
        alert = self.alert_manager.create_system_alert(
            message='Test system message',
            component='test_component',
            severity='critical'
        )
        
        self.assertEqual(alert.alert_type, AlertType.SYSTEM_ALERT.value)
        self.assertEqual(alert.priority, AlertPriority.CRITICAL)
        self.assertEqual(alert.system_message, 'Test system message')
        self.assertEqual(alert.component, 'test_component')
    
    def test_create_test_alert(self):
        """Test creating a test alert."""
        alert = self.alert_manager.create_test_alert(message='Test message')
        
        self.assertEqual(alert.alert_type, AlertType.TEST_NOTIFICATION.value)
        self.assertEqual(alert.priority, AlertPriority.INFO)
        self.assertEqual(alert.test_message, 'Test message')
    
    def test_get_stats(self):
        """Test getting alert manager statistics."""
        # Create some alerts
        self.alert_manager.create_system_alert('Test 1')
        self.alert_manager.create_system_alert('Test 2')
        self.alert_manager.create_test_alert('Test 3')
        
        stats = self.alert_manager.get_stats()
        
        self.assertEqual(stats['alerts_created'], 3)
        self.assertIn('system_alert', stats['alerts_by_type'])
        self.assertIn('test_notification', stats['alerts_by_type'])
    
    def test_empty_recognition_event(self):
        """Test handling of empty recognition event."""
        recognition_event = FaceRecognitionEvent(
            event_type=EventType.NO_FACES_DETECTED,
            recognitions=[],
            event_id='test_event_6'
        )
        
        alerts = self.alert_manager.classify_recognition_event(recognition_event)
        
        # Should return empty list
        self.assertEqual(len(alerts), 0)


class TestAlertPriority(unittest.TestCase):
    """Test suite for AlertPriority enum."""
    
    def test_priority_values(self):
        """Test priority enum values."""
        self.assertEqual(AlertPriority.INFO.value, 'info')
        self.assertEqual(AlertPriority.WARNING.value, 'warning')
        self.assertEqual(AlertPriority.CRITICAL.value, 'critical')
    
    def test_priority_comparison(self):
        """Test priority enum ordering."""
        priorities = [AlertPriority.INFO, AlertPriority.WARNING, AlertPriority.CRITICAL]
        
        # Just verify they exist and are different
        self.assertEqual(len(set(priorities)), 3)


class TestAlertType(unittest.TestCase):
    """Test suite for AlertType enum."""
    
    def test_alert_type_values(self):
        """Test alert type enum values."""
        self.assertEqual(AlertType.KNOWN_PERSON_DETECTED.value, 'known_person_detected')
        self.assertEqual(AlertType.UNKNOWN_PERSON_DETECTED.value, 'unknown_person_detected')
        self.assertEqual(AlertType.BLACKLIST_DETECTED.value, 'blacklist_detected')
        self.assertEqual(AlertType.SYSTEM_ALERT.value, 'system_alert')
        self.assertEqual(AlertType.TEST_NOTIFICATION.value, 'test_notification')


if __name__ == '__main__':
    unittest.main()
