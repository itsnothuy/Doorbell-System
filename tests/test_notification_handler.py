#!/usr/bin/env python3
"""
Test suite for Notification Handler

Comprehensive tests for the main notification processing worker.
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.enrichment.notification_handler import NotificationHandler, NotificationStatus
from src.enrichment.alert_manager import Alert, AlertPriority, AlertType
from src.communication.message_bus import MessageBus, Message
from src.communication.events import (
    FaceRecognitionEvent, FaceRecognition, FaceDetection,
    BoundingBox, RecognitionStatus, EventType
)


class TestNotificationHandler(unittest.TestCase):
    """Test suite for NotificationHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_notifications.db')
        
        # Configuration for testing
        self.config = {
            'enabled': True,
            'debounce_period': 1.0,
            'aggregation_window': 30.0,
            'max_per_hour': 100,
            'alerts': {
                'blacklist_detection': {'enabled': True},
                'known_person_detection': {'enabled': True},
                'unknown_person_detection': {'enabled': True},
                'system_alerts': {'enabled': True}
            },
            'rate_limiting': {
                'enabled': True,
                'max_per_minute': 10,
                'max_per_hour': 100,
                'burst_allowance': 3
            },
            'delivery': {
                'web_interface': {'enabled': False},
                'file_log': {'enabled': False},
                'console': {'enabled': False}
            },
            'storage': {
                'enabled': True,
                'db_path': self.db_path
            }
        }
        
        # Create mock message bus
        self.message_bus = Mock(spec=MessageBus)
        self.message_bus.subscribe = Mock()
        self.message_bus.publish = Mock()
        
        # Create notification handler
        self.handler = NotificationHandler(self.message_bus, self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Close database and remove temp files
        if hasattr(self, 'handler') and self.handler.notification_db:
            self.handler.notification_db.close()
        
        # Clean up temp directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_handler_initialization(self):
        """Test notification handler initialization."""
        self.assertTrue(self.handler.enabled)
        self.assertEqual(self.handler.debounce_period, 1.0)
        self.assertIsNotNone(self.handler.alert_manager)
        self.assertIsNotNone(self.handler.rate_limiter)
        self.assertIsNotNone(self.handler.delivery_manager)
        self.assertIsNotNone(self.handler.notification_db)
    
    def test_subscriptions_setup(self):
        """Test that handler subscribes to correct topics."""
        # Verify subscribe was called for each topic
        call_args_list = self.message_bus.subscribe.call_args_list
        
        topics = [call[0][0] for call in call_args_list]
        
        self.assertIn('faces_recognized', topics)
        self.assertIn('system_alerts', topics)
        self.assertIn('component_errors', topics)
        self.assertIn('notification_test', topics)
        self.assertIn('system_shutdown', topics)
    
    def test_generate_deduplication_key_with_person(self):
        """Test deduplication key generation with person ID."""
        alert = Alert(
            alert_type=AlertType.KNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.INFO,
            person_id='test_person'
        )
        
        key = self.handler._generate_deduplication_key(alert)
        self.assertEqual(key, 'known_person_detected_test_person')
    
    def test_generate_deduplication_key_without_person(self):
        """Test deduplication key generation without person ID."""
        alert = Alert(
            alert_type=AlertType.UNKNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.WARNING
        )
        
        key = self.handler._generate_deduplication_key(alert)
        self.assertEqual(key, 'unknown_person_detected')
    
    def test_is_duplicate_notification(self):
        """Test duplicate notification detection."""
        alert = Alert(
            alert_type=AlertType.KNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.INFO,
            person_id='test_person'
        )
        
        # First check should return False
        self.assertFalse(self.handler._is_duplicate_notification(alert))
        
        # Track the notification
        self.handler._track_recent_notification(alert)
        
        # Second check should return True (within debounce period)
        self.assertTrue(self.handler._is_duplicate_notification(alert))
        
        # After debounce period, should return False again
        time.sleep(1.1)  # Wait longer than debounce period
        self.assertFalse(self.handler._is_duplicate_notification(alert))
    
    def test_track_recent_notification(self):
        """Test tracking recent notifications."""
        alert = Alert(
            alert_type=AlertType.KNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.INFO,
            person_id='test_person'
        )
        
        initial_count = len(self.handler.recent_notifications)
        
        self.handler._track_recent_notification(alert)
        
        self.assertEqual(len(self.handler.recent_notifications), initial_count + 1)
    
    def test_generate_notification_content_blacklist(self):
        """Test notification content generation for blacklist alert."""
        alert = Alert(
            alert_type=AlertType.BLACKLIST_DETECTED.value,
            priority=AlertPriority.CRITICAL,
            person_id='blacklist_person',
            confidence=0.92
        )
        
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        face_detection = FaceDetection(bounding_box=bbox)
        recognition = FaceRecognition(
            face_detection=face_detection,
            status=RecognitionStatus.BLACKLISTED
        )
        recognition_event = FaceRecognitionEvent(
            event_type=EventType.FACE_BLACKLISTED,
            recognitions=[recognition],
            event_id='test_event'
        )
        
        content = self.handler._generate_notification_content(alert, recognition_event)
        
        self.assertIn('🚨 SECURITY ALERT', content['title'])
        self.assertEqual(content['priority'], 'critical')
        self.assertTrue(content['immediate_action_required'])
        self.assertEqual(content['person_id'], 'blacklist_person')
    
    def test_generate_notification_content_known_person(self):
        """Test notification content generation for known person alert."""
        alert = Alert(
            alert_type=AlertType.KNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.INFO,
            person_id='john_doe',
            person_name='John Doe',
            confidence=0.95
        )
        
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        face_detection = FaceDetection(bounding_box=bbox)
        recognition = FaceRecognition(
            face_detection=face_detection,
            status=RecognitionStatus.KNOWN
        )
        recognition_event = FaceRecognitionEvent(
            event_type=EventType.FACE_RECOGNIZED,
            recognitions=[recognition],
            event_id='test_event'
        )
        
        content = self.handler._generate_notification_content(alert, recognition_event)
        
        self.assertIn('John Doe', content['title'])
        self.assertEqual(content['priority'], 'info')
        self.assertEqual(content['person_name'], 'John Doe')
    
    def test_generate_notification_content_unknown_person(self):
        """Test notification content generation for unknown person alert."""
        alert = Alert(
            alert_type=AlertType.UNKNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.WARNING,
            confidence=0.6
        )
        
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        face_detection = FaceDetection(bounding_box=bbox)
        recognition = FaceRecognition(
            face_detection=face_detection,
            status=RecognitionStatus.UNKNOWN
        )
        recognition_event = FaceRecognitionEvent(
            event_type=EventType.FACE_UNKNOWN,
            recognitions=[recognition],
            event_id='test_event'
        )
        
        content = self.handler._generate_notification_content(alert, recognition_event)
        
        self.assertIn('Unknown Person', content['title'])
        self.assertEqual(content['priority'], 'warning')
        self.assertTrue(content['requires_identification'])
    
    def test_handle_recognition_event_creates_alerts(self):
        """Test handling recognition event creates appropriate alerts."""
        # Create recognition event
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
            event_id='test_event'
        )
        
        # Create message
        message = Message(
            topic='faces_recognized',
            data=recognition_event
        )
        
        # Process the message
        initial_count = self.handler.processed_count
        self.handler.handle_recognition_event(message)
        
        # Verify processing happened
        self.assertEqual(self.handler.processed_count, initial_count + 1)
    
    def test_handle_system_alert(self):
        """Test handling system alert."""
        system_data = {
            'message': 'Test system alert',
            'component': 'test_component',
            'severity': 'warning'
        }
        
        message = Message(
            topic='system_alerts',
            data=system_data
        )
        
        # Process system alert
        self.handler.handle_system_alert(message)
        
        # Verify alert was created (check alert_manager stats)
        stats = self.handler.alert_manager.get_stats()
        self.assertGreater(stats['alerts_created'], 0)
    
    def test_handle_test_notification(self):
        """Test handling test notification."""
        test_data = {
            'message': 'This is a test notification'
        }
        
        message = Message(
            topic='notification_test',
            data=test_data
        )
        
        # Process test notification
        self.handler.handle_test_notification(message)
        
        # Should complete without error (actual delivery is mocked)
        # Just verify it doesn't crash
        self.assertTrue(True)
    
    def test_get_metrics(self):
        """Test getting notification handler metrics."""
        metrics = self.handler.get_metrics()
        
        self.assertIn('notifications_sent', metrics)
        self.assertIn('notifications_rate_limited', metrics)
        self.assertIn('notifications_failed', metrics)
        self.assertIn('success_rate', metrics)
        self.assertIn('rate_limit_effectiveness', metrics)
        self.assertIn('active_delivery_channels', metrics)
    
    def test_disabled_handler_skips_processing(self):
        """Test that disabled handler skips processing."""
        # Create handler with disabled config
        disabled_config = self.config.copy()
        disabled_config['enabled'] = False
        
        disabled_handler = NotificationHandler(self.message_bus, disabled_config)
        
        # Create recognition event
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        face_detection = FaceDetection(bounding_box=bbox)
        recognition = FaceRecognition(
            face_detection=face_detection,
            status=RecognitionStatus.KNOWN
        )
        recognition_event = FaceRecognitionEvent(
            event_type=EventType.FACE_RECOGNIZED,
            recognitions=[recognition],
            event_id='test_event'
        )
        
        message = Message(
            topic='faces_recognized',
            data=recognition_event
        )
        
        # Process should return early
        initial_count = disabled_handler.processed_count
        disabled_handler.handle_recognition_event(message)
        
        # Count should not increase
        self.assertEqual(disabled_handler.processed_count, initial_count)


class TestNotificationHandlerIntegration(unittest.TestCase):
    """Integration tests for notification handler with real components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_notifications.db')
        
        # Real configuration with file log enabled
        self.config = {
            'enabled': True,
            'debounce_period': 0.5,
            'max_per_hour': 100,
            'alerts': {
                'blacklist_detection': {'enabled': True},
                'known_person_detection': {'enabled': True},
                'unknown_person_detection': {'enabled': True},
                'system_alerts': {'enabled': True}
            },
            'rate_limiting': {
                'enabled': True,
                'max_per_minute': 10,
                'max_per_hour': 100,
                'burst_allowance': 3
            },
            'delivery': {
                'web_interface': {'enabled': False},
                'file_log': {
                    'enabled': True,
                    'path': os.path.join(self.temp_dir, 'notifications.log'),
                    'format': 'json'
                },
                'console': {'enabled': False}
            },
            'storage': {
                'enabled': True,
                'db_path': self.db_path
            }
        }
        
        self.message_bus = Mock(spec=MessageBus)
        self.handler = NotificationHandler(self.message_bus, self.config)
        self.handler._initialize_worker()
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        if hasattr(self, 'handler'):
            self.handler._cleanup_worker()
        
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_notification_flow(self):
        """Test complete notification flow from event to delivery and storage."""
        # Create recognition event
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        face_detection = FaceDetection(bounding_box=bbox)
        recognition = FaceRecognition(
            face_detection=face_detection,
            status=RecognitionStatus.BLACKLISTED,
            identity='blacklist_person',
            similarity_score=0.95
        )
        recognition_event = FaceRecognitionEvent(
            event_type=EventType.FACE_BLACKLISTED,
            recognitions=[recognition],
            event_id='test_event_integration'
        )
        
        message = Message(
            topic='faces_recognized',
            data=recognition_event
        )
        
        # Process the event
        initial_sent = self.handler.notifications_sent
        self.handler.handle_recognition_event(message)
        
        # Verify notification was sent
        self.assertEqual(self.handler.notifications_sent, initial_sent + 1)
        
        # Verify notification was stored in database
        notifications = self.handler.notification_db.get_notifications(limit=10)
        self.assertGreater(len(notifications), 0)
        
        # Verify notification content
        stored_notif = notifications[0]
        self.assertEqual(stored_notif['alert_type'], 'blacklist_detected')
        self.assertEqual(stored_notif['priority'], 'critical')
        self.assertEqual(stored_notif['status'], 'delivered')


if __name__ == '__main__':
    unittest.main()
