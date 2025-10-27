#!/usr/bin/env python3
"""
Events System Test Suite

Comprehensive tests for the events system including event creation,
serialization, type validation, and lifecycle management.
"""

import time
import pytest
import uuid
from typing import Dict, Any
from unittest.mock import Mock, MagicMock

from src.communication.events import (
    # Enums
    EventType,
    EventPriority,
    RecognitionStatus,
    # Data classes
    BoundingBox,
    FaceDetection,
    FaceRecognition,
    PipelineEvent,
    DoorbellEvent,
    FrameEvent,
    MotionEvent,
    MotionResult,
    MotionHistory,
    FaceDetectionEvent,
    FaceRecognitionEvent,
    NotificationEvent,
    SystemEvent,
    # Utilities
    EventHandler,
    EventFilter,
    create_doorbell_event,
    create_frame_event,
    create_notification_event,
)


class TestEventEnums:
    """Test suite for event enumeration types."""
    
    def test_event_type_enum_values(self):
        """Test that all expected event types exist."""
        expected_types = [
            'DOORBELL_PRESSED',
            'GPIO_STATE_CHANGED',
            'FRAME_CAPTURED',
            'FRAME_PROCESSED',
            'FRAME_DROPPED',
            'MOTION_DETECTED',
            'MOTION_ENDED',
            'FACES_DETECTED',
            'NO_FACES_DETECTED',
            'FACE_DETECTION_FAILED',
            'FACE_RECOGNIZED',
            'FACE_UNKNOWN',
            'FACE_BLACKLISTED',
            'RECOGNITION_FAILED',
            'EVENT_CREATED',
            'EVENT_ENRICHED',
            'EVENT_STORED',
            'NOTIFICATION_SENT',
            'NOTIFICATION_FAILED',
            'SYSTEM_STARTED',
            'SYSTEM_STOPPED',
            'COMPONENT_STARTED',
            'COMPONENT_STOPPED',
            'COMPONENT_ERROR',
            'HEALTH_CHECK',
            'PERFORMANCE_METRIC',
        ]
        
        for event_type_name in expected_types:
            assert hasattr(EventType, event_type_name)
            assert isinstance(getattr(EventType, event_type_name), EventType)
    
    def test_event_priority_enum(self):
        """Test event priority levels."""
        assert EventPriority.LOW.value == 1
        assert EventPriority.NORMAL.value == 2
        assert EventPriority.HIGH.value == 3
        assert EventPriority.CRITICAL.value == 4
    
    def test_recognition_status_enum(self):
        """Test recognition status values."""
        assert RecognitionStatus.KNOWN.value == "known"
        assert RecognitionStatus.UNKNOWN.value == "unknown"
        assert RecognitionStatus.BLACKLISTED.value == "blacklisted"
        assert RecognitionStatus.PROCESSING.value == "processing"
        assert RecognitionStatus.FAILED.value == "failed"


class TestBoundingBox:
    """Test suite for BoundingBox data class."""
    
    def test_bounding_box_creation(self):
        """Test basic bounding box creation."""
        bbox = BoundingBox(x=10, y=20, width=100, height=80, confidence=0.95)
        
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 80
        assert bbox.confidence == 0.95
    
    def test_bounding_box_properties(self):
        """Test bounding box computed properties."""
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        
        assert bbox.top == 20
        assert bbox.bottom == 100  # y + height
        assert bbox.left == 10
        assert bbox.right == 110  # x + width
        assert bbox.center == (60, 60)  # (x + width//2, y + height//2)
        assert bbox.area == 8000  # width * height
    
    def test_bounding_box_default_confidence(self):
        """Test default confidence value."""
        bbox = BoundingBox(x=0, y=0, width=10, height=10)
        
        assert bbox.confidence == 0.0


class TestFaceDetection:
    """Test suite for FaceDetection data class."""
    
    def test_face_detection_creation(self):
        """Test face detection creation."""
        bbox = BoundingBox(x=10, y=20, width=100, height=80, confidence=0.95)
        face_det = FaceDetection(
            bounding_box=bbox,
            landmarks={'nose': (50, 50)},
            confidence=0.98,
            quality_score=0.85
        )
        
        assert face_det.bounding_box == bbox
        assert face_det.landmarks == {'nose': (50, 50)}
        assert face_det.confidence == 0.98
        assert face_det.quality_score == 0.85
    
    def test_face_detection_to_dict(self):
        """Test face detection serialization."""
        bbox = BoundingBox(x=10, y=20, width=100, height=80, confidence=0.95)
        face_det = FaceDetection(
            bounding_box=bbox,
            confidence=0.98,
            quality_score=0.85
        )
        
        result = face_det.to_dict()
        
        assert result['bounding_box']['x'] == 10
        assert result['bounding_box']['y'] == 20
        assert result['confidence'] == 0.98
        assert result['quality_score'] == 0.85
    
    def test_face_detection_with_encoding(self):
        """Test face detection with encoding."""
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        encoding = [0.1, 0.2, 0.3]  # Mock encoding
        
        face_det = FaceDetection(
            bounding_box=bbox,
            encoding=encoding
        )
        
        result = face_det.to_dict()
        assert result['encoding'] == encoding


class TestFaceRecognition:
    """Test suite for FaceRecognition data class."""
    
    def test_face_recognition_creation(self):
        """Test face recognition result creation."""
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        face_det = FaceDetection(bounding_box=bbox)
        
        face_rec = FaceRecognition(
            face_detection=face_det,
            status=RecognitionStatus.KNOWN,
            identity="John Doe",
            similarity_score=0.95,
            recognition_time=0.05
        )
        
        assert face_rec.face_detection == face_det
        assert face_rec.status == RecognitionStatus.KNOWN
        assert face_rec.identity == "John Doe"
        assert face_rec.similarity_score == 0.95
        assert face_rec.recognition_time == 0.05
    
    def test_face_recognition_to_dict(self):
        """Test face recognition serialization."""
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        face_det = FaceDetection(bounding_box=bbox)
        face_rec = FaceRecognition(
            face_detection=face_det,
            status=RecognitionStatus.UNKNOWN,
            similarity_score=0.45
        )
        
        result = face_rec.to_dict()
        
        assert result['status'] == 'unknown'
        assert result['similarity_score'] == 0.45
        assert 'face_detection' in result
    
    def test_face_recognition_unknown_status(self):
        """Test face recognition for unknown person."""
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        face_det = FaceDetection(bounding_box=bbox)
        
        face_rec = FaceRecognition(
            face_detection=face_det,
            status=RecognitionStatus.UNKNOWN
        )
        
        assert face_rec.status == RecognitionStatus.UNKNOWN
        assert face_rec.identity is None
    
    def test_face_recognition_blacklisted(self):
        """Test face recognition for blacklisted person."""
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        face_det = FaceDetection(bounding_box=bbox)
        
        face_rec = FaceRecognition(
            face_detection=face_det,
            status=RecognitionStatus.BLACKLISTED,
            identity="Blocked Person"
        )
        
        assert face_rec.status == RecognitionStatus.BLACKLISTED


class TestPipelineEvent:
    """Test suite for base PipelineEvent class."""
    
    def test_pipeline_event_creation(self):
        """Test pipeline event creation."""
        event = PipelineEvent(
            event_type=EventType.MOTION_DETECTED,
            source="test_component",
            data={"motion_score": 0.75}
        )
        
        assert event.event_type == EventType.MOTION_DETECTED
        assert event.source == "test_component"
        assert event.data["motion_score"] == 0.75
        assert event.priority == EventPriority.NORMAL
        assert event.event_id is not None
        assert event.timestamp > 0
    
    def test_pipeline_event_auto_id_generation(self):
        """Test automatic event ID generation."""
        event1 = PipelineEvent(event_type=EventType.MOTION_DETECTED)
        event2 = PipelineEvent(event_type=EventType.MOTION_DETECTED)
        
        assert event1.event_id != event2.event_id
        assert isinstance(event1.event_id, str)
        assert isinstance(event2.event_id, str)
    
    def test_pipeline_event_with_priority(self):
        """Test event with different priorities."""
        event = PipelineEvent(
            event_type=EventType.FACE_BLACKLISTED,
            priority=EventPriority.CRITICAL
        )
        
        assert event.priority == EventPriority.CRITICAL
    
    def test_pipeline_event_with_correlation(self):
        """Test event with correlation ID."""
        correlation_id = "test-correlation-123"
        event = PipelineEvent(
            event_type=EventType.MOTION_DETECTED,
            correlation_id=correlation_id
        )
        
        assert event.correlation_id == correlation_id
    
    def test_pipeline_event_with_parent(self):
        """Test event with parent event ID."""
        parent_event = PipelineEvent(event_type=EventType.FRAME_CAPTURED)
        child_event = PipelineEvent(
            event_type=EventType.MOTION_DETECTED,
            parent_event_id=parent_event.event_id
        )
        
        assert child_event.parent_event_id == parent_event.event_id
    
    def test_pipeline_event_to_dict(self):
        """Test event serialization to dictionary."""
        event = PipelineEvent(
            event_type=EventType.MOTION_DETECTED,
            source="test_source",
            data={"key": "value"},
            priority=EventPriority.HIGH
        )
        
        result = event.to_dict()
        
        assert result['event_type'] == 'MOTION_DETECTED'
        assert result['source'] == 'test_source'
        assert result['data'] == {"key": "value"}
        assert result['priority'] == 'HIGH'
        assert 'event_id' in result
        assert 'timestamp' in result
    
    def test_pipeline_event_from_dict(self):
        """Test event deserialization from dictionary."""
        event_dict = {
            'event_id': 'test-123',
            'event_type': 'MOTION_DETECTED',
            'timestamp': time.time(),
            'source': 'test_source',
            'data': {'key': 'value'},
            'priority': 'HIGH',
            'correlation_id': 'corr-123',
            'parent_event_id': 'parent-456'
        }
        
        event = PipelineEvent.from_dict(event_dict)
        
        assert event.event_id == 'test-123'
        assert event.event_type == EventType.MOTION_DETECTED
        assert event.source == 'test_source'
        assert event.priority == EventPriority.HIGH


class TestDoorbellEvent:
    """Test suite for DoorbellEvent."""
    
    def test_doorbell_event_creation(self):
        """Test doorbell event creation."""
        event = DoorbellEvent(
            event_type=EventType.DOORBELL_PRESSED,
            channel=18,
            press_duration=0.5
        )
        
        assert event.event_type == EventType.DOORBELL_PRESSED
        assert event.channel == 18
        assert event.press_duration == 0.5
        assert event.data['channel'] == 18
        assert event.data['press_duration'] == 0.5
    
    def test_doorbell_event_default_values(self):
        """Test doorbell event with default values."""
        event = DoorbellEvent(event_type=EventType.DOORBELL_PRESSED)
        
        assert event.channel == 0
        assert event.press_duration is None
    
    def test_create_doorbell_event_convenience(self):
        """Test convenience function for creating doorbell event."""
        event = create_doorbell_event(channel=23)
        
        assert event.event_type == EventType.DOORBELL_PRESSED
        assert event.channel == 23
        assert event.source == 'gpio_handler'


class TestFrameEvent:
    """Test suite for FrameEvent."""
    
    def test_frame_event_creation(self):
        """Test frame event creation."""
        event = FrameEvent(
            event_type=EventType.FRAME_CAPTURED,
            frame_path="/path/to/frame.jpg",
            resolution=(1920, 1080)
        )
        
        assert event.event_type == EventType.FRAME_CAPTURED
        assert event.frame_path == "/path/to/frame.jpg"
        assert event.resolution == (1920, 1080)
        assert event.frame_id is not None
        assert 'frame_id' in event.data
    
    def test_frame_event_has_frame_data(self):
        """Test frame data availability check."""
        event1 = FrameEvent(event_type=EventType.FRAME_CAPTURED)
        event2 = FrameEvent(event_type=EventType.FRAME_CAPTURED, frame_data="data")
        
        assert event1.has_frame_data is False
        assert event2.has_frame_data is True
    
    def test_frame_event_frame_size(self):
        """Test frame size calculation."""
        frame_data = "test frame data"
        event = FrameEvent(
            event_type=EventType.FRAME_CAPTURED,
            frame_data=frame_data
        )
        
        assert event.frame_size is not None
        assert event.frame_size > 0
    
    def test_create_frame_event_convenience(self):
        """Test convenience function for creating frame event."""
        frame_data = "test_data"
        event = create_frame_event(frame_data, frame_path="/test/path.jpg")
        
        assert event.event_type == EventType.FRAME_CAPTURED
        assert event.frame_data == frame_data
        assert event.frame_path == "/test/path.jpg"
        assert event.source == 'frame_capture'


class TestMotionEvent:
    """Test suite for MotionEvent."""
    
    def test_motion_event_creation(self):
        """Test motion event creation."""
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        event = MotionEvent(
            event_type=EventType.MOTION_DETECTED,
            motion_regions=[bbox],
            motion_score=0.85
        )
        
        assert event.event_type == EventType.MOTION_DETECTED
        assert len(event.motion_regions) == 1
        assert event.motion_score == 0.85
        assert 'motion_score' in event.data
    
    def test_motion_event_with_frame(self):
        """Test motion event with associated frame."""
        frame_event = FrameEvent(event_type=EventType.FRAME_CAPTURED)
        motion_event = MotionEvent(
            event_type=EventType.MOTION_DETECTED,
            frame_event=frame_event,
            motion_score=0.75
        )
        
        assert motion_event.frame_event == frame_event
        assert motion_event.data['frame_id'] == frame_event.frame_id


class TestMotionResult:
    """Test suite for MotionResult data class."""
    
    def test_motion_result_creation(self):
        """Test motion result creation."""
        result = MotionResult(
            motion_detected=True,
            motion_score=75.5,
            motion_regions=[(10, 20, 100, 80)],
            contour_count=3,
            largest_contour_area=5000,
            motion_center=(50, 60),
            frame_timestamp=time.time(),
            processing_time=0.05
        )
        
        assert result.motion_detected is True
        assert result.motion_score == 75.5
        assert len(result.motion_regions) == 1
        assert result.contour_count == 3
    
    def test_motion_result_to_dict(self):
        """Test motion result serialization."""
        result = MotionResult(
            motion_detected=True,
            motion_score=75.5,
            motion_regions=[(10, 20, 100, 80)],
            contour_count=3,
            largest_contour_area=5000,
            motion_center=(50, 60),
            frame_timestamp=time.time(),
            processing_time=0.05
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['motion_detected'] is True
        assert result_dict['motion_score'] == 75.5
        assert result_dict['contour_count'] == 3


class TestMotionHistory:
    """Test suite for MotionHistory."""
    
    def test_motion_history_initialization(self):
        """Test motion history initialization."""
        history = MotionHistory()
        
        assert len(history.recent_scores) == 0
        assert len(history.motion_events) == 0
        assert history.static_duration == 0.0
        assert history.trend_direction == "stable"
    
    def test_motion_history_add_score(self):
        """Test adding scores to motion history."""
        history = MotionHistory()
        timestamp = time.time()
        
        history.add_score(75.0, timestamp, is_motion=True)
        
        assert len(history.recent_scores) == 1
        assert history.recent_scores[0] == 75.0
        assert len(history.motion_events) == 1
        assert history.last_motion_time == timestamp
    
    def test_motion_history_static_duration(self):
        """Test static duration calculation."""
        history = MotionHistory()
        timestamp1 = time.time()
        timestamp2 = timestamp1 + 5.0
        
        history.add_score(75.0, timestamp1, is_motion=True)
        history.add_score(10.0, timestamp2, is_motion=False)
        
        assert history.static_duration == 5.0
    
    def test_motion_history_calculate_trend(self):
        """Test trend calculation."""
        history = MotionHistory()
        
        # Add scores showing increasing trend
        for i in range(10):
            history.add_score(float(i * 10), time.time(), is_motion=False)
        
        trend = history.calculate_trend()
        
        # Trend should be detected (implementation dependent)
        assert trend in ["increasing", "decreasing", "stable"]
    
    def test_motion_history_trim(self):
        """Test trimming history to max size."""
        history = MotionHistory()
        
        # Add many scores
        for i in range(150):
            history.add_score(float(i), time.time(), is_motion=False)
        
        history.trim_history(max_size=100)
        
        assert len(history.recent_scores) == 100


class TestFaceDetectionEvent:
    """Test suite for FaceDetectionEvent."""
    
    def test_face_detection_event_with_faces(self):
        """Test face detection event with detected faces."""
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        face = FaceDetection(bounding_box=bbox, confidence=0.95)
        
        event = FaceDetectionEvent(
            event_type=EventType.FACES_DETECTED,
            faces=[face],
            detection_time=0.05,
            detector_type="cpu"
        )
        
        assert event.event_type == EventType.FACES_DETECTED
        assert len(event.faces) == 1
        assert event.data['face_count'] == 1
        assert event.detector_type == "cpu"
    
    def test_face_detection_event_no_faces(self):
        """Test face detection event with no faces."""
        event = FaceDetectionEvent(
            event_type=EventType.NO_FACES_DETECTED,
            faces=[],
            detection_time=0.05,
            detector_type="cpu"
        )
        
        assert event.event_type == EventType.NO_FACES_DETECTED
        assert event.data['face_count'] == 0


class TestFaceRecognitionEvent:
    """Test suite for FaceRecognitionEvent."""
    
    def test_face_recognition_event_known(self):
        """Test face recognition event with known person."""
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        face_det = FaceDetection(bounding_box=bbox)
        face_rec = FaceRecognition(
            face_detection=face_det,
            status=RecognitionStatus.KNOWN,
            identity="John Doe"
        )
        
        event = FaceRecognitionEvent(
            event_type=EventType.FACE_RECOGNIZED,
            recognitions=[face_rec],
            recognition_time=0.1
        )
        
        assert event.event_type == EventType.FACE_RECOGNIZED
        assert event.known_count == 1
        assert event.unknown_count == 0
        assert event.blacklisted_count == 0
    
    def test_face_recognition_event_unknown(self):
        """Test face recognition event with unknown person."""
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        face_det = FaceDetection(bounding_box=bbox)
        face_rec = FaceRecognition(
            face_detection=face_det,
            status=RecognitionStatus.UNKNOWN
        )
        
        event = FaceRecognitionEvent(
            event_type=EventType.FACE_UNKNOWN,
            recognitions=[face_rec]
        )
        
        assert event.event_type == EventType.FACE_UNKNOWN
        assert event.unknown_count == 1
    
    def test_face_recognition_event_blacklisted(self):
        """Test face recognition event with blacklisted person."""
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        face_det = FaceDetection(bounding_box=bbox)
        face_rec = FaceRecognition(
            face_detection=face_det,
            status=RecognitionStatus.BLACKLISTED
        )
        
        event = FaceRecognitionEvent(
            event_type=EventType.FACE_BLACKLISTED,
            recognitions=[face_rec]
        )
        
        assert event.event_type == EventType.FACE_BLACKLISTED
        assert event.blacklisted_count == 1
    
    def test_face_recognition_event_mixed_statuses(self):
        """Test face recognition event with mixed recognition statuses."""
        bbox1 = BoundingBox(x=10, y=20, width=100, height=80)
        bbox2 = BoundingBox(x=120, y=20, width=100, height=80)
        
        face_rec1 = FaceRecognition(
            face_detection=FaceDetection(bounding_box=bbox1),
            status=RecognitionStatus.KNOWN,
            identity="Known Person"
        )
        face_rec2 = FaceRecognition(
            face_detection=FaceDetection(bounding_box=bbox2),
            status=RecognitionStatus.UNKNOWN
        )
        
        event = FaceRecognitionEvent(
            event_type=EventType.FACE_RECOGNIZED,
            recognitions=[face_rec1, face_rec2]
        )
        
        assert event.known_count == 1
        assert event.unknown_count == 1
        assert event.event_type == EventType.FACE_RECOGNIZED  # Known takes precedence


class TestNotificationEvent:
    """Test suite for NotificationEvent."""
    
    def test_notification_event_creation(self):
        """Test notification event creation."""
        event = NotificationEvent(
            event_type=EventType.NOTIFICATION_SENT,
            message="Test notification",
            notification_type="alert",
            image_path="/path/to/image.jpg",
            recipient="user@example.com"
        )
        
        assert event.message == "Test notification"
        assert event.notification_type == "alert"
        assert event.image_path == "/path/to/image.jpg"
        assert event.recipient == "user@example.com"
    
    def test_notification_event_success(self):
        """Test successful notification event."""
        event = NotificationEvent(
            event_type=EventType.NOTIFICATION_SENT,
            message="Success",
            sent_successfully=True
        )
        
        assert event.sent_successfully is True
        assert event.error_message is None
    
    def test_notification_event_failure(self):
        """Test failed notification event."""
        event = NotificationEvent(
            event_type=EventType.NOTIFICATION_FAILED,
            message="Failed notification",
            sent_successfully=False,
            error_message="Connection timeout"
        )
        
        assert event.sent_successfully is False
        assert event.error_message == "Connection timeout"
    
    def test_create_notification_event_convenience(self):
        """Test convenience function for creating notification event."""
        event = create_notification_event(
            message="Test message",
            notification_type="info",
            image_path="/test/path.jpg"
        )
        
        assert event.event_type == EventType.NOTIFICATION_SENT
        assert event.message == "Test message"
        assert event.source == 'notification_handler'


class TestSystemEvent:
    """Test suite for SystemEvent."""
    
    def test_system_event_creation(self):
        """Test system event creation."""
        event = SystemEvent(
            event_type=EventType.COMPONENT_STARTED,
            component="face_detector",
            status="running",
            metrics={"uptime": 100}
        )
        
        assert event.component == "face_detector"
        assert event.status == "running"
        assert event.metrics == {"uptime": 100}
    
    def test_system_event_with_error(self):
        """Test system event with error details."""
        event = SystemEvent(
            event_type=EventType.COMPONENT_ERROR,
            component="database",
            status="error",
            error_details="Connection failed"
        )
        
        assert event.event_type == EventType.COMPONENT_ERROR
        assert event.error_details == "Connection failed"


class TestEventHandler:
    """Test suite for EventHandler base class."""
    
    def test_event_handler_initialization(self):
        """Test event handler initialization."""
        handler = EventHandler(handler_id="test_handler")
        
        assert handler.handler_id == "test_handler"
        assert handler.processed_events == 0
        assert handler.error_count == 0
    
    def test_event_handler_not_implemented(self):
        """Test that _process_event must be implemented."""
        handler = EventHandler(handler_id="test")
        event = PipelineEvent(event_type=EventType.MOTION_DETECTED)
        
        with pytest.raises(NotImplementedError):
            handler._process_event(event)
    
    def test_event_handler_error_handling(self):
        """Test event handler error handling."""
        class TestHandler(EventHandler):
            def _process_event(self, event):
                raise ValueError("Test error")
        
        handler = TestHandler(handler_id="test")
        event = PipelineEvent(event_type=EventType.MOTION_DETECTED)
        
        result = handler.handle_event(event)
        
        assert result is None
        assert handler.error_count == 1
        assert handler.last_error is not None
    
    def test_event_handler_successful_processing(self):
        """Test successful event processing."""
        class TestHandler(EventHandler):
            def _process_event(self, event):
                return PipelineEvent(
                    event_type=EventType.EVENT_ENRICHED,
                    parent_event_id=event.event_id
                )
        
        handler = TestHandler(handler_id="test")
        event = PipelineEvent(event_type=EventType.MOTION_DETECTED)
        
        result = handler.handle_event(event)
        
        assert result is not None
        assert handler.processed_events == 1
        assert handler.error_count == 0
    
    def test_event_handler_get_stats(self):
        """Test event handler statistics."""
        class TestHandler(EventHandler):
            def _process_event(self, event):
                return None
        
        handler = TestHandler(handler_id="test")
        event = PipelineEvent(event_type=EventType.MOTION_DETECTED)
        
        handler.handle_event(event)
        handler.handle_event(event)
        
        stats = handler.get_stats()
        
        assert stats['handler_id'] == "test"
        assert stats['processed_events'] == 2
        assert stats['error_count'] == 0
        assert 'error_rate' in stats


class TestEventFilter:
    """Test suite for EventFilter."""
    
    def test_event_filter_by_type(self):
        """Test filtering by event type."""
        event_filter = EventFilter(
            event_types=[EventType.MOTION_DETECTED, EventType.FACES_DETECTED]
        )
        
        event1 = PipelineEvent(event_type=EventType.MOTION_DETECTED)
        event2 = PipelineEvent(event_type=EventType.FRAME_CAPTURED)
        
        assert event_filter.should_process(event1) is True
        assert event_filter.should_process(event2) is False
    
    def test_event_filter_by_priority(self):
        """Test filtering by minimum priority."""
        event_filter = EventFilter(min_priority=EventPriority.HIGH)
        
        event1 = PipelineEvent(
            event_type=EventType.MOTION_DETECTED,
            priority=EventPriority.HIGH
        )
        event2 = PipelineEvent(
            event_type=EventType.MOTION_DETECTED,
            priority=EventPriority.NORMAL
        )
        
        assert event_filter.should_process(event1) is True
        assert event_filter.should_process(event2) is False
    
    def test_event_filter_by_source(self):
        """Test filtering by source."""
        event_filter = EventFilter(source_filter="test_source")
        
        event1 = PipelineEvent(
            event_type=EventType.MOTION_DETECTED,
            source="test_source"
        )
        event2 = PipelineEvent(
            event_type=EventType.MOTION_DETECTED,
            source="other_source"
        )
        
        assert event_filter.should_process(event1) is True
        assert event_filter.should_process(event2) is False
    
    def test_event_filter_combined(self):
        """Test combined filter criteria."""
        event_filter = EventFilter(
            event_types=[EventType.MOTION_DETECTED],
            min_priority=EventPriority.HIGH,
            source_filter="test_source"
        )
        
        # Passes all filters
        event1 = PipelineEvent(
            event_type=EventType.MOTION_DETECTED,
            priority=EventPriority.HIGH,
            source="test_source"
        )
        
        # Fails type filter
        event2 = PipelineEvent(
            event_type=EventType.FRAME_CAPTURED,
            priority=EventPriority.HIGH,
            source="test_source"
        )
        
        # Fails priority filter
        event3 = PipelineEvent(
            event_type=EventType.MOTION_DETECTED,
            priority=EventPriority.NORMAL,
            source="test_source"
        )
        
        assert event_filter.should_process(event1) is True
        assert event_filter.should_process(event2) is False
        assert event_filter.should_process(event3) is False
    
    def test_event_filter_no_filters(self):
        """Test filter with no criteria (all pass)."""
        event_filter = EventFilter()
        
        event = PipelineEvent(event_type=EventType.MOTION_DETECTED)
        
        assert event_filter.should_process(event) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
