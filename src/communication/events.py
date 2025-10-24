#!/usr/bin/env python3
"""
Events System - Pipeline Event Definitions and Handlers

This module defines all event types and data structures used throughout
the pipeline for type-safe event processing.
"""

import time
import uuid
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


class EventType(Enum):
    """Event types for the pipeline processing."""
    # Hardware events
    DOORBELL_PRESSED = auto()
    GPIO_STATE_CHANGED = auto()
    
    # Frame processing events
    FRAME_CAPTURED = auto()
    FRAME_PROCESSED = auto()
    FRAME_DROPPED = auto()
    
    # Motion detection events
    MOTION_DETECTED = auto()
    MOTION_ENDED = auto()
    
    # Face detection events
    FACES_DETECTED = auto()
    NO_FACES_DETECTED = auto()
    FACE_DETECTION_FAILED = auto()
    
    # Face recognition events
    FACE_RECOGNIZED = auto()
    FACE_UNKNOWN = auto()
    FACE_BLACKLISTED = auto()
    RECOGNITION_FAILED = auto()
    
    # Event processing
    EVENT_CREATED = auto()
    EVENT_ENRICHED = auto()
    EVENT_STORED = auto()
    
    # Notifications
    NOTIFICATION_SENT = auto()
    NOTIFICATION_FAILED = auto()
    
    # System events
    SYSTEM_STARTED = auto()
    SYSTEM_STOPPED = auto()
    COMPONENT_STARTED = auto()
    COMPONENT_STOPPED = auto()
    COMPONENT_ERROR = auto()
    
    # Health and monitoring
    HEALTH_CHECK = auto()
    PERFORMANCE_METRIC = auto()


class EventPriority(Enum):
    """Event priority levels for processing order."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class RecognitionStatus(Enum):
    """Face recognition status values."""
    KNOWN = "known"
    UNKNOWN = "unknown"
    BLACKLISTED = "blacklisted"
    PROCESSING = "processing"
    FAILED = "failed"


@dataclass
class BoundingBox:
    """Represents a bounding box for detected objects."""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
    
    @property
    def top(self) -> int:
        return self.y
    
    @property
    def bottom(self) -> int:
        return self.y + self.height
    
    @property
    def left(self) -> int:
        return self.x
    
    @property
    def right(self) -> int:
        return self.x + self.width
    
    @property
    def center(self) -> tuple:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class FaceDetection:
    """Represents a detected face with metadata."""
    bounding_box: BoundingBox
    landmarks: Optional[Dict[str, Any]] = None
    encoding: Optional[np.ndarray] = None
    confidence: float = 0.0
    quality_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'bounding_box': {
                'x': self.bounding_box.x,
                'y': self.bounding_box.y,
                'width': self.bounding_box.width,
                'height': self.bounding_box.height,
                'confidence': self.bounding_box.confidence
            },
            'landmarks': self.landmarks,
            'confidence': self.confidence,
            'quality_score': self.quality_score,
            'encoding': self.encoding.tolist() if self.encoding is not None else None
        }


@dataclass
class FaceRecognition:
    """Represents a face recognition result."""
    face_detection: FaceDetection
    status: RecognitionStatus
    identity: Optional[str] = None
    similarity_score: float = 0.0
    recognition_time: float = 0.0
    match_details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'face_detection': self.face_detection.to_dict(),
            'status': self.status.value,
            'identity': self.identity,
            'similarity_score': self.similarity_score,
            'recognition_time': self.recognition_time,
            'match_details': self.match_details
        }


@dataclass
class PipelineEvent:
    """Base event class for pipeline processing."""
    event_type: EventType
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.name,
            'timestamp': self.timestamp,
            'source': self.source,
            'data': self.data,
            'priority': self.priority.name,
            'correlation_id': self.correlation_id,
            'parent_event_id': self.parent_event_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineEvent':
        """Create event from dictionary."""
        return cls(
            event_id=data['event_id'],
            event_type=EventType[data['event_type']],
            timestamp=data['timestamp'],
            source=data.get('source'),
            data=data.get('data', {}),
            priority=EventPriority[data.get('priority', 'NORMAL')],
            correlation_id=data.get('correlation_id'),
            parent_event_id=data.get('parent_event_id')
        )


@dataclass
class DoorbellEvent(PipelineEvent):
    """Doorbell press event with GPIO information."""
    channel: int = 0
    press_duration: Optional[float] = None
    
    def __post_init__(self):
        if self.event_type != EventType.DOORBELL_PRESSED:
            self.event_type = EventType.DOORBELL_PRESSED
        self.data.update({
            'channel': self.channel,
            'press_duration': self.press_duration
        })


@dataclass
class FrameEvent(PipelineEvent):
    """Frame capture/processing event."""
    frame_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    frame_data: Optional[np.ndarray] = None
    frame_path: Optional[str] = None
    resolution: Optional[tuple] = None
    capture_time: float = field(default_factory=time.time)
    
    def __post_init__(self):
        self.data.update({
            'frame_id': self.frame_id,
            'frame_path': self.frame_path,
            'resolution': self.resolution,
            'capture_time': self.capture_time
        })
    
    @property
    def has_frame_data(self) -> bool:
        """Check if frame data is available."""
        return self.frame_data is not None
    
    @property
    def frame_size(self) -> Optional[int]:
        """Get frame data size in bytes."""
        if self.frame_data is not None:
            return self.frame_data.nbytes
        return None


@dataclass
class MotionEvent(PipelineEvent):
    """Motion detection event."""
    motion_regions: List[BoundingBox] = field(default_factory=list)
    motion_score: float = 0.0
    frame_event: Optional[FrameEvent] = None
    
    def __post_init__(self):
        self.data.update({
            'motion_regions': [region.__dict__ for region in self.motion_regions],
            'motion_score': self.motion_score,
            'frame_id': self.frame_event.frame_id if self.frame_event else None
        })


@dataclass
class FaceDetectionEvent(PipelineEvent):
    """Face detection event."""
    faces: List[FaceDetection] = field(default_factory=list)
    detection_time: float = 0.0
    detector_type: str = "unknown"
    frame_event: Optional[FrameEvent] = None
    
    def __post_init__(self):
        if self.faces:
            self.event_type = EventType.FACES_DETECTED
        else:
            self.event_type = EventType.NO_FACES_DETECTED
            
        self.data.update({
            'face_count': len(self.faces),
            'faces': [face.to_dict() for face in self.faces],
            'detection_time': self.detection_time,
            'detector_type': self.detector_type,
            'frame_id': self.frame_event.frame_id if self.frame_event else None
        })


@dataclass
class FaceRecognitionEvent(PipelineEvent):
    """Face recognition event."""
    recognitions: List[FaceRecognition] = field(default_factory=list)
    recognition_time: float = 0.0
    known_count: int = 0
    unknown_count: int = 0
    blacklisted_count: int = 0
    
    def __post_init__(self):
        # Count recognition statuses
        self.known_count = sum(1 for r in self.recognitions if r.status == RecognitionStatus.KNOWN)
        self.unknown_count = sum(1 for r in self.recognitions if r.status == RecognitionStatus.UNKNOWN)
        self.blacklisted_count = sum(1 for r in self.recognitions if r.status == RecognitionStatus.BLACKLISTED)
        
        # Set event type based on results
        if self.blacklisted_count > 0:
            self.event_type = EventType.FACE_BLACKLISTED
        elif self.known_count > 0:
            self.event_type = EventType.FACE_RECOGNIZED
        else:
            self.event_type = EventType.FACE_UNKNOWN
        
        self.data.update({
            'recognition_count': len(self.recognitions),
            'recognitions': [r.to_dict() for r in self.recognitions],
            'recognition_time': self.recognition_time,
            'known_count': self.known_count,
            'unknown_count': self.unknown_count,
            'blacklisted_count': self.blacklisted_count
        })


@dataclass
class NotificationEvent(PipelineEvent):
    """Notification event for alerts and messages."""
    message: str = ""
    notification_type: str = "info"
    image_path: Optional[str] = None
    recipient: Optional[str] = None
    sent_successfully: bool = False
    error_message: Optional[str] = None
    
    def __post_init__(self):
        self.data.update({
            'message': self.message,
            'notification_type': self.notification_type,
            'image_path': self.image_path,
            'recipient': self.recipient,
            'sent_successfully': self.sent_successfully,
            'error_message': self.error_message
        })


@dataclass
class SystemEvent(PipelineEvent):
    """System lifecycle and health events."""
    component: str = "system"
    status: str = "unknown"
    metrics: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None
    
    def __post_init__(self):
        self.data.update({
            'component': self.component,
            'status': self.status,
            'metrics': self.metrics,
            'error_details': self.error_details
        })


class EventHandler:
    """Base class for event handlers in the pipeline."""
    
    def __init__(self, handler_id: str):
        self.handler_id = handler_id
        self.processed_events = 0
        self.error_count = 0
        self.last_error: Optional[Exception] = None
    
    def handle_event(self, event: PipelineEvent) -> Optional[PipelineEvent]:
        """
        Process an event and optionally return a new event.
        
        Args:
            event: The event to process
            
        Returns:
            Optional new event to emit, or None
        """
        try:
            result = self._process_event(event)
            self.processed_events += 1
            return result
        except Exception as e:
            self.error_count += 1
            self.last_error = e
            self._handle_error(event, e)
            return None
    
    def _process_event(self, event: PipelineEvent) -> Optional[PipelineEvent]:
        """Override this method to implement event processing logic."""
        raise NotImplementedError("Subclasses must implement _process_event")
    
    def _handle_error(self, event: PipelineEvent, error: Exception) -> None:
        """Handle processing errors (override for custom error handling)."""
        import logging
        logger = logging.getLogger(self.__class__.__name__)
        logger.error(f"Error processing event {event.event_id}: {error}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler performance statistics."""
        return {
            'handler_id': self.handler_id,
            'processed_events': self.processed_events,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(1, self.processed_events),
            'last_error': str(self.last_error) if self.last_error else None
        }


class EventFilter:
    """Filter events based on criteria."""
    
    def __init__(self, 
                 event_types: Optional[List[EventType]] = None,
                 min_priority: Optional[EventPriority] = None,
                 source_filter: Optional[str] = None):
        self.event_types = set(event_types) if event_types else None
        self.min_priority = min_priority
        self.source_filter = source_filter
    
    def should_process(self, event: PipelineEvent) -> bool:
        """Check if event should be processed based on filter criteria."""
        # Event type filter
        if self.event_types and event.event_type not in self.event_types:
            return False
        
        # Priority filter
        if self.min_priority and event.priority.value < self.min_priority.value:
            return False
        
        # Source filter
        if self.source_filter and event.source != self.source_filter:
            return False
        
        return True


def create_doorbell_event(channel: int = 18) -> DoorbellEvent:
    """Convenience function to create a doorbell event."""
    return DoorbellEvent(
        event_type=EventType.DOORBELL_PRESSED,
        channel=channel,
        source='gpio_handler'
    )


def create_frame_event(frame_data: np.ndarray, frame_path: Optional[str] = None) -> FrameEvent:
    """Convenience function to create a frame event."""
    return FrameEvent(
        event_type=EventType.FRAME_CAPTURED,
        frame_data=frame_data,
        frame_path=frame_path,
        resolution=frame_data.shape[:2] if frame_data is not None else None,
        source='frame_capture'
    )


def create_notification_event(message: str, 
                            notification_type: str = "info",
                            image_path: Optional[str] = None) -> NotificationEvent:
    """Convenience function to create a notification event."""
    return NotificationEvent(
        event_type=EventType.NOTIFICATION_SENT,
        message=message,
        notification_type=notification_type,
        image_path=image_path,
        source='notification_handler'
    )