#!/usr/bin/env python3
"""
Alert Manager - Alert Classification and Routing

Classifies recognition events into appropriate alerts based on recognition
results and manages alert metadata and priorities.
"""

import time
import uuid
import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    """Alert priority levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """Alert type categories."""
    KNOWN_PERSON_DETECTED = "known_person_detected"
    UNKNOWN_PERSON_DETECTED = "unknown_person_detected"
    BLACKLIST_DETECTED = "blacklist_detected"
    SYSTEM_ALERT = "system_alert"
    TEST_NOTIFICATION = "test_notification"


@dataclass
class Alert:
    """Represents an alert for notification delivery."""
    alert_type: str
    priority: AlertPriority
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # System alert specific fields
    system_message: Optional[str] = None
    component: Optional[str] = None
    severity: Optional[str] = None
    
    # Test notification specific fields
    test_message: Optional[str] = None


class AlertManager:
    """Manages alert classification and routing based on recognition results."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert manager.
        
        Args:
            config: Alert configuration dictionary
        """
        self.config = config
        
        # Alert type configurations
        self.blacklist_enabled = config.get('blacklist_detection', {}).get('enabled', True)
        self.known_person_enabled = config.get('known_person_detection', {}).get('enabled', True)
        self.unknown_person_enabled = config.get('unknown_person_detection', {}).get('enabled', True)
        self.system_alerts_enabled = config.get('system_alerts', {}).get('enabled', True)
        
        # Priority settings
        self.blacklist_priority = AlertPriority.CRITICAL
        self.known_person_priority = AlertPriority.INFO
        self.unknown_person_priority = AlertPriority.WARNING
        
        # Statistics
        self.alerts_created = 0
        self.alerts_by_type = {}
        
        logger.info(f"Alert manager initialized with config: {config}")
    
    def classify_recognition_event(self, recognition_event) -> List[Alert]:
        """
        Classify a face recognition event into appropriate alerts.
        
        Args:
            recognition_event: FaceRecognitionEvent with recognition results
            
        Returns:
            List of Alert objects for notification delivery
        """
        alerts = []
        
        try:
            # Extract recognition results from event
            recognitions = getattr(recognition_event, 'recognitions', [])
            
            if not recognitions:
                logger.debug("No recognitions in event, skipping alert generation")
                return alerts
            
            # Process each recognition result
            for recognition in recognitions:
                alert = self._create_alert_from_recognition(recognition, recognition_event)
                if alert:
                    alerts.append(alert)
                    self.alerts_created += 1
                    
                    # Track statistics
                    alert_type = alert.alert_type
                    self.alerts_by_type[alert_type] = self.alerts_by_type.get(alert_type, 0) + 1
            
            logger.debug(f"Generated {len(alerts)} alerts from recognition event {recognition_event.event_id}")
            
        except Exception as e:
            logger.error(f"Failed to classify recognition event: {e}")
        
        return alerts
    
    def _create_alert_from_recognition(self, recognition, recognition_event) -> Optional[Alert]:
        """
        Create an alert from a single recognition result.
        
        Args:
            recognition: FaceRecognition object
            recognition_event: Parent recognition event
            
        Returns:
            Alert object or None if alert should not be created
        """
        try:
            # Import RecognitionStatus from events module
            from src.communication.events import RecognitionStatus
            
            status = recognition.status
            identity = recognition.identity
            similarity_score = recognition.similarity_score
            
            # Blacklisted person detection (highest priority)
            if status == RecognitionStatus.BLACKLISTED and self.blacklist_enabled:
                return Alert(
                    alert_type=AlertType.BLACKLIST_DETECTED.value,
                    priority=self.blacklist_priority,
                    person_id=identity or "blacklist_unknown",
                    person_name=identity or "Blacklisted Person",
                    confidence=similarity_score,
                    metadata={
                        'event_id': recognition_event.event_id,
                        'recognition_time': getattr(recognition_event, 'recognition_time', 0),
                        'status': status.value
                    }
                )
            
            # Known person detection
            elif status == RecognitionStatus.KNOWN and self.known_person_enabled:
                return Alert(
                    alert_type=AlertType.KNOWN_PERSON_DETECTED.value,
                    priority=self.known_person_priority,
                    person_id=identity or "known_unknown",
                    person_name=identity or "Known Person",
                    confidence=similarity_score,
                    metadata={
                        'event_id': recognition_event.event_id,
                        'recognition_time': getattr(recognition_event, 'recognition_time', 0),
                        'status': status.value
                    }
                )
            
            # Unknown person detection
            elif status == RecognitionStatus.UNKNOWN and self.unknown_person_enabled:
                return Alert(
                    alert_type=AlertType.UNKNOWN_PERSON_DETECTED.value,
                    priority=self.unknown_person_priority,
                    person_id=None,
                    person_name=None,
                    confidence=similarity_score,
                    metadata={
                        'event_id': recognition_event.event_id,
                        'recognition_time': getattr(recognition_event, 'recognition_time', 0),
                        'status': status.value
                    }
                )
            
            else:
                logger.debug(f"No alert created for recognition status: {status}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create alert from recognition: {e}")
            return None
    
    def create_system_alert(self, message: str, component: str = "system", 
                          severity: str = "warning") -> Alert:
        """
        Create a system alert.
        
        Args:
            message: Alert message
            component: Component that generated the alert
            severity: Alert severity level
            
        Returns:
            System Alert object
        """
        priority_map = {
            'info': AlertPriority.INFO,
            'warning': AlertPriority.WARNING,
            'critical': AlertPriority.CRITICAL
        }
        
        priority = priority_map.get(severity.lower(), AlertPriority.WARNING)
        
        alert = Alert(
            alert_type=AlertType.SYSTEM_ALERT.value,
            priority=priority,
            system_message=message,
            component=component,
            severity=severity,
            metadata={
                'timestamp': time.time()
            }
        )
        
        self.alerts_created += 1
        self.alerts_by_type[AlertType.SYSTEM_ALERT.value] = \
            self.alerts_by_type.get(AlertType.SYSTEM_ALERT.value, 0) + 1
        
        return alert
    
    def create_test_alert(self, message: str = "Test notification") -> Alert:
        """
        Create a test alert.
        
        Args:
            message: Test message
            
        Returns:
            Test Alert object
        """
        alert = Alert(
            alert_type=AlertType.TEST_NOTIFICATION.value,
            priority=AlertPriority.INFO,
            test_message=message,
            metadata={
                'timestamp': time.time(),
                'test': True
            }
        )
        
        self.alerts_created += 1
        
        return alert
    
    def get_stats(self) -> Dict[str, Any]:
        """Get alert manager statistics."""
        return {
            'alerts_created': self.alerts_created,
            'alerts_by_type': self.alerts_by_type.copy(),
            'configuration': {
                'blacklist_enabled': self.blacklist_enabled,
                'known_person_enabled': self.known_person_enabled,
                'unknown_person_enabled': self.unknown_person_enabled,
                'system_alerts_enabled': self.system_alerts_enabled
            }
        }
