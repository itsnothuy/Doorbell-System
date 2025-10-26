#!/usr/bin/env python3
"""
Internal Notification Handler

Privacy-first notification system with alert management, rate limiting,
and multi-channel delivery for doorbell security events.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

from src.pipeline.base_worker import PipelineWorker
from src.communication.message_bus import MessageBus, Message
from src.communication.events import PipelineEvent, EventType, FaceRecognitionEvent
from src.enrichment.alert_manager import AlertManager, Alert, AlertPriority
from src.enrichment.rate_limiter import RateLimiter
from src.enrichment.delivery_channels import DeliveryChannelManager
from src.storage.notification_database import NotificationDatabase

logger = logging.getLogger(__name__)


class NotificationStatus(Enum):
    """Notification delivery status values."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    AGGREGATED = "aggregated"


class NotificationHandler(PipelineWorker):
    """Internal notification handler with privacy-first design."""
    
    def __init__(self, message_bus: MessageBus, config: Dict[str, Any]):
        super().__init__(message_bus, config)
        
        # Core components
        self.alert_manager = AlertManager(config.get('alerts', {}))
        self.rate_limiter = RateLimiter(config.get('rate_limiting', {}))
        self.delivery_manager = DeliveryChannelManager(config.get('delivery', {}))
        self.notification_db = NotificationDatabase(
            config.get('storage', {}).get('db_path', 'data/notifications.db')
        )
        
        # Configuration
        self.enabled = config.get('enabled', True)
        self.debounce_period = config.get('debounce_period', 5.0)
        self.aggregation_window = config.get('aggregation_window', 30.0)
        self.max_notifications_per_hour = config.get('max_per_hour', 100)
        
        # Notification tracking
        self.notifications_sent = 0
        self.notifications_rate_limited = 0
        self.notifications_failed = 0
        self.recent_notifications = {}  # For deduplication
        
        logger.info(f"Initialized {self.worker_id} with {len(self.delivery_manager.channels)} delivery channels")
    
    def _setup_subscriptions(self):
        """Setup message bus subscriptions."""
        # Subscribe to face recognition events
        self.message_bus.subscribe('faces_recognized', self.handle_recognition_event, self.worker_id)
        
        # Subscribe to system events
        self.message_bus.subscribe('system_alerts', self.handle_system_alert, self.worker_id)
        self.message_bus.subscribe('component_errors', self.handle_error_alert, self.worker_id)
        
        # Subscribe to test notifications
        self.message_bus.subscribe('notification_test', self.handle_test_notification, self.worker_id)
        
        # Subscribe to shutdown signal
        self.message_bus.subscribe('system_shutdown', self.handle_shutdown, self.worker_id)
        
        logger.debug(f"{self.worker_id} subscriptions configured")
    
    def _initialize_worker(self):
        """Initialize notification system."""
        try:
            if not self.enabled:
                logger.info(f"{self.worker_id} disabled in configuration")
                return
            
            # Initialize database
            self.notification_db.initialize()
            
            # Initialize delivery channels
            self.delivery_manager.initialize()
            
            # Send startup notification
            self._send_startup_notification()
            
            logger.info(f"{self.worker_id} initialized successfully")
            
        except Exception as e:
            logger.error(f"{self.worker_id} initialization failed: {e}")
            raise
    
    def handle_recognition_event(self, message: Message):
        """Handle face recognition event and generate notifications."""
        if not self.enabled:
            return
            
        recognition_event = message.data
        
        try:
            logger.debug(f"Processing recognition event: {recognition_event.event_id}")
            
            # Classify recognition results into alerts
            alerts = self.alert_manager.classify_recognition_event(recognition_event)
            
            if not alerts:
                logger.debug(f"No alerts generated for recognition event {recognition_event.event_id}")
                return
            
            # Process each alert
            for alert in alerts:
                self._process_alert(alert, recognition_event)
            
            self.processed_count += 1
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Recognition event processing failed: {e}")
    
    def _process_alert(self, alert: Alert, recognition_event: FaceRecognitionEvent):
        """Process individual alert with rate limiting and delivery."""
        try:
            # Check if this is a duplicate recent notification
            if self._is_duplicate_notification(alert):
                logger.debug(f"Skipping duplicate notification for alert {alert.alert_id}")
                return
            
            # Apply rate limiting (except for critical alerts)
            if alert.priority != AlertPriority.CRITICAL:
                if not self.rate_limiter.allow_notification(alert):
                    self.notifications_rate_limited += 1
                    logger.debug(f"Rate limited alert {alert.alert_id}")
                    
                    # Store rate limited notification for history
                    self._store_notification(alert, NotificationStatus.RATE_LIMITED)
                    return
            
            # Generate notification content
            notification_content = self._generate_notification_content(alert, recognition_event)
            
            # Deliver notification
            delivery_success = self._deliver_notification(alert, notification_content)
            
            # Store notification result
            status = NotificationStatus.DELIVERED if delivery_success else NotificationStatus.FAILED
            self._store_notification(alert, status, notification_content)
            
            # Update metrics
            if delivery_success:
                self.notifications_sent += 1
                logger.info(f"Notification delivered for alert {alert.alert_id}: {alert.alert_type}")
            else:
                self.notifications_failed += 1
                logger.warning(f"Notification delivery failed for alert {alert.alert_id}")
            
            # Track recent notifications for deduplication
            self._track_recent_notification(alert)
            
        except Exception as e:
            logger.error(f"Alert processing failed for {alert.alert_id}: {e}")
            self.notifications_failed += 1
    
    def _is_duplicate_notification(self, alert: Alert) -> bool:
        """Check if this is a duplicate of a recent notification."""
        dedup_key = self._generate_deduplication_key(alert)
        
        if dedup_key in self.recent_notifications:
            last_time = self.recent_notifications[dedup_key]
            if time.time() - last_time < self.debounce_period:
                return True
        
        return False
    
    def _generate_deduplication_key(self, alert: Alert) -> str:
        """Generate key for notification deduplication."""
        # Use alert type as base key
        key = alert.alert_type
        
        # Add person identifier for person-specific deduplication
        if hasattr(alert, 'person_id') and alert.person_id:
            key = f"{key}_{alert.person_id}"
        
        return key
    
    def _track_recent_notification(self, alert: Alert):
        """Track notification for deduplication."""
        dedup_key = self._generate_deduplication_key(alert)
        self.recent_notifications[dedup_key] = time.time()
        
        # Clean up old entries
        cutoff_time = time.time() - self.debounce_period * 2
        self.recent_notifications = {
            k: v for k, v in self.recent_notifications.items() 
            if v > cutoff_time
        }
    
    def _generate_notification_content(self, alert: Alert, recognition_event: FaceRecognitionEvent) -> Dict[str, Any]:
        """Generate notification content based on alert type."""
        base_content = {
            'alert_id': alert.alert_id,
            'alert_type': alert.alert_type,
            'priority': alert.priority.value,
            'timestamp': datetime.now().isoformat(),
            'event_id': recognition_event.event_id if recognition_event else None
        }
        
        if alert.alert_type == 'blacklist_detected':
            return {
                **base_content,
                'title': '🚨 SECURITY ALERT - Blacklisted Person Detected',
                'message': f'A person on the blacklist has been detected at your door.',
                'person_id': getattr(alert, 'person_id', 'unknown'),
                'confidence': getattr(alert, 'confidence', 0.0),
                'immediate_action_required': True,
                'image_available': True
            }
        
        elif alert.alert_type == 'known_person_detected':
            person_name = getattr(alert, 'person_name', 'Known Person')
            return {
                **base_content,
                'title': f'👋 {person_name} is at the door',
                'message': f'{person_name} has been detected at your door.',
                'person_id': getattr(alert, 'person_id', 'unknown'),
                'person_name': person_name,
                'confidence': getattr(alert, 'confidence', 0.0),
                'image_available': True
            }
        
        elif alert.alert_type == 'unknown_person_detected':
            return {
                **base_content,
                'title': '👤 Unknown Person at Door',
                'message': 'An unrecognized person has been detected at your door.',
                'confidence': getattr(alert, 'confidence', 0.0),
                'image_available': True,
                'requires_identification': True
            }
        
        elif alert.alert_type == 'system_alert':
            return {
                **base_content,
                'title': f'⚠️ System Alert: {getattr(alert, "system_message", "Unknown")}',
                'message': getattr(alert, 'system_message', 'System alert triggered'),
                'component': getattr(alert, 'component', 'unknown'),
                'severity': getattr(alert, 'severity', 'warning')
            }
        
        else:
            return {
                **base_content,
                'title': 'Doorbell Security Alert',
                'message': 'A security event has been detected.',
                'raw_alert_data': alert.__dict__
            }
    
    def _deliver_notification(self, alert: Alert, content: Dict[str, Any]) -> bool:
        """Deliver notification through configured channels."""
        try:
            # Select delivery channels based on alert priority
            channels = self._select_delivery_channels(alert)
            
            delivery_results = []
            
            for channel in channels:
                try:
                    result = channel.deliver(content)
                    delivery_results.append(result)
                    logger.debug(f"Delivery via {channel.name}: {'success' if result else 'failed'}")
                except Exception as e:
                    logger.error(f"Delivery failed via {channel.name}: {e}")
                    delivery_results.append(False)
            
            # Consider successful if at least one channel succeeded
            success = any(delivery_results) if delivery_results else False
            
            return success
            
        except Exception as e:
            logger.error(f"Notification delivery failed: {e}")
            return False
    
    def _select_delivery_channels(self, alert: Alert) -> List:
        """Select appropriate delivery channels based on alert priority."""
        if alert.priority == AlertPriority.CRITICAL:
            # Critical alerts use all available channels
            return self.delivery_manager.get_all_channels()
        elif alert.priority == AlertPriority.WARNING:
            # Warnings use primary channels
            return self.delivery_manager.get_primary_channels()
        else:
            # Info alerts use minimal channels
            return self.delivery_manager.get_info_channels()
    
    def _store_notification(self, alert: Alert, status: NotificationStatus, content: Dict[str, Any] = None):
        """Store notification in database for history and audit."""
        try:
            notification_record = {
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type,
                'priority': alert.priority.value,
                'status': status.value,
                'timestamp': datetime.now().isoformat(),
                'content': content,
                'metadata': {
                    'person_id': getattr(alert, 'person_id', None),
                    'confidence': getattr(alert, 'confidence', None),
                    'processing_time': time.time() - getattr(alert, 'created_at', time.time())
                }
            }
            
            self.notification_db.store_notification(notification_record)
            
        except Exception as e:
            logger.error(f"Failed to store notification: {e}")
    
    def handle_system_alert(self, message: Message):
        """Handle system-level alerts."""
        if not self.enabled:
            return
            
        try:
            system_alert_data = message.data
            
            # Create system alert
            system_alert = self.alert_manager.create_system_alert(
                message=system_alert_data.get('message', 'System alert'),
                component=system_alert_data.get('component', 'unknown'),
                severity=system_alert_data.get('severity', 'warning')
            )
            
            # Process alert
            self._process_alert(system_alert, None)
            
        except Exception as e:
            logger.error(f"System alert processing failed: {e}")
    
    def handle_error_alert(self, message: Message):
        """Handle component error alerts."""
        if not self.enabled:
            return
            
        try:
            error_data = message.data
            
            # Create error alert (only for critical errors)
            if error_data.get('severity') == 'critical':
                error_alert = self.alert_manager.create_system_alert(
                    message=f"Critical error in {error_data.get('component', 'unknown')}: {error_data.get('error', 'Unknown error')}",
                    component=error_data.get('component', 'unknown'),
                    severity='critical'
                )
                
                self._process_alert(error_alert, None)
            
        except Exception as e:
            logger.error(f"Error alert processing failed: {e}")
    
    def handle_test_notification(self, message: Message):
        """Handle test notification requests."""
        try:
            test_data = message.data
            
            test_alert = self.alert_manager.create_test_alert(
                message=test_data.get('message', 'Test notification')
            )
            
            content = {
                'title': 'Test Notification',
                'message': test_data.get('message', 'This is a test notification from the doorbell security system.'),
                'timestamp': datetime.now().isoformat(),
                'test': True
            }
            
            success = self._deliver_notification(test_alert, content)
            
            logger.info(f"Test notification {'delivered' if success else 'failed'}")
            
        except Exception as e:
            logger.error(f"Test notification failed: {e}")
    
    def _send_startup_notification(self):
        """Send notification when system starts up."""
        try:
            startup_content = {
                'title': '🔄 Doorbell Security System Started',
                'message': 'Your doorbell security system has started successfully and is monitoring for activity.',
                'timestamp': datetime.now().isoformat(),
                'system_status': 'active'
            }
            
            # Use info channels for startup notification
            channels = self.delivery_manager.get_info_channels()
            
            for channel in channels:
                try:
                    channel.deliver(startup_content)
                except Exception as e:
                    logger.warning(f"Startup notification failed via {channel.name}: {e}")
            
        except Exception as e:
            logger.warning(f"Startup notification failed: {e}")
    
    def _cleanup_worker(self):
        """Cleanup notification handler resources."""
        try:
            # Send shutdown notification
            shutdown_content = {
                'title': '🔴 Doorbell Security System Stopping',
                'message': 'Your doorbell security system is shutting down.',
                'timestamp': datetime.now().isoformat(),
                'system_status': 'stopping'
            }
            
            channels = self.delivery_manager.get_info_channels()
            for channel in channels:
                try:
                    channel.deliver(shutdown_content)
                except:
                    pass  # Ignore errors during shutdown
            
            # Cleanup delivery channels
            self.delivery_manager.cleanup()
            
            # Close database
            if self.notification_db:
                self.notification_db.close()
            
            logger.info(f"{self.worker_id} cleanup completed")
            
        except Exception as e:
            logger.error(f"{self.worker_id} cleanup failed: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get notification handler metrics."""
        base_metrics = super().get_metrics()
        
        notification_metrics = {
            'notifications_sent': self.notifications_sent,
            'notifications_rate_limited': self.notifications_rate_limited,
            'notifications_failed': self.notifications_failed,
            'success_rate': self.notifications_sent / max(1, self.notifications_sent + self.notifications_failed),
            'rate_limit_effectiveness': self.notifications_rate_limited / max(1, self.processed_count) if self.processed_count > 0 else 0,
            'active_delivery_channels': len(self.delivery_manager.get_active_channels()),
            'recent_notifications_tracked': len(self.recent_notifications)
        }
        
        return {**base_metrics, **notification_metrics}
