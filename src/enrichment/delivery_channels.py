#!/usr/bin/env python3
"""
Delivery Channels - Multi-Channel Notification Delivery

Implements various delivery channels for notifications with fallback
and error handling capabilities.
"""

import time
import logging
import json
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class DeliveryChannel(ABC):
    """Abstract base class for notification delivery channels."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize delivery channel.
        
        Args:
            name: Channel name
            config: Channel configuration
        """
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
        
        # Statistics
        self.deliveries_attempted = 0
        self.deliveries_succeeded = 0
        self.deliveries_failed = 0
        
        logger.info(f"Initialized delivery channel: {name}")
    
    @abstractmethod
    def deliver(self, content: Dict[str, Any]) -> bool:
        """
        Deliver notification content through this channel.
        
        Args:
            content: Notification content dictionary
            
        Returns:
            True if delivery succeeded, False otherwise
        """
        pass
    
    def is_available(self) -> bool:
        """Check if channel is available for delivery."""
        return self.enabled
    
    def get_stats(self) -> Dict[str, Any]:
        """Get channel delivery statistics."""
        success_rate = self.deliveries_succeeded / max(1, self.deliveries_attempted)
        
        return {
            'name': self.name,
            'enabled': self.enabled,
            'deliveries_attempted': self.deliveries_attempted,
            'deliveries_succeeded': self.deliveries_succeeded,
            'deliveries_failed': self.deliveries_failed,
            'success_rate': success_rate
        }


class WebInterfaceChannel(DeliveryChannel):
    """Delivery channel for web interface real-time notifications."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('web_interface', config)
        
        self.endpoint = config.get('endpoint', '/api/notifications')
        self.real_time = config.get('real_time', True)
        self.notification_queue = []
        self.max_queue_size = config.get('max_queue_size', 100)
    
    def deliver(self, content: Dict[str, Any]) -> bool:
        """Deliver notification to web interface."""
        self.deliveries_attempted += 1
        
        try:
            # Add to notification queue for web interface to poll
            notification = {
                **content,
                'channel': self.name,
                'delivered_at': datetime.now().isoformat()
            }
            
            self.notification_queue.append(notification)
            
            # Maintain queue size limit
            if len(self.notification_queue) > self.max_queue_size:
                self.notification_queue.pop(0)
            
            self.deliveries_succeeded += 1
            logger.debug(f"Web interface notification queued: {content.get('title', 'Untitled')}")
            return True
            
        except Exception as e:
            self.deliveries_failed += 1
            logger.error(f"Web interface delivery failed: {e}")
            return False
    
    def get_notifications(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent notifications for web interface."""
        return self.notification_queue[-limit:]
    
    def clear_notifications(self):
        """Clear notification queue."""
        self.notification_queue.clear()


class FileLogChannel(DeliveryChannel):
    """Delivery channel for file-based notification logging."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('file_log', config)
        
        self.log_path = Path(config.get('path', 'data/logs/notifications.log'))
        self.format = config.get('format', 'json')
        
        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def deliver(self, content: Dict[str, Any]) -> bool:
        """Deliver notification to log file."""
        self.deliveries_attempted += 1
        
        try:
            # Prepare log entry
            log_entry = {
                **content,
                'channel': self.name,
                'logged_at': datetime.now().isoformat()
            }
            
            # Write to file
            with open(self.log_path, 'a') as f:
                if self.format == 'json':
                    f.write(json.dumps(log_entry) + '\n')
                else:
                    # Simple text format
                    timestamp = log_entry.get('timestamp', datetime.now().isoformat())
                    title = log_entry.get('title', 'Notification')
                    message = log_entry.get('message', '')
                    f.write(f"[{timestamp}] {title}: {message}\n")
            
            self.deliveries_succeeded += 1
            logger.debug(f"Notification logged to file: {self.log_path}")
            return True
            
        except Exception as e:
            self.deliveries_failed += 1
            logger.error(f"File log delivery failed: {e}")
            return False


class ConsoleChannel(DeliveryChannel):
    """Delivery channel for console/stdout notifications."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('console', config)
        self.use_colors = config.get('use_colors', True)
    
    def deliver(self, content: Dict[str, Any]) -> bool:
        """Deliver notification to console."""
        self.deliveries_attempted += 1
        
        try:
            title = content.get('title', 'Notification')
            message = content.get('message', '')
            priority = content.get('priority', 'info')
            
            # Format with colors if enabled
            if self.use_colors:
                color_codes = {
                    'info': '\033[94m',      # Blue
                    'warning': '\033[93m',   # Yellow
                    'critical': '\033[91m',  # Red
                }
                reset_code = '\033[0m'
                
                color = color_codes.get(priority, '\033[0m')
                print(f"{color}[{priority.upper()}] {title}{reset_code}")
                print(f"{message}")
            else:
                print(f"[{priority.upper()}] {title}")
                print(f"{message}")
            
            self.deliveries_succeeded += 1
            return True
            
        except Exception as e:
            self.deliveries_failed += 1
            logger.error(f"Console delivery failed: {e}")
            return False


class MockDeliveryChannel(DeliveryChannel):
    """Mock delivery channel for testing."""
    
    def __init__(self, name: str = 'mock', config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config or {})
        self.delivered_notifications = []
    
    def deliver(self, content: Dict[str, Any]) -> bool:
        """Mock delivery that stores notifications."""
        self.deliveries_attempted += 1
        
        try:
            self.delivered_notifications.append({
                **content,
                'delivered_at': time.time()
            })
            
            self.deliveries_succeeded += 1
            logger.debug(f"Mock notification delivered: {content.get('title', 'Untitled')}")
            return True
            
        except Exception as e:
            self.deliveries_failed += 1
            logger.error(f"Mock delivery failed: {e}")
            return False
    
    def get_delivered_notifications(self) -> List[Dict[str, Any]]:
        """Get all delivered notifications (for testing)."""
        return self.delivered_notifications.copy()
    
    def clear(self):
        """Clear delivered notifications."""
        self.delivered_notifications.clear()


class DeliveryChannelManager:
    """Manages multiple delivery channels with prioritization and fallback."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize delivery channel manager.
        
        Args:
            config: Delivery configuration with channel settings
        """
        self.config = config
        self.channels: List[DeliveryChannel] = []
        
        # Initialize channels based on configuration
        self._initialize_channels()
        
        logger.info(f"Delivery channel manager initialized with {len(self.channels)} channels")
    
    def _initialize_channels(self):
        """Initialize delivery channels from configuration."""
        # Web interface channel
        if self.config.get('web_interface', {}).get('enabled', True):
            self.channels.append(WebInterfaceChannel(self.config.get('web_interface', {})))
        
        # File log channel
        if self.config.get('file_log', {}).get('enabled', True):
            self.channels.append(FileLogChannel(self.config.get('file_log', {})))
        
        # Console channel (useful for development)
        if self.config.get('console', {}).get('enabled', False):
            self.channels.append(ConsoleChannel(self.config.get('console', {})))
    
    def initialize(self):
        """Initialize all delivery channels."""
        logger.info(f"Initializing {len(self.channels)} delivery channels")
        
        for channel in self.channels:
            if channel.is_available():
                logger.info(f"Channel '{channel.name}' ready")
    
    def cleanup(self):
        """Cleanup delivery channel resources."""
        logger.info("Cleaning up delivery channels")
        # Any cleanup needed for channels can be added here
    
    def get_all_channels(self) -> List[DeliveryChannel]:
        """Get all available channels."""
        return [ch for ch in self.channels if ch.is_available()]
    
    def get_primary_channels(self) -> List[DeliveryChannel]:
        """Get primary delivery channels (web interface, file log)."""
        primary_names = ['web_interface', 'file_log']
        return [ch for ch in self.channels 
                if ch.is_available() and ch.name in primary_names]
    
    def get_info_channels(self) -> List[DeliveryChannel]:
        """Get channels for info-level notifications (minimal)."""
        # For info notifications, just use file log
        return [ch for ch in self.channels 
                if ch.is_available() and ch.name == 'file_log']
    
    def get_active_channels(self) -> List[DeliveryChannel]:
        """Get list of active (enabled) channels."""
        return [ch for ch in self.channels if ch.enabled]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get delivery statistics for all channels."""
        return {
            'total_channels': len(self.channels),
            'active_channels': len(self.get_active_channels()),
            'channels': [ch.get_stats() for ch in self.channels]
        }
