#!/usr/bin/env python3
"""
Rate Limiter - Notification Rate Limiting and Debounce

Implements rate limiting and debounce logic to prevent notification spam
while ensuring critical alerts are always delivered.
"""

import time
import logging
from typing import Dict, Any, Optional
from collections import deque
from dataclasses import dataclass

from .alert_manager import Alert, AlertPriority

logger = logging.getLogger(__name__)


@dataclass
class RateLimitWindow:
    """Tracks rate limiting for a specific window."""
    window_start: float
    notification_count: int = 0
    last_notification: float = 0.0


class RateLimiter:
    """Manages rate limiting and debounce for notifications."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limiting configuration dictionary
        """
        self.enabled = config.get('enabled', True)
        
        # Rate limiting settings
        self.max_per_minute = config.get('max_per_minute', 10)
        self.max_per_hour = config.get('max_per_hour', 100)
        self.burst_allowance = config.get('burst_allowance', 3)
        self.cooldown_period = config.get('cooldown_period', 60.0)
        
        # Tracking windows
        self.minute_windows: Dict[str, RateLimitWindow] = {}
        self.hour_windows: Dict[str, RateLimitWindow] = {}
        
        # Recent notifications for burst detection
        self.recent_notifications = deque(maxlen=100)
        
        # Statistics
        self.total_allowed = 0
        self.total_blocked = 0
        self.burst_blocks = 0
        self.rate_blocks = 0
        
        logger.info(f"Rate limiter initialized: max_per_minute={self.max_per_minute}, "
                   f"max_per_hour={self.max_per_hour}")
    
    def allow_notification(self, alert: Alert) -> bool:
        """
        Check if notification should be allowed based on rate limits.
        
        Critical priority alerts always bypass rate limiting.
        
        Args:
            alert: Alert to check
            
        Returns:
            True if notification should be sent, False if rate limited
        """
        if not self.enabled:
            self.total_allowed += 1
            return True
        
        # Critical alerts always bypass rate limiting
        if alert.priority == AlertPriority.CRITICAL:
            logger.debug(f"Critical alert {alert.alert_id} bypasses rate limiting")
            self._record_notification(alert)
            self.total_allowed += 1
            return True
        
        current_time = time.time()
        alert_key = self._get_alert_key(alert)
        
        # Check burst allowance
        if not self._check_burst_allowance(alert_key, current_time):
            logger.debug(f"Alert {alert.alert_id} blocked by burst limit")
            self.total_blocked += 1
            self.burst_blocks += 1
            return False
        
        # Check per-minute limit
        if not self._check_minute_limit(alert_key, current_time):
            logger.debug(f"Alert {alert.alert_id} blocked by per-minute limit")
            self.total_blocked += 1
            self.rate_blocks += 1
            return False
        
        # Check per-hour limit
        if not self._check_hour_limit(alert_key, current_time):
            logger.debug(f"Alert {alert.alert_id} blocked by per-hour limit")
            self.total_blocked += 1
            self.rate_blocks += 1
            return False
        
        # All checks passed, allow notification
        self._record_notification(alert)
        self.total_allowed += 1
        logger.debug(f"Alert {alert.alert_id} allowed by rate limiter")
        return True
    
    def _get_alert_key(self, alert: Alert) -> str:
        """
        Generate a key for tracking alert rate limits.
        
        Groups similar alerts together for rate limiting.
        """
        # Use alert type as base key
        key = alert.alert_type
        
        # Add person identifier for person-specific rate limiting
        if alert.person_id:
            key = f"{key}_{alert.person_id}"
        
        return key
    
    def _check_burst_allowance(self, alert_key: str, current_time: float) -> bool:
        """Check if alert exceeds burst allowance."""
        # Get recent notifications for this alert key
        recent_count = sum(
            1 for notif_time, notif_key in self.recent_notifications
            if notif_key == alert_key and current_time - notif_time < 10.0  # 10 second window
        )
        
        return recent_count < self.burst_allowance
    
    def _check_minute_limit(self, alert_key: str, current_time: float) -> bool:
        """Check per-minute rate limit."""
        if alert_key not in self.minute_windows:
            self.minute_windows[alert_key] = RateLimitWindow(window_start=current_time)
        
        window = self.minute_windows[alert_key]
        
        # Reset window if more than 60 seconds have passed
        if current_time - window.window_start >= 60.0:
            window.window_start = current_time
            window.notification_count = 0
        
        # Check if under limit
        if window.notification_count >= self.max_per_minute:
            return False
        
        # Increment counter
        window.notification_count += 1
        window.last_notification = current_time
        
        return True
    
    def _check_hour_limit(self, alert_key: str, current_time: float) -> bool:
        """Check per-hour rate limit."""
        if alert_key not in self.hour_windows:
            self.hour_windows[alert_key] = RateLimitWindow(window_start=current_time)
        
        window = self.hour_windows[alert_key]
        
        # Reset window if more than 3600 seconds have passed
        if current_time - window.window_start >= 3600.0:
            window.window_start = current_time
            window.notification_count = 0
        
        # Check if under limit
        if window.notification_count >= self.max_per_hour:
            return False
        
        # Increment counter
        window.notification_count += 1
        window.last_notification = current_time
        
        return True
    
    def _record_notification(self, alert: Alert):
        """Record that a notification was sent."""
        current_time = time.time()
        alert_key = self._get_alert_key(alert)
        
        self.recent_notifications.append((current_time, alert_key))
        
        # Clean up old windows periodically
        self._cleanup_old_windows(current_time)
    
    def _cleanup_old_windows(self, current_time: float):
        """Clean up old rate limiting windows to prevent memory growth."""
        # Clean up minute windows older than 5 minutes
        expired_minute_keys = [
            key for key, window in self.minute_windows.items()
            if current_time - window.window_start > 300.0
        ]
        for key in expired_minute_keys:
            del self.minute_windows[key]
        
        # Clean up hour windows older than 2 hours
        expired_hour_keys = [
            key for key, window in self.hour_windows.items()
            if current_time - window.window_start > 7200.0
        ]
        for key in expired_hour_keys:
            del self.hour_windows[key]
    
    def get_rate_limit_status(self, alert: Alert) -> Dict[str, Any]:
        """
        Get current rate limit status for an alert.
        
        Args:
            alert: Alert to check
            
        Returns:
            Dictionary with rate limit status information
        """
        current_time = time.time()
        alert_key = self._get_alert_key(alert)
        
        status = {
            'alert_key': alert_key,
            'enabled': self.enabled,
            'would_allow': True,
            'reasons': []
        }
        
        if not self.enabled or alert.priority == AlertPriority.CRITICAL:
            status['reasons'].append('Rate limiting disabled or critical alert')
            return status
        
        # Check burst
        recent_count = sum(
            1 for notif_time, notif_key in self.recent_notifications
            if notif_key == alert_key and current_time - notif_time < 10.0
        )
        status['burst_count'] = recent_count
        status['burst_limit'] = self.burst_allowance
        
        if recent_count >= self.burst_allowance:
            status['would_allow'] = False
            status['reasons'].append('Burst limit exceeded')
        
        # Check minute window
        if alert_key in self.minute_windows:
            window = self.minute_windows[alert_key]
            if current_time - window.window_start < 60.0:
                status['minute_count'] = window.notification_count
                status['minute_limit'] = self.max_per_minute
                
                if window.notification_count >= self.max_per_minute:
                    status['would_allow'] = False
                    status['reasons'].append('Per-minute limit exceeded')
        
        # Check hour window
        if alert_key in self.hour_windows:
            window = self.hour_windows[alert_key]
            if current_time - window.window_start < 3600.0:
                status['hour_count'] = window.notification_count
                status['hour_limit'] = self.max_per_hour
                
                if window.notification_count >= self.max_per_hour:
                    status['would_allow'] = False
                    status['reasons'].append('Per-hour limit exceeded')
        
        return status
    
    def reset(self):
        """Reset all rate limiting state (useful for testing)."""
        self.minute_windows.clear()
        self.hour_windows.clear()
        self.recent_notifications.clear()
        self.total_allowed = 0
        self.total_blocked = 0
        self.burst_blocks = 0
        self.rate_blocks = 0
        logger.info("Rate limiter reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        total_attempts = self.total_allowed + self.total_blocked
        block_rate = self.total_blocked / max(1, total_attempts)
        
        return {
            'enabled': self.enabled,
            'total_allowed': self.total_allowed,
            'total_blocked': self.total_blocked,
            'burst_blocks': self.burst_blocks,
            'rate_blocks': self.rate_blocks,
            'block_rate': block_rate,
            'active_minute_windows': len(self.minute_windows),
            'active_hour_windows': len(self.hour_windows),
            'recent_notification_count': len(self.recent_notifications)
        }
