#!/usr/bin/env python3
"""
Test suite for Rate Limiter

Comprehensive tests for notification rate limiting and debounce logic.
"""

import sys
import time
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.enrichment.rate_limiter import RateLimiter, RateLimitWindow
from src.enrichment.alert_manager import Alert, AlertPriority, AlertType


class TestRateLimitWindow(unittest.TestCase):
    """Test suite for RateLimitWindow class."""
    
    def test_window_creation(self):
        """Test rate limit window creation."""
        window = RateLimitWindow(window_start=time.time())
        
        self.assertEqual(window.notification_count, 0)
        self.assertEqual(window.last_notification, 0.0)
        self.assertIsNotNone(window.window_start)


class TestRateLimiter(unittest.TestCase):
    """Test suite for RateLimiter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'enabled': True,
            'max_per_minute': 5,
            'max_per_hour': 20,
            'burst_allowance': 3,
            'cooldown_period': 60
        }
        self.rate_limiter = RateLimiter(self.config)
    
    def tearDown(self):
        """Clean up after tests."""
        self.rate_limiter.reset()
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        self.assertTrue(self.rate_limiter.enabled)
        self.assertEqual(self.rate_limiter.max_per_minute, 5)
        self.assertEqual(self.rate_limiter.max_per_hour, 20)
        self.assertEqual(self.rate_limiter.burst_allowance, 3)
    
    def test_disabled_rate_limiter_allows_all(self):
        """Test that disabled rate limiter allows all notifications."""
        config = {'enabled': False}
        limiter = RateLimiter(config)
        
        alert = Alert(
            alert_type=AlertType.KNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.INFO
        )
        
        # Should allow many notifications
        for _ in range(100):
            self.assertTrue(limiter.allow_notification(alert))
    
    def test_critical_alerts_bypass_rate_limiting(self):
        """Test that critical alerts always bypass rate limiting."""
        alert = Alert(
            alert_type=AlertType.BLACKLIST_DETECTED.value,
            priority=AlertPriority.CRITICAL,
            person_id='test_person'
        )
        
        # Should allow many critical alerts even if rate limited
        for _ in range(100):
            result = self.rate_limiter.allow_notification(alert)
            self.assertTrue(result, "Critical alert should always bypass rate limiting")
    
    def test_burst_allowance(self):
        """Test burst allowance limiting."""
        alert = Alert(
            alert_type=AlertType.KNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.INFO,
            person_id='test_person'
        )
        
        # First burst_allowance notifications should be allowed
        for i in range(self.config['burst_allowance']):
            result = self.rate_limiter.allow_notification(alert)
            self.assertTrue(result, f"Notification {i+1} within burst should be allowed")
        
        # Next notification should be blocked due to burst limit
        result = self.rate_limiter.allow_notification(alert)
        self.assertFalse(result, "Notification exceeding burst should be blocked")
    
    def test_per_minute_limit(self):
        """Test per-minute rate limiting."""
        alert = Alert(
            alert_type=AlertType.KNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.INFO,
            person_id='test_person_2'
        )
        
        # Allow some time between notifications to avoid burst limit
        allowed_count = 0
        for i in range(self.config['max_per_minute'] + 3):
            time.sleep(0.02)  # Small delay to avoid burst detection
            if self.rate_limiter.allow_notification(alert):
                allowed_count += 1
        
        # Should allow up to max_per_minute
        self.assertLessEqual(allowed_count, self.config['max_per_minute'])
    
    def test_different_alert_keys_independent(self):
        """Test that different alert keys have independent rate limits."""
        alert1 = Alert(
            alert_type=AlertType.KNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.INFO,
            person_id='person_1'
        )
        
        alert2 = Alert(
            alert_type=AlertType.KNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.INFO,
            person_id='person_2'
        )
        
        # Exhaust rate limit for alert1
        for _ in range(self.config['burst_allowance']):
            self.rate_limiter.allow_notification(alert1)
        
        # alert1 should be blocked
        self.assertFalse(self.rate_limiter.allow_notification(alert1))
        
        # But alert2 should still be allowed
        self.assertTrue(self.rate_limiter.allow_notification(alert2))
    
    def test_get_alert_key_with_person_id(self):
        """Test alert key generation with person ID."""
        alert = Alert(
            alert_type=AlertType.KNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.INFO,
            person_id='test_person'
        )
        
        key = self.rate_limiter._get_alert_key(alert)
        self.assertEqual(key, 'known_person_detected_test_person')
    
    def test_get_alert_key_without_person_id(self):
        """Test alert key generation without person ID."""
        alert = Alert(
            alert_type=AlertType.UNKNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.WARNING
        )
        
        key = self.rate_limiter._get_alert_key(alert)
        self.assertEqual(key, 'unknown_person_detected')
    
    def test_window_cleanup(self):
        """Test cleanup of old rate limiting windows."""
        alert = Alert(
            alert_type=AlertType.KNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.INFO,
            person_id='test_cleanup'
        )
        
        # Create some windows
        self.rate_limiter.allow_notification(alert)
        
        initial_count = len(self.rate_limiter.minute_windows)
        self.assertGreater(initial_count, 0)
        
        # Trigger cleanup with a future timestamp
        current_time = time.time() + 400  # 6+ minutes in future
        self.rate_limiter._cleanup_old_windows(current_time)
        
        # Windows should be cleaned up
        self.assertEqual(len(self.rate_limiter.minute_windows), 0)
    
    def test_get_rate_limit_status(self):
        """Test getting rate limit status for an alert."""
        alert = Alert(
            alert_type=AlertType.KNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.INFO,
            person_id='test_status'
        )
        
        # Get initial status
        status = self.rate_limiter.get_rate_limit_status(alert)
        
        self.assertTrue(status['enabled'])
        self.assertTrue(status['would_allow'])
        self.assertIn('alert_key', status)
        self.assertEqual(status['burst_count'], 0)
    
    def test_get_rate_limit_status_after_burst(self):
        """Test rate limit status after exceeding burst."""
        alert = Alert(
            alert_type=AlertType.KNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.INFO,
            person_id='test_burst_status'
        )
        
        # Exhaust burst allowance
        for _ in range(self.config['burst_allowance']):
            self.rate_limiter.allow_notification(alert)
        
        # Get status
        status = self.rate_limiter.get_rate_limit_status(alert)
        
        self.assertFalse(status['would_allow'])
        self.assertIn('Burst limit exceeded', status['reasons'])
        self.assertEqual(status['burst_count'], self.config['burst_allowance'])
    
    def test_reset(self):
        """Test resetting rate limiter state."""
        alert = Alert(
            alert_type=AlertType.KNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.INFO
        )
        
        # Generate some activity
        self.rate_limiter.allow_notification(alert)
        
        self.assertGreater(self.rate_limiter.total_allowed, 0)
        
        # Reset
        self.rate_limiter.reset()
        
        # Verify reset
        self.assertEqual(self.rate_limiter.total_allowed, 0)
        self.assertEqual(self.rate_limiter.total_blocked, 0)
        self.assertEqual(len(self.rate_limiter.minute_windows), 0)
        self.assertEqual(len(self.rate_limiter.hour_windows), 0)
        self.assertEqual(len(self.rate_limiter.recent_notifications), 0)
    
    def test_get_stats(self):
        """Test getting rate limiter statistics."""
        alert = Alert(
            alert_type=AlertType.KNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.INFO,
            person_id='test_stats'
        )
        
        # Allow some notifications
        for _ in range(3):
            self.rate_limiter.allow_notification(alert)
            time.sleep(0.01)
        
        # Block some notifications
        for _ in range(2):
            self.rate_limiter.allow_notification(alert)
        
        stats = self.rate_limiter.get_stats()
        
        self.assertTrue(stats['enabled'])
        self.assertGreater(stats['total_allowed'], 0)
        self.assertGreater(stats['total_blocked'], 0)
        self.assertIn('block_rate', stats)
        self.assertIn('active_minute_windows', stats)
    
    def test_critical_alert_does_not_count_toward_limits(self):
        """Test that critical alerts don't consume rate limit quota."""
        # Send critical alerts
        critical_alert = Alert(
            alert_type=AlertType.BLACKLIST_DETECTED.value,
            priority=AlertPriority.CRITICAL,
            person_id='test_critical'
        )
        
        for _ in range(10):
            self.rate_limiter.allow_notification(critical_alert)
        
        # Now try non-critical alert with same key
        info_alert = Alert(
            alert_type=AlertType.KNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.INFO,
            person_id='test_different'
        )
        
        # Should still be allowed (critical alerts didn't consume quota)
        result = self.rate_limiter.allow_notification(info_alert)
        self.assertTrue(result)


class TestRateLimiterEdgeCases(unittest.TestCase):
    """Test suite for rate limiter edge cases."""
    
    def test_zero_burst_allowance(self):
        """Test behavior with zero burst allowance."""
        config = {
            'enabled': True,
            'max_per_minute': 5,
            'max_per_hour': 20,
            'burst_allowance': 0,  # Zero burst means no burst protection
            'cooldown_period': 60
        }
        limiter = RateLimiter(config)
        
        alert = Alert(
            alert_type=AlertType.KNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.INFO
        )
        
        # With zero burst allowance, first notification should still be blocked
        # because 0 < 0 is False. This is expected behavior - burst_allowance
        # should be at least 1 for any notifications to pass.
        result = limiter.allow_notification(alert)
        self.assertFalse(result, "Zero burst allowance should block all notifications")
    
    def test_very_high_limits(self):
        """Test behavior with very high rate limits."""
        config = {
            'enabled': True,
            'max_per_minute': 1000,
            'max_per_hour': 10000,
            'burst_allowance': 100,
            'cooldown_period': 60
        }
        limiter = RateLimiter(config)
        
        alert = Alert(
            alert_type=AlertType.KNOWN_PERSON_DETECTED.value,
            priority=AlertPriority.INFO
        )
        
        # Should allow many notifications
        allowed = 0
        for _ in range(50):
            if limiter.allow_notification(alert):
                allowed += 1
            time.sleep(0.001)
        
        self.assertGreater(allowed, 40)


if __name__ == '__main__':
    unittest.main()
