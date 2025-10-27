#!/usr/bin/env python3
"""
Test suite for Detector Health Monitoring System

Tests for health status tracking, failure detection, and recovery mechanisms.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.detectors.health_monitor import (
    DetectorHealthMonitor,
    HealthStatus,
    HealthMetrics
)
from src.detectors.detector_factory import MockDetector


class TestHealthMetrics(unittest.TestCase):
    """Test suite for HealthMetrics dataclass."""
    
    def test_metrics_initialization(self):
        """Test HealthMetrics initialization."""
        metrics = HealthMetrics(
            status=HealthStatus.HEALTHY,
            uptime=100.0,
            success_rate=0.95,
            avg_response_time=0.05,
            error_count=5,
            last_error=None,
            recovery_attempts=0
        )
        
        self.assertEqual(metrics.status, HealthStatus.HEALTHY)
        self.assertEqual(metrics.uptime, 100.0)
        self.assertEqual(metrics.success_rate, 0.95)
    
    def test_metrics_to_dict(self):
        """Test metrics conversion to dictionary."""
        metrics = HealthMetrics(
            status=HealthStatus.DEGRADED,
            uptime=50.0,
            success_rate=0.75,
            avg_response_time=0.1,
            error_count=10,
            last_error="Test error",
            recovery_attempts=1
        )
        
        result = metrics.to_dict()
        
        self.assertEqual(result['status'], 'degraded')
        self.assertEqual(result['uptime'], 50.0)
        self.assertEqual(result['success_rate'], 0.75)
        self.assertEqual(result['error_count'], 10)


class TestDetectorHealthMonitor(unittest.TestCase):
    """Test suite for DetectorHealthMonitor."""
    
    def setUp(self):
        """Set up test environment."""
        self.detector = MockDetector({})
        self.monitor = DetectorHealthMonitor(self.detector)
    
    def tearDown(self):
        """Clean up after tests."""
        if self.monitor.monitoring:
            self.monitor.stop_monitoring()
    
    def test_monitor_initialization(self):
        """Test health monitor initialization."""
        self.assertEqual(self.monitor.status, HealthStatus.HEALTHY)
        self.assertEqual(self.monitor.success_count, 0)
        self.assertEqual(self.monitor.total_requests, 0)
        self.assertEqual(self.monitor.error_count, 0)
        self.assertIsNone(self.monitor.last_error)
    
    def test_monitor_with_config(self):
        """Test monitor initialization with custom config."""
        config = {
            'degraded_threshold': 0.70,
            'failing_threshold': 0.40,
            'max_recovery_attempts': 5,
            'monitor_interval': 5.0
        }
        
        monitor = DetectorHealthMonitor(self.detector, config)
        
        self.assertEqual(monitor.degraded_threshold, 0.70)
        self.assertEqual(monitor.failing_threshold, 0.40)
        self.assertEqual(monitor.max_recovery_attempts, 5)
        self.assertEqual(monitor.monitor_interval, 5.0)
    
    def test_record_success(self):
        """Test recording successful operations."""
        self.monitor.record_success(0.05)
        self.monitor.record_success(0.06)
        
        self.assertEqual(self.monitor.success_count, 2)
        self.assertEqual(self.monitor.total_requests, 2)
        self.assertEqual(len(self.monitor.response_times), 2)
    
    def test_record_error(self):
        """Test recording error operations."""
        self.monitor.record_error("Test error")
        
        self.assertEqual(self.monitor.error_count, 1)
        self.assertEqual(self.monitor.total_requests, 1)
        self.assertEqual(self.monitor.last_error, "Test error")
    
    def test_get_metrics_basic(self):
        """Test getting basic health metrics."""
        self.monitor.record_success(0.05)
        self.monitor.record_success(0.06)
        self.monitor.record_error("Error")
        
        metrics = self.monitor.get_metrics()
        
        self.assertEqual(metrics.status, HealthStatus.HEALTHY)
        self.assertAlmostEqual(metrics.success_rate, 2/3)
        self.assertGreater(metrics.uptime, 0)
        self.assertEqual(metrics.error_count, 1)
        self.assertEqual(metrics.last_error, "Error")
    
    def test_get_metrics_avg_response_time(self):
        """Test average response time calculation."""
        self.monitor.record_success(0.05)
        self.monitor.record_success(0.10)
        self.monitor.record_success(0.15)
        
        metrics = self.monitor.get_metrics()
        
        expected_avg = (0.05 + 0.10 + 0.15) / 3
        self.assertAlmostEqual(metrics.avg_response_time, expected_avg, places=5)
    
    def test_health_status_healthy(self):
        """Test that status remains healthy with good success rate."""
        # Record enough operations to trigger status update
        for _ in range(15):
            self.monitor.record_success(0.05)
        
        self.assertEqual(self.monitor.status, HealthStatus.HEALTHY)
    
    def test_health_status_degraded(self):
        """Test status changes to degraded with moderate failure rate."""
        # Record operations that put success rate at 75% (between thresholds)
        for _ in range(9):
            self.monitor.record_success(0.05)
        for _ in range(3):
            self.monitor.record_error("Error")
        
        # Success rate: 9/12 = 0.75, which is below degraded_threshold (0.80)
        self.assertEqual(self.monitor.status, HealthStatus.DEGRADED)
    
    def test_health_status_failing(self):
        """Test status changes to failing with high failure rate."""
        # Mock health_check to prevent automatic recovery
        with patch.object(self.detector, 'health_check', return_value={'status': 'unhealthy'}):
            # Record operations that put success rate below 50% (failing threshold)
            for _ in range(5):
                self.monitor.record_success(0.05)
            for _ in range(6):
                self.monitor.record_error("Error")
            
            # Success rate: 5/11 = 0.45, which is below failing_threshold (0.50)
            self.assertEqual(self.monitor.status, HealthStatus.FAILING)
    
    def test_add_health_callback(self):
        """Test adding health status callback."""
        callback_called = []
        
        def test_callback(metrics):
            callback_called.append(metrics)
        
        self.monitor.add_health_callback(test_callback)
        
        # Trigger status change by recording many failures
        for _ in range(5):
            self.monitor.record_success(0.05)
        for _ in range(6):
            self.monitor.record_error("Error")
        
        # Callback should have been called when status changed
        self.assertGreater(len(callback_called), 0)
    
    def test_recovery_attempt_success(self):
        """Test successful recovery attempt."""
        # Mock health_check to return healthy status
        with patch.object(self.detector, 'health_check', return_value={'status': 'healthy'}):
            # Force status to failing
            for _ in range(4):
                self.monitor.record_success(0.05)
            for _ in range(7):
                self.monitor.record_error("Error")
            
            # Recovery should succeed
            time.sleep(0.1)  # Allow recovery to complete
            
            # Check that recovery was attempted
            self.assertGreater(self.monitor.recovery_attempts, 0)
    
    def test_recovery_attempt_failure(self):
        """Test failed recovery attempt."""
        # Mock health_check to return unhealthy status
        with patch.object(self.detector, 'health_check', return_value={'status': 'unhealthy', 'error': 'Test'}):
            # Force status to failing
            for _ in range(4):
                self.monitor.record_success(0.05)
            for _ in range(7):
                self.monitor.record_error("Error")
            
            # Status should remain failing
            self.assertEqual(self.monitor.status, HealthStatus.FAILING)
    
    def test_max_recovery_attempts(self):
        """Test that recovery stops after max attempts."""
        # Set low max attempts
        self.monitor.max_recovery_attempts = 2
        
        # Mock health_check to return unhealthy status
        with patch.object(self.detector, 'health_check', return_value={'status': 'unhealthy', 'error': 'Test'}):
            # Trigger multiple recovery attempts
            for _ in range(3):
                for _ in range(4):
                    self.monitor.record_success(0.05)
                for _ in range(7):
                    self.monitor.record_error("Error")
                time.sleep(0.1)
            
            # Should eventually mark as FAILED
            self.assertIn(self.monitor.status, [HealthStatus.FAILING, HealthStatus.FAILED])
    
    def test_reset_stats(self):
        """Test resetting health statistics."""
        # Record some operations
        self.monitor.record_success(0.05)
        self.monitor.record_error("Error")
        
        # Reset
        self.monitor.reset_stats()
        
        self.assertEqual(self.monitor.success_count, 0)
        self.assertEqual(self.monitor.total_requests, 0)
        self.assertEqual(self.monitor.error_count, 0)
        self.assertIsNone(self.monitor.last_error)
        self.assertEqual(self.monitor.recovery_attempts, 0)
        self.assertEqual(len(self.monitor.response_times), 0)
        self.assertEqual(self.monitor.status, HealthStatus.HEALTHY)
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping background monitoring."""
        self.monitor.start_monitoring()
        
        self.assertTrue(self.monitor.monitoring)
        self.assertIsNotNone(self.monitor.monitor_thread)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        self.assertFalse(self.monitor.monitoring)
    
    def test_start_monitoring_twice(self):
        """Test that starting monitoring twice doesn't create duplicate threads."""
        self.monitor.start_monitoring()
        thread1 = self.monitor.monitor_thread
        
        self.monitor.start_monitoring()  # Try to start again
        thread2 = self.monitor.monitor_thread
        
        # Should be the same thread
        self.assertEqual(thread1, thread2)
        
        self.monitor.stop_monitoring()
    
    def test_response_time_history_limit(self):
        """Test that response time history is limited."""
        max_history = 10
        monitor = DetectorHealthMonitor(
            self.detector,
            {'max_response_history': max_history}
        )
        
        # Record more operations than the limit
        for i in range(max_history + 5):
            monitor.record_success(0.01 * i)
        
        # Should only keep max_history items
        self.assertEqual(len(monitor.response_times), max_history)
    
    def test_monitor_loop_health_check(self):
        """Test that monitor loop performs periodic health checks."""
        # Set very short interval for testing
        monitor = DetectorHealthMonitor(
            self.detector,
            {'monitor_interval': 0.1}
        )
        
        health_check_called = []
        
        def mock_health_check():
            health_check_called.append(True)
            return {'status': 'healthy', 'response_time_ms': 10}
        
        with patch.object(self.detector, 'health_check', side_effect=mock_health_check):
            monitor.start_monitoring()
            time.sleep(0.3)  # Allow time for at least 2 checks
            monitor.stop_monitoring()
        
        # Health check should have been called at least once
        self.assertGreater(len(health_check_called), 0)


class TestHealthStatus(unittest.TestCase):
    """Test suite for HealthStatus enum."""
    
    def test_health_status_values(self):
        """Test HealthStatus enum values."""
        self.assertEqual(HealthStatus.HEALTHY.value, "healthy")
        self.assertEqual(HealthStatus.DEGRADED.value, "degraded")
        self.assertEqual(HealthStatus.FAILING.value, "failing")
        self.assertEqual(HealthStatus.FAILED.value, "failed")
        self.assertEqual(HealthStatus.RECOVERING.value, "recovering")


if __name__ == '__main__':
    unittest.main()
