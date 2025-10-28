#!/usr/bin/env python3
"""
Detector Health Monitoring System

Real-time monitoring and recovery for detector instances.
Provides health status tracking, failure detection, and automatic recovery mechanisms.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum

from src.detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Detector health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class HealthMetrics:
    """Health monitoring metrics for a detector."""
    
    status: HealthStatus
    uptime: float
    success_rate: float
    avg_response_time: float
    error_count: int
    last_error: Optional[str]
    recovery_attempts: int
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'status': self.status.value,
            'uptime': self.uptime,
            'success_rate': self.success_rate,
            'avg_response_time_ms': self.avg_response_time * 1000,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'recovery_attempts': self.recovery_attempts,
            'timestamp': self.timestamp,
        }


class DetectorHealthMonitor:
    """
    Monitor detector health and performance in real-time.
    
    Tracks success rates, response times, and errors to determine detector health.
    Provides callbacks for health status changes and automatic recovery mechanisms.
    """
    
    def __init__(
        self,
        detector: BaseDetector,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize health monitor for a detector.
        
        Args:
            detector: Detector instance to monitor
            config: Optional configuration for monitoring parameters
        """
        self.detector = detector
        self.config = config or {}
        
        # Health state
        self.status = HealthStatus.HEALTHY
        self.start_time = time.time()
        self.success_count = 0
        self.total_requests = 0
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.recovery_attempts = 0
        
        # Response time tracking
        self.response_times: List[float] = []
        self.max_response_history = self.config.get('max_response_history', 100)
        
        # Thresholds
        self.degraded_threshold = self.config.get('degraded_threshold', 0.80)  # 80% success rate
        self.failing_threshold = self.config.get('failing_threshold', 0.50)    # 50% success rate
        self.max_recovery_attempts = self.config.get('max_recovery_attempts', 3)
        
        # Monitoring thread
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_interval = self.config.get('monitor_interval', 10.0)  # seconds
        
        # Callbacks
        self.health_callbacks: List[Callable[[HealthMetrics], None]] = []
        
        logger.info(
            f"Initialized health monitor for {detector.detector_type.value} detector"
        )
    
    def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self.monitoring:
            logger.warning("Health monitoring already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("Started health monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Stopped health monitoring")
    
    def record_success(self, response_time: float) -> None:
        """
        Record successful detection operation.
        
        Args:
            response_time: Time taken for the operation in seconds
        """
        self.success_count += 1
        self.total_requests += 1
        
        # Track response time
        self.response_times.append(response_time)
        if len(self.response_times) > self.max_response_history:
            self.response_times.pop(0)
        
        # Update status
        self._update_health_status()
    
    def record_error(self, error: str) -> None:
        """
        Record detection error.
        
        Args:
            error: Error message
        """
        self.error_count += 1
        self.total_requests += 1
        self.last_error = error
        
        logger.warning(f"Detector error recorded: {error}")
        
        # Update status
        self._update_health_status()
        
        # Trigger recovery if needed
        if self.status == HealthStatus.FAILING:
            self._attempt_recovery()
    
    def get_metrics(self) -> HealthMetrics:
        """
        Get current health metrics.
        
        Returns:
            HealthMetrics with current status
        """
        uptime = time.time() - self.start_time
        
        success_rate = (
            self.success_count / max(1, self.total_requests)
        )
        
        avg_response_time = (
            sum(self.response_times) / len(self.response_times)
            if self.response_times else 0.0
        )
        
        return HealthMetrics(
            status=self.status,
            uptime=uptime,
            success_rate=success_rate,
            avg_response_time=avg_response_time,
            error_count=self.error_count,
            last_error=self.last_error,
            recovery_attempts=self.recovery_attempts
        )
    
    def add_health_callback(
        self,
        callback: Callable[[HealthMetrics], None]
    ) -> None:
        """
        Add callback to be called on health status changes.
        
        Args:
            callback: Function to call with HealthMetrics
        """
        self.health_callbacks.append(callback)
    
    def _update_health_status(self) -> None:
        """Update health status based on current metrics."""
        if self.total_requests < 10:
            # Not enough data yet
            return
        
        success_rate = self.success_count / self.total_requests
        old_status = self.status
        
        # Determine new status
        if success_rate >= self.degraded_threshold:
            new_status = HealthStatus.HEALTHY
        elif success_rate >= self.failing_threshold:
            new_status = HealthStatus.DEGRADED
        else:
            new_status = HealthStatus.FAILING
        
        # Update status
        if new_status != old_status:
            self.status = new_status
            logger.info(
                f"Health status changed: {old_status.value} -> {new_status.value} "
                f"(success rate: {success_rate:.2%})"
            )
            
            # Notify callbacks
            self._notify_health_change()
    
    def _attempt_recovery(self) -> None:
        """Attempt to recover a failing detector."""
        if self.recovery_attempts >= self.max_recovery_attempts:
            logger.error("Max recovery attempts reached, marking as failed")
            self.status = HealthStatus.FAILED
            self._notify_health_change()
            return
        
        self.recovery_attempts += 1
        self.status = HealthStatus.RECOVERING
        
        logger.info(f"Attempting recovery (attempt {self.recovery_attempts})")
        
        try:
            # Run health check on detector
            health_result = self.detector.health_check()
            
            if health_result.get('status') == 'healthy':
                logger.info("Recovery successful")
                self.status = HealthStatus.HEALTHY
                self.error_count = 0  # Reset error count on successful recovery
                self._notify_health_change()
            else:
                logger.warning("Recovery failed, detector still unhealthy")
                self.status = HealthStatus.FAILING
        
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            self.status = HealthStatus.FAILING
    
    def _notify_health_change(self) -> None:
        """Notify all registered callbacks of health status change."""
        metrics = self.get_metrics()
        
        for callback in self.health_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"Health callback failed: {e}")
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        logger.info("Health monitor loop started")
        
        while self.monitoring:
            try:
                # Periodic health check
                time.sleep(self.monitor_interval)
                
                if not self.monitoring:
                    break
                
                # Run detector health check
                try:
                    health_result = self.detector.health_check()
                    
                    if health_result.get('status') != 'healthy':
                        logger.warning(
                            f"Periodic health check failed: {health_result.get('error', 'Unknown')}"
                        )
                        self.record_error(health_result.get('error', 'Health check failed'))
                    else:
                        # Health check counts as a successful operation
                        response_time = health_result.get('response_time_ms', 0) / 1000.0
                        self.record_success(response_time)
                
                except Exception as e:
                    logger.error(f"Health check error: {e}")
                    self.record_error(str(e))
            
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
        
        logger.info("Health monitor loop stopped")
    
    def reset_stats(self) -> None:
        """Reset health statistics."""
        self.success_count = 0
        self.total_requests = 0
        self.error_count = 0
        self.last_error = None
        self.recovery_attempts = 0
        self.response_times.clear()
        self.status = HealthStatus.HEALTHY
        
        logger.info("Health statistics reset")
