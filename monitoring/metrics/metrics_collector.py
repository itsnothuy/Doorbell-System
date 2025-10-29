#!/usr/bin/env python3
"""
Metrics Collector

Collects and manages custom metrics for the doorbell security system.
"""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects custom application metrics.

    This class provides a simple interface for collecting application-specific
    metrics that complement the system-level metrics from Prometheus.
    """

    def __init__(self, monitoring_system: Optional[Any] = None):
        """
        Initialize metrics collector.

        Args:
            monitoring_system: Optional ProductionMonitoringSystem instance
        """
        self.monitoring_system = monitoring_system
        self.metrics_cache = {}
        self.last_collection_time = time.time()

    def collect_pipeline_metrics(self, pipeline_data: Dict[str, Any]) -> None:
        """
        Collect pipeline-specific metrics.

        Args:
            pipeline_data: Dictionary containing pipeline metrics
        """
        if not self.monitoring_system:
            return

        try:
            # Event processing metrics
            if "events_processed" in pipeline_data:
                self.monitoring_system.record_counter(
                    "pipeline_events_total",
                    pipeline_data["events_processed"],
                    {"event_type": pipeline_data.get("event_type", "unknown"), "status": "success"},
                )

            # Queue size metrics
            if "queue_sizes" in pipeline_data:
                for queue_name, size in pipeline_data["queue_sizes"].items():
                    self.monitoring_system.record_gauge(
                        "pipeline_queue_size",
                        size,
                        {"queue_name": queue_name, "worker_type": "pipeline"},
                    )

            # Worker health metrics
            if "worker_health" in pipeline_data:
                for worker_id, health in pipeline_data["worker_health"].items():
                    self.monitoring_system.record_gauge(
                        "pipeline_worker_health",
                        1.0 if health else 0.0,
                        {"worker_id": worker_id, "worker_type": "pipeline"},
                    )

        except Exception as e:
            logger.warning(f"Failed to collect pipeline metrics: {e}")

    def collect_face_recognition_metrics(self, recognition_data: Dict[str, Any]) -> None:
        """
        Collect face recognition metrics.

        Args:
            recognition_data: Dictionary containing face recognition metrics
        """
        if not self.monitoring_system:
            return

        try:
            # Recognition requests
            if "result_type" in recognition_data:
                confidence = recognition_data.get("confidence", 0.0)
                confidence_bucket = self._get_confidence_bucket(confidence)

                self.monitoring_system.record_counter(
                    "face_recognition_requests_total",
                    1.0,
                    {
                        "result_type": recognition_data["result_type"],
                        "confidence_bucket": confidence_bucket,
                    },
                )

            # Database size
            if "known_faces_count" in recognition_data:
                self.monitoring_system.record_gauge(
                    "known_faces_database_size", recognition_data["known_faces_count"]
                )

            # Accuracy
            if "accuracy" in recognition_data:
                self.monitoring_system.record_gauge(
                    "face_recognition_accuracy",
                    recognition_data["accuracy"],
                    {"model_version": recognition_data.get("model_version", "default")},
                )

        except Exception as e:
            logger.warning(f"Failed to collect face recognition metrics: {e}")

    def collect_doorbell_metrics(self, doorbell_data: Dict[str, Any]) -> None:
        """
        Collect doorbell event metrics.

        Args:
            doorbell_data: Dictionary containing doorbell event data
        """
        if not self.monitoring_system:
            return

        try:
            # Doorbell triggers
            if "trigger_source" in doorbell_data:
                time_of_day = self._get_time_of_day()

                self.monitoring_system.record_counter(
                    "doorbell_triggers_total",
                    1.0,
                    {
                        "trigger_source": doorbell_data["trigger_source"],
                        "time_of_day": time_of_day,
                    },
                )

        except Exception as e:
            logger.warning(f"Failed to collect doorbell metrics: {e}")

    def collect_notification_metrics(self, notification_data: Dict[str, Any]) -> None:
        """
        Collect notification metrics.

        Args:
            notification_data: Dictionary containing notification data
        """
        if not self.monitoring_system:
            return

        try:
            # Notifications sent
            if "channel" in notification_data:
                self.monitoring_system.record_counter(
                    "notifications_sent_total",
                    1.0,
                    {
                        "channel": notification_data["channel"],
                        "notification_type": notification_data.get("type", "unknown"),
                        "status": notification_data.get("status", "unknown"),
                    },
                )

        except Exception as e:
            logger.warning(f"Failed to collect notification metrics: {e}")

    def collect_error_metrics(self, error_data: Dict[str, Any]) -> None:
        """
        Collect error metrics.

        Args:
            error_data: Dictionary containing error data
        """
        if not self.monitoring_system:
            return

        try:
            # Error tracking
            if "component" in error_data:
                self.monitoring_system.record_counter(
                    "errors_total",
                    1.0,
                    {
                        "component": error_data["component"],
                        "error_type": error_data.get("error_type", "unknown"),
                        "severity": error_data.get("severity", "error"),
                    },
                )

        except Exception as e:
            logger.warning(f"Failed to collect error metrics: {e}")

    def collect_security_metrics(self, security_data: Dict[str, Any]) -> None:
        """
        Collect security event metrics.

        Args:
            security_data: Dictionary containing security event data
        """
        if not self.monitoring_system:
            return

        try:
            # Security events
            if "event_type" in security_data:
                self.monitoring_system.record_counter(
                    "security_events_total",
                    1.0,
                    {
                        "event_type": security_data["event_type"],
                        "source": security_data.get("source", "unknown"),
                        "action_taken": security_data.get("action_taken", "none"),
                    },
                )

        except Exception as e:
            logger.warning(f"Failed to collect security metrics: {e}")

    def _get_confidence_bucket(self, confidence: float) -> str:
        """Get confidence bucket label."""
        if confidence >= 0.9:
            return "high"
        elif confidence >= 0.7:
            return "medium"
        elif confidence >= 0.5:
            return "low"
        else:
            return "very_low"

    def _get_time_of_day(self) -> str:
        """Get time of day label."""
        from datetime import datetime

        hour = datetime.now().hour

        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"

    def get_cached_metrics(self) -> Dict[str, Any]:
        """Get cached metrics."""
        return self.metrics_cache.copy()

    def clear_cache(self) -> None:
        """Clear metrics cache."""
        self.metrics_cache.clear()
        self.last_collection_time = time.time()
