#!/usr/bin/env python3
"""
Monitoring Configuration

Configuration for production monitoring, alerting, and observability.
"""

import os
from typing import Any, Dict, List


class MonitoringConfig:
    """Configuration for production monitoring system."""

    def __init__(self):
        """Initialize monitoring configuration."""
        # General monitoring settings
        self.enabled = os.getenv("MONITORING_ENABLED", "True").lower() == "true"
        self.prometheus_port = int(os.getenv("PROMETHEUS_PORT", "8000"))
        self.metrics_path = os.getenv("METRICS_PATH", "/metrics")

        # Prometheus settings
        self.prometheus_config = {
            "enabled": os.getenv("PROMETHEUS_ENABLED", "True").lower() == "true",
            "port": self.prometheus_port,
            "scrape_interval": int(os.getenv("PROMETHEUS_SCRAPE_INTERVAL", "15")),
            "retention_days": int(os.getenv("PROMETHEUS_RETENTION_DAYS", "15")),
        }

        # Distributed tracing settings
        self.tracing = {
            "enabled": os.getenv("TRACING_ENABLED", "False").lower() == "true",
            "agent_host": os.getenv("JAEGER_AGENT_HOST", "localhost"),
            "agent_port": int(os.getenv("JAEGER_AGENT_PORT", "6831")),
            "sample_rate": float(os.getenv("TRACING_SAMPLE_RATE", "0.1")),
        }

        # Alert configuration
        self.alerting = {
            "enabled": os.getenv("ALERTING_ENABLED", "True").lower() == "true",
            "alert_manager_url": os.getenv("ALERT_MANAGER_URL", "http://localhost:9093"),
            "notification_channels": self._parse_notification_channels(),
        }

        # Health check configuration
        self.health_checks = {
            "enabled": os.getenv("HEALTH_CHECKS_ENABLED", "True").lower() == "true",
            "interval": int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
            "timeout": int(os.getenv("HEALTH_CHECK_TIMEOUT", "10")),
            "cpu_threshold_warn": float(os.getenv("CPU_THRESHOLD_WARN", "80.0")),
            "cpu_threshold_critical": float(os.getenv("CPU_THRESHOLD_CRITICAL", "95.0")),
            "memory_threshold_warn": float(os.getenv("MEMORY_THRESHOLD_WARN", "80.0")),
            "memory_threshold_critical": float(os.getenv("MEMORY_THRESHOLD_CRITICAL", "95.0")),
            "disk_threshold_warn": float(os.getenv("DISK_THRESHOLD_WARN", "85.0")),
            "disk_threshold_critical": float(os.getenv("DISK_THRESHOLD_CRITICAL", "95.0")),
        }

        # Logging configuration
        self.logging = {
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "format": os.getenv("LOG_FORMAT", "json"),
            "structured": os.getenv("STRUCTURED_LOGGING", "True").lower() == "true",
            "audit_enabled": os.getenv("AUDIT_LOGGING", "True").lower() == "true",
        }

    def _parse_notification_channels(self) -> List[Dict[str, str]]:
        """Parse notification channels from environment."""
        channels_str = os.getenv("NOTIFICATION_CHANNELS", "")
        if not channels_str:
            return []

        channels = []
        for channel in channels_str.split(","):
            if ":" in channel:
                channel_type, channel_config = channel.split(":", 1)
                channels.append({"type": channel_type.strip(), "config": channel_config.strip()})

        return channels

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "enabled": self.enabled,
            "prometheus_port": self.prometheus_port,
            "metrics_path": self.metrics_path,
            "prometheus": self.prometheus_config,
            "tracing": self.tracing,
            "alerting": self.alerting,
            "health_checks": self.health_checks,
            "logging": self.logging,
        }


# Singleton instance
_monitoring_config = None


def get_monitoring_config() -> MonitoringConfig:
    """
    Get monitoring configuration singleton.

    Returns:
        MonitoringConfig instance
    """
    global _monitoring_config
    if _monitoring_config is None:
        _monitoring_config = MonitoringConfig()
    return _monitoring_config
