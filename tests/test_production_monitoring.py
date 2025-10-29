#!/usr/bin/env python3
"""
Tests for Production Monitoring System
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from monitoring.metrics.prometheus_config import (
    ProductionMonitoringSystem,
    MetricType,
    AlertSeverity,
    MetricDefinition,
    AlertRule,
)


class TestProductionMonitoringSystem:
    """Test suite for ProductionMonitoringSystem."""

    @pytest.fixture
    def monitoring_config(self):
        """Create test monitoring configuration."""
        return {
            "monitoring": {
                "enabled": True,
                "prometheus_port": 8000,
            },
            "tracing": {
                "enabled": False,
            },
        }

    @pytest.fixture
    def monitoring_system(self, monitoring_config):
        """Create test monitoring system."""
        return ProductionMonitoringSystem(monitoring_config)

    def test_initialization(self, monitoring_system):
        """Test monitoring system initializes correctly."""
        assert monitoring_system.enabled is True
        assert len(monitoring_system.metrics_definitions) > 0
        assert len(monitoring_system.alert_rules) > 0

    def test_metric_definitions(self, monitoring_system):
        """Test metric definitions are properly configured."""
        metric_names = [m.name for m in monitoring_system.metrics_definitions]

        # Check for key metrics
        assert "pipeline_events_total" in metric_names
        assert "system_cpu_usage_percent" in metric_names
        assert "face_recognition_requests_total" in metric_names
        assert "doorbell_triggers_total" in metric_names
        assert "errors_total" in metric_names

    def test_alert_rules(self, monitoring_system):
        """Test alert rules are properly configured."""
        rule_names = [r.name for r in monitoring_system.alert_rules]

        # Check for key alert rules
        assert "HighCPUUsage" in rule_names
        assert "HighMemoryUsage" in rule_names
        assert "PipelineWorkerDown" in rule_names
        assert "SecurityEvent" in rule_names

    def test_record_metric_disabled(self):
        """Test recording metrics when monitoring is disabled."""
        config = {"monitoring": {"enabled": False}}
        system = ProductionMonitoringSystem(config)

        # Should not raise exception
        system.record_metric("test_metric", 1.0)
        system.record_counter("test_counter", 1.0)
        system.record_gauge("test_gauge", 1.0)

    @patch("monitoring.metrics.prometheus_config.PSUTIL_AVAILABLE", True)
    @patch("monitoring.metrics.prometheus_config.psutil")
    def test_collect_system_metrics(self, mock_psutil, monitoring_system):
        """Test system metrics collection."""
        # Mock psutil functions
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.virtual_memory.return_value = Mock(
            used=1024**3,  # 1GB
            available=1024**3,  # 1GB
            percent=50.0,
        )
        mock_psutil.disk_partitions.return_value = [
            Mock(mountpoint="/", fstype="ext4", device="/dev/sda1")
        ]
        mock_psutil.disk_usage.return_value = Mock(
            total=100 * 1024**3,  # 100GB
            used=50 * 1024**3,  # 50GB
            free=50 * 1024**3,  # 50GB
        )

        # Collect metrics
        monitoring_system.collect_system_metrics()

        # Verify psutil was called
        mock_psutil.cpu_percent.assert_called_once()
        mock_psutil.virtual_memory.assert_called_once()

    @patch("monitoring.metrics.prometheus_config.PSUTIL_AVAILABLE", True)
    @patch("monitoring.metrics.prometheus_config.psutil")
    def test_evaluate_alert_condition_high_cpu(self, mock_psutil, monitoring_system):
        """Test alert condition evaluation for high CPU."""
        # Mock high CPU usage
        mock_psutil.cpu_percent.return_value = 85.0

        # Find HighCPUUsage rule
        rule = next(r for r in monitoring_system.alert_rules if r.name == "HighCPUUsage")

        # Evaluate condition
        result = monitoring_system._evaluate_alert_condition(rule)

        assert result is True

    @patch("monitoring.metrics.prometheus_config.PSUTIL_AVAILABLE", True)
    @patch("monitoring.metrics.prometheus_config.psutil")
    def test_evaluate_alert_condition_normal_cpu(self, mock_psutil, monitoring_system):
        """Test alert condition evaluation for normal CPU."""
        # Mock normal CPU usage
        mock_psutil.cpu_percent.return_value = 50.0

        # Find HighCPUUsage rule
        rule = next(r for r in monitoring_system.alert_rules if r.name == "HighCPUUsage")

        # Evaluate condition
        result = monitoring_system._evaluate_alert_condition(rule)

        assert result is False

    def test_get_monitoring_status(self, monitoring_system):
        """Test getting monitoring status."""
        status = monitoring_system.get_monitoring_status()

        assert "enabled" in status
        assert "running" in status
        assert "active_alerts" in status
        assert "metrics_collected" in status
        assert "system_health" in status

    @patch("monitoring.metrics.prometheus_config.PSUTIL_AVAILABLE", True)
    @patch("monitoring.metrics.prometheus_config.psutil")
    def test_calculate_health_score(self, mock_psutil, monitoring_system):
        """Test health score calculation."""
        # Test perfect health
        score = monitoring_system._calculate_health_score(0.0, 0.0)
        assert score == 1.0

        # Test degraded health
        score = monitoring_system._calculate_health_score(50.0, 50.0)
        assert 0.4 < score < 0.6

        # Test critical health
        score = monitoring_system._calculate_health_score(100.0, 100.0)
        assert score == 0.0

    def test_shutdown(self, monitoring_system):
        """Test monitoring system shutdown."""
        monitoring_system.running = True
        monitoring_system.shutdown()

        assert monitoring_system.running is False


class TestMetricDefinition:
    """Test suite for MetricDefinition."""

    def test_metric_definition_creation(self):
        """Test creating a metric definition."""
        metric = MetricDefinition(
            name="test_metric",
            metric_type=MetricType.COUNTER,
            description="Test metric",
            labels=["label1", "label2"],
        )

        assert metric.name == "test_metric"
        assert metric.metric_type == MetricType.COUNTER
        assert metric.description == "Test metric"
        assert len(metric.labels) == 2

    def test_histogram_with_buckets(self):
        """Test histogram metric with custom buckets."""
        buckets = [0.1, 0.5, 1.0, 5.0]
        metric = MetricDefinition(
            name="test_histogram",
            metric_type=MetricType.HISTOGRAM,
            description="Test histogram",
            buckets=buckets,
        )

        assert metric.buckets == buckets


class TestAlertRule:
    """Test suite for AlertRule."""

    def test_alert_rule_creation(self):
        """Test creating an alert rule."""
        rule = AlertRule(
            name="TestAlert",
            condition="test_metric > 100",
            severity=AlertSeverity.WARNING,
            description="Test alert",
            threshold=100.0,
            duration=60.0,
            cooldown=300.0,
        )

        assert rule.name == "TestAlert"
        assert rule.severity == AlertSeverity.WARNING
        assert rule.threshold == 100.0

    def test_alert_severity_levels(self):
        """Test different alert severity levels."""
        critical = AlertRule(
            name="CriticalAlert",
            condition="test",
            severity=AlertSeverity.CRITICAL,
            description="Critical",
            threshold=1.0,
        )

        warning = AlertRule(
            name="WarningAlert",
            condition="test",
            severity=AlertSeverity.WARNING,
            description="Warning",
            threshold=1.0,
        )

        info = AlertRule(
            name="InfoAlert",
            condition="test",
            severity=AlertSeverity.INFO,
            description="Info",
            threshold=1.0,
        )

        assert critical.severity == AlertSeverity.CRITICAL
        assert warning.severity == AlertSeverity.WARNING
        assert info.severity == AlertSeverity.INFO
