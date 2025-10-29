#!/usr/bin/env python3
"""
Tests for Health Checker
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from production.health.health_checker import (
    HealthChecker,
    HealthStatus,
    HealthCheckResult,
)


class TestHealthChecker:
    """Test suite for HealthChecker."""

    @pytest.fixture
    def health_config(self):
        """Create test health check configuration."""
        return {
            "cpu_threshold_warn": 80.0,
            "cpu_threshold_critical": 95.0,
            "memory_threshold_warn": 80.0,
            "memory_threshold_critical": 95.0,
            "disk_threshold_warn": 85.0,
            "disk_threshold_critical": 95.0,
        }

    @pytest.fixture
    def health_checker(self, health_config):
        """Create test health checker instance."""
        return HealthChecker(health_config)

    def test_initialization(self, health_checker):
        """Test health checker initializes correctly."""
        assert health_checker.cpu_threshold_warn == 80.0
        assert health_checker.memory_threshold_warn == 80.0
        assert health_checker.disk_threshold_warn == 85.0

    def test_register_custom_check(self, health_checker):
        """Test registering custom health check."""

        def custom_check():
            return HealthCheckResult(
                check_name="custom",
                status=HealthStatus.HEALTHY,
                message="OK",
                timestamp=datetime.now(),
            )

        health_checker.register_check("custom", custom_check)

        assert "custom" in health_checker.custom_checks

    @patch("production.health.health_checker.PSUTIL_AVAILABLE", True)
    @patch("production.health.health_checker.psutil")
    def test_check_system_resources_healthy(self, mock_psutil, health_checker):
        """Test system resource check when healthy."""
        # Mock healthy system
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.virtual_memory.return_value = Mock(
            percent=50.0, available=2 * 1024**3, used=2 * 1024**3
        )
        mock_psutil.disk_partitions.return_value = [
            Mock(mountpoint="/", fstype="ext4", device="/dev/sda1")
        ]
        mock_psutil.disk_usage.return_value = Mock(
            total=100 * 1024**3, used=50 * 1024**3, free=50 * 1024**3
        )

        result = health_checker.check_system_resources()

        assert result.status == HealthStatus.HEALTHY
        assert "healthy" in result.message.lower()

    @patch("production.health.health_checker.PSUTIL_AVAILABLE", True)
    @patch("production.health.health_checker.psutil")
    def test_check_system_resources_degraded(self, mock_psutil, health_checker):
        """Test system resource check when degraded."""
        # Mock degraded system (high CPU)
        mock_psutil.cpu_percent.return_value = 85.0
        mock_psutil.virtual_memory.return_value = Mock(
            percent=50.0, available=2 * 1024**3, used=2 * 1024**3
        )
        mock_psutil.disk_partitions.return_value = [
            Mock(mountpoint="/", fstype="ext4", device="/dev/sda1")
        ]
        mock_psutil.disk_usage.return_value = Mock(
            total=100 * 1024**3, used=50 * 1024**3, free=50 * 1024**3
        )

        result = health_checker.check_system_resources()

        assert result.status == HealthStatus.DEGRADED
        assert "high" in result.message.lower()

    @patch("production.health.health_checker.PSUTIL_AVAILABLE", True)
    @patch("production.health.health_checker.psutil")
    def test_check_system_resources_unhealthy(self, mock_psutil, health_checker):
        """Test system resource check when unhealthy."""
        # Mock unhealthy system (critical CPU)
        mock_psutil.cpu_percent.return_value = 98.0
        mock_psutil.virtual_memory.return_value = Mock(
            percent=50.0, available=2 * 1024**3, used=2 * 1024**3
        )
        mock_psutil.disk_partitions.return_value = [
            Mock(mountpoint="/", fstype="ext4", device="/dev/sda1")
        ]
        mock_psutil.disk_usage.return_value = Mock(
            total=100 * 1024**3, used=50 * 1024**3, free=50 * 1024**3
        )

        result = health_checker.check_system_resources()

        assert result.status == HealthStatus.UNHEALTHY
        assert "critical" in result.message.lower()

    @patch("production.health.health_checker.PSUTIL_AVAILABLE", False)
    def test_check_system_resources_unavailable(self, health_checker):
        """Test system resource check when psutil unavailable."""
        result = health_checker.check_system_resources()

        assert result.status == HealthStatus.UNKNOWN

    def test_check_application_components(self, health_checker):
        """Test application component health check."""
        result = health_checker.check_application_components()

        assert result.check_name == "application_components"
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.UNHEALTHY]

    def test_check_database_connectivity(self, health_checker):
        """Test database connectivity check."""
        result = health_checker.check_database_connectivity()

        assert result.check_name == "database_connectivity"
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.UNHEALTHY]

    def test_perform_all_checks(self, health_checker):
        """Test performing all health checks."""
        results = health_checker.perform_all_checks()

        # Should have at least built-in checks
        assert "system_resources" in results
        assert "application_components" in results
        assert "database_connectivity" in results

        # All results should be HealthCheckResult instances
        for result in results.values():
            assert isinstance(result, HealthCheckResult)

    def test_perform_all_checks_with_custom(self, health_checker):
        """Test performing all checks including custom checks."""

        def custom_check():
            return HealthCheckResult(
                check_name="custom",
                status=HealthStatus.HEALTHY,
                message="OK",
                timestamp=datetime.now(),
            )

        health_checker.register_check("custom", custom_check)

        results = health_checker.perform_all_checks()

        assert "custom" in results
        assert results["custom"].status == HealthStatus.HEALTHY

    @patch("production.health.health_checker.PSUTIL_AVAILABLE", True)
    @patch("production.health.health_checker.psutil")
    def test_get_overall_status_healthy(self, mock_psutil, health_checker):
        """Test overall status when all checks are healthy."""
        # Mock healthy system
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.virtual_memory.return_value = Mock(
            percent=50.0, available=2 * 1024**3, used=2 * 1024**3
        )
        mock_psutil.disk_partitions.return_value = [
            Mock(mountpoint="/", fstype="ext4", device="/dev/sda1")
        ]
        mock_psutil.disk_usage.return_value = Mock(
            total=100 * 1024**3, used=50 * 1024**3, free=50 * 1024**3
        )

        status = health_checker.get_overall_status()

        assert status == HealthStatus.HEALTHY

    @patch("production.health.health_checker.PSUTIL_AVAILABLE", True)
    @patch("production.health.health_checker.psutil")
    def test_get_health_report(self, mock_psutil, health_checker):
        """Test getting comprehensive health report."""
        # Mock healthy system
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.virtual_memory.return_value = Mock(
            percent=50.0, available=2 * 1024**3, used=2 * 1024**3
        )
        mock_psutil.disk_partitions.return_value = [
            Mock(mountpoint="/", fstype="ext4", device="/dev/sda1")
        ]
        mock_psutil.disk_usage.return_value = Mock(
            total=100 * 1024**3, used=50 * 1024**3, free=50 * 1024**3
        )

        report = health_checker.get_health_report()

        assert "overall_status" in report
        assert "timestamp" in report
        assert "checks" in report
        assert "summary" in report

        # Check summary
        summary = report["summary"]
        assert "total_checks" in summary
        assert "healthy" in summary
        assert "degraded" in summary
        assert "unhealthy" in summary


class TestHealthCheckResult:
    """Test suite for HealthCheckResult."""

    def test_health_check_result_creation(self):
        """Test creating health check result."""
        result = HealthCheckResult(
            check_name="test",
            status=HealthStatus.HEALTHY,
            message="All good",
            timestamp=datetime.now(),
            details={"key": "value"},
        )

        assert result.check_name == "test"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"
        assert result.details["key"] == "value"

    def test_health_check_result_to_dict(self):
        """Test converting health check result to dictionary."""
        timestamp = datetime.now()
        result = HealthCheckResult(
            check_name="test",
            status=HealthStatus.HEALTHY,
            message="All good",
            timestamp=timestamp,
        )

        data = result.to_dict()

        assert data["check_name"] == "test"
        assert data["status"] == "healthy"
        assert data["message"] == "All good"
        assert "timestamp" in data


class TestHealthStatus:
    """Test suite for HealthStatus."""

    def test_status_values(self):
        """Test health status enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"
