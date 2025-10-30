#!/usr/bin/env python3
"""
Tests for Test Orchestrator Framework

Comprehensive tests for the centralized testing framework orchestrator.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from tests.framework.orchestrator import (
    TestConfiguration,
    TestEnvironment,
    TestExecutionResult,
    TestOrchestrator,
    TestSuite,
    TestSuiteResult,
)
from tests.framework.performance import PerformanceMetrics, PerformanceRegressor


class TestTestOrchestrator:
    """Test suite for TestOrchestrator class."""

    def test_orchestrator_initialization(self):
        """Test orchestrator initializes correctly."""
        config = TestConfiguration(
            suites={TestSuite.UNIT},
            environment=TestEnvironment.LOCAL,
            parallel_workers=2,
        )

        orchestrator = TestOrchestrator(config)

        assert orchestrator.config == config
        assert orchestrator.project_root.exists()
        assert orchestrator.test_root.exists()
        assert orchestrator.config.output_dir.exists()

    def test_build_pytest_command_unit(self):
        """Test pytest command building for unit tests."""
        config = TestConfiguration(
            suites={TestSuite.UNIT},
            verbose=True,
            parallel_workers=4,
            coverage_analysis=True,
        )

        orchestrator = TestOrchestrator(config)
        cmd = orchestrator._build_pytest_command(TestSuite.UNIT)

        assert "pytest" in cmd[2]  # ['python', '-m', 'pytest', ...]
        assert "-v" in cmd
        assert "--cov=src" in cmd
        assert any("unit" in str(arg) for arg in cmd)

    def test_build_pytest_command_performance(self):
        """Test pytest command building for performance tests."""
        config = TestConfiguration(
            suites={TestSuite.PERFORMANCE}, verbose=False, coverage_analysis=False
        )

        orchestrator = TestOrchestrator(config)
        cmd = orchestrator._build_pytest_command(TestSuite.PERFORMANCE)

        assert "pytest" in cmd[2]  # ['python', '-m', 'pytest', ...]
        assert "-q" in cmd
        assert "--cov=src" not in cmd
        assert any("performance" in str(arg) for arg in cmd)

    def test_determine_overall_status_passed(self):
        """Test overall status determination for passed tests."""
        config = TestConfiguration()
        orchestrator = TestOrchestrator(config)

        suite_results = [
            TestSuiteResult(
                suite=TestSuite.UNIT,
                total_tests=10,
                passed=10,
                failed=0,
                skipped=0,
                errors=0,
                duration=5.0,
            ),
            TestSuiteResult(
                suite=TestSuite.INTEGRATION,
                total_tests=5,
                passed=5,
                failed=0,
                skipped=0,
                errors=0,
                duration=3.0,
            ),
        ]

        status = orchestrator._determine_overall_status(suite_results)
        assert status == "passed"

    def test_determine_overall_status_failed(self):
        """Test overall status determination for failed tests."""
        config = TestConfiguration()
        orchestrator = TestOrchestrator(config)

        suite_results = [
            TestSuiteResult(
                suite=TestSuite.UNIT,
                total_tests=10,
                passed=8,
                failed=2,
                skipped=0,
                errors=0,
                duration=5.0,
            ),
        ]

        status = orchestrator._determine_overall_status(suite_results)
        assert status == "failed"

    def test_determine_overall_status_no_tests(self):
        """Test overall status determination for no tests."""
        config = TestConfiguration()
        orchestrator = TestOrchestrator(config)

        status = orchestrator._determine_overall_status([])
        assert status == "no_tests"

    def test_parse_stdout(self):
        """Test parsing pytest stdout output."""
        config = TestConfiguration()
        orchestrator = TestOrchestrator(config)

        stdout = """
        collected 15 items

        tests/test_example.py::test_one PASSED
        tests/test_example.py::test_two FAILED
        tests/test_example.py::test_three SKIPPED

        ===== 10 passed, 3 failed, 2 skipped in 5.5s =====
        """

        result = orchestrator._parse_stdout(TestSuite.UNIT, 5.5, 1, stdout)

        assert result.suite == TestSuite.UNIT
        assert result.passed == 10
        assert result.failed == 3
        assert result.skipped == 2
        assert result.duration == 5.5

    @pytest.mark.asyncio
    async def test_generate_html_report(self, tmp_path):
        """Test HTML report generation."""
        config = TestConfiguration(output_dir=tmp_path)
        orchestrator = TestOrchestrator(config)

        result = TestExecutionResult(
            configuration=config,
            start_time=time.time() - 10,
            end_time=time.time(),
            total_duration=10.0,
            suite_results=[
                TestSuiteResult(
                    suite=TestSuite.UNIT,
                    total_tests=10,
                    passed=8,
                    failed=2,
                    skipped=0,
                    errors=0,
                    duration=5.0,
                )
            ],
            overall_status="failed",
        )

        html_path = await orchestrator._generate_html_report(result)

        assert html_path.exists()
        assert html_path.suffix == ".html"

        content = html_path.read_text()
        assert "Test Execution Report" in content
        assert "UNIT" in content
        assert "10" in content  # total tests
        assert "8" in content  # passed
        assert "2" in content  # failed

    @pytest.mark.asyncio
    async def test_generate_json_report(self, tmp_path):
        """Test JSON report generation."""
        config = TestConfiguration(output_dir=tmp_path)
        orchestrator = TestOrchestrator(config)

        result = TestExecutionResult(
            configuration=config,
            start_time=time.time() - 10,
            end_time=time.time(),
            total_duration=10.0,
            suite_results=[
                TestSuiteResult(
                    suite=TestSuite.UNIT,
                    total_tests=10,
                    passed=8,
                    failed=2,
                    skipped=0,
                    errors=0,
                    duration=5.0,
                )
            ],
            overall_status="failed",
        )

        json_path = await orchestrator._generate_json_report(result)

        assert json_path.exists()
        assert json_path.suffix == ".json"

        data = json.loads(json_path.read_text())
        assert "overall_status" in data
        assert data["overall_status"] == "failed"
        assert "suite_results" in data
        assert len(data["suite_results"]) == 1


class TestPerformanceRegressor:
    """Test suite for PerformanceRegressor class."""

    def test_performance_regressor_initialization(self, tmp_path):
        """Test performance regressor initializes correctly."""
        regressor = PerformanceRegressor(baseline_path=tmp_path)

        assert regressor.baseline_path == tmp_path
        assert regressor.baseline_path.exists()
        assert isinstance(regressor.baselines, dict)

    def test_measure_performance(self, tmp_path):
        """Test performance measurement."""
        regressor = PerformanceRegressor(baseline_path=tmp_path)

        def test_function():
            time.sleep(0.1)
            return "result"

        metrics = regressor.measure_performance("test_func", test_function)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.duration >= 0.1
        assert metrics.memory_peak >= 0
        assert metrics.cpu_avg >= 0

    def test_check_regression_no_baseline(self, tmp_path):
        """Test regression check with no baseline."""
        regressor = PerformanceRegressor(baseline_path=tmp_path)

        metrics = PerformanceMetrics(
            duration=1.0,
            memory_peak=100000,
            memory_avg=90000,
            cpu_peak=50.0,
            cpu_avg=30.0,
        )

        result = regressor.check_regression("test_no_baseline", metrics)

        assert not result["regression_detected"]
        assert result["reason"] == "no_baseline"

    def test_update_and_check_baseline(self, tmp_path):
        """Test baseline update and regression check."""
        regressor = PerformanceRegressor(baseline_path=tmp_path)

        # Create baseline
        baseline_metrics = [
            PerformanceMetrics(
                duration=1.0,
                memory_peak=100000,
                memory_avg=90000,
                cpu_peak=50.0,
                cpu_avg=30.0,
            )
            for _ in range(5)
        ]

        regressor.update_baseline("test_baseline", baseline_metrics)

        # Check against baseline - no regression
        good_metrics = PerformanceMetrics(
            duration=1.05,
            memory_peak=102000,
            memory_avg=91000,
            cpu_peak=51.0,
            cpu_avg=31.0,
        )

        result = regressor.check_regression("test_baseline", good_metrics)
        assert not result["regression_detected"]

        # Check against baseline - with regression
        bad_metrics = PerformanceMetrics(
            duration=1.5,  # 50% slower
            memory_peak=150000,  # 50% more memory
            memory_avg=135000,
            cpu_peak=75.0,  # 50% more CPU
            cpu_avg=60.0,
        )

        result = regressor.check_regression(
            "test_baseline", bad_metrics, threshold=0.15
        )
        assert result["regression_detected"]
        assert len(result["regressions"]) > 0

    def test_baseline_persistence(self, tmp_path):
        """Test baseline save and load."""
        baseline_path = tmp_path / "baselines"
        regressor1 = PerformanceRegressor(baseline_path=baseline_path)

        # Create and save baseline
        metrics = [
            PerformanceMetrics(
                duration=1.0,
                memory_peak=100000,
                memory_avg=90000,
                cpu_peak=50.0,
                cpu_avg=30.0,
            )
        ]

        regressor1.update_baseline("test_persist", metrics)

        # Load in new instance
        regressor2 = PerformanceRegressor(baseline_path=baseline_path)

        assert "test_persist" in regressor2.baselines
        assert regressor2.baselines["test_persist"].mean_duration == 1.0

    def test_benchmark_function(self, tmp_path):
        """Test function benchmarking."""
        regressor = PerformanceRegressor(baseline_path=tmp_path)

        def test_function():
            time.sleep(0.01)
            return "result"

        result = regressor.benchmark_function(
            test_name="bench_test", test_func=test_function, iterations=3, warmup=1
        )

        assert "test_name" in result
        assert result["iterations"] == 3
        assert "mean_duration" in result
        assert result["mean_duration"] >= 0.01
        assert "metrics" in result
        assert len(result["metrics"]) == 3


class TestTestConfiguration:
    """Test suite for TestConfiguration class."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = TestConfiguration()

        assert TestSuite.ALL in config.suites
        assert config.environment == TestEnvironment.LOCAL
        assert config.parallel_workers == 4
        assert config.timeout_seconds == 3600
        assert config.generate_reports is True
        assert config.coverage_analysis is True

    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = TestConfiguration(
            suites={TestSuite.UNIT, TestSuite.INTEGRATION},
            environment=TestEnvironment.DOCKER,
            parallel_workers=8,
            timeout_seconds=1800,
            fail_fast=True,
            verbose=False,
        )

        assert TestSuite.UNIT in config.suites
        assert TestSuite.INTEGRATION in config.suites
        assert config.environment == TestEnvironment.DOCKER
        assert config.parallel_workers == 8
        assert config.timeout_seconds == 1800
        assert config.fail_fast is True
        assert config.verbose is False
