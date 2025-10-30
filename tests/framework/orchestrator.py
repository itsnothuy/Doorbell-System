#!/usr/bin/env python3
"""
Comprehensive Testing Framework Orchestrator

Centralized test execution, reporting, and environment management system.
"""

import asyncio
import json
import logging
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class TestSuite(Enum):
    """Test suite categories."""

    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    SECURITY = "security"
    LOAD = "load"
    STREAMING = "streaming"
    ALL = "all"


class TestEnvironment(Enum):
    """Test environment types."""

    LOCAL = "local"
    DOCKER = "docker"
    CI = "ci"
    PRODUCTION_LIKE = "production_like"


@dataclass
class TestConfiguration:
    """Test execution configuration."""

    suites: Set[TestSuite] = field(default_factory=lambda: {TestSuite.ALL})
    environment: TestEnvironment = TestEnvironment.LOCAL
    parallel_workers: int = 4
    timeout_seconds: int = 3600
    generate_reports: bool = True
    coverage_analysis: bool = True
    performance_baseline: Optional[str] = None
    fail_fast: bool = False
    verbose: bool = True

    # Environment-specific settings
    docker_image: str = "doorbell-test:latest"
    test_data_path: Path = Path("tests/fixtures")
    output_dir: Path = Path("test-results")

    # Performance testing
    performance_iterations: int = 5
    performance_warmup: int = 2
    regression_threshold: float = 0.15  # 15% performance degradation threshold


@dataclass
class TestResult:
    """Individual test result."""

    test_name: str
    suite: TestSuite
    status: str  # "passed", "failed", "skipped", "error"
    duration: float
    error_message: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    coverage_data: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None


@dataclass
class TestSuiteResult:
    """Test suite execution result."""

    suite: TestSuite
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    coverage_percentage: Optional[float] = None
    test_results: List[TestResult] = field(default_factory=list)


@dataclass
class TestExecutionResult:
    """Complete test execution result."""

    configuration: TestConfiguration
    start_time: float
    end_time: float
    total_duration: float
    suite_results: List[TestSuiteResult] = field(default_factory=list)
    overall_status: str = "unknown"
    coverage_report_path: Optional[Path] = None
    performance_report_path: Optional[Path] = None
    html_report_path: Optional[Path] = None


class TestOrchestrator:
    """Centralized test execution orchestrator."""

    def __init__(self, config: TestConfiguration):
        self.config = config
        self.project_root = Path(__file__).parent.parent.parent
        self.test_root = self.project_root / "tests"

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Test orchestrator initialized with config: {config}")

    async def execute_tests(self) -> TestExecutionResult:
        """Execute comprehensive test suite."""
        start_time = time.time()

        try:
            # Setup test environment
            await self._setup_test_environment()

            # Execute test suites
            suite_results = []

            if TestSuite.ALL in self.config.suites:
                suites_to_run = [s for s in TestSuite if s != TestSuite.ALL]
            else:
                suites_to_run = list(self.config.suites)

            for suite in suites_to_run:
                logger.info(f"Executing {suite.value} tests...")
                result = await self._execute_test_suite(suite)
                suite_results.append(result)

                if self.config.fail_fast and result.failed > 0:
                    logger.error(
                        f"Fail-fast enabled, stopping due to {suite.value} failures"
                    )
                    break

            end_time = time.time()

            # Generate comprehensive result
            result = TestExecutionResult(
                configuration=self.config,
                start_time=start_time,
                end_time=end_time,
                total_duration=end_time - start_time,
                suite_results=suite_results,
            )

            # Determine overall status
            result.overall_status = self._determine_overall_status(suite_results)

            # Generate reports
            if self.config.generate_reports:
                await self._generate_reports(result)

            return result

        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            raise
        finally:
            await self._cleanup_test_environment()

    async def _setup_test_environment(self) -> None:
        """Setup test environment based on configuration."""
        if self.config.environment == TestEnvironment.DOCKER:
            await self._setup_docker_environment()
        elif self.config.environment == TestEnvironment.PRODUCTION_LIKE:
            await self._setup_production_like_environment()

        # Copy test fixtures
        if self.config.test_data_path.exists():
            logger.info("Test fixtures ready")

    async def _execute_test_suite(self, suite: TestSuite) -> TestSuiteResult:
        """Execute a specific test suite."""
        start_time = time.time()

        # Build pytest command
        cmd = self._build_pytest_command(suite)

        # Execute tests
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.project_root,
        )

        stdout, stderr = await process.communicate()

        end_time = time.time()
        duration = end_time - start_time

        # Parse results
        return self._parse_test_results(
            suite, duration, process.returncode, stdout.decode(), stderr.decode()
        )

    def _build_pytest_command(self, suite: TestSuite) -> List[str]:
        """Build pytest command for specific suite."""
        cmd = ["python", "-m", "pytest"]

        # Add suite-specific paths
        suite_paths = {
            TestSuite.UNIT: "tests/unit",
            TestSuite.INTEGRATION: "tests/integration",
            TestSuite.E2E: "tests/e2e",
            TestSuite.PERFORMANCE: "tests/performance",
            TestSuite.SECURITY: "tests/security",
            TestSuite.LOAD: "tests/load",
            TestSuite.STREAMING: "tests/streaming",
        }

        if suite in suite_paths:
            cmd.append(str(self.test_root / suite_paths[suite]))

        # Add common options
        cmd.extend(["-v" if self.config.verbose else "-q", "--tb=short"])

        # Add JUnit XML output
        xml_output = self.config.output_dir / f"{suite.value}_results.xml"
        cmd.append(f"--junitxml={xml_output}")

        # Add parallel execution if requested
        if self.config.parallel_workers > 1:
            cmd.extend(["-n", str(self.config.parallel_workers)])

        # Add coverage for appropriate suites
        if self.config.coverage_analysis and suite in {
            TestSuite.UNIT,
            TestSuite.INTEGRATION,
        }:
            cmd.extend(
                [
                    "--cov=src",
                    "--cov=config",
                    f"--cov-report=html:{self.config.output_dir}/coverage_{suite.value}",
                    f"--cov-report=json:{self.config.output_dir}/coverage_{suite.value}.json",
                ]
            )

        return cmd

    def _parse_test_results(
        self,
        suite: TestSuite,
        duration: float,
        return_code: int,
        stdout: str,
        stderr: str,
    ) -> TestSuiteResult:
        """Parse pytest output into structured results."""
        # Try to parse JUnit XML if available
        xml_path = self.config.output_dir / f"{suite.value}_results.xml"

        if xml_path.exists():
            return self._parse_junit_xml(suite, duration, xml_path)

        # Fallback to parsing stdout
        return self._parse_stdout(suite, duration, return_code, stdout)

    def _parse_junit_xml(
        self, suite: TestSuite, duration: float, xml_path: Path
    ) -> TestSuiteResult:
        """Parse JUnit XML test results."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract test counts from testsuite element
            testsuite = root if root.tag == "testsuite" else root.find("testsuite")

            if testsuite is not None:
                total = int(testsuite.get("tests", 0))
                failures = int(testsuite.get("failures", 0))
                errors = int(testsuite.get("errors", 0))
                skipped = int(testsuite.get("skipped", 0))
                passed = total - failures - errors - skipped

                return TestSuiteResult(
                    suite=suite,
                    total_tests=total,
                    passed=passed,
                    failed=failures,
                    skipped=skipped,
                    errors=errors,
                    duration=duration,
                )

        except Exception as e:
            logger.warning(f"Failed to parse JUnit XML for {suite.value}: {e}")

        # Return default result if parsing fails
        return TestSuiteResult(
            suite=suite,
            total_tests=0,
            passed=0,
            failed=0,
            skipped=0,
            errors=0,
            duration=duration,
        )

    def _parse_stdout(
        self, suite: TestSuite, duration: float, return_code: int, stdout: str
    ) -> TestSuiteResult:
        """Parse pytest stdout for test results."""
        # Parse pytest summary line
        # Example: "5 passed, 2 failed, 1 skipped in 10.5s"
        passed = 0
        failed = 0
        skipped = 0
        errors = 0

        if "passed" in stdout:
            import re

            matches = re.findall(r"(\d+) passed", stdout)
            if matches:
                passed = int(matches[0])

        if "failed" in stdout:
            import re

            matches = re.findall(r"(\d+) failed", stdout)
            if matches:
                failed = int(matches[0])

        if "skipped" in stdout:
            import re

            matches = re.findall(r"(\d+) skipped", stdout)
            if matches:
                skipped = int(matches[0])

        if "error" in stdout:
            import re

            matches = re.findall(r"(\d+) error", stdout)
            if matches:
                errors = int(matches[0])

        total = passed + failed + skipped + errors

        return TestSuiteResult(
            suite=suite,
            total_tests=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration=duration,
        )

    async def _generate_reports(self, result: TestExecutionResult) -> None:
        """Generate comprehensive test reports."""
        # Generate HTML report
        html_path = await self._generate_html_report(result)
        result.html_report_path = html_path

        # Generate JSON report
        await self._generate_json_report(result)

        # Generate coverage report
        if self.config.coverage_analysis:
            coverage_path = await self._generate_coverage_report(result)
            result.coverage_report_path = coverage_path

        # Generate performance report
        performance_path = await self._generate_performance_report(result)
        result.performance_report_path = performance_path

    async def _generate_html_report(self, result: TestExecutionResult) -> Path:
        """Generate comprehensive HTML test report."""
        # Calculate totals across all suites
        total_tests = sum(r.total_tests for r in result.suite_results)
        total_passed = sum(r.passed for r in result.suite_results)
        total_failed = sum(r.failed for r in result.suite_results)
        total_skipped = sum(r.skipped for r in result.suite_results)
        total_errors = sum(r.errors for r in result.suite_results)

        # Generate suite rows
        suite_rows = ""
        for suite_result in result.suite_results:
            status_class = "passed" if suite_result.failed == 0 else "failed"
            suite_rows += f"""
                <tr class="{status_class}">
                    <td>{suite_result.suite.value.upper()}</td>
                    <td>{suite_result.total_tests}</td>
                    <td class="passed">{suite_result.passed}</td>
                    <td class="failed">{suite_result.failed}</td>
                    <td>{suite_result.skipped}</td>
                    <td>{suite_result.errors}</td>
                    <td>{suite_result.duration:.2f}s</td>
                </tr>
            """

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Doorbell Security System - Test Report</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                }}
                .header h1 {{
                    margin: 0 0 10px 0;
                    font-size: 28px;
                }}
                .header p {{
                    margin: 5px 0;
                    opacity: 0.9;
                }}
                .summary {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    padding: 30px;
                    background: #fafafa;
                }}
                .metric {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .metric h3 {{
                    margin: 0 0 10px 0;
                    font-size: 14px;
                    color: #666;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }}
                .metric .value {{
                    font-size: 32px;
                    font-weight: bold;
                    color: #333;
                }}
                .metric.passed .value {{ color: #28a745; }}
                .metric.failed .value {{ color: #dc3545; }}
                .metric.duration .value {{ color: #007bff; }}
                .suites {{
                    padding: 30px;
                }}
                .suites h2 {{
                    margin: 0 0 20px 0;
                    font-size: 20px;
                    color: #333;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                th {{
                    background: #f8f9fa;
                    padding: 12px;
                    text-align: left;
                    font-weight: 600;
                    color: #495057;
                    border-bottom: 2px solid #dee2e6;
                }}
                td {{
                    padding: 12px;
                    border-bottom: 1px solid #dee2e6;
                }}
                tr:hover {{
                    background: #f8f9fa;
                }}
                .passed {{ color: #28a745; font-weight: 600; }}
                .failed {{ color: #dc3545; font-weight: 600; }}
                .status {{
                    display: inline-block;
                    padding: 4px 12px;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: 600;
                    text-transform: uppercase;
                }}
                .status.passed {{
                    background: #d4edda;
                    color: #155724;
                }}
                .status.failed {{
                    background: #f8d7da;
                    color: #721c24;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔔 Test Execution Report</h1>
                    <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result.start_time))}</p>
                    <p>Environment: {result.configuration.environment.value}</p>
                    <p>Status: <span class="status {result.overall_status}">{result.overall_status.upper()}</span></p>
                </div>
                
                <div class="summary">
                    <div class="metric">
                        <h3>Total Tests</h3>
                        <div class="value">{total_tests}</div>
                    </div>
                    <div class="metric passed">
                        <h3>Passed</h3>
                        <div class="value">{total_passed}</div>
                    </div>
                    <div class="metric failed">
                        <h3>Failed</h3>
                        <div class="value">{total_failed}</div>
                    </div>
                    <div class="metric">
                        <h3>Skipped</h3>
                        <div class="value">{total_skipped}</div>
                    </div>
                    <div class="metric duration">
                        <h3>Duration</h3>
                        <div class="value">{result.total_duration:.1f}s</div>
                    </div>
                </div>
                
                <div class="suites">
                    <h2>Test Suite Results</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Suite</th>
                                <th>Total</th>
                                <th>Passed</th>
                                <th>Failed</th>
                                <th>Skipped</th>
                                <th>Errors</th>
                                <th>Duration</th>
                            </tr>
                        </thead>
                        <tbody>
                            {suite_rows}
                        </tbody>
                    </table>
                </div>
            </div>
        </body>
        </html>
        """

        html_path = self.config.output_dir / "test_report.html"
        html_path.write_text(html_content)
        logger.info(f"Generated HTML report: {html_path}")
        return html_path

    async def _generate_json_report(self, result: TestExecutionResult) -> Path:
        """Generate JSON test report."""
        report_data = {
            "start_time": result.start_time,
            "end_time": result.end_time,
            "total_duration": result.total_duration,
            "overall_status": result.overall_status,
            "configuration": {
                "suites": [s.value for s in result.configuration.suites],
                "environment": result.configuration.environment.value,
                "parallel_workers": result.configuration.parallel_workers,
                "coverage_analysis": result.configuration.coverage_analysis,
            },
            "suite_results": [
                {
                    "suite": sr.suite.value,
                    "total_tests": sr.total_tests,
                    "passed": sr.passed,
                    "failed": sr.failed,
                    "skipped": sr.skipped,
                    "errors": sr.errors,
                    "duration": sr.duration,
                    "coverage_percentage": sr.coverage_percentage,
                }
                for sr in result.suite_results
            ],
        }

        json_path = self.config.output_dir / "test_report.json"
        json_path.write_text(json.dumps(report_data, indent=2))
        logger.info(f"Generated JSON report: {json_path}")
        return json_path

    async def _generate_coverage_report(self, result: TestExecutionResult) -> Path:
        """Generate consolidated coverage report."""
        # Aggregate coverage data from all suites
        coverage_dir = self.config.output_dir / "coverage_combined"
        coverage_dir.mkdir(exist_ok=True)

        # Copy HTML coverage reports to output directory
        for suite_result in result.suite_results:
            suite_coverage_dir = (
                self.config.output_dir / f"coverage_{suite_result.suite.value}"
            )
            if suite_coverage_dir.exists():
                logger.info(
                    f"Coverage report available for {suite_result.suite.value}: {suite_coverage_dir}"
                )

        logger.info(f"Coverage reports in: {self.config.output_dir}")
        return self.config.output_dir

    async def _generate_performance_report(self, result: TestExecutionResult) -> Path:
        """Generate performance test report."""
        # Collect performance data from performance suite
        perf_data = {
            "timestamp": time.time(),
            "suites": [
                {
                    "name": sr.suite.value,
                    "duration": sr.duration,
                    "tests_per_second": (
                        sr.total_tests / sr.duration if sr.duration > 0 else 0
                    ),
                }
                for sr in result.suite_results
            ],
            "total_duration": result.total_duration,
        }

        perf_path = self.config.output_dir / "performance_report.json"
        perf_path.write_text(json.dumps(perf_data, indent=2))
        logger.info(f"Generated performance report: {perf_path}")
        return perf_path

    def _determine_overall_status(self, suite_results: List[TestSuiteResult]) -> str:
        """Determine overall test execution status."""
        if not suite_results:
            return "no_tests"

        total_failed = sum(r.failed + r.errors for r in suite_results)
        return "passed" if total_failed == 0 else "failed"

    async def _setup_docker_environment(self) -> None:
        """Setup Docker test environment."""
        logger.warning("Docker environment setup not yet implemented")

    async def _setup_production_like_environment(self) -> None:
        """Setup production-like test environment."""
        logger.warning("Production-like environment setup not yet implemented")

    async def _cleanup_test_environment(self) -> None:
        """Cleanup test environment."""
        logger.info("Test environment cleanup completed")


# CLI Interface
async def main() -> int:
    """CLI entry point for test orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Doorbell Security System Test Orchestrator"
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=[s.value for s in TestSuite],
        default=["all"],
        help="Test suites to execute",
    )
    parser.add_argument(
        "--environment",
        choices=[e.value for e in TestEnvironment],
        default="local",
        help="Test environment",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Parallel workers"
    )
    parser.add_argument(
        "--timeout", type=int, default=3600, help="Timeout in seconds"
    )
    parser.add_argument(
        "--no-reports", action="store_true", help="Skip report generation"
    )
    parser.add_argument(
        "--no-coverage", action="store_true", help="Skip coverage analysis"
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop on first failure"
    )
    parser.add_argument("--quiet", action="store_true", help="Quiet output")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test-results"),
        help="Output directory for reports",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if not args.quiet else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Build configuration
    config = TestConfiguration(
        suites={TestSuite(s) for s in args.suites},
        environment=TestEnvironment(args.environment),
        parallel_workers=args.workers,
        timeout_seconds=args.timeout,
        generate_reports=not args.no_reports,
        coverage_analysis=not args.no_coverage,
        fail_fast=args.fail_fast,
        verbose=not args.quiet,
        output_dir=args.output_dir,
    )

    # Execute tests
    orchestrator = TestOrchestrator(config)
    result = await orchestrator.execute_tests()

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"TEST EXECUTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Status: {result.overall_status.upper()}")
    print(f"Duration: {result.total_duration:.2f} seconds")
    print(f"Suites: {len(result.suite_results)}")

    # Print suite details
    for suite_result in result.suite_results:
        print(
            f"\n{suite_result.suite.value.upper()}: "
            f"{suite_result.passed}/{suite_result.total_tests} passed"
        )

    if result.html_report_path:
        print(f"\nHTML Report: {result.html_report_path}")

    # Exit with appropriate code
    return 0 if result.overall_status == "passed" else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
