#!/usr/bin/env python3
"""
Comprehensive Test Runner Script

Provides a unified interface for running different types of tests with various configurations.
Supports unit, integration, e2e, performance, security, and load tests.

Usage:
    python scripts/testing/run_tests.py --all
    python scripts/testing/run_tests.py --unit --coverage
    python scripts/testing/run_tests.py --integration --verbose
    python scripts/testing/run_tests.py --performance --benchmark
    python scripts/testing/run_tests.py --security
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


class TestRunner:
    """Unified test runner for the Doorbell Security System."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tests_dir = project_root / "tests"

    def run_unit_tests(self, verbose: bool = False, coverage: bool = True) -> int:
        """Run unit tests."""
        print("🧪 Running unit tests...")
        
        cmd = ["pytest", "tests/", "-m", "unit"]
        
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        if coverage:
            cmd.extend([
                "--cov=src",
                "--cov=config",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/unit",
                "--cov-report=xml:coverage-unit.xml"
            ])
        
        cmd.extend(["--tb=short", "--maxfail=10"])
        
        return self._execute_command(cmd)

    def run_integration_tests(self, verbose: bool = False, coverage: bool = True) -> int:
        """Run integration tests."""
        print("🔗 Running integration tests...")
        
        cmd = ["pytest", "tests/integration/", "-v" if verbose else "-q"]
        
        if coverage:
            cmd.extend([
                "--cov=src",
                "--cov=config",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/integration",
                "--cov-report=xml:coverage-integration.xml"
            ])
        
        cmd.extend(["--tb=short"])
        
        return self._execute_command(cmd)

    def run_e2e_tests(self, verbose: bool = False) -> int:
        """Run end-to-end tests."""
        print("🌍 Running end-to-end tests...")
        
        cmd = [
            "pytest", "tests/e2e/",
            "-v" if verbose else "-q",
            "-m", "e2e",
            "--tb=short",
            "--maxfail=3"
        ]
        
        return self._execute_command(cmd)

    def run_performance_tests(self, benchmark: bool = True) -> int:
        """Run performance tests."""
        print("⚡ Running performance tests...")
        
        cmd = [
            "pytest", "tests/performance/",
            "-v",
            "-m", "performance"
        ]
        
        if benchmark:
            cmd.extend([
                "--benchmark-only",
                "--benchmark-json=benchmark-results.json",
                "--benchmark-autosave"
            ])
        
        return self._execute_command(cmd)

    def run_security_tests(self, verbose: bool = False) -> int:
        """Run security tests."""
        print("🔒 Running security tests...")
        
        # Run pytest security tests
        pytest_cmd = [
            "pytest", "tests/security/",
            "-v" if verbose else "-q",
            "-m", "security",
            "--tb=short"
        ]
        
        result = self._execute_command(pytest_cmd)
        if result != 0:
            return result
        
        # Run bandit security scan
        print("\n🔍 Running Bandit security scan...")
        bandit_cmd = [
            "bandit", "-r", "src/", "config/", "app.py",
            "-f", "json", "-o", "bandit-report.json"
        ]
        
        bandit_result = self._execute_command(bandit_cmd, check=False)
        
        # Run safety check
        print("\n🛡️ Running Safety dependency check...")
        safety_cmd = ["safety", "check", "--json", "--output", "safety-report.json"]
        
        self._execute_command(safety_cmd, check=False)
        
        return result

    def run_load_tests(self, users: int = 10, runtime: int = 60) -> int:
        """Run load tests with Locust."""
        print(f"🏋️ Running load tests ({users} users, {runtime}s)...")
        
        locustfile = self.tests_dir / "load" / "locustfile.py"
        
        if not locustfile.exists():
            print(f"⚠️ Locustfile not found at {locustfile}")
            return 1
        
        cmd = [
            "locust",
            "-f", str(locustfile),
            "--headless",
            "--users", str(users),
            "--spawn-rate", "2",
            "--run-time", f"{runtime}s",
            "--html", "load-test-report.html"
        ]
        
        return self._execute_command(cmd)

    def run_all_tests(self, verbose: bool = False, quick: bool = False) -> int:
        """Run all test suites."""
        print("🚀 Running all tests...")
        
        results = []
        
        # Unit tests (always run)
        results.append(("Unit Tests", self.run_unit_tests(verbose=verbose)))
        
        # Integration tests
        results.append(("Integration Tests", self.run_integration_tests(verbose=verbose)))
        
        if not quick:
            # E2E tests (skip in quick mode)
            results.append(("E2E Tests", self.run_e2e_tests(verbose=verbose)))
            
            # Performance tests
            results.append(("Performance Tests", self.run_performance_tests()))
            
            # Security tests
            results.append(("Security Tests", self.run_security_tests(verbose=verbose)))
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 Test Results Summary")
        print("=" * 60)
        
        all_passed = True
        for test_name, result_code in results:
            status = "✅ PASSED" if result_code == 0 else "❌ FAILED"
            print(f"{test_name}: {status}")
            if result_code != 0:
                all_passed = False
        
        print("=" * 60)
        
        return 0 if all_passed else 1

    def run_coverage_report(self) -> int:
        """Generate comprehensive coverage report."""
        print("📈 Generating comprehensive coverage report...")
        
        cmd = [
            "pytest", "tests/",
            "--cov=src",
            "--cov=config",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-report=json:coverage.json",
            "--cov-report=term-missing",
            "-q"
        ]
        
        result = self._execute_command(cmd)
        
        if result == 0:
            print("\n✅ Coverage report generated:")
            print(f"   HTML: {self.project_root / 'htmlcov' / 'index.html'}")
            print(f"   XML:  {self.project_root / 'coverage.xml'}")
            print(f"   JSON: {self.project_root / 'coverage.json'}")
        
        return result

    def _execute_command(self, cmd: List[str], check: bool = True) -> int:
        """Execute a command and return the exit code."""
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                check=check,
                env={**os.environ, "PYTHONPATH": str(self.project_root)}
            )
            return result.returncode
        except subprocess.CalledProcessError as e:
            return e.returncode
        except FileNotFoundError as e:
            print(f"❌ Command not found: {cmd[0]}")
            print(f"   Make sure it's installed: pip install {cmd[0]}")
            return 1


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for Doorbell Security System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python scripts/testing/run_tests.py --all
  
  # Run only unit tests with coverage
  python scripts/testing/run_tests.py --unit --coverage
  
  # Run integration tests verbosely
  python scripts/testing/run_tests.py --integration --verbose
  
  # Run performance benchmarks
  python scripts/testing/run_tests.py --performance --benchmark
  
  # Run security tests
  python scripts/testing/run_tests.py --security
  
  # Quick test run (unit + integration only)
  python scripts/testing/run_tests.py --all --quick
        """
    )
    
    # Test selection
    parser.add_argument("--all", action="store_true", help="Run all test suites")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--security", action="store_true", help="Run security tests")
    parser.add_argument("--load", action="store_true", help="Run load tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    
    # Options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", action="store_true", help="Quick mode (skip slow tests)")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--users", type=int, default=10, help="Number of users for load tests")
    parser.add_argument("--runtime", type=int, default=60, help="Runtime for load tests (seconds)")
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    runner = TestRunner(project_root)
    
    # Determine what to run
    if not any([args.all, args.unit, args.integration, args.e2e, 
                args.performance, args.security, args.load, args.coverage]):
        parser.print_help()
        return 1
    
    exit_code = 0
    
    # Run selected tests
    if args.all:
        exit_code = runner.run_all_tests(verbose=args.verbose, quick=args.quick)
    else:
        if args.unit:
            exit_code |= runner.run_unit_tests(verbose=args.verbose, coverage=True)
        
        if args.integration:
            exit_code |= runner.run_integration_tests(verbose=args.verbose, coverage=True)
        
        if args.e2e:
            exit_code |= runner.run_e2e_tests(verbose=args.verbose)
        
        if args.performance:
            exit_code |= runner.run_performance_tests(benchmark=args.benchmark)
        
        if args.security:
            exit_code |= runner.run_security_tests(verbose=args.verbose)
        
        if args.load:
            exit_code |= runner.run_load_tests(users=args.users, runtime=args.runtime)
        
        if args.coverage:
            exit_code |= runner.run_coverage_report()
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
