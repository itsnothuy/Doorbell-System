#!/usr/bin/env python3
"""
Run Full Test Suite

Execute comprehensive test suite with proper reporting and coverage.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'=' * 80}")
    print(f"{description}")
    print(f"{'=' * 80}\n")

    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def main():
    """Main test execution function."""
    parser = argparse.ArgumentParser(description="Run comprehensive test suite")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end tests only")
    parser.add_argument(
        "--performance", action="store_true", help="Run performance tests only"
    )
    parser.add_argument("--security", action="store_true", help="Run security tests only")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-n", type=int, help="Number of parallel workers")

    args = parser.parse_args()

    # Build pytest command
    pytest_cmd = ["pytest"]

    # Add verbosity
    if args.verbose:
        pytest_cmd.append("-vv")
    else:
        pytest_cmd.append("-v")

    # Add parallel execution
    if args.parallel:
        pytest_cmd.extend(["-n", str(args.parallel)])

    # Add coverage
    if args.coverage or not any([args.unit, args.integration, args.e2e, args.performance, args.security]):
        pytest_cmd.extend(["--cov=src", "--cov=config", "--cov-report=term-missing", "--cov-report=html"])

    # Select test types
    if args.unit:
        pytest_cmd.extend(["-m", "unit"])
    elif args.integration:
        pytest_cmd.extend(["-m", "integration"])
    elif args.e2e:
        pytest_cmd.extend(["-m", "e2e"])
    elif args.performance:
        pytest_cmd.extend(["-m", "performance"])
    elif args.security:
        pytest_cmd.extend(["-m", "security"])

    # Add test directory
    pytest_cmd.append("tests/")

    # Execute tests
    success = run_command(" ".join(pytest_cmd), "Running Test Suite")

    if not success:
        print("\n❌ Tests failed!")
        sys.exit(1)

    print("\n✅ All tests passed!")

    # Generate coverage report if requested
    if args.coverage or not any([args.unit, args.integration, args.e2e, args.performance, args.security]):
        print("\n📊 Coverage report generated in htmlcov/index.html")

    sys.exit(0)


if __name__ == "__main__":
    main()
