#!/usr/bin/env python3
"""
Run Performance Tests

Execute performance benchmarks and generate reports.
"""

import sys
import subprocess
import argparse


def run_performance_tests():
    """Run performance tests."""
    cmd = [
        "pytest",
        "tests/performance/",
        "-v",
        "--benchmark-only",
        "--benchmark-autosave",
        "--benchmark-compare",
    ]

    print("Running performance tests...")
    result = subprocess.run(cmd)

    return result.returncode == 0


def run_specific_benchmark(benchmark_name):
    """Run a specific performance benchmark."""
    cmd = ["pytest", f"tests/performance/test_{benchmark_name}.py", "-v", "--benchmark-only"]

    print(f"Running {benchmark_name} benchmark...")
    result = subprocess.run(cmd)

    return result.returncode == 0


def main():
    """Main performance test function."""
    parser = argparse.ArgumentParser(description="Run performance tests")
    parser.add_argument("--benchmark", help="Run specific benchmark")
    parser.add_argument(
        "--compare", help="Compare with baseline", action="store_true"
    )
    parser.add_argument("--profile", help="Enable profiling", action="store_true")

    args = parser.parse_args()

    if args.benchmark:
        success = run_specific_benchmark(args.benchmark)
    else:
        success = run_performance_tests()

    if not success:
        print("\n❌ Performance tests failed!")
        sys.exit(1)

    print("\n✅ Performance tests passed!")
    sys.exit(0)


if __name__ == "__main__":
    main()
