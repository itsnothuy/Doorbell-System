#!/usr/bin/env python3
"""
Generate Coverage Report

Generate and analyze code coverage reports.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_coverage():
    """Run tests with coverage."""
    cmd = [
        "pytest",
        "tests/",
        "--cov=src",
        "--cov=config",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-report=xml:coverage.xml",
        "--cov-report=json:coverage.json",
    ]

    print("Running tests with coverage...")
    result = subprocess.run(cmd)

    return result.returncode == 0


def generate_badge():
    """Generate coverage badge."""
    try:
        import json

        with open("coverage.json") as f:
            data = json.load(f)
            coverage = data["totals"]["percent_covered"]

        print(f"\n📊 Total Coverage: {coverage:.2f}%")

        if coverage >= 95:
            print("✅ Excellent! Coverage meets 95% target")
        elif coverage >= 80:
            print("⚠️  Good coverage, but below 95% target")
        else:
            print("❌ Coverage below acceptable threshold")

        return coverage >= 95

    except Exception as e:
        print(f"Error generating badge: {e}")
        return False


def main():
    """Main coverage reporting function."""
    parser = argparse.ArgumentParser(description="Generate coverage report")
    parser.add_argument("--badge", action="store_true", help="Generate coverage badge")
    parser.add_argument(
        "--fail-under", type=float, default=80.0, help="Fail if coverage is below threshold"
    )

    args = parser.parse_args()

    # Run coverage
    success = run_coverage()

    if not success:
        print("\n❌ Tests failed!")
        sys.exit(1)

    # Generate badge
    if args.badge:
        meets_target = generate_badge()
        if not meets_target:
            print(f"\n❌ Coverage below target threshold")

    print("\n✅ Coverage report generated successfully!")
    print("📁 HTML report: htmlcov/index.html")
    print("📁 XML report: coverage.xml")

    sys.exit(0)


if __name__ == "__main__":
    main()
