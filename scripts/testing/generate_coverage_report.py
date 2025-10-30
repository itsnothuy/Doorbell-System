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


def check_threshold(threshold: float) -> bool:
    """Check if coverage meets the specified threshold."""
    try:
        import json

        with open("coverage.json") as f:
            data = json.load(f)
            coverage = data["totals"]["percent_covered"]

        print(f"\n📊 Checking coverage threshold...")
        print(f"   Current Coverage: {coverage:.2f}%")
        print(f"   Required Threshold: {threshold:.2f}%")

        if coverage >= threshold:
            print(f"✅ Coverage meets threshold!")
            return True
        else:
            print(f"❌ Coverage below threshold by {threshold - coverage:.2f}%")
            return False

    except Exception as e:
        print(f"Error checking threshold: {e}")
        return False


def generate_markdown_report():
    """Generate a markdown coverage report."""
    try:
        import json
        import xml.etree.ElementTree as ET

        # Parse coverage.xml for detailed info
        tree = ET.parse("coverage.xml")
        root = tree.getroot()

        line_rate = float(root.attrib.get("line-rate", 0)) * 100
        branch_rate = float(root.attrib.get("branch-rate", 0)) * 100

        report = []
        report.append("# 📊 Coverage Report\n")
        report.append(f"**Overall Line Coverage**: {line_rate:.2f}%\n")
        report.append(f"**Overall Branch Coverage**: {branch_rate:.2f}%\n")

        # Coverage badge
        if line_rate >= 90:
            badge_color = "brightgreen"
        elif line_rate >= 80:
            badge_color = "green"
        elif line_rate >= 70:
            badge_color = "yellowgreen"
        else:
            badge_color = "yellow"

        report.append(
            f"\n![Coverage](https://img.shields.io/badge/coverage-{line_rate:.0f}%25-{badge_color})\n"
        )

        # Write to file
        with open("COVERAGE_REPORT.md", "w") as f:
            f.write("\n".join(report))

        print("\n✅ Markdown report generated: COVERAGE_REPORT.md")
        return True

    except Exception as e:
        print(f"Error generating markdown report: {e}")
        return False


def main():
    """Main coverage reporting function."""
    parser = argparse.ArgumentParser(description="Generate coverage report")
    parser.add_argument("--badge", action="store_true", help="Generate coverage badge")
    parser.add_argument("--markdown", action="store_true", help="Generate markdown report")
    parser.add_argument(
        "--fail-under", type=float, help="Fail if coverage is below threshold (percentage)"
    )
    parser.add_argument(
        "--no-run", action="store_true", help="Don't run tests, just analyze existing coverage"
    )

    args = parser.parse_args()

    # Run coverage unless --no-run is specified
    if not args.no_run:
        success = run_coverage()

        if not success:
            print("\n❌ Tests failed!")
            sys.exit(1)

    # Generate badge
    if args.badge:
        meets_target = generate_badge()
        if not meets_target:
            print(f"\n⚠️ Coverage below 95% target threshold")

    # Generate markdown report
    if args.markdown:
        generate_markdown_report()

    # Check threshold if specified
    if args.fail_under is not None:
        if not check_threshold(args.fail_under):
            print(f"\n❌ Coverage does not meet {args.fail_under}% threshold!")
            sys.exit(1)

    print("\n✅ Coverage report generated successfully!")
    print("📁 HTML report: htmlcov/index.html")
    print("📁 XML report: coverage.xml")
    print("📁 JSON report: coverage.json")

    sys.exit(0)


if __name__ == "__main__":
    main()
