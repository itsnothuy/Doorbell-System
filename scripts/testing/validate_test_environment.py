#!/usr/bin/env python3
"""
Validate Test Environment

Check that all test dependencies and requirements are met.
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python version {version.major}.{version.minor} not supported. Need 3.10+")
        return False


def check_pytest():
    """Check if pytest is installed."""
    print("Checking pytest...")
    try:
        import pytest

        print(f"✅ pytest {pytest.__version__}")
        return True
    except ImportError:
        print("❌ pytest not installed")
        return False


def check_coverage():
    """Check if coverage is installed."""
    print("Checking coverage...")
    try:
        import coverage

        print(f"✅ coverage {coverage.__version__}")
        return True
    except ImportError:
        print("❌ coverage not installed")
        return False


def check_test_directories():
    """Check test directory structure."""
    print("Checking test directories...")
    required_dirs = [
        "tests/unit",
        "tests/integration",
        "tests/e2e",
        "tests/performance",
        "tests/security",
        "tests/fixtures",
        "tests/utils",
    ]

    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} not found")
            all_exist = False

    return all_exist


def check_dependencies():
    """Check required dependencies."""
    print("Checking dependencies...")
    required = [
        "numpy",
        "opencv-python",
        "pytest-cov",
        "pytest-mock",
        "pytest-asyncio",
    ]

    all_installed = True
    for package in required:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} not installed")
            all_installed = False

    return all_installed


def main():
    """Main validation function."""
    print("=" * 80)
    print("Validating Test Environment")
    print("=" * 80)
    print()

    checks = [
        check_python_version(),
        check_pytest(),
        check_coverage(),
        check_test_directories(),
        check_dependencies(),
    ]

    print()
    print("=" * 80)

    if all(checks):
        print("✅ Test environment validation passed!")
        sys.exit(0)
    else:
        print("❌ Test environment validation failed!")
        print("\nTo fix, run:")
        print("  pip install -e '.[dev,testing]'")
        sys.exit(1)


if __name__ == "__main__":
    main()
