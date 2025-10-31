#!/usr/bin/env python3
"""
Smart Test Selection

Intelligently select which tests to run based on changed files in the repository.
This optimizes CI/CD performance by only running relevant tests.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Set


def get_git_changed_files(base_branch: str = "master") -> List[str]:
    """Get list of changed files compared to base branch."""
    try:
        # Try to get changes from git
        result = subprocess.run(
            ["git", "diff", "--name-only", f"origin/{base_branch}...HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        
        if result.returncode == 0 and result.stdout.strip():
            return [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        
        # Fallback: get all modified/added files
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~1..HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        
        if result.returncode == 0 and result.stdout.strip():
            return [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        
        # If git fails, return empty list (will run all tests)
        return []
        
    except Exception as e:
        print(f"Warning: Could not get changed files from git: {e}")
        return []


def map_source_to_test(source_file: str) -> List[Path]:
    """Map a source file to its corresponding test files."""
    test_files = []
    
    # Convert source file path to potential test file paths
    if source_file.startswith("src/"):
        # Remove src/ prefix and .py extension
        module_path = source_file[4:].replace(".py", "")
        
        # Pattern 1: tests/test_module.py
        test_file_1 = Path(f"tests/test_{Path(module_path).name}.py")
        if test_file_1.exists():
            test_files.append(test_file_1)
        
        # Pattern 2: tests/unit/test_module.py
        test_file_2 = Path(f"tests/unit/test_{Path(module_path).name}.py")
        if test_file_2.exists():
            test_files.append(test_file_2)
        
        # Pattern 3: tests/integration/test_module_integration.py
        test_file_3 = Path(f"tests/integration/test_{Path(module_path).name}_integration.py")
        if test_file_3.exists():
            test_files.append(test_file_3)
        
        # Pattern 4: Look for any test file that contains the module name
        for test_pattern in ["tests/", "tests/unit/", "tests/integration/"]:
            test_dir = Path(test_pattern)
            if test_dir.exists():
                for test_file in test_dir.glob(f"*test*{Path(module_path).name}*.py"):
                    if test_file not in test_files:
                        test_files.append(test_file)
    
    elif source_file.startswith("config/"):
        # Configuration changes - run config tests
        config_tests = Path("tests").rglob("*test*config*.py")
        test_files.extend(config_tests)
    
    return test_files


def get_core_test_files() -> List[Path]:
    """Get core test files that should always run."""
    core_tests = []
    
    # Core system tests
    core_patterns = [
        "tests/test_system.py",
        "tests/test_config*.py",
        "tests/test_doorbell_security.py",
        "tests/unit/test_doorbell_security.py",
    ]
    
    for pattern in core_patterns:
        for test_file in Path(".").glob(pattern):
            if test_file.exists():
                core_tests.append(test_file)
    
    return core_tests


def get_test_files_for_changes(base_branch: str = "master", include_core: bool = True) -> List[str]:
    """Get list of test files to run based on changed files."""
    changed_files = get_git_changed_files(base_branch)
    
    if not changed_files:
        print("No changed files detected or git not available - will run all tests")
        return []
    
    print(f"Found {len(changed_files)} changed files")
    
    test_files: Set[Path] = set()
    
    # Map source files to test files
    for file in changed_files:
        if file.startswith("tests/"):
            # Changed file is already a test
            test_files.add(Path(file))
        elif file.startswith("src/") or file.startswith("config/"):
            # Map source/config to tests
            mapped_tests = map_source_to_test(file)
            test_files.update(mapped_tests)
        elif file.endswith(".py"):
            # Other Python files (e.g., app.py)
            # Run integration tests
            for integration_test in Path("tests/integration").glob("*.py"):
                test_files.add(integration_test)
    
    # Add core tests that should always run
    if include_core:
        test_files.update(get_core_test_files())
    
    # Convert to sorted list of strings
    test_file_list = sorted([str(f) for f in test_files if f.exists()])
    
    if test_file_list:
        print(f"\nSelected {len(test_file_list)} test files to run:")
        for test_file in test_file_list[:10]:  # Show first 10
            print(f"  - {test_file}")
        if len(test_file_list) > 10:
            print(f"  ... and {len(test_file_list) - 10} more")
    else:
        print("No specific test files selected - will run all tests")
    
    return test_file_list


def main():
    """Main function for smart test selection."""
    parser = argparse.ArgumentParser(
        description="Smart test selection based on changed files"
    )
    parser.add_argument(
        "--base-branch",
        default="master",
        help="Base branch to compare against (default: master)",
    )
    parser.add_argument(
        "--no-core",
        action="store_true",
        help="Don't include core tests that always run",
    )
    parser.add_argument(
        "--output-file",
        help="Write selected test files to this file (one per line)",
    )
    parser.add_argument(
        "--format",
        choices=["paths", "pytest-args"],
        default="paths",
        help="Output format: 'paths' for file paths, 'pytest-args' for pytest arguments",
    )
    
    args = parser.parse_args()
    
    # Get selected test files
    test_files = get_test_files_for_changes(
        base_branch=args.base_branch,
        include_core=not args.no_core,
    )
    
    if not test_files:
        # No specific tests selected - return success and let caller run all tests
        print("\nRunning full test suite (no smart selection applicable)")
        sys.exit(0)
    
    # Format output
    if args.format == "pytest-args":
        output = " ".join(test_files)
    else:
        output = "\n".join(test_files)
    
    # Write to file if requested
    if args.output_file:
        Path(args.output_file).write_text(output)
        print(f"\nTest selection written to {args.output_file}")
    else:
        print(f"\n{output}")
    
    sys.exit(0)


if __name__ == "__main__":
    main()
