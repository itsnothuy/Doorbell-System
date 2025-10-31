#!/usr/bin/env python3
"""
Automatically add pytest markers to test files based on directory structure.

This script helps ensure all tests have appropriate markers for CI/CD filtering.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple


def should_add_marker(content: str, marker: str) -> bool:
    """Check if marker should be added to file."""
    # Check if marker already exists
    marker_pattern = f"@pytest.mark.{marker}"
    if marker_pattern in content:
        return False
    
    # Check if file has test functions or classes
    if not (re.search(r'def test_', content) or re.search(r'class Test', content)):
        return False
    
    return True


def add_marker_to_file(file_path: Path, marker: str) -> bool:
    """Add marker to a test file if needed."""
    try:
        content = file_path.read_text()
        
        if not should_add_marker(content, marker):
            return False
        
        # Find where to insert the marker (after imports, before first test)
        lines = content.split('\n')
        insert_index = None
        
        # Find the first test function or class
        for i, line in enumerate(lines):
            if re.match(r'^(def test_|class Test)', line.strip()):
                insert_index = i
                break
        
        if insert_index is None:
            print(f"  ⚠️  No test functions found in {file_path.name}")
            return False
        
        # Insert marker before the first test
        # Add a blank line if there isn't one
        if insert_index > 0 and lines[insert_index - 1].strip():
            lines.insert(insert_index, '')
            insert_index += 1
        
        # Add the marker
        lines.insert(insert_index, f'@pytest.mark.{marker}')
        
        # Write back
        file_path.write_text('\n'.join(lines))
        print(f"  ✅ Added @pytest.mark.{marker} to {file_path.name}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {file_path}: {e}")
        return False


def process_directory(directory: Path, marker: str) -> Tuple[int, int]:
    """Process all test files in a directory."""
    print(f"\nProcessing {directory} with marker '{marker}':")
    
    if not directory.exists():
        print(f"  ⚠️  Directory does not exist")
        return 0, 0
    
    test_files = list(directory.glob('test_*.py'))
    if not test_files:
        print(f"  ⚠️  No test files found")
        return 0, 0
    
    updated = 0
    total = len(test_files)
    
    for test_file in test_files:
        if add_marker_to_file(test_file, marker):
            updated += 1
    
    return updated, total


def main():
    """Main function to add markers to tests."""
    print("=" * 80)
    print("Pytest Test Marker Automation")
    print("=" * 80)
    
    # Define directory to marker mappings
    marker_mappings = {
        'tests/unit': 'unit',
        'tests/integration': 'integration',
        'tests/e2e': 'e2e',
        'tests/performance': 'performance',
        'tests/security': 'security',
        'tests/load': 'load',
    }
    
    total_updated = 0
    total_files = 0
    
    for directory_str, marker in marker_mappings.items():
        directory = Path(directory_str)
        updated, files = process_directory(directory, marker)
        total_updated += updated
        total_files += files
    
    print("\n" + "=" * 80)
    print(f"Summary: Updated {total_updated} of {total_files} test files")
    print("=" * 80)
    
    if total_updated > 0:
        print("\n⚠️  Remember to:")
        print("  1. Review the changes")
        print("  2. Run tests to ensure they still work")
        print("  3. Commit the changes")


if __name__ == '__main__':
    main()
