#!/usr/bin/env python3
"""
GitHub Test Summary Generator

Generate comprehensive test summaries for GitHub Actions step summary.
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional


def parse_junit_xml(junit_xml_path: str) -> Dict:
    """Parse pytest JUnit XML results."""
    try:
        tree = ET.parse(junit_xml_path)
        root = tree.getroot()
        
        # Extract test suite information
        total = int(root.get('tests', 0))
        failures = int(root.get('failures', 0))
        errors = int(root.get('errors', 0))
        skipped = int(root.get('skipped', 0))
        time = float(root.get('time', 0))
        
        passed = total - failures - errors - skipped
        
        # Get individual test results
        test_cases = []
        failed_tests = []
        error_tests = []
        skipped_tests = []
        slow_tests = []
        
        for testcase in root.iter('testcase'):
            test_name = f"{testcase.get('classname', '')}.{testcase.get('name', '')}"
            test_time = float(testcase.get('time', 0))
            
            test_info = {
                'name': test_name,
                'time': test_time,
            }
            
            # Check for failures
            failure = testcase.find('failure')
            if failure is not None:
                test_info['failure'] = failure.get('message', 'Unknown failure')
                failed_tests.append(test_info)
            
            # Check for errors
            error = testcase.find('error')
            if error is not None:
                test_info['error'] = error.get('message', 'Unknown error')
                error_tests.append(test_info)
            
            # Check for skipped
            skip = testcase.find('skipped')
            if skip is not None:
                test_info['skip_reason'] = skip.get('message', 'Skipped')
                skipped_tests.append(test_info)
            
            # Track slow tests (> 5 seconds)
            if test_time > 5.0:
                slow_tests.append(test_info)
            
            test_cases.append(test_info)
        
        return {
            'total': total,
            'passed': passed,
            'failed': failures,
            'errors': errors,
            'skipped': skipped,
            'time': time,
            'test_cases': test_cases,
            'failed_tests': failed_tests,
            'error_tests': error_tests,
            'skipped_tests': skipped_tests,
            'slow_tests': sorted(slow_tests, key=lambda x: x['time'], reverse=True),
        }
    except FileNotFoundError:
        print(f"⚠️ JUnit XML file not found: {junit_xml_path}")
        return None
    except Exception as e:
        print(f"Error parsing JUnit XML: {e}")
        return None


def parse_coverage_json(coverage_json_path: str) -> Optional[Dict]:
    """Parse coverage.json results."""
    try:
        with open(coverage_json_path, 'r') as f:
            data = json.load(f)
        
        totals = data.get('totals', {})
        
        return {
            'percent_covered': totals.get('percent_covered', 0),
            'num_statements': totals.get('num_statements', 0),
            'missing_lines': totals.get('missing_lines', 0),
            'covered_lines': totals.get('covered_lines', 0),
        }
    except FileNotFoundError:
        print(f"⚠️ Coverage JSON file not found: {coverage_json_path}")
        return None
    except Exception as e:
        print(f"Error parsing coverage JSON: {e}")
        return None


def generate_summary_markdown(
    test_results: Optional[Dict],
    coverage_data: Optional[Dict],
    performance_data: Optional[Dict] = None,
) -> str:
    """Generate markdown summary for GitHub."""
    
    lines = []
    lines.append("# 🧪 Test Results Summary\n")
    
    # Test Results Section
    if test_results:
        total = test_results['total']
        passed = test_results['passed']
        failed = test_results['failed']
        errors = test_results['errors']
        skipped = test_results['skipped']
        time = test_results['time']
        
        # Overall status
        if failed + errors == 0:
            status_emoji = "✅"
            status_text = "All tests passed!"
        elif failed + errors < total * 0.1:
            status_emoji = "⚠️"
            status_text = "Most tests passed with some failures"
        else:
            status_emoji = "❌"
            status_text = "Significant test failures detected"
        
        lines.append(f"## {status_emoji} Overall Status: {status_text}\n")
        lines.append("### Test Statistics\n")
        lines.append("| Metric | Count | Percentage |")
        lines.append("|--------|-------|------------|")
        lines.append(f"| ✅ Passed | {passed} | {(passed/total*100):.1f}% |")
        lines.append(f"| ❌ Failed | {failed} | {(failed/total*100):.1f}% |")
        lines.append(f"| 💥 Errors | {errors} | {(errors/total*100):.1f}% |")
        lines.append(f"| ⏭️  Skipped | {skipped} | {(skipped/total*100):.1f}% |")
        lines.append(f"| **Total** | **{total}** | **100%** |")
        lines.append(f"| ⏱️  Duration | {time:.2f}s | - |\n")
        
        # Failed tests details
        if test_results['failed_tests']:
            lines.append("### ❌ Failed Tests\n")
            for test in test_results['failed_tests'][:10]:  # Show first 10
                lines.append(f"- **{test['name']}** ({test['time']:.3f}s)")
                if 'failure' in test:
                    lines.append(f"  ```")
                    lines.append(f"  {test['failure'][:200]}")  # Truncate long messages
                    lines.append(f"  ```")
            if len(test_results['failed_tests']) > 10:
                remaining = len(test_results['failed_tests']) - 10
                lines.append(f"\n*... and {remaining} more failed tests*\n")
        
        # Slow tests
        if test_results['slow_tests']:
            lines.append("### 🐌 Slow Tests (>5s)\n")
            for test in test_results['slow_tests'][:5]:  # Show top 5
                lines.append(f"- {test['name']}: {test['time']:.2f}s")
            lines.append("")
    else:
        lines.append("⚠️ No test results available\n")
    
    # Coverage Section
    if coverage_data:
        percent = coverage_data['percent_covered']
        
        if percent >= 90:
            coverage_emoji = "🟢"
            coverage_status = "Excellent"
        elif percent >= 80:
            coverage_emoji = "🟡"
            coverage_status = "Good"
        elif percent >= 70:
            coverage_emoji = "🟠"
            coverage_status = "Acceptable"
        else:
            coverage_emoji = "🔴"
            coverage_status = "Needs Improvement"
        
        lines.append(f"## {coverage_emoji} Code Coverage: {percent:.2f}% ({coverage_status})\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Coverage | {percent:.2f}% |")
        lines.append(f"| Covered Lines | {coverage_data['covered_lines']} |")
        lines.append(f"| Missing Lines | {coverage_data['missing_lines']} |")
        lines.append(f"| Total Statements | {coverage_data['num_statements']} |\n")
        
        # Coverage badge
        if percent >= 90:
            badge_color = "brightgreen"
        elif percent >= 80:
            badge_color = "green"
        elif percent >= 70:
            badge_color = "yellowgreen"
        else:
            badge_color = "yellow"
        
        lines.append(f"![Coverage](https://img.shields.io/badge/coverage-{percent:.0f}%25-{badge_color})\n")
    
    # Performance Section
    if performance_data and 'regressions' in performance_data:
        regressions = performance_data.get('regressions', [])
        if regressions:
            lines.append(f"## ⚠️  Performance Regressions Detected\n")
            lines.append(f"Found {len(regressions)} performance regression(s):\n")
            for reg in regressions[:5]:  # Show top 5
                lines.append(f"- {reg['test']}: {reg['regression_pct']:.1f}% slower")
            if len(regressions) > 5:
                lines.append(f"\n*... and {len(regressions) - 5} more regressions*\n")
        else:
            lines.append("## ✅ Performance: No Regressions\n")
    
    # Footer
    lines.append("\n---")
    lines.append("*Generated by automated test reporting*")
    
    return "\n".join(lines)


def main():
    """Main function for test summary generation."""
    parser = argparse.ArgumentParser(
        description="Generate GitHub test summary"
    )
    parser.add_argument(
        "--junit-xml",
        default="pytest-results.xml",
        help="Path to pytest JUnit XML results",
    )
    parser.add_argument(
        "--coverage-json",
        default="coverage.json",
        help="Path to coverage JSON file",
    )
    parser.add_argument(
        "--performance-json",
        help="Path to performance regression JSON file",
    )
    parser.add_argument(
        "--output",
        help="Output file (defaults to stdout)",
    )
    
    args = parser.parse_args()
    
    # Parse test results
    test_results = parse_junit_xml(args.junit_xml)
    
    # Parse coverage data
    coverage_data = parse_coverage_json(args.coverage_json)
    
    # Parse performance data
    performance_data = None
    if args.performance_json and Path(args.performance_json).exists():
        try:
            with open(args.performance_json, 'r') as f:
                performance_data = json.load(f)
        except Exception as e:
            print(f"Error loading performance data: {e}")
    
    # Generate summary
    summary = generate_summary_markdown(test_results, coverage_data, performance_data)
    
    # Output summary
    if args.output:
        with open(args.output, 'w') as f:
            f.write(summary)
        print(f"✅ Summary written to {args.output}")
    else:
        print(summary)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
