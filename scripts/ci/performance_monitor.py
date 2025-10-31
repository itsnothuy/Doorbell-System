#!/usr/bin/env python3
"""
Performance Monitor for CI/CD

Monitor test performance and detect regressions automatically.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


class PerformanceMonitor:
    """Monitor test performance and detect regressions."""
    
    def __init__(self, baseline_file: str = "tests/baselines/performance.json"):
        self.baseline_file = Path(baseline_file)
        self.current_metrics: Dict[str, float] = {}
        self.regressions: List[Dict[str, any]] = []
        
        # Ensure baseline directory exists
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
    
    def load_baseline(self) -> Dict[str, float]:
        """Load baseline performance metrics."""
        if not self.baseline_file.exists():
            print(f"No baseline file found at {self.baseline_file}")
            return {}
        
        try:
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading baseline: {e}")
            return {}
    
    def save_baseline(self, metrics: Optional[Dict[str, float]] = None) -> None:
        """Save current metrics as baseline."""
        if metrics is None:
            metrics = self.current_metrics
        
        try:
            with open(self.baseline_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"✅ Baseline saved to {self.baseline_file}")
        except Exception as e:
            print(f"Error saving baseline: {e}")
    
    def record_test_time(self, test_name: str, duration: float) -> None:
        """Record test execution time."""
        self.current_metrics[test_name] = duration
    
    def load_pytest_timing(self, junit_xml_path: str = "pytest-results.xml") -> None:
        """Load test timing from pytest JUnit XML output."""
        import xml.etree.ElementTree as ET
        
        try:
            tree = ET.parse(junit_xml_path)
            root = tree.getroot()
            
            for testcase in root.iter('testcase'):
                test_name = f"{testcase.get('classname', '')}.{testcase.get('name', '')}"
                duration = float(testcase.get('time', 0))
                self.record_test_time(test_name, duration)
            
            print(f"✅ Loaded {len(self.current_metrics)} test timings from {junit_xml_path}")
        except FileNotFoundError:
            print(f"⚠️ JUnit XML file not found: {junit_xml_path}")
        except Exception as e:
            print(f"Error loading pytest timing: {e}")
    
    def check_regressions(self, threshold: float = 0.2) -> List[Dict[str, any]]:
        """
        Check for performance regressions.
        
        Args:
            threshold: Percentage threshold for regression (0.2 = 20%)
        
        Returns:
            List of regressions with details
        """
        baseline = self.load_baseline()
        
        if not baseline:
            print("No baseline available - establishing new baseline")
            self.save_baseline()
            return []
        
        regressions = []
        
        for test_name, current_time in self.current_metrics.items():
            baseline_time = baseline.get(test_name)
            
            if baseline_time is None:
                # New test - add to baseline
                continue
            
            # Check if current time exceeds baseline by threshold
            if current_time > baseline_time * (1 + threshold):
                regression_pct = ((current_time - baseline_time) / baseline_time) * 100
                
                regression = {
                    'test': test_name,
                    'baseline_time': baseline_time,
                    'current_time': current_time,
                    'regression_pct': regression_pct,
                    'slowdown': current_time - baseline_time,
                }
                
                regressions.append(regression)
        
        self.regressions = regressions
        return regressions
    
    def report_regressions(self, regressions: Optional[List[Dict[str, any]]] = None) -> None:
        """Print regression report."""
        if regressions is None:
            regressions = self.regressions
        
        if not regressions:
            print("✅ No performance regressions detected")
            return
        
        print(f"\n⚠️  Found {len(regressions)} performance regression(s):\n")
        
        # Sort by regression percentage
        regressions.sort(key=lambda x: x['regression_pct'], reverse=True)
        
        for i, reg in enumerate(regressions[:10], 1):  # Show top 10
            print(f"{i}. {reg['test']}")
            print(f"   Baseline: {reg['baseline_time']:.3f}s → Current: {reg['current_time']:.3f}s")
            print(f"   Regression: {reg['regression_pct']:.1f}% slower (+{reg['slowdown']:.3f}s)\n")
        
        if len(regressions) > 10:
            print(f"   ... and {len(regressions) - 10} more regressions")
    
    def save_regression_report(self, output_file: str = "performance-regressions.json") -> None:
        """Save regression report to JSON file."""
        report = {
            'total_regressions': len(self.regressions),
            'regressions': self.regressions,
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"✅ Regression report saved to {output_file}")
        except Exception as e:
            print(f"Error saving regression report: {e}")
    
    def get_summary(self) -> Dict[str, any]:
        """Get performance summary statistics."""
        if not self.current_metrics:
            return {}
        
        timings = list(self.current_metrics.values())
        
        return {
            'total_tests': len(timings),
            'total_time': sum(timings),
            'average_time': sum(timings) / len(timings),
            'min_time': min(timings),
            'max_time': max(timings),
            'regressions_count': len(self.regressions),
        }


def main():
    """Main function for performance monitoring."""
    parser = argparse.ArgumentParser(
        description="Monitor test performance and detect regressions"
    )
    parser.add_argument(
        "--junit-xml",
        default="pytest-results.xml",
        help="Path to pytest JUnit XML results",
    )
    parser.add_argument(
        "--baseline",
        default="tests/baselines/performance.json",
        help="Path to baseline performance file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Regression threshold as decimal (0.2 = 20%)",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save current metrics as new baseline",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with error code if regressions detected",
    )
    parser.add_argument(
        "--report-file",
        help="Save regression report to this file",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Only output the count of regressions (for scripting)",
    )
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = PerformanceMonitor(baseline_file=args.baseline)
    
    # Load test timing from pytest results
    monitor.load_pytest_timing(args.junit_xml)
    
    if not monitor.current_metrics:
        print("⚠️ No test metrics found")
        sys.exit(0)
    
    # Save baseline if requested
    if args.save_baseline:
        monitor.save_baseline()
        print("✅ New baseline saved")
        sys.exit(0)
    
    # Check for regressions
    regressions = monitor.check_regressions(threshold=args.threshold)
    
    # Output count only if requested
    if args.count:
        print(len(regressions))
        sys.exit(0)
    
    # Report regressions
    if regressions:
        monitor.report_regressions(regressions)
        
        # Save regression report
        if args.report_file:
            monitor.save_regression_report(args.report_file)
        
        # Fail if requested
        if args.fail_on_regression:
            print(f"\n❌ Performance regressions detected ({len(regressions)} tests)")
            sys.exit(1)
    else:
        print("✅ No performance regressions detected")
    
    # Print summary
    summary = monitor.get_summary()
    if summary:
        print(f"\n📊 Performance Summary:")
        print(f"   Total tests: {summary['total_tests']}")
        print(f"   Total time: {summary['total_time']:.2f}s")
        print(f"   Average time: {summary['average_time']:.3f}s")
        print(f"   Min/Max: {summary['min_time']:.3f}s / {summary['max_time']:.3f}s")
    
    sys.exit(0)


if __name__ == "__main__":
    main()
