#!/usr/bin/env python3
"""
Performance Regression Testing Framework

Automated performance benchmarking with baseline comparison and regression detection.
"""

import json
import logging
import platform
import statistics
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

    # Create mock psutil for environments where it's not available
    class MockProcess:
        def memory_info(self) -> Any:
            class MemInfo:
                rss = 0

            return MemInfo()

        def cpu_percent(self) -> float:
            return 0.0

    class MockPsutil:
        @staticmethod
        def Process() -> MockProcess:
            return MockProcess()

        @staticmethod
        def cpu_count() -> int:
            return 1

        @staticmethod
        def virtual_memory() -> Any:
            class VirtMem:
                total = 0

            return VirtMem()

    psutil = MockPsutil()  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression testing."""

    test_name: str
    mean_duration: float
    std_duration: float
    mean_memory: float
    std_memory: float
    mean_cpu: float
    std_cpu: float
    sample_count: int
    timestamp: float
    environment: dict[str, Any]


@dataclass
class PerformanceMetrics:
    """Performance measurement data."""

    duration: float
    memory_peak: float
    memory_avg: float
    cpu_peak: float
    cpu_avg: float
    custom_metrics: dict[str, float] = field(default_factory=dict)


class PerformanceRegressor:
    """Performance regression testing framework."""

    def __init__(self, baseline_path: Path = Path("tests/baselines")):
        self.baseline_path = baseline_path
        self.baseline_path.mkdir(parents=True, exist_ok=True)
        self.baselines: dict[str, PerformanceBaseline] = {}
        self._load_baselines()

    def measure_performance(
        self, test_name: str, test_func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> PerformanceMetrics:
        """Measure performance of a test function."""
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, performance metrics will be limited")

        # Start monitoring
        process = psutil.Process()
        start_time = time.time()

        try:
            start_memory = process.memory_info().rss
        except Exception:
            start_memory = 0

        cpu_samples: list[float] = []
        memory_samples: list[float] = []

        # Create monitoring task
        monitoring = True

        def monitor() -> None:
            while monitoring:
                try:
                    cpu_samples.append(process.cpu_percent())
                    memory_samples.append(process.memory_info().rss)
                    time.sleep(0.1)
                except Exception:
                    break

        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()

        try:
            # Execute test function
            test_func(*args, **kwargs)
        finally:
            monitoring = False
            monitor_thread.join(timeout=1.0)

        end_time = time.time()

        try:
            end_memory = process.memory_info().rss
        except Exception:
            end_memory = start_memory

        # Calculate metrics
        duration = end_time - start_time
        memory_peak = max(memory_samples) if memory_samples else end_memory
        memory_avg = statistics.mean(memory_samples) if memory_samples else end_memory
        cpu_peak = max(cpu_samples) if cpu_samples else 0.0
        cpu_avg = statistics.mean(cpu_samples) if cpu_samples else 0.0

        return PerformanceMetrics(
            duration=duration,
            memory_peak=memory_peak,
            memory_avg=memory_avg,
            cpu_peak=cpu_peak,
            cpu_avg=cpu_avg,
        )

    def check_regression(
        self,
        test_name: str,
        metrics: PerformanceMetrics,
        threshold: float = 0.15,
    ) -> dict[str, Any]:
        """Check for performance regression against baseline."""
        if test_name not in self.baselines:
            return {
                "regression_detected": False,
                "reason": "no_baseline",
                "message": f"No baseline found for {test_name}",
            }

        baseline = self.baselines[test_name]

        # Check duration regression
        duration_change = (metrics.duration - baseline.mean_duration) / baseline.mean_duration
        memory_change = (metrics.memory_peak - baseline.mean_memory) / baseline.mean_memory
        cpu_change = (metrics.cpu_peak - baseline.mean_cpu) / baseline.mean_cpu

        regressions = []

        if duration_change > threshold:
            regressions.append(f"Duration: {duration_change:.1%} slower")

        if memory_change > threshold:
            regressions.append(f"Memory: {memory_change:.1%} more")

        if cpu_change > threshold:
            regressions.append(f"CPU: {cpu_change:.1%} higher")

        return {
            "regression_detected": len(regressions) > 0,
            "regressions": regressions,
            "changes": {
                "duration": duration_change,
                "memory": memory_change,
                "cpu": cpu_change,
            },
            "threshold": threshold,
        }

    def update_baseline(self, test_name: str, metrics: list[PerformanceMetrics]) -> None:
        """Update performance baseline with new measurements."""
        if not metrics:
            return

        # Calculate baseline statistics
        durations = [m.duration for m in metrics]
        memories = [m.memory_peak for m in metrics]
        cpus = [m.cpu_peak for m in metrics]

        baseline = PerformanceBaseline(
            test_name=test_name,
            mean_duration=statistics.mean(durations),
            std_duration=statistics.stdev(durations) if len(durations) > 1 else 0.0,
            mean_memory=statistics.mean(memories),
            std_memory=statistics.stdev(memories) if len(memories) > 1 else 0.0,
            mean_cpu=statistics.mean(cpus),
            std_cpu=statistics.stdev(cpus) if len(cpus) > 1 else 0.0,
            sample_count=len(metrics),
            timestamp=time.time(),
            environment=self._get_environment_info(),
        )

        self.baselines[test_name] = baseline
        self._save_baselines()

    def _load_baselines(self) -> None:
        """Load performance baselines from disk."""
        baseline_file = self.baseline_path / "performance_baselines.json"
        if baseline_file.exists():
            try:
                data = json.loads(baseline_file.read_text())
                self.baselines = {
                    name: PerformanceBaseline(**baseline_data)
                    for name, baseline_data in data.items()
                }
                logger.info(f"Loaded {len(self.baselines)} performance baselines")
            except Exception as e:
                logger.warning(f"Could not load baselines: {e}")

    def _save_baselines(self) -> None:
        """Save performance baselines to disk."""
        baseline_file = self.baseline_path / "performance_baselines.json"
        data = {
            name: {
                "test_name": baseline.test_name,
                "mean_duration": baseline.mean_duration,
                "std_duration": baseline.std_duration,
                "mean_memory": baseline.mean_memory,
                "std_memory": baseline.std_memory,
                "mean_cpu": baseline.mean_cpu,
                "std_cpu": baseline.std_cpu,
                "sample_count": baseline.sample_count,
                "timestamp": baseline.timestamp,
                "environment": baseline.environment,
            }
            for name, baseline in self.baselines.items()
        }
        baseline_file.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved {len(self.baselines)} performance baselines")

    def _get_environment_info(self) -> dict[str, Any]:
        """Get current environment information."""
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count() if PSUTIL_AVAILABLE else 1,
            "memory_total": (psutil.virtual_memory().total if PSUTIL_AVAILABLE else 0),
            "timestamp": time.time(),
        }

    def benchmark_function(
        self,
        test_name: str,
        test_func: Callable[..., Any],
        iterations: int = 5,
        warmup: int = 2,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Benchmark a function with multiple iterations.

        Args:
            test_name: Name of the test
            test_func: Function to benchmark
            iterations: Number of benchmark iterations
            warmup: Number of warmup iterations
            *args: Arguments for test function
            **kwargs: Keyword arguments for test function

        Returns:
            Dictionary with benchmark results and regression analysis
        """
        logger.info(f"Benchmarking {test_name} with {warmup} warmup and {iterations} iterations")

        # Warmup runs
        for i in range(warmup):
            logger.debug(f"Warmup iteration {i + 1}/{warmup}")
            test_func(*args, **kwargs)

        # Benchmark runs
        metrics_list = []
        for i in range(iterations):
            logger.debug(f"Benchmark iteration {i + 1}/{iterations}")
            metrics = self.measure_performance(test_name, test_func, *args, **kwargs)
            metrics_list.append(metrics)

        # Calculate statistics
        durations = [m.duration for m in metrics_list]
        mean_duration = statistics.mean(durations)
        std_duration = statistics.stdev(durations) if len(durations) > 1 else 0.0

        # Check for regression
        regression_result = self.check_regression(test_name, metrics_list[0])

        return {
            "test_name": test_name,
            "iterations": iterations,
            "mean_duration": mean_duration,
            "std_duration": std_duration,
            "min_duration": min(durations),
            "max_duration": max(durations),
            "metrics": [
                {
                    "duration": m.duration,
                    "memory_peak": m.memory_peak,
                    "memory_avg": m.memory_avg,
                    "cpu_peak": m.cpu_peak,
                    "cpu_avg": m.cpu_avg,
                }
                for m in metrics_list
            ],
            "regression": regression_result,
        }
