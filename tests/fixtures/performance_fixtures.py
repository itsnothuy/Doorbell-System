#!/usr/bin/env python3
"""
Performance Testing Fixtures

Fixtures for performance monitoring, benchmarking, and load testing.
"""

import pytest
import time
import psutil
import threading
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    operation: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        return self.duration * 1000


class PerformanceBenchmark:
    """Performance benchmarking utility."""

    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.process = psutil.Process()

    def measure(self, operation: str, func: Callable, *args, **kwargs) -> Any:
        """Measure performance of a function execution."""
        metric = PerformanceMetrics(operation=operation)
        metric.start_time = time.time()

        # Measure resource usage before
        cpu_before = self.process.cpu_percent()
        mem_before = self.process.memory_info().rss / 1024 / 1024

        try:
            result = func(*args, **kwargs)
            metric.success = True
        except Exception as e:
            metric.success = False
            metric.error = str(e)
            result = None

        # Measure resource usage after
        metric.end_time = time.time()
        metric.duration = metric.end_time - metric.start_time
        metric.cpu_percent = self.process.cpu_percent() - cpu_before
        metric.memory_mb = (self.process.memory_info().rss / 1024 / 1024) - mem_before

        self.metrics.append(metric)
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all measurements."""
        if not self.metrics:
            return {}

        operations = defaultdict(list)
        for metric in self.metrics:
            operations[metric.operation].append(metric)

        summary = {}
        for operation, metrics_list in operations.items():
            durations = [m.duration for m in metrics_list]
            summary[operation] = {
                "count": len(metrics_list),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "total_duration": sum(durations),
                "success_rate": sum(m.success for m in metrics_list) / len(metrics_list),
            }

        return summary


class LoadGenerator:
    """Generate load for testing."""

    def __init__(self, rate: float = 1.0):
        """
        Initialize load generator.

        Args:
            rate: Events per second
        """
        self.rate = rate
        self.running = False
        self.events_generated = 0
        self.thread: Optional[threading.Thread] = None

    def start(self, callback: Callable, duration: float = 10.0):
        """Start generating load."""
        self.running = True
        self.events_generated = 0

        def generate():
            end_time = time.time() + duration
            while self.running and time.time() < end_time:
                callback()
                self.events_generated += 1
                time.sleep(1.0 / self.rate)

        self.thread = threading.Thread(target=generate)
        self.thread.start()

    def stop(self):
        """Stop generating load."""
        self.running = False
        if self.thread:
            self.thread.join()


@pytest.fixture
def performance_benchmark():
    """Performance benchmark fixture."""
    return PerformanceBenchmark()


@pytest.fixture
def load_generator():
    """Load generator fixture."""
    generator = LoadGenerator()
    yield generator
    generator.stop()


@pytest.fixture
def performance_requirements():
    """Standard performance requirements for testing."""
    return {
        "frame_capture_duration": 0.1,  # 100ms max per frame
        "motion_detection_duration": 0.05,  # 50ms max
        "face_detection_duration": 0.5,  # 500ms max
        "face_recognition_duration": 0.3,  # 300ms max
        "event_processing_duration": 0.1,  # 100ms max
        "end_to_end_latency": 2.0,  # 2 seconds max from trigger to notification
        "throughput_fps": 10.0,  # 10 frames per second minimum
        "memory_usage_mb": 500.0,  # 500MB max memory usage
        "cpu_usage_percent": 80.0,  # 80% max CPU usage
    }


@pytest.fixture
def stress_test_config():
    """Configuration for stress testing."""
    return {
        "concurrent_users": 10,
        "requests_per_second": 100,
        "duration_seconds": 60,
        "ramp_up_time": 10,
        "test_scenarios": [
            "doorbell_trigger",
            "face_recognition",
            "video_streaming",
            "api_requests",
        ],
    }


class ResourceMonitor:
    """Monitor system resources during tests."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.thread: Optional[threading.Thread] = None
        self.measurements: List[Dict[str, float]] = []
        self.process = psutil.Process()

    def start(self):
        """Start monitoring resources."""
        self.monitoring = True
        self.measurements = []

        def monitor():
            while self.monitoring:
                measurement = {
                    "timestamp": time.time(),
                    "cpu_percent": self.process.cpu_percent(),
                    "memory_mb": self.process.memory_info().rss / 1024 / 1024,
                    "memory_percent": self.process.memory_percent(),
                    "num_threads": self.process.num_threads(),
                }
                self.measurements.append(measurement)
                time.sleep(self.interval)

        self.thread = threading.Thread(target=monitor)
        self.thread.start()

    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return summary."""
        self.monitoring = False
        if self.thread:
            self.thread.join()

        if not self.measurements:
            return {}

        cpu_values = [m["cpu_percent"] for m in self.measurements]
        mem_values = [m["memory_mb"] for m in self.measurements]

        return {
            "duration": self.measurements[-1]["timestamp"] - self.measurements[0]["timestamp"],
            "samples": len(self.measurements),
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
            },
            "memory": {
                "avg": sum(mem_values) / len(mem_values),
                "max": max(mem_values),
                "min": min(mem_values),
            },
        }


@pytest.fixture
def resource_monitor():
    """Resource monitoring fixture."""
    monitor = ResourceMonitor()
    yield monitor
    if monitor.monitoring:
        monitor.stop()
