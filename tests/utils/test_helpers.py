#!/usr/bin/env python3
"""
Common Testing Helpers

Utility functions and classes to assist with testing.
"""

import time
import functools
from typing import Any, Callable, Dict, List, Optional
from contextlib import contextmanager


def retry_on_failure(max_attempts: int = 3, delay: float = 0.1):
    """Decorator to retry a function on failure."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)

        return wrapper

    return decorator


@contextmanager
def assert_execution_time(max_duration: float, operation: str = "operation"):
    """Context manager to assert execution time."""
    start_time = time.time()
    yield
    duration = time.time() - start_time
    assert (
        duration <= max_duration
    ), f"{operation} took {duration:.3f}s, expected <= {max_duration:.3f}s"


@contextmanager
def suppress_logging():
    """Context manager to suppress logging during tests."""
    import logging

    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


def wait_for_condition(
    condition: Callable[[], bool],
    timeout: float = 5.0,
    interval: float = 0.1,
    error_message: str = "Condition not met",
):
    """Wait for a condition to become true."""
    end_time = time.time() + timeout
    while time.time() < end_time:
        if condition():
            return True
        time.sleep(interval)

    raise TimeoutError(f"{error_message} after {timeout}s")


def assert_dict_subset(subset: Dict, superset: Dict, path: str = ""):
    """Assert that one dictionary is a subset of another."""
    for key, value in subset.items():
        current_path = f"{path}.{key}" if path else key
        assert key in superset, f"Missing key: {current_path}"

        if isinstance(value, dict):
            assert isinstance(
                superset[key], dict
            ), f"Type mismatch at {current_path}: expected dict"
            assert_dict_subset(value, superset[key], current_path)
        else:
            assert (
                superset[key] == value
            ), f"Value mismatch at {current_path}: {superset[key]} != {value}"


def generate_mock_data(schema: Dict[str, type], count: int = 1) -> List[Dict[str, Any]]:
    """Generate mock data based on a schema."""
    import random
    import string

    def generate_value(value_type: type) -> Any:
        if value_type == str:
            return "".join(random.choices(string.ascii_letters, k=10))
        elif value_type == int:
            return random.randint(0, 1000)
        elif value_type == float:
            return random.uniform(0.0, 1.0)
        elif value_type == bool:
            return random.choice([True, False])
        else:
            return None

    data = []
    for _ in range(count):
        item = {key: generate_value(value_type) for key, value_type in schema.items()}
        data.append(item)

    return data


class TestTimer:
    """Context manager and decorator for timing test execution."""

    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = 0.0
        self.end_time = 0.0
        self.duration = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        print(f"{self.name} took {self.duration:.3f}s")

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


def create_test_id(prefix: str = "test") -> str:
    """Create a unique test ID."""
    import uuid

    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def compare_images(img1, img2, threshold: float = 0.95) -> bool:
    """Compare two images for similarity."""
    import numpy as np

    if img1.shape != img2.shape:
        return False

    # Calculate correlation coefficient
    correlation = np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
    return correlation >= threshold


def assert_eventually(
    assertion: Callable[[], None],
    timeout: float = 5.0,
    interval: float = 0.1,
    error_message: str = "Assertion not met",
):
    """Assert that a condition eventually becomes true."""
    end_time = time.time() + timeout
    last_error = None

    while time.time() < end_time:
        try:
            assertion()
            return
        except AssertionError as e:
            last_error = e
            time.sleep(interval)

    raise AssertionError(f"{error_message} after {timeout}s. Last error: {last_error}")


class EventCollector:
    """Collect events for testing."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def collect(self, event: Dict[str, Any]):
        """Collect an event."""
        self.events.append(event)

    def get_events(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get collected events, optionally filtered by type."""
        if event_type is None:
            return self.events.copy()
        return [e for e in self.events if e.get("event_type") == event_type]

    def clear(self):
        """Clear collected events."""
        self.events.clear()

    def count(self, event_type: Optional[str] = None) -> int:
        """Count events, optionally filtered by type."""
        return len(self.get_events(event_type))

    def wait_for_event(
        self, event_type: str, timeout: float = 5.0, interval: float = 0.1
    ) -> Optional[Dict[str, Any]]:
        """Wait for a specific event type."""
        end_time = time.time() + timeout
        while time.time() < end_time:
            events = self.get_events(event_type)
            if events:
                return events[-1]
            time.sleep(interval)
        return None
