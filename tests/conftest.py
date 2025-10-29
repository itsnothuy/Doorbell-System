#!/usr/bin/env python3
"""
Enhanced PyTest Configuration

Comprehensive test configuration with fixtures, utilities, and testing infrastructure
for the Doorbell Security System.
"""

import logging
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Generator, Optional
from unittest.mock import MagicMock, Mock

import cv2
import numpy as np
import pytest

# Configure test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# ============================================================================
# Session-level Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Test configuration fixture."""
    return {
        "test_mode": True,
        "mock_hardware": True,
        "use_test_database": True,
        "disable_external_services": True,
        "performance_mode": False,
        "debug_mode": True,
    }


@pytest.fixture(scope="session")
def temp_test_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test files."""
    temp_dir = Path(tempfile.mkdtemp(prefix="doorbell_test_"))

    try:
        # Create test directory structure
        (temp_dir / "data").mkdir()
        (temp_dir / "data" / "known_faces").mkdir()
        (temp_dir / "data" / "blacklist_faces").mkdir()
        (temp_dir / "data" / "captures").mkdir()
        (temp_dir / "data" / "logs").mkdir()
        (temp_dir / "config").mkdir()

        yield temp_dir

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Database Fixtures
# ============================================================================


@pytest.fixture
def test_database(temp_test_dir: Path) -> Generator[str, None, None]:
    """Create test database."""
    db_path = temp_test_dir / "data" / "test.db"

    # Create test database
    conn = sqlite3.connect(str(db_path))

    # Create test tables
    conn.execute(
        """
        CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            event_type TEXT,
            timestamp REAL,
            data TEXT,
            source TEXT
        )
    """
    )

    conn.execute(
        """
        CREATE TABLE known_faces (
            id INTEGER PRIMARY KEY,
            person_name TEXT,
            face_encoding BLOB,
            image_path TEXT,
            created_at REAL
        )
    """
    )

    conn.commit()
    conn.close()

    yield str(db_path)


# ============================================================================
# Hardware Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_camera():
    """Mock camera fixture."""
    camera = Mock()
    camera.capture_array.return_value = np.random.randint(
        0, 255, (480, 640, 3), dtype=np.uint8
    )
    camera.start.return_value = None
    camera.stop.return_value = None
    camera.close.return_value = None
    return camera


@pytest.fixture
def mock_gpio():
    """Mock GPIO fixture."""
    gpio = Mock()
    gpio.setup.return_value = None
    gpio.input.return_value = False
    gpio.output.return_value = None
    gpio.cleanup.return_value = None
    gpio.add_event_detect.return_value = None
    gpio.remove_event_detect.return_value = None
    return gpio


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def sample_face_image() -> np.ndarray:
    """Generate sample face image for testing."""
    # Create a simple test image
    image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

    # Add a simple face-like rectangle (for testing purposes)
    cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), 2)
    cv2.circle(image, (80, 80), 10, (0, 0, 0), -1)  # Left eye
    cv2.circle(image, (120, 80), 10, (0, 0, 0), -1)  # Right eye
    cv2.rectangle(image, (90, 110), (110, 120), (0, 0, 0), -1)  # Nose
    cv2.rectangle(image, (80, 130), (120, 140), (0, 0, 0), -1)  # Mouth

    return image


@pytest.fixture
def sample_face_encoding() -> np.ndarray:
    """Generate sample face encoding for testing."""
    return np.random.random(128).astype(np.float64)


@pytest.fixture
def test_known_faces(
    temp_test_dir: Path, sample_face_image: np.ndarray
) -> Dict[str, Any]:
    """Create test known faces."""
    known_faces_dir = temp_test_dir / "data" / "known_faces"

    # Create test face images
    test_faces = {
        "john_doe": sample_face_image,
        "jane_smith": sample_face_image,
        "test_person": sample_face_image,
    }

    for name, image in test_faces.items():
        image_path = known_faces_dir / f"{name}_001.jpg"
        cv2.imwrite(str(image_path), image)

    return {"directory": known_faces_dir, "faces": test_faces, "count": len(test_faces)}


@pytest.fixture
def test_event_data():
    """Test event data fixture."""
    return {
        "doorbell_event": {
            "event_type": "doorbell_triggered",
            "timestamp": 1635724800.0,
            "data": {"trigger_type": "button_press"},
            "source": "gpio_handler",
        },
        "face_detection_event": {
            "event_type": "face_detected",
            "timestamp": 1635724801.0,
            "data": {
                "face_count": 1,
                "confidence": 0.95,
                "bounding_box": [100, 100, 200, 200],
            },
            "source": "face_detector",
        },
        "recognition_event": {
            "event_type": "face_recognized",
            "timestamp": 1635724802.0,
            "data": {"person_name": "john_doe", "confidence": 0.85, "is_known": True},
            "source": "face_recognizer",
        },
    }


# ============================================================================
# Communication Layer Fixtures
# ============================================================================


@pytest.fixture
def mock_message_bus():
    """Mock message bus fixture."""
    bus = Mock()
    bus.publish = Mock()
    bus.subscribe = Mock()
    bus.unsubscribe = Mock()
    bus.start = Mock()
    bus.stop = Mock()
    bus.get_metrics = Mock(return_value={"messages_sent": 0, "messages_received": 0})

    return bus


# ============================================================================
# Pipeline Component Fixtures
# ============================================================================


@pytest.fixture
def mock_pipeline_orchestrator():
    """Mock pipeline orchestrator fixture."""
    orchestrator = Mock()
    orchestrator.start = Mock()
    orchestrator.stop = Mock()
    orchestrator.get_health_status = Mock()
    orchestrator.trigger_doorbell = Mock(return_value={"status": "success"})
    orchestrator.is_running = Mock(return_value=False)

    return orchestrator


# ============================================================================
# Performance Monitoring Fixtures
# ============================================================================


@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture."""

    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {}

        def start_timer(self, operation: str):
            self.metrics[f"{operation}_start"] = time.time()

        def end_timer(self, operation: str):
            start_time = self.metrics.get(f"{operation}_start")
            if start_time:
                self.metrics[f"{operation}_duration"] = time.time() - start_time

        def get_metrics(self) -> Dict[str, float]:
            return {k: v for k, v in self.metrics.items() if not k.endswith("_start")}

    return PerformanceMonitor()


# ============================================================================
# Test Utilities
# ============================================================================


def assert_performance_requirements(
    metrics: Dict[str, float], requirements: Dict[str, float]
):
    """Assert performance requirements are met."""
    for metric, threshold in requirements.items():
        actual_value = metrics.get(metric)
        assert actual_value is not None, f"Missing performance metric: {metric}"
        assert (
            actual_value <= threshold
        ), f"Performance requirement failed: {metric} = {actual_value}, required <= {threshold}"


def generate_test_load(num_events: int = 100) -> list:
    """Generate test load data."""
    events = []
    base_time = time.time()

    for i in range(num_events):
        events.append(
            {
                "id": i,
                "timestamp": base_time + (i * 0.1),
                "event_type": "test_event",
                "data": {"test_data": f"event_{i}"},
            }
        )

    return events


# ============================================================================
# Pytest Configuration Hooks
# ============================================================================


def pytest_configure(config):
    """Configure pytest."""
    # Add custom markers
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "security: mark test as security test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "hardware: mark test as requiring hardware")


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Auto-mark tests based on location
    for item in items:
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, temp_test_dir):
    """Setup test environment for all tests."""
    # Set environment variables for testing
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("TEST_DATA_DIR", str(temp_test_dir))
    monkeypatch.setenv("DISABLE_HARDWARE", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
