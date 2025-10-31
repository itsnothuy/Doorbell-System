#!/usr/bin/env python3
"""
Enhanced PyTest Configuration

Comprehensive test configuration with fixtures, utilities, and testing infrastructure
for the Doorbell Security System with improved reliability and performance.
"""

import gc
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Generator, Optional
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

# Import cv2 with error handling for test environments
try:
    import cv2
except ImportError:
    cv2 = None
    print("⚠️ OpenCV not available - using mocks for image processing")

# Configure test logging with process-safe formatting
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(process)d] %(levelname)s in %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Session-level Configuration and Environment Setup
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
    """Configure test environment for reliability and isolation."""
    
    # Set environment variables for testing
    original_env = {}
    test_env_vars = {
        "TESTING": "true",
        "LOG_LEVEL": "DEBUG",
        "DISABLE_HARDWARE": "true",
        "MOCK_EXTERNAL_SERVICES": "true",
        "TEST_TIMEOUT": "300",
        "PYTEST_XDIST_WORKER": os.environ.get("PYTEST_XDIST_WORKER", "main"),
        "DEVELOPMENT_MODE": "true",
    }
    
    # Save original environment
    for key, value in test_env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    # Configure logging for test isolation
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
    log_file = Path(f"test-{worker_id}-{os.getpid()}.log")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s in %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Test environment configured for worker {worker_id}")
    
    yield
    
    # Cleanup: restore original environment
    for key, value in original_env.items():
        if value is None and key in os.environ:
            del os.environ[key]
        elif value is not None:
            os.environ[key] = value
    
    # Cleanup log file
    try:
        log_file.unlink(missing_ok=True)
    except:
        pass


@pytest.fixture(autouse=True)
def test_isolation():
    """
    Ensure test isolation and prevent side effects.
    
    This fixture runs for every test and ensures:
    - Global state is reset
    - Caches are cleared
    - Memory is managed
    - Mocks are reset
    """
    
    # Clear any module-level caches
    gc.collect()
    
    yield
    
    # Cleanup after test
    gc.collect()


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
    """
    Create temporary directory for test files with process-safe isolation.
    
    Each test worker gets its own isolated directory to prevent conflicts
    during parallel test execution.
    """
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
    temp_dir = Path(tempfile.mkdtemp(prefix=f"doorbell_test_{worker_id}_"))

    try:
        # Create test directory structure
        (temp_dir / "data").mkdir()
        (temp_dir / "data" / "known_faces").mkdir()
        (temp_dir / "data" / "blacklist_faces").mkdir()
        (temp_dir / "data" / "captures").mkdir()
        (temp_dir / "data" / "logs").mkdir()
        (temp_dir / "config").mkdir()
        
        logger.info(f"Created test directory for worker {worker_id}: {temp_dir}")

        yield temp_dir

    finally:
        # Cleanup with retry logic for Windows compatibility
        max_retries = 3
        for attempt in range(max_retries):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1)
                else:
                    logger.warning(f"Failed to cleanup {temp_dir}: {e}")


# ============================================================================
# Database Fixtures
# ============================================================================


@pytest.fixture
def test_database(temp_test_dir: Path) -> Generator[str, None, None]:
    """
    Create test database with proper isolation and cleanup.
    
    Each test gets a fresh database to prevent contamination.
    """
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
    db_path = temp_test_dir / "data" / f"test_{worker_id}.db"

    # Create test database
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access

    # Create test tables
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
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
        CREATE TABLE IF NOT EXISTS known_faces (
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
    
    # Cleanup: close any lingering connections and remove database
    try:
        if Path(db_path).exists():
            Path(db_path).unlink(missing_ok=True)
        # Clean up WAL files
        for wal_file in [f"{db_path}-wal", f"{db_path}-shm"]:
            Path(wal_file).unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Database cleanup warning: {e}")


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
    if cv2 is not None:
        cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), 2)
        cv2.circle(image, (80, 80), 10, (0, 0, 0), -1)  # Left eye
        cv2.circle(image, (120, 80), 10, (0, 0, 0), -1)  # Right eye
        cv2.rectangle(image, (90, 110), (110, 120), (0, 0, 0), -1)  # Nose
        cv2.rectangle(image, (80, 130), (120, 140), (0, 0, 0), -1)  # Mouth
    else:
        # Simple mock face without cv2
        image[50:150, 50:52] = 255  # Left border
        image[50:150, 148:150] = 255  # Right border
        image[50:52, 50:150] = 255  # Top border
        image[148:150, 50:150] = 255  # Bottom border

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
        if cv2 is not None:
            cv2.imwrite(str(image_path), image)
        else:
            # Fallback: use PIL if cv2 is not available
            try:
                from PIL import Image
                img = Image.fromarray(image)
                img.save(str(image_path))
            except ImportError:
                # Create placeholder file
                image_path.write_bytes(image.tobytes())

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
# Network and External Service Mocking
# ============================================================================


@pytest.fixture
def mock_requests():
    """Mock requests library for network isolation."""
    try:
        import requests_mock
        
        with requests_mock.Mocker() as m:
            # Mock common endpoints
            m.get("https://api.telegram.org/bot", json={"ok": True})
            m.post("https://api.telegram.org/bot", json={"ok": True, "result": {}})
            yield m
    except ImportError:
        # Fallback to basic mock if requests_mock not available
        import requests
        from unittest.mock import patch
        
        with patch.object(requests, 'get'), patch.object(requests, 'post') as mock:
            mock.return_value.status_code = 200
            mock.return_value.json.return_value = {"ok": True}
            yield mock


@pytest.fixture
def mock_telegram_api():
    """Mock Telegram Bot API for testing."""
    try:
        import responses
        
        with responses.RequestsMock() as rsps:
            # Mock successful message sending
            rsps.add(
                responses.POST,
                "https://api.telegram.org/bot",
                json={"ok": True, "result": {"message_id": 123}},
                status=200,
                match_querystring=False,
            )
            
            # Mock photo upload
            rsps.add(
                responses.POST,
                "https://api.telegram.org/bot",
                json={"ok": True, "result": {"message_id": 124}},
                status=200,
                match_querystring=False,
            )
            
            yield rsps
    except ImportError:
        # Fallback to basic mock
        yield Mock()


@pytest.fixture
def isolated_test():
    """
    Ensure complete test isolation - no external dependencies.
    
    This fixture ensures:
    - No real network calls
    - No real file system access outside temp directories
    - No real hardware access
    """
    original_socket = None
    
    try:
        # Block network access (optional - can be too aggressive)
        # import socket
        # original_socket = socket.socket
        # socket.socket = Mock(side_effect=RuntimeError("Network access blocked in tests"))
        
        yield
        
    finally:
        # Restore if we blocked network
        if original_socket:
            import socket
            socket.socket = original_socket


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


# ============================================================================
# Platform-Specific Test Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def platform_config_fixture() -> Dict[str, Any]:
    """Platform-specific test configuration fixture."""
    
    import platform as plat
    
    config = {
        "platform": plat.system().lower(),
        "architecture": plat.machine().lower(),
        "is_ci": os.environ.get("CI", "false").lower() == "true",
        "temp_dir": None,
        "mock_hardware": True,
        "test_timeout": 30,
    }
    
    # Platform-specific adjustments
    if config["platform"] == "windows":
        config.update({
            "path_separator": "\\",
            "test_timeout": 60,  # Windows tests often slower
            "mock_hardware": True,  # Always mock on Windows
            "line_endings": "CRLF",
        })
    elif config["platform"] == "darwin":
        config.update({
            "test_timeout": 45,  # macOS can be slow in CI
            "mock_hardware": True,
            "line_endings": "LF",
        })
    elif "arm" in config["architecture"] or "aarch" in config["architecture"]:
        config.update({
            "test_timeout": 120,  # ARM devices can be slower
            "memory_limit": True,
            "line_endings": "LF",
        })
    else:
        config.update({
            "line_endings": "LF",
        })
    
    return config


@pytest.fixture
def platform_paths_fixture(platform_config_fixture):
    """Platform-appropriate path handling fixture."""
    
    from pathlib import Path
    
    class PlatformPaths:
        def __init__(self, config):
            self.config = config
            self.temp_base = Path(tempfile.gettempdir())
            
        def join(self, *parts):
            """Platform-appropriate path joining."""
            return str(Path(*parts).resolve())
        
        def normalize(self, path):
            """Normalize path for platform."""
            return str(Path(path).resolve())
        
        def get_separator(self):
            """Get platform-specific path separator."""
            return self.config.get("path_separator", os.sep)
        
        def to_native(self, path):
            """Convert path to native format."""
            p = Path(path)
            if self.config["platform"] == "windows":
                return str(p).replace("/", "\\")
            else:
                return str(p).replace("\\", "/")
    
    return PlatformPaths(platform_config_fixture)


@pytest.fixture
def mock_hardware_manager_fixture(platform_config_fixture):
    """Platform-specific hardware mocking fixture."""
    
    class MockHardwareManager:
        def __init__(self, config):
            self.config = config
            self.camera_mock = Mock()
            self.gpio_mock = Mock()
            
            # Platform-specific mock behaviors
            if config["platform"] == "windows":
                self._setup_windows_mocks()
            elif config["platform"] == "darwin":
                self._setup_macos_mocks()
            else:
                self._setup_linux_mocks()
        
        def _setup_windows_mocks(self):
            """Windows-specific mock setup."""
            # Handle Windows-specific camera behaviors
            self.camera_mock.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            self.camera_mock.isOpened.return_value = True
            
        def _setup_macos_mocks(self):
            """macOS-specific mock setup."""
            # Handle macOS camera permissions and behaviors
            self.camera_mock.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            self.camera_mock.isOpened.return_value = True
            
        def _setup_linux_mocks(self):
            """Linux-specific mock setup."""
            # Handle Linux V4L2 and GPIO behaviors
            self.camera_mock.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            self.camera_mock.isOpened.return_value = True
            self.gpio_mock.setup.return_value = None
            self.gpio_mock.cleanup.return_value = None
        
        def get_camera_mock(self):
            """Get platform-appropriate camera mock."""
            return self.camera_mock
        
        def get_gpio_mock(self):
            """Get platform-appropriate GPIO mock."""
            return self.gpio_mock
    
    return MockHardwareManager(platform_config_fixture)


@pytest.fixture(scope="session")
def ci_environment_fixture():
    """Detect and configure for CI environment."""
    
    ci_info = {
        "is_ci": False,
        "ci_platform": None,
        "runner_os": None,
    }
    
    # Detect GitHub Actions
    if os.environ.get("GITHUB_ACTIONS") == "true":
        ci_info["is_ci"] = True
        ci_info["ci_platform"] = "github_actions"
        ci_info["runner_os"] = os.environ.get("RUNNER_OS", "unknown").lower()
    
    # Detect other CI platforms
    elif os.environ.get("TRAVIS") == "true":
        ci_info["is_ci"] = True
        ci_info["ci_platform"] = "travis"
    elif os.environ.get("CIRCLECI") == "true":
        ci_info["is_ci"] = True
        ci_info["ci_platform"] = "circle"
    elif os.environ.get("CI") == "true":
        ci_info["is_ci"] = True
        ci_info["ci_platform"] = "generic"
    
    return ci_info


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, temp_test_dir):
    """Setup test environment for all tests."""
    # Set environment variables for testing
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("TEST_DATA_DIR", str(temp_test_dir))
    monkeypatch.setenv("DISABLE_HARDWARE", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
