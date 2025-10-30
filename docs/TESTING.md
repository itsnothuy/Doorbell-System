# Testing Guide

## 🧪 Testing Strategy Overview

The Doorbell Security System employs a comprehensive testing strategy to ensure reliability, security, and performance across different platforms and configurations. This guide covers all aspects of testing from unit tests to load testing and CI/CD automation.

## 📋 Table of Contents

1. [Testing Principles](#testing-principles)
2. [Test Structure](#test-structure)
3. [Test Types](#test-types)
4. [Running Tests](#running-tests)
5. [Test Utilities](#test-utilities)
6. [CI/CD Integration](#cicd-integration)
7. [Coverage Requirements](#coverage-requirements)
8. [Best Practices](#best-practices)

## 📋 Testing Principles

### Core Testing Principles

1. **Test Pyramid**: Unit tests > Integration tests > E2E tests
2. **Fast Feedback**: Quick test execution for rapid development
3. **Platform Coverage**: Test across different operating systems
4. **Hardware Abstraction**: Test with and without hardware
5. **Security First**: Security testing integrated throughout
6. **Performance Aware**: Performance regression prevention
7. **Property-Based Testing**: Verify properties hold for wide input ranges

### Testing Philosophy

- **Fail Fast**: Catch issues early in development
- **Comprehensive Coverage**: High test coverage across all components (target: 80%+)
- **Real-world Scenarios**: Test realistic usage patterns
- **Edge Cases**: Test boundary conditions and error scenarios
- **Backwards Compatibility**: Ensure existing functionality remains stable

## 🏗️ Test Structure

### Directory Organization

```
tests/
├── unit/                   # Unit tests for individual components
│   ├── test_face_manager.py
│   ├── test_camera_handler.py
│   ├── test_gpio_handler.py
│   └── ...
├── integration/            # Integration tests for component interactions
│   ├── test_complete_integration.py
│   ├── test_orchestrator_integration.py
│   └── test_end_to_end_pipeline.py
├── e2e/                    # End-to-end browser and system tests
│   ├── test_doorbell_scenarios.py
│   └── test_web_interface_playwright.py
├── performance/            # Performance and benchmark tests
│   ├── recognition_bench.py
│   └── test_communication_performance.py
├── security/               # Security-specific tests
│   └── test_input_validation.py
├── load/                   # Load testing scenarios
│   ├── locustfile.py
│   └── test_stress_scenarios.py
├── streaming/              # Streaming-specific tests
├── framework/              # Test framework utilities
│   ├── orchestrator.py
│   ├── performance.py
│   └── environments.py
├── utils/                  # Test utilities and helpers
│   └── property_based_tests.py
├── fixtures/               # Test data and fixtures
├── baselines/              # Baseline data for comparison
└── conftest.py            # Pytest configuration and shared fixtures
```

## 🔧 Test Configuration

### pytest Configuration (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "e2e: End-to-end tests",
    "performance: Performance tests",
    "security: Security tests",
    "load: Load testing scenarios",
    "property: Property-based tests",
    "slow: Slow tests (> 5 seconds)",
    "hardware: Tests requiring hardware",
    "network: Tests requiring network access",
    "gpu: Tests requiring GPU",
]
filterwarnings = [
    "error",
    "ignore::UserWarning",
    "ignore::DeprecationWarning",
]
```

## 🧩 Test Types

```python
# conftest.py
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import numpy as np
import cv2

from src.face_manager import FaceManager
from src.camera_handler import CameraHandler
from src.gpio_handler import GPIOHandler
from src.telegram_notifier import TelegramNotifier
from src.web_interface import WebInterface
from config.settings import load_config


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return {
        "face_recognition": {
            "tolerance": 0.6,
            "model": "hog",
            "unknown_threshold": 0.4
        },
        "camera": {
            "width": 640,
            "height": 480,
            "fps": 30
        },
        "telegram": {
            "enabled": False,  # Disabled for tests
            "bot_token": "test_token",
            "chat_id": "test_chat"
        },
        "web": {
            "host": "localhost",
            "port": 5000,
            "debug": True
        }
    }


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for tests."""
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir) / "data"
    data_dir.mkdir()
    
    # Create subdirectories
    for subdir in ["known_faces", "blacklist_faces", "captures", "logs", "cropped_faces/known", "cropped_faces/unknown"]:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    yield data_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_camera():
    """Mock camera for testing."""
    camera = Mock(spec=CameraHandler)
    
    # Create test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (200, 150), (440, 330), (255, 255, 255), -1)  # White rectangle for face
    
    camera.capture_image.return_value = test_image
    camera.is_available.return_value = True
    camera.cleanup.return_value = None
    
    return camera


@pytest.fixture
def mock_gpio():
    """Mock GPIO handler for testing."""
    gpio = Mock(spec=GPIOHandler)
    gpio.setup_doorbell_pin.return_value = None
    gpio.setup_led_pin.return_value = None
    gpio.set_led.return_value = None
    gpio.cleanup.return_value = None
    return gpio


@pytest.fixture
def mock_telegram():
    """Mock Telegram notifier for testing."""
    telegram = Mock(spec=TelegramNotifier)
    telegram.send_message.return_value = True
    telegram.send_photo.return_value = True
    return telegram


@pytest.fixture
def sample_face_image():
    """Create a sample face image for testing."""
    # Create a simple face-like image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a simple face
    center = (320, 240)
    cv2.circle(image, center, 100, (200, 180, 160), -1)  # Face circle
    cv2.circle(image, (290, 210), 15, (50, 50, 50), -1)   # Left eye
    cv2.circle(image, (350, 210), 15, (50, 50, 50), -1)   # Right eye
    cv2.ellipse(image, center, (30, 20), 0, 0, 180, (50, 50, 50), 2)  # Smile
    
    return image


@pytest.fixture
def face_manager(test_config, temp_data_dir):
    """Create FaceManager instance for testing."""
    config = test_config.copy()
    config["data_dir"] = str(temp_data_dir)
    
    return FaceManager(config["face_recognition"])


@pytest.fixture
def web_client(test_config, temp_data_dir):
    """Create Flask test client."""
    from src.web_interface import create_app
    
    config = test_config.copy()
    config["data_dir"] = str(temp_data_dir)
    config["testing"] = True
    
    app = create_app(config)
    
    with app.test_client() as client:
        with app.app_context():
            yield client
```

## 🧪 Unit Tests

### Face Manager Tests

```python
# tests/unit/test_face_manager.py
import pytest
import numpy as np
from unittest.mock import patch, Mock
import pickle
from pathlib import Path

from src.face_manager import FaceManager


class TestFaceManager:
    """Test suite for FaceManager class."""
    
    def test_init(self, face_manager):
        """Test FaceManager initialization."""
        assert face_manager.tolerance == 0.6
        assert face_manager.model == "hog"
        assert face_manager.known_faces == {}
        assert face_manager.blacklist_faces == {}
    
    def test_load_known_faces_empty_directory(self, face_manager, temp_data_dir):
        """Test loading from empty known faces directory."""
        face_manager.data_dir = temp_data_dir
        face_manager.load_known_faces()
        
        assert len(face_manager.known_faces) == 0
        assert len(face_manager.blacklist_faces) == 0
    
    def test_save_and_load_face_encoding(self, face_manager, temp_data_dir):
        """Test saving and loading face encodings."""
        face_manager.data_dir = temp_data_dir
        
        # Create test encoding
        test_encoding = np.random.rand(128).astype(np.float64)
        test_name = "test_person"
        
        # Save encoding
        face_manager.save_face_encoding(test_name, test_encoding)
        
        # Verify file exists
        encoding_file = temp_data_dir / "known_faces" / f"{test_name}.pkl"
        assert encoding_file.exists()
        
        # Load encodings
        face_manager.load_known_faces()
        
        # Verify encoding loaded correctly
        assert test_name in face_manager.known_faces
        loaded_encoding = face_manager.known_faces[test_name]
        np.testing.assert_array_almost_equal(loaded_encoding, test_encoding)
    
    @patch('face_recognition.face_encodings')
    @patch('face_recognition.face_locations')
    def test_identify_face_known_person(self, mock_locations, mock_encodings, face_manager, sample_face_image):
        """Test face identification for known person."""
        # Setup mocks
        test_encoding = np.random.rand(128).astype(np.float64)
        mock_locations.return_value = [(100, 200, 300, 150)]
        mock_encodings.return_value = [test_encoding]
        
        # Add known face
        face_manager.known_faces["john_doe"] = test_encoding
        
        # Test identification
        result = face_manager.identify_face(sample_face_image)
        
        assert result["status"] == "recognized"
        assert result["name"] == "john_doe"
        assert result["confidence"] > 0.5
        assert "bounding_box" in result
    
    @patch('face_recognition.face_encodings')
    @patch('face_recognition.face_locations')
    def test_identify_face_blacklisted_person(self, mock_locations, mock_encodings, face_manager, sample_face_image):
        """Test face identification for blacklisted person."""
        # Setup mocks
        test_encoding = np.random.rand(128).astype(np.float64)
        mock_locations.return_value = [(100, 200, 300, 150)]
        mock_encodings.return_value = [test_encoding]
        
        # Add blacklisted face
        face_manager.blacklist_faces["unwanted_person"] = test_encoding
        
        # Test identification
        result = face_manager.identify_face(sample_face_image)
        
        assert result["status"] == "blacklisted"
        assert result["name"] == "unwanted_person"
        assert result["alert"] is True
    
    @patch('face_recognition.face_encodings')
    @patch('face_recognition.face_locations')
    def test_identify_face_unknown_person(self, mock_locations, mock_encodings, face_manager, sample_face_image):
        """Test face identification for unknown person."""
        # Setup mocks
        test_encoding = np.random.rand(128).astype(np.float64)
        mock_locations.return_value = [(100, 200, 300, 150)]
        mock_encodings.return_value = [test_encoding]
        
        # Test identification (no known or blacklisted faces)
        result = face_manager.identify_face(sample_face_image)
        
        assert result["status"] == "unknown"
        assert result["name"] is None
        assert "confidence" in result
    
    @patch('face_recognition.face_locations')
    def test_identify_face_no_face_detected(self, mock_locations, face_manager, sample_face_image):
        """Test face identification when no face is detected."""
        # Setup mock to return no faces
        mock_locations.return_value = []
        
        # Test identification
        result = face_manager.identify_face(sample_face_image)
        
        assert result["status"] == "no_face"
        assert result["name"] is None
        assert result["confidence"] == 0.0
    
    def test_add_known_face(self, face_manager, temp_data_dir):
        """Test adding a new known face."""
        face_manager.data_dir = temp_data_dir
        
        test_encoding = np.random.rand(128).astype(np.float64)
        test_name = "new_person"
        
        face_manager.add_known_face(test_name, test_encoding)
        
        # Verify face added to memory
        assert test_name in face_manager.known_faces
        np.testing.assert_array_equal(face_manager.known_faces[test_name], test_encoding)
        
        # Verify face saved to disk
        encoding_file = temp_data_dir / "known_faces" / f"{test_name}.pkl"
        assert encoding_file.exists()
    
    def test_remove_known_face(self, face_manager, temp_data_dir):
        """Test removing a known face."""
        face_manager.data_dir = temp_data_dir
        
        test_encoding = np.random.rand(128).astype(np.float64)
        test_name = "person_to_remove"
        
        # Add face first
        face_manager.add_known_face(test_name, test_encoding)
        assert test_name in face_manager.known_faces
        
        # Remove face
        face_manager.remove_known_face(test_name)
        
        # Verify face removed from memory
        assert test_name not in face_manager.known_faces
        
        # Verify file deleted
        encoding_file = temp_data_dir / "known_faces" / f"{test_name}.pkl"
        assert not encoding_file.exists()
    
    def test_add_blacklist_face(self, face_manager, temp_data_dir):
        """Test adding a face to blacklist."""
        face_manager.data_dir = temp_data_dir
        
        test_encoding = np.random.rand(128).astype(np.float64)
        test_name = "unwanted_person"
        
        face_manager.add_blacklist_face(test_name, test_encoding)
        
        # Verify face added to blacklist
        assert test_name in face_manager.blacklist_faces
        np.testing.assert_array_equal(face_manager.blacklist_faces[test_name], test_encoding)
        
        # Verify face saved to disk
        encoding_file = temp_data_dir / "blacklist_faces" / f"{test_name}.pkl"
        assert encoding_file.exists()
    
    def test_face_encoding_cache(self, face_manager):
        """Test face encoding caching mechanism."""
        # Test that repeated calls use cache
        test_encoding = np.random.rand(128).astype(np.float64)
        
        with patch('face_recognition.face_encodings') as mock_encodings:
            mock_encodings.return_value = [test_encoding]
            
            # First call should hit the API
            result1 = face_manager._get_face_encodings(np.zeros((100, 100, 3)))
            assert mock_encodings.call_count == 1
            
            # Subsequent call with same image should use cache
            result2 = face_manager._get_face_encodings(np.zeros((100, 100, 3)))
            assert mock_encodings.call_count == 1  # No additional calls
            
            np.testing.assert_array_equal(result1[0], result2[0])
```

### Camera Handler Tests

```python
# tests/unit/test_camera_handler.py
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import cv2

from src.camera_handler import CameraHandler, PiCameraHandler, OpenCVCameraHandler, MockCameraHandler


class TestCameraHandler:
    """Test suite for CameraHandler classes."""
    
    @patch('src.platform_detector.is_raspberry_pi')
    @patch('src.platform_detector.has_picamera')
    def test_camera_handler_factory_pi(self, mock_has_picamera, mock_is_pi):
        """Test camera handler factory on Raspberry Pi."""
        mock_is_pi.return_value = True
        mock_has_picamera.return_value = True
        
        handler = CameraHandler.create()
        assert isinstance(handler, PiCameraHandler)
    
    @patch('src.platform_detector.is_raspberry_pi')
    def test_camera_handler_factory_non_pi(self, mock_is_pi):
        """Test camera handler factory on non-Pi systems."""
        mock_is_pi.return_value = False
        
        handler = CameraHandler.create()
        assert isinstance(handler, OpenCVCameraHandler)
    
    def test_mock_camera_handler(self):
        """Test MockCameraHandler functionality."""
        handler = MockCameraHandler()
        
        assert handler.is_available() is True
        
        image = handler.capture_image()
        assert image is not None
        assert image.shape == (480, 640, 3)
        assert image.dtype == np.uint8
        
        handler.cleanup()  # Should not raise


class TestOpenCVCameraHandler:
    """Test suite for OpenCVCameraHandler."""
    
    @patch('cv2.VideoCapture')
    def test_initialization_success(self, mock_video_capture):
        """Test successful camera initialization."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        handler = OpenCVCameraHandler()
        
        assert handler.camera_index == 0
        assert handler.cap is not None
        mock_video_capture.assert_called_once_with(0)
    
    @patch('cv2.VideoCapture')
    def test_initialization_failure(self, mock_video_capture):
        """Test camera initialization failure."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        handler = OpenCVCameraHandler()
        
        assert handler.cap is None
    
    @patch('cv2.VideoCapture')
    def test_capture_image_success(self, mock_video_capture):
        """Test successful image capture."""
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, test_image)
        mock_video_capture.return_value = mock_cap
        
        handler = OpenCVCameraHandler()
        captured_image = handler.capture_image()
        
        np.testing.assert_array_equal(captured_image, test_image)
    
    @patch('cv2.VideoCapture')
    def test_capture_image_failure(self, mock_video_capture):
        """Test image capture failure."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap
        
        handler = OpenCVCameraHandler()
        
        with pytest.raises(RuntimeError, match="Failed to capture image"):
            handler.capture_image()
    
    @patch('cv2.VideoCapture')
    def test_is_available(self, mock_video_capture):
        """Test camera availability check."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        handler = OpenCVCameraHandler()
        assert handler.is_available() is True
        
        # Test when camera is not available
        handler.cap = None
        assert handler.is_available() is False
    
    @patch('cv2.VideoCapture')
    def test_cleanup(self, mock_video_capture):
        """Test camera cleanup."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        handler = OpenCVCameraHandler()
        handler.cleanup()
        
        mock_cap.release.assert_called_once()
        assert handler.cap is None


class TestPiCameraHandler:
    """Test suite for PiCameraHandler."""
    
    @patch('picamera2.Picamera2')
    def test_initialization(self, mock_picamera2):
        """Test Pi camera initialization."""
        mock_camera = Mock()
        mock_picamera2.return_value = mock_camera
        
        handler = PiCameraHandler()
        
        assert handler.camera is not None
        mock_camera.start.assert_called_once()
    
    @patch('picamera2.Picamera2')
    def test_capture_image(self, mock_picamera2):
        """Test Pi camera image capture."""
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        mock_camera = Mock()
        mock_camera.capture_array.return_value = test_image
        mock_picamera2.return_value = mock_camera
        
        handler = PiCameraHandler()
        captured_image = handler.capture_image()
        
        np.testing.assert_array_equal(captured_image, test_image)
        mock_camera.capture_array.assert_called_once()
    
    @patch('picamera2.Picamera2')
    def test_is_available(self, mock_picamera2):
        """Test Pi camera availability."""
        mock_camera = Mock()
        mock_picamera2.return_value = mock_camera
        
        handler = PiCameraHandler()
        assert handler.is_available() is True
        
        # Test when camera is None
        handler.camera = None
        assert handler.is_available() is False
    
    @patch('picamera2.Picamera2')
    def test_cleanup(self, mock_picamera2):
        """Test Pi camera cleanup."""
        mock_camera = Mock()
        mock_picamera2.return_value = mock_camera
        
        handler = PiCameraHandler()
        handler.cleanup()
        
        mock_camera.stop.assert_called_once()
        mock_camera.close.assert_called_once()
        assert handler.camera is None
```

## 🔗 Integration Tests

### Doorbell Workflow Integration

```python
# tests/integration/test_doorbell_workflow.py
import pytest
from unittest.mock import Mock, patch
import time
import threading
from pathlib import Path

from src.doorbell_security import DoorbellSecuritySystem


class TestDoorbellWorkflow:
    """Integration tests for complete doorbell workflow."""
    
    @pytest.fixture
    def doorbell_system(self, test_config, temp_data_dir, mock_camera, mock_gpio, mock_telegram):
        """Create doorbell system with mocked dependencies."""
        config = test_config.copy()
        config["data_dir"] = str(temp_data_dir)
        
        with patch('src.camera_handler.CameraHandler.create', return_value=mock_camera), \
             patch('src.gpio_handler.GPIOHandler', return_value=mock_gpio), \
             patch('src.telegram_notifier.TelegramNotifier', return_value=mock_telegram):
            
            system = DoorbellSecuritySystem(config)
            yield system
            system.cleanup()
    
    def test_doorbell_press_unknown_person(self, doorbell_system, mock_camera, mock_telegram, sample_face_image):
        """Test complete workflow for unknown person at door."""
        # Setup camera to return face image
        mock_camera.capture_image.return_value = sample_face_image
        
        # Trigger doorbell press
        doorbell_system.on_doorbell_pressed(channel=18)
        
        # Wait for processing
        time.sleep(1)
        
        # Verify camera was used
        mock_camera.capture_image.assert_called()
        
        # Verify Telegram notification sent
        mock_telegram.send_message.assert_called()
        assert "Unknown person detected" in str(mock_telegram.send_message.call_args)
    
    def test_doorbell_press_known_person(self, doorbell_system, mock_camera, mock_telegram, sample_face_image, face_manager):
        """Test complete workflow for known person at door."""
        # Add known person
        import numpy as np
        test_encoding = np.random.rand(128).astype(np.float64)
        face_manager.add_known_face("john_doe", test_encoding)
        
        # Mock face recognition to return known person
        with patch.object(doorbell_system.face_manager, 'identify_face') as mock_identify:
            mock_identify.return_value = {
                "status": "recognized",
                "name": "john_doe",
                "confidence": 0.85,
                "bounding_box": (100, 150, 200, 250)
            }
            
            # Setup camera
            mock_camera.capture_image.return_value = sample_face_image
            
            # Trigger doorbell press
            doorbell_system.on_doorbell_pressed(channel=18)
            
            # Wait for processing
            time.sleep(1)
            
            # Verify Telegram notification for known person
            mock_telegram.send_message.assert_called()
            assert "john_doe" in str(mock_telegram.send_message.call_args)
    
    def test_doorbell_press_blacklisted_person(self, doorbell_system, mock_camera, mock_telegram, sample_face_image, face_manager):
        """Test complete workflow for blacklisted person at door."""
        # Add blacklisted person
        import numpy as np
        test_encoding = np.random.rand(128).astype(np.float64)
        face_manager.add_blacklist_face("unwanted_person", test_encoding)
        
        # Mock face recognition to return blacklisted person
        with patch.object(doorbell_system.face_manager, 'identify_face') as mock_identify:
            mock_identify.return_value = {
                "status": "blacklisted",
                "name": "unwanted_person",
                "confidence": 0.90,
                "alert": True,
                "bounding_box": (100, 150, 200, 250)
            }
            
            # Setup camera
            mock_camera.capture_image.return_value = sample_face_image
            
            # Trigger doorbell press
            doorbell_system.on_doorbell_pressed(channel=18)
            
            # Wait for processing
            time.sleep(1)
            
            # Verify security alert sent
            mock_telegram.send_message.assert_called()
            alert_message = str(mock_telegram.send_message.call_args)
            assert "SECURITY ALERT" in alert_message
            assert "unwanted_person" in alert_message
    
    def test_doorbell_debouncing(self, doorbell_system, mock_camera, mock_telegram, sample_face_image):
        """Test doorbell press debouncing mechanism."""
        mock_camera.capture_image.return_value = sample_face_image
        
        # Trigger multiple rapid presses
        for _ in range(5):
            doorbell_system.on_doorbell_pressed(channel=18)
            time.sleep(0.1)  # 100ms between presses
        
        # Wait for processing
        time.sleep(2)
        
        # Should only process once due to debouncing
        assert mock_camera.capture_image.call_count <= 2  # Allow for some timing variation
    
    def test_concurrent_doorbell_presses(self, doorbell_system, mock_camera, mock_telegram, sample_face_image):
        """Test handling of concurrent doorbell presses."""
        mock_camera.capture_image.return_value = sample_face_image
        
        # Create multiple threads to simulate concurrent presses
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=doorbell_system.on_doorbell_pressed, args=(18,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Wait for processing
        time.sleep(2)
        
        # Should handle gracefully without crashes
        assert mock_camera.capture_image.call_count >= 1
```

## 🚀 Performance Tests

### Face Recognition Performance

```python
# tests/performance/test_face_recognition_performance.py
import pytest
import time
import numpy as np
from pathlib import Path
import statistics

from src.face_manager import FaceManager


class TestFaceRecognitionPerformance:
    """Performance tests for face recognition system."""
    
    @pytest.fixture
    def large_face_database(self, face_manager, temp_data_dir):
        """Create large face database for performance testing."""
        face_manager.data_dir = temp_data_dir
        
        # Add 100 known faces
        for i in range(100):
            encoding = np.random.rand(128).astype(np.float64)
            face_manager.add_known_face(f"person_{i:03d}", encoding)
        
        # Add 20 blacklisted faces
        for i in range(20):
            encoding = np.random.rand(128).astype(np.float64)
            face_manager.add_blacklist_face(f"blacklist_{i:03d}", encoding)
        
        return face_manager
    
    def test_face_recognition_speed(self, large_face_database, sample_face_image, benchmark):
        """Benchmark face recognition speed."""
        def recognize_face():
            return large_face_database.identify_face(sample_face_image)
        
        result = benchmark(recognize_face)
        
        # Ensure recognition completes in reasonable time (< 2 seconds)
        assert benchmark.stats.mean < 2.0
        assert result is not None
    
    def test_face_loading_performance(self, temp_data_dir, benchmark):
        """Benchmark face database loading speed."""
        # Create face manager with large database
        face_manager = FaceManager({"tolerance": 0.6, "model": "hog"})
        face_manager.data_dir = temp_data_dir
        
        # Add faces to disk
        for i in range(200):
            encoding = np.random.rand(128).astype(np.float64)
            face_manager.save_face_encoding(f"person_{i:03d}", encoding, is_blacklist=(i >= 150))
        
        # Reset memory
        face_manager.known_faces = {}
        face_manager.blacklist_faces = {}
        
        def load_faces():
            face_manager.load_known_faces()
            return len(face_manager.known_faces) + len(face_manager.blacklist_faces)
        
        result = benchmark(load_faces)
        
        # Should load all faces
        assert result == 200
        # Should complete in reasonable time (< 5 seconds)
        assert benchmark.stats.mean < 5.0
    
    def test_memory_usage_with_large_database(self, large_face_database, sample_face_image):
        """Test memory usage with large face database."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform multiple recognitions
        for _ in range(50):
            result = large_face_database.identify_face(sample_face_image)
            assert result is not None
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB)
        assert memory_increase < 100
    
    def test_concurrent_recognition_performance(self, large_face_database, sample_face_image):
        """Test performance of concurrent face recognition."""
        import threading
        import time
        
        results = []
        start_time = time.time()
        
        def recognize_face():
            result = large_face_database.identify_face(sample_face_image)
            results.append(result)
        
        # Create 10 concurrent recognition threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=recognize_face)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # All recognitions should complete
        assert len(results) == 10
        assert all(result is not None for result in results)
        
        # Should complete in reasonable time (< 30 seconds)
        assert total_time < 30
    
    @pytest.mark.slow
    def test_long_running_performance(self, large_face_database, sample_face_image):
        """Test performance degradation over long running time."""
        recognition_times = []
        
        # Perform 100 recognitions and measure time
        for i in range(100):
            start_time = time.time()
            result = large_face_database.identify_face(sample_face_image)
            end_time = time.time()
            
            recognition_times.append(end_time - start_time)
            assert result is not None
            
            # Small delay between recognitions
            time.sleep(0.1)
        
        # Calculate performance statistics
        mean_time = statistics.mean(recognition_times)
        first_10_avg = statistics.mean(recognition_times[:10])
        last_10_avg = statistics.mean(recognition_times[-10:])
        
        # Performance should not degrade significantly over time
        performance_degradation = (last_10_avg - first_10_avg) / first_10_avg
        assert performance_degradation < 0.5  # Less than 50% degradation
        
        # Overall average should be reasonable
        assert mean_time < 2.0
```

## 🔒 Security Tests

### Input Validation Security Tests

```python
# tests/security/test_input_validation.py
import pytest
from unittest.mock import Mock, patch
import numpy as np

from src.face_manager import FaceManager
from src.web_interface import create_app


class TestInputValidationSecurity:
    """Security tests for input validation."""
    
    def test_face_name_sanitization(self, face_manager, temp_data_dir):
        """Test face name sanitization prevents directory traversal."""
        face_manager.data_dir = temp_data_dir
        
        # Test malicious names
        malicious_names = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "test; rm -rf /",
            "test$(whoami)",
            "test`whoami`",
            "test|whoami",
            "<script>alert('xss')</script>",
            "../../../../../../etc/passwd%00",
        ]
        
        test_encoding = np.random.rand(128).astype(np.float64)
        
        for malicious_name in malicious_names:
            with pytest.raises((ValueError, FileNotFoundError)):
                face_manager.save_face_encoding(malicious_name, test_encoding)
    
    def test_image_validation(self, face_manager):
        """Test image input validation."""
        # Test with None
        with pytest.raises((ValueError, TypeError)):
            face_manager.identify_face(None)
        
        # Test with empty array
        with pytest.raises(ValueError):
            face_manager.identify_face(np.array([]))
        
        # Test with wrong dimensions
        with pytest.raises(ValueError):
            face_manager.identify_face(np.random.rand(100, 100))  # 2D instead of 3D
        
        # Test with wrong data type
        with pytest.raises((ValueError, TypeError)):
            face_manager.identify_face("not_an_image")
    
    def test_web_interface_input_validation(self, web_client):
        """Test web interface input validation."""
        # Test invalid file upload
        response = web_client.post('/upload_face', data={
            'name': 'test',
            'file': 'not_a_file'
        })
        assert response.status_code in [400, 422]
        
        # Test empty name
        response = web_client.post('/upload_face', data={
            'name': '',
            'file': (io.BytesIO(b'fake image data'), 'test.jpg')
        })
        assert response.status_code in [400, 422]
        
        # Test malicious filename
        response = web_client.post('/upload_face', data={
            'name': '../../../etc/passwd',
            'file': (io.BytesIO(b'fake image data'), 'test.jpg')
        })
        assert response.status_code in [400, 422]
    
    def test_sql_injection_protection(self, web_client):
        """Test protection against SQL injection (if database is used)."""
        # Test various SQL injection payloads
        sql_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "'; INSERT INTO users VALUES ('admin', 'password'); --",
            "' UNION SELECT * FROM users --",
        ]
        
        for payload in sql_payloads:
            response = web_client.get(f'/search?q={payload}')
            # Should not return server error or expose database info
            assert response.status_code != 500
            assert b"SQL" not in response.data
            assert b"database" not in response.data.lower()
    
    def test_xss_protection(self, web_client):
        """Test protection against Cross-Site Scripting."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<svg onload=alert('xss')>",
        ]
        
        for payload in xss_payloads:
            response = web_client.post('/upload_face', data={
                'name': payload,
                'file': (io.BytesIO(b'fake image data'), 'test.jpg')
            })
            
            # Check that script tags are not reflected in response
            if response.status_code == 200:
                assert b"<script>" not in response.data
                assert b"javascript:" not in response.data
                assert b"onerror=" not in response.data
```

## 🎯 End-to-End Tests

### Full System E2E Tests

```python
# tests/e2e/test_full_workflow.py
import pytest
import time
import subprocess
import requests
from pathlib import Path
import docker

class TestFullSystemE2E:
    """End-to-end tests for the complete system."""
    
    @pytest.fixture(scope="class")
    def docker_compose_system(self):
        """Start system using docker-compose for E2E testing."""
        # Start the system
        subprocess.run(
            ["docker-compose", "-f", "docker-compose-test.yml", "up", "-d"],
            check=True
        )
        
        # Wait for system to be ready
        max_attempts = 30
        for _ in range(max_attempts):
            try:
                response = requests.get("http://localhost:5000/health")
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
        else:
            pytest.fail("System failed to start within timeout")
        
        yield
        
        # Cleanup
        subprocess.run(
            ["docker-compose", "-f", "docker-compose-test.yml", "down", "-v"],
            check=True
        )
    
    def test_web_interface_accessibility(self, docker_compose_system):
        """Test that web interface is accessible."""
        response = requests.get("http://localhost:5000/")
        assert response.status_code == 200
        assert b"Doorbell Security System" in response.data
    
    def test_api_endpoints(self, docker_compose_system):
        """Test API endpoints functionality."""
        # Test health endpoint
        response = requests.get("http://localhost:5000/health")
        assert response.status_code == 200
        
        # Test faces endpoint
        response = requests.get("http://localhost:5000/api/faces")
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/json"
    
    def test_face_upload_workflow(self, docker_compose_system):
        """Test complete face upload workflow."""
        # Create test image
        test_image = create_test_face_image()
        
        # Upload face
        files = {"file": ("test_face.jpg", test_image, "image/jpeg")}
        data = {"name": "test_person"}
        
        response = requests.post(
            "http://localhost:5000/api/upload_face",
            files=files,
            data=data
        )
        
        assert response.status_code == 200
        
        # Verify face was added
        response = requests.get("http://localhost:5000/api/faces")
        faces = response.json()
        assert any(face["name"] == "test_person" for face in faces)
    
    def test_system_resource_usage(self, docker_compose_system):
        """Test system resource usage under load."""
        client = docker.from_env()
        container = client.containers.get("doorbell-system")
        
        # Get initial stats
        initial_stats = container.stats(stream=False)
        
        # Generate some load
        for _ in range(10):
            requests.get("http://localhost:5000/")
            time.sleep(0.1)
        
        # Get final stats
        final_stats = container.stats(stream=False)
        
        # Check that memory usage is reasonable
        memory_usage = final_stats["memory_stats"]["usage"]
        memory_limit = final_stats["memory_stats"]["limit"]
        memory_percent = (memory_usage / memory_limit) * 100
        
        assert memory_percent < 80  # Should use less than 80% of available memory
```

## 🚀 Running Tests

### Command Reference

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only
pytest -m performance           # Performance tests only
pytest -m security              # Security tests only
pytest -m e2e                   # End-to-end tests only

# Run tests with coverage
pytest --cov=src --cov=config --cov-report=html

# Run tests in parallel
pytest -n auto                  # Use all CPU cores
pytest -n 4                     # Use 4 cores

# Run performance benchmarks
pytest --benchmark-only         # Only benchmark tests
pytest --benchmark-sort=mean    # Sort by mean time

# Run tests on specific platforms
pytest -m "not hardware"        # Skip hardware-dependent tests
pytest -m "not slow"           # Skip slow tests
pytest -m "pi"                 # Raspberry Pi specific tests

# Run with verbose output
pytest -v -s                   # Verbose with print statements

# Run specific test file
pytest tests/unit/test_face_manager.py

# Run specific test function
pytest tests/unit/test_face_manager.py::TestFaceManager::test_identify_face

# Run tests matching pattern
pytest -k "face_recognition"   # Run tests with "face_recognition" in name

# Generate test report
pytest --html=report.html --self-contained-html
```

### Continuous Integration

```yaml
# .github/workflows/test.yml (excerpt)
- name: Run Test Suite
  run: |
    # Unit tests
    pytest tests/unit/ -v --cov=src --cov=config
    
    # Integration tests
    pytest tests/integration/ -v
    
    # Security tests
    pytest tests/security/ -v
    
    # Performance tests (on main branch only)
    if [ "$GITHUB_REF" = "refs/heads/main" ]; then
      pytest tests/performance/ -v --benchmark-json=benchmark.json
    fi

- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
    fail_ci_if_error: true
```

## 📊 Test Coverage Goals

### Coverage Targets

- **Overall Coverage**: ≥ 90%
- **Critical Components**: ≥ 95%
  - `face_manager.py`
  - `doorbell_security.py`
  - `camera_handler.py`
- **Security Components**: 100%
  - Input validation
  - Authentication
  - Data protection
- **Platform-Specific Code**: ≥ 80%
  - GPIO handling
  - Camera interfaces

### Coverage Reports

```bash
# Generate detailed coverage report
pytest --cov=src --cov=config \
       --cov-report=html:htmlcov \
       --cov-report=xml:coverage.xml \
       --cov-report=term-missing

# View HTML report
open htmlcov/index.html

# Check coverage thresholds
pytest --cov=src --cov-fail-under=90
```

---

This comprehensive testing guide ensures that the Doorbell Security System maintains high quality, security, and performance standards through automated testing at all levels of the application stack.
## 🛠️ Test Utilities and Scripts

### Unified Test Runner

The `scripts/testing/run_tests.py` script provides a unified interface for running all test types:

```bash
# Run all tests
python scripts/testing/run_tests.py --all

# Run specific test suites
python scripts/testing/run_tests.py --unit --coverage
python scripts/testing/run_tests.py --integration --verbose
python scripts/testing/run_tests.py --e2e
python scripts/testing/run_tests.py --performance --benchmark
python scripts/testing/run_tests.py --security
python scripts/testing/run_tests.py --load --users 100 --runtime 120

# Quick mode (skip slow tests)
python scripts/testing/run_tests.py --all --quick
```

**Features**:
- Runs all test types (unit, integration, E2E, performance, security, load)
- Generates comprehensive coverage reports
- Provides detailed test summaries
- Supports verbose output for debugging

### Coverage Report Generator

The `scripts/testing/generate_coverage_report.py` script provides enhanced coverage analysis:

```bash
# Generate coverage report and run tests
python scripts/testing/generate_coverage_report.py

# Check if coverage meets threshold
python scripts/testing/generate_coverage_report.py --fail-under 80

# Generate markdown report
python scripts/testing/generate_coverage_report.py --markdown

# Analyze existing coverage without running tests
python scripts/testing/generate_coverage_report.py --no-run --markdown
```

**Features**:
- Multiple output formats (text, markdown, JSON)
- Coverage threshold checking
- Package-level coverage breakdown
- Coverage badges for documentation

### Property-Based Testing Utilities

The `tests/utils/property_based_tests.py` module provides custom Hypothesis strategies:

```python
from tests.utils.property_based_tests import (
    valid_image_array,
    face_encoding_vector,
    detection_confidence,
    bounding_box
)

@given(valid_image_array())
def test_image_processing(image):
    # Test will run with many different valid images
    result = process_image(image)
    assert result is not None
```

**Available Strategies**:
- `valid_image_array()`: Generate valid image arrays
- `face_encoding_vector()`: Generate 128D face encoding vectors
- `detection_confidence()`: Generate confidence values (0.0-1.0)
- `recognition_threshold()`: Generate threshold values (0.1-0.9)
- `bounding_box()`: Generate valid bounding boxes

## 🔄 CI/CD Integration

### Automated Testing Workflow

The comprehensive test suite runs automatically on:
- **Pull Requests**: To main/develop branches
- **Pushes**: To main/develop branches
- **Schedule**: Nightly at 2 AM UTC
- **Manual**: Via workflow_dispatch

### Workflow Jobs

1. **Code Quality** (`code-quality`)
   - Black formatting check
   - Ruff linting
   - isort import sorting

2. **Unit Tests** (`unit-tests`)
   - Python 3.10, 3.11, 3.12
   - Coverage reporting

3. **Integration Tests** (`integration-tests`)
   - Component interaction testing
   - Database integration

4. **E2E Tests** (`e2e-tests`)
   - Browser automation with Playwright
   - Complete workflow testing

5. **Performance Tests** (`performance-tests`)
   - Benchmark execution
   - Performance regression detection
   - Result comparison

6. **Load Tests** (`load-tests`)
   - Locust load testing
   - 50 users, 60-second runtime
   - HTML and CSV reports

7. **Security Tests** (`security-tests`)
   - Input validation tests
   - Bandit security scan
   - Safety dependency check

8. **Coverage Report** (`coverage-report`)
   - Comprehensive coverage analysis
   - Markdown report generation
   - Threshold checking (80%)

9. **Quality Gates** (`quality-gates`)
   - Aggregates all test results
   - Provides comprehensive summary
   - Fails if critical tests fail

### Test Artifacts

All test runs produce artifacts stored for 30 days:
- Coverage reports (HTML, XML, JSON, Markdown)
- Test results (JUnit XML, HTML)
- Security reports (Bandit JSON, Safety JSON)
- Performance benchmarks (JSON)
- Load test reports (HTML, CSV)

### PR Integration

Pull requests automatically receive:
- Coverage summary in PR comments
- Test result summary
- Links to detailed reports
- Security scan results

## 📚 Best Practices

### Writing Tests

1. **Use descriptive names**: Test names should describe what they verify
   ```python
   # Good
   def test_face_recognition_rejects_low_confidence_matches(self):
       # Test implementation here
       pass
       
   # Bad
   def test_face_rec(self):
       pass
   ```

2. **Follow AAA pattern**: Arrange, Act, Assert
   ```python
   def test_face_encoding_generation(self):
       # Arrange
       image = load_test_image("john_doe.jpg")
       
       # Act
       encoding = face_manager.generate_encoding(image)
       
       # Assert
       assert encoding is not None
       assert len(encoding) == 128
   ```

3. **Use fixtures for setup**: Avoid code duplication
   ```python
   @pytest.fixture
   def sample_face_image(self):
       return load_test_image("sample_face.jpg")
   
   def test_detection(self, sample_face_image):
       result = detector.detect_faces(sample_face_image)
       assert len(result) > 0
   ```

4. **Mock external dependencies**: Keep tests isolated
   ```python
   @patch('src.telegram_notifier.TelegramBot')
   def test_notification_sending(self, mock_telegram):
       mock_telegram.send_message.return_value = True
       
       result = notifier.send_alert("Test")
       assert result is True
   ```

5. **Test edge cases**: Include boundary conditions
   ```python
   @pytest.mark.parametrize("confidence", [0.0, 0.5, 0.99, 1.0])
   def test_confidence_thresholds(self, confidence):
       result = is_confident_match(confidence, threshold=0.6)
       expected = confidence >= 0.6
       assert result == expected
   ```

### Test Organization

1. **Group related tests**: Use test classes
   ```python
   class TestFaceRecognition:
       def test_encoding_generation(self):
           pass
       
       def test_face_matching(self):
           pass
   ```

2. **Use markers appropriately**: Tag tests for selective execution
   ```python
   @pytest.mark.slow
   @pytest.mark.integration
   def test_full_pipeline(self):
       pass
   ```

3. **Keep tests independent**: Tests should not depend on each other
   ```python
   # Good - each test is independent
   def test_add_face(self):
       face_manager.add_face("john", encoding)
       assert "john" in face_manager.known_faces
   
   # Bad - depends on previous test
   def test_remove_face(self):
       # Assumes john was added in previous test
       face_manager.remove_face("john")
       assert "john" not in face_manager.known_faces
   ```

### Performance Considerations

1. **Use session-scoped fixtures** for expensive setup
   ```python
   @pytest.fixture(scope="session")
   def trained_model(self):
       return load_expensive_model()
   ```

2. **Run slow tests selectively**
   ```bash
   pytest -m "not slow"  # Skip slow tests in development
   pytest -m slow  # Run only slow tests before commit
   ```

3. **Parallelize when possible**
   ```bash
   pytest tests/ -n auto  # Use all available cores
   ```

### Debugging Tests

1. **Use pytest verbosity**:
   ```bash
   pytest tests/test_face_manager.py -v  # Verbose
   pytest tests/test_face_manager.py -vv  # Very verbose
   ```

2. **Show print statements**:
   ```bash
   pytest tests/test_face_manager.py -s
   ```

3. **Drop into debugger on failure**:
   ```bash
   pytest tests/test_face_manager.py --pdb
   ```

4. **Run specific tests**:
   ```bash
   pytest tests/test_face_manager.py::TestFaceManager::test_encoding
   ```

## 🔗 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Playwright Python Documentation](https://playwright.dev/python/)
- [Locust Documentation](https://docs.locust.io/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Test Scripts README](../scripts/testing/README.md)

## 📝 Summary

This comprehensive testing framework ensures:
- ✅ High code quality through multiple testing layers
- ✅ Fast feedback with unit tests
- ✅ Confidence in integrations with integration tests
- ✅ Validated user workflows with E2E tests
- ✅ Performance tracking with benchmarks
- ✅ Security validation with vulnerability scanning
- ✅ Scalability verification with load testing
- ✅ Automated quality gates in CI/CD

For questions or issues with testing, please refer to the [Contributing Guide](CONTRIBUTING.md) or open an issue on GitHub.
