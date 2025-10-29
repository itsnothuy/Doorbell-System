# Comprehensive Testing Framework and CI/CD Automation

## Issue Summary

**Priority**: Critical  
**Type**: Infrastructure, Quality Assurance  
**Component**: Testing Framework, CI/CD Pipeline  
**Estimated Effort**: 60-80 hours  
**Dependencies**: All core components, Pipeline infrastructure  

## Overview

Implement a comprehensive testing framework covering unit tests, integration tests, end-to-end tests, and automated CI/CD workflows. This issue establishes production-grade quality assurance processes to ensure system reliability, security, and performance across all components of the Doorbell Security System.

## Current Testing State Analysis

### Existing Test Structure
```
tests/
├── conftest.py                    # Pytest configuration and fixtures
├── test_camera_handler.py         # Basic camera functionality tests
├── test_face_manager.py           # Face recognition unit tests
├── test_telegram_notifier.py      # Notification system tests
├── integration/                   # Integration test placeholders
├── e2e/                          # End-to-end test placeholders
└── fixtures/                     # Test data and mock objects
```

### Current Gaps
1. **Coverage**: Limited test coverage (~40% estimated)
2. **Integration Testing**: Minimal cross-component testing
3. **Hardware Testing**: No mock hardware abstraction testing
4. **Performance Testing**: No performance regression testing
5. **Security Testing**: No security vulnerability testing
6. **CI/CD**: Basic GitHub Actions without comprehensive test matrix
7. **E2E Testing**: No browser automation or user journey testing

## Technical Specifications

### Testing Architecture Overview

```mermaid
graph TB
    A[Unit Tests] --> D[Test Runner]
    B[Integration Tests] --> D
    C[E2E Tests] --> D
    D --> E[Coverage Report]
    D --> F[Performance Metrics]
    D --> G[Security Scan]
    E --> H[Quality Gate]
    F --> H
    G --> H
    H --> I[CI/CD Pipeline]
    I --> J[Deployment]
```

### Test Framework Stack

#### Core Testing Dependencies
```toml
# pyproject.toml testing dependencies
[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --strict-markers --strict-config"
testpaths = ["tests"]
markers = [
    "unit: Unit tests for isolated components",
    "integration: Integration tests for component interaction", 
    "e2e: End-to-end tests for complete user journeys",
    "performance: Performance regression tests",
    "security: Security vulnerability tests",
    "hardware: Hardware-specific tests (requires actual hardware)",
    "slow: Tests that take more than 5 seconds",
    "gpu: Tests requiring GPU hardware",
    "pi: Tests requiring Raspberry Pi hardware"
]

[project.optional-dependencies]
test = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-xdist>=3.3.1",           # Parallel test execution
    "pytest-mock>=3.11.1",           # Mocking utilities
    "pytest-asyncio>=0.21.1",        # Async test support
    "pytest-benchmark>=4.0.0",       # Performance benchmarking
    "pytest-html>=3.2.0",            # HTML test reports
    "pytest-timeout>=2.1.0",         # Test timeout handling
    "pytest-randomly>=3.12.0",       # Random test ordering
    "Faker>=19.3.0",                 # Test data generation
    "factory-boy>=3.3.0",            # Test object factories
    "responses>=0.23.3",             # HTTP request mocking
    "freezegun>=1.2.2",              # Time mocking
    "hypothesis>=6.82.0",            # Property-based testing
]

e2e = [
    "playwright>=1.36.0",            # Browser automation
    "selenium>=4.11.0",              # Alternative browser automation
    "requests>=2.31.0",              # HTTP client testing
]

performance = [
    "locust>=2.16.1",                # Load testing
    "memory-profiler>=0.61.0",       # Memory profiling  
    "py-spy>=0.3.14",                # Python profiling
    "psutil>=5.9.5",                 # System monitoring
]

security = [
    "bandit>=1.7.5",                 # Security linting
    "safety>=2.3.4",                 # Dependency vulnerability scanning
    "semgrep>=1.34.0",               # Static security analysis
]
```

### Unit Testing Framework

#### Component Test Structure
```python
# tests/unit/test_face_recognition_engine.py
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st

from src.recognition.face_recognizer import FaceRecognizer
from src.recognition.face_encoder import FaceEncoder
from src.recognition.recognition_cache import RecognitionCache
from src.detectors.cpu_detector import CPUDetector


class TestFaceRecognizer:
    """Comprehensive unit tests for face recognition engine."""
    
    @pytest.fixture
    def mock_detector(self):
        """Create mock face detector."""
        detector = Mock(spec=CPUDetector)
        detector.detect_faces.return_value = ([
            Mock(
                box=(100, 100, 200, 200),
                confidence=0.95,
                landmarks=np.random.rand(68, 2)
            )
        ], Mock(inference_time=0.1))
        return detector
    
    @pytest.fixture
    def mock_encoder(self):
        """Create mock face encoder."""
        encoder = Mock(spec=FaceEncoder)
        encoder.encode_face.return_value = np.random.rand(128)
        return encoder
    
    @pytest.fixture
    def mock_cache(self):
        """Create mock recognition cache."""
        cache = Mock(spec=RecognitionCache)
        cache.get.return_value = None
        cache.set.return_value = None
        return cache
    
    @pytest.fixture
    def face_recognizer(self, mock_detector, mock_encoder, mock_cache):
        """Create face recognizer instance with mocked dependencies."""
        config = {
            'recognition_threshold': 0.6,
            'max_distance': 0.4,
            'enable_cache': True
        }
        
        recognizer = FaceRecognizer(config)
        recognizer._detector = mock_detector
        recognizer._encoder = mock_encoder
        recognizer._cache = mock_cache
        
        return recognizer
    
    def test_recognize_face_success(self, face_recognizer):
        """Test successful face recognition."""
        # Arrange
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        known_encoding = np.random.rand(128)
        face_recognizer._known_faces = {'john_doe': known_encoding}
        
        # Mock encoder to return similar encoding
        face_recognizer._encoder.encode_face.return_value = known_encoding + 0.1
        
        # Act
        result = face_recognizer.recognize_face(test_image)
        
        # Assert
        assert result is not None
        assert result.person_name == 'john_doe'
        assert result.confidence > 0.6
        assert result.recognition_time > 0
        
        # Verify method calls
        face_recognizer._detector.detect_faces.assert_called_once()
        face_recognizer._encoder.encode_face.assert_called_once()
    
    def test_recognize_face_no_detection(self, face_recognizer):
        """Test face recognition when no face is detected."""
        # Arrange
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        face_recognizer._detector.detect_faces.return_value = ([], Mock())
        
        # Act
        result = face_recognizer.recognize_face(test_image)
        
        # Assert
        assert result.person_name == 'unknown'
        assert result.confidence == 0.0
    
    @given(
        image_height=st.integers(min_value=100, max_value=1080),
        image_width=st.integers(min_value=100, max_value=1920),
        threshold=st.floats(min_value=0.1, max_value=0.9)
    )
    def test_recognize_face_property_based(self, face_recognizer, image_height, image_width, threshold):
        """Property-based testing for face recognition."""
        # Arrange
        face_recognizer.config['recognition_threshold'] = threshold
        test_image = np.random.randint(0, 255, (image_height, image_width, 3), dtype=np.uint8)
        
        # Act
        result = face_recognizer.recognize_face(test_image)
        
        # Assert - Properties that should always hold
        assert hasattr(result, 'person_name')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'recognition_time')
        assert 0.0 <= result.confidence <= 1.0
        assert result.recognition_time >= 0
    
    @pytest.mark.performance
    def test_recognition_performance(self, face_recognizer, benchmark):
        """Performance benchmark for face recognition."""
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = benchmark(face_recognizer.recognize_face, test_image)
        
        # Performance assertions
        assert result.recognition_time < 1.0  # Should complete within 1 second
    
    @pytest.mark.timeout(5)
    def test_recognition_timeout(self, face_recognizer):
        """Test that recognition doesn't hang indefinitely."""
        # Simulate slow detection
        face_recognizer._detector.detect_faces.side_effect = lambda x: time.sleep(10)
        
        with pytest.raises(TimeoutError):
            face_recognizer.recognize_face(np.zeros((480, 640, 3)))
    
    def test_memory_usage(self, face_recognizer):
        """Test memory usage during face recognition."""
        import tracemalloc
        
        tracemalloc.start()
        
        # Process multiple images
        for _ in range(10):
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            face_recognizer.recognize_face(test_image)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory assertions (adjust based on requirements)
        assert peak < 100 * 1024 * 1024  # Less than 100MB peak usage
```

#### Hardware Abstraction Testing
```python
# tests/unit/hardware/test_camera_handler.py
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.camera_handler import CameraHandler
from src.hardware.platform.raspberry_pi import RaspberryPiCamera
from src.hardware.platform.macos import MacOSCamera


class TestCameraHandler:
    """Test camera handler with hardware abstraction."""
    
    @pytest.fixture
    def mock_platform_camera(self):
        """Create mock platform-specific camera."""
        camera = Mock()
        camera.initialize.return_value = True
        camera.capture_frame.return_value = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        camera.get_camera_info.return_value = Mock(
            resolution=(640, 480),
            fps=30,
            format='RGB'
        )
        return camera
    
    @patch('src.camera_handler.platform_detector')
    def test_platform_detection(self, mock_platform_detector, mock_platform_camera):
        """Test camera handler selects correct platform implementation."""
        # Test Raspberry Pi detection
        mock_platform_detector.is_raspberry_pi.return_value = True
        mock_platform_detector.is_macos.return_value = False
        
        with patch('src.hardware.platform.raspberry_pi.RaspberryPiCamera', return_value=mock_platform_camera):
            handler = CameraHandler()
            assert isinstance(handler._camera, type(mock_platform_camera))
        
        # Test macOS detection
        mock_platform_detector.is_raspberry_pi.return_value = False
        mock_platform_detector.is_macos.return_value = True
        
        with patch('src.hardware.platform.macos.MacOSCamera', return_value=mock_platform_camera):
            handler = CameraHandler()
            assert isinstance(handler._camera, type(mock_platform_camera))
    
    @pytest.mark.pi
    def test_raspberry_pi_camera_integration(self):
        """Integration test for Raspberry Pi camera (requires hardware)."""
        pytest.skip("Requires Raspberry Pi hardware")
        
        # This test would run on actual Pi hardware
        camera = RaspberryPiCamera({'resolution': (640, 480)})
        
        assert camera.initialize() is True
        frame = camera.capture_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)
```

### Integration Testing Framework

#### Component Integration Tests
```python
# tests/integration/test_detection_pipeline.py
import pytest
import asyncio
import threading
import time
from unittest.mock import patch, Mock

from src.pipeline.orchestrator import PipelineOrchestrator
from src.communication.message_bus import MessageBus
from src.detectors.cpu_detector import CPUDetector
from src.recognition.face_recognizer import FaceRecognizer


class TestDetectionPipeline:
    """Integration tests for the complete detection pipeline."""
    
    @pytest.fixture
    async def pipeline_setup(self):
        """Set up pipeline with all components."""
        config = {
            'frame_capture': {
                'fps': 10,
                'resolution': (640, 480)
            },
            'face_detection': {
                'detector_type': 'cpu',
                'confidence_threshold': 0.5
            },
            'face_recognition': {
                'recognition_threshold': 0.6,
                'enable_cache': True
            }
        }
        
        message_bus = MessageBus()
        orchestrator = PipelineOrchestrator(message_bus, config)
        
        yield {
            'orchestrator': orchestrator,
            'message_bus': message_bus,
            'config': config
        }
        
        # Cleanup
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_end_to_end_face_detection(self, pipeline_setup):
        """Test complete face detection pipeline flow."""
        orchestrator = pipeline_setup['orchestrator']
        message_bus = pipeline_setup['message_bus']
        
        # Mock camera input
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Start pipeline
        await orchestrator.start()
        
        # Inject test image
        await message_bus.publish('frame_captured', {
            'image': test_image,
            'timestamp': time.time()
        })
        
        # Wait for processing
        result = await message_bus.subscribe('face_recognition_complete', timeout=5.0)
        
        # Verify result
        assert result is not None
        assert 'detections' in result
        assert 'processing_time' in result
        assert result['processing_time'] < 2.0  # Should process within 2 seconds
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, pipeline_setup):
        """Test pipeline handles errors gracefully."""
        orchestrator = pipeline_setup['orchestrator']
        message_bus = pipeline_setup['message_bus']
        
        # Start pipeline
        await orchestrator.start()
        
        # Inject invalid image
        await message_bus.publish('frame_captured', {
            'image': None,  # Invalid image
            'timestamp': time.time()
        })
        
        # Should receive error event
        error_event = await message_bus.subscribe('pipeline_error', timeout=2.0)
        
        assert error_event is not None
        assert 'error_type' in error_event
        assert 'component' in error_event
    
    @pytest.mark.performance
    async def test_pipeline_throughput(self, pipeline_setup):
        """Test pipeline can handle expected throughput."""
        orchestrator = pipeline_setup['orchestrator']
        message_bus = pipeline_setup['message_bus']
        
        await orchestrator.start()
        
        # Send 100 frames
        start_time = time.time()
        for i in range(100):
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            await message_bus.publish('frame_captured', {
                'image': test_image,
                'timestamp': time.time()
            })
        
        # Wait for all processing to complete
        processed_count = 0
        while processed_count < 100:
            result = await message_bus.subscribe('face_recognition_complete', timeout=1.0)
            if result:
                processed_count += 1
        
        total_time = time.time() - start_time
        fps = 100 / total_time
        
        # Should maintain at least 5 FPS processing
        assert fps >= 5.0
```

#### Database Integration Tests
```python
# tests/integration/test_storage_integration.py
import pytest
import tempfile
import os
from datetime import datetime, timedelta

from src.storage.event_database import EventDatabase
from src.storage.face_database import FaceDatabase


class TestStorageIntegration:
    """Integration tests for storage layer."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        yield db_path
        
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.fixture
    def event_db(self, temp_db_path):
        """Create event database instance."""
        db = EventDatabase(temp_db_path)
        db.initialize()
        return db
    
    @pytest.fixture
    def face_db(self, temp_db_path):
        """Create face database instance."""
        db = FaceDatabase(temp_db_path)
        db.initialize()
        return db
    
    def test_event_storage_lifecycle(self, event_db):
        """Test complete event storage lifecycle."""
        # Create event
        event_data = {
            'event_type': 'face_detected',
            'person_name': 'john_doe',
            'confidence': 0.95,
            'image_path': '/path/to/image.jpg',
            'timestamp': datetime.now()
        }
        
        # Store event
        event_id = event_db.store_event(event_data)
        assert event_id is not None
        
        # Retrieve event
        retrieved_event = event_db.get_event(event_id)
        assert retrieved_event is not None
        assert retrieved_event['person_name'] == 'john_doe'
        assert retrieved_event['confidence'] == 0.95
        
        # Update event
        event_db.update_event(event_id, {'verified': True})
        updated_event = event_db.get_event(event_id)
        assert updated_event['verified'] is True
        
        # Delete event
        event_db.delete_event(event_id)
        deleted_event = event_db.get_event(event_id)
        assert deleted_event is None
    
    def test_face_encoding_management(self, face_db):
        """Test face encoding storage and retrieval."""
        # Store face encoding
        encoding = np.random.rand(128)
        face_id = face_db.store_face_encoding('jane_doe', encoding)
        
        # Retrieve encoding
        retrieved_encoding = face_db.get_face_encoding('jane_doe')
        assert np.allclose(encoding, retrieved_encoding)
        
        # List all faces
        all_faces = face_db.list_known_faces()
        assert 'jane_doe' in all_faces
        
        # Delete face
        face_db.delete_face('jane_doe')
        deleted_encoding = face_db.get_face_encoding('jane_doe')
        assert deleted_encoding is None
    
    @pytest.mark.performance
    def test_database_performance(self, event_db):
        """Test database performance with large dataset."""
        import time
        
        # Insert 1000 events
        start_time = time.time()
        for i in range(1000):
            event_data = {
                'event_type': 'face_detected',
                'person_name': f'person_{i}',
                'confidence': 0.8 + (i % 20) * 0.01,
                'timestamp': datetime.now() - timedelta(minutes=i)
            }
            event_db.store_event(event_data)
        
        insert_time = time.time() - start_time
        
        # Query events
        start_time = time.time()
        recent_events = event_db.get_recent_events(limit=100)
        query_time = time.time() - start_time
        
        # Performance assertions
        assert insert_time < 10.0  # Should insert 1000 events in under 10 seconds
        assert query_time < 1.0    # Should query 100 events in under 1 second
        assert len(recent_events) == 100
```

### End-to-End Testing Framework

#### Browser Automation Tests
```python
# tests/e2e/test_web_interface.py
import pytest
from playwright.async_api import async_playwright
import asyncio


class TestWebInterface:
    """End-to-end tests for web interface using Playwright."""
    
    @pytest.fixture
    async def browser_context(self):
        """Set up browser context for testing."""
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 720},
            user_agent='Mozilla/5.0 (Playwright Test)'
        )
        
        yield context
        
        await context.close()
        await browser.close()
        await playwright.stop()
    
    @pytest.fixture
    async def page(self, browser_context):
        """Create new page for testing."""
        page = await browser_context.new_page()
        yield page
        await page.close()
    
    @pytest.mark.e2e
    async def test_dashboard_loading(self, page):
        """Test dashboard loads correctly."""
        await page.goto('http://localhost:5000/')
        
        # Wait for page to load
        await page.wait_for_selector('[data-testid="dashboard"]')
        
        # Verify key elements are present
        await expect(page.locator('h1')).to_contain_text('Security Dashboard')
        await expect(page.locator('[data-testid="status-card"]')).to_be_visible()
        await expect(page.locator('[data-testid="camera-feed"]')).to_be_visible()
        await expect(page.locator('[data-testid="recent-events"]')).to_be_visible()
    
    @pytest.mark.e2e
    async def test_camera_stream_functionality(self, page):
        """Test camera stream starts and displays correctly."""
        await page.goto('http://localhost:5000/camera')
        
        # Wait for camera to initialize
        await page.wait_for_selector('[data-testid="camera-video"]')
        
        # Verify video element has video stream
        video_element = page.locator('[data-testid="camera-video"]')
        await expect(video_element).to_be_visible()
        
        # Test camera controls
        await page.click('[data-testid="take-snapshot"]')
        await page.wait_for_selector('[data-testid="snapshot-success"]')
        
        # Verify snapshot was taken
        await expect(page.locator('[data-testid="snapshot-success"]')).to_be_visible()
    
    @pytest.mark.e2e
    async def test_face_management_workflow(self, page):
        """Test complete face management user journey."""
        await page.goto('http://localhost:5000/known-faces')
        
        # Add new known face
        await page.click('[data-testid="add-face-button"]')
        
        # Fill form
        await page.fill('[data-testid="person-name-input"]', 'Test Person')
        await page.set_input_files('[data-testid="face-image-input"]', 'tests/fixtures/test_face.jpg')
        
        # Submit form
        await page.click('[data-testid="save-face-button"]')
        
        # Verify face was added
        await page.wait_for_selector('[data-testid="face-card-test-person"]')
        await expect(page.locator('[data-testid="face-card-test-person"]')).to_be_visible()
        
        # Delete face
        await page.click('[data-testid="delete-face-test-person"]')
        await page.click('[data-testid="confirm-delete"]')
        
        # Verify face was deleted
        await expect(page.locator('[data-testid="face-card-test-person"]')).not_to_be_visible()
    
    @pytest.mark.e2e
    async def test_real_time_notifications(self, page):
        """Test real-time notification system."""
        await page.goto('http://localhost:5000/')
        
        # Simulate face detection event (would need mock WebSocket)
        await page.evaluate("""
            window.mockWebSocket.send({
                type: 'face_detected',
                payload: {
                    person_name: 'John Doe',
                    confidence: 0.95,
                    timestamp: new Date().toISOString()
                }
            });
        """)
        
        # Verify notification appears
        await page.wait_for_selector('[data-testid="notification"]')
        await expect(page.locator('[data-testid="notification"]')).to_contain_text('Face detected: John Doe')
        
        # Verify notification auto-dismisses
        await page.wait_for_timeout(5000)
        await expect(page.locator('[data-testid="notification"]')).not_to_be_visible()
    
    @pytest.mark.e2e
    async def test_mobile_responsiveness(self, page):
        """Test mobile responsiveness."""
        # Set mobile viewport
        await page.set_viewport_size({'width': 375, 'height': 667})
        await page.goto('http://localhost:5000/')
        
        # Verify mobile navigation
        await expect(page.locator('[data-testid="mobile-menu-button"]')).to_be_visible()
        await page.click('[data-testid="mobile-menu-button"]')
        await expect(page.locator('[data-testid="mobile-nav"]')).to_be_visible()
        
        # Test navigation on mobile
        await page.click('[data-testid="mobile-nav-camera"]')
        await expect(page).to_have_url('http://localhost:5000/camera')
    
    @pytest.mark.e2e  
    async def test_accessibility_compliance(self, page):
        """Test accessibility compliance."""
        await page.goto('http://localhost:5000/')
        
        # Run accessibility scan (using axe-core)
        await page.add_script_tag(url='https://unpkg.com/axe-core@4.7.2/axe.min.js')
        
        accessibility_results = await page.evaluate("""
            axe.run().then(results => results)
        """)
        
        # Verify no critical accessibility violations
        violations = accessibility_results.get('violations', [])
        critical_violations = [v for v in violations if v['impact'] in ['critical', 'serious']]
        
        assert len(critical_violations) == 0, f"Critical accessibility violations found: {critical_violations}"
```

### Performance Testing Framework

#### Load Testing with Locust
```python
# tests/performance/test_load_testing.py
from locust import HttpUser, task, between
import json
import random


class WebInterfaceUser(HttpUser):
    """Simulate user behavior for load testing."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Set up user session."""
        self.client.verify = False  # For testing with self-signed certs
    
    @task(3)
    def view_dashboard(self):
        """User views dashboard (most common action)."""
        self.client.get("/")
    
    @task(2)
    def view_camera_feed(self):
        """User views camera feed."""
        self.client.get("/camera")
    
    @task(1)
    def view_events(self):
        """User views event history."""
        self.client.get("/events")
        
        # Also test event API
        self.client.get("/api/events?limit=50")
    
    @task(1)
    def api_system_status(self):
        """User checks system status via API."""
        response = self.client.get("/api/system/status")
        
        if response.status_code == 200:
            data = response.json()
            assert 'camera_status' in data
            assert 'detection_status' in data
    
    @task(1)
    def manage_known_faces(self):
        """User manages known faces."""
        # Get known faces
        self.client.get("/known-faces")
        self.client.get("/api/faces")
    
    def test_websocket_connection(self):
        """Test WebSocket connection handling."""
        # This would require WebSocket testing framework
        pass


class ApiLoadTest(HttpUser):
    """API-focused load testing."""
    
    wait_time = between(0.5, 1.5)
    
    @task(5)
    def get_system_metrics(self):
        """High-frequency metrics polling."""
        self.client.get("/api/system/metrics")
    
    @task(3)
    def get_recent_events(self):
        """Event polling."""
        self.client.get("/api/events/recent")
    
    @task(2)
    def face_recognition_api(self):
        """Simulate face recognition API calls."""
        # Would need test image data
        files = {'image': ('test.jpg', b'fake_image_data', 'image/jpeg')}
        self.client.post("/api/recognition/identify", files=files)
    
    @task(1)
    def configuration_updates(self):
        """Configuration change simulation."""
        config = {
            'detection_threshold': random.uniform(0.5, 0.9),
            'recognition_threshold': random.uniform(0.6, 0.8)
        }
        self.client.post("/api/config/detection", json=config)
```

#### Memory and CPU Profiling
```python
# tests/performance/test_profiling.py
import pytest
import psutil
import time
import memory_profiler
from unittest.mock import patch
import numpy as np

from src.recognition.face_recognizer import FaceRecognizer
from src.detectors.cpu_detector import CPUDetector


class TestPerformanceProfiling:
    """Performance profiling tests."""
    
    @pytest.mark.performance
    def test_memory_usage_face_recognition(self):
        """Profile memory usage during face recognition."""
        
        @memory_profiler.profile
        def run_face_recognition():
            recognizer = FaceRecognizer({'recognition_threshold': 0.6})
            
            # Process 100 images
            for i in range(100):
                test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                result = recognizer.recognize_face(test_image)
            
            return recognizer
        
        # Run profiling
        recognizer = run_face_recognition()
        
        # Memory usage should be reasonable
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        assert memory_mb < 500  # Less than 500MB
    
    @pytest.mark.performance
    def test_cpu_usage_detection(self):
        """Monitor CPU usage during face detection."""
        detector = CPUDetector({'model_type': 'hog'})
        
        # Monitor CPU before
        cpu_before = psutil.cpu_percent(interval=1)
        
        start_time = time.time()
        
        # Run detection for 30 seconds
        end_time = start_time + 30
        detection_count = 0
        
        while time.time() < end_time:
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detector.detect_faces(test_image)
            detection_count += 1
        
        # Monitor CPU after
        cpu_after = psutil.cpu_percent(interval=1)
        
        # Calculate metrics
        total_time = time.time() - start_time
        fps = detection_count / total_time
        
        # Performance assertions
        assert fps >= 5.0  # Should maintain at least 5 FPS
        assert cpu_after < 90  # Should not max out CPU
    
    @pytest.mark.performance
    def test_memory_leak_detection(self):
        """Test for memory leaks in long-running operations."""
        import gc
        
        recognizer = FaceRecognizer({'recognition_threshold': 0.6})
        
        # Baseline memory
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss
        
        # Run many iterations
        for i in range(1000):
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = recognizer.recognize_face(test_image)
            
            # Force garbage collection every 100 iterations
            if i % 100 == 0:
                gc.collect()
        
        # Final memory check
        gc.collect()
        final_memory = psutil.Process().memory_info().rss
        memory_growth = (final_memory - baseline_memory) / 1024 / 1024  # MB
        
        # Memory growth should be minimal
        assert memory_growth < 50  # Less than 50MB growth after 1000 iterations
```

### Security Testing Framework

#### Security Vulnerability Tests
```python
# tests/security/test_security_vulnerabilities.py
import pytest
import requests
import json
from urllib.parse import quote

class TestSecurityVulnerabilities:
    """Security vulnerability testing."""
    
    BASE_URL = "http://localhost:5000"
    
    def test_sql_injection_protection(self):
        """Test protection against SQL injection."""
        # Test various SQL injection payloads
        payloads = [
            "'; DROP TABLE events; --",
            "' OR '1'='1",
            "1' UNION SELECT * FROM users --",
            "'; INSERT INTO events VALUES ('hacked'); --"
        ]
        
        for payload in payloads:
            # Test in various endpoints
            response = requests.get(f"{self.BASE_URL}/api/events", 
                                  params={'person_name': payload})
            
            # Should not return 500 error or expose database structure
            assert response.status_code != 500
            assert 'database' not in response.text.lower()
            assert 'sql' not in response.text.lower()
    
    def test_xss_protection(self):
        """Test protection against Cross-Site Scripting."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert('xss');//"
        ]
        
        for payload in xss_payloads:
            # Test file upload with malicious filename
            files = {
                'image': (payload + '.jpg', b'fake_image_data', 'image/jpeg')
            }
            response = requests.post(f"{self.BASE_URL}/api/faces/add", 
                                   files=files,
                                   data={'name': payload})
            
            # Response should escape dangerous content
            if response.status_code == 200:
                assert '<script>' not in response.text
                assert 'javascript:' not in response.text
    
    def test_file_upload_security(self):
        """Test file upload security."""
        # Test malicious file types
        malicious_files = [
            ('test.php', b'<?php system($_GET["cmd"]); ?>', 'application/php'),
            ('test.exe', b'MZ\x90\x00', 'application/executable'),
            ('test.js', b'require("child_process").exec("rm -rf /")', 'application/javascript')
        ]
        
        for filename, content, content_type in malicious_files:
            files = {'image': (filename, content, content_type)}
            response = requests.post(f"{self.BASE_URL}/api/faces/add", 
                                   files=files,
                                   data={'name': 'test'})
            
            # Should reject non-image files
            assert response.status_code in [400, 415]  # Bad Request or Unsupported Media Type
    
    def test_path_traversal_protection(self):
        """Test protection against path traversal attacks."""
        traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2f",  # URL encoded
            "....//....//....//etc/passwd"
        ]
        
        for payload in traversal_payloads:
            # Test in file serving endpoint
            response = requests.get(f"{self.BASE_URL}/static/{quote(payload)}")
            
            # Should not serve system files
            assert response.status_code == 404
            assert 'root:' not in response.text  # Unix passwd file
            assert 'administrator' not in response.text.lower()  # Windows
    
    def test_rate_limiting(self):
        """Test API rate limiting."""
        # Rapid fire requests
        responses = []
        for i in range(100):
            response = requests.get(f"{self.BASE_URL}/api/system/status")
            responses.append(response.status_code)
        
        # Should see rate limiting kick in
        rate_limited = sum(1 for status in responses if status == 429)
        assert rate_limited > 0  # Should have some rate limited responses
    
    @pytest.mark.security
    def test_sensitive_data_exposure(self):
        """Test for sensitive data exposure."""
        # Check various endpoints for sensitive information
        endpoints = [
            "/api/system/info",
            "/api/config",
            "/api/debug",
            "/.env",
            "/config.py",
            "/settings.py"
        ]
        
        for endpoint in endpoints:
            response = requests.get(f"{self.BASE_URL}{endpoint}")
            
            if response.status_code == 200:
                content = response.text.lower()
                
                # Should not expose sensitive information
                sensitive_patterns = [
                    'password', 'secret', 'key', 'token',
                    'database_url', 'api_key', 'private'
                ]
                
                for pattern in sensitive_patterns:
                    assert pattern not in content, f"Sensitive data '{pattern}' exposed in {endpoint}"
```

### CI/CD Pipeline Configuration

#### GitHub Actions Workflow
```yaml
# .github/workflows/comprehensive-testing.yml
name: Comprehensive Testing Pipeline

on:
  push:
    branches: [ master, develop ]
  pull_request:
    branches: [ master ]
  schedule:
    # Run nightly tests
    - cron: '0 2 * * *'

env:
  PYTHON_VERSION: '3.11'
  NODE_VERSION: '18'

jobs:
  code-quality:
    name: Code Quality Checks
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[test,security]"
    
    - name: Lint with flake8
      run: |
        flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src tests --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Type checking with mypy
      run: mypy src --ignore-missing-imports
    
    - name: Security scanning with bandit
      run: bandit -r src -f json -o bandit-report.json
    
    - name: Dependency vulnerability scan
      run: safety check --json --output safety-report.json
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json

  unit-tests:
    name: Unit Tests
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install system dependencies (Ubuntu)
      if: matrix.os == 'ubuntu-latest'
      run: |
        sudo apt-get update
        sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
    
    - name: Install Python dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[test]"
    
    - name: Run unit tests with pytest
      run: |
        pytest tests/unit -v \
          --cov=src \
          --cov-report=xml \
          --cov-report=html \
          --junit-xml=junit-${{ matrix.os }}-${{ matrix.python-version }}.xml \
          --html=report-${{ matrix.os }}-${{ matrix.python-version }}.html \
          --self-contained-html
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.os }}-${{ matrix.python-version }}
        path: |
          junit-*.xml
          report-*.html
          htmlcov/
          coverage.xml

  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: unit-tests
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_DB: doorbell_test
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[test]"
    
    - name: Run integration tests
      env:
        DATABASE_URL: postgresql://test:test@localhost/doorbell_test
        TEST_MODE: integration
      run: |
        pytest tests/integration -v \
          --junit-xml=integration-results.xml \
          --html=integration-report.html \
          --self-contained-html
    
    - name: Upload integration test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: integration-test-results
        path: |
          integration-results.xml
          integration-report.html

  e2e-tests:
    name: End-to-End Tests
    runs-on: ubuntu-latest
    needs: integration-tests
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: ${{ env.NODE_VERSION }}
    
    - name: Install Python dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[test,e2e]"
    
    - name: Install Playwright
      run: playwright install --with-deps
    
    - name: Start application
      run: |
        python app.py &
        sleep 10  # Wait for app to start
      env:
        FLASK_ENV: testing
        TEST_MODE: e2e
    
    - name: Run E2E tests
      run: |
        pytest tests/e2e -v \
          --junit-xml=e2e-results.xml \
          --html=e2e-report.html \
          --self-contained-html
    
    - name: Upload E2E test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: e2e-test-results
        path: |
          e2e-results.xml
          e2e-report.html
          test-results/

  performance-tests:
    name: Performance Tests
    runs-on: ubuntu-latest
    needs: integration-tests
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[test,performance]"
    
    - name: Start application
      run: |
        python app.py &
        sleep 10
      env:
        FLASK_ENV: testing
    
    - name: Run performance tests
      run: |
        pytest tests/performance -v --benchmark-only --benchmark-json=benchmark.json
    
    - name: Run load tests
      run: |
        locust -f tests/performance/test_load_testing.py \
          --headless \
          --users 50 \
          --spawn-rate 5 \
          --run-time 60s \
          --host http://localhost:5000 \
          --html load-test-report.html
    
    - name: Upload performance test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: performance-test-results
        path: |
          benchmark.json
          load-test-report.html

  security-tests:
    name: Security Tests
    runs-on: ubuntu-latest
    needs: unit-tests
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[test,security]"
    
    - name: Start application
      run: |
        python app.py &
        sleep 10
      env:
        FLASK_ENV: testing
        SECURITY_TEST_MODE: true
    
    - name: Run security tests
      run: |
        pytest tests/security -v \
          --junit-xml=security-results.xml \
          --html=security-report.html \
          --self-contained-html
    
    - name: OWASP ZAP Baseline Scan
      uses: zaproxy/action-baseline@v0.7.0
      with:
        target: 'http://localhost:5000'
        rules_file_name: '.zap/rules.tsv'
        cmd_options: '-a'
    
    - name: Upload security test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: security-test-results
        path: |
          security-results.xml
          security-report.html
          report_html.html
          report_md.md

  coverage-report:
    name: Coverage Report
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Download coverage reports
      uses: actions/download-artifact@v3
      with:
        pattern: test-results-*
        merge-multiple: true
    
    - name: Combine coverage reports
      run: |
        python -m pip install coverage
        coverage combine
        coverage report --show-missing
        coverage html
    
    - name: Upload to Codecov
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml
        fail_ci_if_error: true
    
    - name: Coverage comment
      uses: py-cov-action/python-coverage-comment-action@v3
      with:
        GITHUB_TOKEN: ${{ github.token }}

  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests, e2e-tests, security-tests]
    if: github.ref == 'refs/heads/develop'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to staging environment
      run: |
        echo "Deploying to staging..."
        # Add deployment scripts here
    
    - name: Run smoke tests
      run: |
        # Add smoke tests for staging deployment
        curl -f http://staging.doorbell-system.com/health
```

## Implementation Timeline

### Phase 1: Foundation (Week 1-2)
1. **Test Infrastructure Setup**
   - [ ] Configure pytest with all plugins
   - [ ] Set up test database and fixtures
   - [ ] Create base test classes and utilities
   - [ ] Configure GitHub Actions workflows

2. **Unit Test Coverage**
   - [ ] Write unit tests for all core components (>80% coverage)
   - [ ] Mock external dependencies and hardware
   - [ ] Implement property-based tests for critical functions
   - [ ] Add performance benchmarks

### Phase 2: Integration Testing (Week 2-3)
1. **Component Integration**
   - [ ] Test pipeline component interactions
   - [ ] Database integration tests
   - [ ] Message bus communication tests
   - [ ] Hardware abstraction layer tests

2. **API Integration**
   - [ ] REST API endpoint testing
   - [ ] WebSocket communication testing
   - [ ] File upload/download testing
   - [ ] Authentication and authorization testing

### Phase 3: End-to-End Testing (Week 3-4)
1. **Browser Automation**
   - [ ] Set up Playwright test framework
   - [ ] Create page object models
   - [ ] Test critical user journeys
   - [ ] Mobile responsiveness testing

2. **User Experience Testing**
   - [ ] Accessibility compliance testing
   - [ ] Cross-browser compatibility testing
   - [ ] Performance testing with real browsers
   - [ ] Visual regression testing

### Phase 4: Performance & Security (Week 4-5)
1. **Performance Testing**
   - [ ] Load testing with Locust
   - [ ] Memory profiling and leak detection
   - [ ] CPU usage optimization testing
   - [ ] Database performance testing

2. **Security Testing**
   - [ ] Vulnerability scanning
   - [ ] Penetration testing simulation
   - [ ] Input validation testing
   - [ ] Authentication security testing

### Phase 5: CI/CD Automation (Week 5-6)
1. **Pipeline Optimization**
   - [ ] Parallel test execution
   - [ ] Test result reporting
   - [ ] Coverage tracking and enforcement
   - [ ] Performance regression detection

2. **Production Deployment**
   - [ ] Staging environment testing
   - [ ] Production smoke tests
   - [ ] Monitoring and alerting setup
   - [ ] Rollback procedures testing

## Acceptance Criteria

### Test Coverage Requirements
- [ ] Unit test coverage: >85% for core components
- [ ] Integration test coverage: >70% for component interactions
- [ ] E2E test coverage: 100% for critical user journeys
- [ ] API test coverage: 100% for all endpoints

### Performance Standards
- [ ] Unit tests: Complete in <30 seconds
- [ ] Integration tests: Complete in <5 minutes
- [ ] E2E tests: Complete in <15 minutes
- [ ] Full CI pipeline: Complete in <20 minutes

### Quality Gates
- [ ] All tests pass on supported platforms (Linux, macOS, Windows)
- [ ] No critical security vulnerabilities
- [ ] Performance benchmarks meet requirements
- [ ] Accessibility compliance (WCAG 2.1 AA)

### CI/CD Requirements
- [ ] Automated testing on every pull request
- [ ] Automated deployment to staging environment
- [ ] Performance regression detection
- [ ] Security vulnerability scanning

This comprehensive testing framework ensures the Doorbell Security System meets enterprise-grade quality standards with robust automated testing and continuous integration processes.