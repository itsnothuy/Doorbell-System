# Issue #15: Comprehensive Testing Framework and Quality Assurance

## 📋 **Overview**

Establish a comprehensive testing framework that validates the complete pipeline architecture, ensures quality across all components, and provides confidence for production deployment. This includes end-to-end testing, performance validation, security testing, and continuous integration pipelines.

## 🎯 **Objectives**

### **Primary Goals**
1. **Complete Test Coverage**: Achieve 95%+ code coverage across all pipeline components
2. **End-to-End Validation**: Comprehensive integration testing of the entire system
3. **Performance Testing**: Validate performance requirements and regression detection
4. **Security Testing**: Comprehensive security validation and penetration testing
5. **CI/CD Integration**: Automated testing pipeline with quality gates
6. **Load Testing**: Validate system behavior under high load conditions

### **Success Criteria**
- 95%+ code coverage across all modules
- End-to-end test suite covering all user scenarios
- Performance tests validating 30% improvement over legacy system
- Security tests with zero critical vulnerabilities
- Automated CI/CD pipeline with comprehensive quality gates
- Load testing demonstrating system stability under 10x normal load

## 🏗️ **Testing Architecture**

### **Testing Pyramid Structure**
```
                   /\
                  /  \
                 / E2E \              <- End-to-End Tests (10%)
                /______\
               /        \
              / Integration \         <- Integration Tests (30%)
             /______________\
            /                \
           /   Unit Tests      \      <- Unit Tests (60%)
          /____________________\
```

### **Test Categories**
```
Unit Tests (60%)           - Individual component testing
├── Pipeline Workers       - Frame capture, detection, recognition
├── Communication         - Message bus, events, queues
├── Hardware Abstraction  - Camera, GPIO handlers
├── Storage Layer         - Databases, caching
└── Utilities            - Configuration, logging, helpers

Integration Tests (30%)    - Component interaction testing
├── Pipeline Flow         - End-to-end pipeline processing
├── API Integration       - Web interface and external APIs
├── Hardware Integration  - Camera and GPIO integration
├── Database Integration  - Storage operations
└── Performance Tests     - Latency, throughput, resource usage

End-to-End Tests (10%)     - Complete system scenarios
├── User Scenarios        - Doorbell trigger to notification
├── Error Scenarios       - Failure handling and recovery
├── Security Scenarios    - Authentication, authorization
└── Load Scenarios       - High traffic and stress testing
```

## 📁 **Implementation Specifications**

### **Files to Create**

#### **Testing Framework Core**
```
tests/
├── conftest.py                               # Enhanced pytest configuration
├── fixtures/                                 # Comprehensive test fixtures
│   ├── hardware_fixtures.py                 # Hardware mock fixtures
│   ├── pipeline_fixtures.py                 # Pipeline component fixtures
│   ├── data_fixtures.py                     # Test data fixtures
│   └── performance_fixtures.py              # Performance testing fixtures
├── unit/                                     # Unit tests (60%)
│   ├── test_pipeline/                        # Pipeline worker tests
│   │   ├── test_orchestrator.py             # Orchestrator unit tests
│   │   ├── test_frame_capture.py            # Frame capture tests
│   │   ├── test_motion_detector.py          # Motion detection tests
│   │   ├── test_face_detector.py            # Face detection tests
│   │   ├── test_face_recognizer.py          # Face recognition tests
│   │   └── test_event_processor.py          # Event processing tests
│   ├── test_communication/                  # Communication layer tests
│   │   ├── test_message_bus.py              # Message bus tests
│   │   ├── test_events.py                   # Event system tests
│   │   └── test_queues.py                   # Queue management tests
│   ├── test_detectors/                      # Detector strategy tests
│   │   ├── test_base_detector.py            # Base detector tests
│   │   ├── test_cpu_detector.py             # CPU detector tests
│   │   ├── test_gpu_detector.py             # GPU detector tests
│   │   └── test_edgetpu_detector.py         # EdgeTPU detector tests
│   ├── test_storage/                        # Storage layer tests
│   │   ├── test_event_database.py           # Event database tests
│   │   ├── test_face_database.py            # Face database tests
│   │   └── test_cache_manager.py            # Cache management tests
│   ├── test_hardware/                       # Hardware abstraction tests
│   │   ├── test_camera_handler.py           # Camera handler tests
│   │   └── test_gpio_handler.py             # GPIO handler tests
│   └── test_enrichment/                     # Enrichment processor tests
│       ├── test_notification_handler.py     # Notification tests
│       ├── test_alert_manager.py            # Alert management tests
│       └── test_web_events.py               # Web event tests
├── integration/                             # Integration tests (30%)
│   ├── test_pipeline_integration.py         # Pipeline flow tests
│   ├── test_web_integration.py              # Web interface integration
│   ├── test_api_integration.py              # API integration tests
│   ├── test_hardware_integration.py         # Hardware integration tests
│   ├── test_database_integration.py         # Database integration tests
│   └── test_migration_integration.py        # Migration integration tests
├── e2e/                                     # End-to-end tests (10%)
│   ├── test_doorbell_scenarios.py           # Complete doorbell scenarios
│   ├── test_face_recognition_flow.py        # Face recognition end-to-end
│   ├── test_notification_flow.py            # Notification end-to-end
│   ├── test_web_interface_flow.py           # Web interface scenarios
│   └── test_error_recovery.py               # Error handling scenarios
├── performance/                             # Performance testing
│   ├── test_throughput.py                   # Throughput benchmarks
│   ├── test_latency.py                      # Latency measurements
│   ├── test_resource_usage.py               # Resource utilization tests
│   ├── test_memory_management.py            # Memory leak detection
│   └── test_load_scenarios.py               # Load testing scenarios
├── security/                                # Security testing
│   ├── test_input_validation.py             # Input validation tests
│   ├── test_authentication.py               # Authentication security
│   ├── test_api_security.py                 # API security tests
│   └── test_data_protection.py              # Data protection tests
└── load/                                    # Load and stress testing
    ├── test_stress_scenarios.py             # Stress testing
    ├── test_concurrent_users.py             # Concurrent user testing
    └── test_system_limits.py                # System limit testing
```

#### **Testing Infrastructure**
```
tests/utils/                                 # Testing utilities
├── test_helpers.py                          # Common testing helpers
├── mock_hardware.py                         # Comprehensive hardware mocks
├── test_data_generator.py                   # Test data generation
├── performance_profiler.py                  # Performance profiling tools
└── security_scanner.py                      # Security testing tools

scripts/testing/                             # Testing scripts
├── run_full_test_suite.py                   # Complete test execution
├── generate_coverage_report.py              # Coverage reporting
├── run_performance_tests.py                 # Performance test runner
├── run_security_tests.py                    # Security test runner
└── validate_test_environment.py             # Test environment validation

ci/                                          # CI/CD testing configuration
├── github_actions_tests.yml                 # GitHub Actions workflow
├── quality_gates.yml                        # Quality gate definitions
└── test_matrix.yml                          # Test matrix configuration
```

### **Core Component: Enhanced Test Configuration**
```python
#!/usr/bin/env python3
"""
Enhanced PyTest Configuration

Comprehensive test configuration with fixtures, utilities, and testing infrastructure.
"""

import pytest
import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator, Optional
from unittest.mock import Mock, MagicMock
import numpy as np
import cv2
import sqlite3

# Configure test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Test configuration fixture."""
    return {
        'test_mode': True,
        'mock_hardware': True,
        'use_test_database': True,
        'disable_external_services': True,
        'performance_mode': False,
        'debug_mode': True
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


@pytest.fixture
def test_database(temp_test_dir: Path) -> Generator[str, None, None]:
    """Create test database."""
    db_path = temp_test_dir / "data" / "test.db"
    
    # Create test database
    conn = sqlite3.connect(str(db_path))
    
    # Create test tables
    conn.execute('''
        CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            event_type TEXT,
            timestamp REAL,
            data TEXT,
            source TEXT
        )
    ''')
    
    conn.execute('''
        CREATE TABLE known_faces (
            id INTEGER PRIMARY KEY,
            person_name TEXT,
            face_encoding BLOB,
            image_path TEXT,
            created_at REAL
        )
    ''')
    
    conn.commit()
    conn.close()
    
    yield str(db_path)


@pytest.fixture
def mock_camera():
    """Mock camera fixture."""
    camera = Mock()
    camera.capture_array.return_value = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
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
def test_known_faces(temp_test_dir: Path, sample_face_image: np.ndarray) -> Dict[str, Any]:
    """Create test known faces."""
    known_faces_dir = temp_test_dir / "data" / "known_faces"
    
    # Create test face images
    test_faces = {
        'john_doe': sample_face_image,
        'jane_smith': sample_face_image,
        'test_person': sample_face_image
    }
    
    for name, image in test_faces.items():
        image_path = known_faces_dir / f"{name}_001.jpg"
        cv2.imwrite(str(image_path), image)
    
    return {
        'directory': known_faces_dir,
        'faces': test_faces,
        'count': len(test_faces)
    }


@pytest.fixture
def mock_message_bus():
    """Mock message bus fixture."""
    from src.communication.message_bus import MessageBus
    
    bus = Mock(spec=MessageBus)
    bus.publish = Mock()
    bus.subscribe = Mock()
    bus.unsubscribe = Mock()
    bus.start = Mock()
    bus.stop = Mock()
    bus.get_metrics = Mock(return_value={'messages_sent': 0, 'messages_received': 0})
    
    return bus


@pytest.fixture
def mock_pipeline_orchestrator():
    """Mock pipeline orchestrator fixture."""
    from src.pipeline.orchestrator import PipelineOrchestrator
    
    orchestrator = Mock(spec=PipelineOrchestrator)
    orchestrator.start = Mock()
    orchestrator.stop = Mock()
    orchestrator.get_health_status = Mock()
    orchestrator.trigger_doorbell = Mock(return_value={'status': 'success'})
    orchestrator.is_running = Mock(return_value=False)
    
    return orchestrator


@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture."""
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {}
            
        def start_timer(self, operation: str):
            import time
            self.metrics[f"{operation}_start"] = time.time()
            
        def end_timer(self, operation: str):
            import time
            start_time = self.metrics.get(f"{operation}_start")
            if start_time:
                self.metrics[f"{operation}_duration"] = time.time() - start_time
                
        def get_metrics(self) -> Dict[str, float]:
            return {k: v for k, v in self.metrics.items() if not k.endswith('_start')}
    
    return PerformanceMonitor()


@pytest.fixture
def test_event_data():
    """Test event data fixture."""
    return {
        'doorbell_event': {
            'event_type': 'doorbell_triggered',
            'timestamp': 1635724800.0,
            'data': {'trigger_type': 'button_press'},
            'source': 'gpio_handler'
        },
        'face_detection_event': {
            'event_type': 'face_detected',
            'timestamp': 1635724801.0,
            'data': {
                'face_count': 1,
                'confidence': 0.95,
                'bounding_box': [100, 100, 200, 200]
            },
            'source': 'face_detector'
        },
        'recognition_event': {
            'event_type': 'face_recognized',
            'timestamp': 1635724802.0,
            'data': {
                'person_name': 'john_doe',
                'confidence': 0.85,
                'is_known': True
            },
            'source': 'face_recognizer'
        }
    }


# Test utilities
def assert_performance_requirements(metrics: Dict[str, float], requirements: Dict[str, float]):
    """Assert performance requirements are met."""
    for metric, threshold in requirements.items():
        actual_value = metrics.get(metric)
        assert actual_value is not None, f"Missing performance metric: {metric}"
        assert actual_value <= threshold, f"Performance requirement failed: {metric} = {actual_value}, required <= {threshold}"


def generate_test_load(num_events: int = 100) -> list:
    """Generate test load data."""
    import time
    
    events = []
    base_time = time.time()
    
    for i in range(num_events):
        events.append({
            'id': i,
            'timestamp': base_time + (i * 0.1),
            'event_type': 'test_event',
            'data': {'test_data': f'event_{i}'}
        })
    
    return events


# Pytest configuration
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
```

### **Comprehensive Unit Testing Example**
```python
#!/usr/bin/env python3
"""
Pipeline Orchestrator Unit Tests

Comprehensive unit testing for the pipeline orchestrator component.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any

from src.pipeline.orchestrator import PipelineOrchestrator, OrchestratorState, HealthStatus
from src.communication.message_bus import MessageBus
from src.communication.events import PipelineEvent, EventType


class TestPipelineOrchestrator:
    """Comprehensive unit tests for PipelineOrchestrator."""
    
    @pytest.fixture
    def orchestrator_config(self):
        """Orchestrator configuration for testing."""
        return {
            'frame_capture': {
                'enabled': True,
                'worker_count': 1,
                'debounce_time': 1.0
            },
            'face_detection': {
                'enabled': True,
                'worker_count': 2,
                'detector_type': 'cpu'
            },
            'face_recognition': {
                'enabled': True,
                'worker_count': 1,
                'tolerance': 0.6
            },
            'event_processing': {
                'enabled': True,
                'max_queue_size': 100
            },
            'monitoring': {
                'health_check_interval': 0.5,
                'performance_monitoring': True
            }
        }
    
    @pytest.fixture
    def mock_workers(self):
        """Mock pipeline workers."""
        workers = {
            'frame_capture': Mock(),
            'motion_detector': Mock(),
            'face_detector': Mock(),
            'face_recognizer': Mock(),
            'event_processor': Mock()
        }
        
        # Configure worker mocks
        for worker in workers.values():
            worker.start = Mock()
            worker.stop = Mock()
            worker.is_running = Mock(return_value=False)
            worker.get_metrics = Mock(return_value={'processed_count': 0})
            worker.get_health = Mock(return_value={'status': 'healthy'})
        
        return workers
    
    @pytest.fixture
    def orchestrator(self, orchestrator_config, mock_message_bus, mock_workers):
        """Create orchestrator instance for testing."""
        with patch('src.pipeline.orchestrator.FrameCaptureWorker', return_value=mock_workers['frame_capture']):
            with patch('src.pipeline.orchestrator.MotionDetectorWorker', return_value=mock_workers['motion_detector']):
                with patch('src.pipeline.orchestrator.FaceDetectorWorker', return_value=mock_workers['face_detector']):
                    with patch('src.pipeline.orchestrator.FaceRecognizerWorker', return_value=mock_workers['face_recognizer']):
                        with patch('src.pipeline.orchestrator.EventProcessorWorker', return_value=mock_workers['event_processor']):
                            return PipelineOrchestrator(
                                message_bus=mock_message_bus,
                                config=orchestrator_config
                            )
    
    def test_orchestrator_initialization(self, orchestrator, orchestrator_config):
        """Test orchestrator initialization."""
        assert orchestrator.config == orchestrator_config
        assert orchestrator.state == OrchestratorState.STOPPED
        assert not orchestrator.is_running()
        assert orchestrator.orchestrator_id is not None
        assert orchestrator.start_time is None
    
    def test_orchestrator_start_success(self, orchestrator, mock_workers):
        """Test successful orchestrator startup."""
        # Configure workers to start successfully
        for worker in mock_workers.values():
            worker.start.return_value = True
            worker.is_running.return_value = True
        
        # Start orchestrator
        result = orchestrator.start()
        
        # Verify results
        assert result is True
        assert orchestrator.state == OrchestratorState.RUNNING
        assert orchestrator.is_running()
        assert orchestrator.start_time is not None
        
        # Verify all workers were started
        for worker in mock_workers.values():
            worker.start.assert_called_once()
    
    def test_orchestrator_start_failure(self, orchestrator, mock_workers):
        """Test orchestrator startup failure."""
        # Configure one worker to fail startup
        mock_workers['face_detector'].start.side_effect = Exception("Worker startup failed")
        
        # Start orchestrator
        result = orchestrator.start()
        
        # Verify results
        assert result is False
        assert orchestrator.state == OrchestratorState.ERROR
        assert not orchestrator.is_running()
    
    def test_orchestrator_stop(self, orchestrator, mock_workers):
        """Test orchestrator shutdown."""
        # Start orchestrator first
        for worker in mock_workers.values():
            worker.start.return_value = True
            worker.is_running.return_value = True
        
        orchestrator.start()
        
        # Configure workers to stop successfully
        for worker in mock_workers.values():
            worker.stop.return_value = True
            worker.is_running.return_value = False
        
        # Stop orchestrator
        result = orchestrator.stop()
        
        # Verify results
        assert result is True
        assert orchestrator.state == OrchestratorState.STOPPED
        assert not orchestrator.is_running()
        
        # Verify all workers were stopped
        for worker in mock_workers.values():
            worker.stop.assert_called_once()
    
    def test_health_monitoring(self, orchestrator, mock_workers):
        """Test health monitoring functionality."""
        # Start orchestrator
        for worker in mock_workers.values():
            worker.start.return_value = True
            worker.is_running.return_value = True
            worker.get_health.return_value = {'status': 'healthy', 'errors': 0}
        
        orchestrator.start()
        
        # Get health status
        health = orchestrator.get_health_status()
        
        # Verify health status
        assert isinstance(health, HealthStatus)
        assert health.state == OrchestratorState.RUNNING
        assert health.uptime > 0
        assert health.worker_health is not None
        assert health.performance_score > 0
    
    def test_doorbell_trigger(self, orchestrator, mock_message_bus, mock_workers):
        """Test doorbell trigger functionality."""
        # Start orchestrator
        for worker in mock_workers.values():
            worker.start.return_value = True
            worker.is_running.return_value = True
        
        orchestrator.start()
        
        # Trigger doorbell
        result = orchestrator.trigger_doorbell({'source': 'test'})
        
        # Verify results
        assert result['status'] == 'success'
        assert 'event_id' in result
        
        # Verify event was published
        mock_message_bus.publish.assert_called()
        published_args = mock_message_bus.publish.call_args
        assert published_args[0][0] == 'doorbell_events'  # Topic
        assert published_args[0][1].event_type == EventType.DOORBELL_TRIGGERED
    
    def test_performance_monitoring(self, orchestrator, mock_workers, performance_monitor):
        """Test performance monitoring."""
        # Configure workers with performance metrics
        for worker in mock_workers.values():
            worker.start.return_value = True
            worker.is_running.return_value = True
            worker.get_metrics.return_value = {
                'processed_count': 100,
                'error_count': 1,
                'avg_processing_time': 0.05,
                'queue_size': 5
            }
        
        orchestrator.start()
        
        # Get performance metrics
        performance_monitor.start_timer('get_metrics')
        metrics = orchestrator.get_performance_metrics()
        performance_monitor.end_timer('get_metrics')
        
        # Verify metrics
        assert 'total_processed' in metrics
        assert 'total_errors' in metrics
        assert 'avg_processing_time' in metrics
        assert 'worker_metrics' in metrics
        
        # Verify performance requirements
        perf_metrics = performance_monitor.get_metrics()
        assert_performance_requirements(perf_metrics, {'get_metrics_duration': 0.1})
    
    def test_worker_failure_handling(self, orchestrator, mock_workers):
        """Test worker failure handling and recovery."""
        # Start orchestrator
        for worker in mock_workers.values():
            worker.start.return_value = True
            worker.is_running.return_value = True
        
        orchestrator.start()
        
        # Simulate worker failure
        mock_workers['face_detector'].is_running.return_value = False
        mock_workers['face_detector'].get_health.return_value = {
            'status': 'failed', 
            'error': 'Worker crashed'
        }
        
        # Check health after failure
        health = orchestrator.get_health_status()
        
        # Verify failure detection
        assert health.worker_health['face_detector']['status'] == 'failed'
        assert health.performance_score < 1.0  # Should be degraded
    
    def test_concurrent_operations(self, orchestrator, mock_workers):
        """Test concurrent operations handling."""
        # Start orchestrator
        for worker in mock_workers.values():
            worker.start.return_value = True
            worker.is_running.return_value = True
        
        orchestrator.start()
        
        # Simulate concurrent doorbell triggers
        results = []
        threads = []
        
        def trigger_doorbell():
            result = orchestrator.trigger_doorbell({'source': 'concurrent_test'})
            results.append(result)
        
        # Create multiple threads
        for i in range(5):
            thread = threading.Thread(target=trigger_doorbell)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify all triggers succeeded
        assert len(results) == 5
        for result in results:
            assert result['status'] == 'success'
    
    def test_configuration_validation(self, mock_message_bus):
        """Test configuration validation."""
        # Test with invalid configuration
        invalid_config = {
            'frame_capture': {
                'enabled': True,
                'worker_count': -1  # Invalid
            }
        }
        
        # Should raise configuration error
        with pytest.raises(ValueError, match="Invalid worker count"):
            PipelineOrchestrator(
                message_bus=mock_message_bus,
                config=invalid_config
            )
    
    def test_graceful_shutdown(self, orchestrator, mock_workers):
        """Test graceful shutdown with timeout."""
        # Start orchestrator
        for worker in mock_workers.values():
            worker.start.return_value = True
            worker.is_running.return_value = True
        
        orchestrator.start()
        
        # Configure one worker to take time to stop
        mock_workers['face_detector'].stop.side_effect = lambda: time.sleep(0.1)
        
        # Stop with timeout
        start_time = time.time()
        result = orchestrator.stop(timeout=1.0)
        stop_duration = time.time() - start_time
        
        # Verify graceful shutdown
        assert result is True
        assert stop_duration < 1.5  # Should complete within reasonable time
    
    @pytest.mark.performance
    def test_startup_performance(self, orchestrator, mock_workers, performance_monitor):
        """Test orchestrator startup performance."""
        # Configure workers
        for worker in mock_workers.values():
            worker.start.return_value = True
            worker.is_running.return_value = True
        
        # Measure startup time
        performance_monitor.start_timer('startup')
        result = orchestrator.start()
        performance_monitor.end_timer('startup')
        
        # Verify performance
        assert result is True
        
        perf_metrics = performance_monitor.get_metrics()
        assert_performance_requirements(perf_metrics, {'startup_duration': 2.0})
    
    @pytest.mark.integration
    def test_message_bus_integration(self, orchestrator, mock_workers):
        """Test integration with message bus."""
        # This would be in integration tests, but shown here for completeness
        from src.communication.message_bus import MessageBus
        
        # Use real message bus for integration testing
        real_message_bus = MessageBus()
        real_message_bus.start()
        
        try:
            # Test with real message bus
            orchestrator_with_real_bus = PipelineOrchestrator(
                message_bus=real_message_bus,
                config=orchestrator.config
            )
            
            # Test basic operations
            # This would include more comprehensive integration testing
            
        finally:
            real_message_bus.stop()
```

### **End-to-End Testing Example**
```python
#!/usr/bin/env python3
"""
End-to-End Doorbell Scenarios

Complete user scenario testing from doorbell trigger to notification.
"""

import pytest
import time
import requests
from pathlib import Path
from typing import Dict, Any

from src.integration.orchestrator_manager import OrchestratorManager
from src.web_interface import create_web_app


class TestDoorbellScenarios:
    """End-to-end doorbell scenario tests."""
    
    @pytest.fixture(scope="class")
    def running_system(self, test_config, temp_test_dir):
        """Setup running system for E2E tests."""
        # Configure test environment
        system_config = {
            **test_config,
            'data_dir': str(temp_test_dir),
            'web_interface': {'port': 5001}  # Use different port for testing
        }
        
        # Start orchestrator manager
        manager = OrchestratorManager(config=system_config)
        manager.start()
        
        # Create web app
        legacy_interface = manager.get_legacy_interface()
        app = create_web_app(legacy_interface)
        
        # Start web server in test mode
        app.config['TESTING'] = True
        test_client = app.test_client()
        
        yield {
            'manager': manager,
            'app': app,
            'client': test_client,
            'config': system_config
        }
        
        # Cleanup
        manager.stop()
    
    def test_doorbell_trigger_to_notification_flow(self, running_system, test_known_faces):
        """Test complete flow from doorbell trigger to notification."""
        manager = running_system['manager']
        client = running_system['client']
        
        # Step 1: Trigger doorbell
        trigger_result = manager.trigger_doorbell({
            'source': 'e2e_test',
            'trigger_type': 'button_press'
        })
        
        assert trigger_result['status'] == 'success'
        event_id = trigger_result['event_id']
        
        # Step 2: Wait for processing
        time.sleep(2.0)  # Allow time for pipeline processing
        
        # Step 3: Check event was processed
        response = client.get(f'/api/events/{event_id}')
        assert response.status_code == 200
        
        event_data = response.get_json()
        assert event_data['event_id'] == event_id
        assert event_data['status'] in ['completed', 'processing']
        
        # Step 4: Check notifications were sent
        response = client.get('/api/notifications')
        assert response.status_code == 200
        
        notifications = response.get_json()
        assert len(notifications) > 0
        
        # Find notification for our event
        event_notification = next(
            (n for n in notifications if n.get('event_id') == event_id),
            None
        )
        assert event_notification is not None
        assert event_notification['type'] in ['doorbell_triggered', 'face_detected']
    
    def test_face_recognition_flow_known_person(self, running_system, test_known_faces):
        """Test face recognition flow with known person."""
        manager = running_system['manager']
        client = running_system['client']
        
        # Step 1: Add known face for testing
        known_face_data = {
            'person_name': 'test_person',
            'image_data': 'base64_encoded_image_data'  # Would be actual image data
        }
        
        response = client.post('/api/faces/known', json=known_face_data)
        assert response.status_code == 201
        
        # Step 2: Trigger doorbell with face image
        trigger_result = manager.trigger_doorbell({
            'source': 'e2e_test',
            'image_data': 'base64_encoded_test_image'  # Would be actual image
        })
        
        assert trigger_result['status'] == 'success'
        event_id = trigger_result['event_id']
        
        # Step 3: Wait for face recognition processing
        time.sleep(3.0)
        
        # Step 4: Check recognition results
        response = client.get(f'/api/events/{event_id}/recognition')
        assert response.status_code == 200
        
        recognition_data = response.get_json()
        assert 'faces_detected' in recognition_data
        assert recognition_data['faces_detected'] > 0
        
        # Check if known person was recognized
        if recognition_data.get('known_faces'):
            known_face = recognition_data['known_faces'][0]
            assert known_face['person_name'] == 'test_person'
            assert known_face['confidence'] > 0.5
    
    def test_web_interface_real_time_updates(self, running_system):
        """Test web interface real-time updates."""
        client = running_system['client']
        
        # Step 1: Open web interface
        response = client.get('/')
        assert response.status_code == 200
        
        # Step 2: Check WebSocket endpoint exists
        response = client.get('/api/events/stream')
        assert response.status_code == 200  # Should support streaming
        
        # Step 3: Trigger event and check for updates
        # This would involve WebSocket testing in a real implementation
        pass
    
    def test_system_recovery_after_error(self, running_system):
        """Test system recovery after component failure."""
        manager = running_system['manager']
        
        # Step 1: Simulate component failure
        # This would involve more sophisticated error injection
        
        # Step 2: Check system detects failure
        health = manager.get_health_status()
        initial_score = health.performance_score
        
        # Step 3: Verify system attempts recovery
        time.sleep(5.0)  # Allow time for recovery
        
        health = manager.get_health_status()
        # System should either recover or maintain degraded operation
        assert health.state.value in ['running', 'degraded']
    
    def test_high_load_scenario(self, running_system):
        """Test system behavior under high load."""
        manager = running_system['manager']
        
        # Step 1: Generate multiple rapid triggers
        results = []
        for i in range(10):
            result = manager.trigger_doorbell({
                'source': f'load_test_{i}',
                'batch_id': 'high_load_test'
            })
            results.append(result)
            time.sleep(0.1)  # Rapid succession
        
        # Step 2: Verify all events were accepted
        successful_triggers = [r for r in results if r['status'] == 'success']
        assert len(successful_triggers) >= 8  # Allow some failures under load
        
        # Step 3: Wait for processing
        time.sleep(10.0)
        
        # Step 4: Verify system stability
        health = manager.get_health_status()
        assert health.state.value in ['running', 'degraded']
        assert health.performance_score > 0.3  # Should maintain reasonable performance
    
    def test_api_backward_compatibility(self, running_system):
        """Test API backward compatibility with legacy interface."""
        client = running_system['client']
        
        # Test legacy API endpoints still work
        legacy_endpoints = [
            '/api/status',
            '/api/faces',
            '/api/events',
            '/api/config'
        ]
        
        for endpoint in legacy_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200, f"Legacy endpoint failed: {endpoint}"
            
            # Verify response format is compatible
            data = response.get_json()
            assert isinstance(data, (dict, list)), f"Invalid response format for {endpoint}"
```

### **Performance Testing Framework**
```python
#!/usr/bin/env python3
"""
Performance Testing Framework

Comprehensive performance testing for the pipeline system.
"""

import pytest
import time
import statistics
import threading
from typing import Dict, Any, List
from dataclasses import dataclass

from src.integration.orchestrator_manager import OrchestratorManager


@dataclass
class PerformanceMetric:
    """Performance metric result."""
    name: str
    value: float
    unit: str
    threshold: float
    passed: bool


class PerformanceTestFramework:
    """Framework for performance testing."""
    
    def __init__(self, manager: OrchestratorManager):
        self.manager = manager
        self.results: List[PerformanceMetric] = []
    
    def measure_latency(self, operation_name: str, operation_func, iterations: int = 100) -> PerformanceMetric:
        """Measure operation latency."""
        latencies = []
        
        for _ in range(iterations):
            start_time = time.time()
            operation_func()
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        avg_latency = statistics.mean(latencies)
        threshold = 100.0  # 100ms threshold
        
        metric = PerformanceMetric(
            name=f"{operation_name}_latency",
            value=avg_latency,
            unit="ms",
            threshold=threshold,
            passed=avg_latency <= threshold
        )
        
        self.results.append(metric)
        return metric
    
    def measure_throughput(self, operation_name: str, operation_func, duration: float = 10.0) -> PerformanceMetric:
        """Measure operation throughput."""
        start_time = time.time()
        end_time = start_time + duration
        operation_count = 0
        
        while time.time() < end_time:
            operation_func()
            operation_count += 1
        
        actual_duration = time.time() - start_time
        throughput = operation_count / actual_duration
        threshold = 10.0  # 10 operations/second threshold
        
        metric = PerformanceMetric(
            name=f"{operation_name}_throughput",
            value=throughput,
            unit="ops/sec",
            threshold=threshold,
            passed=throughput >= threshold
        )
        
        self.results.append(metric)
        return metric
    
    def measure_resource_usage(self, operation_name: str, operation_func) -> List[PerformanceMetric]:
        """Measure resource usage during operation."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Baseline measurements
        baseline_cpu = process.cpu_percent()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run operation
        start_time = time.time()
        operation_func()
        duration = time.time() - start_time
        
        # Final measurements
        final_cpu = process.cpu_percent()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        cpu_usage = final_cpu - baseline_cpu
        memory_usage = final_memory - baseline_memory
        
        metrics = [
            PerformanceMetric(
                name=f"{operation_name}_cpu_usage",
                value=cpu_usage,
                unit="%",
                threshold=50.0,
                passed=cpu_usage <= 50.0
            ),
            PerformanceMetric(
                name=f"{operation_name}_memory_usage",
                value=memory_usage,
                unit="MB",
                threshold=100.0,
                passed=memory_usage <= 100.0
            )
        ]
        
        self.results.extend(metrics)
        return metrics
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        passed_tests = len([r for r in self.results if r.passed])
        total_tests = len(self.results)
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'overall_passed': passed_tests == total_tests
            },
            'metrics': [
                {
                    'name': metric.name,
                    'value': metric.value,
                    'unit': metric.unit,
                    'threshold': metric.threshold,
                    'passed': metric.passed
                }
                for metric in self.results
            ]
        }


class TestPerformanceRequirements:
    """Performance requirement tests."""
    
    @pytest.fixture
    def performance_framework(self, running_system):
        """Performance testing framework."""
        return PerformanceTestFramework(running_system['manager'])
    
    @pytest.mark.performance
    def test_doorbell_trigger_latency(self, performance_framework):
        """Test doorbell trigger latency."""
        def trigger_operation():
            return performance_framework.manager.trigger_doorbell({'source': 'perf_test'})
        
        metric = performance_framework.measure_latency('doorbell_trigger', trigger_operation)
        assert metric.passed, f"Latency requirement failed: {metric.value}ms > {metric.threshold}ms"
    
    @pytest.mark.performance
    def test_face_recognition_throughput(self, performance_framework):
        """Test face recognition throughput."""
        # This would involve submitting face recognition tasks
        def recognition_operation():
            # Simulate face recognition task
            time.sleep(0.05)  # Simulate processing time
            return True
        
        metric = performance_framework.measure_throughput('face_recognition', recognition_operation)
        assert metric.passed, f"Throughput requirement failed: {metric.value} ops/sec < {metric.threshold} ops/sec"
    
    @pytest.mark.performance
    def test_memory_usage_limits(self, performance_framework):
        """Test memory usage stays within limits."""
        def memory_intensive_operation():
            # Simulate operation that might use memory
            data = [i for i in range(10000)]
            return len(data)
        
        metrics = performance_framework.measure_resource_usage('memory_test', memory_intensive_operation)
        memory_metric = next(m for m in metrics if 'memory' in m.name)
        
        assert memory_metric.passed, f"Memory usage exceeded limit: {memory_metric.value}MB > {memory_metric.threshold}MB"
```

## 📋 **Continuous Integration Configuration**

### **GitHub Actions Workflow**
```yaml
name: Comprehensive Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=xml --cov-report=html
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v --tb=short
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: integration-test-results
        path: test-results/

  performance-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
    
    - name: Run performance tests
      run: |
        pytest tests/performance/ -v -m performance
    
    - name: Generate performance report
      run: |
        python scripts/testing/generate_performance_report.py

  security-tests:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
        pip install bandit safety
    
    - name: Run security scans
      run: |
        bandit -r src/ -f json -o security-report.json
        safety check --json --output safety-report.json
        pytest tests/security/ -v
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          security-report.json
          safety-report.json

  e2e-tests:
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
    
    - name: Run end-to-end tests
      run: |
        pytest tests/e2e/ -v --tb=short
      env:
        E2E_TEST_MODE: true
    
    - name: Upload E2E test artifacts
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: e2e-test-artifacts
        path: test-artifacts/

  quality-gates:
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests, performance-tests, security-tests]
    
    steps:
    - name: Download all artifacts
      uses: actions/download-artifact@v3
    
    - name: Evaluate quality gates
      run: |
        python scripts/testing/evaluate_quality_gates.py
      env:
        MIN_COVERAGE: 95
        MAX_PERFORMANCE_REGRESSION: 5
        SECURITY_ISSUES_ALLOWED: 0
```

## 📋 **Acceptance Criteria**

### **Test Coverage Requirements**
- [ ] 95%+ code coverage across all modules
- [ ] 100% coverage for critical pipeline components
- [ ] Unit tests for all public functions and classes
- [ ] Integration tests for all component interactions

### **Performance Requirements**
- [ ] Doorbell trigger latency < 100ms
- [ ] Face recognition throughput > 10 faces/second
- [ ] Memory usage < 512MB under normal load
- [ ] System startup time < 5 seconds

### **Quality Requirements**
- [ ] Zero critical security vulnerabilities
- [ ] All performance regression tests pass
- [ ] End-to-end scenarios complete successfully
- [ ] Automated CI/CD pipeline with quality gates

### **Documentation Requirements**
- [ ] Comprehensive testing documentation
- [ ] Performance benchmarking reports
- [ ] Security testing protocols
- [ ] CI/CD setup and maintenance guides

---

**This issue establishes a comprehensive testing framework that ensures quality, performance, and security across the entire pipeline system, providing confidence for production deployment.**