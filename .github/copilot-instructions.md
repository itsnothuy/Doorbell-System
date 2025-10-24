# GitHub Copilot Instructions

## 🤖 Project Context

You are working on the **Doorbell Security System**, an AI-powered privacy-first security solution inspired by **Frigate NVR's architecture**. The system uses a sophisticated **modular monolith** design with **pipeline processing**, **event-driven architecture**, and **multi-processing** to perform local face recognition at doorbell events.

## 🎯 Project Overview

### Core Architecture (Frigate-Inspired)
- **Modular Monolith**: Single container with loosely-coupled internal modules
- **Pipeline Processing**: Frame capture → Motion detection → Face detection → Recognition → Event processing
- **Producer-Consumer**: Event-driven processing with high-performance queues
- **Multi-Processing**: Parallel workers for CPU-intensive face recognition
- **Strategy Pattern**: Pluggable face detection backends (CPU, GPU, EdgeTPU)
- **ZeroMQ-like Communication**: High-performance inter-process messaging
- **Privacy-First**: All biometric processing happens locally, no cloud dependencies

### Technical Stack
- **Language**: Python 3.11+
- **AI/ML**: face_recognition, OpenCV, dlib with pluggable backends
- **Architecture**: Multi-threaded pipeline with worker pools
- **Communication**: Queue-based messaging (inspired by ZeroMQ)
- **Hardware**: RPi.GPIO, PiCamera2 (Raspberry Pi)
- **Web**: Flask API with real-time event streaming
- **Messaging**: python-telegram-bot
- **Storage**: SQLite for event tracking
- **Deployment**: Docker, Docker Compose
- **CI/CD**: GitHub Actions

## 📁 Codebase Structure

```
doorbell-system/
├── src/                              # Core pipeline modules
│   ├── pipeline/                     # Main processing pipeline
│   │   ├── orchestrator.py           # Main pipeline orchestrator (like Frigate's main)
│   │   ├── frame_capture.py          # Frame capture worker
│   │   ├── motion_detector.py        # Motion detection pipeline stage
│   │   ├── face_detector.py          # Face detection worker pool
│   │   ├── face_recognizer.py        # Face recognition worker
│   │   └── event_processor.py        # Event processing and enrichment
│   ├── detectors/                    # Detection strategy implementations
│   │   ├── __init__.py
│   │   ├── base_detector.py          # Abstract detector interface
│   │   ├── cpu_detector.py           # CPU-based face detection
│   │   ├── gpu_detector.py           # GPU-accelerated detection
│   │   └── edgetpu_detector.py       # Coral EdgeTPU detection
│   ├── communication/               # Inter-process communication
│   │   ├── __init__.py
│   │   ├── message_bus.py            # ZeroMQ-like message queue
│   │   ├── events.py                 # Event definitions and handlers
│   │   └── queues.py                 # Queue management
│   ├── hardware/                     # Hardware abstraction layer
│   │   ├── camera_handler.py         # Cross-platform camera abstraction
│   │   └── gpio_handler.py           # Hardware GPIO interface
│   ├── storage/                      # Data persistence layer
│   │   ├── event_database.py         # Event storage (SQLite)
│   │   └── face_database.py          # Face encoding management
│   ├── enrichment/                   # Event enrichment processors
│   │   ├── telegram_notifier.py      # Telegram notification enrichment
│   │   └── web_events.py             # Web interface event streaming
│   ├── web_interface.py              # Flask web application
│   └── platform_detector.py         # Platform detection utilities
├── config/                           # Configuration management
│   ├── settings.py                   # Main configuration loader
│   ├── pipeline_config.py           # Pipeline-specific configuration
│   ├── detector_config.py           # Detector strategy configuration
│   └── logging_config.py            # Logging configuration
├── data/                            # Data storage directories
│   ├── events.db                    # SQLite event database
│   ├── known_faces/                 # Known person face encodings
│   ├── blacklist_faces/             # Blacklisted person encodings
│   └── logs/                        # Application logs
└── deployment files                 # Docker, CI/CD, configuration
```

## 🔧 Key Components

### 1. Pipeline Orchestrator (`src/pipeline/orchestrator.py`)
**Purpose**: Main system coordinator inspired by Frigate's architecture
**Key Features**:
- Multi-process pipeline management
- Event-driven workflow coordination
- Worker pool orchestration
- Queue management and monitoring
- Graceful shutdown and recovery

### 2. Frame Capture Worker (`src/pipeline/frame_capture.py`)
**Purpose**: High-performance frame capture with ring buffer
**Key Features**:
- Ring buffer for continuous frame capture
- GPIO event integration
- Multi-threaded capture handling
- Frame preprocessing and optimization
- Hardware abstraction layer

### 3. Face Detection Strategy (`src/detectors/`)
**Purpose**: Pluggable face detection backends (like Frigate's detector plugins)
**Key Features**:
- Abstract detector interface
- CPU/GPU/EdgeTPU implementations
- Performance benchmarking
- Hardware-specific optimizations
- Fallback mechanisms

### 4. Face Recognition Worker (`src/pipeline/face_recognizer.py`)
**Purpose**: Multi-process face recognition engine
**Key Features**:
- Worker pool for parallel processing
- Face encoding comparison
- Known faces and blacklist management
- Performance monitoring
- Queue-based job processing

### 5. Event Processor (`src/pipeline/event_processor.py`)
**Purpose**: Event lifecycle management and enrichment
**Key Features**:
- Event state machine
- Multi-subscriber event broadcasting
- Enrichment pipeline coordination
- Database persistence
- Performance metrics

### 6. Message Bus (`src/communication/message_bus.py`)
**Purpose**: High-performance inter-process communication
**Key Features**:
- ZeroMQ-inspired messaging
- Event publishing and subscription
- Priority queue management
- Message serialization
- Connection pooling

## 🏗️ Architectural Patterns

### Frigate-Inspired Design Patterns
- **Modular Monolith**: Single process with loosely-coupled internal modules
- **Pipeline Architecture**: Sequential processing stages with clear data flow
- **Producer-Consumer**: High-performance queue-based worker systems
- **Strategy Pattern**: Pluggable detector backends (CPU/GPU/EdgeTPU)
- **Observer/Publisher-Subscriber**: Event broadcasting with multiple consumers
- **Worker Pool**: Multi-threaded/multi-process job processing
- **Circuit Breaker**: Fault tolerance and graceful degradation
- **State Machine**: Event lifecycle management

### Key Principles
- **High Performance**: Optimized for real-time video processing
- **Scalability**: Multi-process architecture for parallel processing
- **Modularity**: Clear separation of concerns with defined interfaces
- **Privacy-First**: Local processing only, no cloud dependencies
- **Resource Optimization**: Designed for edge devices and Pi hardware
- **Fault Tolerance**: Graceful handling of hardware failures
- **Extensibility**: Plugin architecture for new detectors and enrichments

## 💡 Development Guidelines

### Code Style
- Follow PEP 8 with 100-character line limit
- Use type hints for all public functions
- Google-style docstrings for documentation
- Comprehensive error handling with specific exceptions
- Logging at appropriate levels (DEBUG, INFO, WARNING, ERROR)

### Security Best Practices
- Input validation and sanitization for all user inputs
- No hardcoded secrets or credentials
- Secure file permissions for sensitive data
- Rate limiting for API endpoints
- CSRF protection for web forms

### Performance Considerations
- Face recognition optimization for Pi hardware
- Memory management for long-running processes
- Caching strategies for repeated operations
- Threading for non-blocking operations
- Resource cleanup and lifecycle management

### Testing Requirements
- Unit tests for all core components
- Integration tests for component interactions
- Mock implementations for hardware dependencies
- Performance tests for face recognition
- Security tests for input validation

## 🎯 Common Tasks and Patterns

### Adding New Face Recognition Features
```python
class FaceManager:
    def new_recognition_feature(self, image: np.ndarray) -> Dict[str, Any]:
        """Template for new recognition features."""
        try:
            # Validate input
            if image is None or image.size == 0:
                raise ValueError("Invalid image")
            
            # Process image
            result = self._process_image(image)
            
            # Log operation
            logger.info(f"Recognition feature completed: {result['status']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Recognition feature failed: {e}")
            raise
```

### Adding New Hardware Interfaces
```python
class NewHardwareHandler:
    def __init__(self):
        self.is_available = platform_detector.has_hardware()
        
    def initialize(self) -> bool:
        """Initialize hardware with error handling."""
        try:
            if not self.is_available:
                logger.warning("Hardware not available, using mock")
                return False
            
            # Initialize hardware
            return True
            
        except Exception as e:
            logger.error(f"Hardware initialization failed: {e}")
            return False
```

### Adding New Web API Endpoints
```python
@app.route('/api/new_feature', methods=['POST'])
def api_new_feature():
    """Template for new API endpoints."""
    try:
        # Validate input
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Process request
        result = process_feature(data)
        
        # Return response
        return jsonify({"status": "success", "data": result})
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": "Internal server error"}), 500
```

#### Adding New Event Enrichment Processors
```python
from src.enrichment.base_enrichment import BaseEnrichment
from src.communication.events import EventData

class NewEnrichmentProcessor(BaseEnrichment):
    """New enrichment processor for event enhancement."""
    
    def __init__(self, message_bus: MessageBus, config: Dict[str, Any]):
        super().__init__(message_bus, config)
        self.priority = config.get('priority', 5)
        
    def enrich_event(self, event: EventData) -> EventData:
        """Enrich event with additional data."""
        try:
            # Add enrichment data
            enriched_data = self._add_enrichment(event.data)
            
            # Create enriched event
            enriched_event = EventData(
                event_id=event.event_id,
                event_type=event.event_type,
                data=enriched_data,
                timestamp=event.timestamp,
                enrichments=event.enrichments + [self.__class__.__name__]
            )
            
            logger.debug(f"Event {event.event_id} enriched by {self.__class__.__name__}")
            return enriched_event
            
        except Exception as e:
            logger.error(f"Event enrichment failed: {e}")
            return event  # Return original event on failure
```

#### Worker Pool Management
```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.communication.queues import JobQueue

class WorkerPoolManager:
    """Manage worker pools for CPU-intensive tasks."""
    
    def __init__(self, max_workers: int, job_queue: JobQueue):
        self.max_workers = max_workers
        self.job_queue = job_queue
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self.running = False
        
    def start_workers(self) -> None:
        """Start worker pool processing."""
        self.running = True
        while self.running:
            try:
                # Get job from queue
                job = self.job_queue.get(timeout=1.0)
                
                # Submit job to worker pool
                future = self.executor.submit(self._process_job, job)
                
                # Handle completion
                future.add_done_callback(self._job_completed)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker pool error: {e}")
                
    def _process_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual job in worker."""
        # Job processing logic
        return processed_result
        
    def _job_completed(self, future):
        """Handle job completion."""
        try:
            result = future.result()
            # Handle successful completion
        except Exception as e:
            logger.error(f"Job failed: {e}")
```

## 🔍 Pipeline Debugging and Monitoring

### Pipeline Health Monitoring
```python
from src.communication.message_bus import MessageBus
from src.pipeline.metrics import PipelineMetrics

def check_pipeline_health() -> Dict[str, Any]:
    """Check overall pipeline health and performance."""
    metrics = PipelineMetrics()
    
    return {
        "frame_capture": {
            "fps": metrics.get_capture_fps(),
            "queue_size": metrics.get_capture_queue_size(),
            "dropped_frames": metrics.get_dropped_frame_count()
        },
        "face_detection": {
            "avg_inference_time": metrics.get_avg_detection_time(),
            "active_workers": metrics.get_active_detection_workers(),
            "queue_backlog": metrics.get_detection_queue_size()
        },
        "face_recognition": {
            "avg_recognition_time": metrics.get_avg_recognition_time(),
            "cache_hit_rate": metrics.get_recognition_cache_hit_rate(),
            "worker_utilization": metrics.get_recognition_worker_utilization()
        },
        "event_processing": {
            "events_per_minute": metrics.get_event_rate(),
            "enrichment_latency": metrics.get_enrichment_latency(),
            "notification_success_rate": metrics.get_notification_success_rate()
        }
    }
```

### Common Pipeline Issues

#### Performance Bottlenecks
- **High queue backlog**: Increase worker pool size or optimize processing
- **Memory pressure**: Implement better cleanup and resource management
- **CPU saturation**: Balance workload across available cores

#### Pipeline Failures
- **Stage disconnection**: Check message bus connectivity
- **Worker crashes**: Review error logs and implement circuit breakers
- **Queue overflow**: Implement backpressure and queue size limits

## 📝 Documentation Standards

### Code Documentation
- Use clear, descriptive function and variable names
- Add docstrings for all public functions and classes
- Include type hints for parameters and return values
- Comment complex logic and business rules

### Commit Messages
Use conventional commit format:
- `feat:` - New features
- `fix:` - Bug fixes  
- `docs:` - Documentation updates
- `test:` - Test additions/modifications
- `refactor:` - Code refactoring
- `perf:` - Performance improvements

### Pull Request Guidelines
- Clear description of changes and motivation
- Reference related issues
- Include test coverage for new features
- Update documentation as needed
- Verify CI passes before requesting review

## 🎯 Implementation Roadmap

### Phase 1: Foundation (PRs 1-3)
1. **PR #1**: Core Communication Infrastructure
   - Implement `src/communication/message_bus.py`
   - Implement `src/communication/events.py`
   - Implement `src/communication/queues.py`
   - Add comprehensive unit tests

2. **PR #2**: Base Detector Framework
   - Implement `src/detectors/base_detector.py`
   - Implement `src/detectors/__init__.py`
   - Create detector factory pattern
   - Add detector performance benchmarking

3. **PR #3**: Pipeline Configuration System
   - Enhance `config/pipeline_config.py`
   - Add environment-specific configs
   - Create configuration validation
   - Add configuration hot-reloading

### Phase 2: Core Pipeline (PRs 4-8)
4. **PR #4**: Frame Capture Worker
   - Implement `src/pipeline/frame_capture.py`
   - Ring buffer implementation
   - GPIO integration
   - Camera abstraction layer

5. **PR #5**: Motion Detection Worker
   - Implement `src/pipeline/motion_detector.py`
   - Background subtraction algorithms
   - Motion region analysis
   - Performance optimization

6. **PR #6**: Face Detection Worker Pool
   - Implement `src/pipeline/face_detector.py`
   - Multi-process worker management
   - Queue-based job distribution
   - Hardware-specific optimizations

7. **PR #7**: Face Recognition Engine
   - Implement `src/pipeline/face_recognizer.py`
   - Face encoding and matching
   - Database integration
   - Caching mechanisms

8. **PR #8**: Event Processing System
   - Implement `src/pipeline/event_processor.py`
   - Event enrichment pipeline
   - Notification routing
   - Database persistence

### Phase 3: Hardware & Storage (PRs 9-11)
9. **PR #9**: Hardware Abstraction Layer
   - Refactor `src/camera_handler.py` → `src/hardware/camera_handler.py`
   - Refactor `src/gpio_handler.py` → `src/hardware/gpio_handler.py`
   - Platform-specific implementations
   - Mock hardware for testing

10. **PR #10**: Storage Layer
    - Implement `src/storage/event_database.py`
    - Implement `src/storage/face_database.py`
    - Database migrations
    - Data retention policies

11. **PR #11**: Enrichment Processors
    - Refactor `src/telegram_notifier.py` → `src/enrichment/telegram_notifier.py`
    - Implement `src/enrichment/web_events.py`
    - Plugin architecture for enrichments
    - Rate limiting and filtering

### Phase 4: Integration & Orchestration (PRs 12-14)
12. **PR #12**: Pipeline Orchestrator
    - Implement `src/pipeline/orchestrator.py`
    - Worker lifecycle management
    - Health monitoring
    - Graceful shutdown handling

13. **PR #13**: Detector Implementations
    - Implement `src/detectors/cpu_detector.py`
    - Implement `src/detectors/gpu_detector.py`
    - Implement `src/detectors/edgetpu_detector.py`
    - Performance benchmarking

14. **PR #14**: Main Application Integration
    - Update `app.py` to use pipeline orchestrator
    - Backward compatibility layer
    - Migration tools from old architecture
    - Integration testing

### Phase 5: Testing & Documentation (PRs 15-16)
15. **PR #15**: Comprehensive Testing
    - Pipeline integration tests
    - Performance test suite
    - Hardware mock testing
    - Load testing framework

16. **PR #16**: Production Readiness
    - Docker optimization
    - Monitoring and metrics
    - Production configuration
    - Deployment automation

## 🎯 Feature Development Workflow

### 1. Planning Phase
- Create GitHub issue with detailed requirements
- Review architectural implications
- Plan testing strategy
- Consider security implications

### 2. Implementation Phase
- Create feature branch from develop
- Implement core functionality
- Add comprehensive tests
- Update documentation

### 3. Integration Phase
- Test with existing components
- Verify cross-platform compatibility
- Run full test suite
- Performance testing if applicable

### 4. Review Phase
- Code review with security focus
- Test coverage verification
- Documentation review
- CI/CD pipeline validation

## 🔐 Security Considerations

### Data Protection
- All face encodings stored encrypted
- Temporary images automatically cleaned up
- No biometric data transmitted to external services
- Secure file permissions for sensitive data

### Input Validation
- Validate all user inputs (web forms, API calls)
- Sanitize file names and paths
- Check image format and size limits
- Prevent path traversal attacks

### Authentication & Authorization
- Implement authentication for web interface
- Use CSRF tokens for state-changing operations
- Rate limiting for API endpoints
- Secure session management

## 🚀 Performance Optimization

### Face Recognition Optimization
- Use HOG model for CPU efficiency
- Implement encoding caching
- Optimize image preprocessing
- Batch processing where possible

### Memory Management
- Regular cleanup of temporary files
- Limit concurrent processing
- Monitor memory usage
- Implement garbage collection triggers

### Resource Management
- Proper cleanup of camera resources
- GPIO cleanup on shutdown
- Thread pool management
- Connection pooling for external services

---

## 📋 Quick Reference

### Key Files to Understand
1. `src/doorbell_security.py` - Main system orchestrator
2. `src/face_manager.py` - Face recognition engine
3. `config/settings.py` - Configuration management
4. `tests/conftest.py` - Test configuration and fixtures

### Important Configuration
- Face recognition tolerance: 0.6 (adjustable)
- Debounce time: 5 seconds (prevents spam)
- Image retention: 7 days (configurable)
- Log rotation: 10MB files, 5 backups

### Development Commands
```bash
# Setup development environment
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest tests/ -v --cov=src

# Start development server
python app.py

# Run with Docker
docker-compose up --build
```

## 🔧 Coding Agent Guidelines

### Implementation Principles
- **Single Responsibility**: Each module has one clear purpose
- **Dependency Injection**: Pass dependencies through constructors
- **Error Handling**: Comprehensive exception handling with specific error types
- **Logging**: Structured logging with appropriate levels
- **Testing**: Write tests alongside implementation
- **Documentation**: Include docstrings and type hints

### Code Templates

#### Pipeline Worker Template
```python
#!/usr/bin/env python3
"""
Pipeline Worker Template

Base template for implementing pipeline workers in the Frigate-inspired architecture.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from src.communication.message_bus import MessageBus
from src.communication.events import PipelineEvent, EventType

logger = logging.getLogger(__name__)


class PipelineWorker:
    """Base class for pipeline workers."""
    
    def __init__(self, message_bus: MessageBus, config: Dict[str, Any]):
        self.message_bus = message_bus
        self.config = config
        self.worker_id = f"{self.__class__.__name__}_{int(time.time())}"
        
        # Worker state
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Performance metrics
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None
        
        # Subscribe to input events
        self._setup_subscriptions()
        
        logger.info(f"Initialized {self.worker_id}")
    
    def _setup_subscriptions(self) -> None:
        """Setup message bus subscriptions."""
        # Override in subclasses
        pass
    
    def start(self) -> None:
        """Start the worker."""
        if self.running:
            logger.warning(f"{self.worker_id} already running")
            return
        
        self.running = True
        self.start_time = time.time()
        
        try:
            self._initialize_worker()
            self._worker_loop()
        except Exception as e:
            logger.error(f"{self.worker_id} failed: {e}")
            raise
        finally:
            self._cleanup_worker()
    
    def stop(self) -> None:
        """Stop the worker gracefully."""
        logger.info(f"Stopping {self.worker_id}...")
        self.running = False
        self.shutdown_event.set()
    
    def _initialize_worker(self) -> None:
        """Initialize worker-specific resources."""
        pass
    
    def _worker_loop(self) -> None:
        """Main worker processing loop."""
        while self.running and not self.shutdown_event.is_set():
            try:
                # Process events or perform work
                self._process_iteration()
                
                # Brief sleep to prevent busy waiting
                time.sleep(0.01)
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"{self.worker_id} processing error: {e}")
                time.sleep(0.1)  # Longer sleep on error
    
    def _process_iteration(self) -> None:
        """Override this method for worker-specific processing."""
        pass
    
    def _cleanup_worker(self) -> None:
        """Cleanup worker resources."""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get worker performance metrics."""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        return {
            'worker_id': self.worker_id,
            'running': self.running,
            'uptime_seconds': uptime,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(1, self.processed_count),
            'processing_rate': self.processed_count / max(1, uptime)
        }
```

#### Event Handler Template
```python
def handle_event(self, message: Message) -> None:
    """Handle incoming pipeline event."""
    try:
        event = message.data
        logger.debug(f"Processing event {event.event_id} of type {event.event_type}")
        
        # Validate event
        if not self._validate_event(event):
            logger.warning(f"Invalid event {event.event_id}")
            return
        
        # Process event
        result = self._process_event(event)
        
        # Publish result if needed
        if result:
            self.message_bus.publish('output_topic', result)
        
        self.processed_count += 1
        
    except Exception as e:
        self.error_count += 1
        logger.error(f"Event handling failed: {e}")
        
        # Publish error event
        error_event = PipelineEvent(
            event_type=EventType.COMPONENT_ERROR,
            data={'error': str(e), 'original_event': event.event_id},
            source=self.worker_id
        )
        self.message_bus.publish('error_events', error_event)
```

#### Configuration Loading Template
```python
@dataclass
class ComponentConfig:
    """Configuration for a pipeline component."""
    enabled: bool = True
    worker_count: int = 1
    queue_size: int = 100
    timeout: float = 10.0
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ComponentConfig':
        """Create config from dictionary with validation."""
        # Validate required fields
        validated_config = {}
        
        for field in cls.__dataclass_fields__:
            field_type = cls.__dataclass_fields__[field].type
            default_value = cls.__dataclass_fields__[field].default
            
            value = config_dict.get(field, default_value)
            
            # Type validation
            if not isinstance(value, field_type):
                try:
                    value = field_type(value)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid {field} value: {value}, using default")
                    value = default_value
            
            validated_config[field] = value
        
        return cls(**validated_config)
```

### Testing Templates

#### Unit Test Template
```python
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

from src.pipeline.example_worker import ExampleWorker
from src.communication.message_bus import MessageBus
from src.communication.events import PipelineEvent, EventType


class TestExampleWorker:
    """Test suite for ExampleWorker."""
    
    @pytest.fixture
    def mock_message_bus(self):
        """Create mock message bus."""
        bus = Mock(spec=MessageBus)
        bus.subscribe = Mock()
        bus.publish = Mock()
        return bus
    
    @pytest.fixture
    def worker_config(self):
        """Create test configuration."""
        return {
            'enabled': True,
            'worker_count': 1,
            'queue_size': 10,
            'timeout': 5.0
        }
    
    @pytest.fixture
    def worker(self, mock_message_bus, worker_config):
        """Create test worker instance."""
        return ExampleWorker(mock_message_bus, worker_config)
    
    def test_worker_initialization(self, worker, mock_message_bus):
        """Test worker initializes correctly."""
        assert worker.message_bus == mock_message_bus
        assert not worker.running
        assert worker.processed_count == 0
        
        # Verify subscription setup
        mock_message_bus.subscribe.assert_called()
    
    def test_event_processing(self, worker):
        """Test event processing logic."""
        # Create test event
        test_event = PipelineEvent(
            event_type=EventType.FRAME_CAPTURED,
            data={'test': 'data'}
        )
        
        # Process event
        result = worker._process_event(test_event)
        
        # Verify result
        assert result is not None
        assert worker.processed_count > 0
    
    def test_error_handling(self, worker):
        """Test error handling in event processing."""
        # Create invalid event
        invalid_event = None
        
        # Should handle gracefully
        with pytest.raises(Exception):
            worker._process_event(invalid_event)
        
        assert worker.error_count > 0
    
    @pytest.mark.asyncio
    async def test_worker_lifecycle(self, worker):
        """Test worker start/stop lifecycle."""
        # Start worker in background
        import threading
        worker_thread = threading.Thread(target=worker.start)
        worker_thread.start()
        
        # Wait a bit
        await asyncio.sleep(0.1)
        
        # Verify running
        assert worker.running
        
        # Stop worker
        worker.stop()
        worker_thread.join(timeout=1.0)
        
        # Verify stopped
        assert not worker.running
```

### Performance Monitoring Template
```python
class PerformanceMonitor:
    """Monitor pipeline performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def record_metric(self, component: str, metric_name: str, value: float):
        """Record a performance metric."""
        if component not in self.metrics:
            self.metrics[component] = {}
        
        if metric_name not in self.metrics[component]:
            self.metrics[component][metric_name] = []
        
        self.metrics[component][metric_name].append({
            'value': value,
            'timestamp': time.time()
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {
            'uptime': time.time() - self.start_time,
            'components': {}
        }
        
        for component, metrics in self.metrics.items():
            component_summary = {}
            
            for metric_name, values in metrics.items():
                recent_values = [v['value'] for v in values[-100:]]  # Last 100 values
                
                component_summary[metric_name] = {
                    'current': recent_values[-1] if recent_values else 0,
                    'average': sum(recent_values) / len(recent_values) if recent_values else 0,
                    'min': min(recent_values) if recent_values else 0,
                    'max': max(recent_values) if recent_values else 0,
                    'count': len(values)
                }
            
            summary['components'][component] = component_summary
        
        return summary
```