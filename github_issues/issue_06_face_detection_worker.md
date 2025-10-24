# GitHub Issue: Implement Face Detection Worker Pool with Strategy Pattern

## 📋 Overview

### Phase Information
- **Phase**: 2 - Core Pipeline Workers
- **PR Number**: #6
- **Complexity**: High
- **Estimated Duration**: 5-6 days
- **Dependencies**: Detector implementations, Frame capture worker, Message bus system
- **Priority**: High (Critical AI/ML component)

### Goals
Implement a multi-process face detection worker pool with pluggable detector strategies, load balancing, and optimized performance for real-time face detection in doorbell security applications.

## 🎯 Requirements

### Functional Requirements

#### Core Functionality
- **Multi-Process Worker Pool**: 2-4 worker processes for parallel face detection
- **Detector Strategy Selection**: Dynamic selection based on hardware capabilities (CPU/GPU/EdgeTPU)
- **Queue-Based Job Distribution**: Priority queue system for detection jobs
- **Load Balancing**: Intelligent distribution of work across available workers
- **Performance Monitoring**: Real-time metrics collection for worker utilization and detection performance
- **Error Handling**: Automatic worker recovery and job retry mechanisms

#### Detection Processing
- **Frame Event Processing**: Subscribe to `frame_captured` events from frame capture worker
- **Face Detection Results**: Publish `faces_detected` events with detection metadata
- **Batch Processing**: Support for processing multiple faces in single frame
- **Result Aggregation**: Combine results from multiple workers efficiently

### Non-Functional Requirements

#### Performance Targets
- **Detection Latency**: <500ms per frame on Raspberry Pi 4
- **Throughput**: >5 frames/second sustained processing
- **Accuracy**: >95% face detection rate (precision and recall)
- **Worker Utilization**: >80% under normal load
- **Memory Usage**: <200MB per worker process
- **Startup Time**: <5 seconds for full worker pool initialization

#### Reliability Requirements
- **Process Isolation**: Worker failures don't affect other workers or main process
- **Automatic Recovery**: Failed workers automatically restart
- **Job Retry**: Failed detection jobs retry with exponential backoff
- **Graceful Degradation**: System continues operating with reduced worker count
- **Resource Management**: Proper cleanup of worker processes and shared resources

## 🔧 Implementation Specifications

### Files to Create/Modify

#### New Files
```
src/pipeline/face_detector.py       # Main face detection worker pool
src/detectors/cpu_detector.py       # CPU-based face detection implementation
src/detectors/detector_factory.py   # Factory for creating detector instances
src/detectors/detection_result.py   # Face detection result data structures
tests/test_face_detector.py         # Comprehensive unit tests
tests/test_detector_factory.py      # Detector factory tests
tests/performance/detection_bench.py # Performance benchmarking suite
tests/mocks/mock_detector.py        # Mock detector for testing
```

#### Modified Files
```
src/detectors/base_detector.py      # Enhanced base detector interface
config/pipeline_config.py           # Add face detection configuration
```

### Architecture Patterns

#### Multi-Process Worker Pool Pattern
```python
class FaceDetectionWorker(PipelineWorker):
    """Face detection worker pool with strategy pattern."""
    
    def __init__(self, message_bus: MessageBus, config: Dict[str, Any]):
        super().__init__(message_bus, config)
        
        # Worker pool configuration
        self.worker_count = config.get('worker_count', 2)
        self.detector_type = config.get('detector_type', 'cpu')
        
        # Process pool and job management
        self.worker_pool = ProcessPoolExecutor(max_workers=self.worker_count)
        self.pending_jobs = {}
        self.job_queue = PriorityQueue()
        
        # Performance monitoring
        self.detection_metrics = DetectionMetrics()
        
    def _setup_subscriptions(self):
        self.message_bus.subscribe('frame_captured', self.handle_frame_event, self.worker_id)
        self.message_bus.subscribe('worker_health_check', self.handle_health_check, self.worker_id)
```

#### Detector Strategy Factory
```python
class DetectorFactory:
    """Factory for creating face detector instances."""
    
    _detectors = {
        'cpu': CPUDetector,
        'gpu': GPUDetector,
        'edgetpu': EdgeTPUDetector,
        'mock': MockDetector
    }
    
    @classmethod
    def create(cls, detector_type: str, config: Dict[str, Any]) -> BaseDetector:
        """Create detector instance with hardware validation."""
        if detector_type not in cls._detectors:
            logger.warning(f"Unknown detector type: {detector_type}, falling back to CPU")
            detector_type = 'cpu'
        
        detector_class = cls._detectors[detector_type]
        
        # Validate hardware availability
        if not detector_class.is_available():
            logger.warning(f"{detector_type} detector not available, falling back to CPU")
            detector_class = cls._detectors['cpu']
        
        return detector_class(config)
```

### Core Implementation

#### Face Detection Worker Pool
```python
#!/usr/bin/env python3
"""
Face Detection Worker Pool

Multi-process face detection worker with strategy pattern, load balancing,
and performance optimization for real-time face detection.
"""

import time
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import PriorityQueue, Empty
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from src.pipeline.base_worker import PipelineWorker
from src.communication.message_bus import MessageBus, Message
from src.communication.events import PipelineEvent, EventType, FrameEvent, FaceDetectionEvent
from src.detectors.detector_factory import DetectorFactory
from src.detectors.detection_result import FaceDetectionResult, DetectionMetrics

logger = logging.getLogger(__name__)


class DetectionJob:
    """Detection job with priority and metadata."""
    
    def __init__(self, priority: int, frame_event: FrameEvent, timestamp: float):
        self.priority = priority
        self.frame_event = frame_event
        self.timestamp = timestamp
        self.job_id = f"{frame_event.event_id}_{int(timestamp * 1000)}"
    
    def __lt__(self, other):
        return self.priority < other.priority


class FaceDetectionWorker(PipelineWorker):
    """Multi-process face detection worker pool."""
    
    def __init__(self, message_bus: MessageBus, config: Dict[str, Any]):
        super().__init__(message_bus, config)
        
        # Configuration
        self.worker_count = config.get('worker_count', 2)
        self.detector_type = config.get('detector_type', 'cpu')
        self.max_queue_size = config.get('max_queue_size', 100)
        self.job_timeout = config.get('job_timeout', 30.0)
        
        # Worker pool and job management
        self.worker_pool = None
        self.pending_jobs = {}
        self.job_queue = PriorityQueue(maxsize=self.max_queue_size)
        
        # Performance monitoring
        self.detection_count = 0
        self.detection_errors = 0
        self.total_detection_time = 0.0
        self.worker_metrics = {}
        
        logger.info(f"Initialized {self.worker_id} with {self.worker_count} workers, detector: {self.detector_type}")
    
    def _setup_subscriptions(self):
        """Setup message bus subscriptions."""
        self.message_bus.subscribe('frame_captured', self.handle_frame_event, self.worker_id)
        self.message_bus.subscribe('worker_health_check', self.handle_health_check, self.worker_id)
        self.message_bus.subscribe('system_shutdown', self.handle_shutdown, self.worker_id)
        logger.debug(f"{self.worker_id} subscriptions configured")
    
    def _initialize_worker(self):
        """Initialize worker pool and detector strategy."""
        try:
            # Validate detector availability
            detector_class = DetectorFactory.get_detector_class(self.detector_type)
            if not detector_class.is_available():
                logger.warning(f"{self.detector_type} detector not available, falling back to CPU")
                self.detector_type = 'cpu'
            
            # Initialize worker pool
            self.worker_pool = ProcessPoolExecutor(
                max_workers=self.worker_count,
                mp_context=mp.get_context('spawn')  # Use spawn for better isolation
            )
            
            # Test detector creation
            test_detector = DetectorFactory.create(self.detector_type, self.config)
            test_detector.health_check()
            
            logger.info(f"{self.worker_id} initialized with {self.detector_type} detector")
            
        except Exception as e:
            logger.error(f"{self.worker_id} initialization failed: {e}")
            raise
    
    def handle_frame_event(self, message: Message):
        """Handle frame capture event and schedule detection."""
        frame_event = message.data
        
        try:
            # Determine job priority (doorbell events get higher priority)
            priority = 1 if 'doorbell' in frame_event.metadata.get('source', '') else 2
            
            # Create detection job
            detection_job = DetectionJob(
                priority=priority,
                frame_event=frame_event,
                timestamp=time.time()
            )
            
            # Add to queue (non-blocking with queue size limit)
            try:
                self.job_queue.put(detection_job, block=False)
                logger.debug(f"Queued detection job {detection_job.job_id}")
            except:
                logger.warning(f"Detection queue full, dropping frame {frame_event.event_id}")
                return
            
            # Process job queue
            self._process_job_queue()
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Frame event handling failed: {e}")
            self._handle_detection_error(e, frame_event)
    
    def _process_job_queue(self):
        """Process jobs from queue using worker pool."""
        while not self.job_queue.empty() and len(self.pending_jobs) < self.worker_count * 2:
            try:
                # Get next job
                detection_job = self.job_queue.get(block=False)
                
                # Check job age (drop old jobs)
                job_age = time.time() - detection_job.timestamp
                if job_age > self.job_timeout:
                    logger.warning(f"Dropping expired job {detection_job.job_id} (age: {job_age:.2f}s)")
                    continue
                
                # Submit to worker pool
                future = self.worker_pool.submit(
                    detect_faces_worker,
                    detection_job.frame_event.frame_data,
                    self.detector_type,
                    self.config,
                    detection_job.job_id
                )
                
                # Track pending job
                self.pending_jobs[detection_job.job_id] = {
                    'future': future,
                    'job': detection_job,
                    'submit_time': time.time()
                }
                
                # Add completion callback
                future.add_done_callback(
                    lambda f, job_id=detection_job.job_id: self._handle_detection_complete(f, job_id)
                )
                
                logger.debug(f"Submitted detection job {detection_job.job_id} to worker pool")
                
            except Empty:
                break
            except Exception as e:
                logger.error(f"Job processing failed: {e}")
    
    def _handle_detection_complete(self, future, job_id: str):
        """Handle completed detection job."""
        try:
            # Get job info
            job_info = self.pending_jobs.pop(job_id, None)
            if not job_info:
                logger.warning(f"Completed job {job_id} not found in pending jobs")
                return
            
            # Get results
            detection_result = future.result(timeout=1.0)
            
            # Update metrics
            processing_time = time.time() - job_info['submit_time']
            self.detection_count += 1
            self.total_detection_time += processing_time
            
            # Create face detection event
            face_detection_event = FaceDetectionEvent(
                event_id=job_info['job'].frame_event.event_id,
                frame_event_id=job_info['job'].frame_event.event_id,
                faces=detection_result.faces,
                detection_metadata={
                    'detector_type': self.detector_type,
                    'processing_time_ms': processing_time * 1000,
                    'face_count': len(detection_result.faces),
                    'confidence_scores': [face.confidence for face in detection_result.faces],
                    'detection_timestamp': time.time()
                },
                performance_metrics=detection_result.metrics
            )
            
            # Publish detection result
            self.message_bus.publish('faces_detected', face_detection_event)
            
            logger.debug(f"Detection completed for {job_id}: {len(detection_result.faces)} faces found")
            self.processed_count += 1
            
        except Exception as e:
            self.detection_errors += 1
            logger.error(f"Detection completion handling failed for {job_id}: {e}")
            
            # Publish error event
            if job_info:
                self._handle_detection_error(e, job_info['job'].frame_event)
    
    def _handle_detection_error(self, error: Exception, frame_event: FrameEvent):
        """Handle detection errors and publish error events."""
        error_event = PipelineEvent(
            event_type=EventType.COMPONENT_ERROR,
            data={
                'component': self.worker_id,
                'error': str(error),
                'error_type': type(error).__name__,
                'frame_event_id': frame_event.event_id,
                'detector_type': self.detector_type,
                'worker_metrics': self.get_metrics()
            },
            source=self.worker_id
        )
        
        self.message_bus.publish('detection_errors', error_event)
    
    def handle_health_check(self, message: Message):
        """Handle health check requests."""
        health_status = {
            'worker_id': self.worker_id,
            'detector_type': self.detector_type,
            'worker_count': self.worker_count,
            'queue_size': self.job_queue.qsize(),
            'pending_jobs': len(self.pending_jobs),
            'detection_rate': self.detection_count / max(1, time.time() - self.start_time),
            'error_rate': self.detection_errors / max(1, self.detection_count),
            'avg_processing_time': self.total_detection_time / max(1, self.detection_count)
        }
        
        health_event = PipelineEvent(
            event_type=EventType.HEALTH_CHECK_RESPONSE,
            data=health_status,
            source=self.worker_id
        )
        
        self.message_bus.publish('worker_health_responses', health_event)
    
    def _cleanup_worker(self):
        """Cleanup worker pool and resources."""
        try:
            # Cancel pending jobs
            for job_id, job_info in self.pending_jobs.items():
                job_info['future'].cancel()
            
            # Shutdown worker pool
            if self.worker_pool:
                self.worker_pool.shutdown(wait=True, timeout=10.0)
            
            # Clear job queue
            while not self.job_queue.empty():
                try:
                    self.job_queue.get(block=False)
                except Empty:
                    break
            
            logger.info(f"{self.worker_id} cleanup completed")
            
        except Exception as e:
            logger.error(f"{self.worker_id} cleanup failed: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get worker performance metrics."""
        base_metrics = super().get_metrics()
        
        detection_metrics = {
            'detection_count': self.detection_count,
            'detection_errors': self.detection_errors,
            'avg_detection_time': self.total_detection_time / max(1, self.detection_count),
            'detection_rate': self.detection_count / max(1, time.time() - self.start_time),
            'error_rate': self.detection_errors / max(1, self.detection_count),
            'queue_size': self.job_queue.qsize(),
            'pending_jobs': len(self.pending_jobs),
            'worker_count': self.worker_count,
            'detector_type': self.detector_type
        }
        
        return {**base_metrics, **detection_metrics}


def detect_faces_worker(frame_data: np.ndarray, detector_type: str, config: Dict[str, Any], job_id: str) -> 'DetectionResult':
    """Worker function for face detection (runs in separate process)."""
    try:
        # Create detector instance
        detector = DetectorFactory.create(detector_type, config)
        
        # Perform detection
        faces, metrics = detector.detect_faces(frame_data)
        
        return DetectionResult(
            job_id=job_id,
            faces=faces,
            metrics=metrics,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Face detection worker failed for {job_id}: {e}")
        return DetectionResult(
            job_id=job_id,
            faces=[],
            metrics=DetectionMetrics(),
            success=False,
            error=str(e)
        )


class DetectionResult:
    """Result from face detection worker."""
    
    def __init__(self, job_id: str, faces: List[FaceDetectionResult], 
                 metrics: DetectionMetrics, success: bool, error: str = None):
        self.job_id = job_id
        self.faces = faces
        self.metrics = metrics
        self.success = success
        self.error = error
```

### Configuration

#### Face Detection Configuration
```yaml
# config/pipeline_config.py - Face Detection Section
face_detection:
  enabled: true
  worker_count: 2
  detector_type: "cpu"
  max_queue_size: 100
  job_timeout: 30.0
  
  # Detector-specific configuration
  detectors:
    cpu:
      model: "hog"  # or "cnn"
      number_of_times_to_upsample: 1
      batch_size: 1
      
    gpu:
      model: "cnn"
      device: "cuda:0"
      batch_size: 4
      
    edgetpu:
      model_path: "/opt/models/face_detection.tflite"
      delegate: "edgetpu"
      
  # Performance tuning
  performance:
    priority_levels: 3
    doorbell_priority: 1
    continuous_priority: 2
    max_concurrent_jobs: 4
    
  # Error handling
  error_handling:
    max_retries: 3
    retry_delay_seconds: 1.0
    worker_restart_threshold: 10
```

## 🧪 Testing Requirements

### Unit Tests (>90% Coverage)

#### Core Functionality Tests
```python
class TestFaceDetectionWorker:
    """Comprehensive test suite for face detection worker."""
    
    def test_worker_pool_initialization(self):
        """Test worker pool initializes with correct configuration."""
        
    def test_frame_event_processing(self):
        """Test frame event handling and job queue management."""
        
    def test_priority_queue_behavior(self):
        """Test priority queue ordering and processing."""
        
    def test_job_timeout_handling(self):
        """Test expiration of old jobs in queue."""
        
    def test_worker_pool_load_balancing(self):
        """Test distribution of jobs across workers."""
        
    def test_detection_result_publishing(self):
        """Test publishing of face detection results."""
        
    def test_error_handling_and_recovery(self):
        """Test worker failure recovery and error propagation."""
        
    def test_graceful_shutdown(self):
        """Test proper cleanup of worker pool and resources."""
```

#### Performance Tests
```python
class TestFaceDetectionPerformance:
    """Performance and load testing for face detection."""
    
    def test_throughput_benchmark(self):
        """Benchmark detection throughput under load."""
        
    def test_latency_measurement(self):
        """Measure detection latency for various scenarios."""
        
    def test_memory_usage_per_worker(self):
        """Test memory usage of worker processes."""
        
    def test_concurrent_detection_performance(self):
        """Test performance with multiple concurrent detections."""
        
    def test_queue_backlog_handling(self):
        """Test behavior under queue backlog conditions."""
```

### Integration Tests

#### Message Bus Integration
```python
def test_end_to_end_detection_pipeline():
    """Test complete detection workflow with real message bus."""
    
def test_frame_capture_integration():
    """Test integration with frame capture worker."""
    
def test_detector_factory_integration():
    """Test integration with various detector implementations."""
```

### Test Coverage Requirements
- **Minimum Coverage**: 90%
- **Critical Path Coverage**: 100% (job processing, worker management, error handling)
- **Multi-processing Coverage**: 100%
- **Error Scenario Coverage**: 100%

## ✅ Acceptance Criteria

### Definition of Done
- [ ] **Multi-Process Architecture**: Worker pool operates with process isolation
- [ ] **Detector Strategy Pattern**: Pluggable detector implementations working
- [ ] **Performance Targets**: All benchmarks met on target hardware
- [ ] **Load Balancing**: Efficient distribution of work across workers
- [ ] **Error Recovery**: Automatic recovery from worker failures
- [ ] **Integration Testing**: Seamless integration with pipeline components
- [ ] **Documentation**: Comprehensive code and API documentation
- [ ] **Code Review**: Architecture review and approval

### Quality Gates
- [ ] **Process Safety**: Worker crashes don't affect main process
- [ ] **Memory Management**: No memory leaks in long-running workers
- [ ] **Resource Cleanup**: Proper cleanup of all worker processes
- [ ] **Error Handling**: All error scenarios handled gracefully
- [ ] **Performance Stability**: Consistent performance under load

### Performance Benchmarks
- [ ] **<500ms Detection Latency**: Per frame on Raspberry Pi 4
- [ ] **>5 FPS Throughput**: Sustained detection processing
- [ ] **>95% Detection Accuracy**: Precision and recall metrics
- [ ] **>80% Worker Utilization**: Under normal load conditions
- [ ] **<200MB Memory per Worker**: Process memory usage
- [ ] **<5 Second Startup**: Worker pool initialization time
- [ ] **<1% Job Loss Rate**: Under normal operating conditions

## 🏷️ Labels

`enhancement`, `pipeline`, `ai-ml`, `phase-2`, `priority-high`, `complexity-high`

## 📝 Implementation Notes

### Development Approach
1. **Start with Single Worker**: Implement core detection logic first
2. **Add Multi-Processing**: Implement worker pool with proper isolation
3. **Integrate Strategy Pattern**: Add detector factory and pluggable implementations
4. **Performance Optimization**: Optimize for target hardware and throughput
5. **Error Handling**: Add comprehensive error handling and recovery
6. **Load Testing**: Test under realistic load conditions

### Risk Mitigation
- **Process Management**: Use proven multi-processing patterns
- **Memory Management**: Regular memory monitoring and leak detection
- **Performance Issues**: Continuous benchmarking and optimization
- **Hardware Compatibility**: Test with various detector implementations

### Success Metrics
- All acceptance criteria met
- Performance targets achieved consistently
- Zero critical errors in production deployment
- Seamless integration with existing pipeline components