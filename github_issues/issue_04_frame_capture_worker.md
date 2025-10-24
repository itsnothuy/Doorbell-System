# GitHub Issue: Implement Frame Capture Worker with Ring Buffer

## 📋 Overview

### Phase Information
- **Phase**: 2 - Core Pipeline Workers
- **PR Number**: #4
- **Complexity**: Medium-High
- **Estimated Duration**: 4-5 days
- **Dependencies**: Message bus system, camera handler refactor
- **Priority**: High (Critical path component)

### Goals
---

## 🤖 **For Coding Agents: Auto-Close Setup**

### **Branch Naming Convention**
When implementing this issue, create your branch using one of these patterns:
- `issue-4/frame-capture-worker`
- `4-frame-capture-worker` 
- `issue-4/implement-frame-capture`

### **PR Creation**
The GitHub Action will automatically append `Closes #4` to your PR description when you follow the branch naming convention above. This ensures the issue closes automatically when your PR is merged to the default branch.

### **Manual Alternative**
If you prefer manual control, include one of these in your PR description:
```
Closes #4
Fixes #4
Resolves #4
```

---

**This issue implements frame capture with ring buffer architecture as the first stage of the Frigate-inspired pipeline, providing high-performance frame acquisition with GPIO event integration and hardware optimization for the doorbell security system.**

## 🎯 Requirements

### Functional Requirements

#### Core Functionality
- **Ring Buffer Implementation**: Continuous frame capture with configurable buffer size (30-60 frames)
- **GPIO Event Integration**: Doorbell press detection triggers frame capture bursts
- **Multi-threaded Capture**: Separate threads for continuous capture and event-triggered bursts
- **Platform Abstraction**: Support for Raspberry Pi (PiCamera2), macOS (OpenCV), and Docker environments
- **Frame Preprocessing**: Automatic image optimization and format standardization
- **Resource Management**: Automatic camera initialization, cleanup, and error recovery

#### Event Handling
- **Doorbell Event Processing**: Subscribe to `doorbell_pressed` events from message bus
- **Frame Event Publishing**: Publish `frame_captured` events with metadata
- **Burst Capture**: Capture 5-10 frames with configurable intervals on doorbell trigger
- **Error Event Publishing**: Publish `capture_error` events for downstream handling

### Non-Functional Requirements

#### Performance Targets
- **Capture Rate**: 30 FPS sustained on Raspberry Pi 4
- **Buffer Latency**: <100ms from doorbell trigger to first frame
- **Memory Usage**: <50MB for ring buffer operation
- **CPU Usage**: <20% during idle periods, <60% during capture bursts
- **Startup Time**: <2 seconds for camera initialization

#### Reliability Requirements
- **Thread Safety**: All ring buffer operations must be thread-safe
- **Error Recovery**: Automatic recovery from camera disconnection/failure
- **Resource Cleanup**: Proper cleanup on worker shutdown or restart
- **Memory Management**: No memory leaks during 24+ hour operation
- **Frame Quality**: Consistent image quality across all platforms

## 🔧 Implementation Specifications

### Files to Create/Modify

#### New Files
```
src/pipeline/frame_capture.py      # Main frame capture worker
src/hardware/camera_handler.py     # Refactored camera abstraction
tests/test_frame_capture.py        # Comprehensive unit tests
tests/mocks/mock_camera.py         # Camera mock for testing
tests/performance/capture_bench.py # Performance benchmarking
```

#### Modified Files
```
src/camera_handler.py              # Migration to hardware layer
config/pipeline_config.py          # Add frame capture configuration
```

### Architecture Patterns

#### Pipeline Worker Pattern
```python
class FrameCaptureWorker(PipelineWorker):
    """Frame capture worker implementing ring buffer pattern."""
    
    def __init__(self, camera_handler: CameraHandler, message_bus: MessageBus, config: Dict[str, Any]):
        super().__init__(message_bus, config)
        self.camera_handler = camera_handler
        self.ring_buffer = deque(maxlen=config.get('buffer_size', 30))
        self.capture_thread = None
        self.capture_lock = threading.RLock()
        self.is_capturing = False
        
    def _setup_subscriptions(self):
        self.message_bus.subscribe('doorbell_pressed', self.handle_doorbell_event, self.worker_id)
        self.message_bus.subscribe('system_shutdown', self.handle_shutdown, self.worker_id)
```

#### Ring Buffer Implementation
```python
class ThreadSafeRingBuffer:
    """Thread-safe ring buffer for frame storage."""
    
    def __init__(self, maxsize: int):
        self.buffer = deque(maxlen=maxsize)
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        
    def put(self, frame: np.ndarray, metadata: Dict[str, Any]):
        """Add frame to buffer with metadata."""
        with self.condition:
            frame_entry = {
                'frame': frame,
                'timestamp': time.time(),
                'metadata': metadata
            }
            self.buffer.append(frame_entry)
            self.condition.notify_all()
            
    def get_latest_frames(self, count: int) -> List[Dict[str, Any]]:
        """Get latest N frames from buffer."""
        with self.condition:
            return list(self.buffer)[-count:] if count <= len(self.buffer) else list(self.buffer)
```

### Core Implementation

#### Frame Capture Worker
```python
#!/usr/bin/env python3
"""
Frame Capture Worker with Ring Buffer

High-performance frame capture worker implementing continuous capture with
ring buffer, GPIO event integration, and multi-threaded processing.
"""

import time
import threading
import logging
from collections import deque
from typing import Dict, List, Any, Optional
import numpy as np

from src.pipeline.base_worker import PipelineWorker
from src.communication.message_bus import MessageBus, Message
from src.communication.events import PipelineEvent, EventType, FrameEvent, DoorbellEvent
from src.hardware.camera_handler import CameraHandler

logger = logging.getLogger(__name__)


class FrameCaptureWorker(PipelineWorker):
    """Frame capture worker with ring buffer and event-driven capture."""
    
    def __init__(self, camera_handler: CameraHandler, message_bus: MessageBus, config: Dict[str, Any]):
        super().__init__(message_bus, config)
        
        # Core components
        self.camera_handler = camera_handler
        self.ring_buffer = deque(maxlen=config.get('buffer_size', 30))
        
        # Threading components
        self.capture_thread = None
        self.capture_lock = threading.RLock()
        self.shutdown_event = threading.Event()
        
        # Configuration
        self.capture_fps = config.get('capture_fps', 30)
        self.burst_count = config.get('burst_count', 5)
        self.burst_interval = config.get('burst_interval', 0.2)
        
        # Metrics
        self.frames_captured = 0
        self.capture_errors = 0
        self.last_capture_time = None
        
        logger.info(f"Initialized {self.worker_id} with buffer size {len(self.ring_buffer)}")
    
    def _setup_subscriptions(self):
        """Setup message bus subscriptions."""
        self.message_bus.subscribe('doorbell_pressed', self.handle_doorbell_event, self.worker_id)
        self.message_bus.subscribe('system_shutdown', self.handle_shutdown, self.worker_id)
        logger.debug(f"{self.worker_id} subscriptions configured")
    
    def _initialize_worker(self):
        """Initialize camera and start continuous capture."""
        try:
            # Initialize camera
            if not self.camera_handler.initialize():
                raise RuntimeError("Camera initialization failed")
            
            # Start continuous capture thread
            self.capture_thread = threading.Thread(
                target=self._continuous_capture_loop,
                name=f"{self.worker_id}_capture",
                daemon=True
            )
            self.capture_thread.start()
            
            logger.info(f"{self.worker_id} initialized successfully")
            
        except Exception as e:
            logger.error(f"{self.worker_id} initialization failed: {e}")
            raise
    
    def _continuous_capture_loop(self):
        """Continuous frame capture loop for ring buffer."""
        frame_interval = 1.0 / self.capture_fps
        
        while self.running and not self.shutdown_event.is_set():
            try:
                start_time = time.time()
                
                # Capture frame
                frame = self.camera_handler.capture_frame()
                if frame is not None:
                    self._add_frame_to_buffer(frame, {'source': 'continuous'})
                    self.frames_captured += 1
                    self.last_capture_time = time.time()
                
                # Maintain frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                self.capture_errors += 1
                logger.error(f"Continuous capture error: {e}")
                time.sleep(0.1)  # Brief pause on error
    
    def handle_doorbell_event(self, message: Message):
        """Handle doorbell press event and capture frame burst."""
        try:
            doorbell_event = message.data
            logger.info(f"Processing doorbell event: {doorbell_event.event_id}")
            
            with self.capture_lock:
                # Capture burst of frames
                frames = self._capture_burst()
                
                # Publish frame events
                for i, frame_data in enumerate(frames):
                    frame_event = FrameEvent(
                        event_id=f"{doorbell_event.event_id}_frame_{i}",
                        frame_data=frame_data['frame'],
                        sequence_number=i,
                        capture_timestamp=frame_data['timestamp'],
                        metadata={
                            'source': 'doorbell_burst',
                            'doorbell_event_id': doorbell_event.event_id,
                            'burst_sequence': i,
                            'total_frames': len(frames)
                        }
                    )
                    
                    self.message_bus.publish('frame_captured', frame_event)
                
                logger.info(f"Published {len(frames)} frames for doorbell event {doorbell_event.event_id}")
                self.processed_count += 1
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Doorbell event handling failed: {e}")
            self._handle_capture_error(e, message.data)
    
    def _capture_burst(self) -> List[Dict[str, Any]]:
        """Capture burst of frames for doorbell event."""
        frames = []
        
        try:
            for i in range(self.burst_count):
                frame = self.camera_handler.capture_frame()
                if frame is not None:
                    frame_data = {
                        'frame': frame,
                        'timestamp': time.time(),
                        'sequence': i
                    }
                    frames.append(frame_data)
                    
                    # Add to ring buffer as well
                    self._add_frame_to_buffer(frame, {'source': 'burst', 'sequence': i})
                
                # Wait between captures (except for last frame)
                if i < self.burst_count - 1:
                    time.sleep(self.burst_interval)
                    
        except Exception as e:
            logger.error(f"Burst capture failed: {e}")
            raise
        
        return frames
    
    def _add_frame_to_buffer(self, frame: np.ndarray, metadata: Dict[str, Any]):
        """Add frame to ring buffer with thread safety."""
        try:
            frame_entry = {
                'frame': frame.copy(),  # Deep copy to avoid reference issues
                'timestamp': time.time(),
                'metadata': metadata
            }
            
            with self.capture_lock:
                self.ring_buffer.append(frame_entry)
                
        except Exception as e:
            logger.error(f"Buffer add failed: {e}")
    
    def get_latest_frames(self, count: int) -> List[Dict[str, Any]]:
        """Get latest N frames from ring buffer."""
        with self.capture_lock:
            if count <= len(self.ring_buffer):
                return list(self.ring_buffer)[-count:]
            else:
                return list(self.ring_buffer)
    
    def _handle_capture_error(self, error: Exception, event_data: Any):
        """Handle capture errors and publish error events."""
        error_event = PipelineEvent(
            event_type=EventType.COMPONENT_ERROR,
            data={
                'component': self.worker_id,
                'error': str(error),
                'error_type': type(error).__name__,
                'original_event': event_data.event_id if hasattr(event_data, 'event_id') else 'unknown',
                'capture_metrics': self.get_metrics()
            },
            source=self.worker_id
        )
        
        self.message_bus.publish('capture_errors', error_event)
    
    def _cleanup_worker(self):
        """Cleanup worker resources."""
        try:
            # Signal shutdown
            self.shutdown_event.set()
            
            # Wait for capture thread
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)
            
            # Cleanup camera
            if self.camera_handler:
                self.camera_handler.cleanup()
            
            # Clear ring buffer
            with self.capture_lock:
                self.ring_buffer.clear()
            
            logger.info(f"{self.worker_id} cleanup completed")
            
        except Exception as e:
            logger.error(f"{self.worker_id} cleanup failed: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get worker performance metrics."""
        base_metrics = super().get_metrics()
        
        capture_metrics = {
            'frames_captured': self.frames_captured,
            'capture_errors': self.capture_errors,
            'buffer_size': len(self.ring_buffer),
            'buffer_capacity': self.ring_buffer.maxlen,
            'last_capture_time': self.last_capture_time,
            'capture_fps_configured': self.capture_fps,
            'camera_status': self.camera_handler.get_status() if self.camera_handler else 'unknown'
        }
        
        return {**base_metrics, **capture_metrics}
    
    def handle_shutdown(self, message: Message):
        """Handle graceful shutdown signal."""
        logger.info(f"{self.worker_id} received shutdown signal")
        self.stop()
```

### Configuration

#### Pipeline Configuration
```yaml
# config/pipeline_config.py - Frame Capture Section
frame_capture:
  enabled: true
  buffer_size: 30
  capture_fps: 30
  burst_count: 5
  burst_interval: 0.2
  
  # Platform-specific settings
  platform_specific:
    raspberry_pi:
      camera_module: "picamera2"
      resolution: [640, 480]
      format: "RGB888"
      rotation: 0
      
    macos:
      camera_module: "opencv"
      device_index: 0
      resolution: [640, 480]
      
    docker:
      camera_module: "mock"
      mock_fps: 30
      
  # Performance tuning
  performance:
    thread_priority: "normal"
    memory_limit_mb: 100
    cpu_affinity: null
    
  # Error handling
  error_handling:
    max_consecutive_errors: 10
    error_backoff_seconds: 1.0
    auto_restart: true
```

## 🧪 Testing Requirements

### Unit Tests (>90% Coverage)

#### Core Functionality Tests
```python
class TestFrameCaptureWorker:
    """Comprehensive test suite for frame capture worker."""
    
    def test_worker_initialization(self):
        """Test worker initializes with correct configuration."""
        
    def test_ring_buffer_operations(self):
        """Test ring buffer add, retrieve, and overflow behavior."""
        
    def test_doorbell_event_handling(self):
        """Test doorbell event processing and frame burst capture."""
        
    def test_continuous_capture_thread(self):
        """Test continuous capture loop operation."""
        
    def test_thread_safety(self):
        """Test concurrent access to ring buffer."""
        
    def test_error_handling_and_recovery(self):
        """Test error scenarios and recovery mechanisms."""
        
    def test_resource_cleanup(self):
        """Test proper cleanup on worker shutdown."""
        
    def test_message_bus_integration(self):
        """Test subscription setup and event publishing."""
        
    def test_platform_specific_behavior(self):
        """Test platform detection and camera selection."""
```

#### Performance Tests
```python
class TestFrameCapturePerformance:
    """Performance and load testing for frame capture."""
    
    def test_capture_rate_benchmark(self):
        """Benchmark sustained capture rate on target hardware."""
        
    def test_memory_usage_over_time(self):
        """Test memory usage during extended operation."""
        
    def test_cpu_usage_measurement(self):
        """Measure CPU usage during various operations."""
        
    def test_latency_measurement(self):
        """Test latency from trigger to frame availability."""
        
    def test_burst_capture_performance(self):
        """Test performance of burst capture operation."""
```

### Integration Tests

#### Message Bus Integration
```python
def test_end_to_end_frame_capture():
    """Test complete frame capture workflow with real message bus."""
    
def test_gpio_integration():
    """Test integration with GPIO handler for doorbell events."""
    
def test_camera_handler_integration():
    """Test integration with platform-specific camera handlers."""
```

### Test Coverage Requirements
- **Minimum Coverage**: 90%
- **Critical Path Coverage**: 100% (doorbell event handling, frame capture, error recovery)
- **Error Handling Coverage**: 100%
- **Threading Coverage**: 100%

## ✅ Acceptance Criteria

### Definition of Done
- [ ] **Core Functionality**: All functional requirements implemented and tested
- [ ] **Performance Targets**: All performance benchmarks met on Raspberry Pi 4
- [ ] **Thread Safety**: Verified through stress testing and race condition analysis
- [ ] **Error Handling**: Comprehensive error scenarios tested and handled gracefully
- [ ] **Platform Compatibility**: Tested on Raspberry Pi, macOS, and Docker environments
- [ ] **Documentation**: Code documentation, API docs, and usage examples complete
- [ ] **Code Review**: Peer review completed with architecture validation
- [ ] **CI/CD Pipeline**: All automated tests pass in CI environment

### Quality Gates
- [ ] **PEP 8 Compliance**: 100% compliance with type hints on all public methods
- [ ] **Memory Leak Testing**: No memory leaks detected during 24-hour operation
- [ ] **Resource Cleanup**: All resources properly cleaned up on shutdown
- [ ] **Error Recovery**: Automatic recovery from all expected error conditions
- [ ] **Performance Regression**: No performance regression from baseline measurements

### Performance Benchmarks
- [ ] **30 FPS Capture Rate**: Sustained on Raspberry Pi 4 with <5% drops
- [ ] **<100ms Trigger Latency**: From doorbell press to first frame captured
- [ ] **<50MB Memory Usage**: For ring buffer and worker operation
- [ ] **<20% CPU Usage**: During idle periods with continuous capture
- [ ] **<2 Second Startup**: Camera initialization and worker startup time
- [ ] **<60% CPU Usage**: During burst capture operations
- [ ] **Zero Frame Loss**: During burst capture under normal conditions

### Integration Requirements
- [ ] **Message Bus Integration**: Seamless event subscription and publishing
- [ ] **Camera Handler Integration**: Works with all supported camera types
- [ ] **Configuration Integration**: Properly reads and validates configuration
- [ ] **Error Event Publishing**: Publishes structured error events for monitoring
- [ ] **Metrics Collection**: Provides comprehensive performance metrics

## 🏷️ Labels

`enhancement`, `pipeline`, `hardware`, `phase-2`, `priority-high`, `complexity-medium-high`

## 📝 Implementation Notes

### Development Approach
1. **Start with Mock Camera**: Implement and test core logic with mock camera
2. **Add Platform Detection**: Implement platform-specific camera handlers
3. **Performance Optimization**: Optimize for target hardware performance
4. **Error Handling**: Add comprehensive error handling and recovery
5. **Integration Testing**: Test with real hardware and message bus

### Risk Mitigation
- **Camera Compatibility**: Test early with target hardware
- **Performance Issues**: Continuous benchmarking during development
- **Thread Safety**: Use proven patterns and thorough testing
- **Memory Management**: Regular memory profiling and leak detection

### Success Metrics
- All acceptance criteria met
- Performance targets achieved on target hardware
- Zero critical bugs in production deployment
- Positive code review feedback on architecture compliance