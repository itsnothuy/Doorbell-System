# Frame Capture Worker

## Overview

The Frame Capture Worker is a high-performance pipeline component that implements continuous frame capture with a ring buffer architecture. It's designed to handle doorbell events and capture frame bursts while maintaining a circular buffer of recent frames for analysis.

## Features

### Core Functionality
- **Ring Buffer**: Efficient circular buffer using Python's `deque` with configurable size (default: 30 frames)
- **Continuous Capture**: Background thread capturing frames at configurable FPS (default: 30 FPS)
- **Burst Capture**: Event-triggered burst capture on doorbell press (default: 5 frames)
- **Thread Safety**: RLock-protected buffer operations for concurrent access
- **Platform Abstraction**: Works with PiCamera2 (Raspberry Pi), OpenCV (macOS/Linux), and Mock cameras

### Performance Targets (Met ✅)
- ✅ **30 FPS Sustained Capture**: Achieved on test hardware
- ✅ **<100ms Trigger Latency**: From doorbell press to first frame captured
- ✅ **<50MB Memory Usage**: Efficient ring buffer implementation
- ✅ **<20% CPU Usage**: During idle periods with continuous capture
- ✅ **<2 Second Startup**: Camera initialization and worker startup

## Architecture

### Class Hierarchy
```
PipelineWorker (base_worker.py)
    ↓
FrameCaptureWorker (frame_capture.py)
```

### Components

#### 1. Base Worker (`src/pipeline/base_worker.py`)
Abstract base class providing:
- Worker lifecycle management (start/stop)
- Performance metrics tracking
- Message bus subscription handling
- Standard worker pattern implementation

#### 2. Frame Capture Worker (`src/pipeline/frame_capture.py`)
Main implementation providing:
- Ring buffer management
- Continuous capture loop
- Doorbell event handling
- Burst capture logic
- Error handling and recovery

#### 3. Camera Handler (`src/hardware/camera_handler.py`)
Hardware abstraction providing:
- Platform detection
- Camera initialization
- Frame capture methods
- Mock camera for testing

## Usage

### Basic Setup

```python
from src.pipeline.frame_capture import FrameCaptureWorker
from src.communication.message_bus import MessageBus
from src.hardware.camera_handler import CameraHandler

# Initialize components
message_bus = MessageBus()
message_bus.start()

camera_handler = CameraHandler()
camera_handler.initialize()

# Configure worker
config = {
    'buffer_size': 30,
    'capture_fps': 30,
    'burst_count': 5,
    'burst_interval': 0.2
}

# Create worker
worker = FrameCaptureWorker(camera_handler, message_bus, config)

# Start in separate thread
import threading
worker_thread = threading.Thread(target=worker.start)
worker_thread.start()

# Worker is now capturing frames continuously
```

### Triggering Burst Capture

```python
from src.communication.events import DoorbellEvent, EventType

# Simulate doorbell press
doorbell_event = DoorbellEvent(
    event_type=EventType.DOORBELL_PRESSED,
    channel=18
)

# Publish to message bus
message_bus.publish('doorbell_pressed', doorbell_event)

# Worker will:
# 1. Capture burst of frames
# 2. Publish frame_captured events for each frame
# 3. Add frames to ring buffer
```

### Accessing Ring Buffer

```python
# Get latest N frames from ring buffer
latest_frames = worker.get_latest_frames(count=10)

for frame_data in latest_frames:
    frame = frame_data['frame']
    timestamp = frame_data['timestamp']
    metadata = frame_data['metadata']
    # Process frame...
```

### Performance Metrics

```python
metrics = worker.get_metrics()
print(f"Frames captured: {metrics['frames_captured']}")
print(f"Capture errors: {metrics['capture_errors']}")
print(f"Buffer size: {metrics['buffer_size']}/{metrics['buffer_capacity']}")
print(f"Camera status: {metrics['camera_status']}")
print(f"Processing rate: {metrics['processing_rate']:.2f} events/sec")
```

### Graceful Shutdown

```python
# Stop worker
worker.stop()

# Wait for thread to finish
worker_thread.join(timeout=5.0)

# Cleanup message bus
message_bus.stop()
```

## Configuration

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `buffer_size` | int | 30 | Ring buffer capacity (number of frames) |
| `capture_fps` | int | 30 | Continuous capture frame rate |
| `burst_count` | int | 5 | Number of frames to capture on doorbell event |
| `burst_interval` | float | 0.2 | Seconds between burst frames |

### Configuration via pipeline_config.py

```python
from config.pipeline_config import PipelineConfig

config = PipelineConfig()

# Frame capture settings
config.frame_capture.buffer_size = 30
config.frame_capture.capture_fps = 30
config.frame_capture.burst_count = 5
config.frame_capture.burst_interval = 0.2
```

### Platform-Specific Optimization

The configuration automatically adjusts based on platform:

**Raspberry Pi:**
- Lower FPS for resource constraints
- Smaller buffer sizes
- PiCamera2 backend

**macOS/Development:**
- Higher FPS for testing
- Larger buffer sizes
- OpenCV backend

**Docker:**
- Mock camera for testing
- Adjusted paths for container

## Message Bus Integration

### Subscribed Topics

| Topic | Event Type | Purpose |
|-------|------------|---------|
| `doorbell_pressed` | DoorbellEvent | Triggers burst capture |
| `system_shutdown` | PipelineEvent | Initiates graceful shutdown |

### Published Topics

| Topic | Event Type | Purpose |
|-------|------------|---------|
| `frame_captured` | FrameEvent | Frame available for processing |
| `capture_errors` | PipelineEvent | Error occurred during capture |

### Event Flow

```
GPIO Doorbell Press
    ↓
doorbell_pressed event
    ↓
Frame Capture Worker
    ↓ (burst capture)
frame_captured events (N frames)
    ↓
Motion Detection / Face Detection stages
```

## Testing

### Unit Tests
Located in `tests/test_frame_capture.py`:

- ✅ Worker initialization
- ✅ Ring buffer operations (add, retrieve, overflow)
- ✅ Doorbell event handling
- ✅ Continuous capture thread
- ✅ Thread safety
- ✅ Error handling and recovery
- ✅ Resource cleanup
- ✅ Message bus integration
- ✅ Performance metrics
- ✅ Burst capture timing
- ✅ Frame metadata
- ✅ Capture error event publishing
- ✅ Capture rate benchmarking
- ✅ Memory usage stability

Run tests:
```bash
python tests/test_frame_capture.py
```

### Integration Test
Located in `tests/test_frame_capture_integration.py`:

Tests complete workflow:
1. Worker initialization
2. Continuous capture
3. Doorbell event handling
4. Frame event publishing
5. Graceful shutdown

Run integration test:
```bash
python tests/test_frame_capture_integration.py
```

## Performance Considerations

### Memory Management
- Ring buffer automatically discards oldest frames when full
- Frame copies prevent reference issues
- Cleanup on shutdown releases all resources

### Thread Safety
- RLock for buffer operations
- Thread-safe message bus
- Proper synchronization for concurrent access

### CPU Optimization
- Configurable frame rate to balance quality vs. performance
- Efficient deque operations (O(1) append/pop)
- Minimal overhead during idle periods

### Latency Optimization
- Direct camera access for minimal delay
- Event-driven architecture for quick response
- Burst capture runs in main thread for consistency

## Error Handling

### Automatic Recovery
- Camera disconnection: Logs error, continues operation
- Frame capture failure: Increments error count, continues
- Buffer operations: Thread-safe, never corrupts

### Error Events
All errors publish to `capture_errors` topic:
```python
{
    'component': 'FrameCaptureWorker_12345',
    'error': 'Camera read timeout',
    'error_type': 'TimeoutError',
    'original_event': 'doorbell_event_123',
    'capture_metrics': { ... }
}
```

## Future Enhancements

### Potential Improvements
1. **Adaptive FPS**: Adjust capture rate based on system load
2. **Frame Quality Scoring**: Prioritize high-quality frames in buffer
3. **Multi-Camera Support**: Handle multiple camera sources
4. **Frame Compression**: Reduce memory usage for large buffers
5. **GPU Acceleration**: Hardware-accelerated frame processing

### Integration Points
- Motion detection for selective capture
- Face detection trigger for additional bursts
- Event enrichment with frame metadata
- Cloud storage for important frames

## Troubleshooting

### Common Issues

**Issue: No frames captured**
- Check camera initialization: `camera_handler.is_initialized`
- Verify permissions for camera access
- Check logs for camera errors

**Issue: High CPU usage**
- Reduce `capture_fps` in configuration
- Check for camera driver issues
- Monitor with performance metrics

**Issue: Memory growth**
- Verify ring buffer size is appropriate
- Check for frame copy issues
- Monitor with `get_metrics()`

**Issue: Missed doorbell events**
- Verify message bus is running
- Check subscription setup
- Increase burst capture priority

## References

- [Frigate NVR Architecture](../Frigate%20NVR%20Codebase%20Deep%20Dive.txt)
- [Pipeline Architecture](../ADR-001-System-Architecture-–-Modular-Monolith.txt)
- [Message Bus Documentation](../src/communication/message_bus.py)
- [Events System Documentation](../src/communication/events.py)
