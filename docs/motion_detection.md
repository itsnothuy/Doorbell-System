# Motion Detection Worker

The motion detection worker is an optional performance optimization stage in the Frigate-inspired pipeline architecture. It performs background subtraction and motion region analysis to reduce false positives and improve face detection efficiency by only processing frames with significant motion.

## Overview

The motion detection worker sits between frame capture and face detection in the pipeline:

```
Frame Capture Worker → Motion Detection Worker → Face Detection Worker Pool
```

### Key Features

- **Background Subtraction**: Uses MOG2 or KNN algorithms for adaptive background modeling
- **Motion Analysis**: Detects motion regions and calculates motion scores
- **Intelligent Forwarding**: Only forwards frames with significant motion or periodic heartbeats
- **Trend Analysis**: Tracks motion history to detect motion trends and transitions
- **Performance Optimization**: Reduces face detection workload by 60-80% during static periods
- **Platform Optimizations**: Automatically adjusts settings for Raspberry Pi, Docker, or development environments

## Architecture

### Pipeline Position

The motion detector is an optional stage that can be enabled/disabled via configuration. When enabled, it:

1. **Receives Frames**: Subscribes to `frame_captured` events from the frame capture worker
2. **Analyzes Motion**: Performs background subtraction and contour analysis
3. **Makes Forwarding Decisions**: Intelligently decides whether to forward frames
4. **Publishes Results**: Forwards frames with motion metadata to `motion_analyzed` topic

### Processing Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Frame Capture Worker                                       │
│  Publishes: frame_captured                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Motion Detection Worker                                    │
│                                                             │
│  1. Preprocess Frame (resize, grayscale, blur)             │
│  2. Apply Background Subtraction (MOG2/KNN)                │
│  3. Analyze Foreground Mask (contours, regions)            │
│  4. Calculate Motion Score                                  │
│  5. Update Motion History                                   │
│  6. Decide Forward/Drop                                     │
│  7. Enhance Frame with Motion Data                          │
│                                                             │
│  Publishes: motion_analyzed                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Face Detection Worker Pool                                 │
│  Subscribes: motion_analyzed                                │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Basic Configuration

```python
from config.motion_config import MotionConfig

# Create default configuration
config = MotionConfig()

# Or create from dictionary
config = MotionConfig.from_dict({
    'enabled': True,
    'motion_threshold': 25.0,
    'min_contour_area': 500,
    'bg_subtractor_type': 'MOG2'
})
```

### Configuration Parameters

#### Core Motion Detection

- **`motion_threshold`** (float, default: 25.0): Percentage of frame area that must have motion to trigger detection
- **`min_contour_area`** (int, default: 500): Minimum contour area in pixels to be considered valid motion
- **`gaussian_blur_kernel`** (tuple, default: (21, 21)): Kernel size for Gaussian blur preprocessing

#### Background Subtraction

- **`bg_subtractor_type`** (str, default: "MOG2"): Background subtractor algorithm ("MOG2" or "KNN")
- **`bg_learning_rate`** (float, default: 0.01): Learning rate for background model updates
- **`bg_history`** (int, default: 500): Number of frames for background history
- **`bg_var_threshold`** (int, default: 50): Variance threshold for MOG2

#### Motion History & Trends

- **`motion_history_size`** (int, default: 10): Number of recent motion scores to track
- **`motion_smoothing_factor`** (float, default: 0.3): Smoothing factor for motion trend analysis
- **`min_motion_duration`** (float, default: 0.5): Minimum motion duration in seconds
- **`max_static_duration`** (float, default: 30.0): Maximum duration without forwarding frames (heartbeat)

#### Performance Optimization

- **`frame_resize_factor`** (float, default: 0.5): Frame resize factor for processing (0.0-1.0)
- **`skip_frame_count`** (int, default: 0): Process every Nth frame (0 = process all)

#### Region of Interest (ROI)

- **`roi_enabled`** (bool, default: False): Enable ROI processing
- **`roi_coordinates`** (tuple, optional): ROI bounding box (x, y, width, height)

### Platform-Specific Optimizations

The motion detector automatically applies optimizations based on the detected platform:

#### Raspberry Pi

```python
frame_resize_factor = 0.5
bg_subtractor_type = "MOG2"
bg_history = 300
skip_frame_count = 1  # Process every other frame
min_contour_area = 400
```

#### Docker/Container

```python
frame_resize_factor = 0.6
skip_frame_count = 0
```

#### macOS/Development

```python
frame_resize_factor = 0.8
skip_frame_count = 0
bg_subtractor_type = "MOG2"
```

## Usage

### Basic Usage

```python
from src.pipeline.motion_detector import MotionDetector
from src.communication.message_bus import MessageBus
from config.motion_config import MotionConfig

# Create message bus
message_bus = MessageBus()
message_bus.start()

# Create configuration
config = MotionConfig()
config.enabled = True

# Create motion detector
detector = MotionDetector(message_bus, config)

# Start detector (runs in background)
import threading
detector_thread = threading.Thread(target=detector.start)
detector_thread.start()

# Detector will now process frame_captured events
# and publish motion_analyzed events
```

### Integration with Pipeline

```python
from src.pipeline.orchestrator import PipelineOrchestrator
from config.pipeline_config import PipelineConfig

# Enable motion detection in pipeline config
pipeline_config = PipelineConfig()
pipeline_config.motion_detection.enabled = True
pipeline_config.motion_detection.motion_threshold = 30.0

# Create and start orchestrator
orchestrator = PipelineOrchestrator(pipeline_config)
orchestrator.start()
```

### Subscribing to Motion Events

```python
def handle_motion_event(message):
    frame_event = message.data
    
    # Check if motion was detected
    if frame_event.data.get('motion_detected'):
        motion_data = frame_event.data.get('motion_data')
        print(f"Motion detected! Score: {motion_data['motion_score']:.2f}%")
        print(f"Motion regions: {motion_data['motion_regions']}")

# Subscribe to motion analyzed events
message_bus.subscribe('motion_analyzed', handle_motion_event, 'motion_subscriber')
```

## Performance Optimization

### Frame Forwarding Strategy

The motion detector uses intelligent frame forwarding to optimize performance:

1. **Always Forward on Motion**: Frames with motion above threshold are always forwarded
2. **Periodic Heartbeat**: Frames are forwarded periodically even without motion (default: every 30s)
3. **Trend Detection**: Frames are forwarded if motion trend is increasing
4. **Transition Detection**: Frames are forwarded during motion start/stop transitions

### Performance Targets

- **Processing Latency**: <50ms per frame (640x480)
- **Throughput**: >20 FPS processing capability
- **CPU Usage**: <15% on Raspberry Pi 4
- **Memory Usage**: <100MB additional footprint
- **Frame Filtering**: 60-80% reduction in downstream processing

### Tuning for Your Environment

#### Low-Power Devices (e.g., Raspberry Pi Zero)

```python
config = MotionConfig()
config.optimize_for_low_power()
# Sets: frame_resize_factor=0.4, skip_frame_count=2, etc.
```

#### High-Performance Systems

```python
config = MotionConfig()
config.optimize_for_performance()
# Sets: frame_resize_factor=0.8, skip_frame_count=0, etc.
```

## Monitoring and Metrics

### Getting Motion Detector Metrics

```python
metrics = detector.get_metrics()

print(f"Frames processed: {metrics['frames_processed']}")
print(f"Frames forwarded: {metrics['frames_forwarded']}")
print(f"Motion events: {metrics['motion_events']}")
print(f"Forward ratio: {metrics['forward_ratio']:.2%}")
print(f"Motion trend: {metrics['motion_trend']}")
```

### Key Metrics to Monitor

- **Forward Ratio**: Percentage of frames forwarded to face detection
- **Motion Event Ratio**: Percentage of frames with detected motion
- **Average Motion Score**: Mean motion score across recent frames
- **Motion Trend**: Current trend direction (increasing/decreasing/stable)
- **Processing Time**: Average time per frame

### Health Checks

The motion detector provides health information through metrics:

```python
metrics = detector.get_metrics()

# Check if motion detector is functioning
if metrics['error_rate'] > 0.1:
    print("WARNING: High error rate in motion detector")

# Check if background model is converging
if metrics['frames_processed'] > 100 and metrics['forward_ratio'] > 0.9:
    print("WARNING: Most frames being forwarded - check thresholds")
```

## Troubleshooting

### Issue: All Frames Being Forwarded

**Symptoms**: Forward ratio close to 100%

**Solutions**:
- Increase `motion_threshold` (try 30.0 or higher)
- Increase `min_contour_area` (try 800 or higher)
- Check if camera is stable (vibrations can cause false motion)

### Issue: Motion Not Being Detected

**Symptoms**: Forward ratio close to 0% or motion events very low

**Solutions**:
- Decrease `motion_threshold` (try 15.0 or lower)
- Decrease `min_contour_area` (try 300 or lower)
- Check `frame_resize_factor` - very small values may miss small motion
- Verify camera is working and frames are being captured

### Issue: High CPU Usage

**Symptoms**: CPU usage >30% on low-power devices

**Solutions**:
- Increase `skip_frame_count` (process every 2nd or 3rd frame)
- Decrease `frame_resize_factor` (try 0.3 or lower)
- Use MOG2 instead of KNN (faster but slightly less accurate)
- Reduce `bg_history` (try 200 or lower)

### Issue: Missing Motion Transitions

**Symptoms**: Motion starts/stops are not detected smoothly

**Solutions**:
- Increase `motion_history_size` (try 15 or higher)
- Adjust `motion_smoothing_factor` (try 0.5)
- Ensure `skip_frame_count` is 0 or 1 for better temporal resolution

## Examples

### Example 1: Basic Motion Detection

```python
import cv2
import numpy as np
from src.pipeline.motion_detector import MotionDetector
from src.communication.message_bus import MessageBus
from config.motion_config import MotionConfig

# Setup
message_bus = MessageBus()
message_bus.start()

config = MotionConfig()
detector = MotionDetector(message_bus, config)
detector._initialize_worker()

# Create test frame
frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Detect motion
result = detector.detect_motion(frame)

print(f"Motion detected: {result.motion_detected}")
print(f"Motion score: {result.motion_score:.2f}%")
print(f"Motion regions: {len(result.motion_regions)}")
```

### Example 2: Custom ROI

```python
# Configure ROI for doorway area
config = MotionConfig()
config.roi_enabled = True
config.roi_coordinates = (200, 150, 300, 400)  # x, y, width, height

detector = MotionDetector(message_bus, config)
```

### Example 3: Aggressive Filtering

```python
# Maximum performance optimization
config = MotionConfig()
config.motion_threshold = 35.0  # Higher threshold
config.min_contour_area = 1000  # Larger contours only
config.frame_resize_factor = 0.3  # Very small frames
config.skip_frame_count = 2  # Process every 3rd frame
config.max_static_duration = 60.0  # Longer heartbeat interval

detector = MotionDetector(message_bus, config)
```

## API Reference

### MotionDetector Class

```python
class MotionDetector(PipelineWorker):
    """Motion detection worker for pipeline optimization."""
    
    def __init__(self, message_bus: MessageBus, config: MotionConfig)
    def detect_motion(self, frame: np.ndarray) -> MotionResult
    def get_metrics(self) -> Dict[str, Any]
```

### MotionResult Class

```python
@dataclass
class MotionResult:
    """Result of motion detection analysis."""
    
    motion_detected: bool
    motion_score: float
    motion_regions: List[Tuple[int, int, int, int]]
    contour_count: int
    largest_contour_area: int
    motion_center: Optional[Tuple[int, int]]
    frame_timestamp: float
    processing_time: float
```

### MotionHistory Class

```python
@dataclass
class MotionHistory:
    """Motion detection history for trend analysis."""
    
    recent_scores: List[float]
    motion_events: List[float]
    static_duration: float
    trend_direction: str
    last_motion_time: Optional[float]
```

## Best Practices

1. **Start with Default Settings**: The default configuration is well-balanced for most use cases
2. **Monitor Metrics**: Regularly check forward ratio and motion event ratio
3. **Platform Optimization**: Let the system auto-optimize for your platform
4. **ROI for Specific Areas**: Use ROI if you only care about motion in specific areas (e.g., doorway)
5. **Gradual Tuning**: Adjust thresholds gradually and monitor the impact
6. **Test with Real Scenarios**: Test with actual doorbell scenarios, not just test patterns
7. **Consider Lighting**: Motion detection is sensitive to lighting changes - adjust accordingly

## Related Documentation

- [Pipeline Architecture](./pipeline_architecture.md)
- [Frame Capture Worker](./frame_capture.md)
- [Face Detection Worker](./face_detection.md)
- [Configuration Management](./configuration.md)

## Contributing

When modifying the motion detection worker, ensure:

1. All tests pass: `python -m unittest tests.test_motion_detector -v`
2. Performance targets are maintained
3. Platform optimizations are tested on target devices
4. Documentation is updated to reflect changes
5. Backward compatibility is maintained for existing configurations
