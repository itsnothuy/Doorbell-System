# Issue #5: Motion Detection Worker Implementation

## 📋 **Overview**

Implement motion detection worker as an optional performance optimization stage in the pipeline. This worker performs background subtraction and motion region analysis to reduce false positives and improve face detection efficiency by only processing frames with significant motion.

## 🎯 **Objectives**

### **Primary Goals**
1. **Performance Optimization**: Reduce face detection workload by filtering static frames
2. **False Positive Reduction**: Only trigger face detection on motion events
3. **Resource Efficiency**: Minimize CPU usage on inactive periods
4. **Queue Management**: Intelligent frame filtering for downstream workers

### **Success Criteria**
- Motion detection accuracy >95% (no missed genuine motion)
- Processing latency <50ms per frame
- CPU usage reduction of 60-80% during static periods
- Seamless integration with existing pipeline architecture

## 🏗️ **Architecture Requirements**

### **Pipeline Position**
```
Frame Capture Worker → Motion Detection Worker → Face Detection Worker Pool
```

### **Processing Flow**
1. **Background Model**: Maintain rolling background subtraction model
2. **Motion Analysis**: Detect motion regions and calculate motion scores
3. **Filtering Logic**: Apply thresholds and region-based filtering
4. **Frame Forwarding**: Forward only frames with significant motion
5. **State Management**: Update background model and motion history

## 📁 **Implementation Specifications**

### **File Structure**
```
src/pipeline/motion_detector.py        # Main motion detection worker
config/motion_config.py                # Motion detection configuration
tests/test_motion_detector.py          # Comprehensive test suite
```

### **Core Component: `MotionDetector`**
```python
class MotionDetector(PipelineWorker):
    """Motion detection worker for pipeline optimization."""
    
    def __init__(self, message_bus: MessageBus, config: MotionConfig):
        super().__init__(message_bus, config.base_config)
        self.motion_config = config
        
        # Background subtraction model
        self.bg_subtractor = None
        self.background_model = None
        
        # Motion analysis parameters
        self.motion_threshold = config.motion_threshold
        self.min_contour_area = config.min_contour_area
        self.motion_history_size = config.motion_history_size
        
        # Performance tracking
        self.frames_processed = 0
        self.frames_forwarded = 0
        self.motion_events = 0
        
    def initialize_background_model(self, initial_frames: List[np.ndarray]) -> None:
        """Initialize background subtraction model."""
        
    def detect_motion(self, frame: np.ndarray) -> MotionResult:
        """Detect motion in frame and return analysis."""
        
    def should_forward_frame(self, motion_result: MotionResult) -> bool:
        """Determine if frame should be forwarded to face detection."""
        
    def update_background_model(self, frame: np.ndarray, motion_detected: bool) -> None:
        """Update background model based on motion state."""
```

### **Configuration Class: `MotionConfig`**
```python
@dataclass
class MotionConfig:
    """Configuration for motion detection worker."""
    
    # Base worker configuration
    base_config: Dict[str, Any]
    
    # Motion detection parameters
    enabled: bool = True
    motion_threshold: float = 25.0
    min_contour_area: int = 500
    gaussian_blur_kernel: Tuple[int, int] = (21, 21)
    
    # Background subtraction
    bg_subtractor_type: str = "MOG2"  # MOG2, KNN, or GMM
    bg_learning_rate: float = 0.01
    bg_history: int = 500
    bg_var_threshold: int = 50
    
    # Motion analysis
    motion_history_size: int = 10
    motion_smoothing_factor: float = 0.3
    min_motion_duration: float = 0.5  # seconds
    max_static_duration: float = 30.0  # seconds
    
    # Region of interest
    roi_enabled: bool = False
    roi_coordinates: Optional[Tuple[int, int, int, int]] = None
    
    # Performance optimization
    frame_resize_factor: float = 0.5
    skip_frame_count: int = 0  # Process every Nth frame
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MotionConfig':
        """Create configuration from dictionary with validation."""
```

### **Data Structures**
```python
@dataclass
class MotionResult:
    """Result of motion detection analysis."""
    motion_detected: bool
    motion_score: float
    motion_regions: List[Tuple[int, int, int, int]]  # Bounding boxes
    contour_count: int
    largest_contour_area: int
    motion_center: Optional[Tuple[int, int]]
    frame_timestamp: float
    processing_time: float

@dataclass
class MotionHistory:
    """Motion detection history for trend analysis."""
    recent_scores: List[float]
    motion_events: List[float]  # Timestamps
    static_duration: float
    trend_direction: str  # "increasing", "decreasing", "stable"
```

## 🔧 **Implementation Details**

### **1. Background Subtraction Implementation**
```python
def _initialize_bg_subtractor(self) -> cv2.BackgroundSubtractor:
    """Initialize appropriate background subtractor."""
    if self.motion_config.bg_subtractor_type == "MOG2":
        return cv2.createBackgroundSubtractorMOG2(
            history=self.motion_config.bg_history,
            varThreshold=self.motion_config.bg_var_threshold,
            detectShadows=True
        )
    elif self.motion_config.bg_subtractor_type == "KNN":
        return cv2.createBackgroundSubtractorKNN(
            history=self.motion_config.bg_history,
            dist2Threshold=400.0,
            detectShadows=True
        )
    else:
        raise ValueError(f"Unsupported background subtractor: {self.motion_config.bg_subtractor_type}")

def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
    """Preprocess frame for motion detection."""
    # Resize for performance
    if self.motion_config.frame_resize_factor != 1.0:
        height, width = frame.shape[:2]
        new_height = int(height * self.motion_config.frame_resize_factor)
        new_width = int(width * self.motion_config.frame_resize_factor)
        frame = cv2.resize(frame, (new_width, new_height))
    
    # Convert to grayscale
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    frame = cv2.GaussianBlur(frame, self.motion_config.gaussian_blur_kernel, 0)
    
    return frame
```

### **2. Motion Analysis Algorithm**
```python
def _analyze_motion(self, fg_mask: np.ndarray) -> MotionResult:
    """Analyze foreground mask for motion characteristics."""
    start_time = time.time()
    
    # Apply morphological operations to clean up mask
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    valid_contours = [c for c in contours if cv2.contourArea(c) >= self.motion_config.min_contour_area]
    
    # Calculate motion metrics
    motion_regions = []
    total_motion_area = 0
    largest_contour_area = 0
    motion_center = None
    
    if valid_contours:
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            motion_regions.append((x, y, w, h))
            
            area = cv2.contourArea(contour)
            total_motion_area += area
            largest_contour_area = max(largest_contour_area, area)
        
        # Calculate center of motion
        moments = cv2.moments(fg_mask)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            motion_center = (cx, cy)
    
    # Calculate motion score
    frame_area = fg_mask.shape[0] * fg_mask.shape[1]
    motion_score = (total_motion_area / frame_area) * 100
    
    # Determine if motion is significant
    motion_detected = (
        motion_score >= self.motion_config.motion_threshold and
        len(valid_contours) > 0 and
        largest_contour_area >= self.motion_config.min_contour_area
    )
    
    processing_time = time.time() - start_time
    
    return MotionResult(
        motion_detected=motion_detected,
        motion_score=motion_score,
        motion_regions=motion_regions,
        contour_count=len(valid_contours),
        largest_contour_area=largest_contour_area,
        motion_center=motion_center,
        frame_timestamp=time.time(),
        processing_time=processing_time
    )
```

### **3. Intelligent Frame Forwarding**
```python
def _should_forward_frame(self, motion_result: MotionResult) -> bool:
    """Advanced logic for frame forwarding decision."""
    
    # Always forward if motion detected
    if motion_result.motion_detected:
        return True
    
    # Forward periodically even without motion (heartbeat)
    current_time = time.time()
    if current_time - self.last_forwarded_time > self.motion_config.max_static_duration:
        return True
    
    # Forward based on motion trend analysis
    if self._is_motion_trend_increasing():
        return True
    
    # Forward if we're in a transition period
    if self._is_motion_transitioning():
        return True
    
    return False

def _update_motion_history(self, motion_result: MotionResult) -> None:
    """Update motion history for trend analysis."""
    self.motion_history.recent_scores.append(motion_result.motion_score)
    
    # Maintain history size
    if len(self.motion_history.recent_scores) > self.motion_config.motion_history_size:
        self.motion_history.recent_scores.pop(0)
    
    # Update motion events
    if motion_result.motion_detected:
        self.motion_history.motion_events.append(motion_result.frame_timestamp)
        
        # Clean old events
        cutoff_time = motion_result.frame_timestamp - 60.0  # Keep last minute
        self.motion_history.motion_events = [
            t for t in self.motion_history.motion_events if t > cutoff_time
        ]
    
    # Calculate trend
    self._calculate_motion_trend()
```

### **4. Pipeline Integration**
```python
def _setup_subscriptions(self) -> None:
    """Setup message bus subscriptions."""
    self.message_bus.subscribe('frame_captured', self._handle_frame_event)
    self.message_bus.subscribe('pipeline_control', self._handle_control_event)

def _handle_frame_event(self, message: Message) -> None:
    """Process incoming frame from capture worker."""
    try:
        frame_event = message.data
        frame = frame_event.frame
        
        # Skip frames if configured
        if self.motion_config.skip_frame_count > 0:
            self.frame_skip_counter = (self.frame_skip_counter + 1) % (self.motion_config.skip_frame_count + 1)
            if self.frame_skip_counter != 0:
                return
        
        # Process frame
        motion_result = self.detect_motion(frame)
        self.frames_processed += 1
        
        # Update motion history
        self._update_motion_history(motion_result)
        
        # Decide whether to forward frame
        should_forward = self._should_forward_frame(motion_result)
        
        if should_forward:
            # Create enhanced frame event with motion data
            enhanced_event = FrameEvent(
                frame=frame,
                timestamp=frame_event.timestamp,
                source=frame_event.source,
                motion_data=motion_result,
                processing_stage="motion_detected"
            )
            
            # Forward to face detection
            self.message_bus.publish('motion_analyzed', enhanced_event)
            self.frames_forwarded += 1
            
            if motion_result.motion_detected:
                self.motion_events += 1
        
        # Update background model
        self.update_background_model(frame, motion_result.motion_detected)
        
        # Publish motion statistics periodically
        if self.frames_processed % 100 == 0:
            self._publish_motion_stats()
            
    except Exception as e:
        self.error_count += 1
        logger.error(f"Motion detection failed: {e}")
        
        # Forward frame anyway to prevent pipeline stall
        fallback_event = FrameEvent(
            frame=frame,
            timestamp=frame_event.timestamp,
            source=frame_event.source,
            motion_data=None,
            processing_stage="motion_detection_failed"
        )
        self.message_bus.publish('motion_analyzed', fallback_event)
```

## 🧪 **Testing Requirements**

### **Unit Tests**
```python
class TestMotionDetector:
    """Comprehensive test suite for motion detector."""
    
    def test_background_model_initialization(self):
        """Test background model setup and initialization."""
        
    def test_motion_detection_accuracy(self):
        """Test motion detection with known motion patterns."""
        
    def test_false_positive_filtering(self):
        """Test filtering of noise and false motion."""
        
    def test_frame_forwarding_logic(self):
        """Test intelligent frame forwarding decisions."""
        
    def test_performance_benchmarks(self):
        """Test processing speed and resource usage."""
        
    def test_configuration_validation(self):
        """Test configuration loading and validation."""
        
    def test_error_handling(self):
        """Test graceful handling of various error conditions."""
        
    def test_pipeline_integration(self):
        """Test integration with message bus and other workers."""
```

### **Integration Tests**
```python
def test_motion_pipeline_integration():
    """Test full pipeline with motion detection enabled."""
    
def test_motion_detector_performance():
    """Test performance impact of motion detection."""
    
def test_motion_detector_failover():
    """Test behavior when motion detection fails."""
```

## 📊 **Performance Targets**

### **Processing Performance**
- **Latency**: <50ms per frame (640x480)
- **Throughput**: >20 FPS processing capability
- **CPU Usage**: <15% on Raspberry Pi 4
- **Memory Usage**: <100MB additional footprint

### **Detection Accuracy**
- **True Positive Rate**: >95% (genuine motion detected)
- **False Positive Rate**: <5% (static scenes marked as motion)
- **False Negative Rate**: <2% (missed genuine motion)
- **Noise Filtering**: >90% reduction in lighting/shadow false positives

### **Resource Optimization**
- **Frame Filtering**: 60-80% reduction in downstream processing
- **Background Learning**: Adaptive model updates
- **Memory Efficiency**: Bounded memory usage regardless of runtime

## 🔧 **Configuration Example**

### **motion_config.py**
```python
# Motion Detection Configuration
MOTION_CONFIG = {
    "enabled": True,
    "motion_threshold": 25.0,
    "min_contour_area": 500,
    "gaussian_blur_kernel": (21, 21),
    
    # Background subtraction
    "bg_subtractor_type": "MOG2",
    "bg_learning_rate": 0.01,
    "bg_history": 500,
    "bg_var_threshold": 50,
    
    # Motion analysis
    "motion_history_size": 10,
    "motion_smoothing_factor": 0.3,
    "min_motion_duration": 0.5,
    "max_static_duration": 30.0,
    
    # Performance optimization
    "frame_resize_factor": 0.5,
    "skip_frame_count": 0,
    
    # Region of interest (optional)
    "roi_enabled": False,
    "roi_coordinates": None
}
```

## 📈 **Monitoring and Metrics**

### **Key Metrics to Track**
- Motion detection accuracy and false positive rates
- Frame processing latency and throughput
- Background model adaptation performance
- Frame forwarding ratio and efficiency
- CPU and memory usage patterns

### **Health Checks**
- Background model convergence
- Motion threshold effectiveness
- Pipeline stage synchronization
- Error rate monitoring

## 🎯 **Definition of Done**

### **Functional Requirements**
- [ ] Motion detection worker correctly identifies motion in frames
- [ ] Background subtraction model adapts to environmental changes
- [ ] Frame forwarding logic optimizes downstream processing
- [ ] Integration with existing pipeline architecture complete
- [ ] Configuration system supports all motion detection parameters

### **Non-Functional Requirements**
- [ ] Processing latency meets performance targets (<50ms)
- [ ] Memory usage bounded and efficient
- [ ] CPU usage optimized for edge devices
- [ ] Error handling prevents pipeline failures
- [ ] Comprehensive test coverage (>90%)

### **Documentation Requirements**
- [ ] Code documentation with clear docstrings
- [ ] Configuration guide for motion detection tuning
- [ ] Performance tuning recommendations
- [ ] Integration examples with other pipeline components
- [ ] Troubleshooting guide for common issues

---

## 🔗 **Dependencies**

### **Previous Issues**
- **Issue #4**: Frame Capture Worker (provides input frames)

### **Next Issues**
- **Issue #6**: Face Detection Worker Pool (receives filtered frames)
- **Issue #8**: Event Processing System (processes motion events)

### **External Dependencies**
- OpenCV 4.x for computer vision operations
- NumPy for array operations
- Message bus infrastructure
- Configuration management system

---

## 🤖 **For Coding Agents: Auto-Close Setup**

### **Branch Naming Convention**
When implementing this issue, create your branch using one of these patterns:
- `issue-5/motion-detection-worker`
- `5-motion-detection-worker` 
- `issue-5/implement-motion-detection`

### **PR Creation**
The GitHub Action will automatically append `Closes #5` to your PR description when you follow the branch naming convention above. This ensures the issue closes automatically when your PR is merged to the default branch.

### **Manual Alternative**
If you prefer manual control, include one of these in your PR description:
```
Closes #5
Fixes #5
Resolves #5
```

---

**This issue implements motion detection as a performance optimization stage in the Frigate-inspired pipeline architecture, reducing computational load on face detection while maintaining high accuracy for genuine motion events.**