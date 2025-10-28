# Performance Optimization Guide

This guide covers performance optimization strategies for face detection in the Doorbell Security System across CPU, GPU, and EdgeTPU detectors.

## Table of Contents
- [Performance Overview](#performance-overview)
- [Benchmarking](#benchmarking)
- [Optimization Strategies](#optimization-strategies)
- [Platform-Specific Tuning](#platform-specific-tuning)
- [Production Best Practices](#production-best-practices)

## Performance Overview

### Detector Performance Comparison

Typical performance on 640x480 resolution images:

| Detector | Hardware | Model | FPS | Latency | Power | Cost |
|----------|----------|-------|-----|---------|-------|------|
| CPU HOG | Pi 4 | HOG | 2-3 | 300-500ms | 5W | $55 |
| CPU CNN | Pi 4 | CNN | 0.5-1 | 1000-2000ms | 6W | $55 |
| CPU HOG | Desktop i7 | HOG | 15-20 | 50-70ms | 65W | $300 |
| GPU | RTX 3060 | RetinaFace | 100-150 | 6-10ms | 170W | $400 |
| GPU | RTX 4090 | RetinaFace | 300-400 | 2-3ms | 450W | $1600 |
| EdgeTPU | Coral USB | MobileNet | 40-50 | 20-25ms | 7W | $60 |
| EdgeTPU | Coral USB | BlazeFace | 60-80 | 12-16ms | 7W | $60 |

### Performance vs Accuracy Trade-offs

```
High Accuracy (Slower)
├── GPU: RetinaFace (640x640)
├── GPU: EfficientDet
├── EdgeTPU: EfficientDet Face
├── EdgeTPU: MobileNet Face
├── CPU: CNN Model
└── CPU: HOG Model

High Speed (Lower Accuracy)
├── EdgeTPU: BlazeFace
├── GPU: YOLOv5 Face (320x320)
├── CPU: HOG (low upsample)
└── Mock Detector
```

## Benchmarking

### Running Benchmarks

Use the built-in performance profiler:

```python
from src.detectors import DetectorFactory, create_detector
from src.detectors.performance_profiler import PerformanceProfiler
import numpy as np
import cv2

# Load test images
test_images = [
    cv2.imread(f'test_data/image{i}.jpg') 
    for i in range(10)
]

# Or create synthetic test images
test_images = [
    np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    for _ in range(10)
]

# Benchmark single detector
detector = create_detector('gpu', {'model': 'retinaface'})
profiler = PerformanceProfiler()

result = profiler.benchmark_detector(
    detector,
    test_images,
    iterations=10,
    warmup_iterations=3
)

print(profiler.get_detailed_report('gpu'))
```

### Comparing Multiple Detectors

```python
# Compare all available detectors
detectors_to_test = [
    ('cpu', {'model': 'hog'}),
    ('gpu', {'model': 'retinaface'}),
    ('edgetpu', {'model': 'mobilenet_face'})
]

profiler = PerformanceProfiler()
results = {}

for detector_type, config in detectors_to_test:
    try:
        detector = create_detector(detector_type, config)
        result = profiler.benchmark_detector(detector, test_images, iterations=5)
        results[detector_type] = result
        print(f"{detector_type}: {result.fps:.1f} FPS")
    except Exception as e:
        print(f"Skipping {detector_type}: {e}")

# Print comparison report
print(profiler.get_comparison_report())
```

### Automated Performance Testing

```python
def run_performance_suite():
    """Run comprehensive performance tests."""
    from src.detectors.benchmarking import DetectorBenchmark
    
    benchmark = DetectorBenchmark()
    
    # Test different image sizes
    image_sizes = [(320, 240), (640, 480), (1280, 720), (1920, 1080)]
    
    # Test different batch sizes (GPU only)
    batch_sizes = [1, 2, 4, 8, 16]
    
    # Run benchmarks
    results = benchmark.run_full_suite(
        image_sizes=image_sizes,
        batch_sizes=batch_sizes,
        iterations=10
    )
    
    # Generate report
    benchmark.save_report('performance_report.json')
    return results
```

## Optimization Strategies

### 1. Model Selection

Choose the right model for your requirements:

```python
# For maximum accuracy (slower)
config = {
    'model': 'retinaface',
    'input_size': (640, 640),
    'confidence_threshold': 0.7
}

# For balanced performance
config = {
    'model': 'mobilenet_face',
    'input_size': (320, 320),
    'confidence_threshold': 0.6
}

# For maximum speed (lower accuracy)
config = {
    'model': 'blazeface',
    'input_size': (224, 224),
    'confidence_threshold': 0.5
}
```

### 2. Input Resolution

Lower resolution dramatically improves speed:

```python
import cv2

# Resize before detection
def preprocess_for_speed(image):
    # Downscale to 320x240 for faster detection
    resized = cv2.resize(image, (320, 240))
    return resized

# Maintain aspect ratio
def smart_resize(image, target_size=640):
    height, width = image.shape[:2]
    scale = target_size / max(height, width)
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height))
    
    return image
```

### 3. Confidence Threshold Tuning

Higher thresholds filter more detections but may miss faces:

```python
# Conservative (fewer false positives, may miss faces)
config = {'confidence_threshold': 0.8}

# Balanced
config = {'confidence_threshold': 0.6}

# Aggressive (more detections, more false positives)
config = {'confidence_threshold': 0.3}
```

### 4. Batch Processing (GPU only)

Process multiple images simultaneously on GPU:

```python
# Single image processing (slower)
for image in images:
    detections, _ = detector.detect_faces(image)
    process(detections)

# Batch processing (faster on GPU)
config = {'batch_size': 8}
detector = create_detector('gpu', config)

# Process in batches
batch_size = 8
for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    # GPU processes batch together
    for image in batch:
        detections, _ = detector.detect_faces(image)
        process(detections)
```

### 5. Detection Region of Interest (ROI)

Only detect faces in specific regions:

```python
def detect_in_roi(detector, image, roi):
    """Detect faces only in region of interest."""
    x, y, w, h = roi
    roi_image = image[y:y+h, x:x+w]
    
    detections, metrics = detector.detect_faces(roi_image)
    
    # Adjust coordinates back to full image
    for detection in detections:
        top, right, bottom, left = detection.bounding_box
        detection.bounding_box = (
            top + y,
            right + x,
            bottom + y,
            left + x
        )
    
    return detections, metrics

# Example: Only detect in doorway area
doorway_roi = (200, 100, 400, 600)  # x, y, width, height
detections, _ = detect_in_roi(detector, image, doorway_roi)
```

### 6. Frame Skipping

Don't process every frame in video:

```python
def process_video_optimized(video_path, skip_frames=3):
    """Process video with frame skipping."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process every Nth frame
        if frame_count % skip_frames == 0:
            detections, _ = detector.detect_faces(frame)
            process(detections, frame_count)
        
        frame_count += 1
    
    cap.release()
```

### 7. Motion-Based Detection

Only run face detection when motion is detected:

```python
import cv2

class MotionDetector:
    def __init__(self, threshold=25):
        self.background = None
        self.threshold = threshold
    
    def has_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.background is None:
            self.background = gray
            return False
        
        # Compute difference
        frame_delta = cv2.absdiff(self.background, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Update background
        self.background = gray
        
        # Check if significant motion
        return cv2.countNonZero(thresh) > 1000

# Use with face detection
motion_detector = MotionDetector()

for frame in video_frames:
    if motion_detector.has_motion(frame):
        # Only run expensive face detection when motion detected
        detections, _ = detector.detect_faces(frame)
        process(detections)
```

## Platform-Specific Tuning

### CPU Optimization

```python
# 1. Use HOG model (faster than CNN)
config = {'model': 'hog', 'number_of_times_to_upsample': 0}

# 2. Reduce image size
image = cv2.resize(image, (320, 240))

# 3. Use multiple processes
from multiprocessing import Pool

def detect_wrapper(args):
    image, config = args
    detector = create_detector('cpu', config)
    return detector.detect_faces(image)

with Pool(processes=4) as pool:
    results = pool.map(detect_wrapper, [(img, config) for img in images])

# 4. Optimize upsample parameter
config = {
    'model': 'hog',
    'number_of_times_to_upsample': 0  # Faster, may miss small faces
}
```

### GPU Optimization

```python
# 1. Maximize batch size
config = {
    'model': 'retinaface',
    'batch_size': 16,  # Adjust based on GPU memory
    'device': 'cuda:0'
}

# 2. Enable AMP (Automatic Mixed Precision)
config = {
    'model': 'retinaface',
    'enable_amp': True,  # Use FP16 for faster inference
    'device': 'cuda:0'
}

# 3. Optimize memory usage
config = {
    'model': 'retinaface',
    'memory_fraction': 0.7,  # Don't use all GPU memory
    'device': 'cuda:0'
}

# 4. Keep detector session alive
detector = create_detector('gpu', config)
# Reuse detector for all inferences (avoid session creation overhead)
for image in images:
    detections, _ = detector.detect_faces(image)

# 5. Use appropriate model size
# RetinaFace 640x640 for accuracy
# RetinaFace 320x320 for speed
# YOLOv5 for maximum speed
```

### EdgeTPU Optimization

```python
# 1. Use optimal model
config = {
    'model': 'blazeface',  # Fastest
    # 'model': 'mobilenet_face',  # Balanced
    # 'model': 'efficientdet_face',  # Most accurate
}

# 2. Enable max performance runtime
# sudo apt install libedgetpu1-max

# 3. Ensure USB 3.0 connection
# Check with: lsusb -t

# 4. Add cooling
# Attach heatsink or use active cooling

# 5. Process single images efficiently
# EdgeTPU excels at single-image inference
# Don't try to batch (not supported)

# 6. Pipeline preprocessing
import threading
from queue import Queue

def preprocessing_thread(input_queue, output_queue):
    while True:
        image = input_queue.get()
        if image is None:
            break
        processed = preprocess(image)
        output_queue.put(processed)

def detection_thread(input_queue, output_queue, detector):
    while True:
        image = input_queue.get()
        if image is None:
            break
        detections, _ = detector.detect_faces(image)
        output_queue.put(detections)

# Pipeline: Preprocess -> Detect -> Process
```

## Production Best Practices

### 1. Detector Pooling

Create a pool of detector instances for parallel processing:

```python
from src.detectors import DetectorFactory

# Create detector pool
pool_size = 4
detector_pool = DetectorFactory.create_detector_pool(
    detector_type='cpu',
    pool_size=pool_size,
    config={'model': 'hog'}
)

# Use with ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor

def process_with_pool(images):
    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        futures = []
        for i, image in enumerate(images):
            detector = detector_pool[i % pool_size]
            future = executor.submit(detector.detect_faces, image)
            futures.append(future)
        
        results = [f.result() for f in futures]
    
    return results
```

### 2. Caching and Deduplication

Avoid processing duplicate frames:

```python
import hashlib

class DetectionCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
    
    def get_hash(self, image):
        return hashlib.md5(image.tobytes()).hexdigest()
    
    def get(self, image):
        img_hash = self.get_hash(image)
        return self.cache.get(img_hash)
    
    def set(self, image, detections):
        img_hash = self.get_hash(image)
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))
        self.cache[img_hash] = detections

# Use cache
cache = DetectionCache()

def detect_with_cache(image):
    cached = cache.get(image)
    if cached is not None:
        return cached
    
    detections, _ = detector.detect_faces(image)
    cache.set(image, detections)
    return detections
```

### 3. Adaptive Quality

Adjust detection quality based on load:

```python
class AdaptiveDetector:
    def __init__(self):
        self.detector = None
        self.current_quality = 'high'
        self.load_threshold = 0.8
    
    def adjust_quality(self, cpu_usage):
        if cpu_usage > self.load_threshold and self.current_quality != 'low':
            # Switch to faster model
            self.current_quality = 'low'
            self.detector = create_detector('cpu', {
                'model': 'hog',
                'number_of_times_to_upsample': 0
            })
        elif cpu_usage < 0.5 and self.current_quality != 'high':
            # Switch to better model
            self.current_quality = 'high'
            self.detector = create_detector('cpu', {
                'model': 'hog',
                'number_of_times_to_upsample': 1
            })
    
    def detect(self, image):
        import psutil
        cpu_usage = psutil.cpu_percent() / 100.0
        self.adjust_quality(cpu_usage)
        return self.detector.detect_faces(image)
```

### 4. Monitoring and Alerting

Monitor performance in production:

```python
import time
from collections import deque

class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.inference_times = deque(maxlen=window_size)
        self.fps_window = deque(maxlen=window_size)
        self.last_time = time.time()
    
    def record(self, inference_time):
        self.inference_times.append(inference_time)
        
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.fps_window.append(fps)
        self.last_time = current_time
    
    def get_stats(self):
        if not self.inference_times:
            return {}
        
        import numpy as np
        return {
            'avg_inference_ms': np.mean(self.inference_times) * 1000,
            'p95_inference_ms': np.percentile(self.inference_times, 95) * 1000,
            'avg_fps': np.mean(self.fps_window),
            'min_fps': np.min(self.fps_window)
        }
    
    def check_degradation(self, threshold_fps=10):
        stats = self.get_stats()
        if stats.get('avg_fps', 0) < threshold_fps:
            # Alert: Performance degradation
            return True
        return False

# Use monitor
monitor = PerformanceMonitor()

for image in images:
    start = time.time()
    detections, _ = detector.detect_faces(image)
    inference_time = time.time() - start
    
    monitor.record(inference_time)
    
    if monitor.check_degradation():
        print(f"WARNING: Performance degraded: {monitor.get_stats()}")
```

### 5. Graceful Degradation

Fallback to simpler detection when performance drops:

```python
class RobustDetector:
    def __init__(self):
        self.primary = create_detector('gpu', {'model': 'retinaface'})
        self.fallback = create_detector('cpu', {'model': 'hog'})
        self.use_fallback = False
        self.error_count = 0
    
    def detect(self, image):
        try:
            if self.use_fallback:
                return self.fallback.detect_faces(image)
            
            detections, metrics = self.primary.detect_faces(image)
            
            # Check if inference is too slow
            if metrics.inference_time > 0.5:  # 500ms threshold
                self.error_count += 1
                if self.error_count > 5:
                    self.use_fallback = True
                    print("Switching to fallback detector")
            else:
                self.error_count = max(0, self.error_count - 1)
            
            return detections, metrics
            
        except Exception as e:
            print(f"Primary detector failed: {e}, using fallback")
            return self.fallback.detect_faces(image)
```

## Performance Checklist

Before deploying to production:

- [ ] Benchmark all available detectors on target hardware
- [ ] Profile with realistic images and workloads
- [ ] Test under different lighting conditions
- [ ] Verify performance with expected image sizes
- [ ] Test memory usage and limits
- [ ] Implement monitoring and alerting
- [ ] Set up graceful degradation
- [ ] Document performance characteristics
- [ ] Create performance regression tests
- [ ] Optimize for 95th percentile latency, not just average

## Additional Resources

- [Face Detection Benchmarks](https://paperswithcode.com/task/face-detection)
- [ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/)
- [TensorFlow Lite Performance Best Practices](https://www.tensorflow.org/lite/performance/best_practices)
- [Coral Performance Optimization](https://coral.ai/docs/edgetpu/inference/)
