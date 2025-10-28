# Face Detector Implementations

This directory contains the face detection backend implementations for the Doorbell Security System, following a strategy pattern design inspired by Frigate NVR's detector plugin architecture.

## Overview

The system supports multiple face detection backends with automatic hardware detection and fallback strategies:

- **CPU Detector**: dlib HOG/CNN models via face_recognition library
- **GPU Detector**: ONNX Runtime with CUDA acceleration
- **EdgeTPU Detector**: Coral EdgeTPU with TensorFlow Lite
- **Mock Detector**: Testing and development purposes

## Architecture

```
BaseDetector (Abstract)
├── CPUDetector - CPU-based face detection (dlib)
├── GPUDetector - GPU-accelerated (ONNX Runtime + CUDA)
├── EdgeTPUDetector - Coral EdgeTPU optimized (TFLite)
└── MockDetector - Testing mock
```

## Quick Start

### Auto-Detection

```python
from src.detectors import create_detector

# Automatically select best available detector
detector = create_detector()  # Auto-detects: EdgeTPU > GPU > CPU

# Detect faces
import cv2
image = cv2.imread('image.jpg')
detections, metrics = detector.detect_faces(image)

print(f"Found {len(detections)} faces")
print(f"Inference time: {metrics.inference_time * 1000:.2f}ms")
```

### Manual Selection

```python
from src.detectors import DetectorFactory

# Create specific detector type
gpu_detector = DetectorFactory.create('gpu', {
    'model': 'retinaface',
    'device': 'cuda:0',
    'batch_size': 4
})

edgetpu_detector = DetectorFactory.create('edgetpu', {
    'model': 'mobilenet_face',
    'confidence_threshold': 0.6
})

cpu_detector = DetectorFactory.create('cpu', {
    'model': 'hog',
    'number_of_times_to_upsample': 1
})
```

### Hardware Detection

```python
from src.detectors import HardwareDetector

hw = HardwareDetector()
print(hw.get_hardware_summary())

# Check specific hardware
if hw.has_cuda_gpu():
    print("GPU available!")
    gpus = hw.detect_gpus()
    for gpu in gpus:
        print(f"  {gpu.name}: {gpu.memory_mb}MB")

if hw.has_edgetpu():
    print("EdgeTPU available!")
    tpus = hw.detect_edgetpus()
    for tpu in tpus:
        print(f"  {tpu.device_type} at {tpu.device_path}")
```

## Performance Benchmarking

```python
from src.detectors import PerformanceProfiler
import numpy as np

# Create test images
test_images = [
    np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    for _ in range(10)
]

# Benchmark detector
profiler = PerformanceProfiler()
result = profiler.benchmark_detector(
    detector,
    test_images,
    iterations=10,
    warmup_iterations=3
)

print(f"Average: {result.avg_inference_time_ms:.2f}ms")
print(f"FPS: {result.fps:.1f}")
print(f"P95 latency: {result.p95_inference_time_ms:.2f}ms")

# Compare multiple detectors
print(profiler.get_comparison_report())
```

## File Structure

```
src/detectors/
├── __init__.py                   # Package exports
├── base_detector.py              # Abstract base class
├── cpu_detector.py               # CPU implementation
├── gpu_detector.py               # GPU implementation (NEW)
├── edgetpu_detector.py           # EdgeTPU implementation (NEW)
├── detector_factory.py           # Factory pattern implementation
├── model_manager.py              # Model download/caching (NEW)
├── hardware_detector.py          # Hardware detection (NEW)
├── performance_profiler.py       # Benchmarking tools (NEW)
├── detection_result.py           # Detection result classes
├── benchmarking.py               # Performance benchmarking
├── ensemble_detector.py          # Ensemble detection
└── health_monitor.py             # Health monitoring

tests/
├── test_cpu_detector.py          # CPU detector tests
├── test_gpu_detector.py          # GPU detector tests (NEW)
├── test_edgetpu_detector.py      # EdgeTPU detector tests (NEW)
├── test_hardware_detector.py     # Hardware detection tests (NEW)
└── test_detector_integration.py  # Integration tests

docs/detectors/
├── gpu_setup.md                  # GPU setup guide (NEW)
├── edgetpu_setup.md              # EdgeTPU setup guide (NEW)
└── performance_guide.md          # Performance optimization (NEW)
```

## Installation

### Base Requirements

```bash
pip install -r requirements.txt
```

### GPU Support

```bash
pip install -r requirements-gpu.txt
# Requires: NVIDIA GPU, CUDA driver, onnxruntime-gpu
```

### EdgeTPU Support

```bash
pip install -r requirements-edgetpu.txt
# Requires: Coral EdgeTPU device, libedgetpu runtime
```

## Performance Comparison

Typical performance on 640x480 images:

| Detector | Hardware | Model | FPS | Latency | Power | Cost |
|----------|----------|-------|-----|---------|-------|------|
| CPU | Pi 4 | HOG | 2-3 | 300-500ms | 5W | $55 |
| CPU | Desktop i7 | HOG | 15-20 | 50-70ms | 65W | $300 |
| GPU | RTX 3060 | RetinaFace | 100-150 | 6-10ms | 170W | $400 |
| GPU | RTX 4090 | RetinaFace | 300-400 | 2-3ms | 450W | $1600 |
| EdgeTPU | Coral USB | MobileNet | 40-50 | 20-25ms | 7W | $60 |
| EdgeTPU | Coral USB | BlazeFace | 60-80 | 12-16ms | 7W | $60 |

## Key Features

### GPU Detector
- ✅ ONNX Runtime with CUDA support
- ✅ 5-10x speedup over CPU
- ✅ Batch processing
- ✅ Multiple models (RetinaFace, YOLOv5)
- ✅ Automatic Mixed Precision (AMP)
- ✅ Memory management

### EdgeTPU Detector
- ✅ Coral EdgeTPU optimized
- ✅ Sub-100ms inference
- ✅ Low power consumption (~7W)
- ✅ Temperature monitoring
- ✅ Multiple models (MobileNet, EfficientDet, BlazeFace)
- ✅ Ideal for edge deployment

### Infrastructure
- ✅ Automatic hardware detection
- ✅ Graceful fallback to CPU
- ✅ Model caching and management
- ✅ Performance profiling
- ✅ Health monitoring
- ✅ Ensemble detection support

## Documentation

- [GPU Setup Guide](../../docs/detectors/gpu_setup.md) - Installation, configuration, troubleshooting
- [EdgeTPU Setup Guide](../../docs/detectors/edgetpu_setup.md) - Coral device setup and optimization
- [Performance Guide](../../docs/detectors/performance_guide.md) - Optimization strategies and best practices

## Testing

Run all detector tests:

```bash
# All detector tests
python -m unittest discover tests -p "test_*detector*.py"

# Specific detector tests
python -m unittest tests.test_gpu_detector
python -m unittest tests.test_edgetpu_detector
python -m unittest tests.test_hardware_detector
```

## Development

### Adding a New Detector

1. Create detector class inheriting from `BaseDetector`
2. Implement required abstract methods:
   - `is_available()` - Check if detector can run
   - `_get_detector_type()` - Return detector type
   - `_initialize_model()` - Initialize detection model
   - `_run_inference()` - Run face detection

3. Register with factory:
```python
DetectorFactory.register_detector('new_type', NewDetector)
```

4. Add tests in `tests/test_new_detector.py`

### Example New Detector

```python
from src.detectors.base_detector import BaseDetector, DetectorType

class NewDetector(BaseDetector):
    @classmethod
    def is_available(cls) -> bool:
        # Check if hardware/library available
        return True
    
    def _get_detector_type(self) -> DetectorType:
        return DetectorType.NEW
    
    def _initialize_model(self) -> None:
        # Initialize detection model
        pass
    
    def _run_inference(self, image):
        # Run detection and return results
        return []
```

## Contributing

When contributing detector implementations:

1. Follow existing architecture patterns
2. Add comprehensive tests (>80% coverage)
3. Include performance benchmarks
4. Document setup and configuration
5. Handle errors gracefully
6. Support automatic fallback

## License

MIT License - See LICENSE file for details

## Support

- Issues: https://github.com/itsnothuy/Doorbell-System/issues
- Documentation: docs/detectors/
- Examples: examples/detector_usage.py
