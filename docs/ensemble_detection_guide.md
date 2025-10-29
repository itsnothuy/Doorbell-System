# Ensemble Detection System Guide

## Overview

The Enhanced Ensemble Detection System combines multiple face detection models with intelligent fusion algorithms to maximize detection accuracy while optimizing performance for real-time applications. This guide provides comprehensive documentation on using the new ensemble detection features.

## Features

### 1. Advanced Fusion Strategies

The system supports 8 different fusion strategies:

- **Simple Voting**: Majority voting where detections agreed upon by majority of detectors are kept
- **Weighted Voting**: Voting weighted by detector confidence scores
- **Consensus**: Requires agreement from a configurable percentage of detectors
- **Union**: Keeps all unique detections from all detectors
- **Intersection**: Only keeps detections found by all detectors
- **Adaptive**: Dynamically selects strategy based on conditions
- **Confidence-Weighted**: Emphasizes high-confidence detections
- **NMS Fusion**: Non-Maximum Suppression to remove overlapping detections

### 2. Adaptive Detector Selection

Automatically selects optimal detector combinations based on:
- Speed requirements (low latency)
- Accuracy requirements (high precision)
- Historical performance data
- Detector priorities

### 3. Parallel Execution

Run multiple detectors simultaneously using thread pools for maximum throughput.

### 4. Performance Tracking

Comprehensive metrics including:
- Fusion times
- Per-detector performance
- Detection counts
- Success rates

## Quick Start

### Basic Usage (Legacy API)

```python
from src.detectors.ensemble_detector import EnsembleDetector, EnsembleStrategy
from src.detectors.detector_factory import DetectorFactory

# Create individual detectors
cpu_detector = DetectorFactory.create('cpu', {'model': 'hog'})
gpu_detector = DetectorFactory.create('gpu', {'model': 'cnn'})

# Create ensemble with legacy API
ensemble = EnsembleDetector(
    detectors=[cpu_detector, gpu_detector],
    strategy=EnsembleStrategy.VOTING
)

# Detect faces
import cv2
image = cv2.imread('test.jpg')
detections, metrics = ensemble.detect_faces(image)
```

### Advanced Usage (New API)

```python
from src.detectors.ensemble_detector import (
    EnsembleDetector, 
    FusionStrategy,
    DetectorPriority
)
from src.detectors.detector_factory import DetectorFactory

# Create ensemble
ensemble = EnsembleDetector(config={
    'fusion_strategy': 'nms_fusion',
    'parallel_execution': True,
    'max_workers': 4,
    'timeout': 10.0
})

# Add detectors with custom configuration
cpu_detector = DetectorFactory.create('cpu', {'model': 'hog'})
ensemble.add_detector(
    'cpu_fast',
    cpu_detector,
    weight=1.0,
    priority=DetectorPriority.HIGH,
    use_for_speed=True,
    use_for_accuracy=False
)

gpu_detector = DetectorFactory.create('gpu', {'model': 'cnn'})
ensemble.add_detector(
    'gpu_accurate',
    gpu_detector,
    weight=1.5,
    priority=DetectorPriority.CRITICAL,
    use_for_speed=False,
    use_for_accuracy=True
)

# Initialize detectors
ensemble.load_model()

# Detect with performance requirements
performance_requirements = {
    'max_latency': 0.5,  # seconds
    'min_accuracy': 0.85,
    'prefer_speed': False  # prefer accuracy
}

detections, metrics = ensemble.detect_faces(
    image,
    performance_requirements=performance_requirements
)
```

## Configuration Options

### EnsembleDetector Configuration

```python
config = {
    # Fusion strategy
    'fusion_strategy': 'weighted_voting',  # or use FusionStrategy enum
    
    # Execution mode
    'parallel_execution': True,
    'max_workers': 4,
    'timeout': 10.0,
    
    # Detection thresholds
    'confidence_threshold': 0.5,
    'min_face_size': (30, 30),
    'max_face_size': (1000, 1000),
    
    # Fusion configuration
    'fusion_config': {
        'iou_threshold': 0.5,
        'confidence_threshold': 0.5,
        'consensus_threshold': 0.6
    },
    
    # Adaptive selector configuration
    'adaptive_config': {
        'selection_criteria': {
            'latency_weight': 0.4,
            'accuracy_weight': 0.6,
            'min_accuracy': 0.8,
            'max_latency': 0.5
        }
    }
}

ensemble = EnsembleDetector(config=config)
```

### DetectorConfig Options

When adding a detector, you can specify:

```python
ensemble.add_detector(
    name='my_detector',
    detector=detector_instance,
    weight=1.5,                        # Voting weight (higher = more influence)
    priority=DetectorPriority.HIGH,    # Priority for adaptive selection
    timeout=5.0,                       # Individual detector timeout (seconds)
    enabled=True,                      # Whether detector is active
    min_confidence=0.3,                # Minimum confidence for detections
    max_detections=10,                 # Maximum detections to consider
    use_for_speed=True,                # Use in speed-optimized scenarios
    use_for_accuracy=True,             # Use in accuracy-optimized scenarios
    metadata={'custom': 'data'}        # Custom metadata dictionary
)
```

## Performance Requirements

When calling `detect_faces`, you can specify performance requirements:

```python
# For high-speed scenarios
speed_requirements = {
    'max_latency': 0.1,      # Maximum acceptable latency (seconds)
    'prefer_speed': True,    # Prioritize speed over accuracy
    'min_accuracy': 0.7      # Minimum acceptable accuracy
}

# For high-accuracy scenarios
accuracy_requirements = {
    'max_latency': 1.0,      # Allow more time
    'prefer_speed': False,   # Prioritize accuracy over speed
    'min_accuracy': 0.95     # High accuracy requirement
}

detections, metrics = ensemble.detect_faces(image, speed_requirements)
```

## Fusion Strategies Comparison

| Strategy | Best For | Pros | Cons |
|----------|----------|------|------|
| Simple Voting | General use | Simple, fast | May miss unique detections |
| Weighted Voting | Trusted detectors | Respects detector quality | Needs proper weight tuning |
| Consensus | High confidence needed | Very reliable | May miss valid detections |
| Union | Recall optimization | Maximizes detections | May include false positives |
| Intersection | Precision optimization | Minimizes false positives | May miss valid detections |
| Confidence-Weighted | Variable confidence | Adapts to detection quality | More complex |
| NMS Fusion | Overlapping detections | Removes duplicates well | Needs IoU tuning |
| Adaptive | Dynamic scenarios | Flexible | Most complex |

## Examples

### Example 1: Speed-Optimized Detection

```python
# Create ensemble with fast detectors
ensemble = EnsembleDetector(config={
    'fusion_strategy': 'simple_voting',
    'parallel_execution': True
})

# Add CPU detector for speed
cpu_detector = DetectorFactory.create('cpu', {'model': 'hog'})
ensemble.add_detector(
    'cpu',
    cpu_detector,
    priority=DetectorPriority.HIGH,
    use_for_speed=True
)

# Detect with speed preference
requirements = {'prefer_speed': True, 'max_latency': 0.1}
detections, metrics = ensemble.detect_faces(image, requirements)
print(f"Detection time: {metrics.total_time * 1000:.2f}ms")
```

### Example 2: Accuracy-Optimized Detection

```python
# Create ensemble with accurate detectors
ensemble = EnsembleDetector(config={
    'fusion_strategy': 'confidence_weighted',
    'parallel_execution': True
})

# Add multiple high-accuracy detectors
gpu_detector = DetectorFactory.create('gpu', {'model': 'cnn'})
ensemble.add_detector(
    'gpu',
    gpu_detector,
    weight=2.0,
    priority=DetectorPriority.CRITICAL,
    use_for_accuracy=True
)

edgetpu_detector = DetectorFactory.create('edgetpu', {})
ensemble.add_detector(
    'edgetpu',
    edgetpu_detector,
    weight=1.5,
    priority=DetectorPriority.HIGH,
    use_for_accuracy=True
)

# Detect with accuracy preference
requirements = {'prefer_speed': False, 'min_accuracy': 0.95}
detections, metrics = ensemble.detect_faces(image, requirements)
print(f"Average confidence: {metrics.confidence:.3f}")
```

### Example 3: Dynamic Detector Management

```python
# Create empty ensemble
ensemble = EnsembleDetector()

# Add detectors dynamically
detectors = {
    'cpu': DetectorFactory.create('cpu', {'model': 'hog'}),
    'gpu': DetectorFactory.create('gpu', {'model': 'cnn'})
}

for name, detector in detectors.items():
    ensemble.add_detector(name, detector)

# Remove a detector
ensemble.remove_detector('cpu')

# Check which detectors are active
active = [name for name, config in ensemble.detector_configs.items() 
          if config.enabled]
print(f"Active detectors: {active}")
```

### Example 4: Performance Monitoring

```python
# Run detection
ensemble = EnsembleDetector(config={'fusion_strategy': 'nms_fusion'})
# ... add detectors ...

for i in range(10):
    detections, metrics = ensemble.detect_faces(test_images[i])

# Check performance metrics
print("Ensemble Metrics:")
print(f"Total detections: {ensemble.ensemble_metrics['total_detections']}")
print(f"Successful: {ensemble.ensemble_metrics['successful_detections']}")

if ensemble.ensemble_metrics['fusion_times']:
    avg_fusion = sum(ensemble.ensemble_metrics['fusion_times']) / len(
        ensemble.ensemble_metrics['fusion_times'])
    print(f"Average fusion time: {avg_fusion * 1000:.2f}ms")

# Per-detector performance
for detector_name, perf in ensemble.ensemble_metrics['detector_performance'].items():
    avg_detections = perf['total_detections'] / perf['calls']
    print(f"{detector_name}: {avg_detections:.1f} detections/call")
```

## Troubleshooting

### Issue: Slow Performance

**Solution**: Use parallel execution and speed-optimized detectors
```python
config = {
    'parallel_execution': True,
    'max_workers': 4,
    'fusion_strategy': 'simple_voting'  # Faster than complex strategies
}
ensemble = EnsembleDetector(config=config)

# Add only fast detectors
ensemble.add_detector('cpu', cpu_detector, use_for_speed=True)
```

### Issue: Low Detection Accuracy

**Solution**: Use consensus or confidence-weighted fusion
```python
config = {
    'fusion_strategy': 'consensus',
    'fusion_config': {
        'consensus_threshold': 0.8  # Require 80% agreement
    }
}
ensemble = EnsembleDetector(config=config)

# Add multiple accurate detectors
ensemble.add_detector('gpu', gpu_detector, weight=2.0, use_for_accuracy=True)
ensemble.add_detector('edgetpu', edgetpu_detector, weight=1.5, use_for_accuracy=True)
```

### Issue: Too Many False Positives

**Solution**: Use intersection strategy or higher thresholds
```python
config = {
    'fusion_strategy': 'intersection',  # Only detections found by all
    'confidence_threshold': 0.7  # Higher threshold
}
```

### Issue: Missing Some Faces

**Solution**: Use union strategy or lower thresholds
```python
config = {
    'fusion_strategy': 'union',  # Keep all detections
    'confidence_threshold': 0.3,  # Lower threshold
    'fusion_config': {
        'iou_threshold': 0.3  # Group more aggressively
    }
}
```

## Best Practices

1. **Start Simple**: Begin with `simple_voting` strategy and adjust as needed

2. **Tune Weights**: Adjust detector weights based on their known accuracy
   ```python
   ensemble.add_detector('accurate', detector, weight=2.0)
   ensemble.add_detector('fast', detector, weight=1.0)
   ```

3. **Use Parallel Execution**: Enable for multiple detectors
   ```python
   config = {'parallel_execution': True, 'max_workers': 4}
   ```

4. **Monitor Performance**: Track metrics to identify bottlenecks
   ```python
   print(ensemble.ensemble_metrics)
   ```

5. **Handle Failures Gracefully**: The ensemble continues even if detectors fail
   ```python
   # Detectors that fail are automatically skipped
   ensemble.add_detector('experimental', detector)  # May fail without breaking ensemble
   ```

6. **Adjust for Your Hardware**: Configure based on available resources
   ```python
   # For Raspberry Pi
   config = {
       'parallel_execution': False,  # Single-threaded
       'max_workers': 1
   }
   
   # For powerful server
   config = {
       'parallel_execution': True,
       'max_workers': 8
   }
   ```

## API Reference

### Classes

#### `FusionStrategy(Enum)`
Available fusion strategies:
- `SIMPLE_VOTING`
- `WEIGHTED_VOTING`
- `CONSENSUS`
- `UNION`
- `INTERSECTION`
- `ADAPTIVE`
- `CONFIDENCE_WEIGHTED`
- `NMS_FUSION`

#### `DetectorPriority(Enum)`
Priority levels for adaptive selection:
- `LOW = 1`
- `MEDIUM = 2`
- `HIGH = 3`
- `CRITICAL = 4`

#### `DetectorConfig(dataclass)`
Configuration for individual detectors.

#### `EnsembleDetection(dataclass)`
Enhanced detection result with metadata.

#### `DetectionFuser`
Fusion algorithm implementation.

#### `AdaptiveEnsembleSelector`
Intelligent detector selection.

#### `EnsembleDetector(BaseDetector)`
Main ensemble detector class.

### Methods

#### `EnsembleDetector.add_detector(name, detector, **kwargs)`
Add a detector to the ensemble.

#### `EnsembleDetector.remove_detector(name)`
Remove a detector from the ensemble.

#### `EnsembleDetector.load_model()`
Initialize all detectors.

#### `EnsembleDetector.detect_faces(image, performance_requirements=None)`
Detect faces using the ensemble.

## Migration from Legacy API

If you're using the old API:

```python
# Old way
ensemble = EnsembleDetector(
    detectors=[det1, det2],
    strategy=EnsembleStrategy.VOTING
)
```

It still works! But you can enhance it:

```python
# New way with more control
ensemble = EnsembleDetector(config={
    'fusion_strategy': 'weighted_voting',
    'parallel_execution': True
})
ensemble.add_detector('det1', det1, weight=1.5)
ensemble.add_detector('det2', det2, weight=1.0)
```

## Contributing

When adding new fusion strategies or features:

1. Add the strategy to `FusionStrategy` enum
2. Implement in `DetectionFuser` class
3. Add tests in `tests/test_ensemble_enhanced.py`
4. Update this documentation

## Support

For issues or questions:
- Check existing tests for examples
- Review code comments in `src/detectors/ensemble_detector.py`
- Open an issue on GitHub
