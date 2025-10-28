# GPU Setup Guide

This guide covers setting up GPU-accelerated face detection for the Doorbell Security System using NVIDIA CUDA GPUs.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA Compute Capability 6.0 or higher
- **Memory**: Minimum 2GB GPU memory (4GB+ recommended)
- **Driver**: NVIDIA driver version 450.80.02 or higher

### Recommended GPUs
- **Desktop**: GTX 1060 (6GB) or higher, RTX series
- **Laptop**: GTX 1650 or higher, RTX mobile series
- **Workstation**: Quadro P1000 or higher, RTX A series
- **Data Center**: Tesla T4, V100, A100

## Installation

### 1. Install NVIDIA Drivers

#### Ubuntu/Debian
```bash
# Add NVIDIA driver repository
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install recommended driver
sudo ubuntu-drivers autoinstall

# Or install specific version
sudo apt install nvidia-driver-535

# Reboot
sudo reboot

# Verify installation
nvidia-smi
```

#### Windows
1. Download drivers from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
2. Run installer and follow instructions
3. Verify with `nvidia-smi` in PowerShell

### 2. Install CUDA Toolkit (Optional)

ONNX Runtime GPU will use the bundled CUDA libraries, but you can install the full toolkit for development.

```bash
# Ubuntu 22.04 example
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-2
```

### 3. Install Python Dependencies

```bash
# Install GPU-specific requirements
pip install -r requirements-gpu.txt

# Core dependencies:
pip install onnxruntime-gpu>=1.16.0
pip install pynvml>=11.5.0
```

### 4. Verify GPU Detection

Run the hardware detection script:

```python
from src.detectors.hardware_detector import HardwareDetector

detector = HardwareDetector()
print(detector.get_hardware_summary())
```

Expected output should show your GPU:
```
Hardware Capabilities:
  System: Linux x86_64
  Python: 3.11.0
  
  GPUs:
    - NVIDIA GeForce RTX 3060 (12288MB)
      Compute: 8.6
  
  Recommended device: GPU
```

## Configuration

### Basic Configuration

```python
from src.detectors import DetectorFactory

# Create GPU detector
config = {
    'model': 'retinaface',  # or 'yolov5_face'
    'device': 'cuda:0',     # Use first GPU
    'batch_size': 4,        # Adjust based on GPU memory
    'confidence_threshold': 0.7,
    'enable_amp': True      # Automatic Mixed Precision
}

detector = DetectorFactory.create('gpu', config)
```

### Multi-GPU Configuration

```python
# Use specific GPU
config = {
    'model': 'retinaface',
    'device': 'cuda:1',  # Use second GPU
    'batch_size': 8
}

detector = DetectorFactory.create('gpu', config)
```

### Memory Management

```python
# Limit GPU memory usage
config = {
    'model': 'retinaface',
    'device': 'cuda:0',
    'memory_fraction': 0.5,  # Use only 50% of GPU memory
    'batch_size': 2
}

detector = DetectorFactory.create('gpu', config)
```

## Performance Tuning

### Batch Size Optimization

Find optimal batch size for your GPU:

```python
from src.detectors.performance_profiler import PerformanceProfiler
import numpy as np

# Create test images
test_images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]

# Test different batch sizes
for batch_size in [1, 2, 4, 8, 16]:
    config = {'model': 'retinaface', 'device': 'cuda:0', 'batch_size': batch_size}
    detector = DetectorFactory.create('gpu', config)
    
    profiler = PerformanceProfiler()
    result = profiler.benchmark_detector(detector, test_images, iterations=5)
    
    print(f"Batch {batch_size}: {result.fps:.1f} FPS, {result.avg_inference_time_ms:.2f}ms")
```

### Model Selection

Different models have different performance characteristics:

| Model | Speed | Accuracy | Memory | Recommended For |
|-------|-------|----------|--------|-----------------|
| RetinaFace | Medium | High | 10MB | Best balance |
| YOLOv5 Face | Fast | Medium | 15MB | Real-time processing |

### Input Resolution

Lower resolution = faster inference:

```python
# High accuracy, slower
config = {'model': 'retinaface', 'input_size': (640, 640)}

# Balanced
config = {'model': 'retinaface', 'input_size': (416, 416)}

# Fast, lower accuracy
config = {'model': 'blazeface', 'input_size': (224, 224)}
```

## Troubleshooting

### GPU Not Detected

**Problem**: `is_available()` returns `False`

**Solutions**:
1. Verify NVIDIA driver: `nvidia-smi`
2. Check CUDA availability:
   ```python
   import onnxruntime as ort
   print(ort.get_available_providers())
   # Should include 'CUDAExecutionProvider'
   ```
3. Reinstall onnxruntime-gpu:
   ```bash
   pip uninstall onnxruntime onnxruntime-gpu
   pip install onnxruntime-gpu
   ```

### Out of Memory Errors

**Problem**: `CUDA out of memory` errors

**Solutions**:
1. Reduce batch size:
   ```python
   config['batch_size'] = 1
   ```
2. Limit memory fraction:
   ```python
   config['memory_fraction'] = 0.5
   ```
3. Use smaller model:
   ```python
   config['model'] = 'blazeface'
   ```

### Slow Performance

**Problem**: GPU performance slower than expected

**Solutions**:
1. Check GPU utilization: `nvidia-smi`
2. Enable AMP (if supported):
   ```python
   config['enable_amp'] = True
   ```
3. Increase batch size:
   ```python
   config['batch_size'] = 8
   ```
4. Check for CPU bottlenecks in preprocessing

### Driver Compatibility Issues

**Problem**: CUDA version mismatch errors

**Solutions**:
1. Check driver/CUDA compatibility:
   ```bash
   nvidia-smi  # Check CUDA version
   python -c "import torch; print(torch.version.cuda)"
   ```
2. Upgrade NVIDIA driver
3. Use compatible onnxruntime-gpu version

## Monitoring

### GPU Utilization

Monitor GPU usage in real-time:

```bash
# Watch GPU stats
watch -n 1 nvidia-smi

# Or use dedicated tool
pip install nvitop
nvitop
```

### Performance Metrics

```python
detector = DetectorFactory.create('gpu', config)

# Get performance stats
metrics = detector.get_performance_metrics()
print(f"Average inference time: {metrics['average_inference_time_ms']:.2f}ms")
print(f"FPS: {metrics['fps']:.1f}")
print(f"GPU Memory: {metrics['gpu_memory_usage']['allocated']:.1f}MB")
```

## Best Practices

1. **Batch Processing**: Process multiple images together when possible
2. **Persistent Session**: Reuse detector instance across multiple inferences
3. **Warm-up**: First inference is slower; run warm-up before benchmarking
4. **Memory Management**: Monitor and limit GPU memory usage
5. **Model Selection**: Choose model based on accuracy vs speed requirements
6. **Resolution Tuning**: Balance input resolution with performance needs

## Production Deployment

### Docker with GPU Support

```dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt requirements-gpu.txt ./
RUN pip3 install -r requirements.txt -r requirements-gpu.txt

# Copy application
COPY . /app
WORKDIR /app

CMD ["python3", "app.py"]
```

Run with GPU access:
```bash
docker run --gpus all -it doorbell-system
```

### Kubernetes Deployment

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: doorbell-detector
spec:
  containers:
  - name: detector
    image: doorbell-system:gpu
    resources:
      limits:
        nvidia.com/gpu: 1
```

## Additional Resources

- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [ONNX Runtime GPU Documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [GPU Performance Optimization](https://docs.nvidia.com/deeplearning/performance/index.html)
