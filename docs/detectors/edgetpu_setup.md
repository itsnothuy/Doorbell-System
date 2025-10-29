# EdgeTPU Setup Guide

This guide covers setting up Coral EdgeTPU-accelerated face detection for the Doorbell Security System.

## Prerequisites

### Hardware Requirements

#### Supported Devices
- **USB Accelerator**: Coral USB Accelerator (most common)
- **Dev Board**: Coral Dev Board / Dev Board Mini
- **PCIe Accelerator**: Coral M.2/PCIe Accelerator
- **System on Module**: Coral SoM with compatible baseboard

### Recommended Configurations
- **Raspberry Pi**: Pi 4 (4GB+) with Coral USB Accelerator
- **PC/Server**: Any x86_64 system with USB 3.0 and Coral USB Accelerator
- **Embedded**: Coral Dev Board for standalone deployment

## Installation

### 1. Install EdgeTPU Runtime

#### Raspberry Pi / Debian-based Linux

```bash
# Add Coral repository
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
  sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

# Add repository key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
  sudo apt-key add -

# Update and install runtime
sudo apt update
sudo apt install libedgetpu1-std

# For maximum performance (runs hotter):
# sudo apt install libedgetpu1-max
```

#### macOS

```bash
# Install via Homebrew
brew tap coral/coral
brew install libedgetpu
```

#### Windows

Download and install from [Coral USB Accelerator guide](https://coral.ai/docs/accelerator/get-started/)

### 2. Connect EdgeTPU Device

#### USB Accelerator
1. Plug Coral USB Accelerator into USB 3.0 port (USB 2.0 also works)
2. Wait for device to be recognized
3. Verify connection:
   ```bash
   lsusb | grep "Global Unichip"
   ```
   Should show: `Bus 001 Device 002: ID 1a6e:089a Global Unichip Corp.`

#### Dev Board
1. Flash operating system if needed
2. Connect via USB or network
3. Access shell and verify TPU:
   ```bash
   cat /sys/class/apex/apex_0/device/chip_model
   ```

### 3. Set up USB Permissions (Linux only)

```bash
# Add udev rules
sudo sh -c "echo 'SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"1a6e\", GROUP=\"plugdev\"' \
  >> /etc/udev/rules.d/99-edgetpu-accelerator.rules"

sudo sh -c "echo 'SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"18d1\", GROUP=\"plugdev\"' \
  >> /etc/udev/rules.d/99-edgetpu-accelerator.rules"

# Reload rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Add user to plugdev group
sudo usermod -aG plugdev $USER

# Log out and back in for changes to take effect
```

### 4. Install Python Dependencies

```bash
# Install EdgeTPU-specific requirements
pip install -r requirements-edgetpu.txt

# Core dependencies:
pip install pycoral>=2.0.0
pip install tflite-runtime>=2.14.0
```

### 5. Verify EdgeTPU Detection

Run the hardware detection script:

```python
from src.detectors.hardware_detector import HardwareDetector

detector = HardwareDetector()
print(detector.get_hardware_summary())
```

Expected output should show your EdgeTPU:
```
Hardware Capabilities:
  System: Linux armv7l
  Python: 3.11.0
  
  GPUs: None detected
  
  EdgeTPUs:
    - usb at /dev/bus/usb/001/002
  
  Recommended device: EDGETPU
```

## Configuration

### Basic Configuration

```python
from src.detectors import DetectorFactory

# Create EdgeTPU detector
config = {
    'model': 'mobilenet_face',  # or 'efficientdet_face', 'blazeface'
    'confidence_threshold': 0.6,
    'enable_monitoring': True,
    'temperature_limit': 85.0  # Celsius
}

detector = DetectorFactory.create('edgetpu', config)
```

### Multi-Device Configuration

If you have multiple EdgeTPU devices:

```python
# Use specific device
config = {
    'model': 'mobilenet_face',
    'device_path': 'usb:0',  # First USB device
    'confidence_threshold': 0.6
}

detector = DetectorFactory.create('edgetpu', config)
```

### Temperature Monitoring

```python
# Enable thermal throttling protection
config = {
    'model': 'mobilenet_face',
    'enable_monitoring': True,
    'temperature_limit': 80.0  # Lower limit for cooler operation
}

detector = DetectorFactory.create('edgetpu', config)
```

## Performance Tuning

### Model Selection

EdgeTPU models are optimized for specific use cases:

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| BlazeFace | Fastest | Good | High FPS, mobile |
| MobileNet Face | Fast | Better | Best balance |
| EfficientDet Face | Medium | Best | Higher accuracy |

### Inference Optimization

```python
from src.detectors.performance_profiler import PerformanceProfiler
import numpy as np

# Create test images
test_images = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(20)]

# Benchmark different models
models = ['blazeface', 'mobilenet_face', 'efficientdet_face']

for model in models:
    config = {'model': model}
    detector = DetectorFactory.create('edgetpu', config)
    
    profiler = PerformanceProfiler()
    result = profiler.benchmark_detector(detector, test_images, iterations=10)
    
    print(f"{model}: {result.fps:.1f} FPS, {result.avg_inference_time_ms:.2f}ms")
```

Expected performance (Coral USB Accelerator):
- **BlazeFace**: ~60-80 FPS
- **MobileNet Face**: ~40-50 FPS  
- **EfficientDet Face**: ~20-30 FPS

## Troubleshooting

### EdgeTPU Not Detected

**Problem**: `is_available()` returns `False`

**Solutions**:
1. Check USB connection:
   ```bash
   lsusb | grep "Global Unichip"
   ```

2. Verify runtime installation:
   ```bash
   dpkg -l | grep edgetpu
   ```

3. Check USB permissions:
   ```bash
   ls -l /dev/bus/usb/001/002
   # Should show group 'plugdev'
   ```

4. Reinstall pycoral:
   ```bash
   pip uninstall pycoral tflite-runtime
   pip install pycoral tflite-runtime
   ```

### Slow Performance

**Problem**: Lower than expected FPS

**Solutions**:
1. Use maximum performance runtime:
   ```bash
   sudo apt remove libedgetpu1-std
   sudo apt install libedgetpu1-max
   ```
   ⚠️ **Warning**: Max performance mode increases power consumption and heat

2. Check USB bus speed:
   ```bash
   lsusb -t
   # Verify device is on USB 3.0 (5000M)
   ```

3. Use lighter model:
   ```python
   config['model'] = 'blazeface'
   ```

### Temperature Issues

**Problem**: Thermal throttling or overheating

**Solutions**:
1. Add cooling:
   - Attach heatsink to EdgeTPU
   - Use case with fan
   - Ensure good airflow

2. Switch to standard runtime:
   ```bash
   sudo apt remove libedgetpu1-max
   sudo apt install libedgetpu1-std
   ```

3. Lower temperature limit:
   ```python
   config['temperature_limit'] = 75.0
   ```

### Model Loading Errors

**Problem**: Model file not found or incompatible

**Solutions**:
1. Check model format (must be `.tflite` with EdgeTPU delegate)
2. Verify model is EdgeTPU-compiled
3. Use included models:
   ```python
   # Models are automatically downloaded/cached
   config = {'model': 'mobilenet_face'}
   detector = DetectorFactory.create('edgetpu', config)
   ```

### Permission Denied Errors

**Problem**: Permission denied accessing `/dev/bus/usb/...`

**Solutions**:
1. Add user to plugdev group:
   ```bash
   sudo usermod -aG plugdev $USER
   ```

2. Log out and back in

3. Or run with sudo (not recommended):
   ```bash
   sudo python app.py
   ```

## Monitoring

### Real-time Performance

```python
detector = DetectorFactory.create('edgetpu', config)

# Get performance stats
metrics = detector.get_performance_metrics()
print(f"Average inference time: {metrics['average_inference_time_ms']:.2f}ms")
print(f"FPS: {metrics['fps']:.1f}")
print(f"Throttle events: {metrics['throttle_events']}")

if 'avg_temperature_c' in metrics:
    print(f"Temperature: {metrics['avg_temperature_c']:.1f}°C")
```

### Temperature Monitoring

```bash
# Check EdgeTPU temperature (if supported)
cat /sys/class/apex/apex_0/temp
```

## Best Practices

1. **Cooling**: Always use heatsink or active cooling for continuous operation
2. **Model Selection**: Choose based on accuracy vs speed requirements
3. **Batch Processing**: EdgeTPU processes one image at a time efficiently
4. **Power Management**: Consider power consumption for battery/mobile deployments
5. **Thermal Management**: Monitor temperature in production environments
6. **USB Port**: Use USB 3.0 for best performance (USB 2.0 also works)

## Production Deployment

### Raspberry Pi Setup

```bash
#!/bin/bash
# deploy.sh - EdgeTPU deployment script for Raspberry Pi

# Update system
sudo apt update && sudo apt upgrade -y

# Install EdgeTPU runtime
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
  sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install -y libedgetpu1-std python3-pip

# Install Python dependencies
pip3 install -r requirements.txt -r requirements-edgetpu.txt

# Set up USB permissions
sudo sh -c "echo 'SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"1a6e\", GROUP=\"plugdev\"' \
  >> /etc/udev/rules.d/99-edgetpu-accelerator.rules"
sudo udevadm control --reload-rules
sudo udevadm trigger

# Configure auto-start
sudo cp doorbell.service /etc/systemd/system/
sudo systemctl enable doorbell
sudo systemctl start doorbell

echo "EdgeTPU setup complete!"
```

### Docker Deployment

```dockerfile
FROM debian:bullseye-slim

# Install EdgeTPU runtime
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    > /etc/apt/sources.list.d/coral-edgetpu.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && \
    apt-get install -y libedgetpu1-std python3 python3-pip

# Install Python dependencies
COPY requirements.txt requirements-edgetpu.txt ./
RUN pip3 install -r requirements.txt -r requirements-edgetpu.txt

# Copy application
COPY . /app
WORKDIR /app

CMD ["python3", "app.py"]
```

Run with USB device access:
```bash
docker run --device /dev/bus/usb -it doorbell-system
```

### Systemd Service

```ini
[Unit]
Description=Doorbell Security System with EdgeTPU
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/doorbell-system
ExecStart=/usr/bin/python3 app.py
Restart=always
RestartSec=10
Environment="DETECTOR_TYPE=edgetpu"

[Install]
WantedBy=multi-user.target
```

## Performance Comparison

### EdgeTPU vs CPU vs GPU

Typical performance on 640x480 images:

| Detector | Device | FPS | Latency | Power |
|----------|--------|-----|---------|-------|
| CPU HOG | Pi 4 | ~2-3 | 300-500ms | 5W |
| EdgeTPU | Pi 4 + Coral | ~40-50 | 20-25ms | 7W |
| GPU | Desktop RTX 3060 | ~100-150 | 6-10ms | 170W |

**EdgeTPU Advantages:**
- ✅ Low latency (sub-100ms)
- ✅ Low power consumption
- ✅ Excellent for edge deployment
- ✅ Cost-effective ($60-80)
- ✅ No thermal throttling with cooling

**EdgeTPU Limitations:**
- ❌ Requires specific model format (.tflite)
- ❌ Limited to INT8 quantization
- ❌ Single-threaded (one inference at a time)
- ❌ Requires USB connection (USB Accelerator)

## Model Conversion

To use custom models with EdgeTPU:

1. Train model in TensorFlow
2. Convert to TensorFlow Lite:
   ```python
   converter = tf.lite.TFLiteConverter.from_saved_model('model/')
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   tflite_model = converter.convert()
   ```

3. Compile for EdgeTPU:
   ```bash
   edgetpu_compiler model.tflite
   # Generates: model_edgetpu.tflite
   ```

4. Deploy to system:
   ```bash
   cp model_edgetpu.tflite ~/.doorbell_models/models/
   ```

## Additional Resources

- [Coral EdgeTPU Documentation](https://coral.ai/docs/)
- [PyCoral API Reference](https://coral.ai/docs/reference/py/)
- [Model Zoo](https://coral.ai/models/)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [EdgeTPU Compiler](https://coral.ai/docs/edgetpu/compiler/)
